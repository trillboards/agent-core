package com.trillboards.ctv.core.ml

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import com.trillboards.ctv.core.VASWeightConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.security.KeyFactory
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.Signature
import java.security.spec.PKCS8EncodedKeySpec
import java.security.spec.X509EncodedKeySpec
import android.util.Base64
import kotlinx.coroutines.*
import kotlin.math.abs
import java.util.Locale
import java.util.Calendar

/**
 * Federated Trainer for on-device gradient computation.
 *
 * After each 10s aggregation window, computes local gradient updates for
 * emotion + audio models. Uses top-k sparsification (10% of gradients)
 * to reduce bandwidth by 90%. Uploads gradients to server every 6 hours.
 *
 * Privacy: Only mathematical gradient deltas leave the device, never raw data.
 *
 * Thread Safety: All gradient buffer access is synchronized via [lock].
 * Key Persistence: Ed25519 keys are stored in EncryptedSharedPreferences
 *   (fallback to regular SharedPreferences) so publicKeyId is stable across restarts.
 */
/**
 * Multi-slice context for federated learning.
 * Each combination of these dimensions gets its own model variant on the server.
 * E.g., bar/sports_bar/evening/miami/RK3588 → separate weights from
 *       mall/food_court/afternoon/nyc/MediaTek.
 */
data class FederatedSliceContext(
    val venueType: String,
    val venueSubtype: String = "",
    val daypart: String = "",
    val geo: String = "",
    val deviceProfile: String = ""
) {
    fun toQueryParams(): String {
        val params = mutableListOf("venue_type=$venueType")
        if (venueSubtype.isNotEmpty()) params.add("venue_subtype=$venueSubtype")
        if (daypart.isNotEmpty()) params.add("daypart=$daypart")
        if (geo.isNotEmpty()) params.add("geo=$geo")
        if (deviceProfile.isNotEmpty()) params.add("device_profile=$deviceProfile")
        return params.joinToString("&")
    }
}

class FederatedTrainer(
    private val context: Context,
    private val apiBaseUrl: String,
    private val deviceFingerprint: String,
    private val screenMongoId: String,
    private val venueType: String
) {
    companion object {
        private const val TAG = "FederatedTrainer"
        private const val UPLOAD_INTERVAL_MS = 6 * 60 * 60 * 1000L // 6 hours
        private const val TOP_K_RATIO = 0.1f // Keep top 10% of gradients
        private const val MIN_SAMPLES_FOR_UPLOAD = 100
        private const val MAX_GRADIENT_BUFFER_SIZE = 10000
        private const val PREFS_NAME = "vas_attestation_keys"
        private const val PREF_PRIVATE_KEY = "vas_private_key"
        private const val PREF_PUBLIC_KEY = "vas_public_key"
        private const val PREF_KEY_ID = "vas_key_id"

        /**
         * Compute current daypart from hour of day.
         * morning: 6-11, afternoon: 12-16, evening: 17-21, late_night: 22-5
         */
        fun computeDaypart(): String {
            val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
            return when {
                hour in 6..11 -> "morning"
                hour in 12..16 -> "afternoon"
                hour in 17..21 -> "evening"
                else -> "late_night"
            }
        }
    }

    // Coroutine scope for async gradient uploads (replaces raw Thread)
    private val uploadScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Thread safety lock for all gradient buffer access
    private val lock = Any()

    // Multi-slice context — updated at runtime via setSliceContext()
    @Volatile private var sliceContext = FederatedSliceContext(venueType = venueType)

    // Gradient accumulation buffer per model
    private val gradientBuffers = mutableMapOf<String, MutableList<FloatArray>>()
    private val sampleCounts = mutableMapOf<String, Int>()
    private val localLosses = mutableMapOf<String, MutableList<Float>>()
    @Volatile private var lastUploadTime = System.currentTimeMillis()
    private var currentModelVersions = mutableMapOf<String, Int>()

    // Local adaptation layer: last 100 observations per model (NOT uploaded)
    private val localAdaptationBuffer = mutableMapOf<String, MutableList<FloatArray>>()
    private val localAdaptationMaxSize = 100

    // Ed25519 key pair for VAS attestation (requires API 33+)
    private var attestationKeyPair: KeyPair? = null
    var publicKeyId: String? = null
        private set

    init {
        initAttestationKeys()
    }

    /**
     * Initialize Ed25519 key pair for VAS attestation.
     * Keys are persisted so that publicKeyId remains stable across app restarts.
     * Uses EncryptedSharedPreferences when available, falls back to regular prefs.
     */
    private fun initAttestationKeys() {
        if (Build.VERSION.SDK_INT < 33) {
            Log.i(TAG, "Ed25519 requires API 33+, attestation disabled on API ${Build.VERSION.SDK_INT}")
            return
        }

        try {
            val prefs = getKeyPreferences()

            val existingPrivateKey = prefs.getString(PREF_PRIVATE_KEY, null)
            val existingPublicKey = prefs.getString(PREF_PUBLIC_KEY, null)
            val existingKeyId = prefs.getString(PREF_KEY_ID, null)

            if (existingPrivateKey != null && existingPublicKey != null && existingKeyId != null) {
                val keyFactory = KeyFactory.getInstance("Ed25519")
                val privateKeyBytes = Base64.decode(existingPrivateKey, Base64.NO_WRAP)
                val publicKeyBytes = Base64.decode(existingPublicKey, Base64.NO_WRAP)
                val privateKey = keyFactory.generatePrivate(PKCS8EncodedKeySpec(privateKeyBytes))
                val publicKey = keyFactory.generatePublic(X509EncodedKeySpec(publicKeyBytes))
                attestationKeyPair = KeyPair(publicKey, privateKey)
                publicKeyId = existingKeyId
                Log.i(TAG, "Restored persisted attestation key pair: $publicKeyId")
            } else {
                val keyGen = KeyPairGenerator.getInstance("Ed25519")
                val newKeyPair = keyGen.generateKeyPair()
                val newKeyId = "dev_${deviceFingerprint.take(16)}_${System.currentTimeMillis().toString(36)}"
                prefs.edit()
                    .putString(PREF_PRIVATE_KEY, Base64.encodeToString(newKeyPair.private.encoded, Base64.NO_WRAP))
                    .putString(PREF_PUBLIC_KEY, Base64.encodeToString(newKeyPair.public.encoded, Base64.NO_WRAP))
                    .putString(PREF_KEY_ID, newKeyId)
                    .apply()
                attestationKeyPair = newKeyPair
                publicKeyId = newKeyId
                Log.i(TAG, "Generated and persisted new attestation key pair: $publicKeyId")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Ed25519 not available, attestation disabled: ${e.message}")
        }
    }

    private fun getKeyPreferences(): SharedPreferences {
        try {
            val masterKey = androidx.security.crypto.MasterKey.Builder(context)
                .setKeyScheme(androidx.security.crypto.MasterKey.KeyScheme.AES256_GCM)
                .build()
            return androidx.security.crypto.EncryptedSharedPreferences.create(
                context,
                PREFS_NAME,
                masterKey,
                androidx.security.crypto.EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                androidx.security.crypto.EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )
        } catch (e: Exception) {
            Log.w(TAG, "EncryptedSharedPreferences unavailable, using regular prefs: ${e.message}")
            return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        }
    }

    /**
     * Update the multi-slice context for federated learning.
     * Called when venue info is loaded or daypart changes.
     */
    fun setSliceContext(
        venueSubtype: String = "",
        geo: String = "",
        deviceProfile: String = ""
    ) {
        sliceContext = FederatedSliceContext(
            venueType = venueType,
            venueSubtype = venueSubtype,
            daypart = computeDaypart(),
            geo = geo,
            deviceProfile = deviceProfile
        )
        Log.d(TAG, "Slice context updated: $sliceContext")
    }

    /**
     * Store a local adaptation observation (NOT uploaded to server).
     * These capture device-specific calibration that should not be averaged
     * with other devices' gradients.
     */
    fun recordLocalAdaptation(modelType: String, observation: FloatArray) {
        synchronized(lock) {
            val buffer = localAdaptationBuffer.getOrPut(modelType) { mutableListOf() }
            buffer.add(observation)
            if (buffer.size > localAdaptationMaxSize) {
                buffer.removeAt(0)
            }
        }
    }

    /**
     * Get local adaptation weights for a model (applied on top of federated weights).
     */
    fun getLocalAdaptation(modelType: String): List<FloatArray> {
        synchronized(lock) {
            return localAdaptationBuffer[modelType]?.toList() ?: emptyList()
        }
    }

    fun accumulateGradients(modelType: String, predictions: FloatArray, labels: FloatArray) {
        if (predictions.size != labels.size || predictions.isEmpty()) return

        val gradients = FloatArray(predictions.size) { i -> predictions[i] - labels[i] }
        val loss = gradients.map { it * it }.average().toFloat()

        synchronized(lock) {
            val buffer = gradientBuffers.getOrPut(modelType) { mutableListOf() }
            buffer.add(gradients)
            sampleCounts[modelType] = (sampleCounts[modelType] ?: 0) + 1
            localLosses.getOrPut(modelType) { mutableListOf() }.add(loss)
            if (buffer.size > MAX_GRADIENT_BUFFER_SIZE) {
                buffer.removeAt(0)
            }
        }

        if (System.currentTimeMillis() - lastUploadTime >= UPLOAD_INTERVAL_MS) {
            uploadAllGradients()
        }
    }

    fun computeSparseGradients(modelType: String): Pair<FloatArray, IntArray>? {
        val bufferCopy: List<FloatArray>
        synchronized(lock) {
            val buffer = gradientBuffers[modelType] ?: return null
            if (buffer.isEmpty()) return null
            bufferCopy = buffer.toList()
        }
        return sparseTopK(bufferCopy)
    }

    fun uploadAllGradients() {
        val uploadsToPerform = mutableListOf<UploadData>()

        synchronized(lock) {
            for (modelType in gradientBuffers.keys.toList()) {
                val samples = sampleCounts[modelType] ?: 0
                if (samples < MIN_SAMPLES_FOR_UPLOAD) {
                    Log.d(TAG, "Skipping $modelType: only $samples samples (need $MIN_SAMPLES_FOR_UPLOAD)")
                    continue
                }

                val buffer = gradientBuffers[modelType] ?: continue
                if (buffer.isEmpty()) continue
                val sparse = sparseTopK(buffer.toList()) ?: continue
                val (values, indices) = sparse
                val avgLoss = localLosses[modelType]?.average()?.toFloat() ?: 0f
                val modelVersion = currentModelVersions[modelType] ?: 0
                uploadsToPerform.add(UploadData(modelType, values, indices, samples, avgLoss, modelVersion))

                gradientBuffers[modelType]?.clear()
                sampleCounts[modelType] = 0
                localLosses[modelType]?.clear()
            }
        }

        // Update daypart in slice context before uploading
        sliceContext = sliceContext.copy(daypart = computeDaypart())

        for (upload in uploadsToPerform) {
            uploadScope.launch {
                try {
                    uploadGradients(upload.modelType, upload.values, upload.indices, upload.sampleCount, upload.avgLoss, upload.modelVersion)
                    Log.i(TAG, "Uploaded gradients for ${upload.modelType}: ${upload.values.size} sparse values, ${upload.sampleCount} samples")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to upload gradients for ${upload.modelType}: ${e.message}")
                }
            }
        }

        lastUploadTime = System.currentTimeMillis()
    }

    /**
     * Release coroutine scope and cancel any pending upload coroutines.
     * Call on service shutdown for structured cancellation.
     */
    fun release() {
        uploadScope.cancel()
    }

    /**
     * Shared top-k sparsification: average gradients then keep top 10% by magnitude.
     */
    private fun sparseTopK(bufferCopy: List<FloatArray>): Pair<FloatArray, IntArray>? {
        if (bufferCopy.isEmpty()) return null
        val gradientSize = bufferCopy[0].size
        val avgGradients = FloatArray(gradientSize)
        for (grad in bufferCopy) {
            for (i in grad.indices) { avgGradients[i] += grad[i] / bufferCopy.size }
        }
        val k = maxOf(1, (gradientSize * TOP_K_RATIO).toInt())
        val indexed = avgGradients.mapIndexed { idx, value -> Pair(idx, value) }
            .sortedByDescending { abs(it.second) }.take(k)
        return Pair(
            FloatArray(indexed.size) { indexed[it].second },
            IntArray(indexed.size) { indexed[it].first }
        )
    }

    private fun uploadGradients(modelType: String, gradients: FloatArray, gradientIndices: IntArray, sampleCount: Int, localLoss: Float, modelVersion: Int) {
        val url = URL("$apiBaseUrl/v1/federated/gradients")
        val connection = url.openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("x-device-fingerprint", deviceFingerprint)
            connection.connectTimeout = 30000
            connection.readTimeout = 30000
            connection.doOutput = true

            val ctx = sliceContext
            val body = JSONObject().apply {
                put("deviceId", deviceFingerprint)
                put("screenMongoId", screenMongoId)
                put("modelType", modelType)
                put("gradients", JSONArray(gradients.toList()))
                put("gradientIndices", JSONArray(gradientIndices.toList()))
                put("sampleCount", sampleCount)
                put("localLoss", localLoss)
                put("modelVersion", modelVersion)
                // Multi-slice context
                put("venueType", ctx.venueType)
                put("venueSubtype", ctx.venueSubtype)
                put("daypart", ctx.daypart)
                put("geo", ctx.geo)
                put("deviceProfile", ctx.deviceProfile)
            }

            connection.outputStream.use { os ->
                OutputStreamWriter(os).use { writer ->
                    writer.write(body.toString())
                    writer.flush()
                }
            }

            val responseCode = connection.responseCode
            if (responseCode == 200) {
                val response = connection.inputStream.use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream)).use { reader ->
                        JSONObject(reader.readText())
                    }
                }

                // Check if server has a newer model version for this slice
                val serverVersion = response.optInt("current_model_version", 0)
                if (serverVersion > modelVersion) {
                    synchronized(lock) { currentModelVersions[modelType] = serverVersion }
                    Log.i(TAG, "Server has newer model v$serverVersion for $modelType (current: v$modelVersion)")
                }

                // Check for slice-specific weight update notification
                val sliceUpdate = response.optBoolean("slice_update_available", false)
                if (sliceUpdate) {
                    Log.i(TAG, "Slice-specific update available for $modelType [${ctx.venueType}/${ctx.venueSubtype}/${ctx.daypart}]")
                }
            } else {
                Log.w(TAG, "Gradient upload failed with status $responseCode")
            }
        } finally {
            connection.disconnect()
        }
    }

    fun computeVAS(
        avgAttention: Float,
        avgDwellTimeMs: Float,
        avgFaceCount: Float,
        emotionalEngagement: Float = 0f,
        bodyEngagement: Float = 0f,
        focusRegion: Int = 4,
        windowMs: Long = 10000L,
        adId: String? = null,
        daypart: String? = null
    ): VASResult {
        if (avgAttention < 0.5f || avgFaceCount == 0f) {
            return VASResult(0f, 0f, 0f, null, publicKeyId, null)
        }
        val attentionSeconds = minOf(avgDwellTimeMs, windowMs.toFloat()) / 1000f
        val vasRaw = attentionSeconds * avgFaceCount
        val focusBonus = when (focusRegion) {
            4 -> 1.0f; 1, 3, 5, 7 -> 0.7f; else -> 0.5f
        }
        val weights = VASWeightConfig.getWeights(venueType, daypart)
        val qualityMultiplier = (
            weights.attention * minOf(avgAttention, 1f) +
            weights.emotion * minOf(emotionalEngagement, 1f) +
            weights.body * minOf(bodyEngagement, 1f) +
            weights.focus * focusBonus
        )
        val vasWeighted = vasRaw * qualityMultiplier
        val signResult = signVAS(vasWeighted, adId ?: "")
        return VASResult(
            vasRaw, vasWeighted, qualityMultiplier,
            signResult?.first, publicKeyId, signResult?.second
        )
    }

    /**
     * Sign VAS measurement with Ed25519. Returns Pair(signature, signingTimestamp)
     * so the CTV can transmit the exact timestamp used in signing for server verification.
     * Uses Locale.US for float formatting to ensure deterministic signature payloads.
     */
    private fun signVAS(vasScore: Float, adId: String): Pair<String, Long>? {
        val keyPair = attestationKeyPair ?: return null
        if (Build.VERSION.SDK_INT < 33) return null
        return try {
            val timestamp = System.currentTimeMillis()
            val vasStr = String.format(Locale.US, "%.6f", vasScore)
            val payload = "vas:$vasStr|ts:$timestamp|fp:$deviceFingerprint|ad:$adId"
            val signature = Signature.getInstance("Ed25519")
            signature.initSign(keyPair.private)
            signature.update(payload.toByteArray())
            Pair(Base64.encodeToString(signature.sign(), Base64.NO_WRAP), timestamp)
        } catch (e: Exception) {
            Log.w(TAG, "VAS signing failed: ${e.message}")
            null
        }
    }

    fun getPublicKeyPem(): String? {
        val keyPair = attestationKeyPair ?: return null
        return try {
            val base64 = Base64.encodeToString(keyPair.public.encoded, Base64.NO_WRAP)
            "-----BEGIN PUBLIC KEY-----\n$base64\n-----END PUBLIC KEY-----"
        } catch (e: Exception) { null }
    }

    fun getStats(): Map<String, Any> {
        val stats = mutableMapOf<String, Any>()
        synchronized(lock) {
            for ((model, buffer) in gradientBuffers) {
                stats[model] = mapOf(
                    "buffered_windows" to buffer.size,
                    "sample_count" to (sampleCounts[model] ?: 0),
                    "avg_loss" to (localLosses[model]?.average() ?: 0.0),
                    "model_version" to (currentModelVersions[model] ?: 0),
                    "local_adaptation_size" to (localAdaptationBuffer[model]?.size ?: 0)
                )
            }
        }
        stats["last_upload"] = lastUploadTime
        stats["has_attestation_key"] = attestationKeyPair != null
        stats["public_key_id"] = publicKeyId ?: "none"
        val ctx = sliceContext
        stats["slice_context"] = mapOf(
            "venue_type" to ctx.venueType,
            "venue_subtype" to ctx.venueSubtype,
            "daypart" to ctx.daypart,
            "geo" to ctx.geo,
            "device_profile" to ctx.deviceProfile
        )
        return stats
    }

    /**
     * Cancel pending coroutine uploads. Call on service shutdown.
     * @see release
     */
    fun destroy() {
        release()
    }

    data class VASResult(
        val vasRaw: Float,
        val vasWeighted: Float,
        val qualityMultiplier: Float,
        val attestationSignature: String?,
        val publicKeyId: String?,
        val signingTimestamp: Long?
    )

    private class UploadData(
        val modelType: String,
        val values: FloatArray,
        val indices: IntArray,
        val sampleCount: Int,
        val avgLoss: Float,
        val modelVersion: Int
    )
}
