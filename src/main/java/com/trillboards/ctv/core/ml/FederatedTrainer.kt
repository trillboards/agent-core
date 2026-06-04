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
import kotlin.math.exp
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

        // ─── FEIN screen-taste constants ─────────────────────────────────
        // Hard-coded mirror of `@trillboards/iab-taxonomy/fein-constants`
        // (data/fein-constants.cjs). The TS parity test in that package
        // pins src/fein.ts <-> data/fein-constants.cjs; this Kotlin block is
        // the third surface and MUST stay in sync. Drift is caught by the
        // Phase-7 cross-language conformance fixture (deferred work).
        const val AUDIENCE_VECTOR_DIM = 64
        const val SCREEN_TASTE_DIM = 64
        const val SGD_LEARNING_RATE = 0.01f
        const val SGD_STEP_EVERY_N_AD_COMPLETIONS = 20
        const val PRIOR_BLEND_ALPHA_INITIAL = 0.3f

        // Persistence keys for the per-screen taste embedding. The taste
        // vector + meta lives in EncryptedSharedPreferences alongside the
        // VAS attestation keypair so a single secure-store handle covers
        // both. Stored as a comma-separated float string for portability
        // (avoids JSON parse cost on hot path; debounced write means the
        // serialization cost is amortized).
        const val PREF_SCREEN_TASTE = "screen_taste_vector"
        const val PREF_TASTE_VERSION = "screen_taste_version"
        const val PREF_TASTE_SAMPLE_COUNT = "screen_taste_sample_count"
        const val PREF_TASTE_UPDATED_AT = "screen_taste_updated_at"

        // Debounce window for write-back of taste state to encrypted prefs.
        // Each SGD step is a small in-place mutation; serializing 64 floats
        // every step would be wasteful. 5 seconds covers the realistic
        // burst of N=20-step SGD events while keeping crash-loss bounded.
        private const val TASTE_PERSIST_DEBOUNCE_MS = 5_000L

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

    // ─── Screen-taste embedding state (FEIN P2 + P3) ─────────────────────
    //
    // Per-screen 64-dim taste vector. Lazy-init: starts null, populated
    // either by Phase 2's `sgdStep()` (zeros on first step, then SGD-
    // updated in place) OR by Phase 3's `setScreenTaste()` (overwritten
    // from a cloud-uploaded vector). Debounced writes persist it to
    // EncryptedSharedPreferences so a re-hydration on construction picks
    // up where the last SGD step left off.
    //
    // Mutations of `screenTaste` are guarded by `lock` (shared with the
    // existing gradient buffers — same I/O contention surface).
    @Volatile private var screenTaste: FloatArray? = null
    // Phase 2 SGD counters (local-side):
    @Volatile private var tasteVersion: Int = 0
    @Volatile private var tasteSampleCount: Int = 0
    @Volatile private var tasteUpdatedAtMs: Long = 0L

    // Debounce bookkeeping for taste-state writeback. `lastTastePersistMs`
    // tracks the wall-clock time of the most recent successful write so
    // back-to-back SGD steps coalesce into a single flush.
    @Volatile private var lastTastePersistMs: Long = 0L

    // Phase 3 upload bookkeeping (set by setScreenTaste from the FedAvg loop):
    @Volatile private var screenTasteSampleCount: Int = 0
    @Volatile private var screenTasteVersionHash: String? = null

    // Ed25519 key pair for VAS attestation (requires API 33+)
    private var attestationKeyPair: KeyPair? = null
    var publicKeyId: String? = null
        private set

    init {
        initAttestationKeys()
        loadScreenTasteFromPrefs()
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

    // ─── Screen-taste SGD (FEIN P2) ───────────────────────────────────────
    //
    // The per-screen taste embedding is a 64-dim vector learned by online
    // SGD on the (audience_emb ⊙ creative_emb) -> target_score signal. The
    // math is intentionally hand-rolled (no TFLite training graph) so the
    // CTV and the TS edge-federated mirror can stay bit-aligned without a
    // model-binary cross-language conformance step.
    //
    // Forward:
    //   z = taste · (audience_emb ⊙ creative_emb)
    //   affinity_pred = sigmoid(z)
    // Loss (per-sample MSE):
    //   L = (affinity_pred - target)^2
    // Gradient wrt taste (via the chain rule, sigmoid' = sigmoid * (1-sigmoid)):
    //   dL/dz       = 2 * (affinity_pred - target) * sigmoid'(z)
    //   dL/d_taste  = dL/dz * (audience_emb ⊙ creative_emb)
    // Update:
    //   taste <- taste - η * dL/d_taste,   η = SGD_LEARNING_RATE = 0.01
    //
    // Phase 2 stages the math: callers pass `targetScore` directly. The
    // actual training signal (`vas_weighted` from `attention_ledger`) is
    // wired in Phase 3 alongside the upload of the resulting taste vector
    // into the existing `federated_model_slices` aggregator path.

    /**
     * Perform one SGD step on the screen-taste embedding.
     *
     * @param audienceEmb 64-dim audience context vector
     *                    (output of [audienceLookalikeService.audienceToVector]).
     * @param creativeEmb 64-dim creative-content embedding vector.
     * @param targetScore Observed affinity in [0, 1] — the per-completion
     *                    label the cloud will start emitting in Phase 3
     *                    (e.g. `vas_weighted` normalized into [0,1]).
     *
     * Mutates [screenTaste] in place, advances [tasteSampleCount], and
     * schedules a debounced persistence write. Validates input dimensions;
     * silently no-ops on a mismatch so a misbehaving caller cannot crash
     * the pipeline.
     */
    fun sgdStep(audienceEmb: FloatArray, creativeEmb: FloatArray, targetScore: Float) {
        if (audienceEmb.size != SCREEN_TASTE_DIM ||
            creativeEmb.size != SCREEN_TASTE_DIM) {
            Log.w(
                TAG,
                "sgdStep dimension mismatch: audience=${audienceEmb.size} " +
                    "creative=${creativeEmb.size} expected=$SCREEN_TASTE_DIM"
            )
            return
        }

        // Pre-compute the Hadamard product (audience_emb ⊙ creative_emb).
        // This is also the gradient direction wrt taste (up to the scalar
        // dL/dz factor), so we re-use it below without a second allocation.
        val hadamard = FloatArray(SCREEN_TASTE_DIM) { audienceEmb[it] * creativeEmb[it] }

        synchronized(lock) {
            // Lazy-init: first SGD step on a fresh trainer (no prior, no
            // setScreenTaste) seeds the vector to zeros. Mirrors the TS
            // canonical's first-touch behavior.
            val taste = screenTaste ?: FloatArray(SCREEN_TASTE_DIM) { 0f }.also {
                screenTaste = it
            }

            // Forward pass: z = taste · hadamard
            var z = 0f
            for (i in 0 until SCREEN_TASTE_DIM) {
                z += taste[i] * hadamard[i]
            }
            val pred = sigmoid(z)
            val sigDeriv = pred * (1f - pred)  // sigmoid'(z)
            val dLdZ = 2f * (pred - targetScore) * sigDeriv

            // taste <- taste - η * dL/dz * hadamard[i]
            val step = SGD_LEARNING_RATE * dLdZ
            for (i in 0 until SCREEN_TASTE_DIM) {
                taste[i] -= step * hadamard[i]
            }

            tasteSampleCount += 1
            tasteUpdatedAtMs = System.currentTimeMillis()
        }

        maybePersistTaste()
    }

    /**
     * Blend a fleet-FedAvg prior into the local taste embedding.
     *
     * Used on cold-start (Phase 3 OTA path) to seed the per-screen taste
     * vector with a slice-level prior before local SGD takes over.
     *
     * @param prior The prior vector (must be [SCREEN_TASTE_DIM]).
     * @param alpha Mixing weight for the prior — 0 keeps local, 1 replaces.
     *              Defaults to [PRIOR_BLEND_ALPHA_INITIAL] (0.3) so cold
     *              screens lean toward the prior while warm screens that
     *              call this preserve most of their own learnings.
     *
     * Silent no-op on dimension mismatch (same defensive posture as
     * [sgdStep]).
     */
    fun applyPrior(prior: FloatArray, alpha: Float = PRIOR_BLEND_ALPHA_INITIAL) {
        if (prior.size != SCREEN_TASTE_DIM) {
            Log.w(
                TAG,
                "applyPrior dimension mismatch: prior=${prior.size} " +
                    "expected=$SCREEN_TASTE_DIM"
            )
            return
        }
        val clampedAlpha = alpha.coerceIn(0f, 1f)
        synchronized(lock) {
            val current = screenTaste
            if (current == null) {
                // Cold-start: no local SGD state yet — seed directly from
                // the prior so the very first ad request can use a non-zero
                // taste vector.
                screenTaste = prior.copyOf()
                tasteUpdatedAtMs = System.currentTimeMillis()
                Log.i(TAG, "applyPrior: seeded screen_taste state from server prior (alpha=$alpha)")
            } else {
                for (i in 0 until SCREEN_TASTE_DIM) {
                    current[i] = (1f - clampedAlpha) * current[i] + clampedAlpha * prior[i]
                }
                tasteUpdatedAtMs = System.currentTimeMillis()
                Log.i(TAG, "applyPrior: blended screen_taste with server prior (alpha=$alpha)")
            }
        }
        // Prior blending is a meaningful state change — write immediately
        // (no debounce) so a crash after a cold-start prior doesn't
        // re-bootstrap from zeros on next launch.
        persistTaste()
    }

    /**
     * Immutable point-in-time view of the screen-taste embedding.
     * Used by the upload flow (Phase 3) — separate snapshot so callers
     * never observe mid-step taste mutations.
     */
    data class TasteSnapshot(
        val version: Int,
        val sampleCount: Int,
        val taste: FloatArray,
        val updatedAt: Long
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is TasteSnapshot) return false
            return version == other.version &&
                sampleCount == other.sampleCount &&
                updatedAt == other.updatedAt &&
                taste.contentEquals(other.taste)
        }

        override fun hashCode(): Int {
            var result = version
            result = 31 * result + sampleCount
            result = 31 * result + updatedAt.hashCode()
            result = 31 * result + taste.contentHashCode()
            return result
        }
    }

    /**
     * Return a defensive copy of the current taste embedding + metadata.
     */
    fun snapshotTaste(): TasteSnapshot {
        synchronized(lock) {
            val current = screenTaste
            val tasteCopy = current?.copyOf() ?: FloatArray(SCREEN_TASTE_DIM) { 0f }
            return TasteSnapshot(
                version = tasteVersion,
                sampleCount = tasteSampleCount,
                taste = tasteCopy,
                updatedAt = tasteUpdatedAtMs
            )
        }
    }

    /**
     * Sigmoid activation: σ(z) = 1 / (1 + e^-z).
     * Internal — exposed via [sgdStep] for the forward pass.
     */
    private fun sigmoid(z: Float): Float {
        return 1f / (1f + exp(-z.toDouble()).toFloat())
    }

    /**
     * Re-hydrate the screen-taste embedding from EncryptedSharedPreferences.
     * Called once from [init] — failure is logged and left at zeros so the
     * trainer is always usable (a corrupt entry won't brick the SGD loop).
     *
     * Catches [Throwable] (not just [Exception]) because the Android-only
     * `EncryptedSharedPreferences` chain can throw `NoClassDefFoundError` /
     * `ExceptionInInitializerError` on JVM unit-test classpaths that lack
     * the Android Keystore. The fallback path uses Context.getSharedPrefs
     * which is mockable; this widening keeps the trainer constructible in
     * both environments.
     */
    private fun loadScreenTasteFromPrefs() {
        try {
            val prefs = getKeyPreferences()
            val serialized = prefs.getString(PREF_SCREEN_TASTE, null) ?: return
            val parts = serialized.split(',')
            if (parts.size != SCREEN_TASTE_DIM) {
                Log.w(
                    TAG,
                    "Persisted screen_taste has wrong dim: ${parts.size} " +
                        "expected=$SCREEN_TASTE_DIM — discarding"
                )
                return
            }
            val restored = FloatArray(SCREEN_TASTE_DIM)
            for (i in 0 until SCREEN_TASTE_DIM) {
                restored[i] = parts[i].toFloatOrNull() ?: 0f
            }
            screenTaste = restored
            tasteVersion = prefs.getInt(PREF_TASTE_VERSION, 0)
            tasteSampleCount = prefs.getInt(PREF_TASTE_SAMPLE_COUNT, 0)
            tasteUpdatedAtMs = prefs.getLong(PREF_TASTE_UPDATED_AT, 0L)
            Log.i(
                TAG,
                "Restored screen_taste: v=$tasteVersion samples=$tasteSampleCount " +
                    "updatedAt=$tasteUpdatedAtMs"
            )
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to load screen_taste from prefs: ${t.message}")
        }
    }

    /**
     * Persist the current screen-taste state if the debounce window has
     * elapsed since the last write. Cheap fast-path; falls through to
     * [persistTaste] when an actual write is due.
     *
     * Callers that need a guaranteed write (e.g. [applyPrior] after a
     * cold-start prior blend) bypass this and call [persistTaste] directly.
     */
    private fun maybePersistTaste() {
        val now = System.currentTimeMillis()
        if (now - lastTastePersistMs < TASTE_PERSIST_DEBOUNCE_MS) return
        persistTaste()
    }

    /**
     * Write the current screen-taste state to EncryptedSharedPreferences.
     *
     * Always writes, regardless of the debounce window — the wrapper
     * [maybePersistTaste] is the debounce decision point.
     */
    private fun persistTaste() {
        val snapshot = synchronized(lock) {
            // Copy under lock so we don't serialize mid-mutation. Bail out
            // when there's no taste state yet — nothing to persist.
            val current = screenTaste ?: return
            Triple(current.copyOf(), tasteSampleCount, tasteUpdatedAtMs)
        }
        try {
            val prefs = getKeyPreferences()
            val builder = StringBuilder(SCREEN_TASTE_DIM * 12)
            for (i in 0 until SCREEN_TASTE_DIM) {
                if (i > 0) builder.append(',')
                builder.append(snapshot.first[i].toString())
            }
            prefs.edit()
                .putString(PREF_SCREEN_TASTE, builder.toString())
                .putInt(PREF_TASTE_VERSION, tasteVersion)
                .putInt(PREF_TASTE_SAMPLE_COUNT, snapshot.second)
                .putLong(PREF_TASTE_UPDATED_AT, snapshot.third)
                .apply()
            lastTastePersistMs = System.currentTimeMillis()
        } catch (t: Throwable) {
            // See [loadScreenTasteFromPrefs] — same JVM-classpath catch
            // widening so a missing Android Keystore stub never crashes
            // the SGD hot path.
            Log.w(TAG, "Failed to persist screen_taste: ${t.message}")
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

        // Phase 3 — also ship the screen-taste embedding if Phase 2's SGD state
        // is populated. Forward-compatible: when Phase 2 lands `snapshotTaste()`
        // will be wired here directly; for now, `uploadScreenTaste()` no-ops
        // when state is null.
        uploadScope.launch {
            try {
                uploadScreenTaste()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to upload screen_taste embedding: ${e.message}")
            }
        }

        lastUploadTime = System.currentTimeMillis()
    }

    /**
     * Phase 2 hook — populate the screen-taste embedding state. Called by the
     * Phase 2 SGD step inside the aggregation window so that the next
     * `uploadAllGradients()` tick ships the fresh vector to
     * `POST /v1/federated/taste`.
     *
     * The vector MUST be length 64 (matches the canonical contract
     * `FederatedApi.TasteUploadPayload` in
     * `trillboard-api/validation/federatedSchemas.ts`); shorter/longer arrays
     * are dropped on the floor so a misconfigured caller can't break the
     * upload path.
     */
    fun setScreenTaste(embedding: FloatArray, sampleCount: Int, versionHash: String) {
        if (embedding.size != 64) {
            Log.w(TAG, "setScreenTaste rejected: vector length ${embedding.size} != 64")
            return
        }
        synchronized(lock) {
            screenTaste = embedding.copyOf()
            screenTasteSampleCount = sampleCount
            screenTasteVersionHash = versionHash
        }
    }

    /**
     * Upload the screen-taste embedding to `POST /v1/federated/taste`. No-ops
     * when Phase 2's state is not populated (forward-compatible — Phase 3
     * ships the transport, Phase 2 fills the state).
     */
    private fun uploadScreenTaste() {
        val embedding: FloatArray
        val sampleCount: Int
        val versionHash: String
        synchronized(lock) {
            val current = screenTaste ?: return
            if (screenTasteSampleCount < 1) return
            embedding = current.copyOf()
            sampleCount = screenTasteSampleCount
            versionHash = screenTasteVersionHash ?: "unknown"
            // Reset sample counter so the next window starts fresh; we leave
            // the embedding in place so it can keep accumulating until the
            // Phase 2 SGD step overwrites it.
            screenTasteSampleCount = 0
        }

        val url = URL("$apiBaseUrl/v1/federated/taste")
        val connection = url.openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("x-device-fingerprint", deviceFingerprint)
            connection.connectTimeout = 30000
            connection.readTimeout = 30000
            connection.doOutput = true

            val ctx = sliceContext
            val timestamp = System.currentTimeMillis()
            val signature = signTaste(embedding, sampleCount, versionHash, timestamp)
            val body = JSONObject().apply {
                put("screenId", deviceFingerprint)
                put("screenMongoId", screenMongoId)
                put("tasteEmbedding", JSONArray(embedding.toList()))
                put("tasteVersionHash", versionHash)
                put("tasteSampleCount", sampleCount)
                put("venueType", ctx.venueType)
                put("daypart", ctx.daypart)
                put("geo", ctx.geo)
                put("deviceProfile", ctx.deviceProfile)
                put("fingerprint", deviceFingerprint)
                put("publicKeyId", publicKeyId ?: "")
                put("signature", signature ?: "")
                put("timestamp", timestamp)
            }

            connection.outputStream.use { os ->
                OutputStreamWriter(os).use { writer ->
                    writer.write(body.toString())
                    writer.flush()
                }
            }

            val responseCode = connection.responseCode
            if (responseCode == 200) {
                Log.i(TAG, "Uploaded screen_taste embedding: 64 floats, $sampleCount samples")
            } else {
                Log.w(TAG, "screen_taste upload failed with status $responseCode")
            }
        } finally {
            connection.disconnect()
        }
    }

    /**
     * Sign the screen-taste payload with the device's Ed25519 attestation key.
     * Payload format mirrors `signVAS`: short canonical string, signed bytes,
     * base64 output. Returns null when the key pair isn't available.
     */
    private fun signTaste(
        embedding: FloatArray,
        sampleCount: Int,
        versionHash: String,
        timestamp: Long
    ): String? {
        val keyPair = attestationKeyPair ?: return null
        if (Build.VERSION.SDK_INT < 33) return null
        return try {
            // Mirror Locale.US numeric formatting so signatures verify cross-locale.
            val checksum = embedding.fold(0f) { acc, v -> acc + v }
            val payload = "taste:${String.format(Locale.US, "%.6f", checksum)}|n:$sampleCount|h:$versionHash|ts:$timestamp|fp:$deviceFingerprint"
            val signature = Signature.getInstance("Ed25519")
            signature.initSign(keyPair.private)
            signature.update(payload.toByteArray())
            Base64.encodeToString(signature.sign(), Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.w(TAG, "screen_taste signing failed: ${e.message}")
            null
        }
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
            // Surface screen-taste stats under the same per-modelType key
            // shape so `getStats()["screen_taste"]` mirrors the gradient
            // buffers. The upload-side counters (`buffered_windows`,
            // `avg_loss`) stay at zero until Phase 3 wires the gradient
            // path; the local-side counters are populated here.
            stats["screen_taste"] = mapOf(
                "buffered_windows" to 0,
                "sample_count" to tasteSampleCount,
                "avg_loss" to 0.0,
                "model_version" to tasteVersion,
                "local_adaptation_size" to (localAdaptationBuffer["screen_taste"]?.size ?: 0),
                "taste_dim" to SCREEN_TASTE_DIM,
                "taste_updated_at" to tasteUpdatedAtMs
            )
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
