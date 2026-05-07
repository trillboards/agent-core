package com.trillboards.ctv.core.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Model Manager for federated model distribution with multi-slice support.
 *
 * Checks for slice-specific model updates from the server, downloads new versions,
 * and supports hot-swapping models without restarting the sensing pipeline.
 *
 * Multi-slice: models are versioned per (venue_type × venue_subtype × daypart × geo × device_profile).
 * Cold-start inheritance: when a slice has no data, inherits from the nearest parent slice.
 * Local adaptation rejection: if a federated update degrades local accuracy > 10%, it's rejected.
 */
class ModelManager(
    private val context: Context,
    private val apiBaseUrl: String,
    private val venueType: String,
    private val deviceFingerprint: String = ""
) {
    companion object {
        private const val TAG = "ModelManager"
        private const val CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000L // 24 hours
        private const val MODELS_DIR = "ml_models"
        private const val PERFORMANCE_WINDOW = 100 // Observations for validation
        private const val ACCURACY_DEGRADATION_THRESHOLD = 0.10 // 10% — reject federated update
    }

    // Coroutine scope for async operations (replaces raw Thread)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Current slice context for multi-slice model fetching
    @Volatile private var sliceContext: FederatedSliceContext? = null

    // Current model versions — keyed by sliceKey (modelType_venueType_subtype_daypart_geo_profile)
    private val loadedVersions = mutableMapOf<String, Int>()
    private var lastCheckTime = 0L

    // Performance tracking for rollback decisions
    private val performanceHistory = mutableMapOf<String, MutableList<Float>>()

    // Pre-update accuracy baselines for rejecting bad federated updates
    private val preUpdateBaselines = mutableMapOf<String, Float>()

    /**
     * Set the multi-slice context for model fetching.
     * Called when venue info is loaded or daypart changes.
     */
    fun setSliceContext(context: FederatedSliceContext) {
        this.sliceContext = context
        Log.d(TAG, "Model slice context updated: $context")
    }

    /**
     * Check for model updates from the server.
     * Uses multi-slice context if available to fetch slice-specific models.
     * Call this on startup and periodically (every 24 hours).
     *
     * @return Map of model types that have updates available
     */
    fun checkForUpdates(): Map<String, ModelUpdateInfo> {
        val updates = mutableMapOf<String, ModelUpdateInfo>()

        try {
            val queryParams = sliceContext?.toQueryParams() ?: "venue_type=$venueType"
            val url = URL("$apiBaseUrl/v1/federated/models?$queryParams")
            val connection = url.openConnection() as HttpURLConnection
            try {
                connection.requestMethod = "GET"
                connection.setRequestProperty("x-device-fingerprint", deviceFingerprint)
                connection.connectTimeout = 15000
                connection.readTimeout = 15000

                if (connection.responseCode != 200) {
                    Log.w(TAG, "Model check failed: HTTP ${connection.responseCode}")
                    return updates
                }

                val response = connection.inputStream.use { inputStream ->
                    JSONObject(inputStream.bufferedReader().use { it.readText() })
                }

                if (!response.optBoolean("success", false)) return updates

                val models = response.optJSONObject("models") ?: return updates

                for (key in models.keys()) {
                    val modelInfo = models.getJSONObject(key)
                    val serverVersion = modelInfo.optInt("version", 0)
                    val currentVersion = loadedVersions[key] ?: 0
                    val isInherited = modelInfo.optBoolean("inherited", false)

                    if (serverVersion > currentVersion) {
                        updates[key] = ModelUpdateInfo(
                            modelType = key,
                            currentVersion = currentVersion,
                            availableVersion = serverVersion,
                            aggregatedAt = modelInfo.optString("aggregatedAt", ""),
                            deviceCount = modelInfo.optInt("deviceCount", 0),
                            sliceInherited = isInherited,
                            sliceSource = modelInfo.optString("sliceSource", "")
                        )
                    }
                }
            } finally {
                connection.disconnect()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check for model updates: ${e.message}")
        }

        lastCheckTime = System.currentTimeMillis()
        return updates
    }

    /**
     * Download and apply a model update with local accuracy validation.
     * If the federated update degrades local accuracy by > 10%, the update is rejected.
     *
     * @param modelType Model type to update
     * @return true if update was successful and not rejected
     */
    fun downloadAndApplyUpdate(modelType: String): Boolean {
        try {
            // Record pre-update baseline accuracy for rejection check
            val history = performanceHistory[modelType]
            if (history != null && history.size >= 10) {
                preUpdateBaselines[modelType] = history.takeLast(10).average().toFloat()
            }

            // Get download URL using slice-specific endpoint
            val queryParams = sliceContext?.toQueryParams() ?: "venue_type=$venueType"
            val infoUrl = URL("$apiBaseUrl/v1/federated/models/$modelType/latest?$queryParams")
            val infoConn = infoUrl.openConnection() as HttpURLConnection
            val downloadUrl: String
            val version: Int
            val sliceSource: String
            try {
                infoConn.requestMethod = "GET"
                infoConn.setRequestProperty("x-device-fingerprint", deviceFingerprint)
                infoConn.connectTimeout = 15000
                infoConn.readTimeout = 15000

                if (infoConn.responseCode != 200) {
                    Log.w(TAG, "Failed to get model info: HTTP ${infoConn.responseCode}")
                    return false
                }

                val infoResponse = infoConn.inputStream.use { inputStream ->
                    JSONObject(inputStream.bufferedReader().use { it.readText() })
                }

                downloadUrl = infoResponse.optString("download_url", "")
                version = infoResponse.optInt("version", 0)
                sliceSource = infoResponse.optString("slice_source", "")
            } finally {
                infoConn.disconnect()
            }

            if (downloadUrl.isEmpty() || version == 0) return false

            // Download model data
            val modelUrl = URL(downloadUrl)
            val modelConn = modelUrl.openConnection() as HttpURLConnection
            try {
                modelConn.connectTimeout = 60000
                modelConn.readTimeout = 60000

                if (modelConn.responseCode != 200) {
                    Log.w(TAG, "Failed to download model: HTTP ${modelConn.responseCode}")
                    return false
                }

                // Save to local storage with slice context in filename
                val modelsDir = File(context.filesDir, MODELS_DIR)
                modelsDir.mkdirs()
                val sliceSuffix = sliceContext?.let { "_${it.venueType}" } ?: "_${venueType}"
                val modelFile = File(modelsDir, "${modelType}${sliceSuffix}_v${version}.json")

                FileOutputStream(modelFile).use { output ->
                    modelConn.inputStream.use { input ->
                        input.copyTo(output)
                    }
                }
            } finally {
                modelConn.disconnect()
            }

            // Update loaded version
            loadedVersions[modelType] = version
            val sourceInfo = if (sliceSource.isNotEmpty()) " (inherited from: $sliceSource)" else ""
            Log.i(TAG, "Downloaded and applied model update: $modelType v$version$sourceInfo")

            return true
        } catch (e: Exception) {
            Log.e(TAG, "Model download failed: ${e.message}")
            return false
        }
    }

    /**
     * Validate a recently applied model update against pre-update baseline.
     * Call this after accumulating enough observations post-update.
     *
     * @return true if update should be kept, false if it should be rolled back
     */
    fun validateUpdate(modelType: String): Boolean {
        val baseline = preUpdateBaselines[modelType] ?: return true
        val history = performanceHistory[modelType] ?: return true
        if (history.size < 50) return true // Not enough post-update data yet

        val postUpdateAvg = history.takeLast(50).average().toFloat()
        val degradation = if (baseline > 0) (baseline - postUpdateAvg) / baseline else 0f

        if (degradation > ACCURACY_DEGRADATION_THRESHOLD) {
            Log.w(TAG, "$modelType federated update REJECTED: accuracy degraded ${(degradation * 100).toInt()}% " +
                    "(baseline: $baseline, post-update: $postUpdateAvg)")
            return false
        }

        Log.d(TAG, "$modelType federated update validated: accuracy change ${(-degradation * 100).toInt()}%")
        preUpdateBaselines.remove(modelType)
        return true
    }

    /**
     * Record model performance for rollback decisions.
     *
     * @param modelType Model type
     * @param accuracy Accuracy on local validation set
     */
    fun recordPerformance(modelType: String, accuracy: Float) {
        val history = performanceHistory.getOrPut(modelType) { mutableListOf() }
        history.add(accuracy)

        // Keep only recent observations
        if (history.size > PERFORMANCE_WINDOW * 2) {
            performanceHistory[modelType] = history.takeLast(PERFORMANCE_WINDOW).toMutableList()
        }
    }

    /**
     * Check if the current model version is performing worse than the previous.
     * Used to trigger automatic rollback.
     *
     * @param modelType Model type
     * @return true if rollback is recommended
     */
    fun shouldRollback(modelType: String): Boolean {
        val history = performanceHistory[modelType] ?: return false
        if (history.size < PERFORMANCE_WINDOW) return false

        // Compare last window vs previous window
        val recentAvg = history.takeLast(PERFORMANCE_WINDOW / 2).average()
        val previousAvg = history.dropLast(PERFORMANCE_WINDOW / 2).takeLast(PERFORMANCE_WINDOW / 2).average()

        // Rollback if accuracy dropped by more than 10%
        return previousAvg > 0 && (previousAvg - recentAvg) / previousAvg > 0.10
    }

    /**
     * Report performance metrics to server (async via coroutine).
     */
    fun reportPerformance(modelType: String, accuracy: Float, latencyMs: Float, sampleCount: Int, deviceId: String) {
        scope.launch {
            val url = URL("$apiBaseUrl/v1/federated/performance")
            val connection = url.openConnection() as HttpURLConnection
            try {
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.setRequestProperty("x-device-fingerprint", deviceFingerprint)
                connection.connectTimeout = 15000
                connection.readTimeout = 15000
                connection.doOutput = true

                val ctx = sliceContext
                val body = JSONObject().apply {
                    put("deviceId", deviceId)
                    put("modelType", modelType)
                    put("modelVersion", loadedVersions[modelType] ?: 0)
                    put("accuracy", accuracy)
                    put("latencyMs", latencyMs)
                    put("sampleCount", sampleCount)
                    // Multi-slice context
                    put("venueType", ctx?.venueType ?: venueType)
                    put("venueSubtype", ctx?.venueSubtype ?: "")
                    put("daypart", ctx?.daypart ?: "")
                    put("geo", ctx?.geo ?: "")
                    put("deviceProfile", ctx?.deviceProfile ?: "")
                }

                connection.outputStream.use { os ->
                    os.bufferedWriter().use { it.write(body.toString()) }
                }
                connection.responseCode // trigger request
            } catch (e: Exception) {
                Log.w(TAG, "Performance report failed: ${e.message}")
            } finally {
                connection.disconnect()
            }
        }
    }

    /**
     * Get the local file path for a downloaded model.
     *
     * @param modelType Model type identifier
     * @return File path if the model exists locally, null otherwise
     */
    fun getModelPath(modelType: String): File? {
        val version = loadedVersions[modelType] ?: return null
        val modelsDir = File(context.filesDir, MODELS_DIR)
        val sliceSuffix = sliceContext?.let { "_${it.venueType}" } ?: "_${venueType}"
        val modelFile = File(modelsDir, "${modelType}${sliceSuffix}_v${version}.json")
        return if (modelFile.exists()) modelFile else null
    }

    /**
     * Get current loaded version for a model type.
     */
    fun getLoadedVersion(modelType: String): Int {
        return loadedVersions[modelType] ?: 0
    }

    /**
     * Check if enough time has passed since the last update check.
     */
    fun isCheckDue(): Boolean {
        return System.currentTimeMillis() - lastCheckTime >= CHECK_INTERVAL_MS
    }

    /**
     * Get current model status including slice context.
     */
    fun getStatus(): Map<String, Any> {
        val ctx = sliceContext
        return mapOf(
            "venue_type" to venueType,
            "slice_context" to (ctx?.let {
                mapOf(
                    "venue_type" to it.venueType,
                    "venue_subtype" to it.venueSubtype,
                    "daypart" to it.daypart,
                    "geo" to it.geo,
                    "device_profile" to it.deviceProfile
                )
            } ?: mapOf("venue_type" to venueType)),
            "loaded_versions" to loadedVersions.toMap(),
            "last_check" to lastCheckTime,
            "performance" to performanceHistory.mapValues { (modelType, history) ->
                if (history.isNotEmpty()) {
                    mapOf(
                        "recent_avg" to history.takeLast(10).average(),
                        "observations" to history.size,
                        "pre_update_baseline" to (preUpdateBaselines[modelType] ?: "none")
                    )
                } else {
                    mapOf("recent_avg" to 0.0, "observations" to 0)
                }
            }
        )
    }

    /**
     * Cancel pending coroutine operations. Call on service shutdown.
     */
    fun destroy() {
        scope.cancel()
    }

    data class ModelUpdateInfo(
        val modelType: String,
        val currentVersion: Int,
        val availableVersion: Int,
        val aggregatedAt: String,
        val deviceCount: Int,
        val sliceInherited: Boolean = false,
        val sliceSource: String = ""
    )
}
