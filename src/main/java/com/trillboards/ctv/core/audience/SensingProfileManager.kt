package com.trillboards.ctv.core.audience

import android.content.Context
import android.util.Log
import com.trillboards.ctv.core.calibration.DeploymentCalibration
import com.trillboards.ctv.core.calibration.DerivedThresholds
import com.trillboards.ctv.core.ml.ModelDownloadManager
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicReference

/**
 * Manages sensing profiles received from the server.
 * Validates against device tier/capabilities and reconfigures the inference graph.
 *
 * Lifecycle: Server sends `updateSensingProfile` command via Socket.IO ->
 * SensingProfileManager validates -> applies to AudienceSensingService -> ACKs
 *
 * Thread-safe: AtomicReference for active profile, synchronized listener list.
 * Persists the active profile to SharedPreferences for offline boot resilience.
 */
class SensingProfileManager(
    private val context: Context,
    private val fingerprint: String
) {
    companion object {
        private const val TAG = "SensingProfileManager"
        private const val PREFS_NAME = "sensing_profiles"
        private const val KEY_ACTIVE_PROFILE = "active_profile"
    }

    /**
     * A sensing profile defines what models to run, which detection classes to emit,
     * at what cadence, and grouped into which observation families.
     */
    data class SensingProfile(
        val profileId: String,
        val profileName: String,
        val models: List<String>,
        val classes: List<String>,
        val thresholds: Map<String, Any>,
        val observationFamilies: List<String>,
        val captureIntervalMs: Int = 10000,
        val reportIntervalMs: Int = 30000,
        val vlmPrompt: String? = null,
        val programSpecJson: String? = null,
        val programSpecVersion: String? = null
    )

    /**
     * Hardware capability snapshot for a device.
     * Used to validate which models from a profile the device can actually run.
     */
    data class DeviceCapabilities(
        val hasCameraHardware: Boolean,
        val hasMicrophoneHardware: Boolean,
        val hasGpu: Boolean = false,
        val maxMemoryMb: Int = 512,
        val deviceTier: String = "standard"
    )

    /**
     * Result of applying a sensing profile. Reports which models were accepted
     * vs dropped due to hardware limitations.
     */
    data class ApplyResult(
        val success: Boolean,
        val profileId: String? = null,
        val appliedModels: List<String> = emptyList(),
        val droppedModels: List<String> = emptyList(),
        val programSpecVersion: String? = null,
        val error: String? = null
    )

    private val activeProfile = AtomicReference<SensingProfile?>(null)
    private val profileListeners = mutableListOf<(SensingProfile) -> Unit>()
    private val deploymentThresholds = AtomicReference<DerivedThresholds?>(null)

    fun getActiveProfile(): SensingProfile? = activeProfile.get()

    fun addProfileListener(listener: (SensingProfile) -> Unit) {
        synchronized(profileListeners) { profileListeners.add(listener) }
    }

    fun removeProfileListener(listener: (SensingProfile) -> Unit) {
        synchronized(profileListeners) { profileListeners.remove(listener) }
    }

    /**
     * Called when server sends `updateSensingProfile` command.
     * Validates the profile against device capabilities before applying.
     *
     * @param profileJson Raw JSON payload from the server
     * @param capabilities Current device hardware capabilities
     * @return ApplyResult with success/failure and model disposition
     */
    fun applyProfile(profileJson: JSONObject, capabilities: DeviceCapabilities): ApplyResult {
        try {
            val profile = parseProfile(profileJson)
            val validated = validateAgainstCapabilities(profile, capabilities)
            val current = activeProfile.get()
            if (current != null &&
                current.profileId == validated.profileId &&
                current.models == validated.models &&
                current.classes == validated.classes &&
                current.captureIntervalMs == validated.captureIntervalMs &&
                current.reportIntervalMs == validated.reportIntervalMs &&
                current.vlmPrompt == validated.vlmPrompt &&
                current.programSpecJson == validated.programSpecJson &&
                current.programSpecVersion == validated.programSpecVersion
            ) {
                Log.i(TAG, "Skipping duplicate sensing profile: ${validated.profileName}")
                return ApplyResult(
                    success = true,
                    profileId = validated.profileId,
                    appliedModels = validated.models,
                    programSpecVersion = validated.programSpecVersion
                )
            }

            activeProfile.set(validated)
            persistProfile(validated)

            // Apply deployment calibration if present in the profile's thresholds
            applyDeploymentCalibration(profileJson)

            // Notify listeners (AudienceSensingService will reconfigure)
            synchronized(profileListeners) {
                profileListeners.forEach { listener ->
                    try {
                        listener(validated)
                    } catch (e: Exception) {
                        Log.w(TAG, "Profile listener threw exception", e)
                    }
                }
            }

            Log.i(TAG, "Applied sensing profile: ${validated.profileName} " +
                "(${validated.models.size} models, ${validated.classes.size} classes)")

            return ApplyResult(
                success = true,
                profileId = validated.profileId,
                appliedModels = validated.models,
                droppedModels = profile.models - validated.models.toSet(),
                programSpecVersion = validated.programSpecVersion
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to apply sensing profile", e)
            return ApplyResult(success = false, error = e.message)
        }
    }

    /**
     * Get the derived deployment thresholds, if deployment calibration has been applied.
     */
    fun getDeploymentThresholds(): DerivedThresholds? = deploymentThresholds.get()

    /**
     * Extract deployment_config from the profile payload and apply it via
     * DeploymentCalibration. Checks both top-level and nested in thresholds.
     *
     * Expected JSON locations (checked in order):
     *   1. profileJson.deployment_config  (top-level)
     *   2. profileJson.thresholds.deployment_config  (nested in thresholds map)
     */
    private fun applyDeploymentCalibration(profileJson: JSONObject) {
        val deploymentJson = profileJson.optJSONObject("deployment_config")
            ?: profileJson.optJSONObject("thresholds")?.optJSONObject("deployment_config")

        if (deploymentJson == null) {
            Log.d(TAG, "No deployment_config in profile -- skipping deployment calibration")
            return
        }

        try {
            val derived = DeploymentCalibration.fromJsonAndApply(deploymentJson)
            if (derived != null) {
                deploymentThresholds.set(derived)
                Log.i(TAG, "Deployment calibration applied: yaw=${derived.yawThresholdDeg}deg, " +
                    "pitch=${derived.pitchThresholdDeg}deg")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to apply deployment calibration", e)
        }
    }

    /**
     * Parse a SensingProfile from server JSON.
     */
    private fun parseProfile(json: JSONObject): SensingProfile {
        val models = mutableListOf<String>()
        val modelsArray = json.optJSONArray("models") ?: JSONArray()
        for (i in 0 until modelsArray.length()) {
            models.add(modelsArray.getString(i))
        }

        val classes = mutableListOf<String>()
        val classesArray = json.optJSONArray("classes") ?: JSONArray()
        for (i in 0 until classesArray.length()) {
            classes.add(classesArray.getString(i))
        }

        val families = mutableListOf<String>()
        val familiesArray = json.optJSONArray("observation_families") ?: JSONArray()
        for (i in 0 until familiesArray.length()) {
            families.add(familiesArray.getString(i))
        }

        val thresholds = mutableMapOf<String, Any>()
        val thresholdsJson = json.optJSONObject("thresholds")
        if (thresholdsJson != null) {
            thresholdsJson.keys().forEach { key ->
                thresholds[key] = thresholdsJson.get(key)
            }
        }
        val observationProgram = extractObservationProgramMetadata(json)

        return SensingProfile(
            profileId = json.optString("profile_id", "unknown"),
            profileName = json.optString("profile_name", "custom"),
            models = models,
            classes = classes,
            thresholds = thresholds,
            observationFamilies = families,
            captureIntervalMs = json.optInt("capture_interval_ms", 10000),
            reportIntervalMs = json.optInt("report_interval_ms", 30000),
            vlmPrompt = json.optString("vlm_prompt", "").takeIf { it.isNotBlank() },
            programSpecJson = observationProgram.programSpecJson,
            programSpecVersion = observationProgram.programSpecVersion
        )
    }

    /**
     * Map model IDs to their asset file paths for existence validation.
     * - blazeface: Uses ML Kit (Google Play Services) -- no on-device file needed
     * - age_gender: Requires age_gender_model.tflite (optional, may not be bundled)
     * - fer_plus: fer_emotion.tflite
     * - movenet: pose_landmarker_lite.task
     * - yamnet: yamnet.tflite
     * - whisper_tiny: moonshine-tiny/ ONNX directory (check tokens.txt as sentinel)
     * - efficientdet: efficientdet_lite0.tflite
     * - yolov8_nano: NO FILE -- not available on device
     */
    private fun getModelAssetPath(modelId: String): String? = when (modelId) {
        "blazeface" -> null // ML Kit, no local file
        "age_gender" -> "age_gender_model.tflite"
        "fer_plus" -> "fer_emotion.tflite"
        "movenet" -> "pose_landmarker_lite.task"
        "yamnet" -> "yamnet.tflite"
        "whisper_tiny" -> "moonshine-tiny/tokens.txt"
        "efficientdet" -> "efficientdet_lite0.tflite"
        "yolov8_nano" -> null // No model file exists -- marked unavailable below
        else -> null
    }

    /**
     * Check if a model's asset file actually exists on the device.
     * Checks three sources in order:
     *   1. APK-bundled assets (existing)
     *   2. OTA-downloaded model binaries on filesystem
     *   3. ML Kit / cloud-backed models that need no local file (e.g. blazeface)
     * Returns false for yolov8_nano which has no model file available.
     */
    private fun isModelAssetAvailable(modelId: String): Boolean {
        // yolov8_nano has no model file -- always unavailable
        if (modelId == "yolov8_nano") {
            Log.d(TAG, "Model $modelId has no model file available on device")
            return false
        }

        // 1. Check APK-bundled assets
        val assetPath = getModelAssetPath(modelId)
        if (assetPath != null) {
            try {
                context.assets.open(assetPath).close()
                return true
            } catch (_: Exception) {
                // Asset not bundled in APK, continue to OTA check
            }
        }

        // 2. Check OTA-downloaded models on filesystem
        val downloadManager = ModelDownloadManager(context)
        if (downloadManager.isModelAvailable(modelId)) {
            Log.d(TAG, "Model $modelId available via OTA download")
            return true
        }

        // 3. Models with no asset path that use cloud/ML Kit (e.g. blazeface)
        if (assetPath == null) return true

        Log.d(TAG, "Model $modelId not found in APK assets or OTA downloads")
        return false
    }

    /**
     * Get the actual file path for loading a model at runtime.
     * Returns an "asset://" URI for APK-bundled models, or an absolute filesystem
     * path for OTA-downloaded models.
     *
     * @param modelId Model identifier (e.g. "yamnet", "efficientdet")
     * @return Path string suitable for model loading, or null if not available
     */
    fun getModelFilePath(modelId: String): String? {
        // 1. APK asset path
        val assetPath = getModelAssetPath(modelId)
        if (assetPath != null) {
            try {
                context.assets.open(assetPath).close()
                return "asset://$assetPath"
            } catch (_: Exception) {
                // Not in APK, fall through to OTA
            }
        }

        // 2. OTA-downloaded model path
        val otaPath = ModelDownloadManager(context).getModelPath(modelId)
        if (otaPath != null) {
            return otaPath.absolutePath
        }

        return null
    }

    /**
     * Drop models the device can't run. Validates both hardware requirements
     * AND asset file existence. If all models are dropped but camera is
     * available, falls back to blazeface for basic face detection (the
     * minimum viable sensing).
     */
    private fun validateAgainstCapabilities(
        profile: SensingProfile,
        capabilities: DeviceCapabilities
    ): SensingProfile {
        val validModels = profile.models.filter { modelId ->
            // First check hardware requirements
            val hasHardware = when (modelId) {
                "blazeface" -> capabilities.hasCameraHardware
                "age_gender" -> capabilities.hasCameraHardware
                "fer_plus" -> capabilities.hasCameraHardware
                "movenet" -> capabilities.hasCameraHardware
                "yamnet" -> capabilities.hasMicrophoneHardware
                "whisper_tiny" -> capabilities.hasMicrophoneHardware
                "efficientdet" -> capabilities.hasCameraHardware
                "yolov8_nano" -> capabilities.hasCameraHardware
                // VLM models — require camera for frame capture
                "gemma_3n_e2b", "gemma_4_e2b", "moondream_05b", "smolvlm_256m" -> capabilities.hasCameraHardware
                else -> {
                    Log.w(TAG, "Unknown model ID: $modelId, skipping")
                    false
                }
            }

            if (!hasHardware) {
                Log.d(TAG, "Model $modelId not valid for device capabilities " +
                    "(camera=${capabilities.hasCameraHardware}, mic=${capabilities.hasMicrophoneHardware})")
                return@filter false
            }

            // Then check asset file existence
            val hasAsset = isModelAssetAvailable(modelId)
            if (!hasAsset) {
                Log.d(TAG, "Model $modelId skipped: asset file not present on device")
            }
            hasAsset
        }

        // Ensure at least blazeface if camera available (minimum viable sensing)
        val finalModels = if (validModels.isEmpty() && capabilities.hasCameraHardware) {
            Log.i(TAG, "No valid models from profile, falling back to blazeface")
            listOf("blazeface")
        } else {
            validModels
        }

        return profile.copy(models = finalModels)
    }

    /**
     * Persist profile to SharedPreferences for offline boot resilience.
     * On next cold start, the device can resume with the last known profile
     * before the server reconnects.
     */
    private fun persistProfile(profile: SensingProfile) {
        try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val json = JSONObject().apply {
                put("profile_id", profile.profileId)
                put("profile_name", profile.profileName)
                put("models", JSONArray(profile.models))
                put("classes", JSONArray(profile.classes))
                put("observation_families", JSONArray(profile.observationFamilies))
                put("capture_interval_ms", profile.captureIntervalMs)
                put("report_interval_ms", profile.reportIntervalMs)
                // Persist thresholds
                val thresholdsObj = JSONObject()
                profile.thresholds.forEach { (key, value) ->
                    thresholdsObj.put(key, value)
                }
                put("thresholds", thresholdsObj)
                profile.vlmPrompt?.let { put("vlm_prompt", it) }
                appendObservationProgramMetadata(
                    target = this,
                    programSpecJson = profile.programSpecJson,
                    programSpecVersion = profile.programSpecVersion
                )
            }
            prefs.edit().putString(KEY_ACTIVE_PROFILE, json.toString()).apply()
            Log.d(TAG, "Persisted profile ${profile.profileId} to SharedPreferences")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to persist profile", e)
        }
    }

    /**
     * Load the last persisted profile from SharedPreferences.
     * Returns null if no profile was previously persisted or if parsing fails.
     */
    fun loadPersistedProfile(): SensingProfile? {
        return try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val json = prefs.getString(KEY_ACTIVE_PROFILE, null) ?: return null
            val profile = parseProfile(JSONObject(json))
            Log.i(TAG, "Loaded persisted profile: ${profile.profileName}")
            profile
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load persisted profile", e)
            null
        }
    }

    /**
     * Clear the active profile and persisted state. Used when the server
     * explicitly removes the profile assignment.
     */
    fun clearProfile() {
        activeProfile.set(null)
        try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit().remove(KEY_ACTIVE_PROFILE).apply()
            Log.i(TAG, "Cleared active profile")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to clear persisted profile", e)
        }
    }

    /**
     * Convert an ApplyResult to JSON for Socket.IO ACK back to server.
     */
    fun applyResultToJson(result: ApplyResult): JSONObject {
        return JSONObject().apply {
            put("success", result.success)
            put("fingerprint", fingerprint)
            if (result.profileId != null) put("profile_id", result.profileId)
            if (result.appliedModels.isNotEmpty()) put("applied_models", JSONArray(result.appliedModels))
            if (result.droppedModels.isNotEmpty()) put("dropped_models", JSONArray(result.droppedModels))
            if (result.programSpecVersion != null) put("program_spec_version", result.programSpecVersion)
            if (result.error != null) put("error", result.error)
        }
    }
}
