package com.trillboards.ctv.core.audience

import android.content.Context
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
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
        val programSpecVersion: String? = null,
        // Raw sensing_config block from the push payload. Survives restart so
        // SensingConfig overrides (speech.useLlmExtractor, vlm.* thresholds,
        // engagement weights, etc.) re-apply on cold boot rather than reverting
        // to Kotlin defaults. JSON string instead of parsed map because the
        // schema is open-ended (any flag a future profile pushes). Null when
        // the profile didn't include sensing_config (legacy path).
        val sensingConfigJson: String? = null,
        // Raw JSON array string of `program_spec.observation_program.signals`.
        // Survives restart so OnDeviceLlmInsightExtractor can rebuild its
        // OpenApiTool spec from profile signals on cold boot without waiting
        // for the server's profile push. Null when the profile did not declare
        // any signals (legacy profiles — back-compat preserved).
        val signalsJson: String? = null
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
            // PR π (2026-06-01) — include sensingConfigJson in the duplicate-key
            // comparison. Before this change, a WS push or in-process restart
            // that re-sent the same profile_id with ONLY a sensing_config delta
            // (e.g. operator flipped speech.shadowLlmExtractor false→true) would
            // short-circuit here BEFORE applySensingConfigOverrides ran, so the
            // newly-pushed flags silently never took effect on already-running
            // devices. Per Codex P1 on PR #6498 — both the persistence path
            // (createProfile) and the apply path must respect sensing_config or
            // the cold-boot fix is half-finished.
            if (current != null &&
                current.profileId == validated.profileId &&
                current.models == validated.models &&
                current.classes == validated.classes &&
                current.captureIntervalMs == validated.captureIntervalMs &&
                current.reportIntervalMs == validated.reportIntervalMs &&
                current.vlmPrompt == validated.vlmPrompt &&
                current.programSpecJson == validated.programSpecJson &&
                current.programSpecVersion == validated.programSpecVersion &&
                current.signalsJson == validated.signalsJson &&
                current.sensingConfigJson == validated.sensingConfigJson
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

            // Apply sensing_config overrides pushed with this profile (PR L3+).
            // Follows the same SensingConfig.updateFromJson path as sensing_config_update
            // Socket.IO events so remote-pushable flags (e.g. speech.useLlmExtractor) take
            // effect without requiring a separate config push event.
            applySensingConfigOverrides(profileJson)

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
     * Apply the [sensing_config] sub-object from the profile payload via
     * [SensingConfig.updateFromJson]. This allows the server to push
     * section-keyed sensing flags (e.g. `{ speech: { useLlmExtractor: false } }`)
     * alongside the sensing profile without requiring a separate
     * `sensing_config_update` Socket.IO event.
     *
     * No-ops silently when [sensing_config] is absent or malformed —
     * the profile apply still succeeds.
     */
    private fun applySensingConfigOverrides(profileJson: JSONObject) {
        val sensingConfigJson = profileJson.optJSONObject("sensing_config") ?: return
        try {
            SensingConfig.updateFromJson(sensingConfigJson)
            Log.i(TAG, "Applied sensing_config overrides from profile: " +
                sensingConfigJson.keys().asSequence().toList())
        } catch (e: Exception) {
            Log.w(TAG, "Failed to apply sensing_config overrides from profile: ${e.message}")
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

        // Capture sensing_config verbatim so we can reapply on cold boot via
        // loadPersistedProfile() — without this, SensingConfig.updateFromJson
        // overrides (speech.useLlmExtractor, vlm.* thresholds, engagement
        // weights, etc.) silently revert to Kotlin defaults on every restart.
        val sensingConfigJsonString = json.optJSONObject("sensing_config")?.toString()

        // Extract signals[] from program_spec.observation_program.signals.
        // These are the operator-declared field declarations for both speech and
        // vision. Persisted verbatim so OnDeviceLlmInsightExtractor can rebuild
        // its OpenApiTool spec on cold boot without a server round-trip.
        val signalsJsonString = extractSignalsJson(json)

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
            programSpecVersion = observationProgram.programSpecVersion,
            sensingConfigJson = sensingConfigJsonString,
            signalsJson = signalsJsonString
        )
    }

    /**
     * Extract the speech-source field declarations from the push payload JSON.
     *
     * PR ρ unified speech + vision declarations onto a single `metrics_schema`
     * surface (each entry carrying its own `source` discriminator). The
     * preferred read is `program_spec.output_contract.metrics_schema` filtered
     * to `source=='speech'`. For back-compat with pre-ρ pushes still in the
     * fleet's persisted-profile cache, we fall back to the deprecated
     * `program_spec.observation_program.signals` array.
     *
     * Returns the speech entries serialized as a JSON array string so the
     * value can be persisted verbatim and later parsed by
     * [SignalsToToolSpecBuilder.parseSignalsArray]. Returns null when no
     * speech entries are present on either surface.
     */
    private fun extractSignalsJson(json: JSONObject): String? {
        return try {
            val programSpec = json.optJSONObject("program_spec") ?: return null
            // Primary surface (PR ρ): metrics_schema filtered to source='speech'.
            val outputContract = programSpec.optJSONObject("output_contract")
            val metricsSchema = outputContract?.optJSONArray("metrics_schema")
            if (metricsSchema != null && metricsSchema.length() > 0) {
                val speechOnly = JSONArray()
                for (i in 0 until metricsSchema.length()) {
                    val entry = metricsSchema.optJSONObject(i) ?: continue
                    val source = entry.optString("source", "vision")
                    if (source == "speech") speechOnly.put(entry)
                }
                if (speechOnly.length() > 0) return speechOnly.toString()
            }
            // Back-compat surface (pre-ρ): observation_program.signals[].
            // Persisted profiles flashed onto the device before PR ρ rolls
            // out will still arrive here until the server re-pushes them.
            val observationProgram = programSpec.optJSONObject("observation_program")
            val signals = observationProgram?.optJSONArray("signals") ?: return null
            if (signals.length() == 0) null else signals.toString()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract speech signals from profile: ${e.message}")
            null
        }
    }

    /**
     * Map model IDs to their asset file paths for existence validation.
     * - blazeface: Uses ML Kit (Google Play Services) -- no on-device file needed
     * - age_gender: Requires age_gender_model.tflite (optional, may not be bundled)
     * - movenet: pose_landmarker_lite.task
     * - yamnet: yamnet.tflite
     * - whisper_tiny: moonshine-tiny/ ONNX directory (check tokens.txt as sentinel)
     * - efficientdet: efficientdet_lite0.tflite
     * - yolov8_nano: NO FILE -- not available on device
     *
     * NOTE: fer_plus has no dedicated asset path since Phase 3 — its emotion
     * outputs are produced by FaceLandmarkerProcessor (face_landmarker.task)
     * alongside gaze + head-pose. The wire-format model_id is retained for
     * API compatibility; profiles that list it succeed via the no-asset path.
     */
    private fun getModelAssetPath(modelId: String): String? = when (modelId) {
        "blazeface" -> null // ML Kit, no local file
        "age_gender" -> "age_gender_model.tflite"
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
                "gemma_4_e2b", "moondream_05b", "smolvlm_256m" -> capabilities.hasCameraHardware
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
                // Persist sensing_config verbatim so loadPersistedProfile() can
                // re-apply SensingConfig overrides on cold boot (PR #6386 fix:
                // useLlmExtractor / shadowLlmExtractor / vlm.* etc. were silently
                // reverting to Kotlin defaults on every app restart).
                profile.sensingConfigJson?.let { put("sensing_config", JSONObject(it)) }
                // Persist signals[] verbatim so OnDeviceLlmInsightExtractor can
                // rebuild its OpenApiTool spec on cold boot without a server push.
                profile.signalsJson?.let { put("signals", JSONArray(it)) }
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
            val profileJsonObj = JSONObject(json)
            val profile = parseProfile(profileJsonObj)

            // Re-apply sensing_config overrides on cold boot. Without this every
            // restart silently reverted SensingConfig overrides (useLlmExtractor,
            // vlm.* thresholds, engagement weights) to Kotlin defaults — the
            // device kept the persisted profile metadata but quietly dropped any
            // feature flag the server had pushed. Verified live on Tab S11
            // (2026-05-29): after a force-stop the L3-wired LLM extractor never
            // activated because useLlmExtractor=false on cold boot.
            //
            // Idempotency: applySensingConfigOverrides parses the JSONObject and
            // calls SensingConfig.updateFromJson which merges over current state;
            // calling twice in a row (once on cold boot, once again when the
            // server's profile push arrives moments later) is a no-op.
            applySensingConfigOverrides(profileJsonObj)

            Log.i(TAG, "Loaded persisted profile: ${profile.profileName}" +
                if (profile.sensingConfigJson != null) " (with sensing_config overrides)" else "")
            profile
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load persisted profile", e)
            null
        }
    }

    /**
     * PR π (2026-06-01) — Server-first bootstrap.
     *
     * Cold-boot ordering FIX: previously [BaseDeviceAgentService.startAudienceSensing]
     * called [loadPersistedProfile] as the source-of-truth and only got the server's
     * version via a later WS push (`updateSensingProfile` command). Race condition:
     *
     *   1. Operator updates the profile (deploy-profile POST → PG row updated → WS push
     *      queued via deviceCommandDispatcher with persistent=true).
     *   2. Device cold-boots and immediately persists + applies the STALE
     *      "soak_coffee_shop_v1" because the WS push has not arrived yet — or arrives
     *      but is suppressed by `applyProfile`'s "skip duplicate" gate when the cached
     *      shape happens to match.
     *   3. Operator's intent never lands. Verified on Tab S11 (2026-06-01): logcat
     *      showed `Loaded persisted profile: soak_coffee_shop_v1` repeatedly even
     *      after the operator pushed a different intent.
     *
     * Fix: this method runs at startup BEFORE loadPersistedProfile. It hits
     * `/v2/earner/sensing/profiles/:screenId`, picks the FIRST active profile (the
     * route returns them ORDER BY updated_at DESC server-side), and applies it via
     * the standard `applyProfile` path. The same WS push subsystem keeps working
     * for run-time updates; this is the cold-start authority.
     *
     * Fallback semantics:
     *   * `fetcher` returning a JSONObject — server-authoritative; apply + persist.
     *     If a persisted profile with a DIFFERENT profile_id was already on disk,
     *     it is overwritten transparently by applyProfile → persistProfile.
     *   * `fetcher` returning null (network down, 5xx, screen not found, empty
     *     active set) — caller's responsibility to fall back to `loadPersistedProfile`.
     *     Returning null here signals "use the persisted recovery path".
     *
     * The `fetcher` signature is intentionally minimal (suspend (screenId) -> JSONObject?)
     * so SensingProfileManager doesn't take an ApiClient dependency — the agent-core
     * module already owns ApiClient elsewhere, and decoupling here keeps unit tests
     * trivial (Kotlin lambda mock, no okhttp).
     */
    suspend fun fetchAndApplyServerProfile(
        screenId: String,
        capabilities: DeviceCapabilities,
        fetcher: suspend (String) -> JSONObject?
    ): ApplyResult? {
        if (screenId.isBlank()) {
            Log.d(TAG, "fetchAndApplyServerProfile skipped: screenId blank")
            return null
        }
        val serverProfile = try {
            fetcher(screenId)
        } catch (e: Exception) {
            Log.w(TAG, "fetchAndApplyServerProfile network error: ${e.message}")
            null
        }
        if (serverProfile == null) {
            Log.i(TAG, "Server profile fetch returned null for screen=$screenId — caller should fall back to persisted")
            return null
        }

        // PR π — invalidate persisted cache when the server's profile_id differs
        // from what's on disk. The applyProfile() path below would do this anyway
        // (persistProfile overwrites), but the explicit log marker lets us
        // diagnose stale-cache cases from logcat without inspecting prefs.
        val persistedId = loadPersistedProfileIdOnly()
        val serverId = serverProfile.optString("profile_id", "")
        if (!persistedId.isNullOrEmpty() && serverId.isNotEmpty() && persistedId != serverId) {
            Log.w(TAG, "Persisted profile cache stale (persisted=$persistedId, server=$serverId) — overwriting")
        }

        val applyResult = applyProfile(serverProfile, capabilities)
        if (!applyResult.success) {
            Log.w(TAG, "applyProfile failed in fetchAndApplyServerProfile: ${applyResult.error}")
        }
        return applyResult
    }

    /**
     * Read just the persisted profile's profile_id (cheap parse for stale-cache
     * detection in [fetchAndApplyServerProfile]). Returns null when nothing was
     * persisted or when the JSON cannot be parsed.
     */
    fun loadPersistedProfileIdOnly(): String? {
        return try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val json = prefs.getString(KEY_ACTIVE_PROFILE, null) ?: return null
            JSONObject(json).optString("profile_id", "").ifEmpty { null }
        } catch (e: Exception) {
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
