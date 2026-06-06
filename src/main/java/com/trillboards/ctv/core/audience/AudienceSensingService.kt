package com.trillboards.ctv.core.audience

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.trillboards.ctv.core.socket.AgentSocketManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import com.trillboards.ctv.core.calibration.CalibratedConfidence
import com.trillboards.ctv.core.calibration.CalibrationTelemetry
import com.trillboards.ctv.core.calibration.ConfidenceCalibrator
import com.trillboards.ctv.core.calibration.SignalBundle
import com.trillboards.ctv.core.inference.HardwareManifest
import com.trillboards.ctv.core.inference.InferenceOutput
import com.trillboards.ctv.core.inference.vlm.PerceptionTrigger
import com.trillboards.ctv.core.inference.vlm.VLMInferenceProcessor
import com.trillboards.ctv.core.inference.vlm.VlmFrameAdmissionGate
import com.trillboards.ctv.core.ml.FederatedTrainer
import com.trillboards.ctv.core.ml.ModelManager
import com.trillboards.ctv.core.sensing.CsiAggregator
import com.trillboards.ctv.core.sensing.CsiNodeDiscovery
import com.trillboards.ctv.core.stability.MemoryAttenuationManager
import com.trillboards.ctv.core.stability.SubsystemHealthRegistry
import java.util.UUID

internal fun <T> swapReleasedReference(current: T?, next: T?, release: (T) -> Unit): T? {
    if (current != null && current !== next) {
        release(current)
    }
    return next
}

internal fun <T> wireExistingCaptureIfReady(
    isCameraBound: Boolean,
    existingCapture: T?,
    wireCapture: (T) -> Unit
): Boolean {
    if (!isCameraBound || existingCapture == null) {
        return false
    }

    wireCapture(existingCapture)
    return true
}

/**
 * CRITICAL-only memory backstop decision (pure, so it is unit-tested without an
 * Android Context). Returns CRITICAL only when the device is at the OOM brink (lmkd
 * kill imminent), otherwise NORMAL.
 *
 * We deliberately do NOT escalate at MEDIUM/HIGH pressure: a 3-4 GB CTV device idles
 * near those thresholds and the product requirement is to "keep the zoo" (full
 * sensing) until genuinely about to crash. Only at CRITICAL do we shed enrichment
 * models to survive — face + YAMNet + VAS stay on at every tier, so the edge->cloud
 * attestation signal is preserved.
 */
internal fun criticalMemoryBackstopTier(
    lowMemory: Boolean,
    availableRamPercent: Float,
    criticalThresholdPercent: Float = 15f
): MemoryAttenuationManager.AttenuationTier =
    if (lowMemory || availableRamPercent < criticalThresholdPercent) {
        MemoryAttenuationManager.AttenuationTier.CRITICAL
    } else {
        MemoryAttenuationManager.AttenuationTier.NORMAL
    }

/**
 * Orchestrates audience sensing by coordinating face detection and audio classification.
 *
 * Features:
 * - Hardware detection with graceful degradation
 * - Combined metrics aggregation (30-second windows)
 * - Socket.io emission to backend
 * - Privacy-preserving (no raw data stored)
 * - Gemini Vision demographics via FrameCaptureManager (every 5 min)
 *
 * Mixed deployment support:
 * - Camera-less devices: Audio-only metrics
 * - Mic-less devices: Face-only metrics
 * - No sensors: Device still functions, no audience data
 */
class AudienceSensingService(
    private val context: Context,
    // Mutable so the SDK can adopt the canonical device fingerprint returned by
    // check-screen (setAudienceFingerprint) once the screen binding resolves —
    // a one-time startup update; reference assignment is atomic.
    private var fingerprint: String,
    private val socketManager: AgentSocketManager,
    private val apiBaseUrl: String = "https://api.trillboards.com",
    private val deviceTokenProvider: () -> String? = { null }
) {
    companion object {
        private const val TAG = "AudienceSensing"
        private const val AGGREGATION_WINDOW_MS = 10_000L  // 10 seconds
        private const val MIN_FACES_TO_REPORT = 0  // Report even with 0 faces (for venue occupancy tracking)
        private const val MIN_FACES_FOR_DEMOGRAPHICS = 1  // Trigger demographics capture when at least 1 face
    }

    // Memory attenuation manager
    private var attenuationManager: MemoryAttenuationManager? = null
    private var originalAggregationWindowMs = AGGREGATION_WINDOW_MS

    private val sensingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Sub-processors
    private var audienceAnalyzer: AudienceAnalyzer? = null
    private var audioProcessor: AudioClassificationProcessor? = null
    private var speechProcessor: SpeechIntelligenceProcessor? = null
    /** Eagerly-initialized LLM extractor (PR L3). Stored here so it is passed to
     *  [SpeechIntelligenceProcessor] and lives for the service lifetime. */
    @Volatile private var onDeviceLlmExtractor: OnDeviceLlmInsightExtractor? = null
    private var frameCaptureManager: FrameCaptureManager? = null

    // VLM perception-triggered inference (Phase 5B)
    private val perceptionTrigger = PerceptionTrigger()
    private var vlmProcessor: VLMInferenceProcessor? = null
    @Volatile private var lastVlmOutput: InferenceOutput? = null
    @Volatile private var vlmFrameJob: Job? = null
    private val vlmFrameAdmissionGate = VlmFrameAdmissionGate()
    private val vlmDownloadLock = Any()
    @Volatile private var vlmDownloadInFlightModelId: String? = null

    // Phase 2 bridge — age/gender foundation-model OTA download lock.
    // Mirrors the VLM lock so concurrent profile-applies don't trigger
    // duplicate downloads of the foundation classifier.
    private val ageGenderDownloadLock = Any()
    @Volatile private var ageGenderDownloadInFlight = false

    // Phase 4 PR 2 — clip-trigger evaluator. PR 3 (this PR) wires the recorder.
    // Active rules are parsed out of the live profile's `metrics_schema` whenever
    // setActiveProgramSpec lands. Evaluator is pure-function; debounce state
    // resets when the active profile changes.
    private val clipTriggerEvaluator = ClipTriggerEvaluator()
    @Volatile private var activeClipTriggerRules: List<ClipTriggerEvaluator.TriggerRule> = emptyList()

    // Phase 4 PR 3 — clip recorder + uploader (DRY-RUN path; PR 4 will gate the
    // actual Gemini call behind this). Both are lazy-init'd on first use so a
    // device with no triggers active never spins them up.
    @Volatile private var clipRecorder: ClipRecorder? = null
    @Volatile private var clipUploadManager: ClipUploadManager? = null

    /** Test seam — let unit tests inject a fake recorder/upload pair. */
    internal fun installClipPipelineForTest(
        recorder: ClipRecorder?,
        uploader: ClipUploadManager?,
    ) {
        clipRecorder = recorder
        clipUploadManager = uploader
    }

    // Multi-signal confidence calibration (Phase 3)
    private val confidenceCalibrator = ConfidenceCalibrator()

    // TFLite snapshot ring buffer — provides temporal context to the VLM prompt.
    // Holds the last 3 aggregation-cycle snapshots so the VLM can see trends
    // (crowd building, winding down, peak, idle) across ~30s of sensor history.
    private val tfliteSnapshots = ArrayDeque<VLMInferenceProcessor.TFLiteSnapshot>(4) // keep max 3

    // Latest demographics from Gemini Vision
    private var latestDemographics: DemographicsResult? = null
    @Volatile private var lastEdgeQualityTelemetry: EdgeQualityTelemetry? = null

    // Mirror of the last per_face_observations array emitted on the
    // audienceSignals WS feed, snapshotted at drain time. Consumed by
    // FrameCaptureManager's HTTP POST (perFaceSnapshotProvider) so the cloud
    // persistence layer — which reads from /v2/earner/audience-analyze, not
    // the WS feed — actually receives per-(face × creative) attribution rows.
    // Without this bridge, the WS emitted them but the HTTP path dropped
    // them, leading to fleet-wide 0% per_face_observations fill. Stays as
    // last-known until the next drain. Empty JSONArray when no faces × ad
    // attribution has happened yet this device session.
    @Volatile private var latestPerFaceSnapshot: org.json.JSONArray = org.json.JSONArray()

    // Cached HardwareManifest — hardware doesn't change at runtime, detect once
    @Volatile private var cachedHardwareManifest: HardwareManifest? = null

    // Hardware capabilities
    private var hasCameraHardware = false
    private var hasMicrophoneHardware = false
    private var hasCameraPermission = false
    private var hasMicrophonePermission = false

    // Camera availability (updated after camera detection)
    private var cameraStatus: CameraStatus = CameraStatus(available = false, type = "none", count = 0)

    // Camera health heartbeat callback — wired from DeviceAgentService to
    // SubsystemHealthRegistry.updateHealth(CAMERA, ACTIVE). Called on every
    // successfully processed frame to prevent RecoveryLadder false-positive
    // "camera failed" detection.
    var onCameraHealthHeartbeat: (() -> Unit)? = null

    // Microphone health heartbeat callback — same pattern as camera.
    // Wired from DeviceAgentService to SubsystemHealthRegistry.updateHealth(MICROPHONE, ACTIVE).
    // Called on every audio classification cycle to prevent RecoveryLadder false-positive
    // "microphone failed" detection.
    var onMicrophoneHealthHeartbeat: (() -> Unit)? = null

    // Fleet-visible health snapshot provider owned by DeviceAgentService.
    var subsystemHealthProvider: (() -> JSONObject?)? = null
    var interactiveUiProvider: (() -> Boolean)? = null

    // Health registry callbacks for edge telemetry.
    var onMlPipelineHealthReport: ((SubsystemHealthRegistry.SubsystemStatus, Map<String, Any>?) -> Unit)? = null
    var onMemoryHealthReport: ((SubsystemHealthRegistry.SubsystemStatus, Map<String, Any>?) -> Unit)? = null

    // Device hotplug detection callbacks
    private val mainHandler = Handler(Looper.getMainLooper())
    private var audioDeviceCallback: AudioDeviceCallback? = null
    private var cameraAvailabilityCallback: CameraManager.AvailabilityCallback? = null
    private var lifecycleOwnerRef: LifecycleOwner? = null
    private var sensingConfig: SensingConfig = SensingConfig()

    // Metrics buffers for aggregation
    private val faceMetricsBuffer = ArrayDeque<AudienceSnapshot>()
    private val audioMetricsBuffer = ArrayDeque<AudioMetrics>()
    private val speechInsightsBuffer = ArrayDeque<SpeechInsights>()


    /**
     * Phase 4 PR 8: per-class binned counts for the YAMNet classifier output.
     * Snapshot lands in audienceSignals.audio_class_histogram (typed map),
     * fanned-out into observation_field_values rows by
     * embeddingWorker.fanOutProfileFields keyed
     * `audio_class_histogram_<bin_offset>_<class_label>`.
     *
     * Reset on every aggregation flush (matching audioMetricsBuffer lifecycle).
     */
    private val yamnetHistogramExtractor = YamnetHistogramExtractor()
    private val yamnetHistogramLock = Any()

    // Class-level foot traffic estimator (reused instead of creating per-aggregation)
    private val footTrafficEstimator = FootTrafficEstimator()

    private var lastAggregationTime = System.currentTimeMillis()

    // PR 10 diagnostic: throttle the AgeGenderProcessor load-gate diag log so
    // we don't flood logcat. aggregateAndEmit runs every 10s; this lets the
    // gate-state breadcrumb print at most once per ~30s. The gate is read in
    // aggregateAndEmit's onDeviceDemographics emit block (~line 2518); the
    // diag captures tierOk / processorPresent / hasFaces independently so the
    // next operator can tell from logcat whether FaceXFormer is bypassed by
    // (a) tier attenuation, (b) model not OTA-downloaded, or (c) no faces
    // accumulated this window (the symptom 0/857,819 audience_metrics rows
    // had on_device_age_distribution populated in the last 7d).
    private var lastAgeGenderDiagLogMs = 0L

    // Screen ID (set when screen is registered)
    @Volatile private var screenId: String? = null

    // Venue type (set when resolved from API alongside screen ID)
    @Volatile private var venueType: String? = null

    // Venue ID (set when venue is assigned to this screen)
    @Volatile private var venueId: String? = null

    // Offline signal buffer
    private var signalBuffer: SignalBufferManager? = null

    // VAS + Federated Learning integration — atomic venue config to prevent race condition
    // where federatedTrainer and modelManager could be read as a partial pair
    data class VenueConfig(val type: String, val trainer: FederatedTrainer, val manager: ModelManager)
    @Volatile private var venueConfig: VenueConfig? = null

    // Content state provider (set by tablet-agent to correlate ads with audience)
    @Volatile private var contentStateProvider: ContentStateProvider? = null

    // Current combined state
    private val _currentState = MutableStateFlow(SensingState())
    val currentState: StateFlow<SensingState> = _currentState

    @Volatile private var isRunning = false
    @Volatile private var startTimeMs = 0L

    /**
     * Detect available hardware and permissions.
     *
     * IMPORTANT: This method checks for ACTUAL connected hardware, not just chipset capabilities.
     * Many signage devices (Rockchip, Amlogic, etc.) declare FEATURE_CAMERA_ANY even without
     * a camera connected. We must enumerate actual cameras and test audio recording availability.
     */
    fun detectCapabilities(): SensingCapabilities {
        val pm = context.packageManager

        // Check runtime permissions first
        hasCameraPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        hasMicrophonePermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        // Check for ACTUAL connected cameras (not just chipset capability)
        // Many devices report FEATURE_CAMERA_ANY but have no camera connected
        hasCameraHardware = detectActualCameraHardware()

        // Check for ACTUAL working microphone (not just chipset capability)
        // Some signage devices report FEATURE_MICROPHONE but have no mic connected
        hasMicrophoneHardware = if (hasMicrophonePermission) {
            detectActualMicrophoneHardware()
        } else {
            // Fall back to feature check if we don't have permission to test
            pm.hasSystemFeature(PackageManager.FEATURE_MICROPHONE)
        }

        // Check if speech intelligence model is available (Moonshine ASR via sherpa-onnx)
        val hasSpeechModel = try {
            context.assets.open("moonshine-tiny/tokens.txt").close()
            true
        } catch (e: Exception) {
            false
        }

        // Check if pose model is available for emotional engagement
        val hasPoseModel = try {
            context.assets.open("pose_landmarker_lite.task").close()
            true
        } catch (e: Exception) {
            false
        }

        // Emotional engagement requires camera (gaze always works with ML Kit)
        val emotionalEngagementAvailable = hasCameraHardware && hasCameraPermission

        val capabilities = SensingCapabilities(
            hasCameraHardware = hasCameraHardware,
            hasMicrophoneHardware = hasMicrophoneHardware,
            hasCameraPermission = hasCameraPermission,
            hasMicrophonePermission = hasMicrophonePermission,
            faceDetectionAvailable = hasCameraHardware && hasCameraPermission,
            audioClassificationAvailable = hasMicrophoneHardware && hasMicrophonePermission,
            speechIntelligenceAvailable = hasMicrophoneHardware && hasMicrophonePermission && hasSpeechModel,
            emotionalEngagementAvailable = emotionalEngagementAvailable
        )

        Log.i(TAG, "Detected capabilities: $capabilities")

        // Emit capabilities to backend
        emitCapabilities(capabilities)

        return capabilities
    }

    /**
     * Post a `stabilityEvent` of type `CameraInitializationFailed` so the
     * operator sees a loud signal in `device_stability_events` when retries
     * are exhausted. Reuses the existing telemetry path — no new table.
     *
     * The metadata block carries everything an operator needs to debug:
     * failure class, retry count, Build.* fields, OS API level, ABI,
     * camera count, time since first attempt. Reads
     * `audienceAnalyzer?.cameraHealth.buildEscalationContextFromBuild(...)`
     * to construct the diagnostic snapshot.
     */
    private fun emitCameraInitializationFailed(decision: CameraHealthMonitor.RetryDecision) {
        val analyzer = audienceAnalyzer ?: return
        val ctx = try {
            analyzer.cameraHealth.buildEscalationContextFromBuild(
                totalCameraCount = cameraStatus.count
            )
        } catch (e: Exception) {
            Log.w(TAG, "[CameraHealth] Failed to build escalation context: ${e.message}")
            return
        }

        val metadata = JSONObject().apply {
            put("failure_class", ctx.failureClass.name.lowercase())
            put("failure_message", ctx.failureMessage ?: "")
            put("retry_count", ctx.retryCount)
            put("max_retries", ctx.maxRetries)
            put("device_manufacturer", ctx.deviceManufacturer)
            put("device_model", ctx.deviceModel)
            put("os_version", ctx.osVersion)
            put("os_api_level", ctx.osApiLevel)
            put("cpu_abi", ctx.cpuAbi)
            put("total_camera_count", ctx.totalCameraCount)
            put("camera_type", cameraStatus.type)
            put("duration_ms_since_first_attempt", ctx.durationMsSinceFirstAttempt)
            put("sensing_mode", _currentState.value.mode.name)
        }

        // device_stability_events INSERT path — wire format mirrors
        // CrashTelemetryReporter.buildStabilityEventJson() so the existing
        // socket.on('stabilityEvent', ...) handler in trillboard-api/socket.js
        // accepts it without adapter changes.
        val event = JSONObject().apply {
            put("event_type", "CameraInitializationFailed")
            put("fingerprint", fingerprint)
            put("screen_id", screenId ?: "")
            put("rendering_mode", "native_camera")
            put("agent_version", "")  // populated server-side from socket auth context
            put("device_model", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
            put("chipset", android.os.Build.HARDWARE ?: "")
            put("attenuation_tier", "NORMAL")
            put("memory_used_mb", 0)
            put("memory_total_mb", 0)
            put("metadata", metadata)
            put("timestamp", System.currentTimeMillis())
        }

        try {
            socketManager.emit("stabilityEvent", event)
            Log.i(
                TAG,
                "[CameraHealth] Emitted CameraInitializationFailed stabilityEvent — class=${ctx.failureClass}, retryCount=${ctx.retryCount}"
            )
        } catch (e: Exception) {
            Log.e(TAG, "[CameraHealth] Failed to emit CameraInitializationFailed stabilityEvent: ${e.message}", e)
        }
    }

    /**
     * Emit device sensing capabilities to backend.
     */
    private fun emitCapabilities(capabilities: SensingCapabilities) {
        val payload = JSONObject().apply {
            put("fingerprint", fingerprint)
            put("screenId", screenId)
            put("sensingCapabilities", JSONArray().apply {
                if (capabilities.faceDetectionAvailable) put("face_detection")
                if (capabilities.audioClassificationAvailable) put("audio_classification")
                if (capabilities.speechIntelligenceAvailable) put("speech_intelligence")
                if (capabilities.emotionalEngagementAvailable) put("emotional_engagement")
            })
            put("hasCameraHardware", capabilities.hasCameraHardware)
            put("hasMicrophoneHardware", capabilities.hasMicrophoneHardware)
            put("hasCameraPermission", capabilities.hasCameraPermission)
            put("hasMicrophonePermission", capabilities.hasMicrophonePermission)
            put("speechIntelligenceAvailable", capabilities.speechIntelligenceAvailable)
            put("emotionalEngagementAvailable", capabilities.emotionalEngagementAvailable)
            // Camera status for fleet dashboard (USB vs internal vs none)
            put("cameraStatus", JSONObject().apply {
                put("available", cameraStatus.available)
                put("type", cameraStatus.type)
                put("count", cameraStatus.count)
            })
        }

        socketManager.emit("deviceCapabilities", payload)
    }

    /**
     * Detect if actual camera hardware is connected.
     *
     * This uses CameraManager to enumerate cameras instead of relying on
     * PackageManager.FEATURE_CAMERA_ANY, which only indicates chipset capability.
     *
     * @return true if at least one camera is actually connected
     */
    private fun detectActualCameraHardware(): Boolean {
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            if (cameraManager == null) {
                Log.w(TAG, "[HardwareDetect] CameraManager not available")
                return false
            }

            val cameraIds = cameraManager.cameraIdList
            val hasCamera = cameraIds.isNotEmpty()

            if (hasCamera) {
                Log.i(TAG, "[HardwareDetect] Found ${cameraIds.size} camera(s): ${cameraIds.joinToString()}")
            } else {
                Log.i(TAG, "[HardwareDetect] No cameras found - device has no connected camera hardware")
            }

            hasCamera
        } catch (e: Exception) {
            Log.w(TAG, "[HardwareDetect] Failed to enumerate cameras: ${e.message}")
            false
        }
    }

    /**
     * Detect if actual microphone hardware is available using Android's device enumeration API.
     *
     * Uses AudioManager.getDevices(GET_DEVICES_INPUTS) to enumerate actual audio input devices,
     * same pattern as CameraManager.getCameraIdList() for cameras.
     *
     * @return true if a real microphone device is connected
     */
    private fun detectActualMicrophoneHardware(): Boolean {
        return try {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
            if (audioManager == null) {
                Log.w(TAG, "[HardwareDetect] AudioManager not available")
                return false
            }

            // Get all audio input devices (API 23+)
            val inputDevices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)

            // Filter for actual microphone types (not virtual/telephony)
            val microphoneTypes = setOf(
                AudioDeviceInfo.TYPE_BUILTIN_MIC,
                AudioDeviceInfo.TYPE_USB_DEVICE,
                AudioDeviceInfo.TYPE_USB_HEADSET,
                AudioDeviceInfo.TYPE_WIRED_HEADSET,
                AudioDeviceInfo.TYPE_BLUETOOTH_SCO,
                AudioDeviceInfo.TYPE_BLE_HEADSET
            )

            val microphones = inputDevices.filter { it.type in microphoneTypes }

            if (microphones.isNotEmpty()) {
                val micList = microphones.joinToString { "'${it.productName}' (type=${it.type})" }
                Log.i(TAG, "[HardwareDetect] Found ${microphones.size} microphone(s): $micList")
                true
            } else {
                val allInputs = inputDevices.joinToString { "'${it.productName}' (type=${it.type})" }
                Log.i(TAG, "[HardwareDetect] No microphones found. Input devices: ${allInputs.ifEmpty { "none" }}")
                false
            }
        } catch (e: Exception) {
            Log.w(TAG, "[HardwareDetect] Failed to enumerate audio devices: ${e.message}")
            false
        }
    }

    /**
     * Get detailed camera status including type (USB, internal, none) and count.
     */
    private fun getCameraStatus(): CameraStatus {
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
                ?: return CameraStatus(available = false, type = "none", count = 0)

            val cameraIds = cameraManager.cameraIdList
            if (cameraIds.isEmpty()) {
                return CameraStatus(available = false, type = "none", count = 0)
            }

            // Check camera types - look for USB (external) cameras
            var hasUsb = false
            var hasInternal = false

            for (cameraId in cameraIds) {
                try {
                    val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                    val lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)

                    // External cameras (USB) don't have a lens facing value or have EXTERNAL
                    when (lensFacing) {
                        CameraCharacteristics.LENS_FACING_EXTERNAL -> hasUsb = true
                        CameraCharacteristics.LENS_FACING_FRONT,
                        CameraCharacteristics.LENS_FACING_BACK -> hasInternal = true
                        null -> hasUsb = true  // No facing info usually means external/USB
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "[CameraStatus] Failed to get characteristics for camera $cameraId: ${e.message}")
                }
            }

            val type = when {
                hasUsb && hasInternal -> "mixed"
                hasUsb -> "usb"
                hasInternal -> "internal"
                else -> "unknown"
            }

            CameraStatus(available = true, type = type, count = cameraIds.size)
        } catch (e: Exception) {
            Log.w(TAG, "[CameraStatus] Failed to get camera status: ${e.message}")
            CameraStatus(available = false, type = "none", count = 0)
        }
    }

    /**
     * True only for EXTERNAL (USB) cameras — the one camera type that can be
     * physically attached or detached at runtime.
     *
     * [CameraManager.AvailabilityCallback] reports a built-in camera as
     * "unavailable" whenever it is merely in use (by our own analyzer, the
     * demographics capture, or any other process) and "available" again when
     * released. Treating those in-use transitions as hotplug events drives a storm
     * of capability re-evaluations, and when one catches a transient camera-query
     * glitch it triggers a destructive analyzer restart that races in-flight camera
     * frames. Restricting the hotplug path to external cameras matches its stated
     * intent (USB hotplug) and removes the spurious-restart root cause.
     *
     * Fails OPEN: if the characteristics query throws, the camera is most likely
     * already gone (a genuine USB removal), so allow re-evaluation to proceed.
     */
    private fun isExternalCamera(cameraId: String): Boolean {
        return try {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
                ?: return true
            val facing = cameraManager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.LENS_FACING)
            facing == CameraCharacteristics.LENS_FACING_EXTERNAL || facing == null
        } catch (e: Exception) {
            Log.w(TAG, "[Hotplug] Could not read facing for camera $cameraId (likely removed): ${e.message}")
            true
        }
    }

    /**
     * Register callbacks for USB device hotplug detection.
     * Automatically re-evaluates capabilities when USB cameras or microphones are connected/disconnected.
     */
    private fun registerDeviceHotplugCallbacks() {
        Log.i(TAG, "[Hotplug] Registering device hotplug callbacks...")

        // Register audio device callback for USB microphone detection
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        if (audioManager != null) {
            audioDeviceCallback = object : AudioDeviceCallback() {
                override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>) {
                    try {
                        val inputDevices = addedDevices.filter { it.isSource }
                        if (inputDevices.isNotEmpty()) {
                            val deviceList = inputDevices.joinToString { "'${it.productName}' (type=${it.type})" }
                            Log.i(TAG, "[Hotplug] >>> AUDIO INPUT DEVICE(S) CONNECTED: $deviceList")
                            onDeviceHotplug(audioAdded = true)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "[Hotplug] Error handling audio device addition: ${e.message}", e)
                    }
                }

                override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>) {
                    try {
                        val inputDevices = removedDevices.filter { it.isSource }
                        if (inputDevices.isNotEmpty()) {
                            val deviceList = inputDevices.joinToString { "'${it.productName}' (type=${it.type})" }
                            Log.i(TAG, "[Hotplug] >>> AUDIO INPUT DEVICE(S) DISCONNECTED: $deviceList")
                            onDeviceHotplug(audioRemoved = true)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "[Hotplug] Error handling audio device removal: ${e.message}", e)
                    }
                }
            }
            audioManager.registerAudioDeviceCallback(audioDeviceCallback, mainHandler)
            Log.d(TAG, "[Hotplug] Audio device callback registered")
        }

        // Register camera availability callback for USB camera detection
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        if (cameraManager != null) {
            cameraAvailabilityCallback = object : CameraManager.AvailabilityCallback() {
                override fun onCameraAvailable(cameraId: String) {
                    // Built-in cameras report "available"/"unavailable" every time they
                    // go idle/in-use — including when OUR OWN analyzer (or the
                    // demographics capture) opens and releases them. Those are not
                    // hotplug events. Only USB/external cameras are physically attached
                    // or detached at runtime, so ignore internal-camera transitions;
                    // otherwise our own camera usage drives a storm of capability
                    // re-evaluations and a transient camera-query glitch tears the
                    // analyzer down mid-frame (the spurious device-swap crash).
                    if (!isExternalCamera(cameraId)) return
                    Log.i(TAG, "[Hotplug] >>> EXTERNAL CAMERA AVAILABLE: $cameraId")
                    onDeviceHotplug(cameraAdded = true, cameraId = cameraId)
                }

                override fun onCameraUnavailable(cameraId: String) {
                    if (!isExternalCamera(cameraId)) return
                    Log.i(TAG, "[Hotplug] >>> EXTERNAL CAMERA UNAVAILABLE: $cameraId")
                    onDeviceHotplug(cameraRemoved = true, cameraId = cameraId)
                }
            }
            cameraManager.registerAvailabilityCallback(cameraAvailabilityCallback!!, mainHandler)
            Log.d(TAG, "[Hotplug] Camera availability callback registered")
        }

        Log.i(TAG, "[Hotplug] Device hotplug callbacks registered successfully")
    }

    /**
     * Unregister device hotplug callbacks.
     */
    private fun unregisterDeviceHotplugCallbacks() {
        Log.d(TAG, "[Hotplug] Unregistering device hotplug callbacks...")

        audioDeviceCallback?.let { callback ->
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
            audioManager?.unregisterAudioDeviceCallback(callback)
            Log.d(TAG, "[Hotplug] Audio device callback unregistered")
        }
        audioDeviceCallback = null

        cameraAvailabilityCallback?.let { callback ->
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            cameraManager?.unregisterAvailabilityCallback(callback)
            Log.d(TAG, "[Hotplug] Camera availability callback unregistered")
        }
        cameraAvailabilityCallback = null
    }

    /**
     * Handle device hotplug events - re-evaluate capabilities and restart sensing if needed.
     */
    private fun onDeviceHotplug(
        cameraAdded: Boolean = false,
        cameraRemoved: Boolean = false,
        audioAdded: Boolean = false,
        audioRemoved: Boolean = false,
        cameraId: String? = null
    ) {
        Log.i(TAG, "[Hotplug] Device change detected: cameraAdded=$cameraAdded, cameraRemoved=$cameraRemoved, " +
                "audioAdded=$audioAdded, audioRemoved=$audioRemoved")

        // Debounce rapid events (USB devices can trigger multiple callbacks)
        mainHandler.removeCallbacksAndMessages("hotplug_reevaluate")
        mainHandler.postDelayed({
            reevaluateCapabilitiesAndRestart()
        }, 1000)  // Wait 1 second for device enumeration to stabilize
    }

    /**
     * Re-evaluate device capabilities and restart sensing with new devices.
     */
    private fun reevaluateCapabilitiesAndRestart() {
        val lifecycleOwner = lifecycleOwnerRef
        if (lifecycleOwner == null) {
            Log.w(TAG, "[Hotplug] No lifecycle owner available, cannot restart sensing")
            return
        }

        Log.i(TAG, "========================================")
        Log.i(TAG, "[Hotplug] >>> RE-EVALUATING CAPABILITIES")
        Log.i(TAG, "========================================")

        val oldMode = _currentState.value.mode
        val oldCameraStatus = cameraStatus
        val oldHasMic = hasMicrophoneHardware

        // Re-detect capabilities
        val newCapabilities = detectCapabilities()

        // Update camera status
        cameraStatus = getCameraStatus()

        Log.i(TAG, "[Hotplug] Old state: mode=$oldMode, camera=${oldCameraStatus.type}/${oldCameraStatus.count}, mic=$oldHasMic")
        Log.i(TAG, "[Hotplug] New state: camera=${cameraStatus.type}/${cameraStatus.count}, mic=$hasMicrophoneHardware")

        // Determine if we need to restart sensing
        val cameraChanged = (oldCameraStatus.available != cameraStatus.available) ||
                (oldCameraStatus.count != cameraStatus.count)
        val micChanged = oldHasMic != hasMicrophoneHardware

        if (!cameraChanged && !micChanged) {
            Log.i(TAG, "[Hotplug] No capability changes detected, skipping restart")
            return
        }

        // Determine new mode
        val newMode = when {
            newCapabilities.faceDetectionAvailable && newCapabilities.audioClassificationAvailable -> SensingMode.FULL
            newCapabilities.faceDetectionAvailable -> SensingMode.FACE_ONLY
            newCapabilities.audioClassificationAvailable -> SensingMode.AUDIO_ONLY
            else -> SensingMode.NONE
        }

        Log.i(TAG, "[Hotplug] Mode transition: $oldMode -> $newMode")

        // Handle mode transitions
        when {
            // Upgrade: NONE -> something, or gained new capability
            newMode.ordinal > oldMode.ordinal -> {
                Log.i(TAG, "[Hotplug] >>> UPGRADING sensing capabilities")
                handleSensingUpgrade(lifecycleOwner, oldMode, newMode, newCapabilities)
            }
            // Downgrade: Lost capability
            newMode.ordinal < oldMode.ordinal -> {
                Log.i(TAG, "[Hotplug] >>> DOWNGRADING sensing capabilities")
                handleSensingDowngrade(oldMode, newMode, newCapabilities)
            }
            // Same mode but device changed (e.g., different USB camera)
            cameraChanged || micChanged -> {
                Log.i(TAG, "[Hotplug] >>> DEVICE CHANGED in same mode, restarting affected components")
                handleDeviceSwap(lifecycleOwner, cameraChanged, micChanged)
            }
        }

        // Update state
        _currentState.value = SensingState(
            isRunning = isRunning,
            mode = newMode,
            error = if (newMode == SensingMode.NONE) "No camera or microphone available" else null
        )

        // Emit updated capabilities to backend
        emitCapabilities(newCapabilities)

        Log.i(TAG, "========================================")
        Log.i(TAG, "[Hotplug] >>> CAPABILITIES UPDATED: $newMode")
        Log.i(TAG, "========================================")
    }

    /**
     * Handle upgrade - start new sensing components.
     */
    private fun handleSensingUpgrade(
        lifecycleOwner: LifecycleOwner,
        oldMode: SensingMode,
        newMode: SensingMode,
        capabilities: SensingCapabilities
    ) {
        // Start camera if gained camera capability
        if (!oldMode.hasCamera() && newMode.hasCamera() && capabilities.faceDetectionAvailable) {
            Log.i(TAG, "[Hotplug] Starting face detection (new camera available)")
            try {
                startFaceDetection(lifecycleOwner, sensingConfig)

                // Also start frame capture for demographics if enabled
                if (sensingConfig.enableDemographicsCapture) {
                    initializeFrameCapture(lifecycleOwner)
                }
            } catch (e: Exception) {
                Log.e(TAG, "[Hotplug] Failed to start face detection: ${e.message}", e)
            }
        }

        // Start audio if gained microphone capability
        if (!oldMode.hasAudio() && newMode.hasAudio() && capabilities.audioClassificationAvailable) {
            Log.i(TAG, "[Hotplug] Starting audio classification (new microphone available)")
            try {
                startAudioClassification(sensingConfig)

                // Also start speech intelligence if enabled and available
                if (capabilities.speechIntelligenceAvailable && sensingConfig.enableSpeechIntelligence) {
                    startSpeechIntelligence(sensingConfig)
                }
            } catch (e: Exception) {
                Log.e(TAG, "[Hotplug] Failed to start audio classification: ${e.message}", e)
            }
        }

        // Ensure aggregation loop is running
        if (!isRunning && newMode != SensingMode.NONE) {
            isRunning = true
            startTimeMs = System.currentTimeMillis()
            startAggregationLoop()
        }
    }

    /**
     * Handle downgrade - stop removed sensing components.
     */
    private fun handleSensingDowngrade(
        oldMode: SensingMode,
        newMode: SensingMode,
        capabilities: SensingCapabilities
    ) {
        // Stop camera if lost camera capability
        if (oldMode.hasCamera() && !newMode.hasCamera()) {
            Log.i(TAG, "[Hotplug] Stopping face detection (camera removed)")
            audienceAnalyzer?.stop()
            audienceAnalyzer = null

            frameCaptureManager?.release()
            frameCaptureManager = null
        }

        // Stop audio if lost microphone capability
        if (oldMode.hasAudio() && !newMode.hasAudio()) {
            Log.i(TAG, "[Hotplug] Stopping audio classification (microphone removed)")
            audioProcessor?.stop()
            audioProcessor = null

            speechProcessor?.stop()
            speechProcessor = null
        }

        // If no sensing left, stop the aggregation loop
        if (newMode == SensingMode.NONE) {
            isRunning = false
        }
    }

    /**
     * Handle device swap - restart affected components with new device.
     */
    private fun handleDeviceSwap(
        lifecycleOwner: LifecycleOwner,
        cameraChanged: Boolean,
        micChanged: Boolean
    ) {
        if (cameraChanged && audienceAnalyzer != null) {
            Log.i(TAG, "[Hotplug] Restarting face detection with new camera")
            audienceAnalyzer?.stop()
            audienceAnalyzer = null
            frameCaptureManager?.release()
            frameCaptureManager = null

            try {
                startFaceDetection(lifecycleOwner, sensingConfig)
                if (sensingConfig.enableDemographicsCapture) {
                    initializeFrameCapture(lifecycleOwner)
                }
            } catch (e: Exception) {
                Log.e(TAG, "[Hotplug] Failed to restart face detection: ${e.message}", e)
            }
        }

        if (micChanged && audioProcessor != null) {
            Log.i(TAG, "[Hotplug] Restarting audio classification with new microphone")
            audioProcessor?.stop()
            audioProcessor = null
            speechProcessor?.stop()
            speechProcessor = null

            try {
                startAudioClassification(sensingConfig)
                val capabilities = detectCapabilities()
                if (capabilities.speechIntelligenceAvailable && sensingConfig.enableSpeechIntelligence) {
                    startSpeechIntelligence(sensingConfig)
                }
            } catch (e: Exception) {
                Log.e(TAG, "[Hotplug] Failed to restart audio classification: ${e.message}", e)
            }
        }
    }

    /**
     * Start audience sensing with available hardware.
     *
     * @param lifecycleOwner Required for CameraX lifecycle binding
     * @param config Optional configuration
     */
    fun start(lifecycleOwner: LifecycleOwner, config: SensingConfig = SensingConfig()) {
        Log.i(TAG, "========================================")
        Log.i(TAG, "[Service] >>> STARTING AUDIENCE SENSING")
        Log.i(TAG, "========================================")

        // Log device profile for diagnostics
        val profile = DeviceProfile.detect(context)
        Log.i(TAG, "[Service] Device profile: ${profile.chipsetVendor} (${profile.chipsetName})")
        Log.i(TAG, "[Service] Adaptive settings: targetFps=${profile.targetFps}, " +
                "resolution=${profile.resolution.width}x${profile.resolution.height}, " +
                "emotionalEngagement=${profile.enableEmotionalEngagement}")

        Log.d(TAG, "[Service] fingerprint=$fingerprint")
        Log.d(TAG, "[Service] apiBaseUrl=$apiBaseUrl")
        Log.d(TAG, "[Service] Config: faceDetection=${config.enableFaceDetection}, " +
                "audioClassification=${config.enableAudioClassification}, " +
                "demographics=${config.enableDemographicsCapture}")

        if (isRunning) {
            Log.w(TAG, "[Service] Already running, ignoring start()")
            return
        }

        // Initialize offline signal buffer
        signalBuffer = SignalBufferManager(context)

        // Store references for hotplug handling
        lifecycleOwnerRef = lifecycleOwner
        sensingConfig = config

        // Initialize memory attenuation manager.
        // zeroAllocPipelineActive intentionally stays FALSE: codex P1 review on
        // PR #6229 correctly observed that setting it true short-circuits
        // MemoryAttenuationManager.getCurrentTier() / transitionTo() to NORMAL
        // unconditionally, which disables shedding of age/gender/pose/gaze
        // even when the loaded models themselves push us into low-RAM/OOM
        // territory. Zero per-frame allocation does NOT eliminate model-weight
        // memory pressure. Proper fix (separating GC-pressure attenuation from
        // model-memory attenuation in the manager itself) is tracked in the
        // follow-up plan; this stays defensive.
        attenuationManager = MemoryAttenuationManager(
            zeroAllocPipelineActive = false,
            onTierChanged = { oldTier, newTier, modelStatusMap ->
                Log.i(TAG, "[Attenuation] Tier changed: $oldTier → $newTier")
                // Emit attenuation event via socket
                val payload = JSONObject().apply {
                    put("fingerprint", fingerprint)
                    put("screenId", screenId)
                    put("oldTier", oldTier.name)
                    put("newTier", newTier.name)
                    put("modelStatus", JSONObject(modelStatusMap as Map<*, *>))
                    put("timestamp", System.currentTimeMillis())
                }
                socketManager.emit("memoryAttenuation", payload)

                // When de-escalating (recovering), restore models to the new tier
                if (newTier.ordinal < oldTier.ordinal) {
                    Log.i(TAG, "[Attenuation] De-escalating from $oldTier to $newTier — restoring models")
                    restoreToTier(newTier)
                }
            }
        )

        // Register device hotplug callbacks for USB camera/mic detection
        registerDeviceHotplugCallbacks()

        val capabilities = detectCapabilities()
        Log.d(TAG, "[Service] Detected capabilities: camera=${capabilities.faceDetectionAvailable}, " +
                "audio=${capabilities.audioClassificationAvailable}")

        // Update camera status
        cameraStatus = getCameraStatus()
        Log.d(TAG, "[Service] Camera status: type=${cameraStatus.type}, count=${cameraStatus.count}")

        if (!capabilities.hasAnySensing) {
            Log.w(TAG, "[Service] No sensing capabilities available - running in no-sensor mode")
            Log.i(TAG, "[Service] Hotplug callbacks are active - sensing will start when USB devices are connected")
            _currentState.value = SensingState(
                isRunning = false,
                mode = SensingMode.NONE,
                error = "No camera or microphone available"
            )
            return
        }

        isRunning = true
        startTimeMs = System.currentTimeMillis()

        // Determine sensing mode
        val mode = when {
            capabilities.faceDetectionAvailable && capabilities.audioClassificationAvailable -> SensingMode.FULL
            capabilities.faceDetectionAvailable -> SensingMode.FACE_ONLY
            capabilities.audioClassificationAvailable -> SensingMode.AUDIO_ONLY
            else -> SensingMode.NONE
        }

        Log.i(TAG, "[Service] Sensing mode determined: $mode")

        _currentState.value = SensingState(
            isRunning = true,
            mode = mode
        )

        // Start face detection if available (wrapped in try-catch for edge cases)
        if (capabilities.faceDetectionAvailable && config.enableFaceDetection) {
            try {
                Log.d(TAG, "[Service] Starting face detection...")
                startFaceDetection(lifecycleOwner, config)
            } catch (e: Exception) {
                Log.e(TAG, "[Service] Failed to start face detection: ${e.message}", e)
                // Downgrade to audio-only mode if face detection fails
                _currentState.value = _currentState.value.copy(
                    mode = if (capabilities.audioClassificationAvailable) SensingMode.AUDIO_ONLY else SensingMode.NONE,
                    error = "Face detection failed: ${e.message}"
                )
            }
        }

        // Start audio classification if available (wrapped in try-catch for edge cases)
        if (capabilities.audioClassificationAvailable && config.enableAudioClassification) {
            try {
                Log.d(TAG, "[Service] Starting audio classification...")
                startAudioClassification(config)
            } catch (e: Exception) {
                Log.e(TAG, "[Service] Failed to start audio classification: ${e.message}", e)
                // Update state to reflect audio failure
                val currentMode = _currentState.value.mode
                val newMode = when (currentMode) {
                    SensingMode.FULL -> SensingMode.FACE_ONLY
                    SensingMode.AUDIO_ONLY -> SensingMode.NONE
                    else -> currentMode
                }
                _currentState.value = _currentState.value.copy(
                    mode = newMode,
                    error = "Audio classification failed: ${e.message}"
                )
            }
        }

        // Start speech intelligence if available and enabled (wrapped in try-catch)
        // RAM guard: skip speech processor on devices with <= 2GB RAM
        // Moonshine Tiny int8 ONNX is ~30MB model + ~50MB runtime = ~80MB total
        // 3GB devices have ~1.5GB available, so 80MB = 5.3% of free RAM — safe
        if (capabilities.speechIntelligenceAvailable && config.enableSpeechIntelligence) {
            val memInfo = ActivityManager.MemoryInfo()
            (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memInfo)
            val totalRamGb = memInfo.totalMem / (1024.0 * 1024 * 1024)
            if (totalRamGb <= 2.0) {
                Log.w(TAG, "Skipping speech processor - device has ${String.format("%.1f", totalRamGb)}GB RAM (minimum 2GB required)")
            } else {
                try {
                    Log.d(TAG, "[Service] Starting speech intelligence (device has ${String.format("%.1f", totalRamGb)}GB RAM)...")
                    startSpeechIntelligence(config)
                } catch (e: Exception) {
                    Log.e(TAG, "[Service] Failed to start speech intelligence: ${e.message}", e)
                    // Speech intelligence is optional, continue without it
                }
            }
        }

        // Initialize frame capture for Gemini demographics (requires camera, wrapped in try-catch)
        if (capabilities.faceDetectionAvailable && config.enableDemographicsCapture) {
            try {
                Log.d(TAG, "[Service] Initializing frame capture for Gemini demographics...")
                initializeFrameCapture(lifecycleOwner)
            } catch (e: Exception) {
                Log.e(TAG, "[Service] Failed to initialize frame capture: ${e.message}", e)
                // Frame capture is optional, continue without it
            }
        }

        // Start aggregation loop
        Log.d(TAG, "[Service] Starting aggregation loop (${AGGREGATION_WINDOW_MS}ms window)")
        startAggregationLoop()

        Log.i(TAG, "========================================")
        Log.i(TAG, "[Service] >>> AUDIENCE SENSING ACTIVE: $mode")
        Log.i(TAG, "========================================")
    }

    private fun startFaceDetection(lifecycleOwner: LifecycleOwner, config: SensingConfig) {
        Log.i(TAG, "[FaceDetection] >>> Initializing face detection...")
        Log.d(TAG, "[FaceDetection] Config: interval=${config.faceDetectionIntervalMs}ms, " +
                "aggregationWindow=${AGGREGATION_WINDOW_MS}ms")

        val analyzerConfig = AudienceConfig(
            enabled = true,
            captureIntervalMs = config.faceDetectionIntervalMs,
            reportIntervalMs = AGGREGATION_WINDOW_MS,
            enableAttentionTracking = true,
            enableEnvironmentSensors = true
        )

        audienceAnalyzer = AudienceAnalyzer(context, analyzerConfig).apply {
            // Wire camera health heartbeat to prevent RecoveryLadder false-positive
            onFrameProcessed = {
                onCameraHealthHeartbeat?.invoke()
            }

            // Route camera frames to VLM processor for on-device vision-language inference.
            // CRITICAL: Copy the bitmap synchronously before launching the coroutine,
            // because imageProxy.close() in addOnCompleteListener will release the
            // backing buffer before the coroutine starts.
            onBitmapAvailable = vlmFrameCallback@{ bitmap ->
                val processor = vlmProcessor
                if (processor != null && processor.isReady()) {
                    val nowMs = SystemClock.elapsedRealtime()
                    val samplingIntervalMs = processor.getSamplingIntervalMs()
                    val maxInFlightMs = processor.getInferenceTimeoutMs() + 5_000L
                    val staleAgeMs = vlmFrameAdmissionGate.inFlightAgeMs(nowMs)
                    if (!vlmFrameAdmissionGate.tryAcquire(nowMs, samplingIntervalMs, maxInFlightMs)) {
                        if (staleAgeMs != null && staleAgeMs > maxInFlightMs) {
                            Log.w(TAG, "[VLM] Frame admission still blocked after stale in-flight age=${staleAgeMs}ms")
                        }
                        return@vlmFrameCallback
                    }
                    if (staleAgeMs != null && staleAgeMs > maxInFlightMs) {
                        Log.w(TAG, "[VLM] Recovered stale frame admission gate age=${staleAgeMs}ms")
                    }

                    // Copy only after the frame is admitted, otherwise the hot path
                    // allocates a large transient bitmap on every camera frame.
                    val bitmapCopy = try {
                        bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                    } catch (e: Exception) {
                        vlmFrameAdmissionGate.release()
                        Log.w(TAG, "[VLM] Bitmap copy failed: ${e.message}")
                        return@vlmFrameCallback
                    }

                    val job = sensingScope.launch(Dispatchers.Default) {
                        try {
                            Log.d(TAG, "[VLM] Frame inference job start")
                            val input = com.trillboards.ctv.core.inference.InferenceInput.CameraFrame(bitmapCopy, System.currentTimeMillis())
                            val output = processor.process(input)
                            if (output != null && output.fields.isNotEmpty()) {
                                lastVlmOutput = output
                                Log.d(TAG, "[VLM] Frame inference output accepted: model=${output.modelId}, fields=${output.fields.size}, latency=${output.latencyMs}ms")
                            } else {
                                Log.w(TAG, "[VLM] Frame inference returned no structured output")
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "[VLM] Frame processing error: ${e.message}")
                        } finally {
                            if (!bitmapCopy.isRecycled) bitmapCopy.recycle()
                            vlmFrameAdmissionGate.release()
                            vlmFrameJob = null
                        }
                    }
                    vlmFrameJob = job
                }
            }

            // Wire up face detection callback for demographics trigger
            onFacesDetected = { faceCount ->
                if (faceCount >= MIN_FACES_FOR_DEMOGRAPHICS) {
                    Log.d(TAG, "[FaceDetection] Faces detected: $faceCount - checking demographics trigger")
                    frameCaptureManager?.onFacesDetected(faceCount)
                }
            }

            // Wire up camera ready callback for FrameCaptureManager
            // This replaces the polling approach - callback is invoked when camera is bound
            onCameraReady = { imageCapture ->
                frameCaptureManager?.setImageCapture(imageCapture)
                Log.i(TAG, "[FaceDetection] >>> ImageCapture ready - wired to FrameCaptureManager")

                // Update camera status now that camera is bound
                cameraStatus = getCameraStatus()
                Log.i(TAG, "[FaceDetection] Camera status: type=${cameraStatus.type}, count=${cameraStatus.count}")

                // Re-emit capabilities with updated camera status
                val capabilities = detectCapabilities()
                emitCapabilities(capabilities)
            }

            // Wire up no-camera callback for graceful degradation (Fire TV without USB camera)
            onNoCameraAvailable = {
                Log.w(TAG, "[FaceDetection] >>> NO CAMERA AVAILABLE - switching to audio-only mode")
                cameraStatus = CameraStatus(available = false, type = "none", count = 0)

                // Update sensing state to audio-only
                val currentMode = _currentState.value.mode
                if (currentMode == SensingMode.FULL || currentMode == SensingMode.FACE_ONLY) {
                    val newMode = if (hasMicrophonePermission && hasMicrophoneHardware) {
                        SensingMode.AUDIO_ONLY
                    } else {
                        SensingMode.NONE
                    }

                    _currentState.value = SensingState(
                        isRunning = isRunning,
                        mode = newMode,
                        error = "No camera available - running in ${newMode.name} mode"
                    )

                    Log.i(TAG, "[FaceDetection] Sensing mode changed: $currentMode -> $newMode")

                    // Re-emit capabilities with updated camera status
                    val capabilities = detectCapabilities()
                    emitCapabilities(capabilities)
                }
            }

            // Wire up camera-start escalation. Fires once per "give up" decision
            // (max-retries hit, or hard failure class like EMULATOR_OR_HEADLESS).
            // Posts a stabilityEvent with full diagnostic context — class,
            // retry count, Build.* fields, total camera count — so the operator
            // sees a loud signal in `device_stability_events` rather than 30
            // days of silent fail-zero like Adam @ Focus Media.
            onCameraStartFailed = { decision ->
                Log.e(
                    TAG,
                    "[CameraHealth] >>> CAMERA INITIALIZATION FAILED (escalating) - class=${decision.failureClass}, msg=${decision.failureMessage}"
                )
                emitCameraInitializationFailed(decision)
            }
        }

        if (activeProfileModels.isNotEmpty()) {
            audienceAnalyzer?.applyVisionProcessorSelection(
                VisionProcessorSelection.fromModels(activeProfileModels)
            )
        }

        // Collect snapshots as they update
        sensingScope.launch {
            audienceAnalyzer?.currentSnapshot?.collect { snapshot ->
                synchronized(faceMetricsBuffer) {
                    // Trim first to prevent momentary overflow
                    while (faceMetricsBuffer.size >= 100) {
                        faceMetricsBuffer.removeFirst()
                    }
                    faceMetricsBuffer.add(snapshot)
                }

                // Log periodic face count updates
                if (snapshot.viewerCount > 0) {
                    Log.d(TAG, "[FaceDetection] Snapshot: viewers=${snapshot.viewerCount}, " +
                            "attention=${String.format("%.2f", snapshot.attentionScore)}, " +
                            "avgDwell=${String.format("%.1f", snapshot.dwellTime.averageSeconds)}s")
                }
            }
        }

        audienceAnalyzer?.start(lifecycleOwner)
        Log.i(TAG, "[FaceDetection] >>> Face detection started - waiting for camera bind...")

        // Age/gender (FaceXFormer) foundation-model OTA at pipeline startup.
        //
        // Root cause this fixes: the only path that previously kicked the
        // age_gender download was applyProfileToInference() (a profile (re)apply
        // event). On a freshly-booted device the persisted-profile apply runs
        // BEFORE this start() creates `audienceAnalyzer`, so
        // triggerAgeGenderOtaDownloadIfNeeded() short-circuited on its
        // `audienceAnalyzer ?: return` guard and the 178MB model was never
        // fetched — AudienceAnalyzer.initializeEmotionalEngagement() then found
        // no model, logged "Age/gender model not available (optional)", and left
        // ageGenderProcessor = null forever. We now (re)attempt the download from
        // the pipeline-startup path too, gated by the same eligibility the
        // profile already enforces (active profile includes `age_gender`).
        startAgeGenderOtaStartupRetryIfEligible()
    }

    /**
     * Drive a BOUNDED, backed-off retry of the age/gender model OTA download at
     * vision-pipeline startup. Eligibility, cadence and the stop condition live
     * in [AgeGenderOtaRetryPolicy] (pure + unit-tested).
     *
     * - Attempt 0 fires immediately (analyzer now exists, so the existing
     *   trigger's `audienceAnalyzer ?: return` guard passes).
     * - Each attempt calls the EXISTING [triggerAgeGenderOtaDownloadIfNeeded] —
     *   we do NOT reinvent the manifest fetch/download. That method already
     *   probes AgeGenderProcessor.hasModel(); when the model is on disk it kicks
     *   AudienceAnalyzer.startAgeGenderProcessor() and the next heartbeat emits
     *   real demographics.
     * - The loop exits early the moment the processor is live (model landed),
     *   and otherwise gives up after [AgeGenderOtaRetryPolicy.MAX_ATTEMPTS] so a
     *   permanently-offline kiosk never polls the manifest endpoint forever. A
     *   later profile (re)apply still re-triggers the existing path.
     *
     * This is the watchdog-style "idempotent recovery re-invoked on a backed-off
     * schedule" pattern (mirrors LocationCollector.start() re-invocation and the
     * CameraHealthMonitor backoff ladder), adapted for a one-shot model fetch.
     */
    /**
     * Is `age_gender` wanted by the CURRENT EFFECTIVE model set? Resolved the
     * SAME way [applyProfileToInference] resolves processor activation —
     * `resolveEffectiveModels(activeProfileModels, activeProgramRuntimeContract)`
     * — so the startup OTA gate never diverges from the processor lifecycle.
     * When a runtime contract supplies explicit worker models that drop
     * `age_gender`, the effective set drops it too and we must NOT download the
     * ~178MB artifact. Reads the @Volatile profile + contract fields live, so
     * callers re-evaluating mid-loop pick up a profile re-apply.
     */
    private fun isAgeGenderEffectivelyWanted(): Boolean {
        val effectiveModels = resolveEffectiveModels(activeProfileModels, activeProgramRuntimeContract)
        return AgeGenderOtaRetryPolicy.isEligible(effectiveModels)
    }

    private fun startAgeGenderOtaStartupRetryIfEligible() {
        if (!isAgeGenderEffectivelyWanted()) {
            Log.d(TAG, "[Profile] Age/gender not in effective model set — skipping startup OTA trigger")
            return
        }

        // Already on disk + processor live (e.g. asset-bundled build, or a prior
        // profile-apply already landed it) — nothing to download.
        if (audienceAnalyzer?.getAgeGenderProcessor() != null) {
            Log.d(TAG, "[Profile] Age/gender processor already active — skipping startup OTA trigger")
            return
        }

        Log.i(TAG, "[Profile] Age/gender eligible but processor not active — starting bounded startup OTA retry")

        sensingScope.launch(Dispatchers.IO) {
            var attempt = 0
            while (isActive && isRunning && !AgeGenderOtaRetryPolicy.hasExhausted(attempt)) {
                // Re-check eligibility against the CURRENT effective model set on
                // every iteration: if the profile is re-applied without
                // age_gender (or a runtime contract drops it) while we're
                // looping, stop — demographics are no longer requested, so we
                // must not keep retrying / downloading.
                if (!isAgeGenderEffectivelyWanted()) {
                    Log.i(TAG, "[Profile] Age/gender no longer in effective model set after $attempt attempt(s) — stopping retry")
                    return@launch
                }

                // Stop the moment a prior attempt (or a concurrent profile-apply)
                // landed the model and brought the processor up.
                if (audienceAnalyzer?.getAgeGenderProcessor() != null) {
                    Log.i(TAG, "[Profile] Age/gender processor active after $attempt startup attempt(s) — stopping retry")
                    return@launch
                }

                Log.i(TAG, "[Profile] Age/gender startup OTA attempt ${attempt + 1}/${AgeGenderOtaRetryPolicy.MAX_ATTEMPTS}")
                triggerAgeGenderOtaDownloadIfNeeded()

                attempt++
                if (AgeGenderOtaRetryPolicy.hasExhausted(attempt)) break
                delay(AgeGenderOtaRetryPolicy.computeNextRetryDelayMs(attempt - 1))
            }

            if (audienceAnalyzer?.getAgeGenderProcessor() == null) {
                Log.w(TAG, "[Profile] Age/gender startup OTA retry exhausted after $attempt attempts — " +
                    "processor still inactive (a later profile re-apply will retry)")
            }
        }
    }

    private fun startAudioClassification(config: SensingConfig) {
        val audioConfig = AudioConfig(
            enabled = true,
            classificationIntervalMs = config.audioClassificationIntervalMs
        )

        audioProcessor = AudioClassificationProcessor(context, audioConfig).apply {
            onMetricsReady = { metrics ->
                synchronized(audioMetricsBuffer) {
                    // Trim first to prevent momentary overflow
                    while (audioMetricsBuffer.size >= 100) {
                        audioMetricsBuffer.removeFirst()
                    }
                    audioMetricsBuffer.add(metrics)
                }
                // Heartbeat on every successful audio classification cycle
                onMicrophoneHealthHeartbeat?.invoke()
            }

            // Phase 4 PR 8: per-class histogram extraction (default ON).
            onCategoriesReady = { categories, timestampMs ->
                synchronized(yamnetHistogramLock) {
                    try {
                        yamnetHistogramExtractor.add(categories, timestampMs)
                    } catch (e: Exception) {
                        Log.w(TAG, "[YamnetHistogram] add() failed (non-fatal): ${e.message}")
                    }
                }
            }

            // Wire up audio state callbacks (matching camera pattern)
            onAudioReady = {
                Log.i(TAG, "[Audio] >>> AUDIO READY - microphone recording started")
                // Send initial microphone health heartbeat
                onMicrophoneHealthHeartbeat?.invoke()
                // Re-emit capabilities with confirmed audio status
                val capabilities = detectCapabilities()
                emitCapabilities(capabilities)
            }

            onAudioFailed = { error ->
                Log.e(TAG, "[Audio] >>> AUDIO FAILED: $error")

                // Update sensing mode if audio fails
                val currentMode = _currentState.value.mode
                if (currentMode == SensingMode.FULL || currentMode == SensingMode.AUDIO_ONLY) {
                    val newMode = if (hasCameraPermission && hasCameraHardware && cameraStatus.available) {
                        SensingMode.FACE_ONLY
                    } else {
                        SensingMode.NONE
                    }

                    _currentState.value = SensingState(
                        isRunning = isRunning,
                        mode = newMode,
                        error = "Audio failed: $error - running in ${newMode.name} mode"
                    )

                    Log.w(TAG, "[Audio] Sensing mode changed: $currentMode -> $newMode")

                    // Re-emit capabilities with updated audio status
                    val capabilities = detectCapabilities()
                    emitCapabilities(capabilities)
                }
            }
        }

        audioProcessor?.start()
        Log.i(TAG, "[Audio] Audio classification initialization requested...")
    }

    /**
     * Start speech intelligence for on-device conversation analysis.
     *
     * PRIVACY: Transcripts are ephemeral - processed in memory and immediately deleted.
     * Only structured SpeechInsights are emitted, never raw text.
     *
     * Eagerly initialises the [OnDeviceLlmInsightExtractor] in a background coroutine
     * so the model is warm before the first speech window arrives. If the model has not
     * been downloaded yet, triggers an OTA download via [ModelDownloadManager].
     * PR L3 will wire the ready extractor into [SpeechIntelligenceProcessor]; for now
     * the eager-init path exists so cold-start download completes before L3 ships.
     */
    private fun startSpeechIntelligence(config: SensingConfig) {
        val speechConfig = SpeechConfig(
            enabled = true,
            transcriptionIntervalMs = config.speechTranscriptionIntervalMs,
            audioBufferLengthMs = 30_000,  // 30 second rolling buffer
            minConfidenceThreshold = 0.3f
        )

        // Eagerly kick off LLM extractor init in the background — NOT lazy on first speech
        // window, which would block the real-time audio pipeline on cold-start download.
        // PR L3: the extractor is stored in [onDeviceLlmExtractor] so SpeechIntelligenceProcessor
        // can call it when SensingConfig.speech.useLlmExtractor becomes true (PR L5 SSM flip).
        val downloadManager = com.trillboards.ctv.core.ml.ModelDownloadManager(context)
        // Reuse an already-created extractor across config re-applies, hotplug
        // restarts and sensing upgrades — startSpeechIntelligence() runs on all of
        // those paths, and recreating the extractor each time would discard the warm
        // 271MB model and force a cold reload + re-warmup. The schema is kept current
        // by setActiveSignalsJson → initialize(), which reloads only when the
        // operator's declared speech fields actually change.
        val llmExtractor = onDeviceLlmExtractor ?: OnDeviceLlmInsightExtractor(
            context = context,
            modelDownloadManager = downloadManager,
            signalsJsonSupplier = { activeSignalsJson },
            attenuationManager = attenuationManager
        ).also { onDeviceLlmExtractor = it }

        sensingScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            val initialized = llmExtractor.initialize()
            if (!initialized) {
                // Model not yet downloaded — fetch the manifest and start OTA download so the
                // model is resident before L5 flips the SSM flag to activate the primary path.
                if (!downloadManager.isModelAvailable("functiongemma_270m")) {
                    Log.i(TAG, "[Speech] FunctionGemma 270M not on device — fetching manifest for OTA download")
                    try {
                        val params = StringBuilder("device_tier=standard")
                        screenId?.let { params.append("&screen_id=$it") }
                        if (fingerprint.isNotEmpty()) params.append("&fingerprint=$fingerprint")
                        val manifestUrl = "${apiBaseUrl}/v2/earner/ml/model-manifest?$params"

                        val request = okhttp3.Request.Builder().url(manifestUrl).build()
                        val client = okhttp3.OkHttpClient.Builder()
                            .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                            .readTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                            .build()
                        val response = client.newCall(request).execute()
                        val body = response.body?.string()
                        response.close()

                        if (body == null || !response.isSuccessful) {
                            Log.w(TAG, "[Speech] FunctionGemma manifest fetch failed: HTTP ${response.code}")
                        } else {
                            val json = org.json.JSONObject(body)
                            val models = json.optJSONObject("models")
                            val entry = models?.optJSONObject("functiongemma_270m")
                            val downloadUrl = entry?.optString("url", "") ?: ""
                            if (downloadUrl.isEmpty()) {
                                Log.i(TAG, "[Speech] FunctionGemma 270M not yet in manifest — " +
                                    "will download on next profile sync when server registers the model")
                            } else {
                                val checksum = if (entry!!.isNull("hash")) "" else entry.optString("hash", "")
                                val sizeBytes = entry.optLong("size_bytes", 0)
                                val format = entry.optString("format", "")
                                Log.i(TAG, "[Speech] FunctionGemma OTA download starting: " +
                                    "size=${sizeBytes / (1024 * 1024)}MB, format=$format")
                                val manifest = com.trillboards.ctv.core.ml.ModelDownloadManager.ModelManifest(
                                    modelId = "functiongemma_270m",
                                    downloadUrl = downloadUrl,
                                    checksumSha256 = checksum,
                                    sizeBytes = sizeBytes,
                                    modelFormat = format
                                )
                                downloadManager.downloadModel(manifest).collect { progress ->
                                    when (progress.state) {
                                        com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.DOWNLOADING -> {
                                            if (progress.percentComplete.toInt() % 25 == 0) {
                                                Log.i(TAG, "[Speech] FunctionGemma download: ${progress.percentComplete.toInt()}%")
                                            }
                                        }
                                        com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.COMPLETE -> {
                                            Log.i(TAG, "[Speech] FunctionGemma 270M download complete — " +
                                                "model resident and wired into SpeechIntelligenceProcessor; " +
                                                "will activate when L5 flips speech.useLlmExtractor=true")
                                        }
                                        com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.FAILED -> {
                                            Log.e(TAG, "[Speech] FunctionGemma 270M download FAILED — " +
                                                "will retry on next profile apply")
                                        }
                                        else -> { /* ignore other states */ }
                                    }
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "[Speech] FunctionGemma OTA download error: ${e.message}", e)
                    }
                } else {
                    Log.d(TAG, "[Speech] LLM extractor initialization blocked (policy/memory) — " +
                        "will retry when SensingConfig.speech.useLlmExtractor=true next heartbeat")
                }
            } else {
                Log.i(TAG, "[Speech] OnDeviceLlmInsightExtractor ready (model=functiongemma_270m) — " +
                    "wired into SpeechIntelligenceProcessor; PRIMARY path activates at L5 SSM flip")
            }
        }

        // Mirror the audioClassificationProcessor wiring pattern (line 1421):
        // construct SpeechIntelligenceProcessor with the already-stored llmExtractor.
        speechProcessor = SpeechIntelligenceProcessor(
            context = context,
            config = speechConfig,
            fingerprint = fingerprint,
            apiBaseUrl = apiBaseUrl,
            onDeviceLlmExtractor = onDeviceLlmExtractor,
            // Device-auth the /analyze-speech edge persist POST so the server
            // resolves THIS screen's declared speech schema (not the retail default).
            deviceTokenProvider = deviceTokenProvider,
            // Send screenId in the POST body (mirrors the frame-analyze POST) so the
            // server's screen-binding lookup resolves the active profile deterministically.
            screenIdProvider = { screenId }
        ).apply {
            // Wire noise profile from audio classifier for noise-robust preprocessing
            audioClassificationProcessor = audioProcessor

            onInsightsReady = { insights ->
                synchronized(speechInsightsBuffer) {
                    // Trim first to prevent momentary overflow
                    while (speechInsightsBuffer.size >= 50) {
                        speechInsightsBuffer.removeFirst()
                    }
                    speechInsightsBuffer.add(insights)
                }

                // Log significant insights immediately
                if (insights.profileFields.isNotEmpty()) {
                    // Sense-Anything on-device LLM path — the operator's declared fields.
                    Log.i(TAG, "[Speech] >>> PROFILE FIELDS (${insights.classificationSource}): " +
                            insights.profileFields)
                } else if (insights.hasBrandMentions() || insights.hasActionablePurchaseSignals()) {
                    Log.i(TAG, "[Speech] >>> ACTIONABLE INSIGHT: " +
                            "brands=${insights.brandMentions}, " +
                            "stage=${insights.purchaseJourney.stage}, " +
                            "objections=${insights.objections}")
                }
            }

        }

        speechProcessor?.start()
        Log.i(TAG, "[Speech] Speech intelligence started (interval: ${config.speechTranscriptionIntervalMs}ms)")
    }

    /**
     * Initialize frame capture for Gemini Vision demographics analysis.
     *
     * Note: FrameCaptureManager no longer manages its own camera binding.
     * Instead, it receives the ImageCapture use case from AudienceAnalyzer,
     * which binds both ImageAnalysis (ML Kit) and ImageCapture (Gemini) together.
     */
    private fun initializeFrameCapture(lifecycleOwner: LifecycleOwner) {
        Log.i(TAG, "[FrameCapture] >>> Initializing frame capture for demographics...")

        frameCaptureManager?.release()
        frameCaptureManager = FrameCaptureManager(context, apiBaseUrl, fingerprint, deviceTokenProvider).apply {
            // Set screen ID if already known
            screenId?.let {
                setScreenId(it)
                Log.d(TAG, "[FrameCapture] Screen ID set: $it")
            }

            // Phase 1c → HTTP-POP bridge. Mirror the WS-emitted per_face
            // snapshot onto every /v2/earner/audience-analyze POST so the
            // cloud persistence path actually receives per-(face × creative)
            // attribution rows. Pre-fix the WS feed emitted them, the HTTP
            // body didn't, so audienceMetricsService never persisted them
            // → fleet-wide 0% per_face_observations fill.
            perFaceSnapshotProvider = { latestPerFaceSnapshot }

            edgeQualityProvider = {
                buildEdgeQualityTelemetry(
                    avgFaceCountWindow = lastEdgeQualityTelemetry?.sensing?.avgFaceCountWindow,
                    avgPersonCountWindow = lastEdgeQualityTelemetry?.sensing?.avgPersonCountWindow,
                    maxFaceCountWindow = lastEdgeQualityTelemetry?.sensing?.maxFaceCountWindow,
                    estimatedOccupancy = lastEdgeQualityTelemetry?.sensing?.estimatedOccupancy,
                    speechSignalsPresent = lastEdgeQualityTelemetry?.sensing?.speechSignalsPresent ?: false,
                    proxySignalsPresent = lastEdgeQualityTelemetry?.proxySignalsPresent,
                    proxySignalMismatch = lastEdgeQualityTelemetry?.proxySignalMismatch,
                    diagnosticReason = lastEdgeQualityTelemetry?.captureDiagnostics?.reason,
                    shouldCaptureDiagnostic = lastEdgeQualityTelemetry?.captureDiagnostics?.shouldCaptureDiagnostic
                        ?: false
                )
            }

            // Edge intelligence context — VLM + TFLite signals for Gemini cloud enrichment
            edgeIntelligenceProvider = provider@{
                val vlmOutput = lastVlmOutput
                val telemetry = lastEdgeQualityTelemetry
                // Nothing to report if both VLM and TFLite are empty
                if (vlmOutput == null && telemetry == null) return@provider null

                JSONObject().apply {
                    // VLM scene understanding (dynamic fields from on-device VLM)
                    if (vlmOutput != null) {
                        put("recent_vlm_scene_description",
                            vlmOutput.fields["scene_description"]?.toString() ?: "")
                        val vlmPersonCount = vlmOutput.fields["person_count"]
                        if (vlmPersonCount is Number) {
                            put("recent_vlm_person_count", vlmPersonCount.toInt())
                        }
                    }

                    // TFLite running averages from last aggregation cycle
                    telemetry?.sensing?.let { sensing ->
                        put("tflite_avg_face_count", sensing.avgFaceCountWindow)
                    }

                    // Live attention score from BlazeFace tracker (non-destructive read)
                    audienceAnalyzer?.getCurrentAttentionScore()?.let { attention ->
                        put("tflite_avg_attention", attention.toDouble())
                    }

                    // Dominant ambience from latest audio classification snapshot
                    val latestAmbience = synchronized(audioMetricsBuffer) {
                        audioMetricsBuffer.lastOrNull()?.ambienceType
                    }
                    if (latestAmbience != null) {
                        put("tflite_dominant_ambience", latestAmbience)
                    }

                    // Dominant emotion from emotional engagement StateFlow (non-destructive)
                    val dominantEmotion = audienceAnalyzer
                        ?.currentEmotionalEngagement?.value
                        ?.emotion?.dominantEmotion?.name
                    if (dominantEmotion != null) {
                        put("tflite_dominant_emotion", dominantEmotion)
                    }
                }
            }

            // Phase 1b: thread content state through to the JPEG REST payload
            // so audience_metrics / observation_stream rows carry currentAdId.
            // Forward the service-level ContentStateProvider (set by tablet-agent
            // via setContentStateProvider) into the capture manager. Android-TV
            // builds leave both nulls, which the manager serializes as JSON null.
            this@AudienceSensingService.contentStateProvider?.let { provider ->
                contentStateProvider = provider
            }

            // Handle demographics results
            onDemographicsReceived = { demographics ->
                latestDemographics = demographics
                Log.i(TAG, "[FrameCapture] >>> DEMOGRAPHICS RECEIVED <<<")
                Log.i(TAG, "[FrameCapture] viewers=${demographics.estimatedViewerCount}, " +
                        "attention=${demographics.attentionLevel}, " +
                        "confidence=${String.format("%.2f", demographics.confidence)}")
                Log.d(TAG, "[FrameCapture] Ages: ${demographics.ageRanges}")
                Log.d(TAG, "[FrameCapture] Genders: ${demographics.genderDistribution}")

                // Emit demographics to backend via Socket.io
                emitDemographics(demographics)
            }

            // Handle errors
            onError = { error ->
                Log.e(TAG, "[FrameCapture] ERROR: $error")
            }
        }

        val analyzer = audienceAnalyzer
        val wiredExistingCapture = wireExistingCaptureIfReady(
            isCameraBound = analyzer?.isCameraBound == true,
            existingCapture = analyzer?.imageCapture
        ) { capture ->
            frameCaptureManager?.setImageCapture(capture)
        }

        if (wiredExistingCapture) {
            Log.i(TAG, "[FrameCapture] Frame capture manager created - re-used already-bound ImageCapture")
        } else {
            Log.i(TAG, "[FrameCapture] Frame capture manager created - ImageCapture will be wired via onCameraReady callback")
        }
    }

    /**
     * Emit demographics to backend via Socket.io.
     */
    private fun emitDemographics(demographics: DemographicsResult) {
        if (
            demographics.observationFamily != ObservationSignalClassifier.SCREEN_AUDIENCE ||
            demographics.decisionability != ObservationSignalClassifier.DECISIONABLE
        ) {
            Log.i(
                TAG,
                "[Socket] Skipping audienceDemographics emission - family=${demographics.observationFamily}, " +
                    "decisionability=${demographics.decisionability}, reasons=${demographics.decisionBlockReasons}"
            )
            return
        }

        Log.d(TAG, "[Socket] Emitting audienceDemographics...")

        val payload = JSONObject().apply {
            put("fingerprint", fingerprint)
            put("screenId", screenId)
            put("demographics", demographics.toJson())
            demographics.captureMode?.let { put("captureMode", it) }
            demographics.measurementQuality?.let { put("measurementQuality", it) }
            demographics.observationFamily?.let { put("observationFamily", it) }
            demographics.evidenceGrade?.let { put("evidenceGrade", it) }
            demographics.decisionability?.let { put("decisionability", it) }
            put("decisionBlockReasons", JSONArray(demographics.decisionBlockReasons))
            demographics.edgeQualityJson?.let { put("edgeQuality", it) }
            put("timestamp", System.currentTimeMillis())
            // Include active sensing profile for pipeline linkage
            activeProfileId?.let { put("profile_id", it) }
            appendObservationProgramMetadata(
                target = this,
                programSpecJson = activeProgramSpecJson,
                programSpecVersion = activeProgramSpecVersion
            )
        }

        socketManager.emit("audienceDemographics", payload)
        Log.i(TAG, "[Socket] >>> audienceDemographics emitted - screenId=$screenId, " +
                "viewers=${demographics.estimatedViewerCount}")
    }

    private fun buildEdgeQualityTelemetry(
        avgFaceCountWindow: Double? = null,
        avgPersonCountWindow: Double? = null,
        maxFaceCountWindow: Int? = null,
        estimatedOccupancy: String? = null,
        speechSignalsPresent: Boolean = false,
        proxySignalsPresent: Boolean? = null,
        proxySignalMismatch: Boolean? = null,
        diagnosticReason: String? = null,
        shouldCaptureDiagnostic: Boolean = false
    ): EdgeQualityTelemetry {
        val runtimeSnapshot = audienceAnalyzer?.getRuntimeSnapshot()
        val memoryInfo = runtimeSnapshot?.memoryInfo ?: readSystemMemoryInfo()
        val currentFaceCount = audienceAnalyzer?.getCurrentViewerCount() ?: 0
        val currentPersonCount = audienceAnalyzer?.getCurrentPersonCount() ?: 0
        val faceWindow = avgFaceCountWindow ?: lastEdgeQualityTelemetry?.sensing?.avgFaceCountWindow ?: currentFaceCount.toDouble()
        val personWindow = avgPersonCountWindow ?: lastEdgeQualityTelemetry?.sensing?.avgPersonCountWindow ?: currentPersonCount.toDouble()
        val maxFaces = maxFaceCountWindow ?: lastEdgeQualityTelemetry?.sensing?.maxFaceCountWindow ?: currentFaceCount
        val occupancy = estimatedOccupancy ?: lastEdgeQualityTelemetry?.sensing?.estimatedOccupancy
        val proxySignals = proxySignalsPresent
            ?: (personWindow >= 0.75 || speechSignalsPresent || occupancy !in setOf(null, "0-5", "unknown"))
        val signalMismatch = proxySignalMismatch
            ?: (faceWindow < 0.5 && proxySignals)
        val signalClassification = ObservationSignalClassifier.classify(
            avgFaceCountWindow = faceWindow,
            currentFaceCount = currentFaceCount,
            avgPersonCountWindow = personWindow,
            currentPersonCount = currentPersonCount,
            speechSignalsPresent = speechSignalsPresent,
            proxySignalsPresent = proxySignals,
            proxySignalMismatch = signalMismatch
        )

        val reliability = when {
            signalClassification.decisionability == ObservationSignalClassifier.DECISIONABLE
                    && (runtimeSnapshot?.actualFps ?: 0f) >= 4f
                    && memoryInfo.pressure == ResourceMonitor.MemoryPressure.LOW -> "high"
            signalClassification.decisionability == ObservationSignalClassifier.DECISIONABLE -> "medium"
            signalMismatch -> "low"
            else -> "low"
        }

        return EdgeQualityTelemetry(
            measurementQuality = signalClassification.measurementQuality,
            observationFamily = signalClassification.observationFamily,
            evidenceGrade = signalClassification.evidenceGrade,
            decisionability = signalClassification.decisionability,
            decisionBlockReasons = signalClassification.decisionBlockReasons,
            reliability = reliability,
            proxySignalsPresent = proxySignals,
            proxySignalMismatch = signalMismatch,
            attenuationTier = (attenuationManager?.getCurrentTier() ?: MemoryAttenuationManager.AttenuationTier.NORMAL).name,
            performance = EdgePerformanceSnapshot(
                actualFps = runtimeSnapshot?.actualFps ?: 0f,
                targetFps = runtimeSnapshot?.targetFps ?: 0,
                skippedFrames = runtimeSnapshot?.skippedFrames ?: 0,
                skipRatio = runtimeSnapshot?.skipRatio ?: 0f,
                avgInferenceMs = runtimeSnapshot?.avgInferenceMs ?: 0.0,
                lastInferenceMs = runtimeSnapshot?.lastInferenceMs ?: 0L,
                webviewFps = com.trillboards.ctv.core.bridge.NativeDeviceBridge.lastWebViewFps
            ),
            memory = EdgeMemorySnapshot(
                availableRamMb = memoryInfo.availableMb,
                totalRamMb = memoryInfo.totalMb,
                thresholdMb = memoryInfo.thresholdMb,
                javaHeapMb = memoryInfo.javaHeapMb,
                javaHeapMaxMb = memoryInfo.javaHeapMaxMb,
                nativeHeapMb = memoryInfo.nativeHeapMb,
                pressure = memoryInfo.pressure.name,
                lowMemory = memoryInfo.lowMemory,
                thermalStatus = getCurrentThermalStatus()
            ),
            sensing = EdgeSensingSnapshot(
                currentFaceCount = currentFaceCount,
                currentPersonCount = currentPersonCount,
                avgFaceCountWindow = faceWindow,
                avgPersonCountWindow = personWindow,
                maxFaceCountWindow = maxFaces,
                estimatedOccupancy = occupancy,
                speechSignalsPresent = speechSignalsPresent
            ),
            captureDiagnostics = CaptureDiagnostics(
                reason = diagnosticReason,
                shouldCaptureDiagnostic = shouldCaptureDiagnostic
            ),
            activeModels = attenuationManager?.getModelStatus() ?: emptyMap(),
            zeroAllocPipelineActive = attenuationManager
                ?.getAttenuationSnapshot()
                ?.get("zeroAllocPipelineActive") as? Boolean
        )
    }

    private fun readSystemMemoryInfo(): ResourceMonitor.MemoryInfo {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val runtime = Runtime.getRuntime()
        val availableRatio = if (memInfo.totalMem > 0) {
            memInfo.availMem.toFloat() / memInfo.totalMem.toFloat()
        } else {
            1f
        }
        val pressure = when {
            memInfo.lowMemory || availableRatio < 0.15f -> ResourceMonitor.MemoryPressure.CRITICAL
            availableRatio < 0.25f -> ResourceMonitor.MemoryPressure.HIGH
            availableRatio < 0.40f -> ResourceMonitor.MemoryPressure.MEDIUM
            else -> ResourceMonitor.MemoryPressure.LOW
        }

        return ResourceMonitor.MemoryInfo(
            totalMb = (memInfo.totalMem / (1024 * 1024)).toInt(),
            availableMb = (memInfo.availMem / (1024 * 1024)).toInt(),
            thresholdMb = (memInfo.threshold / (1024 * 1024)).toInt(),
            lowMemory = memInfo.lowMemory,
            javaHeapMb = ((runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)).toInt(),
            javaHeapMaxMb = (runtime.maxMemory() / (1024 * 1024)).toInt(),
            nativeHeapMb = (android.os.Debug.getNativeHeapAllocatedSize() / (1024 * 1024)).toInt(),
            pressure = pressure
        )
    }

    private fun getCurrentThermalStatus(): Int? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return null
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        return powerManager?.currentThermalStatus
    }

    private fun reportSubsystemHealth(edgeQuality: EdgeQualityTelemetry) {
        val memoryStatus = when (edgeQuality.memory.pressure) {
            ResourceMonitor.MemoryPressure.LOW.name -> SubsystemHealthRegistry.SubsystemStatus.NORMAL
            ResourceMonitor.MemoryPressure.MEDIUM.name -> SubsystemHealthRegistry.SubsystemStatus.MEDIUM
            else -> SubsystemHealthRegistry.SubsystemStatus.HIGH
        }
        onMemoryHealthReport?.invoke(memoryStatus, mapOf(
            "availableRamMb" to edgeQuality.memory.availableRamMb,
            "totalRamMb" to edgeQuality.memory.totalRamMb,
            "nativeHeapMb" to edgeQuality.memory.nativeHeapMb,
            "pressure" to edgeQuality.memory.pressure,
            "measurementQuality" to edgeQuality.measurementQuality
        ))

        val mlStatus = when {
            edgeQuality.attenuationTier == MemoryAttenuationManager.AttenuationTier.CRITICAL.name -> {
                SubsystemHealthRegistry.SubsystemStatus.CRITICAL
            }
            edgeQuality.proxySignalMismatch || edgeQuality.attenuationTier != MemoryAttenuationManager.AttenuationTier.NORMAL.name -> {
                SubsystemHealthRegistry.SubsystemStatus.DEGRADED
            }
            else -> SubsystemHealthRegistry.SubsystemStatus.FULL
        }
        onMlPipelineHealthReport?.invoke(mlStatus, mapOf(
            "measurementQuality" to edgeQuality.measurementQuality,
            "reliability" to edgeQuality.reliability,
            "actualFps" to edgeQuality.performance.actualFps,
            "targetFps" to edgeQuality.performance.targetFps,
            "avgInferenceMs" to edgeQuality.performance.avgInferenceMs,
            "proxySignalMismatch" to edgeQuality.proxySignalMismatch
        ))
    }

    // Note: checkDemographicsCapture removed - now using AudienceAnalyzer.onFacesDetected callback
    // which triggers directly in startFaceDetection()

    private fun startAggregationLoop() {
        sensingScope.launch {
            var cycleCount = 0
            while (isActive && isRunning) {
                delay(AGGREGATION_WINDOW_MS)
                aggregateAndEmit()

                // Prune old buffered signals every ~1 hour (360 cycles × 10s)
                cycleCount++
                if (cycleCount % 360 == 0) {
                    launch(Dispatchers.IO) {
                        signalBuffer?.pruneOldSignals()
                    }
                }
            }
        }
    }

    /**
     * Aggregate buffered metrics and emit to backend.
     */
    private fun aggregateAndEmit() {
        // Memory monitoring (log only — never block data emission)
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        val maxMemory = runtime.maxMemory()
        val javaHeapPercent = (usedMemory * 100 / maxMemory).toInt()

        val nativeHeapAllocated = android.os.Debug.getNativeHeapAllocatedSize()
        val nativeHeapMb = nativeHeapAllocated / (1024 * 1024)

        // System-level memory check
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val systemMemoryPercent = ((1.0 - memInfo.availMem.toDouble() / memInfo.totalMem.toDouble()) * 100).toInt()

        // Check for memory attenuation restoration
        val availableRamPercent = ((memInfo.availMem.toDouble() / memInfo.totalMem.toDouble()) * 100).toFloat()
        attenuationManager?.checkForRestoration(availableRamPercent)

        // CRITICAL-only memory backstop. This 10s loop computes pressure but used to
        // only ever DE-escalate (checkForRestoration above); the sole escalation path
        // was onTrimMemory, which doesn't fire under zram on Android 16, so the device
        // could thrash straight into an lmkd kill with no defense. Close the loop: at
        // the OOM brink only, shed enrichment (age/gender, pose, emotion, gaze, speech,
        // VLM) to survive. Face + YAMNet + VAS stay on at CRITICAL, so attestation is
        // preserved; the full zoo runs untouched at MEDIUM/HIGH. transitionTo() applies
        // hysteresis so this can't thrash.
        if (criticalMemoryBackstopTier(memInfo.lowMemory, availableRamPercent)
                == MemoryAttenuationManager.AttenuationTier.CRITICAL) {
            Log.w(TAG, "[MemoryPressure] CRITICAL backstop: avail=${availableRamPercent.toInt()}%, " +
                    "lowMemory=${memInfo.lowMemory} — shedding enrichment to avoid OOM (face/YAMNet/VAS stay on)")
            attenuateToTier(MemoryAttenuationManager.AttenuationTier.CRITICAL)
        }

        if (javaHeapPercent > 85 || systemMemoryPercent > 90) {
            Log.w(TAG, "[MemoryPressure] java=$javaHeapPercent%, native=${nativeHeapMb}MB, system=$systemMemoryPercent% — emitting anyway (data must flow)")
        }

        // Read venueConfig once to prevent partial reads from concurrent setVenueType()
        val currentVenueConfig = venueConfig

        val now = System.currentTimeMillis()
        val windowStart = lastAggregationTime
        val windowEnd = now
        val windowDurationSec = (windowEnd - windowStart) / 1000.0
        lastAggregationTime = now

        Log.d(TAG, "[Aggregation] Processing ${windowDurationSec.toInt()}s window...")

        // Aggregate face metrics
        val faceSnapshots: List<AudienceSnapshot>
        synchronized(faceMetricsBuffer) {
            faceSnapshots = faceMetricsBuffer.toList()
            faceMetricsBuffer.clear()
        }

        // Aggregate audio metrics
        val audioSnapshots: List<AudioMetrics>
        synchronized(audioMetricsBuffer) {
            audioSnapshots = audioMetricsBuffer.toList()
            audioMetricsBuffer.clear()
        }

        // Aggregate speech insights
        val speechSnapshots: List<SpeechInsights>
        synchronized(speechInsightsBuffer) {
            speechSnapshots = speechInsightsBuffer.toList()
            speechInsightsBuffer.clear()
        }

        Log.d(TAG, "[Aggregation] Samples: ${faceSnapshots.size} face, ${audioSnapshots.size} audio, ${speechSnapshots.size} speech")

        // Skip emission when face samples have not landed yet but detection is active.
        // Requeue audio/speech snapshots so debug injections and live speech aren't
        // dropped during the deferred window.
        val currentViewerCount = audienceAnalyzer?.getCurrentViewerCount() ?: 0
        val deferredAggregationPlan = buildDeferredAggregationPlan(
            faceSnapshotsEmpty = faceSnapshots.isEmpty(),
            currentViewerCount = currentViewerCount,
            audioSnapshots = audioSnapshots,
            speechSnapshots = speechSnapshots
        )
        if (deferredAggregationPlan.shouldDefer) {
            synchronized(audioMetricsBuffer) {
                restoreDeferredSnapshots(audioMetricsBuffer, deferredAggregationPlan.audioSnapshotsToRetain, 100)
            }
            synchronized(speechInsightsBuffer) {
                restoreDeferredSnapshots(speechInsightsBuffer, deferredAggregationPlan.speechSnapshotsToRetain, 50)
            }
            Log.d(TAG, "[Aggregation] Skipping cycle — face buffer empty, detection active. Next cycle will have data.")
            return
        }

        // Calculate aggregated face metrics
        val avgFaceCount = if (faceSnapshots.isNotEmpty()) {
            faceSnapshots.map { it.viewerCount }.average()
        } else 0.0

        val avgPersonCount = if (faceSnapshots.isNotEmpty()) {
            faceSnapshots.map { it.personCount }.average()
        } else 0.0

        val maxFaceCount = if (faceSnapshots.isNotEmpty()) {
            faceSnapshots.maxOfOrNull { it.viewerCount } ?: 0
        } else 0

        val avgAttention = if (faceSnapshots.isNotEmpty()) {
            faceSnapshots.map { it.attentionScore.toDouble() }.average()
        } else 0.0

        val avgDwellMs = if (faceSnapshots.isNotEmpty()) {
            faceSnapshots.map { it.dwellTime.averageSeconds * 1000.0 }.average()
        } else 0.0

        // Calculate aggregated audio metrics
        val avgNoiseTier = if (audioSnapshots.isNotEmpty()) {
            audioSnapshots.map { it.ambientNoiseLevel.toDouble() }.average()
        } else -1.0

        // Compute ambient light and lux variance from face snapshot environment data
        val luxValues = faceSnapshots.mapNotNull {
            it.environment.ambientLightLux.takeIf { l -> l >= 0f }
        }
        val avgAmbientLux = if (luxValues.isNotEmpty()) luxValues.average().toFloat() else -1f
        val computedLuxVariance = if (luxValues.size >= 2) {
            val mean = luxValues.average().toFloat()
            luxValues.map { (it - mean) * (it - mean) }.average().toFloat()
        } else 0f
        // Real IAB MRC-shaped viewability: face_count > 0 AND dwell ≥ 1s
        // AND attention sustained, multiplied by environmental lighting
        // factor. Returns null when no face was observed in the window
        // (no face = nothing was viewable). The prior implementation
        // returned a lux-only bucket that was 1.0 for any indoor light
        // (50–500 lux), masquerading as a buyer-grade signal.
        val environmentalFactor = audienceAnalyzer?.getSensorCollector()
            ?.getEnvironmentalViewabilityFactor()
        val viewabilityScore: Float? = if (avgFaceCount < 0.5) {
            // No face seen → nothing was viewable; emit null
            null
        } else {
            // avgDwellMs and avgAttention are Double (from .average()); compute
            // in Double then narrow to Float at emission. Keeps math precise
            // and lets the wire field stay Float? for backward-compat.
            val dwellSec = (avgDwellMs / 1000.0).coerceAtLeast(0.0)
            val durationFactor = (dwellSec / 1.0).coerceIn(0.0, 1.0)  // IAB MRC ≥1s = full
            val attentionFactor = avgAttention.coerceIn(0.0, 1.0)
            // environmentalFactor null → no light reading; treat as 1.0
            // (don't penalize devices with absent light sensor)
            val envFactor = (environmentalFactor ?: 1f).toDouble()
            (durationFactor * attentionFactor * envFactor).coerceIn(0.0, 1.0).toFloat()
        }

        val crowdedPct = if (audioSnapshots.isNotEmpty()) {
            audioSnapshots.count { it.isCrowded } / audioSnapshots.size.toFloat()
        } else 0f

        val dominantAmbience = audioSnapshots
            .groupBy { it.ambienceType }
            .maxByOrNull { it.value.size }
            ?.key ?: "unknown"

        // Aggregate enhanced audio metrics. cosmic-brewing-bear C2: drop the
        // `?: 0.5` and `?: "0-5"` magic fallbacks. When no audio data exists
        // in the window, emit null upward — the server's
        // networkAggregationService already tolerates null per PR #5003.
        val avgAdReceptivity: Double? = if (audioSnapshots.isNotEmpty()) {
            audioSnapshots.map { it.adReceptivityScore.toDouble() }.average()
        } else null

        // Mic-noise-tier "0-5"/"5-20"/"20-50"/"50+" buckets were deleted from
        // AudioMetrics in cosmic-brewing-bear C1. The aggregated occupancy
        // string is gone with them; consumers should read raw scalars
        // (audio_db_max, crowd_voice_count_estimate, audio_class_event_counts)
        // and derive their own thresholds, or read face-backed occupancy from
        // the server resolution path.
        val estimatedOccupancy: String? = null

        val inferredVenueType = audioSnapshots
            .groupBy { it.inferredVenueType }
            .maxByOrNull { it.value.size }
            ?.key ?: "unknown"

        // Raw audio scalars from the AudioMetrics windows. dBFS values use
        // -120 as the silence floor (NOT a magic data value — it's the
        // documented dynamic-range bottom for PCM_FLOAT). When the buffer is
        // empty we emit null instead of a fabricated zero/floor.
        val audioDbMax: Float? = if (audioSnapshots.isNotEmpty()) {
            audioSnapshots.map { it.audioDbMax }.max()
        } else null
        val audioDbMean: Float? = if (audioSnapshots.isNotEmpty()) {
            audioSnapshots.map { it.audioDbMean.toDouble() }.average().toFloat()
        } else null
        // crowd_voice_count: peak across the window when at least one window
        // had a confident reading. Returns null when every window was null
        // (no confident speech anywhere in the window). NOT zero — null is
        // the honest "we don't know".
        val crowdVoiceCountEstimate: Int? = audioSnapshots
            .mapNotNull { it.crowdVoiceCountEstimate }
            .maxOrNull()
        // event_counts: sum across windows. Empty map (no events ever fired)
        // is intentional — server-side fanout treats absent as no events,
        // identical to a zero-count map.
        val audioClassEventCounts: Map<String, Int> = audioSnapshots
            .flatMap { it.audioClassEventCounts.entries }
            .groupBy({ it.key }, { it.value })
            .mapValues { (_, counts) -> counts.sum() }

        // Aggregate speech insights
        // Skip if tier >= CRITICAL (speech processor unloaded)
        val currentTier = attenuationManager?.getCurrentTier()
        val aggregatedSpeech = if (currentTier == null || currentTier.ordinal < MemoryAttenuationManager.AttenuationTier.CRITICAL.ordinal) {
            aggregateSpeechInsights(speechSnapshots)
        } else {
            null
        }

        // Aggregate emotional engagement
        // Skip if tier >= HIGH (pose/emotion/gaze processors unloaded)
        val emotionalEngagement = if (currentTier == null || currentTier.ordinal < MemoryAttenuationManager.AttenuationTier.HIGH.ordinal) {
            audienceAnalyzer?.aggregateEmotionalEngagement()
        } else {
            null
        }

        // Push TFLite snapshot into the ring buffer for temporal VLM context.
        // The VLM's contextProvider reads these snapshots to inject sensor history
        // into the prompt so the model can reason about crowd trends.
        val snapshot = VLMInferenceProcessor.TFLiteSnapshot(
            timestampMs = System.currentTimeMillis(),
            faceCount = avgFaceCount.toFloat(),
            avgAttention = avgAttention.toFloat(),
            dominantEmotion = emotionalEngagement?.emotion?.dominantEmotion?.name ?: "unknown",
            dominantAmbience = dominantAmbience
        )
        synchronized(tfliteSnapshots) {
            if (tfliteSnapshots.size >= 3) tfliteSnapshots.removeFirst()
            tfliteSnapshots.addLast(snapshot)
        }

        // Perception-triggered VLM inference (Phase 5B)
        // Build SceneState from current perception outputs and trigger VLM only
        // when the scene has meaningfully changed. Instead of directly calling
        // process() here (we don't have a camera frame in the aggregation loop),
        // we call requestImmediateInference() which resets the VLM sampling timer.
        // The next CameraX frame will then run VLM inference immediately.
        val activeVlmProcessor = vlmProcessor
        if (activeVlmProcessor != null && activeVlmProcessor.isReady()) {
            val sceneState = PerceptionTrigger.SceneState(
                personCount = avgFaceCount.toInt(),
                objectCounts = emptyMap(),  // Populated when object detection processor is wired
                dominantEmotion = emotionalEngagement?.emotion?.dominantEmotion?.name,
                noiseLevel = if (avgNoiseTier >= 0) avgNoiseTier.toFloat() else 0f,
                timestamp = System.currentTimeMillis()
            )

            if (perceptionTrigger.shouldTriggerVLM(sceneState)) {
                activeVlmProcessor.requestImmediateInference()
                perceptionTrigger.markTriggered(sceneState)
                Log.d(TAG, "[VLM] Perception trigger fired: persons=${sceneState.personCount}, " +
                    "emotion=${sceneState.dominantEmotion}, noise=${sceneState.noiseLevel}")
            }

            // Cache latest VLM output for emission in model_outputs.
            // Compute temporal agreement score from TFLite snapshots and attach
            // it to the output fields so the server can weigh VLM observations.
            activeVlmProcessor.getLastKnownGoodOutput()?.let { output ->
                val snapshotsCopy = synchronized(tfliteSnapshots) { tfliteSnapshots.toList() }
                val agreement = activeVlmProcessor.computeTemporalAgreement(output.fields, snapshotsCopy)
                val enrichedOutput = output.copy(
                    fields = output.fields + ("temporal_agreement" to agreement)
                )
                lastVlmOutput = enrichedOutput
            }
        }

        // Get content state once — reused for both payload and VAS computation
        val contentState = try {
            contentStateProvider?.getCurrentContentState()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get content state: ${e.message}")
            null
        }

        // Phase 1c — propagate the current creative ID into the
        // AudienceAnalyzer's per-(face × creative) accumulator. Subsequent
        // per-frame samples will bucket under this pair until the next
        // window flush (when drainPerAdAttention runs below).
        //
        // 2026-05-11 — creativeId broadens this beyond IMA programmatic:
        // self_promo, sponsored, default_stream items now attribute attention
        // too. Pre-fix, contentState.adId was only non-null for IMA ads, so
        // perAdAttention stayed empty for the 95%+ of fleet playback that's
        // non-IMA, and Phase 1c emitted nothing fleet-wide. Falling back to
        // adId preserves the IMA-only behavior when the JS bridge somehow
        // didn't populate creativeId.
        audienceAnalyzer?.setCurrentAd(
            contentState?.creativeId ?: contentState?.adId,
            contentState?.impressionId,
            contentState?.creativeSource
        )

        val shadowEvents = audienceAnalyzer?.getSensorCollector()?.getAndResetShadowEvents() ?: 0

        // --- Phase 3: Multi-signal confidence calibration ---
        // Build a SignalBundle from all processor outputs in this window and compute
        // calibrated confidence. This replaces the flat 0.95 parse confidence with
        // real P(observation correct | sensor data) measured by inter-signal agreement.
        val currentVlmOutputForCalibration = lastVlmOutput
        // Edge-side occupancy bucketing was deleted with cosmic-brewing-bear C1;
        // the SignalBundle audioOccupancy slot is null until a replacement
        // signal lands (e.g. crowd_voice_count_estimate when buyer-side
        // calibration uses it).
        val audioOccupancyNumeric: Int? = null
        val signalBundle = SignalBundle(
            faceCount = avgFaceCount.toInt(),
            personCount = avgPersonCount.toInt(),
            vlmFaceCount = currentVlmOutputForCalibration?.fields?.get("face_count")?.let {
                (it as? Number)?.toInt()
            },
            vlmPersonCount = currentVlmOutputForCalibration?.fields?.get("person_count")?.let {
                (it as? Number)?.toInt()
            },
            shadowEvents = shadowEvents,
            audioOccupancy = audioOccupancyNumeric,
            ferEmotion = emotionalEngagement?.emotion?.dominantEmotion?.name,
            vlmActivity = currentVlmOutputForCalibration?.fields?.get("activity")?.toString()
                ?: currentVlmOutputForCalibration?.fields?.get("mood")?.toString(),
            gazeAttention = emotionalEngagement?.gaze?.gazeAttentionScore ?: 0f,
            bodyEngagement = emotionalEngagement?.pose?.bodyEngagementScore ?: 0f
        )
        val calibratedConfidence = confidenceCalibrator.calibrate(signalBundle)

        // Record calibration result for rolling telemetry averages
        CalibrationTelemetry.recordCalibration(calibratedConfidence)
        // Record VLM vs TFLite face count agreement when both are available
        val vlmFaceForAgreement = signalBundle.vlmFaceCount
        if (vlmFaceForAgreement != null) {
            CalibrationTelemetry.recordFaceAgreement(signalBundle.faceCount, vlmFaceForAgreement)
        }

        // Enrich VLM output with semantic confidence if available
        if (currentVlmOutputForCalibration != null) {
            lastVlmOutput = currentVlmOutputForCalibration.copy(
                semanticConfidence = calibratedConfidence.overall,
                fields = currentVlmOutputForCalibration.fields + mapOf(
                    "calibrated_confidence" to calibratedConfidence.overall,
                    "face_count_confidence" to calibratedConfidence.faceCountConfidence,
                    "emotion_confidence" to calibratedConfidence.emotionConfidence,
                    "engagement_confidence" to calibratedConfidence.engagementConfidence,
                    "presence_confidence" to calibratedConfidence.presenceConfidence,
                    "signal_count" to calibratedConfidence.signalCount
                )
            )
        }

        val footTrafficMetrics = if (currentTier == null || currentTier.ordinal < MemoryAttenuationManager.AttenuationTier.MEDIUM.ordinal) {
            footTrafficEstimator.estimate(
                shadowEvents = shadowEvents,
                maxFaceCount = maxFaceCount,
                estimatedOccupancy = estimatedOccupancy,
                luxVariance = computedLuxVariance
            )
        } else {
            null
        }

        val speechSignalsPresent = aggregatedSpeech?.hasBrandMentions() == true
            || aggregatedSpeech?.hasActionablePurchaseSignals() == true
        // estimatedOccupancy is permanently null after cosmic-brewing-bear C1
        // (the bucket strings are gone). Drop that branch from the proxy
        // OR-chain — the remaining signals (person count, foot traffic, shadow
        // events, speech) carry the load.
        val strongProxySignalsPresent = avgPersonCount >= 0.75
            || (footTrafficMetrics?.estimatedCount ?: 0) >= 2
            || shadowEvents > 0
            || speechSignalsPresent
        // avgAdReceptivity is now nullable; treat null as "below threshold".
        val proxySignalsPresent = strongProxySignalsPresent || (avgAdReceptivity ?: 0.0) >= 0.7
        val diagnosticReason = if (avgFaceCount < 0.5 && strongProxySignalsPresent) {
            "proxy_signal_mismatch"
        } else {
            null
        }

        // estimatedOccupancy=null on the wire — the magic-bucket strings
        // ("0-5", "5-20", "20-50", "50+") are an audio-noise-tier proxy that
        // misrepresents face-backed occupancy. Server's
        // networkAggregationService overrides occupancy with String(totalViewers)
        // when face count > 0; dropping the edge string lets the server's
        // null-fallback path show "—" instead of a fake bucket. The internal
        // FootTrafficEstimator heuristic above still consumes the
        // edge-bucketed audio signal — that's edge-internal and doesn't reach
        // the wire. Raw audio scalars (audio_db_max, crowd_voice_count_estimate,
        // event_counts) are tracked under Task #7 in cosmic-brewing-bear.
        var edgeQualityTelemetry = buildEdgeQualityTelemetry(
            avgFaceCountWindow = avgFaceCount,
            avgPersonCountWindow = avgPersonCount,
            maxFaceCountWindow = maxFaceCount,
            estimatedOccupancy = null,
            speechSignalsPresent = speechSignalsPresent,
            proxySignalsPresent = proxySignalsPresent,
            proxySignalMismatch = diagnosticReason != null,
            diagnosticReason = diagnosticReason,
            shouldCaptureDiagnostic = diagnosticReason != null
        )
        lastEdgeQualityTelemetry = edgeQualityTelemetry

        if (diagnosticReason != null) {
            val requested = frameCaptureManager?.requestDiagnosticCapture(diagnosticReason) == true
            if (requested != edgeQualityTelemetry.captureDiagnostics.shouldCaptureDiagnostic) {
                edgeQualityTelemetry = edgeQualityTelemetry.copy(
                    captureDiagnostics = edgeQualityTelemetry.captureDiagnostics.copy(
                        shouldCaptureDiagnostic = requested
                    )
                )
                lastEdgeQualityTelemetry = edgeQualityTelemetry
            }
        }

        reportSubsystemHealth(edgeQualityTelemetry)

        // Generate inference snapshot ID for outcome correlation
        val inferenceSnapshotId = UUID.randomUUID().toString()
        val currentVlmOutput = lastVlmOutput
        val currentVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.withOutput(
            hasLastOutput = currentVlmOutput != null,
            lastOutputModelId = currentVlmOutput?.modelId
        )
        val currentCsiSnapshot = try {
            CsiAggregator.aggregate()
        } catch (csiErr: Exception) {
            Log.w(TAG, "[CSI] Failed to aggregate snapshot (non-fatal): ${csiErr.message}")
            null
        }
        val currentCsiNodeCount = try {
            CsiNodeDiscovery.getNodeCount()
        } catch (csiErr: Exception) {
            Log.w(TAG, "[CSI] Failed to resolve node count (non-fatal): ${csiErr.message}")
            0
        }
        val effectiveCsiNodeCount = if (currentCsiSnapshot != null) {
            maxOf(currentCsiNodeCount, 1)
        } else {
            currentCsiNodeCount
        }

        // Build payload (unified audience signals)
        val payload = JSONObject().apply {
            put("fingerprint", fingerprint)
            put("screenId", screenId)
            put("venue_id", venueId ?: JSONObject.NULL)
            put("inference_snapshot_id", inferenceSnapshotId)
            put("profile_id", activeProfileId ?: JSONObject.NULL)
            // Only emit vlm_runtime when the VLM is actively producing
            // inference. Disabled/blocked/initializing/failed states are dead
            // weight on the wire. Server consumer (normalizeVlmRuntimeTelemetry,
            // audienceMetricsService.js#393) tolerates the field being
            // absent — it falls back to `null`.
            if (currentVlmRuntimeTelemetry.state == VlmRuntimeTelemetry.STATE_ACTIVE) {
                put("vlm_runtime", currentVlmRuntimeTelemetry.toJson())
            }
            appendObservationProgramMetadata(
                target = this,
                programSpecJson = activeProgramSpecJson,
                programSpecVersion = activeProgramSpecVersion
            )
            put("windowStart", windowStart)
            put("windowEnd", windowEnd)

            // Per-model outputs keyed by model ID (for profile-aware telemetry)
            val modelOutputs = JSONObject()
            if (audienceAnalyzer != null) {
                modelOutputs.put("blazeface", JSONObject().apply {
                    put("face_count", avgFaceCount)
                    put("attention", avgAttention)
                })
            }
            if (audioProcessor != null) {
                modelOutputs.put("yamnet", JSONObject().apply {
                    put("noise_tier", avgNoiseTier)
                    put("crowded_pct", crowdedPct)
                    put("ambience", dominantAmbience)
                })
            }
            if (aggregatedSpeech != null) {
                // purchase_intent emission removed in cosmic-brewing-bear
                // task #26 — cloud Gemini's speech_purchase_intent enum is
                // canonical (audienceVisionService.js#_assembleProfilePrompt).
                modelOutputs.put("whisper_tiny", JSONObject().apply {
                    put("speaker_count", aggregatedSpeech.estimatedSpeakerCount)
                    put("brand_mentions", aggregatedSpeech.brandMentions.size)
                })
            }
            if (emotionalEngagement != null) {
                emotionalEngagement.emotion?.let { emotion ->
                    modelOutputs.put("fer_plus", JSONObject().apply {
                        put("dominant_emotion", emotion.dominantEmotion.name)
                        put("engagement_score", emotion.emotionalEngagementScore)
                    })
                }
                emotionalEngagement.pose?.let { pose ->
                    modelOutputs.put("movenet", JSONObject().apply {
                        put("body_engagement", pose.bodyEngagementScore)
                        // Self-describing distribution — adding a new MovementState surfaces
                        // through here automatically. Server reads keys directly off the map.
                        put("movement_distribution", JSONObject().apply {
                            pose.movementDistribution.forEach { (state, count) -> put(state.name, count) }
                        })
                        // Legacy convenience keys kept for one APK release while the server-side
                        // reader migrates to the distribution map. Drop both in the next PR.
                        put("stopped_count", pose.movementDistribution[MovementState.STOPPED] ?: 0)
                        put("walking_past_count", pose.movementDistribution[MovementState.WALKING_FAST] ?: 0)
                    })
                }
            }
            if (currentCsiSnapshot != null) {
                modelOutputs.put("csi_sensing", JSONObject().apply {
                    put("node_id", currentCsiSnapshot.nodeId)
                    put("occupant_count", currentCsiSnapshot.occupantCount)
                    put("motion_score", currentCsiSnapshot.motionScore.toDouble())
                    put("signal_quality", currentCsiSnapshot.signalQuality.toDouble())
                    put("subcarrier_count", currentCsiSnapshot.subcarrierCount)
                    put("capture_rate_hz", currentCsiSnapshot.captureRateHz.toDouble())
                    put("avg_rssi_dbm", currentCsiSnapshot.avgRssiDbm)
                    put("frames_processed", currentCsiSnapshot.framesProcessed)
                    put("frames_dropped", currentCsiSnapshot.framesDropped)
                    put("hardware_type", currentCsiSnapshot.hardwareType)
                    put("window_start_ms", currentCsiSnapshot.windowStartMs)
                    put("window_end_ms", currentCsiSnapshot.windowEndMs)
                    put("node_count", effectiveCsiNodeCount)
                })
            }
            // efficientdet: vehicle + pedestrian counts from ObjectDetectionProcessor
            // (Phase 4 wire-up). VehicleFlowProcessor and QueueEstimationProcessor
            // accumulate per-frame snapshots and expose per-window aggregates here.
            // These ride the existing 10s → audienceSignals → metricsWorker →
            // audience_metrics path as model_outputs.efficientdet fields.
            val vehicleMetrics = audienceAnalyzer?.getVehicleFlowMetrics()
            val queueMetrics = audienceAnalyzer?.getQueueMetrics()
            if (vehicleMetrics != null || queueMetrics != null) {
                modelOutputs.put("efficientdet", JSONObject().apply {
                    vehicleMetrics?.let { vm ->
                        put("vehicle_count", vm.currentVehicleCount)
                        put("avg_vehicle_count", vm.avgVehicleCount)
                        put("peak_vehicle_count", vm.peakVehicleCount)
                        put("fill_rate_estimate", vm.fillRateEstimate)
                        put("vehicle_trend", vm.trend)
                    }
                    queueMetrics?.let { qm ->
                        put("pedestrian_count", qm.estimatedQueueLength)
                        put("queue_length", qm.estimatedQueueLength)
                        put("queue_detected", qm.isQueueDetected)
                        put("queue_trend", qm.trend)
                    }
                })
            }
            val ageGenderProc = audienceAnalyzer?.getAgeGenderProcessor()
            if (ageGenderProc != null && ageGenderProc.hasFaces()) {
                modelOutputs.put("age_gender", JSONObject().apply {
                    put("has_data", true)
                    put("confidence", ageGenderProc.getAvgConfidence())
                })
            }
            // VLM perception-triggered output (Phase 5B) + calibrated confidence (Phase 3)
            if (currentVlmOutput != null) {
                modelOutputs.put(currentVlmOutput.modelId, JSONObject().apply {
                    for ((key, value) in currentVlmOutput.fields) {
                        put(key, value)
                    }
                    put("latency_ms", currentVlmOutput.latencyMs)
                    // Phase 3: emit both parse and effective confidence
                    put("parse_confidence", currentVlmOutput.confidence.toDouble())
                    put("confidence", currentVlmOutput.effectiveConfidence.toDouble())
                    if (currentVlmOutput.semanticConfidence != null) {
                        put("semantic_confidence", currentVlmOutput.semanticConfidence.toDouble())
                    }
                    put("perception_triggered", true)
                    val (triggerCount, skipCount) = perceptionTrigger.getTelemetry()
                    put("trigger_count", triggerCount)
                    put("skip_count", skipCount)
                })
            }
            // Phase 3: attach calibrated confidence to top-level payload
            put("calibratedConfidence", JSONObject().apply {
                put("overall", calibratedConfidence.overall.toDouble())
                put("faceCount", calibratedConfidence.faceCountConfidence.toDouble())
                put("emotion", calibratedConfidence.emotionConfidence.toDouble())
                put("engagement", calibratedConfidence.engagementConfidence.toDouble())
                put("presence", calibratedConfidence.presenceConfidence.toDouble())
                put("signalCount", calibratedConfidence.signalCount)
            })
            if (modelOutputs.length() > 0) {
                put("model_outputs", modelOutputs)
            }

            // Phase 5d: forward cloud-computed dynamic fields (politeness_score etc.)
            // from the most recent /v2/earner/audience-analyze response onto the
            // 10s audienceSignals wire so analytics + WebSocket consumers see them.
            // latestDemographics is set by onDemographicsReceived when the HTTP call
            // returns (~every 30s); null between device boots or before first cloud call.
            // profile_fields_timestamp lets downstream (metricsWorker, analytics) deduplicate
            // repeated 10s emits that carry the same cloud analysis result — same timestamp
            // = same cloud call, different timestamp = fresh analysis.
            latestDemographics?.let { demographics ->
                demographics.profileFields?.let { fields ->
                    put("profile_fields", fields)
                    put("profile_fields_timestamp", demographics.analysisTimestamp)
                }
            }

            // Face metrics (vision signals)
            put("avgFaceCount", avgFaceCount)
            put("avgPersonCount", avgPersonCount)
            put("maxFaceCount", maxFaceCount)
            put("avgAttention", avgAttention)
            put("avgDwellTimeMs", avgDwellMs)

            // Audio metrics (enhanced with derived signals)
            put("avgNoiseTier", avgNoiseTier)
            put("crowdedPct", crowdedPct)
            put("dominantAmbience", dominantAmbience)
            // estimatedOccupancy: null on the wire — the magic-bucket strings
            // ("0-5"…"50+") were deleted in cosmic-brewing-bear C1; server
            // resolves occupancy from face count via networkAggregationService.
            // adReceptivityScore: now Double? — null when no audio data this
            // window (cosmic-brewing-bear C2 deletes the `?: 0.5` fallback).
            put("adReceptivityScore", avgAdReceptivity ?: JSONObject.NULL)
            put("inferredVenueType", inferredVenueType)

            // Raw audio scalars (cosmic-brewing-bear C1) — buyer-grade
            // honest signals replacing the deleted occupancy buckets.
            // dBFS is digital full-scale (NOT calibrated SPL). Server reads
            // these as deltas/relatives. Null when no audio window observed.
            put("audio_db_max", audioDbMax?.toDouble() ?: JSONObject.NULL)
            put("audio_db_mean", audioDbMean?.toDouble() ?: JSONObject.NULL)
            put("crowd_voice_count_estimate", crowdVoiceCountEstimate ?: JSONObject.NULL)
            // Empty map → emit empty object; downstream fanout treats absent
            // and empty identically.
            put("audio_class_event_counts", JSONObject().apply {
                for ((cls, count) in audioClassEventCounts) put(cls, count)
            })

            // Phase 4 PR 8: YAMNet per-class histogram (per-class counts in
            // 0.5s bins). Snapshot + reset is atomic under yamnetHistogramLock.
            // Empty histogram → omit the field; embeddingWorker fanout treats
            // a missing field as no rows produced.
            try {
                val histogramSnapshot: Map<String, Map<String, Int>> = synchronized(yamnetHistogramLock) {
                    val snap = yamnetHistogramExtractor.snapshotForPayload()
                    yamnetHistogramExtractor.reset()
                    snap
                }
                if (histogramSnapshot.isNotEmpty()) {
                    val histogramJson = JSONObject()
                    for ((binKey, perClass) in histogramSnapshot) {
                        val perClassJson = JSONObject()
                        for ((clsLabel, count) in perClass) {
                            perClassJson.put(clsLabel, count)
                        }
                        histogramJson.put(binKey, perClassJson)
                    }
                    put("audio_class_histogram", histogramJson)
                }
            } catch (e: Exception) {
                Log.w(TAG, "[YamnetHistogram] snapshot/emit failed (non-fatal): ${e.message}")
            }

            // Ambient light metrics (from SensorCollector via face snapshot environment)
            put("ambientLux", if (avgAmbientLux >= 0f) avgAmbientLux else JSONObject.NULL)
            put("luxVariance", computedLuxVariance)
            put("viewabilityScore", viewabilityScore ?: JSONObject.NULL)

            // WiFi CSI sensing (ESP32 node + UDP aggregation)
            if (currentCsiSnapshot != null) {
                put("csiOccupantCount", currentCsiSnapshot.occupantCount)
                put("csiMotionScore", currentCsiSnapshot.motionScore.toDouble())
                put("csiSignalQuality", currentCsiSnapshot.signalQuality.toDouble())
                put("csiSubcarrierCount", currentCsiSnapshot.subcarrierCount)
                put("csiCaptureRateHz", currentCsiSnapshot.captureRateHz.toDouble())
                put("csiHardwareType", currentCsiSnapshot.hardwareType)
                put("csiNodeCount", effectiveCsiNodeCount)
                put("csiNodeId", currentCsiSnapshot.nodeId)
                put("csiAvgRssiDbm", currentCsiSnapshot.avgRssiDbm)
                put("csiFramesProcessed", currentCsiSnapshot.framesProcessed)
                put("csiFramesDropped", currentCsiSnapshot.framesDropped)
                put("csiWindowStartMs", currentCsiSnapshot.windowStartMs)
                put("csiWindowEndMs", currentCsiSnapshot.windowEndMs)
            } else {
                put("csiNodeCount", effectiveCsiNodeCount)
            }

            // Speech OFV is the CLOUD's job now (cloud owns speech OFV). The
            // SpeechIntelligenceProcessor sends EVERY usable transcript to
            // /v2/earner/analyze-speech, where the server writes the declared
            // fields + the IAB chip-cloud (provenance cloud_speech +
            // cloud_synthesis) — the same structured-output machinery vision uses.
            // Re-emitting a `speech` block on /audience-analyze would double-write
            // the same utterance under edge_baseline (Codex P2, PR #6622), racing
            // the OFV PK under ON CONFLICT DO NOTHING and inflating speech counts.
            // aggregatedSpeech still feeds the live-panes summary + telemetry
            // counts/logs above; it is intentionally NOT re-persisted here.

            // Emotional engagement metrics (pose, emotion, gaze)
            if (emotionalEngagement != null) {
                put("emotionalEngagement", emotionalEngagement.toJson())

                // Derive behavioralContext from on-device ML signals
                // Backend maps behavioralContext.primaryMood → primary_mood column
                // and behavioralContext.movementPace → movement_pace column
                val dominantEmotion = emotionalEngagement.emotion?.dominantEmotion
                val primaryMood = when (dominantEmotion) {
                    EmotionType.HAPPY -> "happy"
                    EmotionType.SURPRISED -> "excited"
                    EmotionType.ANGRY -> "frustrated"
                    EmotionType.SAD -> "bored"
                    EmotionType.DISGUSTED -> "frustrated"
                    EmotionType.CONTEMPT -> "skeptical"
                    EmotionType.FEARFUL -> "anxious"
                    EmotionType.NEUTRAL -> {
                        // Differentiate neutral: high attention = focused, low = idle
                        if (avgAttention > 0.5) "focused" else "neutral"
                    }
                    else -> null
                }

                val pose = emotionalEngagement.pose
                val moveCount = { state: MovementState -> pose?.movementDistribution?.get(state) ?: 0 }
                val stopped = moveCount(MovementState.STOPPED)
                val walking = moveCount(MovementState.WALKING_FAST)
                val approaching = moveCount(MovementState.APPROACHING)
                val movementPace = when {
                    pose == null -> "strolling"
                    stopped > walking -> "stationary"
                    walking > 0 && approaching > 0 -> "strolling"
                    walking > stopped -> "hurrying"
                    else -> "strolling"
                }

                val gaze = emotionalEngagement.gaze
                val screenEngagement = when {
                    gaze != null && gaze.lookingAtScreenPct > 0.7f -> "engaged"
                    gaze != null && gaze.lookingAtScreenPct > 0.3f -> "glancing"
                    gaze != null -> "ignoring"
                    else -> "unknown"
                }

                put("behavioralContext", JSONObject().apply {
                    put("primaryMood", primaryMood ?: JSONObject.NULL)
                    put("movementPace", movementPace)
                    put("screenEngagement", screenEngagement)
                })
            }

            // Ad-audience correlation: what content is playing during this sensing window
            if (contentState != null) {
                put("currentAdId", contentState.adId ?: JSONObject.NULL)
                put("currentContentType", contentState.contentType)
                put("currentImpressionId", contentState.impressionId ?: JSONObject.NULL)
            } else {
                put("currentAdId", JSONObject.NULL)
                put("currentContentType", "idle")
                put("currentImpressionId", JSONObject.NULL)
            }

            // Phase 1c — per-(face_track × ad_id) attention observations.
            // Additive to the existing aggregate histograms (movement /
            // emotion / gaze) — older payload consumers ignore the field;
            // the new server-side consumer fans each row into
            // observation_field_values with field_key
            // 'per_face_obs_ad_<adId>_face_<faceId>_<dim>' for typed OFV
            // access. The accumulator drains AND clears state; the next
            // window starts fresh.
            val perFaceDrain = audienceAnalyzer?.drainPerAdAttention() ?: emptyMap()
            if (perFaceDrain.isNotEmpty()) {
                val perFaceJson = JSONArray()
                for ((adId, faceMap) in perFaceDrain) {
                    // Source recorded at setCurrentAd time via the 3-arg
                    // overload. Null for legacy windows where the JS bridge
                    // didn't populate creativeSource (e.g. older WebView build).
                    val creativeSource = audienceAnalyzer?.getSourceFor(adId)
                    for ((faceTrackId, attention) in faceMap) {
                        perFaceJson.put(JSONObject().apply {
                            // `ad_id` kept for back-compat with cloud
                            // consumers that key off the IMA programmatic
                            // ad name. `creative_id` is the unified key
                            // (== ad_id when source==ima_programmatic,
                            // == stream item id otherwise).
                            put("ad_id", adId)
                            put("creative_id", adId)
                            put("creative_source", creativeSource ?: JSONObject.NULL)
                            put("face_track_id", faceTrackId)
                            put("dwell_seconds", attention.dwellSecondsTotal.toDouble())
                            put("gaze_seconds", attention.gazeSecondsTotal.toDouble())
                            put("attention_p50", attention.attentionP50.toDouble())
                            put("attention_max", attention.attentionMax.toDouble())
                            put("dominant_emotion", attention.dominantEmotion.name)
                            put("emotion_confidence_max", attention.emotionConfidenceMax.toDouble())
                            put("gaze_pct_on_screen", attention.gazePctOnScreen.toDouble())
                            put("primary_focus_region", attention.primaryFocusRegion)
                            put("age_bucket", attention.ageBucket ?: JSONObject.NULL)
                            put("gender", attention.gender ?: JSONObject.NULL)
                            put("first_seen_ms", attention.firstSeenMs)
                            put("last_seen_ms", attention.lastSeenMs)
                            put("sample_count", attention.sampleCount)
                        })
                    }
                }
                put("per_face_observations", perFaceJson)
                // Mirror onto the field FrameCaptureManager's HTTP POST will
                // read. The snapshot lives until the NEXT drain replaces it,
                // covering the gap between this 10-sec WS window and the
                // 30-sec HTTP cadence. Worst case the HTTP body carries the
                // previous window's per-face — acceptable since the cloud
                // joins on observation_id (not time-aligned anyway).
                latestPerFaceSnapshot = perFaceJson
            }

            // On-device demographics (10s granularity, augments Gemini 5-min captures)
            // Skip if tier >= MEDIUM (age/gender processor unloaded).
            //
            // Phase 2 (2026-05-04): backed by FaceXFormer Swin-B foundation
            // model delivered via OTA (sensing_model_registry → S3 presigned
            // URL → ModelDownloadManager). If hasModel()=false (download not
            // yet complete) OR no faces seen this window, we EMIT NOTHING
            // (no onDeviceDemographics block). Honest null is the only
            // acceptable degraded state — the previous heuristic fallback
            // fabricated data and was removed in this PR.
            // PR 10 diagnostic — emit the 3 gate predicates independently so
            // the next operator can tell from logcat WHY FaceXFormer isn't
            // emitting demographics on a given device. Throttled to once per
            // ~30s (3 emission windows) to avoid logcat flood. Symptom we are
            // diagnosing: 0 of 857,819 audience_metrics rows in the last 7d
            // had on_device_age_distribution populated.
            val tierOk = currentTier == null || currentTier.ordinal < MemoryAttenuationManager.AttenuationTier.MEDIUM.ordinal
            val gateAgeGenderProcessor = audienceAnalyzer?.getAgeGenderProcessor()
            val processorPresent = gateAgeGenderProcessor != null
            val hasFaces = gateAgeGenderProcessor?.hasFaces() ?: false
            // Codex P2 (#5763): monotonic clock so NTP / timezone / manual time
            // adjustments don't make `nowMs - lastAgeGenderDiagLogMs` jump
            // negative (silent log starvation) or huge (log flood).
            val nowMs = android.os.SystemClock.elapsedRealtime()
            if (nowMs - lastAgeGenderDiagLogMs >= 30_000L) {
                Log.d(TAG, "[AgeGender-Diag] tier=${currentTier?.name ?: "NORMAL"} " +
                    "tierOk=$tierOk processorPresent=$processorPresent hasFaces=$hasFaces")
                lastAgeGenderDiagLogMs = nowMs
            }

            if (tierOk) {
                val ageGenderProcessor = gateAgeGenderProcessor
                if (ageGenderProcessor != null && hasFaces) {
                    put("onDeviceDemographics", JSONObject().apply {
                        put("ageDistribution", ageGenderProcessor.getAgeDistribution())
                        put("genderSplit", ageGenderProcessor.getGenderSplit())
                        put("confidence", ageGenderProcessor.getAvgConfidence())
                        put("source", "on_device_facexformer")
                    })
                    ageGenderProcessor.resetAccumulators()
                }
                // else: foundation model not loaded or no faces seen — emit
                // no onDeviceDemographics block. on_device_age_distribution
                // and on_device_gender_split remain NULL in CH for this row.
            }

            // Creative zone attention heatmap (3×3 viewport grid).
            // Skip if tier >= HIGH (FaceLandmarker unloaded under memory pressure).
            if (currentTier == null || currentTier.ordinal < MemoryAttenuationManager.AttenuationTier.HIGH.ordinal) {
                val analyzer = audienceAnalyzer
                if (analyzer != null && analyzer.isFaceLandmarkerActive()) {
                    val zoneDistribution = analyzer.getGazeZoneDistribution()
                    if (zoneDistribution.isNotEmpty()) {
                        put("creativeZones", JSONObject().apply {
                            zoneDistribution.forEach { (zone, pct) ->
                                put(GazeZoneLabels.zoneToLabel(zone), pct)
                            }
                        })
                    }
                    analyzer.resetZoneDwell()
                }
            }

            // Read shadow events from sensor collector (reset counter for this window)
            put("shadowEvents", shadowEvents)

            // Foot traffic estimation (fusing face count + audio occupancy + shadow events)
            // Skip if tier >= MEDIUM (foot traffic estimator unloaded, depends on face extrapolation)
            if (footTrafficMetrics != null && footTrafficMetrics.estimatedCount > 0) {
                put("footTraffic", footTrafficEstimator.toJson(footTrafficMetrics).also {
                    it.put("shadowEvents", shadowEvents)
                })
            }

            put("measurementQuality", edgeQualityTelemetry.measurementQuality)
            put("observationFamily", edgeQualityTelemetry.observationFamily)
            put("evidenceGrade", edgeQualityTelemetry.evidenceGrade)
            put("decisionability", edgeQualityTelemetry.decisionability)
            put("decisionBlockReasons", JSONArray(edgeQualityTelemetry.decisionBlockReasons))
            put("visualEvidence", JSONObject().apply {
                put("currentFaceCount", edgeQualityTelemetry.sensing.currentFaceCount)
                put("currentPersonCount", edgeQualityTelemetry.sensing.currentPersonCount)
                put("avgFaceCountWindow", edgeQualityTelemetry.sensing.avgFaceCountWindow)
                put("avgPersonCountWindow", edgeQualityTelemetry.sensing.avgPersonCountWindow)
                put("maxFaceCountWindow", edgeQualityTelemetry.sensing.maxFaceCountWindow)
                put("estimatedOccupancy", edgeQualityTelemetry.sensing.estimatedOccupancy ?: JSONObject.NULL)
            })
            put("edgeQuality", edgeQualityTelemetry.toJson())
            subsystemHealthProvider?.invoke()?.let { put("subsystemHealth", it) }

            // Calibration telemetry — effective thresholds, config sources, signal agreement
            try {
                put("calibration", CalibrationTelemetry.buildTelemetry())
            } catch (e: Exception) {
                Log.w(TAG, "[CalibrationTelemetry] Failed to build telemetry (non-fatal): ${e.message}")
            }

            // Metadata
            put("faceSnapshotCount", faceSnapshots.size)
            put("audioSnapshotCount", audioSnapshots.size)
            put("speechSnapshotCount", speechSnapshots.size)
            put("sensingMode", _currentState.value.mode.name)
            put("attenuationTier", (currentTier ?: MemoryAttenuationManager.AttenuationTier.NORMAL).name)
            put("venueType", venueType ?: JSONObject.NULL)
        }

        // --- VAS + Federated Learning Integration ---
        val trainer = currentVenueConfig?.trainer
        if (trainer != null && avgFaceCount > 0) {
            try {
                val emotionalEng = emotionalEngagement?.overallEngagementScore ?: 0f
                val bodyEng = emotionalEngagement?.pose?.bodyEngagementScore ?: 0f
                val focusReg = audienceAnalyzer?.getAverageFocusRegion() ?: 4
                val currentAdId = contentState?.adId

                val currentDaypart = FederatedTrainer.computeDaypart()

                val vasResult = trainer.computeVAS(
                    avgAttention.toFloat(), avgDwellMs.toFloat(), avgFaceCount.toFloat(),
                    emotionalEng, bodyEng, focusReg, AGGREGATION_WINDOW_MS, currentAdId,
                    currentDaypart
                )

                // Add VAS to payload (include signingTimestamp for server attestation verification)
                payload.put("vasRaw", vasResult.vasRaw)
                payload.put("vasWeighted", vasResult.vasWeighted)
                payload.put("vasQualityMultiplier", vasResult.qualityMultiplier)
                payload.put("attestationSignature", vasResult.attestationSignature ?: JSONObject.NULL)
                payload.put("publicKeyId", vasResult.publicKeyId ?: JSONObject.NULL)
                payload.put("vasTimestamp", vasResult.signingTimestamp ?: JSONObject.NULL)
                payload.put("bodyEngagementScore", bodyEng)
                payload.put("focusRegion", focusReg)
            } catch (vasErr: Exception) {
                Log.w(TAG, "[VAS] Computation failed (non-fatal): ${vasErr.message}")
            }
        }

        // Phase 4 PR 2/3 — clip-trigger evaluation + DRY-RUN capture. Snapshot
        // is built from the same aggregated values that make up the
        // audienceSignals payload so the evaluator sees exactly what the
        // server sees. PR 3 (this PR): when a rule fires we kick off
        // ClipRecorder → ClipUploadManager on a separate IO coroutine; the
        // emit path is never blocked.
        try {
            val rules = activeClipTriggerRules
            if (rules.isNotEmpty()) {
                val signalSnapshot = mapOf<String, Any?>(
                    "face_count" to avgFaceCount,
                    "person_count" to avgPersonCount,
                    "max_face_count" to maxFaceCount,
                    "attention" to avgAttention,
                    "dwell_ms" to avgDwellMs,
                    "noise_level" to avgNoiseTier.takeIf { it >= 0 },
                    "noise_tier" to avgNoiseTier.takeIf { it >= 0 },
                    "crowded_pct" to crowdedPct,
                    "ambient_lux" to avgAmbientLux.takeIf { it >= 0f },
                    "ambience" to dominantAmbience,
                    "estimated_occupancy" to estimatedOccupancy,
                    "ad_receptivity" to avgAdReceptivity,
                    "dominant_emotion" to (emotionalEngagement?.emotion?.dominantEmotion?.name),
                    "engagement_score" to (emotionalEngagement?.overallEngagementScore?.toDouble()),
                    "body_engagement" to (emotionalEngagement?.pose?.bodyEngagementScore?.toDouble()),
                    // "purchase_intent" removed — cloud Gemini emits the
                    // canonical signal; clip triggers can read it from the
                    // post-merge audience_metrics row server-side.
                    "speaker_count" to (aggregatedSpeech?.estimatedSpeakerCount),
                    "brand_mentions" to (aggregatedSpeech?.brandMentions?.size),
                )
                val fired = clipTriggerEvaluator.evaluate(rules, signalSnapshot)
                if (fired.isNotEmpty()) {
                    Log.i(TAG, "[ClipTriggers] ${fired.size} trigger(s) fired this cycle: " +
                        fired.joinToString { it.rule.triggerId })
                    // PR 3 — fire-and-forget DRY-RUN clip capture per fired rule.
                    // Do not await; the aggregation/emit path stays on its 10s
                    // budget. dispatchClipDryrunCapture() handles dedup via the
                    // recorder's own busy-flag.
                    fired.forEach { dispatchClipDryrunCapture(it) }
                }
            }
        } catch (e: Exception) {
            // Evaluator is pure; failures should be impossible unless the
            // signal map throws. Belt-and-suspenders to keep the emit path
            // alive — if the trigger logic blows up, we still ship signals.
            Log.w(TAG, "[ClipTriggers] evaluation failed (non-fatal): ${e.message}")
        }

        // Tee the anonymous aggregate to the host-app callback (partner SDK).
        // Wrapped + fire-and-forget — a throwing host callback must never break
        // the emit path or the ~10s aggregation budget. No-op when unset
        // (first-party agents pay nothing).
        onSensingResult?.let { cb ->
            try {
                cb(SensingResult(
                    faceCount = avgFaceCount,
                    personCount = avgPersonCount,
                    attention = avgAttention,
                    dwellMs = avgDwellMs,
                    noiseLevel = avgNoiseTier.takeIf { it >= 0 },
                    ambientLux = avgAmbientLux.takeIf { it >= 0f },
                    ambience = dominantAmbience,
                    adReceptivity = avgAdReceptivity,
                    dominantEmotion = emotionalEngagement?.emotion?.dominantEmotion?.name,
                    engagementScore = emotionalEngagement?.overallEngagementScore?.toDouble(),
                    bodyEngagement = emotionalEngagement?.pose?.bodyEngagementScore?.toDouble(),
                    speakerCount = aggregatedSpeech?.estimatedSpeakerCount,
                    brandMentionCount = aggregatedSpeech?.brandMentions?.size,
                    windowEndEpochMs = System.currentTimeMillis(),
                ))
            } catch (e: Exception) {
                Log.w(TAG, "[SensingResult] host callback threw (non-fatal): ${e.message}")
            }
        }

        // Emit unified audience signals to backend (with offline buffering)
        if (socketManager.isConnected()) {
            socketManager.emit("audienceSignals", payload)
            // Also flush any previously buffered signals
            sensingScope.launch(Dispatchers.IO) {
                try {
                    val flushed = signalBuffer?.flushBuffer(socketManager, 3) ?: 0
                    if (flushed > 0) {
                        Log.i(TAG, "[Buffer] Flushed $flushed previously buffered signals")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "[Buffer] Flush failed: ${e.message}")
                }
            }
        } else {
            // Socket disconnected — buffer for later
            sensingScope.launch(Dispatchers.IO) {
                signalBuffer?.bufferSignal(payload)
                val queued = signalBuffer?.count() ?: 0
                Log.w(TAG, "[Buffer] Socket disconnected, buffered signal ($queued queued)")
            }
        }

        // Log summary
        val speechSummary = if (aggregatedSpeech != null && aggregatedSpeech.confidence > 0) {
            val shoppingContexts = aggregatedSpeech.shoppingContexts.take(2).joinToString("|").ifBlank { "none" }
            "speech=${aggregatedSpeech.brandMentions.size}brands/stage=${aggregatedSpeech.purchaseJourney.stage}/contexts=$shoppingContexts"
        } else "speech=none"

        Log.d(TAG, "[Aggregation] EmotionalEngagement: " +
                "result=${emotionalEngagement != null}, " +
                "confidence=${emotionalEngagement?.confidence ?: -1}, " +
                "score=${emotionalEngagement?.overallEngagementScore ?: -1}")

        val emotionalSummary = if (emotionalEngagement != null && emotionalEngagement.confidence > 0) {
            "engagement=${String.format("%.2f", emotionalEngagement.overallEngagementScore)}/${emotionalEngagement.audienceReaction.name}"
        } else "engagement=none"

        Log.i(TAG, "[VLM] Runtime ${currentVlmRuntimeTelemetry.toLogSummary()}")

        Log.i(TAG, "[Socket] >>> audienceSignals emitted - " +
                "avgFaces=${String.format("%.1f", avgFaceCount)}, " +
                "avgPersons=${String.format("%.1f", avgPersonCount)}, " +
                "maxFaces=$maxFaceCount, " +
                "attention=${String.format("%.2f", avgAttention)}, " +
                "ambience=$dominantAmbience, " +
                "receptivity=${avgAdReceptivity?.let { String.format("%.2f", it) } ?: "n/a"}, " +
                "$speechSummary, $emotionalSummary")
    }

    /**
     * Aggregate multiple speech insight snapshots into a single combined insight.
     * Takes the strongest signals from the window.
     */
    private fun aggregateSpeechInsights(snapshots: List<SpeechInsights>): SpeechInsights? {
        if (snapshots.isEmpty()) return null

        // Collect all unique brands and products mentioned
        val allBrands = snapshots.flatMap { it.brandMentions }.distinct()
        val allProducts = snapshots.flatMap { it.productCategories }.distinct()
        val allObjections = snapshots.flatMap { it.objections }.distinct()
        val allInterests = snapshots.flatMap { it.interests }.distinct()

        // strongestIntent aggregation removed — SpeechInsights.purchaseIntent
        // was deleted in cosmic-brewing-bear task #26 (cloud Gemini emits
        // the canonical enum).

        // Any price or availability inquiry in the window
        val anyPriceInquiry = snapshots.any { it.priceInquiry }
        val anyAvailabilityInquiry = snapshots.any { it.availabilityInquiry }
        val anyBuyingToday = snapshots.any { it.mentionedBuyingToday }

        // Dominant sentiment and conversation type
        val dominantSentiment = snapshots
            .groupBy { it.sentimentTone }
            .maxByOrNull { it.value.size }
            ?.key ?: SentimentTone.NEUTRAL

        val dominantConversationType = snapshots
            .groupBy { it.conversationType }
            .maxByOrNull { it.value.size }
            ?.key ?: ConversationType.UNKNOWN

        // Max speaker count observed
        val maxSpeakers = snapshots.maxOfOrNull { it.estimatedSpeakerCount } ?: 0

        // Average confidence
        val avgConfidence = snapshots.map { it.confidence }.average().toFloat()
        val mergedPurchaseJourney = mergePurchaseJourney(snapshots)
        val speechSemantics = mergeSpeechSemantics(
            snapshots = snapshots,
            mergedPurchaseJourney = mergedPurchaseJourney,
            avgConfidence = avgConfidence
        )

        // Sense-Anything (speech): merge the generic operator-declared fields
        // TYPE-AWARE (an untyped "latest" would corrupt counters — the flagship
        // "count how many people asked for X" use case). Carry the dominant
        // classificationSource forward so the payload builder can detect a
        // cloud-sourced window and avoid a double OFV write.
        val mergedProfileFields = mergeProfileFields(snapshots, activeSignalsJson)
        val dominantClassificationSource = snapshots
            .map { it.classificationSource }
            .filter { it.isNotBlank() && it != "unknown" }
            .groupingBy { it }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key ?: "unknown"

        return SpeechInsights(
            brandMentions = allBrands,
            brandMatches = snapshots
                .flatMap { it.brandMatches }
                .distinctBy { "${it.canonicalName}:${it.observedText}:${it.matchType}" },
            productCategories = allProducts,
            shoppingContexts = snapshots.flatMap { it.shoppingContexts }.distinct(),
            priceInquiry = anyPriceInquiry,
            availabilityInquiry = anyAvailabilityInquiry,
            purchaseJourney = mergedPurchaseJourney,
            mentionedBuyingToday = anyBuyingToday,
            objections = allObjections,
            objectionInsights = snapshots
                .flatMap { it.objectionInsights }
                .distinctBy { "${it.theme}:${it.summaryLabel}:${it.evidencePhrase}" },
            interests = allInterests,
            sentimentTone = dominantSentiment,
            conversationType = dominantConversationType,
            estimatedSpeakerCount = maxSpeakers,
            confidence = avgConfidence,
            classificationSource = dominantClassificationSource,
            speechSemantics = speechSemantics,
            profileFields = mergedProfileFields
        )
    }

    /**
     * Merge the generic profile-field maps across the speech snapshots in a
     * window, TYPE-AWARE. Each declared field reduces per its declared type:
     *   list                     → union, de-duplicated
     *   counter                  → sum (per-window counts accumulate)
     *   gauge                    → mean
     *   boolean                  → OR (true if any window saw it)
     *   enum / category / string → mode (dominant value; ties → latest)
     *   unknown declared type    → latest non-null value
     * Untyped "latest" would corrupt counters — the whole point of declaring a
     * counter (e.g. "oat-milk requests this window") is that it accumulates.
     */
    private fun mergeProfileFields(
        snapshots: List<SpeechInsights>,
        signalsJson: String?
    ): Map<String, Any> {
        val withFields = snapshots.filter { it.profileFields.isNotEmpty() }
        if (withFields.isEmpty()) return emptyMap()

        val types = parseSpeechFieldTypes(signalsJson)
        val out = LinkedHashMap<String, Any>()
        // Preserve first-seen key order across snapshots for stable payloads.
        val allKeys = LinkedHashSet<String>().apply {
            withFields.forEach { addAll(it.profileFields.keys) }
        }

        for (key in allKeys) {
            val values = withFields.mapNotNull { it.profileFields[key] }
            if (values.isEmpty()) continue
            val merged: Any? = when (types[key]) {
                "list" -> values
                    .flatMap { v -> (v as? List<*>)?.map { it.toString() } ?: listOf(v.toString()) }
                    .distinct()
                "counter" -> values.sumOf { v ->
                    (v as? Number)?.toInt() ?: v.toString().toIntOrNull() ?: 0
                }
                "gauge" -> values
                    .mapNotNull { v -> (v as? Number)?.toDouble() ?: v.toString().toDoubleOrNull() }
                    .takeIf { it.isNotEmpty() }
                    ?.average()
                "boolean" -> values.any { v -> (v as? Boolean) ?: v.toString().toBoolean() }
                // enum / category / string → mode (dominant value); tie → latest
                "enum", "category", "string" -> values
                    .map { it.toString() }
                    .groupingBy { it }
                    .eachCount()
                    .maxByOrNull { it.value }
                    ?.key
                // unknown declared type → latest non-null value
                else -> values.last()
            }
            if (merged != null && !(merged is List<*> && merged.isEmpty())) {
                out[key] = merged
            }
        }
        return out
    }

    /** Parse the active profile's signals JSON into field → declared type (speech only). */
    private fun parseSpeechFieldTypes(signalsJson: String?): Map<String, String> {
        if (signalsJson.isNullOrBlank()) return emptyMap()
        return SignalsToToolSpecBuilder.parseSignalsArray(signalsJson)
            .filter { it.source == "speech" }
            .associate { it.field to it.type }
    }

    private fun mergeSpeechSemantics(
        snapshots: List<SpeechInsights>,
        mergedPurchaseJourney: PurchaseJourney,
        avgConfidence: Float
    ): SpeechSemantics? {
        val entities = snapshots
            .mapNotNull { it.speechSemantics }
            .flatMap { it.entities }
            .distinctBy { "${it.name}:${it.type}:${it.role}:${it.polarity}" }
        val reasons = snapshots
            .mapNotNull { it.speechSemantics }
            .flatMap { it.reasons }
            .distinctBy { "${it.target}:${it.type}:${it.detail}" }
        val questions = snapshots
            .mapNotNull { it.speechSemantics }
            .flatMap { it.questions }
            .distinctBy { "${it.type}:${it.text}" }
        val evidencePhrases = linkedSetOf<String>().apply {
            snapshots.mapNotNull { it.speechSemantics }
                .flatMapTo(this) { it.evidencePhrases }
            mergedPurchaseJourney.evidencePhrases.forEach { add(it) }
        }.toList()
        val merged = SpeechSemantics(
            entities = entities,
            reasons = reasons,
            questions = questions,
            journeyState = mergedPurchaseJourney.stage.takeUnless { it == PurchaseJourney.NONE_STAGE },
            confidence = avgConfidence,
            evidencePhrases = evidencePhrases,
            brandMatches = snapshots
                .flatMap { it.brandMatches }
                .distinctBy { "${it.canonicalName}:${it.observedText}:${it.matchType}" },
            shoppingContexts = snapshots.flatMap { it.shoppingContexts }.distinct(),
            purchaseJourney = mergedPurchaseJourney,
            objectionInsights = snapshots
                .flatMap { it.objectionInsights }
                .distinctBy { "${it.theme}:${it.summaryLabel}:${it.evidencePhrase}" }
        )

        return merged.takeIf { it.hasSignals() }
    }

    private fun SpeechSemantics.toJson(): JSONObject = JSONObject().apply {
        put("entities", JSONArray(entities.map { entity ->
            JSONObject().apply {
                put("name", entity.name)
                put("type", entity.type ?: JSONObject.NULL)
                put("role", entity.role ?: JSONObject.NULL)
                put("polarity", entity.polarity ?: JSONObject.NULL)
            }
        }))
        put("reasons", JSONArray(reasons.map { reason ->
            JSONObject().apply {
                put("target", reason.target ?: JSONObject.NULL)
                put("type", reason.type ?: JSONObject.NULL)
                put("detail", reason.detail ?: JSONObject.NULL)
            }
        }))
        put("questions", JSONArray(questions.map { question ->
            JSONObject().apply {
                put("type", question.type ?: JSONObject.NULL)
                put("text", question.text ?: JSONObject.NULL)
            }
        }))
        put("journeyState", journeyState ?: JSONObject.NULL)
        put("confidence", confidence?.toDouble() ?: JSONObject.NULL)
        put("evidencePhrases", JSONArray(evidencePhrases))
        put("brandMatches", JSONArray(brandMatches.map { match ->
            JSONObject().apply {
                put("canonicalName", match.canonicalName)
                put("observedText", match.observedText)
                put("confidence", match.confidence.toDouble())
                put("matchType", match.matchType)
            }
        }))
        put("shoppingContexts", JSONArray(shoppingContexts))
        put("purchaseJourney", purchaseJourney?.let { journey ->
            JSONObject().apply {
                put("stage", journey.stage)
                put("urgency", journey.urgency)
                put("journeySignals", JSONArray(journey.journeySignals))
                put("evidencePhrases", JSONArray(journey.evidencePhrases))
            }
        } ?: JSONObject.NULL)
        put("objectionInsights", JSONArray(objectionInsights.map { objection ->
            JSONObject().apply {
                put("theme", objection.theme)
                put("summaryLabel", objection.summaryLabel ?: JSONObject.NULL)
                put("evidencePhrase", objection.evidencePhrase ?: JSONObject.NULL)
                put("confidence", objection.confidence.toDouble())
            }
        }))
    }

    private fun mergePurchaseJourney(snapshots: List<SpeechInsights>): PurchaseJourney {
        if (snapshots.isEmpty()) return PurchaseJourney()

        val stagePriority = mapOf(
            "ready_to_buy" to 5,
            "repeat_consideration" to 4,
            "active_comparison" to 3,
            "active_consideration" to 2,
            "casual_browsing" to 1,
            PurchaseJourney.NONE_STAGE to 0
        )
        val urgencyPriority = mapOf(
            "immediate" to 4,
            "high" to 3,
            "medium" to 2,
            "low" to 1,
            "none" to 0
        )

        val selectedStage = snapshots
            .map { it.purchaseJourney.stage }
            .maxByOrNull { stagePriority[it] ?: 0 }
            ?: PurchaseJourney.NONE_STAGE
        val selectedUrgency = snapshots
            .map { it.purchaseJourney.urgency }
            .maxByOrNull { urgencyPriority[it] ?: 0 }
            ?: "none"

        return PurchaseJourney(
            stage = selectedStage,
            urgency = selectedUrgency,
            journeySignals = snapshots
                .flatMap { it.purchaseJourney.journeySignals }
                .distinct(),
            evidencePhrases = snapshots
                .flatMap { it.purchaseJourney.evidencePhrases }
                .distinct()
                .take(8)
        )
    }

    /**
     * Debug-only hook used by the tablet debug APK to force a structured speech payload
     * through the normal aggregation path for deterministic end-to-end verification.
     */
    fun injectDebugSpeechInsights(insights: SpeechInsights) {
        synchronized(speechInsightsBuffer) {
            while (speechInsightsBuffer.size >= 50) {
                speechInsightsBuffer.removeFirst()
            }
            speechInsightsBuffer.add(insights)
        }
        Log.i(
            TAG,
            "[Speech][Debug] Injected structured speech - " +
                "stage=${insights.purchaseJourney.stage}, contexts=${insights.shoppingContexts.joinToString("|")}"
        )
        aggregateAndEmit()
    }

    /**
     * Parse the audio occupancy range string (e.g., "0-5", "10-20") to a numeric midpoint.
     * Returns null for "unknown" or unparseable strings.
     */
    private fun parseOccupancyToInt(occupancy: String): Int? {
        if (occupancy == "unknown") return null
        val parts = occupancy.split("-")
        return if (parts.size == 2) {
            val low = parts[0].toIntOrNull() ?: return null
            val high = parts[1].toIntOrNull() ?: return null
            (low + high) / 2
        } else {
            occupancy.toIntOrNull()
        }
    }

    /**
     * Stop audience sensing.
     */
    fun stop() {
        Log.i(TAG, "========================================")
        Log.i(TAG, "[Service] >>> STOPPING AUDIENCE SENSING")
        Log.i(TAG, "========================================")

        if (!isRunning) {
            Log.w(TAG, "[Service] Not running, ignoring stop()")
            return
        }

        isRunning = false

        Log.d(TAG, "[Service] Stopping face detection...")
        audienceAnalyzer?.stop()
        audienceAnalyzer = null

        Log.d(TAG, "[Service] Stopping audio classification...")
        audioProcessor?.stop()
        audioProcessor = null

        Log.d(TAG, "[Service] Stopping speech intelligence...")
        speechProcessor?.stop()
        speechProcessor = null

        Log.d(TAG, "[Service] Releasing frame capture...")
        frameCaptureManager?.release()
        frameCaptureManager = null

        // Reset perception trigger and VLM state
        perceptionTrigger.reset()
        lastVlmOutput = null
        vlmFrameJob?.cancel()
        vlmFrameJob = null
        vlmFrameAdmissionGate.reset()

        // Unregister device hotplug callbacks
        Log.d(TAG, "[Service] Unregistering device hotplug callbacks...")
        unregisterDeviceHotplugCallbacks()
        lifecycleOwnerRef = null

        // Emit final aggregation
        Log.d(TAG, "[Service] Emitting final aggregation...")
        aggregateAndEmit()

        _currentState.value = SensingState(isRunning = false)

        // Destroy federated learning resources to prevent leaks
        venueConfig?.let { vc ->
            Log.d(TAG, "[Service] Destroying FederatedTrainer and ModelManager...")
            vc.trainer.destroy()
            vc.manager.destroy()
        }
        venueConfig = null
        venueType = null

        Log.i(TAG, "========================================")
        Log.i(TAG, "[Service] >>> AUDIENCE SENSING STOPPED")
        Log.i(TAG, "========================================")
    }

    /**
     * Rebind camera after app returns from background.
     *
     * When the app goes to background, Android releases the camera. This method
     * forces the camera to rebind so face detection resumes when returning to foreground.
     *
     * @param lifecycleOwner The lifecycle owner to bind the camera to
     */
    fun rebindCamera(lifecycleOwner: LifecycleOwner) {
        if (!isRunning) {
            Log.w(TAG, "[Service] rebindCamera requested while stopped — recovery should call restartAudienceSensing, not rebind")
            return
        }

        Log.i(TAG, "[Service] >>> REBINDING CAMERA after app resume...")
        audienceAnalyzer?.rebindCamera(lifecycleOwner)
    }

    /**
     * Restart audio classification after app returns from background.
     *
     * When the app goes to background, audio resources may be released. This method
     * restarts audio classification when returning to foreground.
     */
    fun rebindAudio() {
        if (!isRunning) {
            Log.w(TAG, "[Service] rebindAudio requested while stopped — recovery should call restartAudienceSensing, not rebind")
            return
        }

        if (!hasMicrophoneHardware || !hasMicrophonePermission) {
            Log.d(TAG, "[Service] No microphone available, skipping audio rebind")
            return
        }

        Log.i(TAG, "[Service] >>> REBINDING AUDIO after app resume...")
        audioProcessor?.restart()
    }

    /**
     * Set the screen ID (call when screen registration is complete).
     */
    fun setScreenId(id: String) {
        screenId = id
        frameCaptureManager?.setScreenId(id)
        Log.i(TAG, "Screen ID set: $id")
    }

    /**
     * Adopt the canonical device fingerprint resolved by check-screen. A partner
     * SDK initially identifies by its external_id (config.deviceId); once the
     * backend resolves the real binding it calls this so audience-analyze +
     * audienceSignals carry the canonical finger_print and attribute in both
     * soft- and strict-enforce modes. One-time startup update; no-op for a blank
     * or unchanged value (e.g. first-party agents already on their hardware fp).
     */
    fun setAudienceFingerprint(fp: String) {
        if (fp.isBlank() || fp == fingerprint) return
        fingerprint = fp
        frameCaptureManager?.setFingerprint(fp)
        Log.i(TAG, "Audience fingerprint adopted: ${fp.take(12)}…")
    }

    /**
     * Optional host-app callback, invoked once per aggregation window (~10s)
     * with the anonymous [SensingResult] for that window. Wired by the partner
     * SDK facade from `SensingSdkConfig.onSensingResult`; null for first-party
     * agents (zero overhead when unset). Never blocks the emit path.
     */
    @Volatile
    var onSensingResult: ((SensingResult) -> Unit)? = null

    /**
     * Set the venue ID for this screen's venue.
     * Called by DeviceAgentService when venue assignment is resolved from the API.
     */
    fun setVenueId(id: String) {
        venueId = id
        Log.i(TAG, "Venue ID set: $id")
    }

    /**
     * Set the venue type and initialize VAS + Federated Learning components.
     * Called by DeviceAgentService when venue_type is resolved from the API.
     */
    fun setVenueType(type: String) {
        // Idempotent: skip if already initialized with same venue type
        val currentConfig = venueConfig
        if (venueType == type && currentConfig != null) {
            Log.d(TAG, "setVenueType($type) already initialized, skipping")
            return
        }
        venueType = type
        val sid = screenId ?: ""

        // Destroy old VenueConfig resources before creating new ones
        currentConfig?.let { oldConfig ->
            Log.d(TAG, "Destroying old VenueConfig for type=${oldConfig.type}")
            oldConfig.trainer.destroy()
            oldConfig.manager.destroy()
        }

        // Construct the full VenueConfig atomically before assigning
        val trainer = FederatedTrainer(context, apiBaseUrl, fingerprint, sid, type)
        val manager = ModelManager(context, apiBaseUrl, type, fingerprint)
        venueConfig = VenueConfig(type, trainer, manager)

        // Register public key with server for attestation verification
        trainer.getPublicKeyPem()?.let { pem ->
            trainer.publicKeyId?.let { keyId ->
                socketManager.emit("registerPublicKey", JSONObject().apply {
                    put("publicKeyId", keyId)
                    put("publicKeyPem", pem)
                })
                Log.i(TAG, "Registered VAS public key $keyId with server")
            }
        }

        // Check for model updates in background
        sensingScope.launch(Dispatchers.IO) {
            try {
                manager.checkForUpdates()
            } catch (e: Exception) {
                Log.w(TAG, "Model update check failed: ${e.message}")
            }
        }

        Log.i(TAG, "Venue type set: $type, VAS + Federated Learning initialized")
    }

    /**
     * Set the content state provider for ad-audience correlation.
     * Called by tablet-agent to provide current content state (which ad is playing).
     *
     * Phase 1b: also forward to FrameCaptureManager so the JPEG REST payload
     * carries currentAdId / currentImpressionId. Provider injection may happen
     * before OR after initializeFrameCapture — handle both orderings.
     */
    fun setContentStateProvider(provider: ContentStateProvider) {
        contentStateProvider = provider
        // Race-safe forwarding: if frame capture is already up, push the
        // provider through; if not, initializeFrameCapture will pick it up
        // from this.contentStateProvider when it runs.
        frameCaptureManager?.contentStateProvider = provider
        Log.i(TAG, "Content state provider set")
    }

    /**
     * Set the VLM inference processor for perception-triggered reasoning (Phase 5B).
     *
     * When set, the aggregation loop will evaluate perception state changes
     * (person count, object counts, emotion, noise level) and trigger VLM
     * inference only when the scene has meaningfully changed. This replaces
     * the fixed 5-second sampling interval with event-driven triggering.
     *
     * @param processor The initialized VLMInferenceProcessor, or null to disable
     */
    fun setVlmProcessor(processor: VLMInferenceProcessor?) {
        vlmFrameJob?.cancel()
        vlmFrameJob = null
        vlmFrameAdmissionGate.reset()
        val previousProcessor = vlmProcessor
        vlmProcessor = swapReleasedReference(previousProcessor, processor) { prior ->
            try {
                prior.release()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to release previous VLM processor: ${e.message}", e)
            }
        }
        if (processor != null) {
            perceptionTrigger.reset()
            lastVlmOutput = null
            Log.i(TAG, "VLM processor set for perception-triggered reasoning")
        } else {
            lastVlmOutput = null
            Log.i(TAG, "VLM processor cleared")
        }
    }

    // Active sensing profile state
    @Volatile private var activeProfileId: String? = null
    @Volatile private var activeProfileModels: List<String> = emptyList()
    @Volatile private var activeVlmPrompt: String? = null
    @Volatile private var activeProgramSpecJson: String? = null
    @Volatile private var activeProgramSpecVersion: String? = null
    @Volatile private var activeProgramRuntimeContract: ObservationProgramRuntimeContract? = null
    @Volatile private var activeVlmRuntimeTelemetry: VlmRuntimeTelemetry = VlmRuntimeTelemetry()
    // Raw JSON array string of `program_spec.observation_program.signals` from the
    // active profile. Fed to OnDeviceLlmInsightExtractor at (re)init time so it
    // builds its OpenApiTool spec from the profile's declared speech fields.
    @Volatile private var activeSignalsJson: String? = null

    /** Set the VLM prompt for the active profile (called before applyProfileToInference). */
    fun setVlmPrompt(prompt: String?) {
        activeVlmPrompt = prompt
    }

    /** Set the signals[] JSON array from the active profile's program_spec.observation_program.
     *  Used by [OnDeviceLlmInsightExtractor] to build its OpenApiTool spec at runtime.
     *
     *  L9 single-path speech (2026-06-02): the LLM extractor must re-attempt
     *  initialization whenever fresh signals arrive — the eager init at
     *  service start fires BEFORE the server profile fetch completes (race),
     *  so without this retry the LLM stays dormant forever and every speech
     *  window emits empty SpeechInsights(confidence=0f). [OnDeviceLlmInsightExtractor.initialize]
     *  is idempotent (no-op when `loaded`), so re-firing on every profile
     *  apply is safe. */
    fun setActiveSignalsJson(signalsJson: String?) {
        val previous = activeSignalsJson
        activeSignalsJson = signalsJson
        // Trigger (re)init whenever the signals materially change. initialize() is
        // mutex-guarded and schema-aware: it no-ops when the speech schema is
        // unchanged (heartbeat re-applies of the same profile), initialises when the
        // engine isn't loaded yet (startup race), and RELOADS with the new tool spec
        // when the operator's declared speech fields change — so a freshly-deployed
        // profile takes effect live, without a device restart.
        val changed = previous != signalsJson
        val extractor = onDeviceLlmExtractor
        if (changed && signalsJson != null && extractor != null) {
            Log.i(TAG, "[Speech] activeSignalsJson changed — (re)initializing OnDeviceLlmInsightExtractor")
            sensingScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                try {
                    val initialized = extractor.initialize()
                    if (initialized) {
                        Log.i(TAG, "[Speech] OnDeviceLlmInsightExtractor ready after signals update")
                    } else {
                        Log.w(TAG, "[Speech] OnDeviceLlmInsightExtractor.initialize() returned false after signals update " +
                            "— extractor logs above explain why")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "[Speech] OnDeviceLlmInsightExtractor.initialize() threw after signals update", e)
                }
            }
        }
    }

    /** Preserve observation-program metadata for downstream payload compatibility. */
    fun setActiveProgramSpec(programSpecJson: String?, programSpecVersion: String?) {
        activeProgramSpecJson = programSpecJson
        activeProgramSpecVersion = programSpecVersion
        activeProgramRuntimeContract = parseObservationProgramRuntimeContract(programSpecJson)
        activeProgramRuntimeContract?.let { contract ->
            Log.i(
                TAG,
                "[Profile] Observation program set: objective=${contract.objective ?: "unknown"}, " +
                    "fusion=${contract.fusionMode ?: "default"}, workers=" +
                    "static=${contract.staticSemantics?.enabled}, temporal=${contract.temporalSemantics?.enabled}, " +
                    "speech=${contract.speechSemantics?.enabled}, physical=${contract.physicalCorroboration?.enabled}"
            )
        }

        // Phase 4 PR 2 — refresh active clip triggers from the new program's
        // metrics_schema. Reset the debounce map so a new profile gets a fresh
        // window (no stale fires from a prior profile carrying over).
        activeClipTriggerRules = extractClipTriggersFromProgramSpec(programSpecJson)
        clipTriggerEvaluator.reset()
        if (activeClipTriggerRules.isNotEmpty()) {
            Log.i(TAG, "[ClipTriggers] Active rules: ${activeClipTriggerRules.size} " +
                activeClipTriggerRules.joinToString(prefix = "[", postfix = "]") { rule ->
                    "${rule.triggerId}:${rule.condition.field}${rule.condition.op}${rule.condition.value}"
                })
        }
    }

    /**
     * Phase 4 PR 3 — fire the DRY-RUN clip capture pipeline for one fired
     * trigger. Fire-and-forget: the aggregation/emit path never blocks on
     * recording (which can take up to 5s for the default duration).
     *
     * Recording + uploading runs on the existing [sensingScope] with
     * `Dispatchers.IO`. The recorder's busy-flag dedupes overlap when two
     * triggers fire in the same cycle.
     */
    internal fun dispatchClipDryrunCapture(fired: ClipTriggerEvaluator.FiredTrigger) {
        // Lazy-init: a device with no triggers active never spins these up.
        // Keep both `clipRecorder` and `clipUploadManager` paired — if either
        // is missing we skip the capture but still preserve the trigger fire
        // log line for debugging.
        if (clipRecorder == null) {
            clipRecorder = ClipRecorder(context = context)
        }
        if (clipUploadManager == null) {
            clipUploadManager = ClipUploadManager(
                client = okhttp3.OkHttpClient(),
                socketManager = socketManager,
                apiBaseUrl = apiBaseUrl,
            )
        }

        val recorder = clipRecorder ?: return
        val uploader = clipUploadManager ?: return
        val triggerId = fired.rule.triggerId
        // Use the recorder's deterministic id format so the API row id is
        // recognizable as a clip dryrun observation. PR 4 will mirror the
        // server-side deterministicObservationId() once we wire Gemini.
        val observationId = recorder.generateObservationId(triggerId)
        val durationMs = fired.rule.captureDurationSeconds.toLong() * 1000L

        sensingScope.launch(Dispatchers.IO) {
            try {
                val recording = recorder.recordClip(
                    observationId = observationId,
                    requestedDurationMs = durationMs,
                ) ?: run {
                    Log.w(TAG, "[ClipDryrun] recording failed observation_id=$observationId trigger=$triggerId")
                    return@launch
                }
                val result = uploader.uploadAndEmit(
                    recording = recording,
                    fingerprint = fingerprint,
                    triggerId = triggerId,
                    observedValue = fired.observedValue,
                )
                if (result == null) {
                    Log.w(TAG, "[ClipDryrun] upload failed observation_id=$observationId trigger=$triggerId")
                } else {
                    Log.i(TAG, "[ClipDryrun] dryrun emitted observation_id=$observationId trigger=$triggerId " +
                        "s3_key=${result.s3Key} size=${result.sizeBytes}B duration=${result.durationMs}ms " +
                        "emitted=${result.emitted}")
                }
            } catch (e: Exception) {
                Log.w(TAG, "[ClipDryrun] capture-pipeline failure (non-fatal)", e)
            }
        }
    }

    /**
     * Extract `metrics_schema` from a program-spec JSON string and parse out
     * any `type='trigger'` entries. The metrics_schema lives at
     * `program_spec.output_contract.metrics_schema` (see programSpec.js).
     */
    private fun extractClipTriggersFromProgramSpec(programSpecJson: String?): List<ClipTriggerEvaluator.TriggerRule> {
        val parsed = parseObservationProgramJson(programSpecJson) ?: return emptyList()
        val outputContract = parsed.optJSONObject("output_contract") ?: return emptyList()
        val metricsSchema = outputContract.optJSONArray("metrics_schema") ?: return emptyList()
        return parseClipTriggersFromMetricsSchema(metricsSchema)
    }

    private fun resolveEffectiveModels(
        appliedModels: List<String>,
        runtimeContract: ObservationProgramRuntimeContract?
    ): List<String> {
        val runtimeModels = runtimeContract?.enabledWorkerModelIds().orEmpty()
        return if (runtimeContract?.hasExplicitWorkerModels() == true && runtimeModels.isNotEmpty()) {
            runtimeModels
        } else {
            appliedModels
        }
    }

    /**
     * Reconfigure the inference graph based on an applied sensing profile.
     * Enables/disables processors based on which models are in the applied profile.
     *
     * Model ID to processor mapping:
     * - blazeface -> AudienceAnalyzer (ML Kit face detection, always-on core)
     * - age_gender -> AgeGenderProcessor (on-device demographics)
     * - fer_plus -> FaceLandmarkerProcessor (MediaPipe blendshape-derived emotion;
     *                wire-format model_id retained for API compatibility)
     * - movenet -> PoseEngagementProcessor (MediaPipe Pose)
     * - yamnet -> AudioClassificationProcessor (ambient audio)
     * - whisper_tiny -> SpeechIntelligenceProcessor (Moonshine ASR)
     * - efficientdet -> PersonDetectionProcessor (TFLite person detection)
     *                   + ObjectDetectionProcessor → VehicleFlowProcessor + QueueEstimationProcessor
     *                   (vehicle_count, pedestrian_count — Phase 4 wire-up)
     *
     * This method is safe to call multiple times; it updates state in-place.
     */
    fun applyProfileToInference(appliedModels: List<String>) {
        activeProfileModels = appliedModels
        val runtimeContract = activeProgramRuntimeContract
        val effectiveModels = resolveEffectiveModels(appliedModels, runtimeContract)
        val temporalWorker = runtimeContract?.temporalSemantics
        val speechWorker = runtimeContract?.speechSemantics
        val physicalWorker = runtimeContract?.physicalCorroboration
        Log.i(
            TAG,
            "[Profile] Applying inference profile with models: $appliedModels, effective=$effectiveModels, " +
                "objective=${runtimeContract?.objective ?: "legacy_compatibility"}"
        )

        // Face detection (blazeface) -- controlled by whether analyzer exists
        // blazeface is the core pipeline; if it's absent, we don't tear down the analyzer
        // since it manages the camera lifecycle. Other vision models depend on it.

        // Ambient audio is treated as a base signal whenever microphone capture
        // is available. Profiles can choose whether to emphasize or ignore it,
        // but rotating profiles should not churn the microphone pipeline.
        val wantsAudio = physicalWorker?.enabled ?: ("yamnet" in effectiveModels)
        val canRunAmbientAudio =
            sensingConfig.enableAudioClassification && hasMicrophoneHardware && hasMicrophonePermission
        if (runtimeContract != null && physicalWorker != null && !physicalWorker.enabled) {
            if (audioProcessor != null) {
                Log.i(TAG, "[Profile] Disabling ambient audio classification via observation-program contract")
                audioProcessor?.stop()
                audioProcessor = null
            }
        } else if (audioProcessor == null && canRunAmbientAudio) {
            Log.i(
                TAG,
                if (wantsAudio) {
                    "[Profile] Enabling ambient audio classification (yamnet in profile)"
                } else {
                    "[Profile] Retaining ambient audio classification as base telemetry"
                }
            )
            try {
                startAudioClassification(sensingConfig)
            } catch (e: Exception) {
                Log.e(TAG, "[Profile] Failed to initialize ambient audio classification: ${e.message}", e)
            }
        } else if (!wantsAudio && audioProcessor != null) {
            Log.i(TAG, "[Profile] Keeping ambient audio classification active for continuous sensing")
        }

        // Speech intelligence remains part of continuous microphone telemetry.
        val speechDecision = if (runtimeContract != null && speechWorker != null && !speechWorker.enabled) {
            SpeechPipelinePolicy.Decision.STOP
        } else {
            SpeechPipelinePolicy.decide(
                appliedModels = speechWorker?.modelIds?.takeIf { it.isNotEmpty() } ?: effectiveModels,
                hasMicrophoneHardware = hasMicrophoneHardware,
                hasMicrophonePermission = hasMicrophonePermission
            )
        }

        when (speechDecision) {
            SpeechPipelinePolicy.Decision.STOP -> {
                if (speechProcessor != null) {
                    Log.i(
                        TAG,
                        if (runtimeContract != null && speechWorker != null && !speechWorker.enabled) {
                            "[Profile] Disabling speech intelligence via observation-program contract"
                        } else {
                            "[Profile] Disabling speech intelligence (microphone unavailable)"
                        }
                    )
                    speechProcessor?.stop()
                    speechProcessor = null
                }
            }
            SpeechPipelinePolicy.Decision.KEEP_OR_START -> {
                if (speechProcessor == null) {
                    // Honor the same RAM guard used in start() — devices with <= 2GB RAM
                    // must not spin up speech processor on profile reapply
                    val memInfo = ActivityManager.MemoryInfo()
                    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memInfo)
                    val totalRamGb = memInfo.totalMem / (1024.0 * 1024 * 1024)
                    if (totalRamGb <= 2.0) {
                        Log.w(TAG, "[Profile] Skipping speech processor restart - device has ${String.format("%.1f", totalRamGb)}GB RAM (minimum 2GB required)")
                    } else {
                        Log.i(TAG, "[Profile] Keeping speech intelligence active as base telemetry")
                        try {
                            startSpeechIntelligence(sensingConfig)
                        } catch (e: Exception) {
                            Log.e(TAG, "[Profile] Failed to restart speech intelligence: ${e.message}", e)
                        }
                    }
                } else {
                    Log.i(TAG, "[Profile] Keeping speech intelligence active for continuous sensing")
                }
            }
        }

        // Emotional engagement processors are managed by AudienceAnalyzer internally.
        // Log the model disposition for fleet visibility.
        val enableCameraPresenceBaseline = runtimeContract?.let {
            (it.temporalSemantics?.enabled == true) || (it.physicalCorroboration?.enabled == true)
        } ?: true
        val selection = VisionProcessorSelection.fromModels(
            appliedModels = effectiveModels,
            enableCameraPresenceBaseline = enableCameraPresenceBaseline
        )
        audienceAnalyzer?.applyVisionProcessorSelection(selection)
        Log.i(TAG, "[Profile] Vision sub-models active: ${selection.activeModelIds()}")

        // VLM model activation/deactivation
        val vlmModelIds = setOf("gemma_4_e2b", "moondream_05b", "smolvlm_256m")
        val wantedVlmId = when {
            temporalWorker != null && !temporalWorker.enabled -> null
            temporalWorker?.modelIds?.isNotEmpty() == true -> temporalWorker.modelIds.firstOrNull { it in vlmModelIds }
            else -> effectiveModels.firstOrNull { it in vlmModelIds }
        }

        if (wantedVlmId == null) {
            activeVlmRuntimeTelemetry = VlmRuntimeTelemetry(
                state = VlmRuntimeTelemetry.STATE_DISABLED,
                reason = if (temporalWorker != null && !temporalWorker.enabled) {
                    "temporal_worker_disabled"
                } else {
                    "profile_has_no_vlm"
                },
                promptChars = activeVlmPrompt?.length
            )
        }

        if (wantedVlmId == null && vlmProcessor != null) {
            Log.i(TAG, "[Profile] Temporal worker disabled or absent - unloading active VLM")
            setVlmProcessor(null)
        }

        if (wantedVlmId != null && (vlmProcessor == null || vlmProcessor?.modelId != wantedVlmId)) {
            // Tear down existing VLM if switching models
            vlmProcessor?.let {
                Log.i(TAG, "[Profile] Unloading previous VLM: ${it.modelId}")
                setVlmProcessor(null)
            }

            // VLM format is detected from the cached model file extension, or from the OTA manifest
            // during download. The server selects the best format variant per device GPU/NPU.
            // LiteRT-LM is the sole VLM runtime (GGUF/llama.cpp removed).
            var modelFormat = "litert-lm" // default, overridden by file detection or manifest

            // Get model path from ModelDownloadManager (OTA-downloaded models)
            val downloadManager = com.trillboards.ctv.core.ml.ModelDownloadManager(context!!)
            val modelPath = downloadManager.getModelPath(wantedVlmId)
            val memoryInfo = readSystemMemoryInfo()
            val manifest = getHardwareManifestOrNull()
            val activationDecision = VlmActivationPolicy.decide(
                VlmActivationInput(
                    modelId = wantedVlmId,
                    totalRamMb = memoryInfo.totalMb,
                    availableRamMb = memoryInfo.availableMb,
                    nativeHeapMb = memoryInfo.nativeHeapMb,
                    lowMemory = memoryInfo.lowMemory,
                    pressure = memoryInfo.pressure,
                    attenuationTier = attenuationManager?.getCurrentTier()
                        ?: MemoryAttenuationManager.AttenuationTier.NORMAL,
                    maxVlmSizeMb = manifest?.maxVlmSizeMb ?: 0,
                    hasForegroundUi = interactiveUiProvider?.invoke() ?: true,
                    modelFileSizeMb = modelPath?.length()?.div(1024 * 1024)?.toInt()
                )
            )

            if (!activationDecision.allowed) {
                activeVlmRuntimeTelemetry = VlmRuntimeTelemetry(
                    requestedModelId = wantedVlmId,
                    activeModelId = null,
                    state = VlmRuntimeTelemetry.STATE_BLOCKED,
                    reason = activationDecision.reason,
                    requiredAvailableRamMb = activationDecision.requiredAvailableRamMb,
                    estimatedRuntimeFootprintMb = activationDecision.estimatedRuntimeFootprintMb,
                    availableRamMb = memoryInfo.availableMb,
                    nativeHeapMb = memoryInfo.nativeHeapMb,
                    promptChars = activeVlmPrompt?.length
                )
                Log.w(
                    TAG,
                    "[Profile] Skipping VLM model $wantedVlmId: ${activationDecision.reason} " +
                        "(available=${memoryInfo.availableMb}MB, nativeHeap=${memoryInfo.nativeHeapMb}MB, " +
                        "pressure=${memoryInfo.pressure}, ui=${interactiveUiProvider?.invoke() ?: true}, " +
                        "required=${activationDecision.requiredAvailableRamMb}MB)"
                )
                if (vlmProcessor != null) setVlmProcessor(null)
                Log.i(TAG, "[Profile] Inference graph reconfigured successfully")
                return
            }

            // Detect format from the actual model file on disk (model.litertlm, model.onnx, etc.)
            // getModelPath() returns the model File itself (not directory)
            if (modelPath != null) {
                modelFormat = when {
                    modelPath.name.endsWith(".litertlm") -> "litert-lm"
                    modelPath.name.endsWith(".onnx") -> "onnx-vlm"
                    modelPath.name.endsWith(".pte") -> "executorch"
                    else -> "litert-lm"
                }
                Log.i(TAG, "[Profile] Detected VLM format from cached file: ${modelPath.name} → $modelFormat")
            }

            if (modelPath != null) {
                val metricsPrompt = activeVlmPrompt
                    ?: runtimeContract?.temporalPromptFallback()
                    ?: "Analyze this image. Return JSON with: person_count, scene_description, activity_description, object_list"
                activeVlmRuntimeTelemetry = VlmRuntimeTelemetry(
                    requestedModelId = wantedVlmId,
                    activeModelId = null,
                    state = VlmRuntimeTelemetry.STATE_INITIALIZING,
                    reason = activationDecision.reason,
                    requiredAvailableRamMb = activationDecision.requiredAvailableRamMb,
                    estimatedRuntimeFootprintMb = activationDecision.estimatedRuntimeFootprintMb,
                    availableRamMb = memoryInfo.availableMb,
                    nativeHeapMb = memoryInfo.nativeHeapMb,
                    promptChars = metricsPrompt.length
                )

                try {
                    // Use CPU backend for VLM to avoid GPU memory doubling (Mali GPU
                    // mirrors model weights into process RSS, causing OOM on 8-12GB devices).
                    // CPU inference is ~20% slower but uses ~50% less memory.
                    val vlmConfig = com.trillboards.ctv.core.inference.vlm.VLMConfig(
                        maxTokens = 512,
                        temperature = 0.1f,
                        useGpu = false,
                        numThreads = 4
                    )
                    val processor = VLMInferenceProcessor(
                        context = context,
                        modelId = wantedVlmId,
                        metricsPrompt = metricsPrompt,
                        modelPath = modelPath.absolutePath,
                        modelFormat = modelFormat,
                        vlmConfig = vlmConfig
                    )

                    // Wire temporal context from the TFLite snapshot ring buffer.
                    // Each aggregation cycle pushes a snapshot; the VLM prompt
                    // receives the last ~30s of sensor history so it can reason
                    // about crowd trends (building, peak, winding_down, idle).
                    processor.contextProvider = {
                        val snapshots = synchronized(tfliteSnapshots) { tfliteSnapshots.toList() }
                        if (snapshots.isEmpty()) ""
                        else {
                            val now = System.currentTimeMillis()
                            val lines = snapshots.reversed().map { s ->
                                val agoSec = (now - s.timestampMs) / 1000
                                "[${agoSec}s ago] faces=${s.faceCount.toInt()}, attn=${String.format("%.2f", s.avgAttention)}, mood=${s.dominantEmotion}, noise=${s.dominantAmbience}"
                            }
                            "Recent sensor readings from this screen:\n${lines.joinToString("\n")}\n"
                        }
                    }

                    if (processor.initialize()) {
                        setVlmProcessor(processor)
                        activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                            activeModelId = wantedVlmId,
                            state = VlmRuntimeTelemetry.STATE_ACTIVE,
                            reason = activationDecision.reason
                        )
                        Log.i(TAG, "[Profile] VLM processor activated: $wantedVlmId ($modelFormat)")
                    } else {
                        activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                            activeModelId = null,
                            state = VlmRuntimeTelemetry.STATE_INITIALIZE_FAILED,
                            reason = "processor_initialize_returned_false"
                        )
                        Log.e(TAG, "[Profile] VLM processor failed to initialize: $wantedVlmId")
                        processor.release()
                    }
                } catch (e: Exception) {
                    activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                        activeModelId = null,
                        state = VlmRuntimeTelemetry.STATE_INITIALIZE_FAILED,
                        reason = e.javaClass.simpleName
                    )
                    Log.e(TAG, "[Profile] Failed to create VLM processor: ${e.message}", e)
                }
            } else {
                val shouldStartDownload = synchronized(vlmDownloadLock) {
                    if (vlmDownloadInFlightModelId == wantedVlmId) {
                        false
                    } else {
                        vlmDownloadInFlightModelId = wantedVlmId
                        true
                    }
                }
                if (!shouldStartDownload) {
                    activeVlmRuntimeTelemetry = VlmRuntimeTelemetry(
                        requestedModelId = wantedVlmId,
                        activeModelId = null,
                        state = VlmRuntimeTelemetry.STATE_DOWNLOAD_IN_FLIGHT,
                        reason = "model_missing",
                        requiredAvailableRamMb = activationDecision.requiredAvailableRamMb,
                        estimatedRuntimeFootprintMb = activationDecision.estimatedRuntimeFootprintMb,
                        availableRamMb = memoryInfo.availableMb,
                        nativeHeapMb = memoryInfo.nativeHeapMb,
                        promptChars = activeVlmPrompt?.length
                    )
                    Log.i(TAG, "[Profile] VLM download already in flight for $wantedVlmId -- skipping duplicate trigger")
                    Log.i(TAG, "[Profile] Inference graph reconfigured successfully")
                    return
                }

                activeVlmRuntimeTelemetry = VlmRuntimeTelemetry(
                    requestedModelId = wantedVlmId,
                    activeModelId = null,
                    state = VlmRuntimeTelemetry.STATE_DOWNLOADING,
                    reason = "model_missing",
                    requiredAvailableRamMb = activationDecision.requiredAvailableRamMb,
                    estimatedRuntimeFootprintMb = activationDecision.estimatedRuntimeFootprintMb,
                    availableRamMb = memoryInfo.availableMb,
                    nativeHeapMb = memoryInfo.nativeHeapMb,
                    promptChars = activeVlmPrompt?.length
                )
                Log.w(TAG, "[Profile] VLM model $wantedVlmId not on device — fetching manifest and downloading")
                // Fetch model manifest from API, then download the VLM binary via OTA.
                // On completion, re-apply the profile to activate the VLM processor.
                kotlinx.coroutines.GlobalScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                    try {
                        val downloadManager = com.trillboards.ctv.core.ml.ModelDownloadManager(context!!)
                        val manifestParams = StringBuilder("device_tier=standard")
                        screenId?.let { manifestParams.append("&screen_id=$it") }
                        if (fingerprint.isNotEmpty()) manifestParams.append("&fingerprint=$fingerprint")
                        val manifestUrl = "${apiBaseUrl}/v2/earner/ml/model-manifest?$manifestParams"
                        Log.i(TAG, "[Profile] Fetching model manifest: $manifestUrl")

                        // Fetch model manifest from API
                        val request = okhttp3.Request.Builder().url(manifestUrl).build()
                        val client = okhttp3.OkHttpClient.Builder()
                            .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                            .readTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                            .build()
                        val response = client.newCall(request).execute()
                        val body = response.body?.string()
                        response.close()

                        if (body == null || !response.isSuccessful) {
                            activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                                state = VlmRuntimeTelemetry.STATE_MANIFEST_FETCH_FAILED,
                                reason = "http_${response.code}"
                            )
                            Log.e(TAG, "[Profile] Failed to fetch model manifest: HTTP ${response.code}")
                            return@launch
                        }

                        val json = org.json.JSONObject(body)
                        val models = json.optJSONObject("models")
                        if (models == null) {
                            activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                                state = VlmRuntimeTelemetry.STATE_MANIFEST_FETCH_FAILED,
                                reason = "manifest_missing_models"
                            )
                            Log.e(TAG, "[Profile] Model manifest has no 'models' key")
                            return@launch
                        }

                        // Find the VLM model entry that matches wantedVlmId
                        var downloadUrl: String? = null
                        var checksumSha256 = ""
                        var sizeBytes = 0L
                        val keys = models.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            val entry = models.getJSONObject(key)
                            if (entry.optString("model") == wantedVlmId) {
                                downloadUrl = entry.optString("url", "")
                                checksumSha256 = if (entry.isNull("hash")) "" else entry.optString("hash", "")
                                sizeBytes = entry.optLong("size_bytes", 0)
                                // Read format from manifest — server selects best variant per device
                                val manifestFormat = entry.optString("format", "")
                                if (manifestFormat.isNotEmpty()) modelFormat = manifestFormat
                                break
                            }
                        }

                        if (downloadUrl.isNullOrEmpty()) {
                            activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                                state = VlmRuntimeTelemetry.STATE_MANIFEST_MISSING_MODEL,
                                reason = "model_not_in_manifest"
                            )
                            Log.w(TAG, "[Profile] VLM model $wantedVlmId not found in manifest. " +
                                "Available: ${models.keys().asSequence().toList()}")
                            return@launch
                        }

                        // Parse companion files (auxiliary files for multimodal vision)
                        val companionFiles = mutableListOf<com.trillboards.ctv.core.ml.ModelDownloadManager.CompanionFile>()
                        // Re-find the entry to get companion_files
                        val allKeys2 = models.keys()
                        while (allKeys2.hasNext()) {
                            val k2 = allKeys2.next()
                            val e2 = models.getJSONObject(k2)
                            if (e2.optString("model") == wantedVlmId) {
                                val cfArray = e2.optJSONArray("companion_files")
                                if (cfArray != null) {
                                    for (i in 0 until cfArray.length()) {
                                        val cf = cfArray.getJSONObject(i)
                                        companionFiles.add(com.trillboards.ctv.core.ml.ModelDownloadManager.CompanionFile(
                                            filename = cf.getString("filename"),
                                            downloadUrl = cf.getString("url"),
                                            checksumSha256 = if (cf.isNull("hash")) "" else cf.optString("hash", ""),
                                            sizeBytes = cf.optLong("size_bytes", 0)
                                        ))
                                    }
                                }
                                break
                            }
                        }

                        Log.i(TAG, "[Profile] Found VLM model $wantedVlmId in manifest: " +
                            "url=$downloadUrl, size=${sizeBytes / (1024 * 1024)}MB, format=$modelFormat, " +
                            "companionFiles=${companionFiles.size}")

                        val manifest = com.trillboards.ctv.core.ml.ModelDownloadManager.ModelManifest(
                            modelId = wantedVlmId,
                            downloadUrl = downloadUrl,
                            checksumSha256 = checksumSha256,
                            sizeBytes = sizeBytes,
                            modelFormat = modelFormat,
                            companionFiles = companionFiles
                        )

                        // Download the model (resumable, checksum-validated)
                        downloadManager.downloadModel(manifest).collect { progress ->
                            when (progress.state) {
                                com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.DOWNLOADING -> {
                                    if (progress.percentComplete.toInt() % 10 == 0) {
                                        Log.i(TAG, "[Profile] VLM download: ${progress.percentComplete.toInt()}% " +
                                            "(${progress.bytesDownloaded / (1024 * 1024)}MB / ${progress.totalBytes / (1024 * 1024)}MB)")
                                    }
                                }
                                com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.COMPLETE -> {
                                    synchronized(vlmDownloadLock) {
                                        if (vlmDownloadInFlightModelId == wantedVlmId) {
                                            vlmDownloadInFlightModelId = null
                                        }
                                    }
                                    Log.i(TAG, "[Profile] VLM model $wantedVlmId download complete! " +
                                        "Activating processor immediately.")
                                    applyProfileToInference(activeProfileModels)
                                }
                                com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.FAILED -> {
                                    synchronized(vlmDownloadLock) {
                                        if (vlmDownloadInFlightModelId == wantedVlmId) {
                                            vlmDownloadInFlightModelId = null
                                        }
                                    }
                                    activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                                        state = VlmRuntimeTelemetry.STATE_DOWNLOAD_FAILED,
                                        reason = "download_failed"
                                    )
                                    Log.e(TAG, "[Profile] VLM model download FAILED for $wantedVlmId")
                                }
                                else -> {}
                            }
                        }
                    } catch (e: Exception) {
                        synchronized(vlmDownloadLock) {
                            if (vlmDownloadInFlightModelId == wantedVlmId) {
                                vlmDownloadInFlightModelId = null
                            }
                        }
                        activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                            state = VlmRuntimeTelemetry.STATE_DOWNLOAD_FAILED,
                            reason = e.javaClass.simpleName
                        )
                        Log.e(TAG, "[Profile] VLM OTA download failed: ${e.message}", e)
                    }
                }
            }
        } else if (wantedVlmId != null && vlmProcessor?.modelId == wantedVlmId) {
            activeVlmRuntimeTelemetry = activeVlmRuntimeTelemetry.copy(
                requestedModelId = wantedVlmId,
                activeModelId = wantedVlmId,
                state = VlmRuntimeTelemetry.STATE_ACTIVE,
                reason = "ok",
                promptChars = activeVlmPrompt?.length
            )
        } else if (wantedVlmId == null && vlmProcessor != null) {
            Log.i(TAG, "[Profile] Profile has no VLM model — clearing VLM processor")
            setVlmProcessor(null)
        }

        // Phase 2 — age/gender foundation-model OTA download.
        //
        // When the active profile includes `age_gender`, check if FaceXFormer
        // is already loaded. If not, query the model-manifest endpoint for
        // a presigned S3 URL to the FP16 artifact and download via
        // ModelDownloadManager. On COMPLETE, restart AgeGenderProcessor so
        // the next heartbeat emits real demographics. Until the download
        // completes, on_device_age_distribution / on_device_gender_split
        // remain NULL — honest null over fabricated data.
        if (effectiveModels.contains("age_gender")) {
            triggerAgeGenderOtaDownloadIfNeeded()
        }

        Log.i(TAG, "[Profile] Inference graph reconfigured successfully")
    }

    /**
     * Trigger an OTA download for the age/gender foundation model when:
     *   1. The active profile asks for `age_gender`
     *   2. AgeGenderProcessor.hasModel() returns false (asset+OTA both empty)
     *   3. No download is already in flight for this model
     *   4. The model-manifest API returns a non-empty download URL
     *
     * Once the download completes, calls audienceAnalyzer.startAgeGenderProcessor()
     * which re-attempts the load (now finding the OTA file) and flips the
     * processor active. From the next heartbeat onward, real FaceXFormer
     * demographics flow into onDeviceDemographics. Until the download
     * completes, the heartbeat emits no onDeviceDemographics block (no
     * fabricated heuristic data).
     */
    private fun triggerAgeGenderOtaDownloadIfNeeded() {
        val ctx = context ?: return
        val analyzer = audienceAnalyzer ?: return

        // If a processor is already initialized OR a candidate would init
        // (meaning hasModel() is true), no download needed.
        if (analyzer.getAgeGenderProcessor() != null) return

        // Probe hasModel() via a fresh candidate — its check covers both
        // assets and OTA paths in AgeGenderProcessor.findOtaModelFile().
        val probe = AgeGenderProcessor(ctx)
        if (probe.hasModel()) {
            // Model is on disk but processor wasn't started — kick it.
            analyzer.startAgeGenderProcessor()
            return
        }

        val shouldStart = synchronized(ageGenderDownloadLock) {
            if (ageGenderDownloadInFlight) {
                Log.d(TAG, "[Profile] Age/gender OTA download already in flight — skip")
                false
            } else {
                ageGenderDownloadInFlight = true
                true
            }
        }
        if (!shouldStart) return

        Log.i(TAG, "[Profile] Age/gender model missing — fetching manifest for OTA download")

        kotlinx.coroutines.GlobalScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            try {
                val downloadManager = com.trillboards.ctv.core.ml.ModelDownloadManager(ctx)
                val params = StringBuilder("device_tier=standard")
                screenId?.let { params.append("&screen_id=$it") }
                if (fingerprint.isNotEmpty()) params.append("&fingerprint=$fingerprint")
                val manifestUrl = "${apiBaseUrl}/v2/earner/ml/model-manifest?$params"

                val request = okhttp3.Request.Builder().url(manifestUrl).build()
                val client = okhttp3.OkHttpClient.Builder()
                    .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                val response = client.newCall(request).execute()
                val body = response.body?.string()
                response.close()

                if (body == null || !response.isSuccessful) {
                    Log.w(TAG, "[Profile] Age/gender manifest fetch failed: HTTP ${response.code}")
                    return@launch
                }

                val json = org.json.JSONObject(body)
                val models = json.optJSONObject("models")
                    ?: run {
                        Log.w(TAG, "[Profile] Age/gender manifest missing 'models' key")
                        return@launch
                    }

                // Look for either the `age_gender` or `face_foundation`
                // registry ID in the manifest. The server's otaManifestService
                // keys by the registry model_id, so this matches whatever
                // Phase 2 PR 2's migration registers.
                val candidates = listOf("age_gender", "face_foundation")
                var entry: org.json.JSONObject? = null
                var entryId: String? = null
                for (id in candidates) {
                    if (models.has(id)) {
                        entry = models.getJSONObject(id)
                        entryId = id
                        break
                    }
                }

                if (entry == null) {
                    // No artifact registered yet — heartbeat will emit no
                    // onDeviceDemographics block until the registry row gets
                    // a download_url + checksum_sha256 set.
                    Log.i(TAG, "[Profile] Age/gender manifest absent — " +
                        "onDeviceDemographics remains null (register face_foundation " +
                        "in sensing_model_registry to deliver the model)")
                    return@launch
                }

                val downloadUrl = entry.optString("url", "")
                if (downloadUrl.isEmpty()) {
                    Log.w(TAG, "[Profile] Age/gender manifest entry has empty url")
                    return@launch
                }
                val checksum = if (entry.isNull("hash")) "" else entry.optString("hash", "")
                val sizeBytes = entry.optLong("size_bytes", 0)
                val format = entry.optString("format", "tflite")

                Log.i(TAG, "[Profile] Age/gender OTA download starting: id=$entryId, " +
                    "url=$downloadUrl, size=${sizeBytes / (1024 * 1024)}MB, format=$format")

                val manifest = com.trillboards.ctv.core.ml.ModelDownloadManager.ModelManifest(
                    modelId = entryId!!,
                    downloadUrl = downloadUrl,
                    checksumSha256 = checksum,
                    sizeBytes = sizeBytes,
                    modelFormat = format
                )

                downloadManager.downloadModel(manifest).collect { progress ->
                    when (progress.state) {
                        com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.DOWNLOADING -> {
                            if (progress.percentComplete.toInt() % 25 == 0) {
                                Log.i(TAG, "[Profile] Age/gender download: ${progress.percentComplete.toInt()}%")
                            }
                        }
                        com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.COMPLETE -> {
                            Log.i(TAG, "[Profile] Age/gender model download complete — activating processor")
                            audienceAnalyzer?.startAgeGenderProcessor()
                        }
                        com.trillboards.ctv.core.ml.ModelDownloadManager.DownloadState.FAILED -> {
                            Log.e(TAG, "[Profile] Age/gender model download FAILED — " +
                                "onDeviceDemographics remains null (will retry on next profile apply)")
                        }
                        else -> { /* ignore other states */ }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "[Profile] Age/gender OTA download error: ${e.message}", e)
            } finally {
                synchronized(ageGenderDownloadLock) {
                    ageGenderDownloadInFlight = false
                }
            }
        }
    }

    /**
     * Set the active profile ID (called from profile manager after successful apply).
     */
    fun setActiveProfileId(profileId: String?) {
        activeProfileId = profileId
    }

    /**
     * Get the active sensing profile ID for emission in audienceSignals payload.
     */
    fun getActiveProfileId(): String? = activeProfileId

    /**
     * Update the dynamic brand list for speech intelligence.
     * Called when the server sends active campaign brands for this screen.
     * Brand awareness is now operator-declared via profile signals; this is a no-op
     * retained for call-site compatibility until callers are removed.
     */
    fun updateBrandList(brands: List<String>) {
        Log.i(TAG, "Dynamic brand list received (${brands.size} brands) — " +
            "brand awareness is profile-driven; no-op on device side")
    }

    /**
     * Get the latest demographics result from Gemini Vision.
     */
    fun getLatestDemographics(): DemographicsResult? = latestDemographics

    /**
     * Get current face count.
     */
    fun getCurrentFaceCount(): Int {
        return audienceAnalyzer?.getCurrentViewerCount() ?: 0
    }

    /**
     * Get current attention score (0.0 to 1.0).
     */
    fun getCurrentAttentionScore(): Float {
        return audienceAnalyzer?.getCurrentAttentionScore() ?: 0f
    }

    /**
     * Check if sensing is currently active.
     */
    fun isSensing(): Boolean = isRunning

    /**
     * Flush any buffered audience signals that accumulated while the socket was disconnected.
     * Called by DeviceAgentService when the socket reconnects to drain the buffer immediately
     * rather than waiting for the next aggregation cycle.
     */
    fun flushBufferedSignals() {
        sensingScope.launch(Dispatchers.IO) {
            try {
                val count = signalBuffer?.count() ?: 0
                if (count == 0) return@launch
                Log.i(TAG, "[Buffer] Socket reconnected — flushing $count buffered signals")
                val flushed = signalBuffer?.flushBuffer(socketManager, 10) ?: 0
                Log.i(TAG, "[Buffer] Reconnect flush complete: $flushed/$count signals sent")
            } catch (e: Exception) {
                Log.w(TAG, "[Buffer] Reconnect flush failed: ${e.message}")
            }
        }
    }

    /**
     * Get audience sensing capabilities for heartbeat.
     * Returns camera status, microphone status, and current sensing mode.
     *
     * Camera health snapshot (2026-05-03, fix-ctv-camera-never-started-
     * product-fix): includes the [CameraHealthMonitor] state + retry count
     * + last failure class so the server's per-screen camera-health gauge
     * lights up when devices like Adam @ Focus Media's RockChip never
     * actually start the camera. Default-empty when the analyzer hasn't
     * been instantiated yet (FACE_ONLY mode but service still pending boot).
     */
    fun getAudienceSensingCapabilities(): com.trillboards.ctv.core.models.DeviceCapabilityPayload.AudienceSensing {
        val healthSnapshot = audienceAnalyzer?.cameraHealth?.snapshot()
        val cameraHealth = if (healthSnapshot != null) {
            com.trillboards.ctv.core.models.DeviceCapabilityPayload.CameraHealth(
                state = healthSnapshot.state.name.lowercase(),
                lastFrameAtMs = healthSnapshot.lastFrameAtMs,
                retryCount = healthSnapshot.retryCount,
                lastFailureClass = healthSnapshot.lastFailureClass?.name?.lowercase(),
                lastFailureMessage = healthSnapshot.lastFailureMessage
            )
        } else {
            // Analyzer not yet instantiated — surface "never_started" so the
            // server doesn't see a hole. cap_face_detection devolution
            // intentionally does NOT trigger from this state; only after a
            // real start attempt has been made and FAILED for ≥ 24h.
            com.trillboards.ctv.core.models.DeviceCapabilityPayload.CameraHealth()
        }

        // Self-correcting cap: when the agent claims face-detection capability
        // but the camera has been FAILED for >24h, devolve so cap_face_detection
        // matches reality. Reverts automatically when the camera recovers
        // (state→OPEN flips shouldDevolveFaceCap back to false).
        val faceCapHonest = !shouldDevolveFaceCap()

        // Belt-and-suspenders for server-side cap_face_detection / cap_audio_classification
        // derivation (peppy-cooking-blum PR 1). Mirror the server's logic so a single
        // boolean read replaces the cameraAvailable+sensingMode tuple. faceDetection
        // honors the `faceCapHonest` devolution so a camera FAILED >24h stops claiming
        // the capability — matches what the existing `cameraAvailable` line does.
        val effectiveCameraAvailable = cameraStatus.available && faceCapHonest
        val effectiveMicAvailable = hasMicrophoneHardware && hasMicrophonePermission
        val sensingModeName = _currentState.value.mode.name
        val faceDetection = effectiveCameraAvailable && (sensingModeName == SensingMode.FACE_ONLY.name || sensingModeName == SensingMode.FULL.name)
        val audioClassification = effectiveMicAvailable && (sensingModeName == SensingMode.AUDIO_ONLY.name || sensingModeName == SensingMode.FULL.name)

        return com.trillboards.ctv.core.models.DeviceCapabilityPayload.AudienceSensing(
            cameraAvailable = effectiveCameraAvailable,
            cameraType = cameraStatus.type,
            cameraCount = cameraStatus.count,
            microphoneAvailable = effectiveMicAvailable,
            sensingMode = sensingModeName,
            cameraHealth = cameraHealth,
            faceDetection = faceDetection,
            audioClassification = audioClassification
        )
    }

    /**
     * Whether `cap_face_detection` should devolve to false on this heartbeat.
     * Mirrors `CameraHealthMonitor.shouldDevolveFaceCap()` so external
     * callers (DeviceAgentService capability builder) can also query.
     */
    fun shouldDevolveFaceCap(): Boolean = audienceAnalyzer?.cameraHealth?.shouldDevolveFaceCap() == true

    /**
     * Get ML hardware capabilities for heartbeat.
     * Lazily detects HardwareManifest on first call and caches it —
     * hardware doesn't change at runtime.
     */
    fun getMLCapabilities(): com.trillboards.ctv.core.models.DeviceCapabilityPayload.MLCapabilities {
        val manifest = getHardwareManifestOrNull() ?: run {
            Log.w(TAG, "HardwareManifest.detect() unavailable, returning defaults")
            return com.trillboards.ctv.core.models.DeviceCapabilityPayload.MLCapabilities()
        }

        return com.trillboards.ctv.core.models.DeviceCapabilityPayload.MLCapabilities(
            chipsetVendor = manifest.chipsetVendor.name.lowercase(),
            chipsetName = manifest.chipsetName,
            totalRamMb = manifest.totalRamMb,
            availableRamMb = manifest.availableRamMb,
            gpuName = manifest.gpuName,
            hasNpu = manifest.hasNpu,
            npuName = manifest.npuName,
            gpuDelegateSupported = manifest.gpuDelegateSupported,
            nnapiSupported = manifest.nnapiSupported,
            recommendedModelTier = manifest.recommendedModelTier.name,
            maxVlmSizeMb = manifest.maxVlmSizeMb,
            // Phase 2 prereq PR 2 — surface ABI / OS API / display geometry
            // from HardwareManifest into the wire payload so server-side
            // earner_screen_devices.{cpu_abi, os_api_level, screen_width_px,
            // screen_height_px, density_dpi} stop being NULL on full-agent
            // heartbeats.
            cpuAbi = manifest.cpuAbi,
            osApiLevel = manifest.osApiLevel,
            screenWidthPx = manifest.screenWidthPx,
            screenHeightPx = manifest.screenHeightPx,
            densityDpi = manifest.densityDpi
        )
    }

    private fun getHardwareManifestOrNull(): HardwareManifest? {
        return cachedHardwareManifest ?: try {
            HardwareManifest.detect(context).also { cachedHardwareManifest = it }
        } catch (e: Exception) {
            Log.w(TAG, "HardwareManifest.detect() failed", e)
            null
        }
    }

    /**
     * Telemetry getters for heartbeat metadata.
     */
    fun getSensingMode(): String = if (isRunning) _currentState.value.mode.name ?: "UNKNOWN" else "OFF"
    fun getSensingUptime(): Long = if (isRunning && startTimeMs > 0) System.currentTimeMillis() - startTimeMs else 0L
    fun getTrackedFaceCount(): Int = audienceAnalyzer?.getCurrentViewerCount() ?: 0

    /**
     * Get current attenuation tier.
     */
    fun getAttenuationTier(): MemoryAttenuationManager.AttenuationTier? = attenuationManager?.getCurrentTier()

    /**
     * Attenuate to a specific tier to reduce memory pressure.
     */
    fun attenuateToTier(tier: MemoryAttenuationManager.AttenuationTier) {
        val manager = attenuationManager ?: run {
            Log.w(TAG, "[Attenuation] Manager not initialized, cannot attenuate")
            return
        }

        val oldTier = manager.getCurrentTier()
        val actualTier = manager.transitionTo(tier)

        if (actualTier == oldTier) {
            Log.d(TAG, "[Attenuation] No tier change (hysteresis or same tier)")
            return
        }

        Log.i(TAG, "[Attenuation] Attenuating from $oldTier to $actualTier")

        // Apply attenuation based on tier
        when (actualTier) {
            MemoryAttenuationManager.AttenuationTier.MEDIUM -> {
                // Close age/gender processor
                audienceAnalyzer?.closeAgeGenderProcessor()
                // Increase aggregation window to 15s
                lastAggregationTime = System.currentTimeMillis() - (15_000 - AGGREGATION_WINDOW_MS)
                Log.d(TAG, "[Attenuation] MEDIUM: Closed age/gender, set aggregation to 15s")
            }
            MemoryAttenuationManager.AttenuationTier.HIGH -> {
                // Also stop pose/emotion/gaze
                audienceAnalyzer?.closeAgeGenderProcessor()
                audienceAnalyzer?.stopPoseDetection()
                audienceAnalyzer?.stopEmotionDetection()
                audienceAnalyzer?.stopGazeTracking()
                if (vlmProcessor != null) {
                    Log.w(TAG, "[Attenuation] HIGH: Unloading VLM processor under memory pressure")
                    setVlmProcessor(null)
                }
                lastAggregationTime = System.currentTimeMillis() - (15_000 - AGGREGATION_WINDOW_MS)
                Log.d(TAG, "[Attenuation] HIGH: Closed age/gender + pose/emotion/gaze, aggregation 15s")
            }
            MemoryAttenuationManager.AttenuationTier.CRITICAL -> {
                // Also stop speech processor
                audienceAnalyzer?.closeAgeGenderProcessor()
                audienceAnalyzer?.stopPoseDetection()
                audienceAnalyzer?.stopEmotionDetection()
                audienceAnalyzer?.stopGazeTracking()
                speechProcessor?.stop()
                speechProcessor = null
                if (vlmProcessor != null) {
                    Log.w(TAG, "[Attenuation] CRITICAL: Unloading VLM processor under memory pressure")
                    setVlmProcessor(null)
                }
                lastAggregationTime = System.currentTimeMillis() - (15_000 - AGGREGATION_WINDOW_MS)
                Log.d(TAG, "[Attenuation] CRITICAL: Stopped speech processor, aggregation 15s")
            }
            MemoryAttenuationManager.AttenuationTier.NORMAL -> {
                // This case handled by restoreToTier
            }
        }

        val estimatedMB = estimateMemoryFreed(oldTier, actualTier)
        Log.i(TAG, "[Attenuation] Tier: $oldTier → $actualTier, freed ~${estimatedMB}MB")
    }

    /**
     * Restore to a less aggressive tier when memory pressure drops.
     */
    fun restoreToTier(tier: MemoryAttenuationManager.AttenuationTier) {
        val manager = attenuationManager ?: run {
            Log.w(TAG, "[Attenuation] Manager not initialized, cannot restore")
            return
        }

        val oldTier = manager.getCurrentTier()
        val actualTier = manager.transitionTo(tier)

        if (actualTier == oldTier) {
            Log.d(TAG, "[Attenuation] No tier change (hysteresis or same tier)")
            return
        }

        Log.i(TAG, "[Attenuation] Restoring from $oldTier to $actualTier")

        // Restore models based on tier
        when (actualTier) {
            MemoryAttenuationManager.AttenuationTier.NORMAL -> {
                // Restore everything
                audienceAnalyzer?.startAgeGenderProcessor()
                audienceAnalyzer?.startPoseDetection()
                audienceAnalyzer?.startEmotionDetection()
                audienceAnalyzer?.startGazeTracking()
                // Restore original aggregation window
                lastAggregationTime = System.currentTimeMillis() - (originalAggregationWindowMs - AGGREGATION_WINDOW_MS)
                // Restart speech if enabled
                val capabilities = detectCapabilities()
                if (capabilities.speechIntelligenceAvailable && sensingConfig.enableSpeechIntelligence) {
                    startSpeechIntelligence(sensingConfig)
                }
                maybeRestoreVlmAfterAttenuation()
                Log.d(TAG, "[Attenuation] NORMAL: Restored all models, aggregation ${originalAggregationWindowMs}ms")
            }
            MemoryAttenuationManager.AttenuationTier.MEDIUM -> {
                // Restore pose/emotion/gaze, keep age/gender closed
                audienceAnalyzer?.startPoseDetection()
                audienceAnalyzer?.startEmotionDetection()
                audienceAnalyzer?.startGazeTracking()
                // Restart speech if enabled
                val capabilities = detectCapabilities()
                if (capabilities.speechIntelligenceAvailable && sensingConfig.enableSpeechIntelligence) {
                    startSpeechIntelligence(sensingConfig)
                }
                maybeRestoreVlmAfterAttenuation()
                Log.d(TAG, "[Attenuation] MEDIUM: Restored pose/emotion/gaze/speech, aggregation 15s")
            }
            MemoryAttenuationManager.AttenuationTier.HIGH -> {
                // Restore speech only
                val capabilities = detectCapabilities()
                if (capabilities.speechIntelligenceAvailable && sensingConfig.enableSpeechIntelligence) {
                    startSpeechIntelligence(sensingConfig)
                }
                Log.d(TAG, "[Attenuation] HIGH: Restored speech processor, aggregation 15s")
            }
            MemoryAttenuationManager.AttenuationTier.CRITICAL -> {
                // Already at most aggressive, nothing to restore
            }
        }

        val estimatedMB = estimateMemoryFreed(oldTier, actualTier)
        Log.i(TAG, "[Attenuation] Tier: $oldTier → $actualTier, changed ~${estimatedMB}MB")
    }

    private fun maybeRestoreVlmAfterAttenuation() {
        val currentModels = activeProfileModels
        val runtimeContract = activeProgramRuntimeContract
        val wantsVlm = if (runtimeContract?.temporalSemantics != null) {
            runtimeContract.temporalSemantics.enabled &&
                runtimeContract.temporalSemantics.modelIds.any { it == "gemma_4_e2b" }
        } else {
            currentModels.any { it == "gemma_4_e2b" }
        }
        if (!wantsVlm || vlmProcessor != null) {
            return
        }

        Log.i(TAG, "[Attenuation] Re-evaluating VLM activation after memory recovery")
        applyProfileToInference(currentModels)
    }

    /**
     * Estimate memory freed/allocated during tier transition.
     */
    private fun estimateMemoryFreed(oldTier: MemoryAttenuationManager.AttenuationTier, newTier: MemoryAttenuationManager.AttenuationTier): Int {
        val modelSizes = mapOf(
            "whisper" to 100,
            "ageGender" to 20,
            "pose" to 15,
            "emotion" to 10,
            "gaze" to 5
        )

        var freed = 0
        when {
            oldTier == MemoryAttenuationManager.AttenuationTier.NORMAL && newTier == MemoryAttenuationManager.AttenuationTier.MEDIUM -> {
                freed = modelSizes["ageGender"] ?: 0
            }
            oldTier == MemoryAttenuationManager.AttenuationTier.MEDIUM && newTier == MemoryAttenuationManager.AttenuationTier.HIGH -> {
                freed = (modelSizes["pose"] ?: 0) + (modelSizes["emotion"] ?: 0) + (modelSizes["gaze"] ?: 0)
            }
            oldTier == MemoryAttenuationManager.AttenuationTier.HIGH && newTier == MemoryAttenuationManager.AttenuationTier.CRITICAL -> {
                freed = modelSizes["whisper"] ?: 0
            }
            // Reverse (restoration)
            oldTier == MemoryAttenuationManager.AttenuationTier.CRITICAL && newTier == MemoryAttenuationManager.AttenuationTier.HIGH -> {
                freed = -(modelSizes["whisper"] ?: 0)
            }
            oldTier == MemoryAttenuationManager.AttenuationTier.HIGH && newTier == MemoryAttenuationManager.AttenuationTier.MEDIUM -> {
                freed = -((modelSizes["pose"] ?: 0) + (modelSizes["emotion"] ?: 0) + (modelSizes["gaze"] ?: 0))
            }
            oldTier == MemoryAttenuationManager.AttenuationTier.MEDIUM && newTier == MemoryAttenuationManager.AttenuationTier.NORMAL -> {
                freed = -(modelSizes["ageGender"] ?: 0)
            }
        }
        return freed
    }
}

/**
 * Device sensing capabilities.
 */
data class SensingCapabilities(
    val hasCameraHardware: Boolean = false,
    val hasMicrophoneHardware: Boolean = false,
    val hasCameraPermission: Boolean = false,
    val hasMicrophonePermission: Boolean = false,
    val faceDetectionAvailable: Boolean = false,
    val audioClassificationAvailable: Boolean = false,
    val speechIntelligenceAvailable: Boolean = false,
    val emotionalEngagementAvailable: Boolean = false  // Pose + Emotion + Gaze
) {
    val hasAnySensing: Boolean
        get() = faceDetectionAvailable || audioClassificationAvailable || speechIntelligenceAvailable
}

/**
 * Sensing mode based on available capabilities.
 */
enum class SensingMode {
    NONE,           // No sensors available
    FACE_ONLY,      // Camera only (no microphone)
    AUDIO_ONLY,     // Microphone only (no camera)
    FULL;           // Both camera and microphone

    fun hasCamera(): Boolean = this == FACE_ONLY || this == FULL
    fun hasAudio(): Boolean = this == AUDIO_ONLY || this == FULL
}

// Note: CameraStatus is defined in AudienceAnalyzer.kt

/**
 * Current sensing state.
 */
data class SensingState(
    val isRunning: Boolean = false,
    val mode: SensingMode = SensingMode.NONE,
    val error: String? = null
)

/**
 * Configuration for audience sensing.
 */
data class SensingConfig(
    val enableFaceDetection: Boolean = true,
    val enableAudioClassification: Boolean = true,
    val enableSpeechIntelligence: Boolean = true,   // Enable on-device speech analysis
    val enableDemographicsCapture: Boolean = true,  // Enable Gemini Vision demographics (every 5 min)
    val enableEmotionalEngagement: Boolean = true,  // Enable pose/emotion/gaze analysis
    val faceDetectionIntervalMs: Long = 1000,       // Analyze camera every 1 second
    val audioClassificationIntervalMs: Long = 1000, // Classify audio every 1 second
    val speechTranscriptionIntervalMs: Long = 10_000 // Transcribe speech every 10 seconds
)
