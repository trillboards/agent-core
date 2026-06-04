package com.trillboards.ctv.core.audience

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import com.trillboards.ctv.core.SensingConfig
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit

/**
 * Camera status information for heartbeat reporting.
 */
data class CameraStatus(
    val available: Boolean,
    val type: String,  // "usb", "internal", "none"
    val count: Int
)

internal inline fun shouldReinitializeSharedBitmap(
    hasBitmap: Boolean,
    isRecycled: () -> Boolean,
    widthProvider: () -> Int,
    heightProvider: () -> Int,
    expectedWidth: Int,
    expectedHeight: Int
): Boolean {
    if (!hasBitmap) {
        return true
    }
    if (isRecycled()) {
        return true
    }
    return widthProvider() != expectedWidth || heightProvider() != expectedHeight
}

/**
 * Why we need to (re)initialize the shared bitmap. Used to differentiate "no
 * bitmap yet, just allocate one quietly" from "we have a bitmap but the HAL
 * delivered a different resolution than we asked for", which is the only case
 * worth a WARN. Without this distinction the analyzer logged "expected
 * nullxnull, actual ..." every frame between [reset()] and the first
 * [Bitmap.createBitmap] inside [initializeBuffers] — 53/min on Tab S11.
 */
internal enum class SharedBitmapReinitCause {
    NONE,                  // bitmap is live and matches frame dimensions — no reinit
    NEEDS_ALLOCATION,      // bitmap is null or recycled — first frame after teardown
    RESOLUTION_MISMATCH    // bitmap is live but its width/height differ from the frame
}

internal inline fun classifyReinitCause(
    hasBitmap: Boolean,
    isRecycled: () -> Boolean,
    widthProvider: () -> Int,
    heightProvider: () -> Int,
    expectedWidth: Int,
    expectedHeight: Int
): SharedBitmapReinitCause {
    if (!hasBitmap || isRecycled()) {
        return SharedBitmapReinitCause.NEEDS_ALLOCATION
    }
    if (widthProvider() != expectedWidth || heightProvider() != expectedHeight) {
        return SharedBitmapReinitCause.RESOLUTION_MISMATCH
    }
    return SharedBitmapReinitCause.NONE
}

internal data class SharedBitmapBufferState(
    val needsReinitialization: Boolean,
    val reinitCause: SharedBitmapReinitCause,
    val currentWidth: Int?,
    val currentHeight: Int?
)

/**
 * Confidence floor below which a HAPPY frame is dropped from the
 * "mid-band positive valence" tally that promotes NEUTRAL → INTERESTED.
 *
 * FaceLandmarker emits per-face emotion confidences as the average of
 * smile/surprise blendshape activations. A frame with a fleeting twitch
 * (confidence ~0.05) is not the same signal as a sustained smile
 * (confidence ~0.30+). The floor keeps single-frame noise out of the
 * INTERESTED promotion path while letting genuine but mid-intensity
 * positive expression carry the audience above NEUTRAL.
 */
internal const val HAPPY_FRAME_CONFIDENCE_FLOOR: Float = 0.15f

/**
 * Pure ladder helper: maps the engagement composite + valence counts +
 * happy-frame tally to an [AudienceReaction] tier.
 *
 * Cosmic-brewing-bear plan task #33: the previous ladder collapsed any
 * non-strong positive/negative valence into NEUTRAL, hiding "audience
 * leaning positive but not fully engaged" — meaningful advertiser-audit
 * signal. Add an explicit INTERESTED branch for windows where ANY HAPPY
 * frames cleared [HAPPY_FRAME_CONFIDENCE_FLOOR] but the strong-positive
 * (HIGHLY_ENGAGED) threshold isn't met.
 *
 * Order of precedence (must match aggregateEmotionalEngagement()):
 *   1. Negative valence wins regardless of intensity.
 *   2. High composite-score → HIGHLY_ENGAGED.
 *   3. Mid composite-score → INTERESTED (existing intensity branch).
 *   4. ANY HAPPY frames above the confidence floor → INTERESTED (NEW —
 *      mid-band positive valence escapes the NEUTRAL collapse).
 *   5. Above neutral threshold OR positive valence (any intensity) →
 *      NEUTRAL.
 *   6. Catch-all → NEUTRAL.
 *
 * Pure / no Android dependencies / unit-testable.
 */
internal fun computeAudienceReaction(
    overallScore: Float,
    highlyEngagedThreshold: Float,
    interestedThreshold: Float,
    neutralThreshold: Float,
    isNegativeValence: Boolean,
    isPositiveValence: Boolean,
    happyFrameCount: Int
): AudienceReaction = when {
    isNegativeValence -> AudienceReaction.NEGATIVE
    overallScore > highlyEngagedThreshold -> AudienceReaction.HIGHLY_ENGAGED
    overallScore > interestedThreshold -> AudienceReaction.INTERESTED
    happyFrameCount > 0 -> AudienceReaction.INTERESTED
    overallScore > neutralThreshold || isPositiveValence -> AudienceReaction.NEUTRAL
    else -> AudienceReaction.NEUTRAL
}

/**
 * Analyzes camera feed to detect and track audience members.
 *
 * Privacy-first design:
 * - All processing happens on-device using ML Kit
 * - No images are stored or transmitted
 * - Only anonymous aggregate data is collected
 * - Face tracking IDs are session-local and ephemeral
 *
 * Camera Management:
 * This class manages the shared camera lifecycle for both:
 * - ImageAnalysis (ML Kit face detection - continuous)
 * - ImageCapture (Gemini demographics capture - on-demand via FrameCaptureManager)
 *
 * CameraX only allows one camera binding per provider, so we bind both
 * use cases together here and expose ImageCapture for external use.
 *
 * USB Camera Support:
 * Automatically detects and prefers USB cameras (via CameraX UVC support).
 * Falls back to front camera, then back camera if no USB camera available.
 *
 * Adaptive Performance:
 * Uses DeviceProfile to detect chipset and apply optimal settings.
 * Frame throttling reduces CPU/memory pressure on MediaTek devices.
 */
class AudienceAnalyzer(
    private val context: Context,
    private val config: AudienceConfig = AudienceConfig()
) {
    companion object {
        private const val TAG = "AudienceAnalyzer"
        private const val MAX_TRACKED_FACES = 50
        private const val HEALTHY_CAMERA_REBIND_GRACE_MS = 20_000L

        /**
         * Maximum wait for the camera HAL to deliver its first frame after
         * a successful `bindToLifecycle()`. Most CameraX implementations
         * deliver within ~2-3s; RockChip class HALs can take up to ~6s on
         * cold boot. 15s is well past that — anything longer means the bind
         * succeeded but the HAL is stalled (the silent fail-zero pattern
         * that bricked Adam @ Focus Media's screen for 30+ days).
         */
        internal const val FIRST_FRAME_GRACE_MS = 15_000L

        // Phase 1c — upper bound on a single sample's frame interval. Without
        // this a long pause (camera teardown, attenuation tier swap) would
        // dump a multi-minute "dwell" into the next bucket. 5 seconds is
        // comfortably beyond any normal frame cadence (typical 0.2-1s on the
        // S11) but small enough to keep dwell calculations honest.
        internal const val PER_FACE_MAX_FRAME_INTERVAL_MS: Long = 5_000L
    }

    // Device profile for adaptive settings
    private val deviceProfile by lazy { DeviceProfile.detect(context) }

    // Frame throttler for skipping frames based on device capabilities.
    // cameraFps defaults from SensingConfig (historically 30); it is corrected to
    // the real production rate once the camera binds and the AE target-FPS range is
    // pinned — see bindCameraUseCases() → frameThrottler.setCameraFps().
    private val frameThrottler by lazy {
        FrameThrottler(initialTargetFps = deviceProfile.targetFps)
    }

    // Resource monitor for memory pressure
    private val resourceMonitor by lazy { ResourceMonitor(context) }

    private val analyzerScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val frameBufferLeaseController = FrameBufferLeaseController()

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var faceDetector: FaceDetector? = null

    // ImageCapture for demographics - shared with FrameCaptureManager
    private var _imageCapture: ImageCapture? = null

    // Camera binding state
    private var _isCameraBound = false

    /**
     * Get the ImageCapture use case for external frame capture.
     * This is used by FrameCaptureManager for Gemini demographics analysis.
     * Returns null until camera is bound.
     */
    val imageCapture: ImageCapture? get() = _imageCapture

    /**
     * Check if camera is currently bound and ready for capture.
     */
    val isCameraBound: Boolean get() = _isCameraBound

    /**
     * Camera lifecycle health gauge — owns retry/backoff/escalation policy
     * for the camera-start path. Wired by AudienceSensingService so the
     * heartbeat surfaces `camera_health` and the cap_face_detection devolve
     * step has a state machine to query.
     *
     * Default-initialized so fresh boots already have a valid snapshot
     * (state=NEVER_STARTED) before any start() call lands.
     */
    val cameraHealth: CameraHealthMonitor = CameraHealthMonitor()

    /**
     * Callback invoked when camera is successfully bound and ImageCapture is ready.
     * Use this to wire up FrameCaptureManager instead of polling.
     */
    var onCameraReady: ((ImageCapture) -> Unit)? = null

    /**
     * Callback invoked when no camera is available (for graceful degradation to audio-only).
     * Fire TV and Android TV without USB cameras will trigger this.
     */
    var onNoCameraAvailable: (() -> Unit)? = null

    /**
     * Callback invoked when the camera-start path fails AFTER all retries are
     * exhausted. AudienceSensingService wires this to a `stabilityEvent`
     * `CameraInitializationFailed` post so the operator gets a loud signal
     * with full diagnostic context.
     */
    var onCameraStartFailed: ((CameraHealthMonitor.RetryDecision) -> Unit)? = null

    // Track faces across frames
    private val trackedFaces = java.util.concurrent.ConcurrentHashMap<Int, DetectedFace>()
    private var peakViewerCount = 0

    // Frame processing statistics
    private var frameCount = 0L
    private var lastFrameLogTime = 0L
    @Volatile private var lastFrameProcessedAtMs = 0L
    private val frameLogIntervalMs = 10_000L  // Log every 10 seconds
    private var lastFpsChangeMs = 0L
    private val FPS_HOLD_PERIOD_MS = 60_000L  // Hold FPS for 60s after any change
    @Volatile private var lastInferenceMs = 0L
    @Volatile private var rollingInferenceMs = 0.0

    // Sensor data
    private val sensorCollector = SensorCollector(context)

    // Emotional engagement processors
    private var poseProcessor: PoseEngagementProcessor? = null
    // FaceLandmarker (cosmic-brewing-bear Task #6 + Phase 3 cleanup) — sole
    // source of per-frame emotion + iris-gaze + head-pose signals. The legacy
    // EmotionClassificationProcessor (FER+ TFLite, 100% NEUTRAL bias on the
    // Tab S11 funny-faces soak) and GazeTrackingProcessor (Google ML Kit
    // FaceMesh head-pose proxy with no iris landmarks) were deleted in this
    // phase; one FaceLandmarker forward pass produces 52 ARKit blendshapes
    // (FACS-explicit emotion) + 478 landmarks including iris 468-477 (true
    // gaze) + a 4×4 facial transformation matrix (head-pose Euler angles).
    private var faceLandmarkerProcessor: FaceLandmarkerProcessor? = null
    private var emotionalEngagementEnabled = false

    // Zone-dwell heatmap (3×3 viewport grid). Populated from the per-frame
    // FaceLandmarker gaze focusRegion in processEmotionalEngagementZeroAlloc;
    // drained by AudienceSensingService at the end of each report window.
    // Replaces the stateful tracking that lived on the deleted
    // GazeTrackingProcessor.
    private val zoneDwellMs = LongArray(9)
    private var currentZone: Int = 4               // 4 == center of the 3×3 grid
    private var lastZoneChangeMs: Long = 0

    // On-device age/gender processor
    private var ageGenderProcessor: AgeGenderProcessor? = null
    private var ageGenderEnabled = false

    // Local TFLite person detector (PoC) — only instantiated if model exists
    private var personDetector: PersonDetectionProcessor? = null
    private var currentPersonCount = 0

    // Multi-class object detector + downstream counting processors.
    // Reuses the same efficientdet_lite0.tflite as personDetector but configured
    // for all 80 COCO classes instead of just "person".
    private var objectDetector: ObjectDetectionProcessor? = null
    private val vehicleFlowProcessor = VehicleFlowProcessor()
    private val queueEstimationProcessor = QueueEstimationProcessor()

    @Volatile private var desiredVisionSelection: VisionProcessorSelection? = null

    // Latest emotional engagement metrics
    private val _currentEmotionalEngagement = MutableStateFlow<EmotionalEngagement?>(null)
    val currentEmotionalEngagement: StateFlow<EmotionalEngagement?> = _currentEmotionalEngagement

    // Buffers for emotional engagement aggregation
    private val poseMetricsBuffer = ArrayDeque<PoseMetrics>()
    private val emotionMetricsBuffer = ArrayDeque<EmotionMetrics>()
    private val gazeMetricsBuffer = ArrayDeque<GazeMetrics>()

    // ============================================================
    // Phase 1c — per-(face × ad) attention accumulator
    // ============================================================
    // Stores the currently-playing ad ID + impression ID so per-frame
    // sample accumulation can bucket by (adId, faceTrackId). Updated via
    // [setCurrentAd] from AudienceSensingService whenever the
    // ContentStateProvider reports an ad-state change.
    //
    // 2026-05-11 — currentAdId is now broadened to "current creative ID"
    // covering ima_programmatic + self_promo + sponsored + default_stream.
    // currentCreativeSource discriminates the origin so downstream
    // attribution can distinguish a programmatic ad bid from a free
    // self-promo placement in the same window.
    @Volatile private var currentAdId: String? = null
    @Volatile private var currentImpressionId: String? = null
    @Volatile private var currentCreativeSource: String? = null

    // perAdAttention[adId][faceTrackId] -> running PerFaceAttention bucket.
    // ConcurrentHashMap on the outer layer (multi-thread put/get from
    // setCurrentAd vs aggregator vs frame producer). Inner map is also
    // concurrent because frames and aggregator drain may overlap.
    private val perAdAttention =
        java.util.concurrent.ConcurrentHashMap<String, java.util.concurrent.ConcurrentHashMap<Int, PerFaceAttention>>()

    // adIdToSource[adId] -> "ima_programmatic" | "self_promo" | "sponsored" |
    // "default_stream". Snapshotted at drain time so the per_face_observations
    // emit stamps each row with the creative_source it actually came from
    // (handles multi-creative windows where a self_promo followed an IMA ad).
    private val adIdToSource =
        java.util.concurrent.ConcurrentHashMap<String, String>()

    // Last per-face sample timestamp — used to compute frameIntervalMs
    // for dwell-seconds accumulation. Reset on window flush.
    @Volatile private var lastPerFaceSampleMs: Long = 0L

    // ============================================================
    // Zero-allocation pre-allocated buffers (Phase 1)
    // Allocated once in initializeBuffers(), reused every frame.
    // ============================================================

    // Shared bitmap: camera frame is rendered here via manual YUV→RGB conversion.
    // Lifetime = camera session. Never recycled during operation by us, BUT external
    // consumers must NOT recycle it either — historically PersonDetectionProcessor's
    // `mpImage.close()` did recycle the source via MediaPipe's
    // BitmapImageContainer.close() semantics (fixed in
    // PersonDetectionProcessor.process() — see commit message). Marked @Volatile
    // defensively in case any future consumer reads/writes from a non-analysis
    // thread.
    @Volatile private var sharedBitmap: Bitmap? = null

    // NV21 byte array for YUV plane extraction (size = ySize + uSize + vSize)
    private var nv21Buffer: ByteArray? = null

    // Int array for holding ARGB pixels during YUV→RGB conversion
    private var yuvPixelBuffer: IntArray? = null

    // ByteBuffer for MediaPipe Pose input (RGB unsigned bytes, 3 bytes per pixel)
    private var poseInputBuffer: ByteBuffer? = null

    // Pre-allocated crop bitmaps for emotion and age/gender processors
    private var emotionCropBitmap: Bitmap? = null      // 48x48 ARGB_8888
    private var ageGenderCropBitmap: Bitmap? = null     // 224x224 ARGB_8888 (FaceXFormer)

    // Adaptive FPS calibration
    private var calibrationFrameCount = 0
    private var calibrationTotalMs = 0L
    private var isCalibrated = false
    private val CALIBRATION_FRAMES = 10

    // Current metrics state
    private val _currentSnapshot = MutableStateFlow(AudienceSnapshot())
    val currentSnapshot: StateFlow<AudienceSnapshot> = _currentSnapshot

    // Callback for when faces are detected (for FrameCaptureManager demographics trigger)
    var onFacesDetected: ((Int) -> Unit)? = null

    // Callback for emotional engagement metrics
    var onEmotionalEngagementReady: ((EmotionalEngagement) -> Unit)? = null

    // Callback for camera health heartbeat — called on every successfully processed frame.
    // AudienceSensingService wires this to SubsystemHealthRegistry.updateHealth(CAMERA, ACTIVE)
    // to prevent RecoveryLadder from false-positive "camera failed" detection.
    var onFrameProcessed: (() -> Unit)? = null

    /** Callback invoked with each camera frame bitmap for VLM inference routing.
     *  The bitmap is valid only during the callback — caller must copy if needed. */
    var onBitmapAvailable: ((Bitmap) -> Unit)? = null

    private var isRunning = false

    /**
     * Initialize ML Kit face detector.
     */
    private fun initializeFaceDetector() {
        Log.d(TAG, "[MLKit] Initializing face detector...")
        val detCfg = SensingConfig.get().detection
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)  // Eye landmarks consumed by HeuristicEmotionalEngagementEstimator fallback path
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)  // For eye/smile
            .setMinFaceSize(detCfg.minFaceSize)  // Minimum face size relative to image
            .enableTracking()  // Enable face tracking across frames
            .build()

        faceDetector = FaceDetection.getClient(options)
        Log.i(TAG, "[MLKit] Face detector initialized - mode=FAST, landmarks=ALL, classification=ALL, minFaceSize=15%, tracking=ON")
    }

    /**
     * Initialize emotional engagement processors (pose + FaceLandmarker for
     * emotion/gaze/head-pose, plus optional age-gender and person detection).
     *
     * Guards against double initialization: if processors already exist, they are
     * closed first to prevent native memory leaks from orphaned model instances.
     * Each PoseLandmarker/MediaPipe interpreter holds ~500MB of native memory.
     */
    private fun initializeEmotionalEngagement(initialSelection: VisionProcessorSelection? = null) {
        Log.d(TAG, "[EmotionalEngagement] Initializing emotional engagement processors...")

        // CRITICAL: Close existing processors before creating new ones.
        // Without this, each re-init leaks ~500MB of native model weights.
        poseProcessor?.release()
        poseProcessor = null
        faceLandmarkerProcessor?.release()
        faceLandmarkerProcessor = null
        ageGenderProcessor?.close()
        ageGenderProcessor = null
        ageGenderEnabled = false
        personDetector?.release()
        personDetector = null
        objectDetector?.release()
        objectDetector = null
        resetZoneDwell()

        val wantsLegacyFullStack = initialSelection == null
        val wantsPose = wantsLegacyFullStack || initialSelection?.wantsPose == true
        val wantsEmotion = wantsLegacyFullStack || initialSelection?.wantsEmotion == true
        val wantsGaze = wantsLegacyFullStack || initialSelection?.wantsGaze == true
        val wantsAgeGender = wantsLegacyFullStack || initialSelection?.wantsAgeGender == true
        val wantsPersonDetection = wantsLegacyFullStack || initialSelection?.wantsPersonDetection == true

        var processorsInitialized = 0

        if (wantsPose) {
            poseProcessor = PoseEngagementProcessor(context).apply {
                if (hasModel() && initialize()) {
                    processorsInitialized++
                    Log.i(TAG, "[EmotionalEngagement] Pose processor initialized")
                } else {
                    Log.w(TAG, "[EmotionalEngagement] Pose processor not available (model missing)")
                    poseProcessor = null
                }
            }
        }

        if (wantsEmotion || wantsGaze) {
            // FaceLandmarker is the sole source of emotion + gaze + head-pose
            // signals in one forward pass per frame. If init fails (asset
            // missing, low-memory device) we degrade to the heuristic estimator
            // in HeuristicEmotionalEngagementEstimator, which derives a coarse
            // engagement signal from ML Kit face metadata only.
            faceLandmarkerProcessor = FaceLandmarkerProcessor(context).apply {
                if (initialize()) {
                    processorsInitialized++
                    Log.i(TAG, "[EmotionalEngagement] FaceLandmarker (blendshape emotion + iris gaze + head-pose) initialized — sole emotion/gaze source")
                } else {
                    Log.w(TAG, "[EmotionalEngagement] FaceLandmarker init failed; degrading to ML Kit heuristic estimator only")
                    faceLandmarkerProcessor = null
                }
            }
        }

        if (wantsAgeGender) {
            val agpCandidate = AgeGenderProcessor(context)
            if (agpCandidate.hasModel() && agpCandidate.initialize()) {
                ageGenderProcessor = agpCandidate
                ageGenderEnabled = true
                Log.i(TAG, "[EmotionalEngagement] Age/gender processor initialized")
            } else {
                agpCandidate.close()
                Log.d(TAG, "[EmotionalEngagement] Age/gender model not available (optional)")
            }
        }

        if (wantsPersonDetection) {
            val pdCandidate = PersonDetectionProcessor(context)
            if (pdCandidate.hasModel() && pdCandidate.initialize()) {
                personDetector = pdCandidate
                Log.i(TAG, "[PersonDetection] TFLite person detector initialized")
            } else {
                pdCandidate.release()
                Log.d(TAG, "[PersonDetection] TFLite model not available — skipping (not on device)")
            }

            // Wire ObjectDetectionProcessor + VehicleFlowProcessor + QueueEstimationProcessor.
            // Reuses the same efficientdet_lite0.tflite — no additional model download.
            // ObjectDetectionProcessor is separate from PersonDetectionProcessor so it can
            // filter all 80 COCO classes (vehicles, persons in queue context, etc.) while
            // personDetector continues its existing person-only count path unchanged.
            val odCandidate = ObjectDetectionProcessor(context)
            if (odCandidate.hasModel() && odCandidate.initialize()) {
                objectDetector = odCandidate
                vehicleFlowProcessor.reset()
                queueEstimationProcessor.reset()
                Log.i(TAG, "[ObjectDetection] Multi-class detector initialized (vehicles + pedestrians)")
            } else {
                odCandidate.release()
                Log.d(TAG, "[ObjectDetection] Model not available — skipping (not on device)")
            }
        }

        refreshProcessorActivationState(processorsInitialized > 0)

        if (emotionalEngagementEnabled) {
            Log.i(TAG, "[EmotionalEngagement] >>> Emotional engagement ENABLED ($processorsInitialized processors)")
        } else {
            Log.w(TAG, "[EmotionalEngagement] Emotional engagement DISABLED (no processors available)")
        }
    }

    private fun hasEmotionalProcessorsEnabled(): Boolean {
        return poseProcessor != null || faceLandmarkerProcessor != null || ageGenderEnabled
    }

    private fun hasSharedBitmapConsumers(): Boolean {
        return hasEmotionalProcessorsEnabled() || personDetector?.isReady() == true || onBitmapAvailable != null
    }

    private fun refreshProcessorActivationState(overrideEnabled: Boolean? = null) {
        emotionalEngagementEnabled = overrideEnabled ?: hasEmotionalProcessorsEnabled()
    }

    /**
     * Initialize pre-allocated buffers for zero-allocation frame pipeline.
     * Called once when camera resolution is known (in bindCameraUseCases).
     *
     * After this call, processImage() performs ZERO native allocations per frame.
     */
    private fun initializeBuffers(width: Int, height: Int) {
        Log.i(TAG, "[ZeroAlloc] Initializing pre-allocated buffers for ${width}x${height}")
        frameBufferLeaseController.reset()

        // Shared bitmap: overwritten every frame via setPixels(), never recycled during operation
        sharedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // NV21 buffer: Y + U + V planes (YUV420 = 1.5 bytes per pixel)
        val yuvSize = width * height * 3 / 2
        nv21Buffer = ByteArray(yuvSize)

        // ARGB pixel buffer for YUV→RGB conversion
        yuvPixelBuffer = IntArray(width * height)

        // MediaPipe Pose input: RGB byte buffer (3 unsigned bytes per pixel)
        // IMAGE_FORMAT_RGB expects width×height×3 bytes, NOT floats
        poseInputBuffer = ByteBuffer.allocateDirect(width * height * 3).apply {
            order(java.nio.ByteOrder.nativeOrder())
        }

        // Emotion processor crop: 48x48 ARGB_8888 (reused for each face)
        emotionCropBitmap = Bitmap.createBitmap(48, 48, Bitmap.Config.ARGB_8888)

        // Age/gender processor crop: 224x224 ARGB_8888 (FaceXFormer input size)
        // Phase 2 (2026-05-04): up from 96×96 (legacy MobileFaceNet) to match
        // FaceXFormer's Swin-B 224×224 input. Bitmap is 224×224×4 bytes = 200 KB,
        // pre-allocated once per processor lifetime; no per-frame allocation.
        ageGenderCropBitmap = Bitmap.createBitmap(
            AgeGenderProcessor.MODEL_INPUT_SIZE,
            AgeGenderProcessor.MODEL_INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )

        // Reset calibration state
        calibrationFrameCount = 0
        calibrationTotalMs = 0L
        isCalibrated = false

        Log.i(TAG, "[ZeroAlloc] Buffers initialized: shared=${width}x${height}, " +
                "nv21=${nv21Buffer?.size}B, pose=${poseInputBuffer?.capacity()}B, " +
                "emotionCrop=48x48, ageGenderCrop=${AgeGenderProcessor.MODEL_INPUT_SIZE}x${AgeGenderProcessor.MODEL_INPUT_SIZE}")
    }

    /**
     * Release pre-allocated buffers.
     */
    private fun releaseBuffers(recycleBitmaps: Boolean = false) {
        val staleSharedBitmap = sharedBitmap
        val staleEmotionCrop = emotionCropBitmap
        val staleAgeGenderCrop = ageGenderCropBitmap

        val releasedCleanly = frameBufferLeaseController.releaseAndWait {
            if (recycleBitmaps) {
                if (staleSharedBitmap != null && !staleSharedBitmap.isRecycled) {
                    staleSharedBitmap.recycle()
                }
                if (staleEmotionCrop != null && !staleEmotionCrop.isRecycled) {
                    staleEmotionCrop.recycle()
                }
                if (staleAgeGenderCrop != null && !staleAgeGenderCrop.isRecycled) {
                    staleAgeGenderCrop.recycle()
                }
            }
        }

        sharedBitmap = null
        nv21Buffer = null
        yuvPixelBuffer = null
        poseInputBuffer = null
        emotionCropBitmap = null
        ageGenderCropBitmap = null

        if (releasedCleanly) {
            Log.d(
                TAG,
                "[ZeroAlloc] Pre-allocated buffers released" +
                    if (recycleBitmaps) " with explicit bitmap recycle" else " without explicit recycle"
            )
        } else {
            Log.w(TAG, "[ZeroAlloc] Active frame lease exceeded release timeout - dropped stale buffer references without recycle")
        }
    }

    /**
     * Convert ImageProxy YUV_420_888 to the pre-allocated sharedBitmap.
     *
     * Manual NV21→ARGB conversion — NO BitmapFactory, NO JPEG roundtrip,
     * NO Bitmap.createBitmap. Writes directly into sharedBitmap via setPixels().
     *
     * On the first frame, checks if the actual camera resolution differs from
     * the target resolution and re-initializes buffers if needed. CameraX
     * setTargetResolution is a hint — the HAL may deliver a different size.
     *
     * @return true if conversion succeeded, false otherwise
     */
    private fun imageProxyToSharedBitmap(imageProxy: ImageProxy): Boolean {
        val width = imageProxy.width
        val height = imageProxy.height

        // Check buffer state under the lease gate so teardown cannot recycle the
        // bitmap between the recycled check and width/height access.
        val bufferState = frameBufferLeaseController.withLease {
            val bitmap = sharedBitmap
            val recycled = bitmap?.isRecycled == true
            val currentWidth = if (bitmap != null && !recycled) bitmap.width else null
            val currentHeight = if (bitmap != null && !recycled) bitmap.height else null

            val cause = classifyReinitCause(
                hasBitmap = bitmap != null,
                isRecycled = { recycled },
                widthProvider = { currentWidth!! },
                heightProvider = { currentHeight!! },
                expectedWidth = width,
                expectedHeight = height
            )
            SharedBitmapBufferState(
                needsReinitialization = cause != SharedBitmapReinitCause.NONE,
                reinitCause = cause,
                currentWidth = currentWidth,
                currentHeight = currentHeight
            )
        } ?: return false

        if (bufferState.needsReinitialization) {
            // Only log a WARN when the HAL actually delivered a different size
            // than what we have allocated — that's a real surprise worth flagging.
            // For NEEDS_ALLOCATION (first frame, post-recycle, or the brief
            // [reset()->createBitmap] window inside [initializeBuffers]) the
            // analyzer simply allocates and moves on; logging WARN every frame
            // during that window produced the "expected nullxnull" spam on
            // Tab S11 (53/min during camera bind).
            when (bufferState.reinitCause) {
                SharedBitmapReinitCause.RESOLUTION_MISMATCH ->
                    Log.w(TAG, "[ZeroAlloc] Camera resolution mismatch: expected ${bufferState.currentWidth}x${bufferState.currentHeight}, " +
                            "actual ${width}x${height} — re-initializing buffers")
                SharedBitmapReinitCause.NEEDS_ALLOCATION ->
                    Log.d(TAG, "[ZeroAlloc] Allocating shared bitmap on first frame at actual ${width}x${height}")
                SharedBitmapReinitCause.NONE -> Unit
            }
            releaseBuffers()
            initializeBuffers(width, height)
        }

        return frameBufferLeaseController.withLease {
            val bitmap = sharedBitmap ?: return@withLease false
            if (bitmap.isRecycled) {
                return@withLease false
            }
            val pixels = yuvPixelBuffer ?: return@withLease false

            // Extract Y, U, V planes
            val yPlane = imageProxy.planes[0]
            val uPlane = imageProxy.planes[1]
            val vPlane = imageProxy.planes[2]

            val yBuffer = yPlane.buffer
            val uBuffer = uPlane.buffer
            val vBuffer = vPlane.buffer

            val yRowStride = yPlane.rowStride
            val uvRowStride = uPlane.rowStride
            val uvPixelStride = uPlane.pixelStride

            // Manual YUV420 → ARGB conversion
            // This is the hot path — optimized for minimal allocation
            var pixelIdx = 0
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val y = (yBuffer.get(row * yRowStride + col).toInt() and 0xFF)
                    val uvRow = row shr 1
                    val uvCol = col shr 1
                    val uvOffset = uvRow * uvRowStride + uvCol * uvPixelStride
                    val u = (uBuffer.get(uvOffset).toInt() and 0xFF) - 128
                    val v = (vBuffer.get(uvOffset).toInt() and 0xFF) - 128

                    // YUV → RGB using integer math (faster than float)
                    var r = y + (1370705 * v shr 20)
                    var g = y - (337633 * u shr 20) - (698001 * v shr 20)
                    var b = y + (1732446 * u shr 20)

                    // Clamp to 0-255
                    r = r.coerceIn(0, 255)
                    g = g.coerceIn(0, 255)
                    b = b.coerceIn(0, 255)

                    pixels[pixelIdx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }

            // Write ARGB pixels into the pre-allocated bitmap — zero allocation
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            true
        } ?: false
    }

    /**
     * Fill the pre-allocated RGB byte ByteBuffer from the shared bitmap.
     * Used as input for MediaPipe Pose (ByteBufferImageBuilder with IMAGE_FORMAT_RGB).
     *
     * IMAGE_FORMAT_RGB expects 3 unsigned bytes per pixel (R, G, B each 0-255).
     *
     * @return true if buffer was filled successfully
     */
    private fun fillRgbByteBuffer(): Boolean {
        val bitmap = sharedBitmap ?: return false
        if (bitmap.isRecycled) {
            return false
        }
        val buffer = poseInputBuffer ?: return false
        val pixels = yuvPixelBuffer ?: return false

        buffer.rewind()

        // yuvPixelBuffer was fully populated by imageProxyToSharedBitmap() this
        // frame via the YUV→RGB loop + bitmap.setPixels(). It has not been
        // touched since (single analysisExecutor thread; MediaPipe ByteBuffer
        // wrappers are read-only consumers), so we read directly from it — no
        // getPixels() round-trip needed (that was a 1.38 MB JNI memcopy per
        // frame at 7 fps ≈ 10 MB/s of wasted native↔JVM bandwidth).
        for (pixel in pixels) {
            buffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            buffer.put(((pixel shr 8) and 0xFF).toByte())  // G
            buffer.put((pixel and 0xFF).toByte())           // B
        }

        return true
    }

    /**
     * Start audience analysis.
     * Requires camera permission to be granted.
     *
     * @param lifecycleOwner Activity or Fragment lifecycle owner
     */
    fun start(lifecycleOwner: LifecycleOwner) {
        if (!config.enabled) {
            Log.i(TAG, "[Analyzer] Audience analysis disabled in config")
            return
        }

        if (isRunning) {
            Log.w(TAG, "[Analyzer] Already running, ignoring start()")
            return
        }

        Log.i(TAG, "[Analyzer] >>> STARTING audience analysis...")
        Log.d(TAG, "[Analyzer] Config: captureInterval=${config.captureIntervalMs}ms, " +
                "reportInterval=${config.reportIntervalMs}ms, minFaceConfidence=${config.minFaceConfidence}")

        isRunning = true
        frameCount = 0
        lastFrameLogTime = System.currentTimeMillis()
        lastFrameProcessedAtMs = 0L
        initializeFaceDetector()

        // Initialize emotional engagement processors
        initializeEmotionalEngagement(desiredVisionSelection)

        if (config.enableEnvironmentSensors) {
            sensorCollector.start()
            Log.d(TAG, "[Analyzer] Environment sensors started")
        }

        // Kick off camera-start with the health monitor in the loop. attemptCameraStart
        // owns retry/backoff/escalation — the failure-mode classifier in
        // CameraHealthMonitor turns each silent fail-zero (RockChip lensFacing
        // throw, ProcessCameraProvider init exception, bindToLifecycle throw)
        // into a loud Log.e + telemetry hop + scheduled retry.
        attemptCameraStart(lifecycleOwner)

        Log.i(TAG, "[Analyzer] Audience analysis initialization complete")
    }

    /**
     * Owns the full camera-start retry loop.
     *
     * Failure modes diagnosed in production (Adam @ Focus Media, RockChip
     * Generic Android 13, 30+ days of silent fail-zero):
     *
     *   1. ProcessCameraProvider.getInstance(...) throws
     *      InitializationException (HAL not ready) — old code logged and
     *      returned. Now: classified as PROVIDER_INIT_FAILED, retried with
     *      backoff.
     *
     *   2. selectBestCamera(...) returned null because every
     *      CameraInfo.lensFacing query threw on the RockChip HAL. Old code
     *      caught the throw and returned `false` from each filter, then
     *      logged "no suitable camera found" and devolved to audio-only
     *      forever. Now: failures inside the lens-facing loop bubble up
     *      via cameraHealth.onStartFailure with class
     *      LENS_FACING_QUERY_FAILED so we can retry and escalate.
     *
     *   3. bindToLifecycle(...) threw with an opaque CameraX exception —
     *      old code logged and gave up. Now: classified as BIND_FAILED,
     *      retried.
     *
     *   4. Camera bound but no frame ever arrived (HAL silently stalled).
     *      Old code: nothing checked. Now: a frame-watchdog handler runs
     *      after the first `bindToLifecycle` returns; if no frame arrives
     *      within `FIRST_FRAME_GRACE_MS` we mark the bind as a transient
     *      failure and retry.
     */
    private fun attemptCameraStart(lifecycleOwner: LifecycleOwner) {
        if (!isRunning) {
            Log.d(TAG, "[Camera] attemptCameraStart skipped — analyzer no longer running")
            return
        }
        cameraHealth.onStartAttempt()

        val cameraProviderFuture = try {
            ProcessCameraProvider.getInstance(context)
        } catch (e: Exception) {
            // Synchronous throw is rare but covered for defensiveness — most
            // OEMs surface init errors via the future.get() codepath.
            handleCameraStartFailure(
                lifecycleOwner,
                e,
                hint = CameraHealthMonitor.FailureClass.PROVIDER_INIT_FAILED,
                context = "ProcessCameraProvider.getInstance() threw synchronously"
            )
            return
        }

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                Log.d(TAG, "[Camera] Camera provider obtained")
                bindCameraUseCases(lifecycleOwner)
                // bindCameraUseCases() will route success/failure through
                // cameraHealth itself, so we don't double-fire here.
            } catch (e: Exception) {
                handleCameraStartFailure(
                    lifecycleOwner,
                    e,
                    hint = CameraHealthMonitor.FailureClass.PROVIDER_INIT_FAILED,
                    context = "ProcessCameraProvider.getInstance().get() failed"
                )
            }
        }, ContextCompat.getMainExecutor(context))

        // Frame watchdog — after the bind path completes, if no frame has
        // arrived within FIRST_FRAME_GRACE_MS we treat it as a stuck HAL and
        // re-enter the retry path. This catches the case where bindToLifecycle
        // returned cleanly but the camera never actually started delivering
        // frames (RockChip "ghost bind" pattern, observed pre-fix).
        scheduleFirstFrameWatchdog(lifecycleOwner)
    }

    private val mainHandler by lazy { android.os.Handler(android.os.Looper.getMainLooper()) }
    private var firstFrameWatchdogToken: Any? = null

    private fun scheduleFirstFrameWatchdog(lifecycleOwner: LifecycleOwner) {
        firstFrameWatchdogToken?.let { mainHandler.removeCallbacksAndMessages(it) }
        val token = Any()
        firstFrameWatchdogToken = token
        mainHandler.postDelayed({
            // Only trigger if we never transitioned to OPEN.
            val snapshot = cameraHealth.snapshot()
            if (snapshot.state != CameraHealthMonitor.CameraState.OPEN && isRunning) {
                Log.e(
                    TAG,
                    "[Camera] FIRST FRAME WATCHDOG fired — bound=$_isCameraBound, state=${snapshot.state}, " +
                            "elapsed=${FIRST_FRAME_GRACE_MS}ms, retryCount=${snapshot.retryCount}"
                )
                cameraHealth.onTransientFailure("first_frame_grace_expired")
                handleCameraStartFailure(
                    lifecycleOwner,
                    error = null,
                    hint = CameraHealthMonitor.FailureClass.FRAME_TIMEOUT,
                    context = "First frame did not arrive within ${FIRST_FRAME_GRACE_MS}ms after bind"
                )
            }
        }, FIRST_FRAME_GRACE_MS)
    }

    /**
     * Centralized handler for every camera-start failure mode. Routes
     * through CameraHealthMonitor for retry/escalation policy, then either
     * schedules a retry or fires the escalation callback for AudienceSensingService
     * to post a `CameraInitializationFailed` stabilityEvent.
     */
    private fun handleCameraStartFailure(
        lifecycleOwner: LifecycleOwner,
        error: Throwable?,
        hint: CameraHealthMonitor.FailureClass,
        context: String
    ) {
        val msg = error?.message ?: "(no message)"
        Log.e(TAG, "[Camera] START FAILURE [$hint]: $context — $msg", error)
        val decision = cameraHealth.onStartFailure(error, hint)
        if (decision.giveUp) {
            // Notify service for escalation. Service publishes a stabilityEvent
            // and (eventually) flips cap_face_detection via the heartbeat
            // devolve step.
            onCameraStartFailed?.invoke(decision)
            return
        }

        if (!isRunning) {
            Log.d(TAG, "[Camera] retry skipped — analyzer no longer running")
            return
        }

        Log.w(
            TAG,
            "[Camera] scheduling retry in ${decision.retryAfterMs}ms (attempt=${cameraHealth.snapshot().retryCount}/${cameraHealth.snapshot().maxRetries})"
        )
        mainHandler.postDelayed({
            if (isRunning) attemptCameraStart(lifecycleOwner)
        }, decision.retryAfterMs)
    }

    /**
     * Stop audience analysis and release resources.
     */
    fun stop() {
        if (!isRunning) {
            Log.d(TAG, "[Analyzer] Not running, ignoring stop()")
            return
        }

        Log.i(TAG, "[Analyzer] >>> STOPPING audience analysis... (processed $frameCount frames)")

        isRunning = false
        _isCameraBound = false
        lastFrameProcessedAtMs = 0L
        cameraHealth.onStopRequested()
        firstFrameWatchdogToken?.let { mainHandler.removeCallbacksAndMessages(it) }
        firstFrameWatchdogToken = null
        imageAnalysis?.clearAnalyzer()
        imageAnalysis = null
        _imageCapture = null
        // unbindAll() MUST run on the main thread — post if we're on a worker thread
        val provider = cameraProvider
        if (provider != null) {
            if (android.os.Looper.myLooper() == android.os.Looper.getMainLooper()) {
                provider.unbindAll()
            } else {
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    try {
                        provider.unbindAll()
                    } catch (e: Exception) {
                        Log.w(TAG, "[Camera] unbindAll() failed on main thread: ${e.message}")
                    }
                }
            }
        }
        faceDetector?.close()
        sensorCollector.stop()
        analysisExecutor.shutdown()
        analyzerScope.cancel()

        // Release emotional engagement processors
        poseProcessor?.release()
        poseProcessor = null
        faceLandmarkerProcessor?.release()
        faceLandmarkerProcessor = null
        resetZoneDwell()
        ageGenderProcessor?.close()
        ageGenderProcessor = null
        ageGenderEnabled = false
        
        personDetector?.release()
        personDetector = null
        currentPersonCount = 0

        objectDetector?.release()
        objectDetector = null

        emotionalEngagementEnabled = false
        poseMetricsBuffer.clear()
        emotionMetricsBuffer.clear()
        gazeMetricsBuffer.clear()

        // Release pre-allocated buffers
        releaseBuffers(recycleBitmaps = true)

        trackedFaces.clear()
        peakViewerCount = 0
        _imageCapture = null

        Log.i(TAG, "[Analyzer] Audience analysis stopped and resources released")
    }

    /**
     * Rebind camera use cases after app returns from background.
     *
     * When the app goes to background, Android releases the camera. When returning
     * to foreground, the camera doesn't automatically rebind because the service
     * lifecycle state doesn't change. This method forces a camera rebind.
     *
     * @param lifecycleOwner Activity or Fragment lifecycle owner
     */
    fun rebindCamera(lifecycleOwner: LifecycleOwner) {
        if (!isRunning) {
            Log.d(TAG, "[Camera] Not running, cannot rebind camera")
            return
        }

        val lastFrameAgeMs = if (lastFrameProcessedAtMs > 0L) {
            SystemClock.elapsedRealtime() - lastFrameProcessedAtMs
        } else {
            Long.MAX_VALUE
        }

        if (_isCameraBound) {
            if (lastFrameAgeMs <= HEALTHY_CAMERA_REBIND_GRACE_MS) {
                Log.i(TAG, "[Camera] Skipping rebind - camera already healthy (lastFrameAge=${lastFrameAgeMs}ms)")
                return
            }
            Log.w(TAG, "[Camera] Camera bound but frame heartbeat is stale (${lastFrameAgeMs}ms) - forcing rebind")
        }

        Log.i(TAG, "[Camera] >>> REBINDING CAMERA after app resume...")

        val provider = cameraProvider
        if (provider == null) {
            Log.w(TAG, "[Camera] Camera provider is null, attempting to re-acquire via attemptCameraStart")
            // Route through the same retry/escalation harness so a rebind
            // failure doesn't silently swallow exceptions either.
            attemptCameraStart(lifecycleOwner)
            return
        }

        // Rebind with existing provider
        bindCameraUseCases(lifecycleOwner)
    }

    /**
     * Select the best available camera with USB preference.
     * Priority: USB/External > Front > Back > None
     *
     * CameraX automatically detects USB cameras via UVC drivers on Android 9+.
     * This enables audience sensing on Fire TV and Android TV devices with USB webcams.
     *
     * @param provider The CameraProvider to query for available cameras
     * @return CameraSelector for the best camera, or null if no camera available
     */
    private fun selectBestCamera(provider: ProcessCameraProvider): CameraSelector? {
        val cameras = provider.availableCameraInfos

        if (cameras.isEmpty()) {
            Log.w(TAG, "[Camera] No cameras available on this device")
            return null
        }

        Log.d(TAG, "[Camera] Detected ${cameras.size} camera(s)")

        // Track lens-facing query failures so we can distinguish "RockChip
        // HAL throws on every lensFacing query" (silent fail-zero, would cause
        // null return + audio-only forever) from "device legitimately has no
        // matching camera". We log each throw with the cameraInfo identity
        // so the operator can debug per-device.
        var lensQueryThrowCount = 0
        var lastLensQueryThrow: Throwable? = null
        fun safeLensFacing(cameraInfo: androidx.camera.core.CameraInfo, expected: Int): Boolean {
            return try {
                cameraInfo.lensFacing == expected
            } catch (e: Exception) {
                lensQueryThrowCount++
                lastLensQueryThrow = e
                Log.w(TAG, "[Camera] lensFacing query threw on $cameraInfo (expected=$expected): ${e.message}")
                false
            }
        }

        // Priority 1: External/USB camera (better positioned for audience on TV devices)
        val externalCamera = cameras.find { safeLensFacing(it, CameraSelector.LENS_FACING_EXTERNAL) }

        if (externalCamera != null) {
            Log.i(TAG, "[Camera] >>> Using EXTERNAL/USB camera (preferred for audience sensing)")
            return CameraSelector.Builder()
                .addCameraFilter { cameraInfos -> cameraInfos.filter { it == externalCamera } }
                .build()
        }

        // Priority 2: Front-facing camera (tablets, phones)
        val frontCamera = cameras.find { safeLensFacing(it, CameraSelector.LENS_FACING_FRONT) }

        if (frontCamera != null) {
            Log.i(TAG, "[Camera] Using front-facing camera")
            return CameraSelector.DEFAULT_FRONT_CAMERA
        }

        // Priority 3: Back camera (last resort - not ideal for audience)
        val backCamera = cameras.find { safeLensFacing(it, CameraSelector.LENS_FACING_BACK) }

        if (backCamera != null) {
            Log.w(TAG, "[Camera] Using back camera as fallback (not ideal for audience sensing)")
            return CameraSelector.DEFAULT_BACK_CAMERA
        }

        // No recognized camera type. If every lens-facing query threw, this
        // is the RockChip HAL pattern — a silent fail-zero in the old code.
        // Surface it loudly so the health monitor can route through the
        // LENS_FACING_QUERY_FAILED retry path.
        if (lensQueryThrowCount > 0 && lensQueryThrowCount >= cameras.size * 2) {
            Log.e(
                TAG,
                "[Camera] ALL lens-facing queries threw ($lensQueryThrowCount throws across ${cameras.size} cameras) — " +
                        "RockChip-class HAL pattern. Last exception: ${lastLensQueryThrow?.message}",
                lastLensQueryThrow
            )
            cameraHealth.onStartFailure(
                lastLensQueryThrow,
                hint = CameraHealthMonitor.FailureClass.LENS_FACING_QUERY_FAILED
            )
        } else {
            Log.w(TAG, "[Camera] No suitable camera found among ${cameras.size} detected cameras")
        }
        return null
    }

    /**
     * Get current camera status for heartbeat reporting.
     * Reports camera availability, type (usb/internal/none), and count.
     *
     * Used by AudienceSensingService to include camera info in device heartbeat,
     * enabling fleet dashboard to show camera adoption rate.
     */
    fun getCameraStatus(): CameraStatus {
        val provider = cameraProvider
        if (provider == null) {
            return CameraStatus(available = false, type = "none", count = 0)
        }

        val cameras = provider.availableCameraInfos

        // Same RockChip-class HAL pattern: lensFacing throws on certain
        // builds. Don't swallow — log so the operator sees it. The status
        // is best-effort (used for heartbeat dashboards), so we still
        // return what we know rather than failing the whole call.
        val hasExternal = cameras.any { cameraInfo ->
            try {
                cameraInfo.lensFacing == CameraSelector.LENS_FACING_EXTERNAL
            } catch (e: Exception) {
                Log.w(TAG, "[Camera] getCameraStatus: lensFacing query (EXTERNAL) threw: ${e.message}")
                false
            }
        }

        val hasInternal = cameras.any { cameraInfo ->
            try {
                cameraInfo.lensFacing in listOf(
                    CameraSelector.LENS_FACING_FRONT,
                    CameraSelector.LENS_FACING_BACK
                )
            } catch (e: Exception) {
                Log.w(TAG, "[Camera] getCameraStatus: lensFacing query (FRONT/BACK) threw: ${e.message}")
                false
            }
        }

        val type = when {
            hasExternal -> "usb"
            hasInternal -> "internal"
            else -> "none"
        }

        return CameraStatus(
            available = cameras.isNotEmpty(),
            type = type,
            count = cameras.size
        )
    }

    /**
     * Pin the camera AE target-FPS range so the HAL produces frames near the
     * analysis rate instead of free-running toward 30fps. Mutates [builder] in
     * place via [Camera2Interop.Extender] and corrects the throttler's assumed
     * production rate. No-op (HAL default left intact) if the device reports no AE
     * ranges or the query fails — never pin an unsupported range.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    private fun applyAeFpsCap(
        builder: ImageAnalysis.Builder,
        provider: ProcessCameraProvider,
        selector: CameraSelector
    ) {
        // Uncapped HAL rate; the throttler is reset to this on every path that does
        // NOT pin a cap, so a rebind that falls back to HAL default never leaves the
        // throttler decimating from a stale cap set by a previous successful bind.
        val defaultCameraFps = SensingConfig.get().capture.cameraFps
        var capApplied = false
        try {
            val cameraInfo = selector.filter(provider.availableCameraInfos).firstOrNull()
            if (cameraInfo == null) {
                Log.w(TAG, "[Camera] No CameraInfo for AE FPS cap — leaving HAL default")
            } else {
                val ranges = Camera2CameraInfo.from(cameraInfo)
                    .getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                if (ranges == null || ranges.isEmpty()) {
                    Log.w(TAG, "[Camera] No AE target-FPS ranges reported — leaving HAL default")
                } else {
                    val available = ranges.map { AeFpsRangeSelector.FpsRange(it.lower, it.upper) }
                    val chosen = AeFpsRangeSelector.select(available, FrameThrottler.MAX_FPS)
                    if (chosen == null) {
                        Log.w(TAG, "[Camera] AE FPS-range selection empty — leaving HAL default")
                    } else {
                        Camera2Interop.Extender(builder).setCaptureRequestOption(
                            CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                            Range(chosen.min, chosen.max)
                        )
                        // Throttler decimates from the real (capped) rate, not 30.
                        frameThrottler.setCameraFps(chosen.max)
                        capApplied = true
                        Log.i(TAG, "[Camera] AE target-FPS pinned to ${chosen.min}-${chosen.max}fps " +
                                "(analyzer ceiling ${FrameThrottler.MAX_FPS}fps) — caps HAL production; " +
                                "available=${available.joinToString { "${it.min}-${it.max}" }}")
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "[Camera] AE FPS cap failed — leaving HAL default", e)
        }
        if (!capApplied) {
            // No cap pinned on this (re)bind: HAL runs at its default rate, so reset
            // the throttler to match instead of inheriting a previous bind's cap.
            frameThrottler.setCameraFps(defaultCameraFps)
        }
    }

    /**
     * Bind camera use cases - BOTH ImageAnalysis (ML Kit) and ImageCapture (Gemini demographics).
     * CameraX requires all use cases to be bound together.
     */
    private fun bindCameraUseCases(lifecycleOwner: LifecycleOwner) {
        val provider = cameraProvider
        if (provider == null) {
            Log.e(TAG, "[Camera] ERROR: Camera provider is null, cannot bind use cases")
            cameraHealth.onStartFailure(
                IllegalStateException("Camera provider is null"),
                hint = CameraHealthMonitor.FailureClass.PROVIDER_INIT_FAILED
            )
            return
        }

        Log.d(TAG, "[Camera] Binding camera use cases...")
        imageAnalysis?.clearAnalyzer()
        imageAnalysis = null
        _imageCapture = null
        _isCameraBound = false
        releaseBuffers()

        // Hardware sanity probe — empty CameraIdList means no physical camera
        // attached. Distinct from "RockChip lensFacing throws on every camera"
        // (LENS_FACING_QUERY_FAILED, which routes through selectBestCamera()).
        // We classify as NO_CAMERA_HARDWARE so the retry path doesn't loop
        // forever on devices that genuinely have no camera (e.g. plain Fire
        // TV without a USB cam). One retry is enough — onNoCameraAvailable
        // routes the service to audio-only.
        if (provider.availableCameraInfos.isEmpty()) {
            Log.w(TAG, "[Camera] No cameras available on this device — degrading to audio-only")
            _isCameraBound = false
            _imageCapture = null
            cameraHealth.onStartFailure(
                t = null,
                hint = CameraHealthMonitor.FailureClass.NO_CAMERA_HARDWARE
            )
            onNoCameraAvailable?.invoke()
            return
        }

        // Select best available camera (USB > Front > Back)
        val cameraSelector = selectBestCamera(provider)
        if (cameraSelector == null) {
            // selectBestCamera already routed any LENS_FACING_QUERY_FAILED
            // through cameraHealth.onStartFailure. If we got here without that
            // (i.e. zero throw count, just nobody matched the priority filters),
            // it's NO_CAMERA_SELECTED — no recognized lens facing.
            val snapshot = cameraHealth.snapshot()
            if (snapshot.lastFailureClass != CameraHealthMonitor.FailureClass.LENS_FACING_QUERY_FAILED) {
                Log.w(TAG, "[Camera] No camera available - switching to audio-only mode")
                cameraHealth.onStartFailure(
                    t = null,
                    hint = CameraHealthMonitor.FailureClass.NO_CAMERA_SELECTED
                )
            }
            _isCameraBound = false
            _imageCapture = null
            onNoCameraAvailable?.invoke()
            return
        }

        // Configure ImageAnalysis for ML Kit face detection (continuous)
        // Use device profile resolution for chipset-specific optimization
        val targetResolution = deviceProfile.resolution
        Log.i(TAG, "[Camera] Using device profile resolution: ${targetResolution.width}x${targetResolution.height} " +
                "(chipset: ${deviceProfile.chipsetVendor}, targetFps: ${deviceProfile.targetFps})")

        // CameraX 1.5.x: setTargetResolution(Size) is deprecated. The modern
        // replacement is ResolutionSelector + ResolutionStrategy. We pick
        // FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER so a device that does not
        // expose the exact requested size still binds to the nearest available
        // surface — same behavior the deprecated API provided.
        val analysisResolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    targetResolution,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val analysisBuilder = ImageAnalysis.Builder()
            .setResolutionSelector(analysisResolutionSelector)  // Adaptive resolution based on chipset
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)

        // Cap camera frame *production* near the analysis rate. Without a pinned AE
        // target-FPS range the HAL free-runs toward 30fps (~21 observed on the Tab
        // A9) for a ≤15fps analyzer; the surplus frames burn the camera-provider
        // core + CMA and are then dropped by the throttler, driving chronic zram
        // swap. This also corrects the throttler's production-rate assumption so its
        // decimation hits the target FPS.
        applyAeFpsCap(analysisBuilder, provider, cameraSelector)

        imageAnalysis = analysisBuilder
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    processImage(imageProxy)
                }
            }

        // Configure ImageCapture for Gemini demographics (on-demand)
        // This is used by FrameCaptureManager for frame capture
        val captureResolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(
                        FrameCaptureManager.MAX_IMAGE_WIDTH,
                        FrameCaptureManager.MAX_IMAGE_HEIGHT
                    ),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        _imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(captureResolutionSelector)
            .setJpegQuality(FrameCaptureManager.JPEG_QUALITY)
            .build()

        try {
            provider.unbindAll()

            // Bind BOTH ImageAnalysis and ImageCapture together
            provider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                imageAnalysis,
                _imageCapture  // Include ImageCapture for FrameCaptureManager
            )

            // Mark camera as bound
            _isCameraBound = true
            cameraHealth.onCameraBound()

            // Recreate shared frame buffers only after the new use cases are bound.
            initializeBuffers(targetResolution.width, targetResolution.height)

            Log.i(TAG, "[Camera] >>> CAMERA BOUND SUCCESSFULLY - ImageAnalysis (ML Kit) + ImageCapture (Gemini)")
            Log.d(TAG, "[Camera] ImageAnalysis: 640x480, STRATEGY_KEEP_ONLY_LATEST")
            Log.d(TAG, "[Camera] ImageCapture: ${FrameCaptureManager.MAX_IMAGE_WIDTH}x${FrameCaptureManager.MAX_IMAGE_HEIGHT}, JPEG quality=${FrameCaptureManager.JPEG_QUALITY}")

            // Notify listeners that camera is ready (for FrameCaptureManager)
            _imageCapture?.let { capture ->
                Log.i(TAG, "[Camera] Invoking onCameraReady callback")
                onCameraReady?.invoke(capture)
            }

        } catch (e: Exception) {
            // bindToLifecycle threw — most common silent fail-zero in the
            // Adam @ Focus Media incident class. Old code logged once and
            // gave up. Now: classify as BIND_FAILED, route through health
            // monitor for retry/escalation. handleCameraStartFailure will
            // either schedule a retry or fire onCameraStartFailed for the
            // service to escalate.
            _isCameraBound = false
            _imageCapture = null
            handleCameraStartFailure(
                lifecycleOwner = lifecycleOwner,
                error = e,
                hint = CameraHealthMonitor.FailureClass.BIND_FAILED,
                context = "ProcessCameraProvider.bindToLifecycle() threw"
            )
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processImage(imageProxy: ImageProxy) {
        if (!isRunning) {
            imageProxy.close()
            return
        }

        frameCount++

        // CRITICAL: Frame throttling - skip frames based on device capabilities
        if (!frameThrottler.shouldProcessFrame()) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val frameStartMs = System.currentTimeMillis()

        // Zero-allocation path: convert YUV→RGB into pre-allocated sharedBitmap
        val hasSharedBitmap = if (hasSharedBitmapConsumers()) {
            try {
                imageProxyToSharedBitmap(imageProxy)
            } catch (e: Exception) {
                Log.w(TAG, "[ZeroAlloc] Failed to convert image: ${e.message}")
                false
            }
        } else {
            false
        }

        // Always use fromMediaImage for ML Kit — native YUV path avoids the
        // double conversion (YUV→Bitmap→ML Kit internal) that triggers the
        // StreamingFormatChecker "inefficient Bitmap" warning.
        val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        // Pin all listeners to analysisExecutor to keep ML inference off the main thread.
        // Google's Task API addOnSuccessListener() without an executor defaults to
        // TaskExecutors.MAIN_THREAD — on the A9, this caused 275ms inference chains
        // blocking WebView rendering (6.96% jank, 5731 frames >200ms).
        faceDetector?.process(inputImage)
            ?.addOnSuccessListener(analysisExecutor) { faces ->
                if (!isRunning) {
                    return@addOnSuccessListener
                }
                processFaces(faces)

                // Process emotional engagement synchronously using pre-allocated buffers.
                // Synchronous ensures all processors finish reading sharedBitmap before
                // imageProxy.close() releases it for the next frame.
                if (hasSharedBitmap) {
                    frameBufferLeaseController.withLease {
                        val leasedBitmap = sharedBitmap
                        if (leasedBitmap == null || leasedBitmap.isRecycled) {
                            return@withLease
                        }

                        // Phase 1E: fill the RGB ByteBuffer ONCE per frame, shared
                        // across processEmotionalEngagementZeroAlloc + personDetector
                        // (was 2 fills/frame = 1.38 MB JNI memcopy waste). Only fill
                        // when at least one consumer is active; avoids the fill in
                        // configurations where both are disabled/null.
                        val sharedPoseBuffer = poseInputBuffer
                        val needsBuffer = emotionalEngagementEnabled || personDetector?.isReady() == true
                        val bufferReady = needsBuffer && sharedPoseBuffer != null && fillRgbByteBuffer()

                        if (emotionalEngagementEnabled) {
                            processEmotionalEngagementZeroAlloc(
                                leasedBitmap,
                                faces,
                                imageProxy.width,
                                imageProxy.height,
                                sharedPoseBuffer,
                                bufferReady
                            )
                        }

                        personDetector?.let { detector ->
                            if (detector.isReady()) {
                                // Zero-alloc: reuse the shared RGB buffer filled once
                                // above instead of a second fillRgbByteBuffer() call.
                                // Falls back to the bitmap path when the buffer isn't
                                // ready.
                                currentPersonCount = if (bufferReady && sharedPoseBuffer != null) {
                                    detector.processWithBuffer(sharedPoseBuffer, imageProxy.width, imageProxy.height)
                                } else {
                                    detector.process(leasedBitmap)
                                }
                            }
                        }

                        // Multi-class object detection — mirrors personDetector pattern.
                        // ObjectDetectionProcessor detects all 80 COCO classes; results are
                        // fed to VehicleFlowProcessor and QueueEstimationProcessor which
                        // aggregate per-window counts at each 10s flush.
                        objectDetector?.let { detector ->
                            if (detector.isReady()) {
                                val detectionResult = detector.detect(leasedBitmap)
                                vehicleFlowProcessor.recordSnapshot(detectionResult)
                                // QueueEstimationProcessor uses face count (from ML Kit) +
                                // dwell time as its snapshot inputs — vehicle frame result
                                // gives us the person count in the current frame for the queue.
                                val pedestrianCount = detectionResult.objectCounts["person"] ?: 0
                                val avgDwellMs = (trackedFaces.values.map { it.getDwellTimeSeconds() * 1000 }
                                    .takeIf { it.isNotEmpty() }?.average()?.toLong() ?: 0L)
                                queueEstimationProcessor.recordSnapshot(pedestrianCount, avgDwellMs)
                            }
                        }

                        // Route camera frame to VLM processor (if set).
                        // The VLM processor copies the bitmap internally and runs inference
                        // asynchronously — this callback returns immediately.
                        onBitmapAvailable?.invoke(leasedBitmap)
                    }
                }
            }
            ?.addOnFailureListener(analysisExecutor) { e ->
                Log.w(TAG, "[MLKit] Face detection failed: ${e.message}")
            }
            ?.addOnCompleteListener(analysisExecutor) {
                imageProxy.close()

                // Camera health heartbeat — prevents RecoveryLadder false-positive.
                // Also notifies CameraHealthMonitor so the OPEN state transitions
                // and `cap_face_detection` self-correction logic see real frames
                // arriving (the difference between "bind succeeded" and "frames
                // are flowing" — the latter is what the heartbeat must reflect).
                if (isRunning) {
                    lastFrameProcessedAtMs = SystemClock.elapsedRealtime()
                    cameraHealth.onFrameProcessed()
                    onFrameProcessed?.invoke()
                }

                // Adaptive FPS calibration (first N frames)
                val frameEndMs = System.currentTimeMillis()
                val frameTimeMs = frameEndMs - frameStartMs
                lastInferenceMs = frameTimeMs
                val smoothAlpha = SensingConfig.get().detection.smoothingAlpha.toDouble()
                rollingInferenceMs = if (rollingInferenceMs <= 0.0) {
                    frameTimeMs.toDouble()
                } else {
                    (rollingInferenceMs * smoothAlpha) + (frameTimeMs * (1.0 - smoothAlpha))
                }
                if (!isCalibrated && calibrationFrameCount < CALIBRATION_FRAMES) {
                    calibrationFrameCount++
                    calibrationTotalMs += frameTimeMs
                    if (calibrationFrameCount >= CALIBRATION_FRAMES) {
                        val avgMs = calibrationTotalMs.toDouble() / CALIBRATION_FRAMES
                        frameThrottler.autoTuneFromLatency(avgMs)
                        isCalibrated = true
                        Log.i(TAG, "[AdaptiveFPS] Calibrated: avgInference=${avgMs.toInt()}ms, " +
                                "newTargetFps=${frameThrottler.getTargetFps()}")
                    }
                }

                // Periodic frame processing stats
                val now = System.currentTimeMillis()
                if (now - lastFrameLogTime >= frameLogIntervalMs) {
                    val throttlerStats = frameThrottler.getStats()
                    val memInfo = resourceMonitor.getMemoryInfo()
                    Log.d(TAG, "[Analyzer] Stats: totalFrames=$frameCount, " +
                            "actualFps=${String.format("%.1f", throttlerStats.actualFps)}, " +
                            "targetFps=${throttlerStats.targetFps}, " +
                            "skipped=${throttlerStats.skippedFrames}, " +
                            "trackedFaces=${trackedFaces.size}, peak=$peakViewerCount, " +
                            "nativeHeap=${memInfo.nativeHeapMb}MB")
                    lastFrameLogTime = now
                    frameCount = 0
                    frameThrottler.resetStats()
                }
            }
    }

    /**
     * Zero-allocation emotional engagement processing.
     *
     * Runs SYNCHRONOUSLY on the analysisExecutor thread via ML Kit's
     * addOnSuccessListener(analysisExecutor) callback.
     * All processors read from sharedBitmap and pre-allocated crop bitmaps.
     * No new Bitmap/ByteBuffer/ByteArray allocations per frame.
     *
     * Thread safety: CameraX STRATEGY_KEEP_ONLY_LATEST ensures the next frame
     * won't arrive until imageProxy.close() is called in addOnCompleteListener,
     * which happens after this method returns.
     */
    private fun processEmotionalEngagementZeroAlloc(
        bitmap: Bitmap,
        faces: List<Face>,
        imageWidth: Int,
        imageHeight: Int,
        poseBuffer: ByteBuffer?,
        bufferReady: Boolean
    ) {
        try {
            // Process pose using pre-allocated ByteBuffer (zero-alloc path)
            val detCfg = SensingConfig.get().detection
            // poseBuffer and bufferReady are hoisted from the withLease call site
            // (Phase 1E): the buffer is filled ONCE per frame and shared across
            // processEmotionalEngagementZeroAlloc + personDetector. Pose AND
            // FaceLandmarker both consume it zero-alloc (ByteBufferImageBuilder
            // wraps it without copying or recycling). MediaPipe detect() is
            // synchronous and doesn't mutate the input, so the sequential readers
            // safely share one filled buffer.
            poseProcessor?.let { pose ->
                if (pose.isReady()) {
                    if (bufferReady && poseBuffer != null) {
                        // Build engagement context from the same frame's face list.
                        // screenEngaged uses the same head-yaw/pitch envelope that
                        // FaceLandmarker's HeadPose.isFacingScreen() applies (gazeCfg
                        // thresholds × 2), so a single face whose head rotation is
                        // within bounds means "user is stably looking at the screen".
                        // Pose's own facing-angle is intentionally NOT used here: it
                        // derives from shoulder geometry (a different signal source)
                        // and would couple two engagement gates to the same noisy
                        // primitive.
                        val gazeCfg = SensingConfig.get().gaze
                        val anyFaceLooking = faces.any { f ->
                            kotlin.math.abs(f.headEulerAngleY) < gazeCfg.headYawThreshold * 2 &&
                                kotlin.math.abs(f.headEulerAngleX) < gazeCfg.headPitchThreshold * 2
                        }
                        val poseContext = PoseEngagementProcessor.MovementContext(
                            faceCount = faces.size,
                            screenEngaged = anyFaceLooking
                        )
                        val poseMetrics = pose.processWithBuffer(
                            poseBuffer, imageWidth, imageHeight, poseContext
                        )
                        if (poseMetrics != null && poseMetrics.confidence > detCfg.minPoseBufferConfidence) {
                            synchronized(poseMetricsBuffer) {
                                poseMetricsBuffer.add(poseMetrics)
                                if (poseMetricsBuffer.size > detCfg.maxBufferedMetrics) poseMetricsBuffer.removeFirst()
                            }
                        }
                    }
                }
            }

            // Process emotions + gaze for each face — FaceLandmarker is the
            // SOLE source after Phase 3 cleanup. One forward pass produces
            // blendshape-derived emotion (FACS-explicit, no NEUTRAL bias)
            // AND iris-based gaze direction AND head-pose Euler angles.
            // When FaceLandmarker is unavailable (init failed, asset missing)
            // these buffers stay empty and aggregateEmotionalEngagement() falls
            // back to HeuristicEmotionalEngagementEstimator on the face list.
            if (faces.isNotEmpty()) {
                val flResult = if (bufferReady && poseBuffer != null) {
                    faceLandmarkerProcessor?.processWithBuffer(poseBuffer, imageWidth, imageHeight)
                } else {
                    faceLandmarkerProcessor?.process(bitmap)
                }
                val flUsable = flResult != null && flResult.faces.isNotEmpty()
                    && kotlin.math.abs(flResult.faces.size - faces.size) <= 1
                if (flUsable) {
                    val now = System.currentTimeMillis()
                    // Phase 1c collapse-point #1 fix: bind FaceLandmarker
                    // output to ML Kit's stable face track IDs via box-IoU
                    // match. Previously emotions/gaze were zipped by list
                    // index — fragile when the two detectors disagree on
                    // ordering, and the faceId was never carried forward at
                    // all so the per-(face × ad) accumulator (collapse
                    // point #3) had nothing to bucket on.
                    //
                    // The mapping is computed ONCE up-front so emotion and
                    // gaze paths agree on which FaceLandmarker face → which
                    // ML Kit faceId.
                    val flFaceIdMap = mutableMapOf<Int, Int>()  // FaceLandmarker index -> ML Kit faceId
                    flResult!!.faces.forEachIndexed { idx, flFace ->
                        val derivedBox = deriveBoundingBoxFromLandmarks(
                            flFace.landmarks, imageWidth, imageHeight
                        )
                        if (derivedBox != null) {
                            val matchedId = findOverlappingFace(derivedBox)
                            if (matchedId != null) {
                                flFaceIdMap[idx] = matchedId
                            } else if (idx < faces.size) {
                                // Fallback: bind to faces[idx]'s tracking
                                // ID when ML Kit has assigned one. Preserves
                                // backward-compat when IoU dedup hasn't
                                // recorded a tracked face for this frame yet.
                                faces[idx].trackingId?.let { flFaceIdMap[idx] = it }
                            }
                        }
                    }
                    val flEmotions = flResult.faces.mapIndexed { idx, face ->
                        val (positiveScore, negativeScore) = when (face.emotion.emotion) {
                            EmotionType.HAPPY, EmotionType.SURPRISED -> face.emotion.confidence to 0f
                            EmotionType.SAD, EmotionType.ANGRY, EmotionType.FEARFUL, EmotionType.DISGUSTED, EmotionType.CONTEMPT -> 0f to face.emotion.confidence
                            else -> 0f to 0f
                        }
                        EmotionMetrics(
                            timestamp = now,
                            dominantEmotion = face.emotion.emotion,
                            emotionConfidence = face.emotion.confidence,
                            happyScore = if (face.emotion.emotion == EmotionType.HAPPY) face.emotion.confidence else 0f,
                            surprisedScore = if (face.emotion.emotion == EmotionType.SURPRISED) face.emotion.confidence else 0f,
                            sadScore = if (face.emotion.emotion == EmotionType.SAD) face.emotion.confidence else 0f,
                            angryScore = if (face.emotion.emotion == EmotionType.ANGRY) face.emotion.confidence else 0f,
                            neutralScore = if (face.emotion.emotion == EmotionType.NEUTRAL) face.emotion.confidence else 0f,
                            isPositiveReaction = positiveScore > 0f,
                            isNegativeReaction = negativeScore > 0f,
                            // Engagement = blendshape activation magnitude
                            // (clamped 0..1). Strong emotion → high engagement,
                            // neutral → zero (matches legacy FER+ semantics).
                            emotionalEngagementScore = if (face.emotion.emotion == EmotionType.NEUTRAL) {
                                0f
                            } else {
                                face.emotion.confidence.coerceIn(0f, 1f)
                            },
                            confidence = face.emotion.confidence,
                            faceId = flFaceIdMap[idx],
                        )
                    }
                    synchronized(emotionMetricsBuffer) {
                        emotionMetricsBuffer.addAll(flEmotions)
                        while (emotionMetricsBuffer.size > detCfg.maxBufferedMetrics) emotionMetricsBuffer.removeFirst()
                    }

                    // Gaze: FaceLandmarker iris landmarks (468-477 in the 478-point
                    // mesh) drive both gaze direction and the 3×3 focus-region zone
                    // for the creativeZones heatmap. Head pose comes from the
                    // facial transformation matrix decomposition (true Euler angles).
                    val flGazeMetrics = flResult.faces.mapIndexed { idx, face ->
                        val gaze = face.gazeDirection
                        val pose = face.headPoseDeg
                        // 3x3 focus region: dx ∈ [-1,1] → col ∈ {0,1,2}; same for dy → row.
                        // Region index = row * 3 + col, with center = 4.
                        val col = when {
                            gaze.dx < -0.33f -> 0
                            gaze.dx > 0.33f -> 2
                            else -> 1
                        }
                        val row = when {
                            gaze.dy < -0.33f -> 0
                            gaze.dy > 0.33f -> 2
                            else -> 1
                        }
                        val focusRegion = row * 3 + col
                        // isLookingAtScreen mirrors legacy thresholds (yaw < 20°, pitch < 15°)
                        val isLooking = pose.isFacingScreen()
                        // Iris-magnitude-based stability proxy: small iris deflection
                        // (looking forward) → high stability; large deflection → erratic.
                        val gazeStability = (1f - gaze.magnitude()).coerceIn(0f, 1f)
                        // Attention score: gazeStability if looking at screen, else 0.
                        val attentionScore = if (isLooking) gazeStability else 0f
                        GazeMetrics(
                            timestamp = now,
                            focusRegion = focusRegion,
                            isLookingAtScreen = isLooking,
                            gazeStability = gazeStability,
                            headRotationX = pose.pitchDeg,
                            headRotationY = pose.yawDeg,
                            headRotationZ = pose.rollDeg,
                            attentionScore = attentionScore,
                            confidence = 1f,
                            faceId = flFaceIdMap[idx],
                        )
                    }
                    synchronized(gazeMetricsBuffer) {
                        gazeMetricsBuffer.addAll(flGazeMetrics)
                        while (gazeMetricsBuffer.size > detCfg.maxBufferedMetrics) gazeMetricsBuffer.removeFirst()
                    }
                    // Update zone-dwell heatmap from the dominant face (first detected)
                    // — the same face whose attentionScore drives the engagement composite.
                    flGazeMetrics.firstOrNull()?.let { primary ->
                        if (primary.isLookingAtScreen) updateZoneDwell(primary.focusRegion)
                    }

                    // Phase 1c collapse-point #3 fix: per-(face × ad)
                    // accumulation. Walk the paired emotion+gaze samples
                    // and bucket each into the per-ad map. Frames captured
                    // while no ad is playing are skipped inside
                    // accumulatePerFaceSample (currentAdId == null guard).
                    // Frame interval: time since last sample, capped at
                    // PER_FACE_MAX_FRAME_INTERVAL_MS to keep long pauses
                    // from inflating dwell time.
                    val lastSampleAt = lastPerFaceSampleMs
                    val frameIntervalMs = if (lastSampleAt > 0L && now > lastSampleAt) {
                        (now - lastSampleAt).coerceAtMost(PER_FACE_MAX_FRAME_INTERVAL_MS)
                    } else {
                        EmotionalEngagementConfig().emotionProcessingIntervalMs
                    }
                    lastPerFaceSampleMs = now
                    val pairCount = minOf(flEmotions.size, flGazeMetrics.size)
                    for (i in 0 until pairCount) {
                        val em = flEmotions[i]
                        val gz = flGazeMetrics[i]
                        val faceId = em.faceId ?: gz.faceId ?: continue
                        accumulatePerFaceSample(
                            faceId = faceId,
                            nowMs = now,
                            attentionScore = gz.attentionScore,
                            gazeAttentionScore = gz.attentionScore,
                            isLookingAtScreen = gz.isLookingAtScreen,
                            focusRegion = gz.focusRegion,
                            emotion = em.dominantEmotion,
                            emotionConfidence = em.emotionConfidence,
                            frameIntervalMs = frameIntervalMs,
                        )
                    }
                }

                // On-device age/gender using pre-allocated crop bitmap (zero-alloc)
                //
                // Phase 1c collapse-point #4 fix: pass the ML Kit face track
                // ID through to `accumulateWithFaceId`. The histogram
                // counters still update (preserves existing aggregate emit);
                // the per-face overlay map records the latest result per
                // faceId so drainPerAdAttention() can thread age/gender
                // into the matching PerFaceAttention bucket.
                if (ageGenderEnabled) {
                    val agCrop = ageGenderCropBitmap
                    ageGenderProcessor?.let { agp ->
                        if (agCrop != null) {
                            for (face in faces) {
                                try {
                                    val result = agp.classifyWithPreallocatedCrop(
                                        bitmap, face.boundingBox, agCrop
                                    )
                                    if (result != null) {
                                        val rawId = face.trackingId
                                        val boundingBox = FaceRect(
                                            left = face.boundingBox.left,
                                            top = face.boundingBox.top,
                                            right = face.boundingBox.right,
                                            bottom = face.boundingBox.bottom,
                                        )
                                        val effId = findOverlappingFace(boundingBox) ?: rawId
                                        if (effId != null) {
                                            agp.accumulateWithFaceId(effId, result)
                                        } else {
                                            agp.accumulate(result)
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.w(TAG, "[AgeGender] Classification failed: ${e.message}")
                                }
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "[EmotionalEngagement] Processing error: ${e.message}")
        }
    }

    /**
     * Aggregate and return current emotional engagement metrics.
     * Called from AudienceSensingService during aggregation window.
     */
    fun aggregateEmotionalEngagement(): EmotionalEngagement? {
        if (!emotionalEngagementEnabled) return null

        val poseSnapshots: List<PoseMetrics>
        val emotionSnapshots: List<EmotionMetrics>
        val gazeSnapshots: List<GazeMetrics>

        synchronized(poseMetricsBuffer) {
            poseSnapshots = poseMetricsBuffer.toList()
            poseMetricsBuffer.clear()
        }
        synchronized(emotionMetricsBuffer) {
            emotionSnapshots = emotionMetricsBuffer.toList()
            emotionMetricsBuffer.clear()
        }
        synchronized(gazeMetricsBuffer) {
            gazeSnapshots = gazeMetricsBuffer.toList()
            gazeMetricsBuffer.clear()
        }

        if (poseSnapshots.isEmpty() && emotionSnapshots.isEmpty() && gazeSnapshots.isEmpty()) {
            val heuristicFallback = HeuristicEmotionalEngagementEstimator.fromFaces(
                faces = trackedFaces.values.toList()
            )
            if (heuristicFallback != null) {
                _currentEmotionalEngagement.value = heuristicFallback
                onEmotionalEngagementReady?.invoke(heuristicFallback)
                Log.w(
                    TAG,
                    "[EmotionalEngagement] Using face-derived fallback: " +
                        "faces=${trackedFaces.size}, overall=${String.format("%.2f", heuristicFallback.overallEngagementScore)}, " +
                        "reaction=${heuristicFallback.audienceReaction}"
                )
                return heuristicFallback
            }
            return null
        }

        // Aggregate pose metrics
        @Suppress("DEPRECATION")
        val aggregatedPose = if (poseSnapshots.isNotEmpty()) {
            AggregatedPoseMetrics(
                windowStart = poseSnapshots.minOfOrNull { it.timestamp } ?: 0,
                windowEnd = poseSnapshots.maxOfOrNull { it.timestamp } ?: 0,
                sampleCount = poseSnapshots.size,
                avgFacingAngle = poseSnapshots.map { it.facingAngle }.average().toFloat(),
                facingScreenPct = poseSnapshots.count { it.isFacingScreen }.toFloat() / poseSnapshots.size,
                leaningInCount = poseSnapshots.count { it.leanDirection == LeanDirection.LEANING_IN },
                leaningBackCount = poseSnapshots.count { it.leanDirection == LeanDirection.LEANING_BACK },
                avgLeanMagnitude = poseSnapshots.map { it.leanMagnitude }.average().toFloat(),
                // One pass over snapshots → Map<MovementState, Int>. Adding a new movement
                // state (e.g. when depth-aware tracking lands and APPROACHING / DEPARTING
                // start emitting) requires only adding to the MovementState enum — the
                // distribution flows through automatically.
                movementDistribution = poseSnapshots.groupingBy { it.movementState }.eachCount(),
                avgMovementSpeed = poseSnapshots.map { it.movementSpeed }.average().toFloat(),
                bodyEngagementScore = poseSnapshots.map { it.bodyEngagementScore }.average().toFloat()
            )
        } else null

        // Aggregate emotion metrics
        val aggregatedEmotion = if (emotionSnapshots.isNotEmpty()) {
            val emotionCounts = emotionSnapshots.groupBy { it.dominantEmotion }.mapValues { it.value.size }
            val dominantEmotion = emotionCounts.maxByOrNull { it.value }?.key ?: EmotionType.NEUTRAL

            AggregatedEmotionMetrics(
                windowStart = emotionSnapshots.minOfOrNull { it.timestamp } ?: 0,
                windowEnd = emotionSnapshots.maxOfOrNull { it.timestamp } ?: 0,
                sampleCount = emotionSnapshots.size,
                dominantEmotion = dominantEmotion,
                positiveReactionCount = emotionSnapshots.count { it.isPositiveReaction },
                neutralCount = emotionSnapshots.count { it.dominantEmotion == EmotionType.NEUTRAL },
                negativeReactionCount = emotionSnapshots.count { it.isNegativeReaction },
                happyCount = emotionSnapshots.count { it.dominantEmotion == EmotionType.HAPPY },
                surprisedCount = emotionSnapshots.count { it.dominantEmotion == EmotionType.SURPRISED },
                confusedCount = emotionSnapshots.count { it.dominantEmotion == EmotionType.CONFUSED },
                avgEmotionIntensity = emotionSnapshots.map { it.emotionConfidence }.average().toFloat(),
                emotionalEngagementScore = emotionSnapshots.map { it.emotionalEngagementScore }.average().toFloat()
            )
        } else null

        // Aggregate gaze metrics — pure helper (no processor instance needed).
        val aggregatedGaze = if (gazeSnapshots.isNotEmpty()) aggregateGazeSnapshots(gazeSnapshots) else null

        // Calculate overall engagement score
        val config = EmotionalEngagementConfig()
        var totalWeight = 0f
        var weightedScore = 0f

        aggregatedPose?.let {
            weightedScore += it.bodyEngagementScore * config.poseWeight
            totalWeight += config.poseWeight
        }
        aggregatedEmotion?.let {
            weightedScore += it.emotionalEngagementScore * config.emotionWeight
            totalWeight += config.emotionWeight
        }
        aggregatedGaze?.let {
            weightedScore += it.gazeAttentionScore * config.gazeWeight
            totalWeight += config.gazeWeight
        }

        val overallScore = if (totalWeight > 0) weightedScore / totalWeight else 0f

        // Determine audience reaction (2026-05-08 reform — soak findings):
        // The OLD ladder treated DISINTERESTED as the catch-all for ANY
        // overallScore < neutralThreshold. That meant a relaxed couch viewer
        // (neutral expression, low body movement, gaze toward something) was
        // labeled DISINTERESTED — a negative-valence reaction — when their
        // actual emotional state was neutral. Tab S11 soak: 62/66 windows
        // tagged DISINTERESTED with dominant emotion NEUTRAL@0.93. Catch-all
        // semantics conflated "low engagement intensity" with "negative
        // valence". Two separate concepts.
        //
        // New L9 ladder separates intensity (engagement composite) from
        // valence (FER classifier). Negative-valence is now an early branch
        // (independent of intensity) and the catch-all for low-intensity is
        // NEUTRAL — the honest default when emotion is neither positive nor
        // negative.
        //
        // Also fixes the precedence bug at the old line: `x ?: 0 > y ?: 0`
        // parses as `x ?: (0 > (y ?: 0))` because elvis is lower precedence
        // than comparison. Result: the NEGATIVE branch never fired correctly.
        // Explicit parens fix the parse.
        //
        // FOLLOW-UP (cosmic-brewing-bear plan B9):
        //   - MediaPipe FaceLandmarker v2 emits 52 blendshape coefficients
        //     per face. Replace the 7-class FER classifier with blendshape-
        //     derived emotion (smile_L/R, brow_lower, mouth_frown). Per-face
        //     classification, aggregated to {pos, neu, neg} distribution
        //     with face-seconds weighting. min-N ≥ 5 face-seconds to display.
        //
        // FOLLOW-UP (cosmic-brewing-bear plan task #33):
        //   - When ANY HAPPY frames clear HAPPY_FRAME_CONFIDENCE_FLOOR but
        //     the strong-POSITIVE (HIGHLY_ENGAGED) intensity isn't met,
        //     promote NEUTRAL → INTERESTED. Mid-band positive valence is
        //     meaningful advertiser-audit signal ("audience leaning
        //     positive but not fully engaged") and should not collapse
        //     into NEUTRAL. Ladder lives in computeAudienceReaction() so
        //     the unit tests can exercise it without spinning up a
        //     CameraX/Android stack.
        val detCfg = SensingConfig.get().detection
        val negativeCount = aggregatedEmotion?.negativeReactionCount ?: 0
        val positiveCount = aggregatedEmotion?.positiveReactionCount ?: 0
        val isNegativeValence = negativeCount > positiveCount
        val isPositiveValence = positiveCount > 0 && positiveCount > negativeCount
        // Tally HAPPY frames whose per-face emotionConfidence cleared the
        // confidence floor. Fleeting <0.15 twitches stay out of the
        // promotion path.
        val happyFrameCount = emotionSnapshots.count {
            it.dominantEmotion == EmotionType.HAPPY &&
                it.emotionConfidence > HAPPY_FRAME_CONFIDENCE_FLOOR
        }
        val audienceReaction = computeAudienceReaction(
            overallScore = overallScore,
            highlyEngagedThreshold = detCfg.highlyEngagedThreshold,
            interestedThreshold = detCfg.interestedThreshold,
            neutralThreshold = detCfg.neutralThreshold,
            isNegativeValence = isNegativeValence,
            isPositiveValence = isPositiveValence,
            happyFrameCount = happyFrameCount
        )

        val engagement = EmotionalEngagement(
            pose = aggregatedPose,
            emotion = aggregatedEmotion,
            gaze = aggregatedGaze,
            overallEngagementScore = overallScore,
            audienceReaction = audienceReaction,
            isHighEngagement = overallScore > config.highEngagementThreshold,
            isNegativeReaction = audienceReaction == AudienceReaction.NEGATIVE,
            confidence = totalWeight / (config.poseWeight + config.emotionWeight + config.gazeWeight)
        )

        _currentEmotionalEngagement.value = engagement
        onEmotionalEngagementReady?.invoke(engagement)

        Log.i(TAG, "[EmotionalEngagement] Aggregated: overall=${String.format("%.2f", overallScore)}, " +
                "reaction=$audienceReaction, pose=${aggregatedPose != null}, emotion=${aggregatedEmotion != null}, gaze=${aggregatedGaze != null}")

        return engagement
    }

    // ============================================================
    // Box geometry helpers — delegate to BoxTracking (shared with
    // VehicleFlowProcessor / QueueEstimationProcessor). FaceRect is
    // pixel-space; BoxTracking.BoxRect is Float-based, so we convert.
    // ============================================================

    private fun FaceRect.toBoxRect() = BoxTracking.BoxRect(
        left = left.toFloat(), top = top.toFloat(),
        right = right.toFloat(), bottom = bottom.toFloat()
    )

    /**
     * Calculate Intersection over Union (IoU) for two bounding boxes.
     * Delegates to [BoxTracking.iou] — shared kernel with object counting.
     */
    private fun calculateIoU(box1: FaceRect, box2: FaceRect): Float =
        BoxTracking.iou(box1.toBoxRect(), box2.toBoxRect())

    /**
     * Calculate centroid distance between two bounding boxes.
     * Delegates to [BoxTracking.centroidDistance].
     */
    private fun calculateCentroidDistance(box1: FaceRect, box2: FaceRect): Float =
        BoxTracking.centroidDistance(box1.toBoxRect(), box2.toBoxRect())

    /**
     * Calculate average face size (for relative distance thresholding).
     * Delegates to [BoxTracking.averageSize].
     */
    private fun averageFaceSize(box1: FaceRect, box2: FaceRect): Float =
        BoxTracking.averageSize(box1.toBoxRect(), box2.toBoxRect())

    /**
     * Derive a pixel-space bounding box from a FaceLandmarker face's
     * normalized landmarks (x,y in [0..1]). Phase 1c uses this to bind
     * FaceLandmarker output back to ML Kit's stable track IDs via
     * [findOverlappingFace] — the existing IoU dedup that operates on
     * pixel-space FaceRect.
     *
     * Returns null when the landmarks list is empty or image dims are bad.
     */
    internal fun deriveBoundingBoxFromLandmarks(
        landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int,
    ): FaceRect? {
        if (landmarks.isEmpty() || imageWidth <= 0 || imageHeight <= 0) return null
        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (lm in landmarks) {
            if (lm.x() < minX) minX = lm.x()
            if (lm.y() < minY) minY = lm.y()
            if (lm.x() > maxX) maxX = lm.x()
            if (lm.y() > maxY) maxY = lm.y()
        }
        if (!minX.isFinite() || !maxX.isFinite()) return null
        return FaceRect(
            left = (minX * imageWidth).toInt().coerceAtLeast(0),
            top = (minY * imageHeight).toInt().coerceAtLeast(0),
            right = (maxX * imageWidth).toInt().coerceAtMost(imageWidth),
            bottom = (maxY * imageHeight).toInt().coerceAtMost(imageHeight),
        )
    }

    // ============================================================
    // Phase 1c — per-(face × ad) attention accumulator API
    // ============================================================

    /**
     * Update the currently-playing ad context.
     *
     * Called from [AudienceSensingService] when the
     * [ContentStateProvider]'s reported ad changes. Subsequent per-frame
     * samples are bucketed under this (adId, faceTrackId) until the ad
     * changes again. Pass `null` adId to indicate idle / no ad playing —
     * frames captured during idle are not accumulated into the per-ad map
     * (only ad-playback windows produce per-face observations).
     */
    fun setCurrentAd(adId: String?, impressionId: String?) {
        setCurrentAd(adId, impressionId, source = null)
    }

    /**
     * 3-arg overload: also records `source` (ima_programmatic / self_promo /
     * sponsored / default_stream) into `adIdToSource` so drainPerAdAttention
     * can stamp each per-face row with the creative_source it actually came
     * from. Existing 2-arg callers default to source=null (back-compat).
     */
    fun setCurrentAd(adId: String?, impressionId: String?, source: String?) {
        currentAdId = adId
        currentImpressionId = impressionId
        currentCreativeSource = source
        if (adId != null && source != null) {
            // Record once per (adId, source). If the same adId reappears with
            // a different source, last write wins — which is correct since a
            // single adId belongs to one source in practice.
            adIdToSource[adId] = source
        }
    }

    /** Reads the currently-set ad for tests / introspection. */
    internal fun getCurrentAdIdForTesting(): String? = currentAdId

    /** Reads the currently-set creative source for tests / introspection. */
    internal fun getCurrentCreativeSourceForTesting(): String? = currentCreativeSource

    /**
     * Look up the source recorded at setCurrentAd time for a given adId.
     * Returns null when no source was provided (legacy 2-arg setCurrentAd
     * caller) or the adId is unknown. Consumed by AudienceSensingService
     * at drain time to stamp creative_source onto each emitted per_face row.
     */
    fun getSourceFor(adId: String): String? = adIdToSource[adId]

    /**
     * Drain accumulated per-(adId, faceId) attention buckets. Called from
     * [AudienceSensingService] at window flush time. Returns the
     * finalize()'d snapshot AND clears internal state so the next window
     * starts fresh. Idempotent — calling twice in a row returns
     * accumulations from the second window only (empty if no frames
     * accumulated in between).
     *
     * Map layout: outer key = adId, inner key = ML Kit face track ID.
     */
    fun drainPerAdAttention(): Map<String, Map<Int, PerFaceAttention>> {
        // Phase 1c — pre-fetch the per-face age/gender overlay so the
        // drain populates ageBucket / gender on each bucket before the
        // window resets. Snapshot ONCE — the underlying map is concurrent.
        val ageGenderOverlay: Map<Int, AgeGenderResult> =
            ageGenderProcessor?.getPerFaceLatestSnapshot() ?: emptyMap()

        val snapshot = mutableMapOf<String, Map<Int, PerFaceAttention>>()
        // Snapshot keys first to avoid holding the iterator under modification
        val adKeys = perAdAttention.keys.toList()
        for (adKey in adKeys) {
            val inner = perAdAttention.remove(adKey) ?: continue
            val finalized = mutableMapOf<Int, PerFaceAttention>()
            for ((faceId, attention) in inner) {
                ageGenderOverlay[faceId]?.let { ag ->
                    attention.ageBucket = ag.ageRange
                    attention.gender = ag.gender
                }
                finalized[faceId] = attention.finalize()
            }
            if (finalized.isNotEmpty()) snapshot[adKey] = finalized
        }
        // Reset frame-interval anchor — next window starts a fresh cadence.
        lastPerFaceSampleMs = 0L
        // adIdToSource is NOT cleared here — drainPerAdAttention returns
        // the per-face map and the caller looks up creative_source per adId
        // via getSourceFor() AFTER drain returns. Lifetime of the source
        // map is bounded by unique creative count per device (~10-100), so
        // memory pressure is negligible. If we cleared here, the very next
        // getSourceFor() call would return null.
        return snapshot
    }

    /**
     * Accumulate a per-frame sample into the (currentAdId, faceId) bucket.
     *
     * No-op when [currentAdId] is null (no ad playing during this frame) —
     * idle frames don't contribute to per-ad attention. Per-face samples
     * arrive from the per-frame [processEmotionalEngagementZeroAlloc] path
     * (collapse points #1 and #2 in P0a audit). The `frameIntervalMs`
     * value translates each sample into dwell-seconds and gaze-seconds.
     */
    internal fun accumulatePerFaceSample(
        faceId: Int,
        nowMs: Long,
        attentionScore: Float,
        gazeAttentionScore: Float,
        isLookingAtScreen: Boolean,
        focusRegion: Int,
        emotion: EmotionType,
        emotionConfidence: Float,
        frameIntervalMs: Long,
    ) {
        val adId = currentAdId ?: return
        val inner = perAdAttention.computeIfAbsent(adId) {
            java.util.concurrent.ConcurrentHashMap()
        }
        val bucket = inner.computeIfAbsent(faceId) { PerFaceAttention() }
        synchronized(bucket) {
            bucket.addSample(
                nowMs = nowMs,
                attentionScore = attentionScore,
                gazeAttentionScore = gazeAttentionScore,
                isLookingAtScreen = isLookingAtScreen,
                focusRegion = focusRegion,
                emotion = emotion,
                emotionConfidence = emotionConfidence,
                frameIntervalMs = frameIntervalMs,
            )
        }
    }

    /**
     * Test-only / introspection: returns the count of (adId, faceId)
     * buckets currently accumulated. Production code MUST use
     * [drainPerAdAttention].
     */
    internal fun perAdAttentionBucketCountForTesting(): Int {
        var n = 0
        for ((_, inner) in perAdAttention) n += inner.size
        return n
    }

    /**
     * Test-only: returns a live reference to the inner per-face map for
     * the given ad. Tests use this to assert intermediate accumulator
     * state without draining the production map.
     */
    internal fun peekPerFaceMapForTesting(adId: String): Map<Int, PerFaceAttention>? {
        return perAdAttention[adId]
    }

    /**
     * Find existing tracked face that matches new bounding box.
     * Uses two strategies:
     * 1. IoU overlap > 30% (catches similar position faces)
     * 2. Centroid distance < 50% of average face size (catches rotated faces)
     * This prevents the same face from being counted multiple times
     * when ML Kit assigns different tracking IDs due to head movement.
     */
    private fun findOverlappingFace(newBox: FaceRect): Int? {
        val detCfg = SensingConfig.get().detection
        return trackedFaces.entries.find { (_, existing) ->
            val iou = calculateIoU(existing.boundingBox, newBox)
            val centroidDist = calculateCentroidDistance(existing.boundingBox, newBox)
            val avgSize = averageFaceSize(existing.boundingBox, newBox)
            iou > detCfg.iouThreshold || (centroidDist < avgSize * detCfg.centroidDistanceThreshold)
        }?.key
    }

    private fun processFaces(faces: List<Face>) {
        val currentTime = System.currentTimeMillis()
        val currentFaceIds = mutableSetOf<Int>()
        val previousFaceCount = trackedFaces.size

        for (face in faces) {
            // Skip faces without tracking ID
            if (face.trackingId == null) continue

            val faceId = face.trackingId!!
            val boundingBox = FaceRect(
                left = face.boundingBox.left,
                top = face.boundingBox.top,
                right = face.boundingBox.right,
                bottom = face.boundingBox.bottom
            )

            // Check if this face overlaps with an existing tracked face
            // This prevents counting the same person multiple times when
            // ML Kit assigns different tracking IDs due to head rotation
            val overlappingId = findOverlappingFace(boundingBox)
            val effectiveId = overlappingId ?: faceId

            currentFaceIds.add(effectiveId)

            val existingFace = trackedFaces[effectiveId]
            val isNewFace = existingFace == null && overlappingId == null

            val detectedFace = DetectedFace(
                id = effectiveId,
                boundingBox = boundingBox,
                headEulerAngleX = face.headEulerAngleX,
                headEulerAngleY = face.headEulerAngleY,
                headEulerAngleZ = face.headEulerAngleZ,
                smilingProbability = face.smilingProbability,
                leftEyeOpenProbability = face.leftEyeOpenProbability,
                rightEyeOpenProbability = face.rightEyeOpenProbability,
                estimatedAge = null,  // Would need separate ML model
                estimatedGender = null,  // Would need separate ML model
                firstSeenTimestamp = existingFace?.firstSeenTimestamp ?: currentTime,
                lastSeenTimestamp = currentTime
            )

            trackedFaces[effectiveId] = detectedFace

            // Log new faces with attention score
            if (isNewFace) {
                val attention = detectedFace.calculateAttentionScore()
                Log.d(TAG, "[MLKit] New face detected: id=$effectiveId, attention=${String.format("%.2f", attention)}, " +
                        "yaw=${String.format("%.1f", face.headEulerAngleY)}°")
            } else if (overlappingId != null && overlappingId != faceId) {
                // Log when we merge a new tracking ID with existing face
                Log.d(TAG, "[MLKit] Merged tracking ID $faceId → $overlappingId (IoU dedup)")
            }
        }

        // Remove faces that haven't been seen for a while (left the frame)
        val staleThreshold = SensingConfig.get().detection.staleFaceRemovalMs
        val removedCount = trackedFaces.entries.count { (id, face) ->
            !currentFaceIds.contains(id) && (currentTime - face.lastSeenTimestamp) > staleThreshold
        }
        trackedFaces.entries.removeIf { (id, face) ->
            !currentFaceIds.contains(id) && (currentTime - face.lastSeenTimestamp) > staleThreshold
        }

        if (removedCount > 0) {
            Log.d(TAG, "[MLKit] Removed $removedCount stale face(s) - now tracking ${trackedFaces.size}")
        }

        // Cap tracked faces to prevent unbounded growth
        if (trackedFaces.size > MAX_TRACKED_FACES) {
            val excess = trackedFaces.size - MAX_TRACKED_FACES
            val oldestIds = trackedFaces.entries
                .sortedBy { it.value.lastSeenTimestamp }
                .take(excess)
                .map { it.key }
            oldestIds.forEach { trackedFaces.remove(it) }
            Log.w(TAG, "[MLKit] Evicted $excess oldest face(s) to cap at $MAX_TRACKED_FACES")
        }

        // Update peak viewer count
        if (trackedFaces.size > peakViewerCount) {
            peakViewerCount = trackedFaces.size
            Log.d(TAG, "[MLKit] New peak viewer count: $peakViewerCount")
        }

        // Log significant changes in face count
        if (trackedFaces.size != previousFaceCount && trackedFaces.size > 0) {
            Log.d(TAG, "[MLKit] Face count changed: $previousFaceCount → ${trackedFaces.size}")
        }

        // Update current snapshot
        updateSnapshot()

        // Notify listeners (for FrameCaptureManager demographics trigger)
        if (trackedFaces.isNotEmpty()) {
            onFacesDetected?.invoke(trackedFaces.size)
        }
    }

    private fun updateSnapshot() {
        val activeFaces = trackedFaces.values.toList()

        val avgAttention = if (activeFaces.isNotEmpty()) {
            activeFaces.map { it.calculateAttentionScore() }.average().toFloat()
        } else {
            0f
        }

        val dwellTimes = activeFaces.map { it.getDwellTimeSeconds() }
        val dwellMetrics = if (dwellTimes.isNotEmpty()) {
            DwellTimeMetrics(
                averageSeconds = dwellTimes.average().toFloat(),
                maxSeconds = dwellTimes.maxOrNull() ?: 0f,
                minSeconds = dwellTimes.minOrNull() ?: 0f,
                totalViewerSeconds = dwellTimes.sum()
            )
        } else {
            DwellTimeMetrics()
        }

        _currentSnapshot.value = AudienceSnapshot(
            timestamp = System.currentTimeMillis(),
            viewerCount = activeFaces.size,
            personCount = currentPersonCount,
            attentionScore = avgAttention,
            demographics = Demographics(),  // Would need additional ML models
            dwellTime = dwellMetrics,
            environment = sensorCollector.getEnvironmentMetrics()
        )
    }

    /**
     * Get current viewer count.
     */
    fun getCurrentViewerCount(): Int = trackedFaces.size

    /**
     * Get current person count from the local TFLite detector.
     */
    fun getCurrentPersonCount(): Int = currentPersonCount

    /**
     * Get vehicle flow metrics from the live VehicleFlowProcessor window.
     * Returns null when object detection is not active (model absent or disabled).
     */
    fun getVehicleFlowMetrics(): VehicleFlowProcessor.FlowMetrics? =
        if (objectDetector?.isReady() == true) vehicleFlowProcessor.getFlowMetrics() else null

    /**
     * Get queue estimation metrics from the live QueueEstimationProcessor window.
     * Returns null when object detection is not active (model absent or disabled).
     */
    fun getQueueMetrics(): QueueEstimationProcessor.QueueMetrics? =
        if (objectDetector?.isReady() == true) queueEstimationProcessor.getQueueMetrics() else null

    /**
     * Get current attention score (0.0 to 1.0).
     */
    fun getCurrentAttentionScore(): Float = _currentSnapshot.value.attentionScore

    /**
     * Get current runtime snapshot for telemetry and adaptive capture decisions.
     */
    fun getRuntimeSnapshot(): AnalyzerRuntimeSnapshot {
        val throttlerStats = frameThrottler.getStats()
        return AnalyzerRuntimeSnapshot(
            actualFps = throttlerStats.actualFps,
            targetFps = throttlerStats.targetFps,
            skippedFrames = throttlerStats.skippedFrames,
            skipRatio = throttlerStats.skipRatio,
            avgInferenceMs = rollingInferenceMs,
            lastInferenceMs = lastInferenceMs,
            memoryInfo = resourceMonitor.getMemoryInfo()
        )
    }

    /**
     * Check if audience analysis is currently running.
     */
    fun isAnalyzing(): Boolean = isRunning

    /**
     * Get the age/gender processor for payload aggregation.
     */
    fun getAgeGenderProcessor(): AgeGenderProcessor? = ageGenderProcessor

    /**
     * Whether the FaceLandmarker (emotion + gaze + head-pose source) is
     * currently loaded. Used by AudienceSensingService to gate the
     * `creativeZones` heatmap emission.
     */
    fun isFaceLandmarkerActive(): Boolean = faceLandmarkerProcessor != null

    /**
     * Snapshot the per-zone dwell-time distribution for the current report
     * window. Zones are 0..8 indices into the 3×3 viewport grid (4 == center).
     * Returns an empty map when no gaze samples accumulated since the last
     * reset. Zone bookkeeping is updated by processEmotionalEngagementZeroAlloc
     * from each frame's FaceLandmarker focusRegion.
     *
     * Replaces the legacy GazeTrackingProcessor.getZoneDistribution().
     */
    fun getGazeZoneDistribution(): Map<Int, Float> {
        synchronized(zoneDwellMs) {
            // Flush the active zone's pending dwell so the snapshot reflects
            // ms accumulated up to "now" rather than only since the last
            // updateZoneDwell() call.
            val now = System.currentTimeMillis()
            if (lastZoneChangeMs > 0 && currentZone in 0..8) {
                val pending = (now - lastZoneChangeMs).coerceAtMost(10_000L)
                zoneDwellMs[currentZone] += pending
                lastZoneChangeMs = now
            }

            val totalMs = zoneDwellMs.sum()
            if (totalMs == 0L) return emptyMap()

            return (0 until 9)
                .filter { zoneDwellMs[it] > 0 }
                .associateWith { zoneDwellMs[it].toFloat() / totalMs.toFloat() }
        }
    }

    /**
     * Reset zone-dwell accumulators at the end of a report window.
     * Replaces the legacy GazeTrackingProcessor.resetZoneDwell().
     */
    fun resetZoneDwell() {
        synchronized(zoneDwellMs) {
            zoneDwellMs.fill(0)
            lastZoneChangeMs = System.currentTimeMillis()
            currentZone = 4
        }
    }

    /**
     * Track time spent in each gaze zone. Called from
     * processEmotionalEngagementZeroAlloc with the dominant face's focusRegion
     * each frame. Caps individual dwell at 10s to avoid stale data from paused
     * processing windows.
     */
    private fun updateZoneDwell(newZone: Int) {
        if (newZone !in 0..8) return
        synchronized(zoneDwellMs) {
            val now = System.currentTimeMillis()
            if (lastZoneChangeMs > 0 && currentZone in 0..8) {
                val dwellMs = now - lastZoneChangeMs
                zoneDwellMs[currentZone] += dwellMs.coerceAtMost(10_000L)
            }
            currentZone = newZone
            lastZoneChangeMs = now
        }
    }

    /**
     * Pure aggregation over per-frame gaze snapshots — replaces the legacy
     * GazeTrackingProcessor.aggregateMetrics() which had no instance state.
     */
    private fun aggregateGazeSnapshots(samples: List<GazeMetrics>): AggregatedGazeMetrics {
        if (samples.isEmpty()) return AggregatedGazeMetrics()

        val windowStart = samples.minOfOrNull { it.timestamp } ?: 0
        val windowEnd = samples.maxOfOrNull { it.timestamp } ?: 0

        val regionCounts = samples
            .filter { it.isLookingAtScreen }
            .groupBy { it.focusRegion }
            .mapValues { it.value.size }

        val primaryRegion = regionCounts.maxByOrNull { it.value }?.key ?: 4

        val lookingCount = samples.count { it.isLookingAtScreen }
        val lookingAtScreenPct = lookingCount.toFloat() / samples.size

        val avgStability = samples
            .filter { it.isLookingAtScreen }
            .map { it.gazeStability }
            .takeIf { it.isNotEmpty() }
            ?.average()
            ?.toFloat() ?: 0f

        val totalLooking = lookingCount.coerceAtLeast(1)
        val heatmap = (0 until 9).associateWith { region ->
            (regionCounts[region] ?: 0).toFloat() / totalLooking
        }

        val avgAttention = samples.map { it.attentionScore }.average().toFloat()

        return AggregatedGazeMetrics(
            windowStart = windowStart,
            windowEnd = windowEnd,
            sampleCount = samples.size,
            primaryFocusRegion = primaryRegion,
            lookingAtScreenPct = lookingAtScreenPct,
            avgGazeStability = avgStability,
            regionHeatmap = heatmap,
            gazeAttentionScore = avgAttention
        )
    }

    /**
     * Get mode of recent gaze focus regions.
     * Focus region is a 0-8 index on a 3x3 grid (4 = center).
     * Returns the most common focus region from recent gaze samples.
     */
    fun getAverageFocusRegion(): Int {
        val recentGaze: List<GazeMetrics>
        synchronized(gazeMetricsBuffer) {
            recentGaze = gazeMetricsBuffer.toList()
        }
        if (recentGaze.isEmpty()) return 4 // Default to center

        // GazeMetrics already has focusRegion (0-8 on a 3x3 grid, 4 = center)
        return recentGaze
            .map { it.focusRegion }
            .groupBy { it }
            .maxByOrNull { it.value.size }
            ?.key ?: 4
    }

    /**
     * Selective model lifecycle methods for memory attenuation.
     */

    fun stopPoseDetection() {
        runVisionMutation { stopPoseDetectionInternal() }
    }

    private fun stopPoseDetectionInternal() {
        poseProcessor?.release()
        poseProcessor = null
        refreshProcessorActivationState()
        Log.d(TAG, "[Attenuation] Pose detection stopped")
    }

    fun startPoseDetection() {
        runVisionMutation { startPoseDetectionInternal() }
    }

    private fun startPoseDetectionInternal() {
        // Close existing instance first to prevent native memory leaks
        poseProcessor?.release()
        poseProcessor = null
        poseProcessor = PoseEngagementProcessor(context).apply {
            if (hasModel() && initialize()) {
                refreshProcessorActivationState()
                Log.d(TAG, "[Attenuation] Pose detection started")
            } else {
                Log.w(TAG, "[Attenuation] Pose detection unavailable (model missing)")
                poseProcessor = null
                refreshProcessorActivationState()
            }
        }
    }

    // Emotion + gaze share one processor (FaceLandmarker) since Phase 3.
    // The legacy stop/start emotion/gaze pairs are kept as public aliases so
    // existing callers in AudienceSensingService (memory-attenuation tier
    // transitions) continue to work; both names target the same lifecycle.

    fun stopEmotionDetection() {
        runVisionMutation { stopFaceLandmarkerInternal() }
    }

    fun startEmotionDetection() {
        runVisionMutation { startFaceLandmarkerInternal() }
    }

    fun stopGazeTracking() {
        runVisionMutation { stopFaceLandmarkerInternal() }
    }

    fun startGazeTracking() {
        runVisionMutation { startFaceLandmarkerInternal() }
    }

    private fun stopFaceLandmarkerInternal() {
        if (faceLandmarkerProcessor == null) return
        faceLandmarkerProcessor?.release()
        faceLandmarkerProcessor = null
        resetZoneDwell()
        refreshProcessorActivationState()
        Log.d(TAG, "[Attenuation] FaceLandmarker (emotion + gaze + head-pose) stopped")
    }

    private fun startFaceLandmarkerInternal() {
        if (faceLandmarkerProcessor != null) return
        faceLandmarkerProcessor = FaceLandmarkerProcessor(context).apply {
            if (initialize()) {
                refreshProcessorActivationState()
                Log.d(TAG, "[Attenuation] FaceLandmarker (emotion + gaze + head-pose) started")
            } else {
                Log.w(TAG, "[Attenuation] FaceLandmarker init failed; emotion/gaze unavailable")
                faceLandmarkerProcessor = null
                refreshProcessorActivationState()
            }
        }
    }

    fun closeAgeGenderProcessor() {
        runVisionMutation { closeAgeGenderProcessorInternal() }
    }

    private fun closeAgeGenderProcessorInternal() {
        ageGenderProcessor?.close()
        ageGenderProcessor = null
        ageGenderEnabled = false
        refreshProcessorActivationState()
        Log.d(TAG, "[Attenuation] Age/gender processor closed")
    }

    /**
     * Public accessor for the SensorCollector instance.
     * Used by AudienceSensingService to read shadow events, ambient light,
     * and viewability score for the audience signal payload.
     */
    fun getSensorCollector(): SensorCollector = sensorCollector

    fun startAgeGenderProcessor() {
        runVisionMutation { startAgeGenderProcessorInternal() }
    }

    private fun startAgeGenderProcessorInternal() {
        // Close existing instance first to prevent native memory leaks
        ageGenderProcessor?.close()
        ageGenderProcessor = null
        ageGenderEnabled = false
        ageGenderProcessor = AgeGenderProcessor(context).apply {
            if (hasModel() && initialize()) {
                ageGenderEnabled = true
                refreshProcessorActivationState()
                Log.d(TAG, "[Attenuation] Age/gender processor started")
            } else {
                Log.w(TAG, "[Attenuation] Age/gender processor unavailable (model missing)")
                ageGenderProcessor = null
                ageGenderEnabled = false
                refreshProcessorActivationState()
            }
        }
    }

    fun stopPersonDetection() {
        runVisionMutation { stopPersonDetectionInternal() }
    }

    private fun stopPersonDetectionInternal() {
        personDetector?.release()
        personDetector = null
        currentPersonCount = 0
        objectDetector?.release()
        objectDetector = null
        Log.d(TAG, "[Attenuation] Person + object detection stopped")
    }

    fun startPersonDetection() {
        runVisionMutation { startPersonDetectionInternal() }
    }

    private fun startPersonDetectionInternal() {
        personDetector?.release()
        personDetector = null
        val pdCandidate = PersonDetectionProcessor(context)
        if (pdCandidate.hasModel() && pdCandidate.initialize()) {
            personDetector = pdCandidate
            Log.d(TAG, "[Attenuation] Person detection started")
        } else {
            pdCandidate.release()
            Log.w(TAG, "[Attenuation] Person detection unavailable (model missing)")
        }

        objectDetector?.release()
        objectDetector = null
        val odCandidate = ObjectDetectionProcessor(context)
        if (odCandidate.hasModel() && odCandidate.initialize()) {
            objectDetector = odCandidate
            vehicleFlowProcessor.reset()
            queueEstimationProcessor.reset()
            Log.d(TAG, "[Attenuation] Multi-class object detection started")
        } else {
            odCandidate.release()
            Log.w(TAG, "[Attenuation] Object detection unavailable (model missing)")
        }
    }

    fun applyVisionProcessorSelection(selection: VisionProcessorSelection) {
        desiredVisionSelection = selection
        if (!isRunning) {
            Log.i(TAG, "[Profile] Stored camera processor selection for next start: ${selection.activeModelIds()}")
            return
        }

        runVisionMutation {
            if (selection.wantsPose) {
                if (poseProcessor == null) startPoseDetectionInternal()
            } else if (poseProcessor != null) {
                stopPoseDetectionInternal()
            }

            // Emotion + gaze are produced by a single FaceLandmarker forward
            // pass; one lifecycle covers both `wants` flags.
            val wantsFaceLandmarker = selection.wantsEmotion || selection.wantsGaze
            if (wantsFaceLandmarker) {
                if (faceLandmarkerProcessor == null) startFaceLandmarkerInternal()
            } else if (faceLandmarkerProcessor != null) {
                stopFaceLandmarkerInternal()
            }

            if (selection.wantsAgeGender) {
                if (!ageGenderEnabled || ageGenderProcessor == null) startAgeGenderProcessorInternal()
            } else if (ageGenderProcessor != null || ageGenderEnabled) {
                closeAgeGenderProcessorInternal()
            }

            if (selection.wantsPersonDetection) {
                if (personDetector == null) startPersonDetectionInternal()
            } else if (personDetector != null) {
                stopPersonDetectionInternal()
            }

            Log.i(TAG, "[Profile] Camera processors active: ${selection.activeModelIds()}")
        }
    }

    private fun runVisionMutation(block: () -> Unit) {
        if (!isRunning) {
            block()
            return
        }

        val latch = CountDownLatch(1)
        try {
            analysisExecutor.execute {
                try {
                    block()
                } finally {
                    latch.countDown()
                }
            }
            if (!latch.await(2, TimeUnit.SECONDS)) {
                Log.w(TAG, "[Profile] Timed out waiting for analyzer-thread model mutation")
            }
        } catch (rejected: RejectedExecutionException) {
            Log.w(TAG, "[Profile] Analyzer executor unavailable during model mutation, applying inline")
            block()
        }
    }
}

data class AnalyzerRuntimeSnapshot(
    val actualFps: Float,
    val targetFps: Int,
    val skippedFrames: Long,
    val skipRatio: Float,
    val avgInferenceMs: Double,
    val lastInferenceMs: Long,
    val memoryInfo: ResourceMonitor.MemoryInfo
)
