package com.trillboards.ctv.core.audience

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import com.trillboards.ctv.core.SensingConfig
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import android.util.Size
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
    }

    // Device profile for adaptive settings
    private val deviceProfile by lazy { DeviceProfile.detect(context) }

    // Frame throttler for skipping frames based on device capabilities
    private val frameThrottler by lazy {
        FrameThrottler(
            initialTargetFps = deviceProfile.targetFps,
            cameraFps = 30
        )
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
    private var emotionProcessor: EmotionClassificationProcessor? = null
    private var gazeProcessor: GazeTrackingProcessor? = null
    private var emotionalEngagementEnabled = false

    // On-device age/gender processor
    private var ageGenderProcessor: AgeGenderProcessor? = null
    private var ageGenderEnabled = false

    // Local TFLite person detector (PoC) — only instantiated if model exists
    private var personDetector: PersonDetectionProcessor? = null
    private var currentPersonCount = 0
    @Volatile private var desiredVisionSelection: VisionProcessorSelection? = null

    // Latest emotional engagement metrics
    private val _currentEmotionalEngagement = MutableStateFlow<EmotionalEngagement?>(null)
    val currentEmotionalEngagement: StateFlow<EmotionalEngagement?> = _currentEmotionalEngagement

    // Buffers for emotional engagement aggregation
    private val poseMetricsBuffer = ArrayDeque<PoseMetrics>()
    private val emotionMetricsBuffer = ArrayDeque<EmotionMetrics>()
    private val gazeMetricsBuffer = ArrayDeque<GazeMetrics>()

    // ============================================================
    // Zero-allocation pre-allocated buffers (Phase 1)
    // Allocated once in initializeBuffers(), reused every frame.
    // ============================================================

    // Shared bitmap: camera frame is rendered here via manual YUV→RGB conversion.
    // Lifetime = camera session. Never recycled during operation.
    private var sharedBitmap: Bitmap? = null

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

    // Callback for when metrics are ready to report
    var onMetricsReady: ((AudienceMetricsPayload) -> Unit)? = null

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

    private var lastReportTime = System.currentTimeMillis()
    private var isRunning = false

    /**
     * Initialize ML Kit face detector.
     */
    private fun initializeFaceDetector() {
        Log.d(TAG, "[MLKit] Initializing face detector...")
        val detCfg = SensingConfig.get().detection
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)  // Required for GazeTrackingProcessor eye landmarks
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)  // For eye/smile
            .setMinFaceSize(detCfg.minFaceSize)  // Minimum face size relative to image
            .enableTracking()  // Enable face tracking across frames
            .build()

        faceDetector = FaceDetection.getClient(options)
        Log.i(TAG, "[MLKit] Face detector initialized - mode=FAST, landmarks=ALL, classification=ALL, minFaceSize=15%, tracking=ON")
    }

    /**
     * Initialize emotional engagement processors (pose, emotion, gaze).
     *
     * Guards against double initialization: if processors already exist, they are
     * closed first to prevent native memory leaks from orphaned model instances.
     * Each PoseLandmarker/TFLite interpreter holds ~500MB of native memory.
     */
    private fun initializeEmotionalEngagement(initialSelection: VisionProcessorSelection? = null) {
        Log.d(TAG, "[EmotionalEngagement] Initializing emotional engagement processors...")

        // CRITICAL: Close existing processors before creating new ones.
        // Without this, each re-init leaks ~500MB of native model weights.
        poseProcessor?.release()
        poseProcessor = null
        emotionProcessor?.release()
        emotionProcessor = null
        gazeProcessor?.reset()
        gazeProcessor = null
        ageGenderProcessor?.close()
        ageGenderProcessor = null
        ageGenderEnabled = false
        personDetector?.release()
        personDetector = null

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

        if (wantsEmotion) {
            emotionProcessor = EmotionClassificationProcessor(context).apply {
                if (hasModel() && initialize()) {
                    processorsInitialized++
                    Log.i(TAG, "[EmotionalEngagement] Emotion processor initialized")
                } else {
                    Log.w(TAG, "[EmotionalEngagement] Emotion processor not available (model missing)")
                    emotionProcessor = null
                }
            }
        }

        if (wantsGaze) {
            gazeProcessor = GazeTrackingProcessor()
            processorsInitialized++
            Log.i(TAG, "[EmotionalEngagement] Gaze processor initialized")
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
        }

        refreshProcessorActivationState(processorsInitialized > 0)

        if (emotionalEngagementEnabled) {
            Log.i(TAG, "[EmotionalEngagement] >>> Emotional engagement ENABLED ($processorsInitialized processors)")
        } else {
            Log.w(TAG, "[EmotionalEngagement] Emotional engagement DISABLED (no processors available)")
        }
    }

    private fun hasEmotionalProcessorsEnabled(): Boolean {
        return poseProcessor != null || emotionProcessor != null || gazeProcessor != null || ageGenderEnabled
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
        val width = bitmap.width
        val height = bitmap.height

        // Pixels are already in yuvPixelBuffer from imageProxyToSharedBitmap()
        // Re-read from bitmap to be safe (setPixels was called on the bitmap)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

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
        emotionProcessor?.release()
        emotionProcessor = null
        gazeProcessor?.reset()
        gazeProcessor = null
        ageGenderProcessor?.close()
        ageGenderProcessor = null
        ageGenderEnabled = false
        
        personDetector?.release()
        personDetector = null
        currentPersonCount = 0
        
        emotionalEngagementEnabled = false
        poseMetricsBuffer.clear()
        emotionMetricsBuffer.clear()
        gazeMetricsBuffer.clear()

        // Release pre-allocated buffers
        releaseBuffers(recycleBitmaps = true)

        // Final report
        generateAndSendReport()

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

        imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(analysisResolutionSelector)  // Adaptive resolution based on chipset
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
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

                        if (emotionalEngagementEnabled) {
                            processEmotionalEngagementZeroAlloc(
                                leasedBitmap,
                                faces,
                                imageProxy.width,
                                imageProxy.height
                            )
                        }

                        personDetector?.let { detector ->
                            if (detector.isReady()) {
                                currentPersonCount = detector.process(leasedBitmap)
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
                checkReportInterval()

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
        imageHeight: Int
    ) {
        try {
            // Process pose using pre-allocated ByteBuffer (zero-alloc path)
            val detCfg = SensingConfig.get().detection
            poseProcessor?.let { pose ->
                if (pose.isReady()) {
                    val poseBuffer = poseInputBuffer
                    if (poseBuffer != null && fillRgbByteBuffer()) {
                        // Build engagement context from the same frame's face list.
                        // screenEngaged uses the same head-yaw/pitch envelope as
                        // GazeTrackingProcessor.isLookingAtScreen (gazeCfg thresholds × 2),
                        // so a single face whose head rotation is within bounds means
                        // "user is stably looking at the screen". Pose's own facing-angle
                        // is intentionally NOT used here: it derives from shoulder
                        // geometry (a different signal source) and would couple two
                        // engagement gates to the same noisy primitive.
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

            // Process emotions for each face using pre-allocated crop bitmap
            if (faces.isNotEmpty()) {
                val emotionCrop = emotionCropBitmap
                emotionProcessor?.let { emotion ->
                    if (emotion.isReady() && emotionCrop != null) {
                        // Zero-alloc path: use Canvas to crop+scale into pre-allocated bitmap
                        val emotionMetrics = emotion.processMultipleWithPreallocatedCrop(
                            bitmap, faces, emotionCrop
                        )
                        synchronized(emotionMetricsBuffer) {
                            emotionMetricsBuffer.addAll(emotionMetrics)
                            while (emotionMetricsBuffer.size > detCfg.maxBufferedMetrics) emotionMetricsBuffer.removeFirst()
                        }
                    } else {
                        // Fall back to ML Kit heuristics (no TFLite model)
                        val heuristicEmotions = faces.map { emotion.classifyFromMLKit(it) }
                        synchronized(emotionMetricsBuffer) {
                            emotionMetricsBuffer.addAll(heuristicEmotions)
                            while (emotionMetricsBuffer.size > detCfg.maxBufferedMetrics) emotionMetricsBuffer.removeFirst()
                        }
                    }
                }

                // Process gaze for each face (pure math from landmarks — already zero-alloc)
                gazeProcessor?.let { gaze ->
                    val gazeMetrics = gaze.processMultiple(faces, imageWidth, imageHeight)
                    synchronized(gazeMetricsBuffer) {
                        gazeMetricsBuffer.addAll(gazeMetrics)
                        while (gazeMetricsBuffer.size > detCfg.maxBufferedMetrics) gazeMetricsBuffer.removeFirst()
                    }
                }

                // On-device age/gender using pre-allocated crop bitmap (zero-alloc)
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
                                        agp.accumulate(result)
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
                stoppedCount = poseSnapshots.count { it.movementState == MovementState.STOPPED },
                walkingPastCount = poseSnapshots.count { it.movementState == MovementState.WALKING_FAST },
                approachingCount = poseSnapshots.count { it.movementState == MovementState.APPROACHING }, // always 0 — requires depth
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

        // Aggregate gaze metrics
        val aggregatedGaze = gazeProcessor?.aggregateMetrics(gazeSnapshots)

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

        // Determine audience reaction
        val detCfg = SensingConfig.get().detection
        val audienceReaction = when {
            overallScore > detCfg.highlyEngagedThreshold -> AudienceReaction.HIGHLY_ENGAGED
            overallScore > detCfg.interestedThreshold -> AudienceReaction.INTERESTED
            overallScore > detCfg.neutralThreshold -> AudienceReaction.NEUTRAL
            aggregatedEmotion?.negativeReactionCount ?: 0 > aggregatedEmotion?.positiveReactionCount ?: 0 ->
                AudienceReaction.NEGATIVE
            else -> AudienceReaction.DISINTERESTED
        }

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

    /**
     * Calculate Intersection over Union (IoU) for two bounding boxes.
     * Used to detect if two face detections are the same person.
     * Higher IoU = more overlap = likely same face.
     */
    private fun calculateIoU(box1: FaceRect, box2: FaceRect): Float {
        val xOverlap = maxOf(0, minOf(box1.right, box2.right) - maxOf(box1.left, box2.left))
        val yOverlap = maxOf(0, minOf(box1.bottom, box2.bottom) - maxOf(box1.top, box2.top))
        val intersection = xOverlap * yOverlap

        val area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
        val area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
        val union = area1 + area2 - intersection

        return if (union > 0) intersection.toFloat() / union else 0f
    }

    /**
     * Calculate centroid distance between two bounding boxes.
     * More robust than IoU for head rotation because face center moves less.
     */
    private fun calculateCentroidDistance(box1: FaceRect, box2: FaceRect): Float {
        val cx1 = (box1.left + box1.right) / 2f
        val cy1 = (box1.top + box1.bottom) / 2f
        val cx2 = (box2.left + box2.right) / 2f
        val cy2 = (box2.top + box2.bottom) / 2f
        return kotlin.math.sqrt((cx2 - cx1) * (cx2 - cx1) + (cy2 - cy1) * (cy2 - cy1))
    }

    /**
     * Calculate average face size (for relative distance thresholding).
     */
    private fun averageFaceSize(box1: FaceRect, box2: FaceRect): Float {
        val size1 = maxOf(box1.right - box1.left, box1.bottom - box1.top)
        val size2 = maxOf(box2.right - box2.left, box2.bottom - box2.top)
        return (size1 + size2) / 2f
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

    private fun checkReportInterval() {
        val now = System.currentTimeMillis()
        if (now - lastReportTime >= config.reportIntervalMs) {
            generateAndSendReport()
            lastReportTime = now
        }
    }

    private fun generateAndSendReport() {
        val snapshot = _currentSnapshot.value
        val intervalMs = System.currentTimeMillis() - lastReportTime

        val payload = AudienceMetricsPayload(
            screenId = null,  // Will be filled in by the service
            fingerprint = "",  // Will be filled in by the service
            timestamp = snapshot.timestamp,
            intervalMs = intervalMs,
            viewerCount = snapshot.viewerCount,
            personCount = snapshot.personCount,
            peakViewerCount = peakViewerCount,
            attentionScore = snapshot.attentionScore,
            demographics = snapshot.demographics,
            dwellTime = snapshot.dwellTime,
            environment = snapshot.environment
        )

        // Reset peak for next interval
        peakViewerCount = snapshot.viewerCount

        analyzerScope.launch {
            onMetricsReady?.invoke(payload)
        }
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
     * Get the gaze tracking processor for creative zone data.
     */
    fun getGazeProcessor(): GazeTrackingProcessor? = gazeProcessor

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
     * Get last emotion classification scores for gradient computation.
     * Returns the most recent emotion probability distribution, or null if unavailable.
     */
    fun getLastEmotionScores(): FloatArray? {
        val recentEmotions: List<EmotionMetrics>
        synchronized(emotionMetricsBuffer) {
            recentEmotions = emotionMetricsBuffer.toList()
        }
        if (recentEmotions.isEmpty()) return null

        val last = recentEmotions.last()
        // Return emotion scores as: [happy, surprised, neutral, sad, angry, disgusted, confused]
        return floatArrayOf(
            last.happyScore,
            last.surprisedScore,
            last.neutralScore,
            last.sadScore,
            last.angryScore,
            last.disgustScore,
            if (last.dominantEmotion == EmotionType.CONFUSED) last.emotionConfidence else 0f
        )
    }

    /**
     * Get self-supervised emotion pseudo-labels from engagement feedback.
     * Uses engagement score as a soft label: high engagement = positive emotions likely correct.
     */
    fun getEmotionPseudoLabels(): FloatArray? {
        val recentEmotions: List<EmotionMetrics>
        synchronized(emotionMetricsBuffer) {
            recentEmotions = emotionMetricsBuffer.toList()
        }
        if (recentEmotions.isEmpty()) return null

        // Average emotion distribution from recent samples as pseudo-labels
        val counts = FloatArray(7) // [happy, surprised, neutral, sad, angry, disgusted, confused]
        val emotionTypes = listOf(EmotionType.HAPPY, EmotionType.SURPRISED, EmotionType.NEUTRAL,
            EmotionType.SAD, EmotionType.ANGRY, EmotionType.DISGUSTED, EmotionType.CONFUSED)

        for (em in recentEmotions) {
            val idx = emotionTypes.indexOf(em.dominantEmotion)
            if (idx >= 0) counts[idx] += 1f
        }

        // Normalize to probability distribution
        val total = counts.sum()
        if (total == 0f) return null
        return FloatArray(7) { counts[it] / total }
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

    fun stopEmotionDetection() {
        runVisionMutation { stopEmotionDetectionInternal() }
    }

    private fun stopEmotionDetectionInternal() {
        emotionProcessor?.release()
        emotionProcessor = null
        refreshProcessorActivationState()
        Log.d(TAG, "[Attenuation] Emotion detection stopped")
    }

    fun startEmotionDetection() {
        runVisionMutation { startEmotionDetectionInternal() }
    }

    private fun startEmotionDetectionInternal() {
        // Close existing instance first to prevent native memory leaks
        emotionProcessor?.release()
        emotionProcessor = null
        emotionProcessor = EmotionClassificationProcessor(context).apply {
            if (hasModel() && initialize()) {
                refreshProcessorActivationState()
                Log.d(TAG, "[Attenuation] Emotion detection started")
            } else {
                Log.w(TAG, "[Attenuation] Emotion detection unavailable (model missing)")
                emotionProcessor = null
                refreshProcessorActivationState()
            }
        }
    }

    fun stopGazeTracking() {
        runVisionMutation { stopGazeTrackingInternal() }
    }

    private fun stopGazeTrackingInternal() {
        gazeProcessor?.reset()
        gazeProcessor = null
        refreshProcessorActivationState()
        Log.d(TAG, "[Attenuation] Gaze tracking stopped")
    }

    fun startGazeTracking() {
        runVisionMutation { startGazeTrackingInternal() }
    }

    private fun startGazeTrackingInternal() {
        if (gazeProcessor == null) {
            gazeProcessor = GazeTrackingProcessor()
            refreshProcessorActivationState()
            Log.d(TAG, "[Attenuation] Gaze tracking started")
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
        Log.d(TAG, "[Attenuation] Person detection stopped")
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

            if (selection.wantsEmotion) {
                if (emotionProcessor == null) startEmotionDetectionInternal()
            } else if (emotionProcessor != null) {
                stopEmotionDetectionInternal()
            }

            if (selection.wantsGaze) {
                if (gazeProcessor == null) startGazeTrackingInternal()
            } else if (gazeProcessor != null) {
                stopGazeTrackingInternal()
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
