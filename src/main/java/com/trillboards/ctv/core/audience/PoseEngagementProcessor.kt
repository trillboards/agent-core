package com.trillboards.ctv.core.audience

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.nio.ByteBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.inference.DelegateSelector
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Body posture engagement processor using MediaPipe Pose Lite.
 *
 * Detects:
 * - Body facing angle (is the person oriented toward the screen?)
 * - Leaning direction (leaning in = high engagement, leaning back = skepticism)
 * - Movement state (stopped, walking slow, walking fast)
 * - Overall body engagement score
 *
 * Model: pose_landmarker_lite.task (~3.5MB)
 * Inference: ~30ms on mid-range devices
 *
 * MediaPipe Pose returns 33 body landmarks:
 * 0-10: Face landmarks
 * 11-12: Shoulders
 * 13-14: Elbows
 * 15-16: Wrists
 * 23-24: Hips
 * 25-26: Knees
 * 27-28: Ankles
 */
class PoseEngagementProcessor(
    private val context: Context,
    private val config: EmotionalEngagementConfig = EmotionalEngagementConfig()
) {
    companion object {
        private const val TAG = "PoseEngagement"

        // Model file (must be in assets/)
        private const val POSE_MODEL_FILE = "pose_landmarker_lite.task"

        // Key landmark indices
        private const val NOSE = 0
        private const val LEFT_SHOULDER = 11
        private const val RIGHT_SHOULDER = 12
        private const val LEFT_HIP = 23
        private const val RIGHT_HIP = 24
        private const val LEFT_ANKLE = 27
        private const val RIGHT_ANKLE = 28

        /**
         * Pure helper that classifies movement state from raw hip-position deltas
         * plus an engagement context (faceCount + screenEngaged).
         *
         * Bug guard (2026-05-02): a user typing at a desk produced hip-center
         * displacement large enough that the speed-only classifier emitted
         * WALKING_FAST. Real walking always produces displacement above
         * [SensingConfig.BodyConfig.engagedStationaryHipThreshold]; typing /
         * sipping / gesturing-while-talking does not. When a single face is
         * stably looking at the screen, we use that as evidence the upper-body
         * sway is NOT walking and force STOPPED.
         *
         * The gate intentionally only applies when the displacement is below the
         * engaged-stationary threshold — a user who happens to be facing the
         * screen while genuinely walking (rare but possible) still classifies
         * correctly because their hip displacement clears the threshold.
         *
         * @return (state, normalizedSpeed) — normalizedSpeed is the same value
         *         the previous implementation produced, so downstream consumers
         *         (engagement-score weighting) are unaffected.
         */
        @JvmStatic
        fun classifyMovement(
            prevHipX: Float, prevHipY: Float,
            currHipX: Float, currHipY: Float,
            timeDeltaSec: Float,
            faceCount: Int,
            screenEngaged: Boolean,
            cfg: SensingConfig.BodyConfig
        ): Pair<MovementState, Float> {
            if (timeDeltaSec <= 0f) {
                return Pair(MovementState.UNKNOWN, 0f)
            }

            val deltaX = currHipX - prevHipX
            val deltaY = currHipY - prevHipY
            val hipMagnitude = sqrt(deltaX * deltaX + deltaY * deltaY)

            // Speed = displacement per second, clamped to [0,1]. Preserved verbatim
            // from the original calculateMovement so downstream weighting is unchanged.
            val speed = (hipMagnitude / timeDeltaSec).coerceIn(0f, 1f)

            val baseState = when {
                speed < cfg.movementSpeedThreshold -> MovementState.STOPPED
                speed < cfg.movementSpeedThreshold * cfg.walkingSlowSpeedMultiplier -> MovementState.WALKING_SLOW
                else -> MovementState.WALKING_FAST
            }

            // Engagement gate: at least one stably-engaged face + small raw hip
            // displacement → upper-body micro-motion (typing, gesturing, sipping)
            // and NOT actual walking. Override to STOPPED.
            //
            // P2.1 fix (audit 2026-05-03): the gate now fires for any
            // `faceCount >= 1 AND screenEngaged`, not the original
            // `faceCount == 1` hardcode. Two coworkers browsing a kiosk or a
            // couple stopping to read produced the same typing-as-walking
            // false-positive the single-face fix was supposed to eliminate. The
            // signal that determines "we have a stationary engaged person near
            // the screen" is the presence of any engaged face — `screenEngaged`
            // is true when at least one face is stably looking at the screen
            // (caller computes engagement as `max(perFaceEngagement)` upstream
            // so two engaged faces don't dilute the signal: the highest-
            // engagement face dominates the gate decision). The hip-
            // displacement threshold still preserves real walking-while-engaged
            // scenes — a person genuinely walking past produces hip motion
            // above `engagedStationaryHipThreshold` regardless of how many
            // faces are in frame.
            if (faceCount >= 1 && screenEngaged &&
                (baseState == MovementState.WALKING_FAST || baseState == MovementState.WALKING_SLOW) &&
                hipMagnitude < cfg.engagedStationaryHipThreshold
            ) {
                return Pair(MovementState.STOPPED, speed)
            }

            return Pair(baseState, speed)
        }
    }

    /**
     * Per-frame engagement context for movement classification.
     *
     * Caller (AudienceAnalyzer) supplies these from the same frame the pose
     * was inferred against:
     * - [faceCount]: number of faces ML Kit detected in the frame
     * - [screenEngaged]: whether at least one face is stably looking at the
     *   screen (head yaw/pitch within engagement bounds — same logic as
     *   FaceLandmarker's HeadPose.isFacingScreen())
     *
     * Default ([NONE]) preserves the original speed-only behaviour, so
     * existing call sites that don't yet supply context continue to work.
     */
    data class MovementContext(
        val faceCount: Int = 0,
        val screenEngaged: Boolean = false
    ) {
        companion object {
            /** Equivalent to "no engagement signal" — preserves pre-fix behaviour. */
            val NONE = MovementContext(faceCount = 0, screenEngaged = false)
        }
    }

    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var poseLandmarker: PoseLandmarker? = null
    private var isInitialized = false

    // Previous frame data for movement detection
    private var previousHipCenter: Pair<Float, Float>? = null
    private var previousTimestamp: Long = 0

    // Current metrics state
    private val _currentMetrics = MutableStateFlow(PoseMetrics())
    val currentMetrics: StateFlow<PoseMetrics> = _currentMetrics

    // Callback for when metrics are ready
    var onMetricsReady: ((PoseMetrics) -> Unit)? = null

    /**
     * Check if the pose model is available.
     */
    fun hasModel(): Boolean {
        return try {
            context.assets.open(POSE_MODEL_FILE).close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Pose model not found: ${e.message}")
            false
        }
    }

    /**
     * Initialize the pose landmarker.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        if (!hasModel()) {
            Log.e(TAG, "Pose model not available - cannot initialize")
            return false
        }

        // Honest GPU/NPU acceleration: live ADB on the Galaxy Tab S11
        // (MediaTek Dimensity 9400, Mali-G720) showed Pose running on CPU
        // XNNPack because this path used to hardcode Delegate.CPU with the
        // comment "CPU is more reliable across devices" — that's exactly
        // what DelegateSelector.withFallback / the test-inference loop
        // exists to handle. Mali GPU sat idle for ~30ms/frame. Mirror the
        // PersonDetectionProcessor.kt:88-134 pattern: try chipset-recommended
        // delegate first with a test inference, fall back to CPU on failure.
        val profile = DeviceProfile.detect(context)
        val recommendation = DelegateSelector.recommendAndLog(profile.chipsetVendor, "vision")
        val delegates = if (recommendation.primary != recommendation.fallback) {
            listOf(recommendation.primary, recommendation.fallback)
        } else {
            listOf(recommendation.primary)
        }

        val bodyCfg = SensingConfig.get().body
        for (delegate in delegates) {
            try {
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath(POSE_MODEL_FILE)
                    .setDelegate(delegate)
                    .build()

                val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMinPoseDetectionConfidence(bodyCfg.minPoseDetectionConfidence)
                    .setMinPosePresenceConfidence(bodyCfg.minPosePresenceConfidence)
                    .setMinTrackingConfidence(bodyCfg.minTrackingConfidence)
                    .setNumPoses(3) // Detect up to 3 people
                    .build()

                val landmarker = PoseLandmarker.createFromOptions(context, options)

                // Validate with a test inference — catches GPU drivers that
                // initialize successfully but fail at runtime in the
                // ToTensorConverter (Samsung Android 16 known issue).
                val testBitmap = Bitmap.createBitmap(192, 192, Bitmap.Config.ARGB_8888)
                try {
                    val testImage = BitmapImageBuilder(testBitmap).build()
                    try {
                        landmarker.detect(testImage)
                    } finally {
                        testImage.close()
                    }
                    Log.i(TAG, "Delegate $delegate passed test inference")
                } finally {
                    if (!testBitmap.isRecycled) testBitmap.recycle()
                }

                poseLandmarker = landmarker
                isInitialized = true
                Log.i(TAG, "Pose landmarker initialized with delegate=$delegate")
                return true
            } catch (e: Exception) {
                Log.w(TAG, "Delegate $delegate failed (init or test inference): ${e.message}")
                // Continue to next delegate
            }
        }

        Log.e(TAG, "All delegates failed for PoseEngagementProcessor")
        return false
    }

    /**
     * Process a bitmap frame and extract pose engagement metrics.
     *
     * @param bitmap The camera frame to analyze
     * @param context Per-frame engagement signal (faceCount + screenEngaged) used
     *        to gate stationary classification when small upper-body sway would
     *        otherwise read as walking. Defaults to [MovementContext.NONE], which
     *        preserves the original speed-only behaviour for callers that don't
     *        yet supply context.
     * @return PoseMetrics with engagement signals
     * @deprecated Use processWithBuffer for zero-allocation pipeline
     */
    @JvmOverloads
    @Deprecated("Use processWithBuffer for zero-allocation pipeline", ReplaceWith("processWithBuffer(buffer, width, height)"))
    fun process(
        bitmap: Bitmap,
        context: MovementContext = MovementContext.NONE
    ): PoseMetrics? {
        if (!isInitialized) {
            Log.w(TAG, "Not initialized - call initialize() first")
            return null
        }

        val landmarker = poseLandmarker ?: return null

        return try {
            // IMPORTANT: MediaPipe's BitmapImageContainer.close() (called via
            // mpImage.close()) calls recycle() on the source bitmap. Passing the
            // caller's bitmap directly would recycle AudienceAnalyzer's pre-allocated
            // zero-alloc sharedBitmap, causing per-frame buffer reinit (~24×/sec).
            // Fix mirrors PersonDetectionProcessor: pass a defensive copy to MediaPipe;
            // recycle the copy in finally; the caller's bitmap stays alive.
            val frameCopy = try {
                bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to copy bitmap for pose detection: ${e.message}")
                return null
            }

            val bitmapWidth = bitmap.width
            val bitmapHeight = bitmap.height

            val result = try {
                val mpImage = BitmapImageBuilder(frameCopy).build()
                try {
                    landmarker.detect(mpImage)
                } finally {
                    // mpImage.close() calls BitmapImageContainer.close() which
                    // calls frameCopy.recycle(). The shared bitmap is unaffected
                    // because we passed a copy.
                    mpImage.close()
                }
            } finally {
                // Defensive: if MediaPipe's BitmapImageContainer ever stops
                // recycling on close (API change), recycle the throwaway here
                // so we don't leak the copy per call.
                if (!frameCopy.isRecycled) {
                    frameCopy.recycle()
                }
            }

            analyzePoseResult(result, bitmapWidth, bitmapHeight, context)
        } catch (e: Exception) {
            Log.e(TAG, "Pose detection error", e)
            null
        }
    }

    /**
     * Zero-allocation pose detection using a pre-allocated RGB byte ByteBuffer.
     *
     * The caller owns the ByteBuffer and reuses it across frames.
     * ByteBufferImageBuilder wraps it without copying; mpImage.close()
     * only frees the lightweight MPImage wrapper, not the underlying buffer.
     *
     * IMAGE_FORMAT_RGB expects 3 unsigned bytes per pixel (R, G, B each 0-255).
     * Buffer size must be exactly width × height × 3 bytes.
     *
     * @param rgbByteBuffer Pre-allocated ByteBuffer containing RGB bytes (3 bytes per pixel, 0-255)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param context Per-frame engagement signal (faceCount + screenEngaged) used
     *        to gate stationary classification when small upper-body sway would
     *        otherwise read as walking. Defaults to [MovementContext.NONE], which
     *        preserves the original speed-only behaviour for callers that don't
     *        yet supply context.
     * @return PoseMetrics with engagement signals, or null on failure
     */
    @JvmOverloads
    fun processWithBuffer(
        rgbByteBuffer: ByteBuffer,
        width: Int,
        height: Int,
        context: MovementContext = MovementContext.NONE
    ): PoseMetrics? {
        if (!isInitialized) {
            Log.w(TAG, "Not initialized - call initialize() first")
            return null
        }

        val landmarker = poseLandmarker ?: return null

        return try {
            rgbByteBuffer.rewind()
            val mpImage = ByteBufferImageBuilder(
                rgbByteBuffer,
                width,
                height,
                MPImage.IMAGE_FORMAT_RGB
            ).build()

            val result = landmarker.detect(mpImage)
            // close() frees only the MPImage wrapper — our buffer survives
            mpImage.close()

            analyzePoseResult(result, width, height, context)
        } catch (e: Exception) {
            Log.e(TAG, "Pose detection (buffer) error", e)
            null
        }
    }

    /**
     * Process a frame asynchronously and emit results via callback.
     */
    @JvmOverloads
    fun processAsync(bitmap: Bitmap, context: MovementContext = MovementContext.NONE) {
        if (!isInitialized) return

        processorScope.launch {
            @Suppress("DEPRECATION")
            val metrics = process(bitmap, context)
            if (metrics != null) {
                _currentMetrics.value = metrics
                onMetricsReady?.invoke(metrics)
            }
        }
    }

    /**
     * Analyze the pose detection result and extract engagement metrics.
     */
    private fun analyzePoseResult(
        result: PoseLandmarkerResult,
        imageWidth: Int,
        imageHeight: Int,
        context: MovementContext = MovementContext.NONE
    ): PoseMetrics {
        if (result.landmarks().isEmpty()) {
            Log.d(TAG, "No poses detected")
            return PoseMetrics(confidence = 0f)
        }

        // Analyze the first (primary) detected person
        val landmarks = result.landmarks()[0]

        // Get key landmarks
        val nose = landmarks.getOrNull(NOSE)
        val leftShoulder = landmarks.getOrNull(LEFT_SHOULDER)
        val rightShoulder = landmarks.getOrNull(RIGHT_SHOULDER)
        val leftHip = landmarks.getOrNull(LEFT_HIP)
        val rightHip = landmarks.getOrNull(RIGHT_HIP)
        val leftAnkle = landmarks.getOrNull(LEFT_ANKLE)
        val rightAnkle = landmarks.getOrNull(RIGHT_ANKLE)

        if (leftShoulder == null || rightShoulder == null ||
            leftHip == null || rightHip == null) {
            Log.d(TAG, "Missing key landmarks")
            return PoseMetrics(confidence = SensingConfig.get().body.missingLandmarksConfidence)
        }

        // Calculate shoulder center
        val shoulderCenterX = (leftShoulder.x() + rightShoulder.x()) / 2f
        val shoulderCenterY = (leftShoulder.y() + rightShoulder.y()) / 2f

        // Calculate hip center
        val hipCenterX = (leftHip.x() + rightHip.x()) / 2f
        val hipCenterY = (leftHip.y() + rightHip.y()) / 2f

        // Calculate facing angle based on shoulder orientation
        val shoulderDeltaZ = if (leftShoulder.z() != 0f && rightShoulder.z() != 0f) {
            rightShoulder.z() - leftShoulder.z()
        } else {
            0f
        }
        val facingAngle = calculateFacingAngle(
            leftShoulder.x(), rightShoulder.x(),
            shoulderDeltaZ
        )
        val bodyCfg = SensingConfig.get().body
        val isFacingScreen = abs(facingAngle) < bodyCfg.facingAngleThreshold

        // Calculate lean direction
        val (leanDirection, leanMagnitude) = calculateLean(
            shoulderCenterX, shoulderCenterY,
            hipCenterX, hipCenterY
        )

        // Calculate movement state, using the per-frame engagement context from
        // the caller (faceCount + screenEngaged) to gate stationary-override.
        val currentTime = System.currentTimeMillis()
        val (movementState, movementSpeed) = calculateMovement(
            hipCenterX, hipCenterY,
            currentTime,
            context
        )

        // Update previous frame data
        previousHipCenter = Pair(hipCenterX, hipCenterY)
        previousTimestamp = currentTime

        // Calculate overall body engagement score
        val bodyEngagementScore = calculateEngagementScore(
            isFacingScreen, leanDirection, leanMagnitude, movementState
        )

        // Average landmark confidence
        val avgConfidence = listOf(
            leftShoulder.visibility().orElse(0f),
            rightShoulder.visibility().orElse(0f),
            leftHip.visibility().orElse(0f),
            rightHip.visibility().orElse(0f)
        ).average().toFloat()

        val metrics = PoseMetrics(
            facingAngle = facingAngle,
            isFacingScreen = isFacingScreen,
            leanDirection = leanDirection,
            leanMagnitude = leanMagnitude,
            movementState = movementState,
            movementSpeed = movementSpeed,
            bodyEngagementScore = bodyEngagementScore,
            confidence = avgConfidence
        )

        Log.d(TAG, "Pose: facing=${facingAngle.toInt()}deg, lean=$leanDirection, " +
                "movement=$movementState, engagement=${String.format("%.2f", bodyEngagementScore)}")

        return metrics
    }

    /**
     * Calculate body facing angle from shoulder positions.
     * Uses Z-depth to determine if shoulders are at different depths (indicating rotation).
     */
    private fun calculateFacingAngle(
        leftShoulderX: Float,
        rightShoulderX: Float,
        shoulderDeltaZ: Float
    ): Float {
        // Calculate angle from shoulder depth difference
        // If Z values are similar, person is facing camera
        // If Z values differ significantly, person is turned

        // Use atan2 for angle calculation
        // Shoulder width in normalized coords is typically ~0.2-0.3
        val shoulderWidth = abs(rightShoulderX - leftShoulderX)

        if (shoulderWidth < SensingConfig.get().body.minShoulderWidth) {
            // Shoulders too close together - probably turned 90 degrees
            return 90f
        }

        // Z delta indicates rotation - larger delta = more turned. Always
        // emit absolute deviation (0..90°) so downstream arithmetic-mean
        // aggregation (AudienceAnalyzer.kt avgFacingAngle) does not cancel
        // signed values toward zero. Every existing call site of
        // .facingAngle (isFacingScreen line 396, calibration thresholds)
        // already wraps abs() before comparing, so the abs() here is the
        // canonical contract — facingAngle is unsigned deviation.
        val angleRad = atan2(shoulderDeltaZ, shoulderWidth)
        return abs(Math.toDegrees(angleRad.toDouble())).toFloat()
    }

    /**
     * Calculate lean direction and magnitude.
     * Uses shoulder-to-hip Y-ratio instead of X-axis displacement.
     * This works regardless of camera position (overhead or eye-level).
     */
    private fun calculateLean(
        shoulderX: Float, shoulderY: Float,
        hipX: Float, hipY: Float
    ): Pair<LeanDirection, Float> {
        // Use torso height ratio (Y difference between hips and shoulders)
        // In normalized coords, Y increases downward
        // Normal standing: torsoHeightRatio is some baseline value (~0.15)
        // Leaning forward: shoulders drop toward hips (ratio decreases)
        // Leaning back: shoulders rise away from hips (ratio increases)
        val cfg = SensingConfig.get().body
        val torsoHeightRatio = hipY - shoulderY

        val baseline = cfg.leanBaseline  // Expected torso height in normalized coords
        val deviation = torsoHeightRatio - baseline

        val magnitude = abs(deviation).coerceIn(0f, cfg.leanMaxMagnitude) / cfg.leanMaxMagnitude  // Normalize to 0-1

        val direction = when {
            magnitude < cfg.leanMagnitudeThreshold -> LeanDirection.NEUTRAL
            deviation < cfg.leanInMinDeviation -> LeanDirection.LEANING_IN   // Torso compressed = leaning forward
            deviation > cfg.leanBackMinDeviation -> LeanDirection.LEANING_BACK   // Torso extended = leaning back
            else -> LeanDirection.NEUTRAL
        }

        return Pair(direction, magnitude)
    }

    /**
     * Calculate movement state by comparing current hip position to previous frame.
     *
     * Delegates to the pure [classifyMovement] helper, supplying the engagement
     * context from the caller. Unit tests cover the helper directly; this
     * method is the orchestrator that owns prior-frame state.
     */
    private fun calculateMovement(
        currentHipX: Float, currentHipY: Float,
        currentTime: Long,
        context: MovementContext
    ): Pair<MovementState, Float> {
        val prevHip = previousHipCenter
        val prevTime = previousTimestamp

        if (prevHip == null || prevTime == 0L) {
            return Pair(MovementState.UNKNOWN, 0f)
        }

        val timeDeltaSec = (currentTime - prevTime) / 1000f
        return classifyMovement(
            prevHipX = prevHip.first, prevHipY = prevHip.second,
            currHipX = currentHipX, currHipY = currentHipY,
            timeDeltaSec = timeDeltaSec,
            faceCount = context.faceCount,
            screenEngaged = context.screenEngaged,
            cfg = SensingConfig.get().body
        )
    }

    /**
     * Calculate overall body engagement score.
     */
    private fun calculateEngagementScore(
        isFacingScreen: Boolean,
        leanDirection: LeanDirection,
        leanMagnitude: Float,
        movementState: MovementState
    ): Float {
        val cfg = SensingConfig.get().body
        var score = 0f

        // Facing screen is important
        if (isFacingScreen) score += cfg.facingWeight

        // Lean direction
        score += when (leanDirection) {
            LeanDirection.LEANING_IN -> cfg.leanWeight * (0.5f + 0.5f * leanMagnitude)
            LeanDirection.NEUTRAL -> cfg.leanNeutralScore
            LeanDirection.LEANING_BACK -> cfg.leanBackScore
            LeanDirection.UNKNOWN -> cfg.leanUnknownScore
        }

        // Movement state
        @Suppress("DEPRECATION")
        score += when (movementState) {
            MovementState.STOPPED -> cfg.stoppedScore          // Best - they stopped to look
            MovementState.WALKING_SLOW -> cfg.walkingSlowScore    // Moderate - some interest
            MovementState.WALKING_FAST -> cfg.walkingFastScore    // Low - walking past
            MovementState.APPROACHING -> 0.40f     // Dead: never emitted (requires depth)
            MovementState.DEPARTING -> 0.10f       // Dead: never emitted (requires depth)
            MovementState.UNKNOWN -> cfg.unknownMovementScore
        }

        return score.coerceIn(0f, 1f)
    }

    /**
     * Release resources.
     */
    fun release() {
        try {
            poseLandmarker?.close()
            poseLandmarker = null
            isInitialized = false
            previousHipCenter = null
            previousTimestamp = 0
            Log.i(TAG, "Pose landmarker released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing pose landmarker", e)
        }
    }

    /**
     * Check if processor is ready.
     */
    fun isReady(): Boolean = isInitialized
}
