package com.trillboards.ctv.core.audience

import android.util.Log
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceLandmark
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Head Attention Zone Processor using ML Kit face landmarks.
 *
 * IMPORTANT: This uses ML Kit face landmarks (eye center + head rotation)
 * which provides HEAD POSE direction, not actual iris-based gaze direction.
 * Accuracy is +/-15-20 degrees for screen zone (3x3 grid).
 * For actual gaze tracking, MediaPipe Iris landmarks are needed (future).
 * Current data should be used as "attention zone" not "gaze point".
 *
 * Estimates where viewers are looking on the screen by analyzing:
 * - Eye position relative to eye socket
 * - Head rotation (yaw/pitch)
 * - Face position in frame
 *
 * Screen is divided into a 3x3 grid:
 *
 *   0 | 1 | 2   (top row)
 *   ---------
 *   3 | 4 | 5   (middle row - center is ideal ad placement)
 *   ---------
 *   6 | 7 | 8   (bottom row)
 *
 * No additional model required - uses ML Kit face landmarks directly.
 * Inference: ~5ms (part of existing ML Kit face detection)
 */
class GazeTrackingProcessor(
    private val config: EmotionalEngagementConfig = EmotionalEngagementConfig()
) {
    companion object {
        private const val TAG = "GazeTracking"

        // Grid regions
        private const val GRID_ROWS = 3
        private const val GRID_COLS = 3
        private const val CENTER_REGION = 4

        // Zone label mapping (3x3 grid: 0=top_left through 8=bottom_right)
        private val ZONE_LABEL_MAP = mapOf(
            0 to "top_left",
            1 to "top_center",
            2 to "top_right",
            3 to "middle_left",
            4 to "center",
            5 to "middle_right",
            6 to "bottom_left",
            7 to "bottom_center",
            8 to "bottom_right"
        )

        fun zoneToLabel(zone: Int): String = ZONE_LABEL_MAP[zone] ?: "unknown"
    }

    // Previous gaze data for stability calculation
    private var previousGazeX: Float? = null
    private var previousGazeY: Float? = null
    private var gazeHistory = mutableListOf<Pair<Float, Float>>()
    private val maxHistorySize = 10

    // Zone dwell time tracking (creative-zone heatmaps)
    private val zoneDwellMs = LongArray(9)      // Milliseconds spent in each grid zone
    private var currentZone: Int = CENTER_REGION // Current gaze zone
    private var lastZoneChangeMs: Long = 0       // Timestamp of last zone transition

    // Current metrics state
    private val _currentMetrics = MutableStateFlow(GazeMetrics())
    val currentMetrics: StateFlow<GazeMetrics> = _currentMetrics

    // Callback for when metrics are ready
    var onMetricsReady: ((GazeMetrics) -> Unit)? = null

    /**
     * Process a face and estimate gaze direction.
     *
     * @param face ML Kit Face with landmarks and head rotation
     * @param imageWidth Camera frame width
     * @param imageHeight Camera frame height
     * @return GazeMetrics with gaze direction and stability
     */
    fun process(face: Face, imageWidth: Int, imageHeight: Int): GazeMetrics {
        // Check confidence
        val trackingId = face.trackingId ?: -1

        // Get head rotation
        val headRotationX = face.headEulerAngleX  // Pitch (up/down)
        val headRotationY = face.headEulerAngleY  // Yaw (left/right)
        val headRotationZ = face.headEulerAngleZ  // Roll (tilt)

        val gazeCfg = SensingConfig.get().gaze

        // Check if looking at screen (not turned away)
        val isLookingAtScreen = abs(headRotationY) < gazeCfg.headYawThreshold * 2 &&
                abs(headRotationX) < gazeCfg.headPitchThreshold * 2

        // Compute gaze confidence from multiple factors:
        // - Whether the face has a stable tracking ID (tracked > untracked)
        // - Head angle magnitude (extreme angles = lower gaze accuracy)
        // - Eye landmark availability (both eyes > one > none)
        // Capped at 0.85 because this is head-pose estimation, not iris tracking
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)
        val trackingConfidence = if (trackingId >= 0) 1.0f else gazeCfg.landmarkFallbackConfidence
        val angleConfidence = (1f - (abs(headRotationY) + abs(headRotationX)) / 180f).coerceIn(0f, 1f)
        val landmarkConfidence = if (leftEye != null && rightEye != null) gazeCfg.landmarkAvailableConfidence else gazeCfg.landmarkFallbackConfidence
        val gazeConfidence = (trackingConfidence * gazeCfg.confidenceTrackingWeight + angleConfidence * gazeCfg.confidenceAngleWeight + landmarkConfidence * gazeCfg.confidenceLandmarkWeight)
            .coerceIn(gazeCfg.minGazeConfidence, gazeCfg.maxGazeConfidence)

        if (!isLookingAtScreen) {
            Log.d(TAG, "Face not looking at screen (yaw=${headRotationY.toInt()}, pitch=${headRotationX.toInt()})")
            return GazeMetrics(
                isLookingAtScreen = false,
                headRotationX = headRotationX,
                headRotationY = headRotationY,
                headRotationZ = headRotationZ,
                confidence = (gazeConfidence * gazeCfg.offScreenConfidenceScale).coerceIn(gazeCfg.offScreenMinConfidence, gazeCfg.offScreenMaxConfidence)
            )
        }

        // Estimate gaze point using eye landmarks and head rotation
        val gazePoint = estimateGazePoint(face, headRotationX, headRotationY, imageWidth, imageHeight)

        // Map gaze point to grid region
        val focusRegion = mapToGridRegion(gazePoint.first, gazePoint.second)

        // Track zone dwell time for creative-zone heatmaps
        updateZoneDwell(focusRegion)

        // Calculate gaze stability
        val gazeStability = calculateGazeStability(gazePoint.first, gazePoint.second)

        // Update history
        updateGazeHistory(gazePoint.first, gazePoint.second)

        // Calculate attention score
        val attentionScore = calculateAttentionScore(
            focusRegion, gazeStability, isLookingAtScreen, headRotationY
        )

        val metrics = GazeMetrics(
            focusRegion = focusRegion,
            isLookingAtScreen = isLookingAtScreen,
            gazeStability = gazeStability,
            headRotationX = headRotationX,
            headRotationY = headRotationY,
            headRotationZ = headRotationZ,
            attentionScore = attentionScore,
            confidence = gazeConfidence
        )

        _currentMetrics.value = metrics
        onMetricsReady?.invoke(metrics)

        Log.d(TAG, "Gaze: region=$focusRegion, stability=${String.format("%.2f", gazeStability)}, " +
                "attention=${String.format("%.2f", attentionScore)}")

        return metrics
    }

    /**
     * Process multiple faces and aggregate gaze metrics.
     */
    fun processMultiple(faces: List<Face>, imageWidth: Int, imageHeight: Int): List<GazeMetrics> {
        return faces.map { face ->
            process(face, imageWidth, imageHeight)
        }
    }

    /**
     * Estimate gaze point in normalized screen coordinates (0-1, 0-1).
     *
     * Uses a combination of:
     * - Eye landmarks (iris position within eye socket)
     * - Head rotation (where the head is pointing)
     * - Face position in frame
     */
    private fun estimateGazePoint(
        face: Face,
        headPitch: Float,
        headYaw: Float,
        imageWidth: Int,
        imageHeight: Int
    ): Pair<Float, Float> {
        // Get face center in normalized coordinates
        val faceBounds = face.boundingBox
        val faceCenterX = (faceBounds.centerX().toFloat() / imageWidth)
        val faceCenterY = (faceBounds.centerY().toFloat() / imageHeight)

        // Get eye landmarks if available
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)

        // Calculate eye center in normalized coordinates
        val eyeCenterX = if (leftEye != null && rightEye != null) {
            ((leftEye.position.x + rightEye.position.x) / 2f) / imageWidth
        } else {
            faceCenterX
        }

        val gazeCfg = SensingConfig.get().gaze
        val eyeCenterY = if (leftEye != null && rightEye != null) {
            ((leftEye.position.y + rightEye.position.y) / 2f) / imageHeight
        } else {
            faceCenterY * gazeCfg.eyeFallbackYFactor  // Eyes are typically in upper portion of face
        }

        // Adjust for head rotation
        // Yaw affects X position (looking left = gaze shifts right in camera view)
        // Pitch affects Y position (looking up = gaze shifts up)
        val yawAdjustment = -headYaw / gazeCfg.yawAngleDivisor * gazeCfg.yawMaxAdjustment
        val pitchAdjustment = -headPitch / gazeCfg.pitchAngleDivisor * gazeCfg.pitchMaxAdjustment

        // Calculate final gaze point
        val gazeX = (eyeCenterX + yawAdjustment).coerceIn(0f, 1f)
        val gazeY = (eyeCenterY + pitchAdjustment).coerceIn(0f, 1f)

        return Pair(gazeX, gazeY)
    }

    /**
     * Map normalized gaze coordinates to 3x3 grid region.
     */
    private fun mapToGridRegion(gazeX: Float, gazeY: Float): Int {
        val col = (gazeX * GRID_COLS).toInt().coerceIn(0, GRID_COLS - 1)
        val row = (gazeY * GRID_ROWS).toInt().coerceIn(0, GRID_ROWS - 1)
        return row * GRID_COLS + col
    }

    /**
     * Calculate gaze stability based on movement from previous frames.
     * Lower movement = higher stability (focused attention).
     */
    private fun calculateGazeStability(currentX: Float, currentY: Float): Float {
        val prevX = previousGazeX
        val prevY = previousGazeY

        if (prevX == null || prevY == null) {
            previousGazeX = currentX
            previousGazeY = currentY
            return SensingConfig.get().gaze.defaultStability  // Unknown - assume moderate stability
        }

        // Calculate distance from previous gaze point
        val distance = sqrt(
            (currentX - prevX).pow(2) + (currentY - prevY).pow(2)
        )

        // Update previous values
        previousGazeX = currentX
        previousGazeY = currentY

        // Convert distance to stability (inverse relationship)
        // Small distance = high stability, large distance = low stability
        // Divisor of 0.15: values > 0.15 normalized units indicate actual head movement
        // (not natural micro-adjustments/saccades which are typically 0.02-0.05 units)
        // Previous divisor of 0.5 meant normal eye movement showed ~50% stability
        return (1f - (distance / SensingConfig.get().gaze.stabilityDivisor)).coerceIn(0f, 1f)
    }

    /**
     * Update gaze history for stability calculations.
     */
    private fun updateGazeHistory(x: Float, y: Float) {
        gazeHistory.add(Pair(x, y))
        if (gazeHistory.size > maxHistorySize) {
            gazeHistory.removeAt(0)
        }
    }

    /**
     * Calculate overall attention score.
     */
    private fun calculateAttentionScore(
        focusRegion: Int,
        gazeStability: Float,
        isLookingAtScreen: Boolean,
        headYaw: Float
    ): Float {
        val cfg = SensingConfig.get().gaze
        if (!isLookingAtScreen) return cfg.notLookingAttentionScore
        var score = 0f

        // Focus region matters (center is best)
        score += when (focusRegion) {
            CENTER_REGION -> cfg.centerFocusScore
            1, 3, 5, 7 -> cfg.adjacentFocusScore
            else -> cfg.cornerFocusScore
        }

        // Gaze stability (focused attention)
        score += gazeStability * cfg.stabilityWeight

        // Head directly facing (not turned)
        val headDirectness = 1f - (abs(headYaw) / cfg.headDirectnessDivisor).coerceIn(0f, 1f)
        score += headDirectness * cfg.headDirectnessWeight

        return score.coerceIn(0f, 1f)
    }

    /**
     * Aggregate gaze metrics from multiple faces over a time window.
     */
    fun aggregateMetrics(samples: List<GazeMetrics>): AggregatedGazeMetrics {
        if (samples.isEmpty()) {
            return AggregatedGazeMetrics()
        }

        val windowStart = samples.minOfOrNull { it.timestamp } ?: 0
        val windowEnd = samples.maxOfOrNull { it.timestamp } ?: 0

        // Calculate primary focus region (most common)
        val regionCounts = samples
            .filter { it.isLookingAtScreen }
            .groupBy { it.focusRegion }
            .mapValues { it.value.size }

        val primaryRegion = regionCounts.maxByOrNull { it.value }?.key ?: CENTER_REGION

        // Calculate looking at screen percentage
        val lookingCount = samples.count { it.isLookingAtScreen }
        val lookingAtScreenPct = lookingCount.toFloat() / samples.size

        // Average gaze stability
        val avgStability = samples
            .filter { it.isLookingAtScreen }
            .map { it.gazeStability }
            .average().toFloat()

        // Build region heatmap (normalized)
        val totalLooking = lookingCount.coerceAtLeast(1)
        val heatmap = (0 until 9).associateWith { region ->
            (regionCounts[region] ?: 0).toFloat() / totalLooking
        }

        // Average attention score
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
     * Update zone dwell time when gaze moves to a new zone.
     */
    private fun updateZoneDwell(newZone: Int) {
        val now = System.currentTimeMillis()

        if (lastZoneChangeMs > 0 && currentZone in 0..8) {
            val dwellMs = now - lastZoneChangeMs
            // Cap individual dwell to 10s to avoid stale data from paused processing
            zoneDwellMs[currentZone] += dwellMs.coerceAtMost(10_000L)
        }

        if (newZone != currentZone || lastZoneChangeMs == 0L) {
            currentZone = newZone
            lastZoneChangeMs = now
        } else {
            lastZoneChangeMs = now
        }
    }

    /**
     * Get normalized zone dwell distribution for the current 10s window.
     * Returns a map of zone index (0-8) to dwell percentage (0.0-1.0).
     * Only includes zones with non-zero dwell time.
     */
    fun getZoneDistribution(): Map<Int, Float> {
        // Flush current zone's accumulated time
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

    /**
     * Reset zone dwell accumulators for a new 10s window.
     */
    fun resetZoneDwell() {
        zoneDwellMs.fill(0)
        lastZoneChangeMs = System.currentTimeMillis()
    }

    /**
     * Reset tracking state.
     */
    fun reset() {
        previousGazeX = null
        previousGazeY = null
        gazeHistory.clear()
        zoneDwellMs.fill(0)
        lastZoneChangeMs = 0
    }
}
