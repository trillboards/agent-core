package com.trillboards.ctv.core.sensing

import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Aggregates raw CSI frames from [CsiUdpListener] into [CsiSnapshot] summaries.
 *
 * Maintains a ring buffer of the last [BUFFER_DURATION_MS] (60 seconds) of frames.
 * On demand, computes a windowed aggregate ([WINDOW_DURATION_MS] = 10 seconds) that
 * produces occupancy estimates, motion scores, and signal quality metrics.
 *
 * ## Occupancy Estimation
 * Uses subcarrier amplitude variance thresholding. When a person moves through the
 * WiFi signal path, subcarrier amplitudes fluctuate. The number of subcarriers
 * exceeding the variance threshold correlates with the number of distinct signal
 * paths being disturbed, which maps to occupant count.
 *
 * ## Motion Score
 * Normalized amplitude variance across all subcarriers in the window.
 * 0.0 = completely still environment (no CSI variation).
 * 1.0 = maximum motion detected.
 *
 * ## Signal Quality
 * Ratio of frames successfully received to frames expected (based on the
 * ESP32's configured capture rate).
 *
 * Thread safety: frame buffer is a [CopyOnWriteArrayList], aggregation reads
 * are atomic snapshots.
 */
object CsiAggregator {

    private const val TAG = "CsiAggregator"

    /** Duration of the ring buffer in milliseconds (60 seconds). */
    const val BUFFER_DURATION_MS = 60_000L

    /** Aggregation window size in milliseconds (10 seconds). */
    const val WINDOW_DURATION_MS = 10_000L

    /**
     * Amplitude variance threshold for detecting a human-disturbed subcarrier.
     * Empirically tuned for ESP32-S3 at 2.4 GHz. Subcarriers with variance
     * above this threshold are considered "active" (disturbed by motion).
     */
    private const val OCCUPANCY_VARIANCE_THRESHOLD = 5.0f

    /**
     * Maximum expected occupant count. Clamps the estimate to avoid
     * unreasonable values from noisy environments.
     */
    private const val MAX_OCCUPANT_COUNT = 50

    /**
     * Number of active subcarriers per detected occupant. Tuned for
     * typical indoor environments with ESP32-S3.
     */
    private const val SUBCARRIERS_PER_OCCUPANT = 8

    /**
     * Maximum amplitude variance for motion score normalization.
     * Variance values above this are clamped to 1.0.
     */
    private const val MAX_MOTION_VARIANCE = 50.0f

    // Ring buffer of recent frames
    private val frameBuffer = CopyOnWriteArrayList<CsiFrame>()

    /**
     * Ingest a new CSI frame into the ring buffer.
     * Automatically evicts frames older than [BUFFER_DURATION_MS].
     */
    fun addFrame(frame: CsiFrame) {
        frameBuffer.add(frame)
        evictOldFrames()
    }

    /**
     * Ingest multiple CSI frames into the ring buffer.
     */
    fun addFrames(frames: List<CsiFrame>) {
        frameBuffer.addAll(frames)
        evictOldFrames()
    }

    /**
     * Compute an aggregated [CsiSnapshot] for the most recent window.
     *
     * @param windowMs Aggregation window duration (default [WINDOW_DURATION_MS])
     * @return [CsiSnapshot] or null if insufficient data
     */
    fun aggregate(windowMs: Long = WINDOW_DURATION_MS): CsiSnapshot? {
        val now = System.currentTimeMillis()
        val windowStart = now - windowMs
        val windowFrames = frameBuffer.filter { it.receivedAtMs >= windowStart }

        if (windowFrames.isEmpty()) {
            Log.d(TAG, "No frames in aggregation window")
            return null
        }

        // Determine the primary node (most frames in window)
        val framesByNode = windowFrames.groupBy { it.nodeId }
        val primaryNodeId = framesByNode.maxByOrNull { it.value.size }?.key ?: return null
        val nodeFrames = framesByNode[primaryNodeId] ?: return null

        val framesProcessed = nodeFrames.size
        val subcarrierCount = nodeFrames.firstOrNull()?.nSubcarriers ?: 0

        // Compute per-subcarrier amplitude variance across the window
        val occupantCount: Int
        val motionScore: Float

        if (subcarrierCount > 0 && nodeFrames.size >= 2) {
            val variances = computeSubcarrierVariances(nodeFrames, subcarrierCount)
            occupantCount = estimateOccupancy(variances)
            motionScore = computeMotionScore(variances)
        } else {
            occupantCount = 0
            motionScore = 0.0f
        }

        // Signal quality: frames received vs expected
        val windowDurationSec = windowMs / 1000.0f
        val captureRateHz = framesProcessed / windowDurationSec

        // Estimate expected frames from the first and last sequence numbers
        val framesExpected = estimateExpectedFrames(nodeFrames)
        val signalQuality = if (framesExpected > 0) {
            (framesProcessed.toFloat() / framesExpected).coerceIn(0.0f, 1.0f)
        } else {
            1.0f  // If we can't estimate, assume good quality
        }

        // Average RSSI
        val avgRssi = nodeFrames.map { it.rssi }.average().toInt()

        // Frames dropped estimate (expected - received, floor at 0)
        val framesDropped = (framesExpected - framesProcessed).coerceAtLeast(0)

        // Window boundaries
        val actualWindowStart = nodeFrames.minOf { it.receivedAtMs }
        val actualWindowEnd = nodeFrames.maxOf { it.receivedAtMs }

        return CsiSnapshot(
            nodeId = primaryNodeId,
            occupantCount = occupantCount,
            motionScore = motionScore,
            signalQuality = signalQuality,
            subcarrierCount = subcarrierCount,
            captureRateHz = captureRateHz,
            avgRssiDbm = avgRssi,
            framesProcessed = framesProcessed,
            framesDropped = framesDropped,
            windowStartMs = actualWindowStart,
            windowEndMs = actualWindowEnd
        )
    }

    /**
     * Clear the frame buffer.
     */
    fun clear() {
        frameBuffer.clear()
    }

    /**
     * Current number of frames in the ring buffer.
     */
    fun bufferSize(): Int = frameBuffer.size

    /**
     * Compute per-subcarrier amplitude variance across a set of frames.
     *
     * For each subcarrier index, computes the variance of its amplitude
     * values across all frames in the window.
     *
     * @return Array of variance values, one per subcarrier
     */
    internal fun computeSubcarrierVariances(
        frames: List<CsiFrame>,
        subcarrierCount: Int
    ): FloatArray {
        val variances = FloatArray(subcarrierCount)

        for (sc in 0 until subcarrierCount) {
            // Collect amplitude for this subcarrier across all frames
            var sum = 0.0f
            var sumSq = 0.0f
            var validCount = 0

            for (frame in frames) {
                if (sc < frame.amplitudes.size) {
                    val amp = frame.amplitudes[sc]
                    sum += amp
                    sumSq += amp * amp
                    validCount++
                }
            }

            if (validCount >= 2) {
                val mean = sum / validCount
                variances[sc] = (sumSq / validCount) - (mean * mean)
                // Clamp to zero for floating point rounding errors
                if (variances[sc] < 0) variances[sc] = 0.0f
            }
        }

        return variances
    }

    /**
     * Estimate occupant count from subcarrier variance distribution.
     *
     * Counts the number of subcarriers with variance above the threshold,
     * then divides by [SUBCARRIERS_PER_OCCUPANT].
     */
    internal fun estimateOccupancy(variances: FloatArray): Int {
        val activeSubcarriers = variances.count { it > OCCUPANCY_VARIANCE_THRESHOLD }
        val estimate = activeSubcarriers / SUBCARRIERS_PER_OCCUPANT
        return estimate.coerceIn(0, MAX_OCCUPANT_COUNT)
    }

    /**
     * Compute normalized motion score from subcarrier variances.
     *
     * Takes the mean variance across all subcarriers and normalizes
     * by [MAX_MOTION_VARIANCE].
     */
    internal fun computeMotionScore(variances: FloatArray): Float {
        if (variances.isEmpty()) return 0.0f
        val meanVariance = variances.average().toFloat()
        return (meanVariance / MAX_MOTION_VARIANCE).coerceIn(0.0f, 1.0f)
    }

    /**
     * Estimate the number of frames expected in the window based on sequence gaps.
     *
     * Uses the difference between first and last sequence numbers as a proxy
     * for expected frame count. Falls back to frame count if sequences
     * are not monotonically increasing (e.g., node restarted).
     */
    private fun estimateExpectedFrames(frames: List<CsiFrame>): Int {
        if (frames.size < 2) return frames.size

        val sorted = frames.sortedBy { it.sequence }
        val firstSeq = sorted.first().sequence
        val lastSeq = sorted.last().sequence

        // If sequence wrapped or is non-monotonic, fall back to actual count
        if (lastSeq <= firstSeq) return frames.size

        val seqRange = lastSeq - firstSeq + 1
        // Sanity check: expected frames should not be more than 10x actual
        return if (seqRange > frames.size * 10) {
            frames.size
        } else {
            seqRange
        }
    }

    /**
     * Remove frames older than [BUFFER_DURATION_MS].
     */
    private fun evictOldFrames() {
        val cutoff = System.currentTimeMillis() - BUFFER_DURATION_MS
        frameBuffer.removeAll { it.receivedAtMs < cutoff }
    }
}
