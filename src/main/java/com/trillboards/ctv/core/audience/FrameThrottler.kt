package com.trillboards.ctv.core.audience

import android.util.Log
import com.trillboards.ctv.core.SensingConfig

/**
 * Frame throttler for adaptive face detection rate.
 *
 * Camera typically delivers frames at 30fps, but ML Kit face detection
 * doesn't need that frequency. This class implements frame skipping
 * to reduce CPU and memory pressure.
 *
 * The throttler is adaptive - it can be adjusted at runtime based on
 * memory pressure or battery state.
 */
class FrameThrottler(
    initialTargetFps: Int = SensingConfig.get().capture.initialTargetFps,
    cameraFps: Int = SensingConfig.get().capture.cameraFps
) {
    // Effective camera production rate. Starts from config (historically assumed
    // 30) but is corrected to the real rate once the AE target-FPS range is pinned
    // at bind time — see setCameraFps(). calculateSkipRate() decimates from this
    // down to targetFps, so a stale value makes the throttler over-skip.
    @Volatile
    private var cameraFps: Int = cameraFps.coerceAtLeast(1)
    companion object {
        private const val TAG = "FrameThrottler"
        val MIN_FPS get() = SensingConfig.get().capture.minFps
        val MAX_FPS get() = SensingConfig.get().capture.maxFps
    }

    @Volatile
    private var targetFps: Int = initialTargetFps.coerceIn(MIN_FPS, MAX_FPS)

    @Volatile
    private var frameCounter: Int = 0

    private var skipRate: Int = calculateSkipRate(targetFps)

    // Statistics
    private var processedFrames: Long = 0
    private var skippedFrames: Long = 0
    private var lastStatsTime: Long = System.currentTimeMillis()

    /**
     * Check if the current frame should be processed.
     * Call this for every frame from the camera.
     *
     * @return true if this frame should be processed, false to skip
     */
    fun shouldProcessFrame(): Boolean {
        frameCounter++

        if (frameCounter >= skipRate) {
            frameCounter = 0
            processedFrames++
            return true
        }

        skippedFrames++
        return false
    }

    /**
     * Update target FPS dynamically.
     * Use this to reduce FPS under memory pressure.
     *
     * @param newTargetFps New target frames per second (1-15)
     */
    fun setTargetFps(newTargetFps: Int) {
        val clamped = newTargetFps.coerceIn(MIN_FPS, MAX_FPS)
        if (clamped != targetFps) {
            val oldFps = targetFps
            targetFps = clamped
            skipRate = calculateSkipRate(clamped)
            frameCounter = 0  // Reset counter to avoid burst

            Log.i(TAG, "Target FPS changed: $oldFps -> $targetFps (skipRate: $skipRate)")
        }
    }

    /**
     * Correct the assumed camera production rate (frames/sec the HAL actually
     * delivers) and recompute the skip rate.
     *
     * Call this once the camera is bound with the effective AE target-FPS cap. If
     * left stale (e.g. assuming 30 while the AE range pins the HAL to 15) the
     * throttler over-skips — skipRate = cameraFps / targetFps would decimate from a
     * rate the camera never produces, so the analyzer runs below its target.
     *
     * @param newCameraFps Effective frames/sec the camera delivers (≥1).
     */
    fun setCameraFps(newCameraFps: Int) {
        val clamped = newCameraFps.coerceAtLeast(1)
        if (clamped != cameraFps) {
            val old = cameraFps
            cameraFps = clamped
            skipRate = calculateSkipRate(targetFps)
            Log.i(TAG, "Camera FPS corrected: $old -> $cameraFps (skipRate: $skipRate)")
        }
    }

    /**
     * Get current target FPS.
     */
    fun getTargetFps(): Int = targetFps

    /**
     * Auto-tune target FPS from measured inference latency.
     *
     * Computes the maximum FPS that leaves [utilizationTarget]% of the frame
     * budget for inference, with the remaining headroom for GC, scheduling, etc.
     *
     * Example: if Pose(30ms) + Face(25ms) + Emotion(20ms) = 75ms per frame,
     * maxFps = floor(1000 / 75 * 0.8) = 10fps
     *
     * @param avgInferenceMs Average end-to-end inference time per frame (milliseconds)
     * @param utilizationTarget Fraction of frame budget to use (0.0-1.0, default 0.8 = 80%)
     */
    fun autoTuneFromLatency(avgInferenceMs: Double, utilizationTarget: Float = 0.8f) {
        if (avgInferenceMs <= 0) return

        val maxFps = kotlin.math.floor(1000.0 / avgInferenceMs * utilizationTarget).toInt()
        val clamped = maxFps.coerceIn(MIN_FPS, MAX_FPS)

        Log.i(TAG, "Auto-tune: avgInference=${avgInferenceMs.toInt()}ms, " +
                "utilization=${(utilizationTarget * 100).toInt()}%, " +
                "computed=${maxFps}fps, clamped=${clamped}fps")

        setTargetFps(clamped)
    }

    /**
     * Calculate skip rate from target FPS.
     * skipRate = cameraFps / targetFps
     * e.g., 30fps camera / 6fps target = process every 5th frame
     */
    private fun calculateSkipRate(fps: Int): Int {
        return (cameraFps / fps.coerceAtLeast(1)).coerceAtLeast(1)
    }

    /**
     * Get processing statistics.
     */
    fun getStats(): ThrottlerStats {
        val now = System.currentTimeMillis()
        val durationMs = now - lastStatsTime
        val actualFps = if (durationMs > 0) {
            (processedFrames * 1000.0 / durationMs).toFloat()
        } else 0f

        return ThrottlerStats(
            targetFps = targetFps,
            actualFps = actualFps,
            skipRate = skipRate,
            processedFrames = processedFrames,
            skippedFrames = skippedFrames,
            skipRatio = if (processedFrames + skippedFrames > 0) {
                skippedFrames.toFloat() / (processedFrames + skippedFrames)
            } else 0f
        )
    }

    /**
     * Reset statistics counters.
     */
    fun resetStats() {
        processedFrames = 0
        skippedFrames = 0
        lastStatsTime = System.currentTimeMillis()
    }

    /**
     * Log current stats (call periodically for monitoring).
     */
    fun logStats() {
        val stats = getStats()
        Log.d(TAG, "Throttler: target=${stats.targetFps}fps, actual=${String.format("%.1f", stats.actualFps)}fps, " +
                "processed=${stats.processedFrames}, skipped=${stats.skippedFrames}, " +
                "skipRatio=${String.format("%.1f", stats.skipRatio * 100)}%")
    }

    /**
     * Throttler statistics.
     */
    data class ThrottlerStats(
        val targetFps: Int,
        val actualFps: Float,
        val skipRate: Int,
        val processedFrames: Long,
        val skippedFrames: Long,
        val skipRatio: Float
    )
}
