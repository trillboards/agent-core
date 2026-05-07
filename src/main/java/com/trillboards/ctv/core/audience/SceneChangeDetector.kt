package com.trillboards.ctv.core.audience

import android.graphics.Bitmap
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import java.util.concurrent.atomic.AtomicReference

/**
 * Detects significant scene changes by comparing frame luminance histograms.
 * Used to trigger re-classification and detect events like store opening/closing,
 * crowd arrivals/departures, or sudden environmental changes.
 *
 * Algorithm:
 * - Computes a luminance histogram from a sampled subset of pixels (~10K)
 * - Compares consecutive histograms using chi-squared distance
 * - Emits a SceneChangeEvent when the distance exceeds a threshold
 * - Rate-limits events to at most 1 per minute to avoid noise
 *
 * Privacy-Preserving: only processes pixel luminance values in-memory.
 * Never stores or transmits raw image data.
 *
 * Thread-safe: AtomicReference for previous histogram, volatile timestamp.
 */
class SceneChangeDetector {
    companion object {
        private const val TAG = "SceneChange"
        private val sceneCfg get() = SensingConfig.get().scene
        private val HISTOGRAM_BINS get() = sceneCfg.histogramBins
        private val CHANGE_THRESHOLD get() = sceneCfg.changeThreshold
        private val MIN_CHANGE_INTERVAL_MS get() = sceneCfg.minChangeIntervalMs
        private val SAMPLE_PIXELS get() = sceneCfg.samplePixels
    }

    /**
     * A detected scene change event with magnitude score.
     */
    data class SceneChangeEvent(
        val timestamp: Long,
        val changeScore: Float,
        val isSignificant: Boolean
    )

    private val lastHistogram = AtomicReference<FloatArray?>(null)
    @Volatile private var lastChangeTimestamp = 0L

    /**
     * Compare current frame against previous frame's histogram.
     * Returns a SceneChangeEvent only if the scene has changed significantly
     * AND the rate limit has not been exceeded.
     *
     * Returns null for the first frame (no baseline to compare against)
     * or if no significant change is detected.
     *
     * @param bitmap The camera frame in-memory bitmap.
     * @return SceneChangeEvent if significant change detected, null otherwise.
     */
    fun analyzeFrame(bitmap: Bitmap): SceneChangeEvent? {
        val histogram = computeHistogram(bitmap)
        val previous = lastHistogram.getAndSet(histogram) ?: return null

        val score = computeHistogramDifference(previous, histogram)
        val now = System.currentTimeMillis()
        val isSignificant = score > CHANGE_THRESHOLD &&
                (now - lastChangeTimestamp) > MIN_CHANGE_INTERVAL_MS

        if (isSignificant) {
            lastChangeTimestamp = now
            Log.i(TAG, "Scene change detected: score=${"%.3f".format(score)}")
        }

        return if (isSignificant) {
            SceneChangeEvent(now, score, true)
        } else null
    }

    /**
     * Compute a luminance histogram from a sampled subset of pixels.
     * Samples approximately [SAMPLE_PIXELS] pixels uniformly across the frame
     * to keep computation fast even on high-resolution inputs.
     */
    private fun computeHistogram(bitmap: Bitmap): FloatArray {
        val histogram = FloatArray(HISTOGRAM_BINS)
        val width = bitmap.width
        val height = bitmap.height

        // Calculate step sizes to sample ~SAMPLE_PIXELS total
        val totalPixels = width * height
        val sampleRatio = if (totalPixels > SAMPLE_PIXELS) {
            kotlin.math.sqrt(totalPixels.toDouble() / SAMPLE_PIXELS).toInt().coerceAtLeast(1)
        } else 1

        var pixelCount = 0
        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                // ITU-R BT.601 luminance
                val luminance = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
                val bin = (luminance * HISTOGRAM_BINS / 256).coerceIn(0, HISTOGRAM_BINS - 1)
                histogram[bin]++
                pixelCount++
                x += sampleRatio
            }
            y += sampleRatio
        }

        // Normalize to probability distribution
        if (pixelCount > 0) {
            for (i in histogram.indices) histogram[i] /= pixelCount.toFloat()
        }

        return histogram
    }

    /**
     * Compute chi-squared distance between two histograms.
     * Returns a value in [0, 1] where 0 = identical, 1 = maximally different.
     */
    private fun computeHistogramDifference(a: FloatArray, b: FloatArray): Float {
        var diff = 0f
        for (i in a.indices) {
            val sum = a[i] + b[i]
            if (sum > 0) {
                diff += (a[i] - b[i]) * (a[i] - b[i]) / sum
            }
        }
        return diff / 2  // Normalize chi-squared to 0-1 range
    }

    /**
     * Get the current change threshold. Can be read for diagnostics.
     */
    fun getThreshold(): Float = CHANGE_THRESHOLD

    /**
     * Reset all state. Call when the device changes location or
     * after a long period of inactivity where the baseline is stale.
     */
    fun reset() {
        lastHistogram.set(null)
        lastChangeTimestamp = 0L
        Log.i(TAG, "SceneChangeDetector reset")
    }
}
