package com.trillboards.ctv.core.inference.vlm

import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import java.io.File
import kotlin.math.max

/**
 * Monitors native memory (VmRSS) after each VLM inference to detect leaks.
 *
 * For 24/7 kiosk operation, even 1KB/inference leaked means 30MB/month.
 * Android has no swap, so this eventually triggers OOM kill.
 *
 * Usage: call [checkAfterInference] after each VLM engine inference.
 * If RSS growth exceeds [thresholdMb] for [consecutiveThreshold] consecutive
 * readings, [onReloadRequired] fires to trigger model reload.
 *
 * @param modelSizeMb Size of the loaded model in megabytes. Used to compute a
 *   dynamic threshold: max(modelSizeMb * 0.15, 100). A 3GB model gets 450MB
 *   threshold; a 175MB model gets the 100MB floor. The old hardcoded 50MB was
 *   wrong for large models — it triggered false-positive reloads constantly.
 * @param thresholdMb Explicit threshold override. When modelSizeMb is provided,
 *   this is computed automatically. Only used when modelSizeMb is 0 (unknown).
 * @param consecutiveThreshold Number of consecutive over-threshold readings
 *   before triggering a reload.
 * @param onReloadRequired Callback invoked when a memory leak is detected.
 */
class VLMMemoryWatchdog(
    private val modelSizeMb: Float = 0f,
    private val thresholdMb: Float = if (modelSizeMb > 0f) max(modelSizeMb * SensingConfig.get().vlm.watchdogThresholdModelFraction, SensingConfig.get().vlm.watchdogThresholdMinMb) else SensingConfig.get().vlm.watchdogThresholdMinMb,
    private val consecutiveThreshold: Int = SensingConfig.get().vlm.watchdogConsecutiveThreshold,
    private val onReloadRequired: (() -> Unit)? = null,
    /**
     * Injectable RSS reader. Defaults to reading `/proc/self/status` (the real
     * Android device path). JVM unit tests pass `{ -1f }` to force the
     * early-return branch — without this, Linux CI runners (which DO expose
     * `/proc/self/status`) advance `inferenceCount` and break test fixtures
     * that were written assuming the call is a no-op outside Android.
     */
    private val rssReader: () -> Float = { defaultReadVmRssMb() }
) {
    companion object {
        private const val TAG = "VLMMemoryWatchdog"

        private fun defaultReadVmRssMb(): Float {
            return try {
                val status = File("/proc/self/status").readText()
                val rssLine = status.lineSequence().firstOrNull { it.startsWith("VmRSS:") }
                    ?: return -1f
                // Format: "VmRSS:    1234 kB"
                val kbStr = rssLine.substringAfter(":").trim().substringBefore(" ")
                kbStr.toFloatOrNull()?.div(1024f) ?: -1f
            } catch (e: Exception) {
                -1f
            }
        }
    }

    private var baselineRssMb: Float = -1f
    private var consecutiveOverThreshold: Int = 0
    private var inferenceCount: Int = 0
    private var lastRssMb: Float = 0f

    fun checkAfterInference() {
        val currentRss = rssReader()
        // currentRss <= 0 means VmRSS is unavailable (JVM tests injecting -1,
        // or pre-API-26 Android stripping /proc). On real devices VmRSS is
        // always positive so the early-return never fires.
        if (currentRss <= 0f) return

        inferenceCount++
        lastRssMb = currentRss

        // Set baseline after first inference
        if (baselineRssMb < 0f) {
            baselineRssMb = currentRss
            Log.i(TAG, "Baseline RSS: ${currentRss.toInt()}MB (inference #$inferenceCount)")
            return
        }

        val delta = currentRss - baselineRssMb

        // Periodic full report every N inferences
        if (inferenceCount % SensingConfig.get().vlm.watchdogReportInterval == 0) {
            Log.i(TAG, "Memory report: RSS=${currentRss.toInt()}MB, " +
                    "baseline=${baselineRssMb.toInt()}MB, delta=${delta.toInt()}MB, " +
                    "inferences=$inferenceCount")
        }

        // Check for sustained memory growth
        if (delta > thresholdMb) {
            consecutiveOverThreshold++
            Log.w(TAG, "RSS growth ${delta.toInt()}MB above baseline " +
                    "($consecutiveOverThreshold/$consecutiveThreshold consecutive)")

            if (consecutiveOverThreshold >= consecutiveThreshold) {
                Log.e(TAG, "Memory leak detected! RSS grew ${delta.toInt()}MB over " +
                        "$inferenceCount inferences. Triggering model reload.")
                consecutiveOverThreshold = 0
                baselineRssMb = -1f // Reset baseline after reload
                onReloadRequired?.invoke()
            }
        } else {
            consecutiveOverThreshold = 0
        }
    }

    fun reset() {
        baselineRssMb = -1f
        consecutiveOverThreshold = 0
        inferenceCount = 0
        lastRssMb = 0f
    }

    fun getLastRssMb(): Float = lastRssMb
    fun getInferenceCount(): Int = inferenceCount
}
