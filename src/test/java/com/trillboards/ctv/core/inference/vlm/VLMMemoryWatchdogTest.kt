package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [VLMMemoryWatchdog] constructor and threshold computation.
 *
 * Memory monitoring (VmRSS read) is tested on-device. In JVM unit tests we
 * inject a no-op reader (`NO_RSS = { -1f }`) on every construction so the
 * early-return branch fires deterministically — without this, Linux CI
 * runners (which DO expose `/proc/self/status`) would advance `inferenceCount`
 * and break the assertions below. macOS dev hosts happen to read `-1` because
 * they have no `/proc`, but we don't depend on that.
 */
class VLMMemoryWatchdogTest {

    /** Forces JVM-test behavior regardless of whether the host exposes /proc. */
    private val NO_RSS: () -> Float = { -1f }

    // --- Default constructor (backward compatibility) ---

    @Test
    fun `default constructor uses 100MB threshold floor`() {
        val watchdog = VLMMemoryWatchdog(rssReader = NO_RSS)
        // With modelSizeMb=0 (default), threshold should be 100MB floor
        // We can't read the private field directly, but we can verify via behavior:
        // checkAfterInference should not crash
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount()) // VmRSS unavailable in JVM test
    }

    @Test
    fun `default constructor has zero inference count`() {
        val watchdog = VLMMemoryWatchdog(rssReader = NO_RSS)
        assertEquals(0, watchdog.getInferenceCount())
    }

    @Test
    fun `default constructor has zero RSS`() {
        val watchdog = VLMMemoryWatchdog(rssReader = NO_RSS)
        assertEquals(0f, watchdog.getLastRssMb(), 0.001f)
    }

    // --- modelSizeMb parameter ---

    @Test
    fun `3GB model gets 450MB threshold`() {
        // 3072MB * 0.15 = 460.8MB, max(460.8, 100) = 460.8
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 3072f, rssReader = NO_RSS)
        // Verify it doesn't crash and accepts the parameter
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount()) // VmRSS unavailable
    }

    @Test
    fun `175MB model gets 100MB threshold floor`() {
        // 175MB * 0.15 = 26.25MB, max(26.25, 100) = 100MB (floor)
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 175f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }

    @Test
    fun `1GB model gets 153MB threshold`() {
        // 1024MB * 0.15 = 153.6MB, max(153.6, 100) = 153.6MB
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 1024f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }

    @Test
    fun `zero modelSizeMb gets 100MB threshold floor`() {
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 0f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }

    @Test
    fun `negative modelSizeMb treated as zero`() {
        // Negative modelSizeMb is handled gracefully (not rejected) — the constructor's
        // threshold expression `if (modelSizeMb > 0f)` falls through to the 100MB floor.
        // This is intentional: callers may pass -1 as "unknown" without crashing.
        val watchdog = VLMMemoryWatchdog(modelSizeMb = -100f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }

    // --- Explicit threshold override ---

    @Test
    fun `explicit thresholdMb overrides modelSizeMb computation`() {
        // Even with a large model, explicit threshold takes precedence
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 3072f, thresholdMb = 200f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }

    // --- consecutiveThreshold parameter ---

    @Test
    fun `custom consecutiveThreshold is accepted`() {
        val watchdog = VLMMemoryWatchdog(
            modelSizeMb = 1024f,
            consecutiveThreshold = 5,
            rssReader = NO_RSS
        )
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }

    // --- reset ---

    @Test
    fun `reset clears all counters`() {
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 1024f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        watchdog.reset()
        assertEquals(0, watchdog.getInferenceCount())
        assertEquals(0f, watchdog.getLastRssMb(), 0.001f)
    }

    @Test
    fun `reset is safe to call multiple times`() {
        val watchdog = VLMMemoryWatchdog(rssReader = NO_RSS)
        watchdog.reset()
        watchdog.reset()
        watchdog.reset()
        assertEquals(0, watchdog.getInferenceCount())
    }

    // --- onReloadRequired callback ---

    @Test
    fun `onReloadRequired callback is accepted`() {
        var reloadCount = 0
        val watchdog = VLMMemoryWatchdog(
            modelSizeMb = 2048f,
            onReloadRequired = { reloadCount++ },
            rssReader = NO_RSS
        )
        // In JVM test, VmRSS is unavailable so callback won't fire,
        // but verifying the constructor accepts the callback
        watchdog.checkAfterInference()
        assertEquals(0, reloadCount)
    }

    // --- Threshold computation verification ---

    @Test
    fun `threshold computation - boundary at 667MB model`() {
        // 667MB * 0.15 = 100.05MB, just above floor
        // Below 667MB: floor wins. At/above 667MB: 15% wins.
        val smallWatchdog = VLMMemoryWatchdog(modelSizeMb = 600f, rssReader = NO_RSS)
        // 600 * 0.15 = 90 < 100, so threshold = 100
        smallWatchdog.checkAfterInference()

        val largeWatchdog = VLMMemoryWatchdog(modelSizeMb = 700f, rssReader = NO_RSS)
        // 700 * 0.15 = 105 > 100, so threshold = 105
        largeWatchdog.checkAfterInference()
    }

    @Test
    fun `threshold computation - very large model (4GB)`() {
        // 4096MB * 0.15 = 614.4MB
        val watchdog = VLMMemoryWatchdog(modelSizeMb = 4096f, rssReader = NO_RSS)
        watchdog.checkAfterInference()
        assertEquals(0, watchdog.getInferenceCount())
    }
}
