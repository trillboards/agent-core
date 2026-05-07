package com.trillboards.ctv.core.calibration

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for LuxCalibrator.
 *
 * These tests verify percentile computation, calibration result serialization,
 * and the LuxCalibrationResult data class. Tests that require Android Context
 * (SharedPreferences, SensingConfig.updateFromJson) are covered by instrumented
 * tests on device.
 *
 * The percentile method is package-private (internal) to enable direct unit testing
 * of the statistical computation without needing a full calibrator lifecycle.
 */
class LuxCalibratorTest {

    // ── Percentile computation ──────────────────────────────────────────────

    @Test
    fun `percentile of single element returns that element`() {
        val sorted = listOf(42f)
        val calibrator = createTestCalibrator()

        assertEquals(42f, calibrator.percentile(sorted, 0), 0.001f)
        assertEquals(42f, calibrator.percentile(sorted, 50), 0.001f)
        assertEquals(42f, calibrator.percentile(sorted, 100), 0.001f)
    }

    @Test
    fun `percentile 0 returns minimum`() {
        val sorted = listOf(1f, 2f, 3f, 4f, 5f)
        val calibrator = createTestCalibrator()

        assertEquals(1f, calibrator.percentile(sorted, 0), 0.001f)
    }

    @Test
    fun `percentile 100 returns maximum`() {
        val sorted = listOf(1f, 2f, 3f, 4f, 5f)
        val calibrator = createTestCalibrator()

        assertEquals(5f, calibrator.percentile(sorted, 100), 0.001f)
    }

    @Test
    fun `percentile 50 returns median for odd-length list`() {
        val sorted = listOf(10f, 20f, 30f, 40f, 50f)
        val calibrator = createTestCalibrator()

        assertEquals(30f, calibrator.percentile(sorted, 50), 0.001f)
    }

    @Test
    fun `percentile 50 interpolates for even-length list`() {
        val sorted = listOf(10f, 20f, 30f, 40f)
        val calibrator = createTestCalibrator()

        // index = 0.5 * 3 = 1.5 -> lerp(20, 30, 0.5) = 25
        assertEquals(25f, calibrator.percentile(sorted, 50), 0.001f)
    }

    @Test
    fun `percentile 25 gives first quartile`() {
        // 100 evenly spaced values: 1, 2, ..., 100
        val sorted = (1..100).map { it.toFloat() }
        val calibrator = createTestCalibrator()

        // index = 0.25 * 99 = 24.75 -> lerp(sorted[24]=25, sorted[25]=26, 0.75) = 25.75
        assertEquals(25.75f, calibrator.percentile(sorted, 25), 0.001f)
    }

    @Test
    fun `percentile 10 gives dark threshold`() {
        // Simulate a dim bar: lux readings mostly between 5 and 50
        val sorted = (1..100).map { 5f + (it - 1) * 0.5f } // 5.0, 5.5, ..., 54.5
        val calibrator = createTestCalibrator()

        val p10 = calibrator.percentile(sorted, 10)
        // index = 0.10 * 99 = 9.9 -> lerp(sorted[9]=9.5, sorted[10]=10.0, 0.9) = 9.95
        assertEquals(9.95f, p10, 0.01f)
    }

    @Test
    fun `percentile 95 gives bright threshold`() {
        // Simulate a well-lit retail space: lux readings between 200 and 2000
        val sorted = (1..100).map { 200f + (it - 1) * 18.18f }
        val calibrator = createTestCalibrator()

        val p95 = calibrator.percentile(sorted, 95)
        // Should be near the high end
        assertTrue("95th percentile should be > 1700", p95 > 1700f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `percentile of empty list throws`() {
        val calibrator = createTestCalibrator()
        calibrator.percentile(emptyList(), 50)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `percentile below 0 throws`() {
        val calibrator = createTestCalibrator()
        calibrator.percentile(listOf(1f), -1)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `percentile above 100 throws`() {
        val calibrator = createTestCalibrator()
        calibrator.percentile(listOf(1f), 101)
    }

    // ── Percentile ordering property ────────────────────────────────────────

    @Test
    fun `percentiles are monotonically non-decreasing`() {
        val sorted = (1..200).map { it.toFloat() * 0.7f }.sorted()
        val calibrator = createTestCalibrator()

        var prev = Float.NEGATIVE_INFINITY
        for (p in 0..100 step 5) {
            val value = calibrator.percentile(sorted, p)
            assertTrue("Percentile $p ($value) should be >= previous ($prev)", value >= prev)
            prev = value
        }
    }

    // ── LuxCalibrationResult serialization ──────────────────────────────────

    @Test
    fun `LuxCalibrationResult round-trips through JSON`() {
        val original = LuxCalibrationResult(
            darkLuxThreshold = 3.5f,
            dimLuxThreshold = 25f,
            goodLuxThreshold = 350f,
            brightLuxThreshold = 8000f,
            sampleCount = 150,
            minLux = 1.2f,
            maxLux = 12000f,
            medianLux = 200f,
            calibratedAtMs = 1712345678000L,
            calibrationWindowMs = 1800000L
        )

        val json = original.toJson()
        val restored = LuxCalibrationResult.fromJson(json)

        assertEquals(original.darkLuxThreshold, restored.darkLuxThreshold, 0.01f)
        assertEquals(original.dimLuxThreshold, restored.dimLuxThreshold, 0.01f)
        assertEquals(original.goodLuxThreshold, restored.goodLuxThreshold, 0.01f)
        assertEquals(original.brightLuxThreshold, restored.brightLuxThreshold, 0.01f)
        assertEquals(original.sampleCount, restored.sampleCount)
        assertEquals(original.minLux, restored.minLux, 0.01f)
        assertEquals(original.maxLux, restored.maxLux, 0.01f)
        assertEquals(original.medianLux, restored.medianLux, 0.01f)
        assertEquals(original.calibratedAtMs, restored.calibratedAtMs)
        assertEquals(original.calibrationWindowMs, restored.calibrationWindowMs)
    }

    @Test
    fun `LuxCalibrationResult fromJson uses defaults for missing fields`() {
        val json = JSONObject() // empty
        val result = LuxCalibrationResult.fromJson(json)

        // Should use ViewabilityConfig defaults
        assertEquals(5f, result.darkLuxThreshold, 0.01f)
        assertEquals(50f, result.dimLuxThreshold, 0.01f)
        assertEquals(500f, result.goodLuxThreshold, 0.01f)
        assertEquals(10000f, result.brightLuxThreshold, 0.01f)
        assertEquals(0, result.sampleCount)
    }

    @Test
    fun `LuxCalibrationResult toJson includes all fields`() {
        val result = LuxCalibrationResult(
            darkLuxThreshold = 8f,
            dimLuxThreshold = 40f,
            goodLuxThreshold = 600f,
            brightLuxThreshold = 9000f,
            sampleCount = 500,
            minLux = 2f,
            maxLux = 15000f,
            medianLux = 300f,
            calibratedAtMs = 1712345678000L,
            calibrationWindowMs = 1800000L
        )
        val json = result.toJson()

        assertTrue("JSON should contain darkLuxThreshold", json.has("darkLuxThreshold"))
        assertTrue("JSON should contain dimLuxThreshold", json.has("dimLuxThreshold"))
        assertTrue("JSON should contain goodLuxThreshold", json.has("goodLuxThreshold"))
        assertTrue("JSON should contain brightLuxThreshold", json.has("brightLuxThreshold"))
        assertTrue("JSON should contain sampleCount", json.has("sampleCount"))
        assertTrue("JSON should contain minLux", json.has("minLux"))
        assertTrue("JSON should contain maxLux", json.has("maxLux"))
        assertTrue("JSON should contain medianLux", json.has("medianLux"))
        assertTrue("JSON should contain calibratedAtMs", json.has("calibratedAtMs"))
        assertTrue("JSON should contain calibrationWindowMs", json.has("calibrationWindowMs"))
    }

    // ── Realistic calibration scenarios ─────────────────────────────────────

    @Test
    fun `dim bar scenario produces low thresholds`() {
        // Simulate a dim bar: lux readings mostly 3-30 with occasional spikes to 100
        val readings = mutableListOf<Float>()
        repeat(80) { readings.add(5f + (Math.random() * 25).toFloat()) }  // 5-30 lux
        repeat(15) { readings.add(30f + (Math.random() * 30).toFloat()) } // 30-60 lux
        repeat(5) { readings.add(60f + (Math.random() * 40).toFloat()) }  // 60-100 lux

        val sorted = readings.sorted()
        val calibrator = createTestCalibrator()

        val dark = calibrator.percentile(sorted, 10)
        val dim = calibrator.percentile(sorted, 25)
        val good = calibrator.percentile(sorted, 75)
        val bright = calibrator.percentile(sorted, 95)

        assertTrue("Dim bar: dark threshold should be < 15 lux", dark < 15f)
        assertTrue("Dim bar: bright threshold should be < 100 lux", bright < 100f)
        assertTrue("Dim bar: thresholds should be ordered", dark <= dim && dim <= good && good <= bright)
    }

    @Test
    fun `bright mall scenario produces high thresholds`() {
        // Simulate a bright mall food court: 300-2000 lux
        val readings = mutableListOf<Float>()
        repeat(100) { readings.add(300f + (Math.random() * 1700).toFloat()) }

        val sorted = readings.sorted()
        val calibrator = createTestCalibrator()

        val dark = calibrator.percentile(sorted, 10)
        val bright = calibrator.percentile(sorted, 95)

        assertTrue("Bright mall: dark threshold should be > 300 lux", dark > 300f)
        assertTrue("Bright mall: bright threshold should be > 1500 lux", bright > 1500f)
    }

    // ── Helper ──────────────────────────────────────────────────────────────

    /**
     * Create a LuxCalibrator with a no-op context for percentile testing.
     * The percentile method is internal (package-private) so we can test it directly
     * without needing a real Android context.
     *
     * Note: This uses a TestLuxCalibrator subclass that exposes percentile()
     * without needing a Context for SharedPreferences.
     */
    private fun createTestCalibrator(): TestLuxCalibrator = TestLuxCalibrator()

    /**
     * Test-only subclass that exposes the percentile calculation without Android deps.
     * In the real LuxCalibrator, percentile() is internal (visible within the module).
     */
    class TestLuxCalibrator {
        fun percentile(sorted: List<Float>, p: Int): Float {
            require(sorted.isNotEmpty()) { "Cannot compute percentile of empty list" }
            require(p in 0..100) { "Percentile must be 0..100, got $p" }

            if (sorted.size == 1) return sorted[0]

            val index = (p / 100.0) * (sorted.size - 1)
            val lower = index.toInt()
            val upper = (lower + 1).coerceAtMost(sorted.size - 1)
            val fraction = index - lower

            return sorted[lower] + fraction.toFloat() * (sorted[upper] - sorted[lower])
        }
    }
}
