package com.trillboards.ctv.core.sensing

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for CsiAggregator (mirrors agent-core-lite test).
 *
 * Tests the aggregation logic: subcarrier variance computation,
 * occupancy estimation, motion scoring, and signal quality.
 */
class CsiAggregatorTest {

    @Before
    fun setUp() {
        CsiAggregator.clear()
    }

    @After
    fun tearDown() {
        CsiAggregator.clear()
    }

    private fun makeFrame(
        nodeId: Int = 1,
        nSubcarriers: Int = 8,
        sequence: Int = 0,
        rssi: Int = -45,
        amplitudes: FloatArray? = null,
        receivedAtMs: Long = System.currentTimeMillis()
    ): CsiFrame {
        val amps = amplitudes ?: FloatArray(nSubcarriers) { 10.0f }
        return CsiFrame(
            nodeId = nodeId,
            nAntennas = 1,
            nSubcarriers = nSubcarriers,
            freqMhz = 2412,
            sequence = sequence,
            rssi = rssi,
            noiseFloor = -90,
            amplitudes = amps,
            phases = FloatArray(nSubcarriers),
            receivedAtMs = receivedAtMs
        )
    }

    @Test
    fun `aggregate should return null with no frames`() {
        val snapshot = CsiAggregator.aggregate()
        assertNull(snapshot)
    }

    @Test
    fun `aggregate should return null for frames outside window`() {
        // Add frames that are 2 minutes old
        val oldTime = System.currentTimeMillis() - 120_000L
        CsiAggregator.addFrame(makeFrame(receivedAtMs = oldTime))

        val snapshot = CsiAggregator.aggregate()
        assertNull(snapshot)
    }

    @Test
    fun `aggregate should produce snapshot from recent frames`() {
        val now = System.currentTimeMillis()
        for (i in 0 until 100) {
            CsiAggregator.addFrame(makeFrame(
                sequence = i,
                receivedAtMs = now - (5000 - i * 50L)  // spread across last 5s, well within 10s window
            ))
        }

        val snapshot = CsiAggregator.aggregate()
        assertNotNull(snapshot)
        assertEquals(1, snapshot!!.nodeId)
        assertEquals(100, snapshot.framesProcessed)
        assertEquals("esp32-s3", snapshot.hardwareType)
    }

    @Test
    fun `aggregate should select primary node with most frames`() {
        val now = System.currentTimeMillis()
        // Add 10 frames from node 1
        for (i in 0 until 10) {
            CsiAggregator.addFrame(makeFrame(nodeId = 1, sequence = i, receivedAtMs = now - 5000))
        }
        // Add 20 frames from node 2
        for (i in 0 until 20) {
            CsiAggregator.addFrame(makeFrame(nodeId = 2, sequence = 100 + i, receivedAtMs = now - 5000))
        }

        val snapshot = CsiAggregator.aggregate()
        assertNotNull(snapshot)
        assertEquals(2, snapshot!!.nodeId)  // Node 2 has more frames
        assertEquals(20, snapshot.framesProcessed)
    }

    @Test
    fun `computeSubcarrierVariances should return zero variance for constant amplitudes`() {
        val frames = (0 until 10).map {
            makeFrame(nSubcarriers = 4, amplitudes = floatArrayOf(10f, 10f, 10f, 10f))
        }

        val variances = CsiAggregator.computeSubcarrierVariances(frames, 4)

        assertEquals(4, variances.size)
        for (v in variances) {
            assertEquals(0.0f, v, 0.001f)
        }
    }

    @Test
    fun `computeSubcarrierVariances should compute positive variance for varying amplitudes`() {
        val frames = listOf(
            makeFrame(nSubcarriers = 2, amplitudes = floatArrayOf(0f, 10f)),
            makeFrame(nSubcarriers = 2, amplitudes = floatArrayOf(10f, 10f)),
            makeFrame(nSubcarriers = 2, amplitudes = floatArrayOf(20f, 10f))
        )

        val variances = CsiAggregator.computeSubcarrierVariances(frames, 2)

        // Subcarrier 0: values [0, 10, 20], mean=10, variance = (100+0+100)/3 = 66.67
        assertTrue(variances[0] > 0)
        // Subcarrier 1: values [10, 10, 10], variance = 0
        assertEquals(0.0f, variances[1], 0.001f)
    }

    @Test
    fun `estimateOccupancy should return 0 for low variance`() {
        val variances = FloatArray(52) { 1.0f }  // All below threshold
        val count = CsiAggregator.estimateOccupancy(variances)
        assertEquals(0, count)
    }

    @Test
    fun `estimateOccupancy should estimate based on active subcarriers`() {
        // 16 subcarriers above threshold, 8 per occupant = 2 occupants
        val variances = FloatArray(52) { if (it < 16) 10.0f else 1.0f }
        val count = CsiAggregator.estimateOccupancy(variances)
        assertEquals(2, count)
    }

    @Test
    fun `estimateOccupancy should clamp to MAX_OCCUPANT_COUNT`() {
        // All 52 subcarriers above threshold = 52/8 = 6, well below max
        val variances = FloatArray(52) { 100.0f }
        val count = CsiAggregator.estimateOccupancy(variances)
        assertTrue(count in 0..50)
    }

    @Test
    fun `computeMotionScore should return 0 for zero variance`() {
        val variances = FloatArray(10) { 0.0f }
        val score = CsiAggregator.computeMotionScore(variances)
        assertEquals(0.0f, score, 0.001f)
    }

    @Test
    fun `computeMotionScore should be between 0 and 1`() {
        val variances = FloatArray(10) { 25.0f }
        val score = CsiAggregator.computeMotionScore(variances)
        assertTrue(score in 0.0f..1.0f)
    }

    @Test
    fun `computeMotionScore should clamp at 1 for high variance`() {
        val variances = FloatArray(10) { 1000.0f }
        val score = CsiAggregator.computeMotionScore(variances)
        assertEquals(1.0f, score, 0.001f)
    }

    @Test
    fun `computeMotionScore should return 0 for empty array`() {
        val score = CsiAggregator.computeMotionScore(FloatArray(0))
        assertEquals(0.0f, score, 0.001f)
    }

    @Test
    fun `aggregate should compute positive captureRateHz`() {
        val now = System.currentTimeMillis()
        for (i in 0 until 50) {
            CsiAggregator.addFrame(makeFrame(sequence = i, receivedAtMs = now - (5000 - i * 100L)))
        }

        val snapshot = CsiAggregator.aggregate()
        assertNotNull(snapshot)
        assertTrue(snapshot!!.captureRateHz > 0)
    }

    @Test
    fun `aggregate should compute signal quality between 0 and 1`() {
        val now = System.currentTimeMillis()
        for (i in 0 until 100) {
            CsiAggregator.addFrame(makeFrame(sequence = i, receivedAtMs = now - 5000))
        }

        val snapshot = CsiAggregator.aggregate()
        assertNotNull(snapshot)
        assertTrue(snapshot!!.signalQuality in 0.0f..1.0f)
    }

    @Test
    fun `bufferSize should reflect added frames`() {
        assertEquals(0, CsiAggregator.bufferSize())

        CsiAggregator.addFrame(makeFrame())
        assertEquals(1, CsiAggregator.bufferSize())

        CsiAggregator.addFrame(makeFrame(sequence = 1))
        assertEquals(2, CsiAggregator.bufferSize())
    }

    @Test
    fun `clear should empty the buffer`() {
        CsiAggregator.addFrame(makeFrame())
        CsiAggregator.addFrame(makeFrame(sequence = 1))
        assertEquals(2, CsiAggregator.bufferSize())

        CsiAggregator.clear()
        assertEquals(0, CsiAggregator.bufferSize())
    }

    @Test
    fun `addFrames should accept batch of frames`() {
        val frames = (0 until 5).map { makeFrame(sequence = it) }
        CsiAggregator.addFrames(frames)
        assertEquals(5, CsiAggregator.bufferSize())
    }

    @Test
    fun `aggregate window should respect custom duration`() {
        val now = System.currentTimeMillis()
        // Add frames 3 seconds ago
        for (i in 0 until 10) {
            CsiAggregator.addFrame(makeFrame(sequence = i, receivedAtMs = now - 3000))
        }
        // Add frames 15 seconds ago
        for (i in 0 until 10) {
            CsiAggregator.addFrame(makeFrame(sequence = 100 + i, receivedAtMs = now - 15000))
        }

        // 5-second window should only get the recent 10 frames
        val snapshot = CsiAggregator.aggregate(windowMs = 5000L)
        assertNotNull(snapshot)
        assertEquals(10, snapshot!!.framesProcessed)
    }

    @Test
    fun `constants should have expected values`() {
        assertEquals(60_000L, CsiAggregator.BUFFER_DURATION_MS)
        assertEquals(10_000L, CsiAggregator.WINDOW_DURATION_MS)
    }
}
