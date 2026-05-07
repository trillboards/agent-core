package com.trillboards.ctv.core.sensing

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [PairedTrainingSample] data integrity:
 * - byte serialization round-trip
 * - identity equality (UUID-based)
 * - keypoint shape validation helpers
 *
 * The serializer is the contract used by [PairedSampleStore]; any change
 * here ripples through every stored sample.
 */
class PairedTrainingSampleTest {

    @Test
    fun `serializeFloats then deserializeFloats round-trips`() {
        val original = floatArrayOf(0.1f, -0.5f, 12345.6789f, 0.0f, Float.MAX_VALUE, Float.MIN_VALUE)
        val bytes = PairedTrainingSample.serializeFloats(original)
        val decoded = PairedTrainingSample.deserializeFloats(bytes)
        assertEquals(original.size, decoded.size)
        assertArrayEquals(original, decoded, 0.0f)
    }

    @Test
    fun `serializeFloats produces 4 bytes per float`() {
        val original = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val bytes = PairedTrainingSample.serializeFloats(original)
        assertEquals(20, bytes.size)
    }

    @Test
    fun `serializeFloats handles empty array`() {
        val bytes = PairedTrainingSample.serializeFloats(floatArrayOf())
        assertEquals(0, bytes.size)
        val decoded = PairedTrainingSample.deserializeFloats(bytes)
        assertEquals(0, decoded.size)
    }

    @Test
    fun `deserializeFloats throws on truncated buffer`() {
        // 9 bytes is not divisible by 4 -> invalid
        val bad = ByteArray(9)
        var threw = false
        try {
            PairedTrainingSample.deserializeFloats(bad)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue("expected IllegalArgumentException for non-multiple-of-4 buffer", threw)
    }

    @Test
    fun `serializeCsiWindow accepts variable subcarrier counts`() {
        val window56 = FloatArray(20 * 56) { it.toFloat() }
        val bytes56 = PairedTrainingSample.serializeFloats(window56)
        assertEquals(20 * 56 * 4, bytes56.size)

        val window128 = FloatArray(20 * 128) { it.toFloat() / 100f }
        val bytes128 = PairedTrainingSample.serializeFloats(window128)
        assertEquals(20 * 128 * 4, bytes128.size)
    }

    @Test
    fun `equals uses sampleId only`() {
        val a = sampleWith(id = "a", overall = 0.9f)
        val a2 = sampleWith(id = "a", overall = 0.1f)  // different overall but same id
        val b = sampleWith(id = "b", overall = 0.9f)
        assertEquals(a, a2)
        assertNotEquals(a, b)
    }

    @Test
    fun `hashCode is consistent with equals`() {
        val a = sampleWith(id = "abc", overall = 0.42f)
        val a2 = sampleWith(id = "abc", overall = 0.99f)
        assertEquals(a.hashCode(), a2.hashCode())
    }

    @Test
    fun `keypoints has 17 joints with x,y pairs (34 floats)`() {
        val sample = sampleWith()
        assertEquals(34, sample.keypoints.size)
        assertEquals(17, sample.keypointConfidences.size)
    }

    @Test
    fun `windowEndMs greater than or equal to windowStartMs`() {
        val sample = sampleWith()
        assertTrue(sample.windowEndMs >= sample.windowStartMs)
    }

    private fun sampleWith(
        id: String = "00000000-0000-0000-0000-000000000001",
        overall: Float = 0.85f
    ): PairedTrainingSample {
        return PairedTrainingSample(
            sampleId = id,
            csiWindowBytes = ByteArray(20 * 56 * 4),
            subcarrierCount = 56,
            keypoints = FloatArray(34) { it.toFloat() / 34f },
            keypointConfidences = FloatArray(17) { 0.9f },
            overallConfidence = overall,
            numCameraFrames = 5,
            windowStartMs = 1_712_000_000_000L,
            windowEndMs = 1_712_000_000_200L,
            screenMongoId = "screen-1",
            venueType = "retail",
            cameraModelVersion = "mediapipe-blazepose-lite-v0.10.33"
        )
    }
}
