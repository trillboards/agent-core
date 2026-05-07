package com.trillboards.ctv.core.sensing

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [PairedSampleStore] using the in-memory implementation.
 *
 * The Room-backed implementation cannot be unit-tested without
 * Robolectric/instrumentation, so [InMemoryPairedSampleStore] mirrors
 * the contract one-to-one and is the production-equivalent under test.
 *
 * Both implementations must satisfy:
 *   - insert is idempotent on sample_id
 *   - getUnuploadedBatch returns oldest first, capped at limit
 *   - markUploaded transitions samples out of the unuploaded set
 *   - clearUploaded only removes uploaded samples
 *   - count and countUnuploaded are accurate
 */
class PairedSampleStoreTest {

    private lateinit var store: PairedSampleStore

    @Before
    fun setUp() {
        store = InMemoryPairedSampleStore()
    }

    @Test
    fun `insert and count returns one`() {
        store.insert(sample(id = "s1", windowStart = 100L))
        assertEquals(1, store.count())
        assertEquals(1, store.countUnuploaded())
    }

    @Test
    fun `insert is idempotent on sample_id`() {
        store.insert(sample(id = "dup", windowStart = 100L))
        store.insert(sample(id = "dup", windowStart = 200L))
        assertEquals(1, store.count())
    }

    @Test
    fun `getUnuploadedBatch returns oldest first`() {
        store.insert(sample(id = "newest", windowStart = 300L))
        store.insert(sample(id = "oldest", windowStart = 100L))
        store.insert(sample(id = "middle", windowStart = 200L))

        val batch = store.getUnuploadedBatch(limit = 10)
        assertEquals(3, batch.size)
        assertEquals("oldest", batch[0].sampleId)
        assertEquals("middle", batch[1].sampleId)
        assertEquals("newest", batch[2].sampleId)
    }

    @Test
    fun `getUnuploadedBatch respects limit`() {
        for (i in 0 until 10) {
            store.insert(sample(id = "s$i", windowStart = i.toLong()))
        }
        val batch = store.getUnuploadedBatch(limit = 3)
        assertEquals(3, batch.size)
        assertEquals("s0", batch[0].sampleId)
    }

    @Test
    fun `getUnuploadedBatch empty store returns empty list`() {
        assertEquals(0, store.getUnuploadedBatch(limit = 10).size)
    }

    @Test
    fun `markUploaded removes ids from unuploaded set`() {
        store.insert(sample(id = "s1", windowStart = 100L))
        store.insert(sample(id = "s2", windowStart = 200L))
        store.insert(sample(id = "s3", windowStart = 300L))

        store.markUploaded(listOf("s1", "s3"))

        assertEquals(3, store.count())
        assertEquals(1, store.countUnuploaded())

        val remaining = store.getUnuploadedBatch(limit = 10)
        assertEquals(1, remaining.size)
        assertEquals("s2", remaining[0].sampleId)
    }

    @Test
    fun `markUploaded ignores unknown ids`() {
        store.insert(sample(id = "s1", windowStart = 100L))
        store.markUploaded(listOf("unknown", "s1", "also-unknown"))
        assertEquals(0, store.countUnuploaded())
    }

    @Test
    fun `clearUploaded only removes uploaded samples`() {
        store.insert(sample(id = "u1", windowStart = 100L))
        store.insert(sample(id = "u2", windowStart = 200L))
        store.insert(sample(id = "kept", windowStart = 300L))

        store.markUploaded(listOf("u1", "u2"))
        store.clearUploaded()

        assertEquals(1, store.count())
        assertEquals(1, store.countUnuploaded())
        val remaining = store.getUnuploadedBatch(limit = 10)
        assertEquals("kept", remaining[0].sampleId)
    }

    @Test
    fun `clearUploaded on empty store is a no-op`() {
        store.clearUploaded()
        assertEquals(0, store.count())
    }

    @Test
    fun `inserted sample preserves byte payload`() {
        val window = FloatArray(20 * 56) { it.toFloat() }
        val keypoints = FloatArray(34) { it.toFloat() / 10f }
        val confidences = FloatArray(17) { 0.5f + it * 0.01f }

        val s = PairedTrainingSample(
            sampleId = "byte-test",
            csiWindowBytes = PairedTrainingSample.serializeFloats(window),
            subcarrierCount = 56,
            keypoints = keypoints,
            keypointConfidences = confidences,
            overallConfidence = 0.83f,
            numCameraFrames = 5,
            windowStartMs = 1_000L,
            windowEndMs = 1_200L,
            screenMongoId = "screen-byte",
            venueType = "retail",
            cameraModelVersion = "mediapipe-blazepose-lite-v0.10.33"
        )

        store.insert(s)
        val read = store.getUnuploadedBatch(10)[0]
        val decodedWindow = PairedTrainingSample.deserializeFloats(read.csiWindowBytes)
        assertArrayEquals(window, decodedWindow, 0.0f)
        assertArrayEquals(keypoints, read.keypoints, 0.0f)
        assertArrayEquals(confidences, read.keypointConfidences, 0.0f)
        assertEquals(56, read.subcarrierCount)
    }

    @Test
    fun `count is sum of uploaded and unuploaded`() {
        for (i in 0 until 5) store.insert(sample(id = "s$i", windowStart = i.toLong()))
        store.markUploaded(listOf("s0", "s1"))
        assertEquals(5, store.count())
        assertEquals(3, store.countUnuploaded())
    }

    private fun sample(id: String, windowStart: Long): PairedTrainingSample {
        return PairedTrainingSample(
            sampleId = id,
            csiWindowBytes = ByteArray(20 * 56 * 4),
            subcarrierCount = 56,
            keypoints = FloatArray(34) { 0f },
            keypointConfidences = FloatArray(17) { 0.9f },
            overallConfidence = 0.9f,
            numCameraFrames = 5,
            windowStartMs = windowStart,
            windowEndMs = windowStart + 200L,
            screenMongoId = null,
            venueType = null,
            cameraModelVersion = "mediapipe-blazepose-lite-v0.10.33"
        )
    }
}
