package com.trillboards.ctv.core.sensing

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Tests for [CsiPairedDataCollector].
 *
 * The collector pairs camera pose observations with the most recent CSI window
 * (20 frames at 100Hz = 200ms). A pair is valid only if:
 *   1) The CSI window has 20 frames within the 200ms window ending at pose time
 *   2) The pose observation's overallConfidence >= minConfidence
 *   3) The CSI subcarrier count is consistent across the window
 *   4) The pose timestamp is within `maxTimestampDriftMs` of the latest CSI frame
 *
 * Tests use:
 *   - [InMemoryPairedSampleStore] for verification
 *   - [FakeCsiFrameSource] for deterministic CSI input
 *   - direct calls to onPoseObservation() so we don't need a coroutine clock
 */
class CsiPairedDataCollectorTest {

    private lateinit var store: InMemoryPairedSampleStore
    private lateinit var csi: FakeCsiFrameSource
    private lateinit var pose: MediaPipePoseAdapter
    private lateinit var collector: CsiPairedDataCollector

    @Before
    fun setUp() {
        store = InMemoryPairedSampleStore()
        csi = FakeCsiFrameSource()
        pose = MediaPipePoseAdapter()
        collector = CsiPairedDataCollector(
            csiFrameSource = csi,
            poseProvider = pose,
            sampleStore = store,
            screenMongoId = "screen-001",
            venueType = "retail",
            cameraModelVersion = "mediapipe-blazepose-lite-v0.10.33",
            minConfidence = 0.5f
        )
    }

    @Test
    fun `pose without csi frames is skipped (timestamp mismatch)`() = runBlocking {
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))
        assertEquals(0, store.count())
        assertEquals(1, collector.getStats().skippedTimestampMismatch)
    }

    @Test
    fun `pose with low confidence is skipped`() = runBlocking {
        // Provide a full CSI window
        for (i in 0 until 20) {
            csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        }
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.3f))

        assertEquals(0, store.count())
        assertEquals(1, collector.getStats().skippedLowConfidence)
        assertEquals(0, collector.getStats().skippedTimestampMismatch)
    }

    @Test
    fun `valid pose with CSI window writes one paired sample`() = runBlocking {
        for (i in 0 until 20) {
            csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        }
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))

        assertEquals(1, store.count())
        val sample = store.getUnuploadedBatch(10)[0]
        assertEquals("screen-001", sample.screenMongoId)
        assertEquals("retail", sample.venueType)
        assertEquals("mediapipe-blazepose-lite-v0.10.33", sample.cameraModelVersion)
        assertEquals(56, sample.subcarrierCount)
        assertEquals(20, sample.numCameraFrames)
        assertTrue(sample.windowEndMs >= sample.windowStartMs)
        assertEquals(0.9f, sample.overallConfidence, 1e-6f)
    }

    @Test
    fun `csi window byte length matches 20 frames x subcarriers x 4 bytes`() = runBlocking {
        for (i in 0 until 20) {
            csi.add(frameAt(time = 800L + i * 10L, sequence = i, nSubcarriers = 56))
        }
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))

        val sample = store.getUnuploadedBatch(1)[0]
        assertEquals(20 * 56 * 4, sample.csiWindowBytes.size)
    }

    @Test
    fun `csi window with mixed subcarrier counts is rejected`() = runBlocking {
        // 19 frames with 56 subcarriers, then one with 64 — collector should reject
        for (i in 0 until 19) {
            csi.add(frameAt(time = 800L + i * 10L, sequence = i, nSubcarriers = 56))
        }
        csi.add(frameAt(time = 990L, sequence = 19, nSubcarriers = 64))

        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))
        assertEquals(0, store.count())
        assertEquals(1, collector.getStats().skippedTimestampMismatch)
    }

    @Test
    fun `partial CSI window (less than 20 frames) is rejected`() = runBlocking {
        for (i in 0 until 10) {
            csi.add(frameAt(time = 850L + i * 10L, sequence = i))
        }
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))
        assertEquals(0, store.count())
        assertEquals(1, collector.getStats().skippedTimestampMismatch)
    }

    @Test
    fun `pose timestamp far ahead of CSI is rejected (drift)`() = runBlocking {
        // CSI window ends at 1000, pose timestamp at 5000 -> 4s drift, way over budget
        for (i in 0 until 20) {
            csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        }
        collector.testOnPoseObservation(observation(time = 5000L, confidence = 0.9f))
        assertEquals(0, store.count())
        assertEquals(1, collector.getStats().skippedTimestampMismatch)
    }

    @Test
    fun `collection stops at targetSampleCount`() = runBlocking {
        collector.setTargetSampleCount(3)

        repeat(5) { idx ->
            for (i in 0 until 20) {
                csi.add(frameAt(time = 800L + idx * 1000L + i * 10L, sequence = idx * 100 + i))
            }
            collector.testOnPoseObservation(
                observation(time = 1000L + idx * 1000L, confidence = 0.9f)
            )
        }

        assertEquals(3, store.count())
        val stats = collector.getStats()
        assertEquals(3, stats.samplesCollected)
        assertEquals(3, stats.samplesTarget)
    }

    @Test
    fun `getStats reflects accumulated counters`() = runBlocking {
        // 1 valid sample
        for (i in 0 until 20) csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))

        // 1 low-confidence skip
        for (i in 0 until 20) csi.add(frameAt(time = 1800L + i * 10L, sequence = 100 + i))
        collector.testOnPoseObservation(observation(time = 2000L, confidence = 0.2f))

        // 1 timestamp-mismatch skip (no fresh CSI)
        collector.testOnPoseObservation(observation(time = 9999L, confidence = 0.9f))

        val stats = collector.getStats()
        assertEquals(1, stats.samplesCollected)
        assertEquals(1, stats.skippedLowConfidence)
        assertEquals(1, stats.skippedTimestampMismatch)
        assertTrue(stats.lastSampleMs > 0)
    }

    @Test
    fun `markUploaded flow updates uploadedCount stat`() = runBlocking {
        for (i in 0 until 20) csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))
        assertEquals(0, collector.getStats().uploadedCount)

        // Pretend we uploaded
        collector.testMarkUploaded(store.getUnuploadedBatch(10).map { it.sampleId })

        assertEquals(1, collector.getStats().uploadedCount)
        assertEquals(0, store.countUnuploaded())
    }

    @Test
    fun `clearUploaded removes uploaded samples but keeps unuploaded`() = runBlocking {
        // 2 valid samples
        for (i in 0 until 20) csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        collector.testOnPoseObservation(observation(time = 1000L, confidence = 0.9f))
        for (i in 0 until 20) csi.add(frameAt(time = 1800L + i * 10L, sequence = 100 + i))
        collector.testOnPoseObservation(observation(time = 2000L, confidence = 0.9f))

        val firstId = store.getUnuploadedBatch(10)[0].sampleId
        collector.testMarkUploaded(listOf(firstId))
        collector.clearUploaded()

        assertEquals(1, store.count())
        assertEquals(1, store.countUnuploaded())
    }

    @Test
    fun `subscribing collector to PoseKeypointProvider auto-pairs on push`() = runBlocking {
        // Wire the collector to the pose provider via the subscription path
        val subscription = collector.startSubscription()
        try {
            for (i in 0 until 20) csi.add(frameAt(time = 800L + i * 10L, sequence = i))
            pose.pushPose(
                PoseObservation(
                    keypoints = FloatArray(34) { 0f },
                    confidences = FloatArray(17) { 0.9f },
                    timestampMs = 1000L,
                    modelVersion = "mediapipe-blazepose-lite-v0.10.33"
                )
            )
            // The subscription path uses overallConfidence = avg(confidences) = 0.9
            assertEquals(1, store.count())
        } finally {
            subscription.close()
        }
    }

    @Test
    fun `closing subscription stops new pushes from being collected`() = runBlocking {
        val subscription = collector.startSubscription()
        for (i in 0 until 20) csi.add(frameAt(time = 800L + i * 10L, sequence = i))
        pose.pushPose(
            PoseObservation(
                keypoints = FloatArray(34) { 0f },
                confidences = FloatArray(17) { 0.9f },
                timestampMs = 1000L,
                modelVersion = "mediapipe-blazepose-lite-v0.10.33"
            )
        )
        assertEquals(1, store.count())
        subscription.close()

        // Push another observation — should NOT increase the count
        for (i in 0 until 20) csi.add(frameAt(time = 1800L + i * 10L, sequence = 100 + i))
        pose.pushPose(
            PoseObservation(
                keypoints = FloatArray(34) { 0f },
                confidences = FloatArray(17) { 0.9f },
                timestampMs = 2000L,
                modelVersion = "mediapipe-blazepose-lite-v0.10.33"
            )
        )
        assertEquals(1, store.count())
    }

    private fun observation(time: Long, confidence: Float): PoseObservation {
        return PoseObservation(
            keypoints = FloatArray(34) { it.toFloat() / 34f },
            confidences = FloatArray(17) { confidence },
            timestampMs = time,
            modelVersion = "mediapipe-blazepose-lite-v0.10.33"
        )
    }

    private fun frameAt(
        time: Long,
        sequence: Int,
        nSubcarriers: Int = 56,
        nodeId: Int = 1
    ): CsiFrame {
        return CsiFrame(
            nodeId = nodeId,
            nAntennas = 1,
            nSubcarriers = nSubcarriers,
            freqMhz = 2412,
            sequence = sequence,
            rssi = -45,
            noiseFloor = -90,
            amplitudes = FloatArray(nSubcarriers) { sequence + it.toFloat() / 100f },
            phases = FloatArray(nSubcarriers) { 0f },
            receivedAtMs = time
        )
    }
}

/**
 * Test-only implementation of [CsiFrameSource] backed by an in-memory list.
 */
private class FakeCsiFrameSource : CsiFrameSource {
    private val frames = mutableListOf<CsiFrame>()

    fun add(frame: CsiFrame) {
        frames.add(frame)
    }

    override fun getRecentFrames(sinceMs: Long): List<CsiFrame> {
        return if (sinceMs <= 0) {
            frames.toList()
        } else {
            frames.filter { it.receivedAtMs >= sinceMs }
        }
    }

    override fun clearBuffer() {
        frames.clear()
    }
}
