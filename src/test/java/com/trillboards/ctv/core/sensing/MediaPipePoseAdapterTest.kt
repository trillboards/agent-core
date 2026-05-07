package com.trillboards.ctv.core.sensing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [MediaPipePoseAdapter].
 *
 * The adapter has two responsibilities:
 *   1) Map MediaPipe Pose Landmarker's 33 BlazePose landmarks → 17 COCO keypoints
 *      (the format MoveNet/COCO/RuView WiFlow expect)
 *   2) Implement [PoseKeypointProvider] so consumers can passively observe pose
 *      results without owning the inference loop
 *
 * It does NOT load any MediaPipe models — it's a pure transform + pub/sub holder.
 */
class MediaPipePoseAdapterTest {

    @Test
    fun `mapBlazePoseToCoco returns 17 keypoints (34 floats)`() {
        // 33 landmarks, each (x, y, visibility)
        val blazepose = FloatArray(33 * 3) { it.toFloat() / 100f }
        val visibilities = FloatArray(33) { 1.0f }

        val (kp, conf) = MediaPipePoseAdapter.mapBlazePoseToCoco(blazepose, visibilities)
        assertEquals(34, kp.size)
        assertEquals(17, conf.size)
    }

    @Test
    fun `mapBlazePoseToCoco preserves nose at index 0`() {
        // BlazePose nose is at landmark 0 -> COCO nose is at COCO index 0 (x=0, y=1)
        val blazepose = FloatArray(33 * 3)
        // landmark 0: x=0.5, y=0.7
        blazepose[0] = 0.5f
        blazepose[1] = 0.7f
        val visibilities = FloatArray(33) { 1.0f }

        val (kp, _) = MediaPipePoseAdapter.mapBlazePoseToCoco(blazepose, visibilities)
        assertEquals(0.5f, kp[0], 1e-6f)  // COCO nose x
        assertEquals(0.7f, kp[1], 1e-6f)  // COCO nose y
    }

    @Test
    fun `mapBlazePoseToCoco preserves left_shoulder at COCO index 5`() {
        // BlazePose left_shoulder = landmark 11
        val blazepose = FloatArray(33 * 3)
        blazepose[11 * 3] = 0.4f
        blazepose[11 * 3 + 1] = 0.6f
        val visibilities = FloatArray(33) { 1.0f }

        val (kp, _) = MediaPipePoseAdapter.mapBlazePoseToCoco(blazepose, visibilities)
        // COCO left_shoulder is index 5 -> kp positions [10, 11]
        assertEquals(0.4f, kp[10], 1e-6f)
        assertEquals(0.6f, kp[11], 1e-6f)
    }

    @Test
    fun `mapBlazePoseToCoco preserves right_ankle at COCO index 16`() {
        // BlazePose right_ankle = landmark 28
        val blazepose = FloatArray(33 * 3)
        blazepose[28 * 3] = 0.85f
        blazepose[28 * 3 + 1] = 0.95f
        val visibilities = FloatArray(33) { 0.7f }

        val (kp, conf) = MediaPipePoseAdapter.mapBlazePoseToCoco(blazepose, visibilities)
        // COCO right_ankle is index 16 -> kp positions [32, 33], confidence index 16
        assertEquals(0.85f, kp[32], 1e-6f)
        assertEquals(0.95f, kp[33], 1e-6f)
        assertEquals(0.7f, conf[16], 1e-6f)
    }

    @Test
    fun `mapBlazePoseToCoco rejects wrong sized arrays`() {
        var threw = false
        try {
            MediaPipePoseAdapter.mapBlazePoseToCoco(FloatArray(10), FloatArray(33))
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue("expected IAE for wrong landmark size", threw)
    }

    @Test
    fun `getLatestPose returns null before any push`() {
        val adapter = MediaPipePoseAdapter()
        assertNull(adapter.getLatestPose())
    }

    @Test
    fun `pushPose then getLatestPose returns the pushed observation`() {
        val adapter = MediaPipePoseAdapter()
        val keypoints = FloatArray(34) { it * 0.01f }
        val confidences = FloatArray(17) { 0.8f }

        adapter.pushPose(
            PoseObservation(
                keypoints = keypoints,
                confidences = confidences,
                timestampMs = 1000L,
                modelVersion = "mediapipe-blazepose-lite-v0.10.33"
            )
        )

        val obs = adapter.getLatestPose()
        assertNotNull(obs)
        assertEquals(1000L, obs!!.timestampMs)
        assertEquals(34, obs.keypoints.size)
        assertEquals("mediapipe-blazepose-lite-v0.10.33", obs.modelVersion)
    }

    @Test
    fun `pushPose overwrites previous latest`() {
        val adapter = MediaPipePoseAdapter()
        adapter.pushPose(observationWith(time = 1L))
        adapter.pushPose(observationWith(time = 2L))
        adapter.pushPose(observationWith(time = 3L))

        assertEquals(3L, adapter.getLatestPose()!!.timestampMs)
    }

    @Test
    fun `subscribePose handler receives subsequent pushes`() {
        val adapter = MediaPipePoseAdapter()
        val received = mutableListOf<Long>()
        val subscription = adapter.subscribePose { obs -> received.add(obs.timestampMs) }

        adapter.pushPose(observationWith(time = 100L))
        adapter.pushPose(observationWith(time = 200L))

        assertEquals(listOf(100L, 200L), received)
        subscription.close()
    }

    @Test
    fun `closing a subscription stops further notifications`() {
        val adapter = MediaPipePoseAdapter()
        val received = mutableListOf<Long>()
        val subscription = adapter.subscribePose { obs -> received.add(obs.timestampMs) }

        adapter.pushPose(observationWith(time = 1L))
        subscription.close()
        adapter.pushPose(observationWith(time = 2L))

        assertEquals(listOf(1L), received)
    }

    @Test
    fun `multiple subscriptions all receive events`() {
        val adapter = MediaPipePoseAdapter()
        val a = mutableListOf<Long>()
        val b = mutableListOf<Long>()
        adapter.subscribePose { obs -> a.add(obs.timestampMs) }
        adapter.subscribePose { obs -> b.add(obs.timestampMs) }

        adapter.pushPose(observationWith(time = 99L))

        assertEquals(listOf(99L), a)
        assertEquals(listOf(99L), b)
    }

    private fun observationWith(time: Long): PoseObservation {
        return PoseObservation(
            keypoints = FloatArray(34) { 0f },
            confidences = FloatArray(17) { 0.5f },
            timestampMs = time,
            modelVersion = "mediapipe-blazepose-lite-v0.10.33"
        )
    }
}
