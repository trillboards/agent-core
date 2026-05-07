package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HeuristicEmotionalEngagementEstimatorTest {

    @Test
    fun `face heuristic fallback emits engagement when visual audience is present`() {
        val engagement = HeuristicEmotionalEngagementEstimator.fromFaces(
            faces = listOf(
                DetectedFace(
                    id = 1,
                    boundingBox = FaceRect(100, 100, 220, 240),
                    headEulerAngleX = -4f,
                    headEulerAngleY = 3f,
                    headEulerAngleZ = 0f,
                    smilingProbability = 0.82f,
                    leftEyeOpenProbability = 0.9f,
                    rightEyeOpenProbability = 0.88f,
                    estimatedAge = null,
                    estimatedGender = null,
                    firstSeenTimestamp = 1_000L,
                    lastSeenTimestamp = 4_500L
                )
            ),
            timestamp = 5_000L
        )

        assertNotNull(engagement)
        assertNotNull(engagement!!.pose)
        assertNotNull(engagement.emotion)
        assertNotNull(engagement.gaze)
        assertEquals(1, engagement.emotion!!.positiveReactionCount)
        assertEquals(4, engagement.gaze!!.primaryFocusRegion)
        assertTrue(engagement.overallEngagementScore > 0.5f)
        assertTrue(engagement.confidence in 0.35f..0.55f)
    }

    @Test
    fun `face heuristic fallback marks averted viewers as low engagement`() {
        val engagement = HeuristicEmotionalEngagementEstimator.fromFaces(
            faces = listOf(
                DetectedFace(
                    id = 2,
                    boundingBox = FaceRect(40, 80, 150, 220),
                    headEulerAngleX = 18f,
                    headEulerAngleY = 35f,
                    headEulerAngleZ = 0f,
                    smilingProbability = 0.05f,
                    leftEyeOpenProbability = 0.4f,
                    rightEyeOpenProbability = 0.35f,
                    estimatedAge = null,
                    estimatedGender = null,
                    firstSeenTimestamp = 1_000L,
                    lastSeenTimestamp = 2_000L
                )
            ),
            timestamp = 2_500L
        )

        assertNotNull(engagement)
        assertTrue(engagement!!.overallEngagementScore < 0.4f)
        assertEquals(AudienceReaction.DISINTERESTED, engagement.audienceReaction)
        assertTrue(engagement.pose!!.facingScreenPct < 1f)
    }
}
