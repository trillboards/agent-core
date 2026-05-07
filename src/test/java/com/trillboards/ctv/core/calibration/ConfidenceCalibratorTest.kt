package com.trillboards.ctv.core.calibration

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [ConfidenceCalibrator].
 *
 * Verifies multi-signal agreement scoring across all signal groups:
 * face count, emotion, engagement, presence, and overall calibration.
 */
class ConfidenceCalibratorTest {

    private lateinit var calibrator: ConfidenceCalibrator

    @Before
    fun setUp() {
        calibrator = ConfidenceCalibrator()
    }

    // --- Helper ---

    private fun bundle(
        faceCount: Int = 0,
        personCount: Int = 0,
        vlmFaceCount: Int? = null,
        vlmPersonCount: Int? = null,
        shadowEvents: Int = 0,
        audioOccupancy: Int? = null,
        ferEmotion: String? = null,
        vlmActivity: String? = null,
        gazeAttention: Float = 0f,
        bodyEngagement: Float = 0f
    ) = SignalBundle(
        faceCount = faceCount,
        personCount = personCount,
        vlmFaceCount = vlmFaceCount,
        vlmPersonCount = vlmPersonCount,
        shadowEvents = shadowEvents,
        audioOccupancy = audioOccupancy,
        ferEmotion = ferEmotion,
        vlmActivity = vlmActivity,
        gazeAttention = gazeAttention,
        bodyEngagement = bodyEngagement
    )

    // ==========================================
    // Face Count Agreement
    // ==========================================

    @Test
    fun `face count - 2 signals agree within spread 1`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 3, personCount = 4)
        )
        assertEquals(
            "Spread of 1 should be strong agreement",
            ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result, 0.001f
        )
    }

    @Test
    fun `face count - 2 signals exact match`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 5, personCount = 5)
        )
        assertEquals(
            "Exact match should be strong agreement",
            ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result, 0.001f
        )
    }

    @Test
    fun `face count - 2 signals moderate spread`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 2, personCount = 5)
        )
        assertEquals(
            "Spread of 3 should be moderate agreement",
            ConfidenceCalibrator.FACE_AGREEMENT_MODERATE, result, 0.001f
        )
    }

    @Test
    fun `face count - 2 signals large spread`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 1, personCount = 8)
        )
        assertEquals(
            "Spread of 7 should be weak agreement",
            ConfidenceCalibrator.FACE_AGREEMENT_WEAK, result, 0.001f
        )
    }

    @Test
    fun `face count - 3 signals all agree`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 3, personCount = 3, vlmFaceCount = 3)
        )
        assertEquals(
            "3 signals with spread 0 should be strong",
            ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result, 0.001f
        )
    }

    @Test
    fun `face count - 3 signals one disagrees`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 3, personCount = 3, vlmFaceCount = 10)
        )
        assertEquals(
            "3 signals with spread 7 should be weak",
            ConfidenceCalibrator.FACE_AGREEMENT_WEAK, result, 0.001f
        )
    }

    @Test
    fun `face count - 4 signals moderate disagreement`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 3, personCount = 4, vlmFaceCount = 5, vlmPersonCount = 6)
        )
        assertEquals(
            "4 signals with spread 3 should be moderate",
            ConfidenceCalibrator.FACE_AGREEMENT_MODERATE, result, 0.001f
        )
    }

    @Test
    fun `face count - all zeros with 2 signals is strong agreement`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 0, personCount = 0)
        )
        assertEquals(
            "Both zero should be strong agreement on empty scene",
            ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result, 0.001f
        )
    }

    @Test
    fun `face count - all zeros with VLM signals is strong agreement`() {
        val result = calibrator.computeFaceCountAgreement(
            bundle(faceCount = 0, personCount = 0, vlmFaceCount = 0, vlmPersonCount = 0)
        )
        assertEquals(
            "All four zeros should be strong agreement on empty scene",
            ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result, 0.001f
        )
    }

    // ==========================================
    // Emotion Agreement
    // ==========================================

    @Test
    fun `emotion - same valence bucket agrees`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "happy", vlmActivity = "engaged")
        )
        assertEquals("Both positive should agree", 0.85f, result, 0.001f)
    }

    @Test
    fun `emotion - opposite valence disagrees`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "happy", vlmActivity = "bored")
        )
        assertEquals("Positive vs negative should disagree", 0.3f, result, 0.001f)
    }

    @Test
    fun `emotion - one neutral one positive is moderate`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "neutral", vlmActivity = "browsing")
        )
        assertEquals("Neutral + positive should be moderate", 0.6f, result, 0.001f)
    }

    @Test
    fun `emotion - both neutral agrees`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "neutral", vlmActivity = "calm")
        )
        assertEquals("Both neutral should agree", 0.85f, result, 0.001f)
    }

    @Test
    fun `emotion - both negative agrees`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "angry", vlmActivity = "frustrated")
        )
        assertEquals("Both negative should agree", 0.85f, result, 0.001f)
    }

    @Test
    fun `emotion - no FER returns neutral confidence`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = null, vlmActivity = "happy")
        )
        assertEquals("Missing FER should return 0.5", 0.5f, result, 0.001f)
    }

    @Test
    fun `emotion - no VLM returns neutral confidence`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "happy", vlmActivity = null)
        )
        assertEquals("Missing VLM should return 0.5", 0.5f, result, 0.001f)
    }

    @Test
    fun `emotion - both null returns neutral confidence`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = null, vlmActivity = null)
        )
        assertEquals("Both null should return 0.5", 0.5f, result, 0.001f)
    }

    @Test
    fun `emotion - unrecognized emotion returns neutral`() {
        val result = calibrator.computeEmotionAgreement(
            bundle(ferEmotion = "happy", vlmActivity = "xyzzy_unknown")
        )
        // unrecognized vlmActivity => null valence => treated as missing
        assertEquals("Unrecognized VLM emotion should return 0.5", 0.5f, result, 0.001f)
    }

    // ==========================================
    // Engagement Agreement
    // ==========================================

    @Test
    fun `engagement - both high is correlated`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0.8f, bodyEngagement = 0.7f)
        )
        assertEquals(
            "Both above threshold should be correlated",
            ConfidenceCalibrator.ENGAGEMENT_CORRELATED, result, 0.001f
        )
    }

    @Test
    fun `engagement - both low is correlated`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0.1f, bodyEngagement = 0.2f)
        )
        assertEquals(
            "Both below threshold should be correlated",
            ConfidenceCalibrator.ENGAGEMENT_CORRELATED, result, 0.001f
        )
    }

    @Test
    fun `engagement - high gaze low body is mixed`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0.8f, bodyEngagement = 0.1f)
        )
        assertEquals(
            "High gaze + low body should be mixed",
            ConfidenceCalibrator.ENGAGEMENT_MIXED, result, 0.001f
        )
    }

    @Test
    fun `engagement - low gaze high body is mixed`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0.1f, bodyEngagement = 0.8f)
        )
        assertEquals(
            "Low gaze + high body should be mixed",
            ConfidenceCalibrator.ENGAGEMENT_MIXED, result, 0.001f
        )
    }

    @Test
    fun `engagement - both zero returns single signal`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0f, bodyEngagement = 0f)
        )
        assertEquals(
            "Both zero should return single signal baseline",
            ConfidenceCalibrator.ENGAGEMENT_SINGLE, result, 0.001f
        )
    }

    @Test
    fun `engagement - only gaze active returns single`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0.7f, bodyEngagement = 0f)
        )
        assertEquals(
            "Only gaze should return single signal",
            ConfidenceCalibrator.ENGAGEMENT_SINGLE, result, 0.001f
        )
    }

    @Test
    fun `engagement - only body active returns single`() {
        val result = calibrator.computeEngagementAgreement(
            bundle(gazeAttention = 0f, bodyEngagement = 0.7f)
        )
        assertEquals(
            "Only body should return single signal",
            ConfidenceCalibrator.ENGAGEMENT_SINGLE, result, 0.001f
        )
    }

    // ==========================================
    // Presence Agreement
    // ==========================================

    @Test
    fun `presence - all 3 modalities confirm presence`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 3, personCount = 2, audioOccupancy = 5, shadowEvents = 2)
        )
        assertEquals(
            "3/3 present should be strong",
            ConfidenceCalibrator.PRESENCE_STRONG, result, 0.001f
        )
    }

    @Test
    fun `presence - vision and audio agree no shadow`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 2, personCount = 2, audioOccupancy = 3, shadowEvents = 0)
        )
        assertEquals(
            "2/3 present should be moderate",
            ConfidenceCalibrator.PRESENCE_MODERATE, result, 0.001f
        )
    }

    @Test
    fun `presence - vision and shadow agree no audio`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 2, personCount = 1, audioOccupancy = null, shadowEvents = 3)
        )
        assertEquals(
            "Vision + shadow should be moderate",
            ConfidenceCalibrator.PRESENCE_MODERATE, result, 0.001f
        )
    }

    @Test
    fun `presence - only vision detects people`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 3, personCount = 2, audioOccupancy = 0, shadowEvents = 0)
        )
        assertEquals(
            "Vision only + audio says no should be conflict",
            ConfidenceCalibrator.PRESENCE_CONFLICT, result, 0.001f
        )
    }

    @Test
    fun `presence - only audio detects occupancy`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 0, personCount = 0, audioOccupancy = 5, shadowEvents = 0)
        )
        assertEquals(
            "Audio only + no vision should be conflict",
            ConfidenceCalibrator.PRESENCE_CONFLICT, result, 0.001f
        )
    }

    @Test
    fun `presence - only shadow detects motion`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 0, personCount = 0, audioOccupancy = null, shadowEvents = 5)
        )
        assertEquals(
            "Shadow only with no audio should be single",
            ConfidenceCalibrator.PRESENCE_SINGLE, result, 0.001f
        )
    }

    @Test
    fun `presence - no signals at all`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 0, personCount = 0, audioOccupancy = null, shadowEvents = 0)
        )
        assertEquals(
            "No presence signals should return neutral",
            ConfidenceCalibrator.PRESENCE_NONE, result, 0.001f
        )
    }

    @Test
    fun `presence - all modalities say empty`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 0, personCount = 0, audioOccupancy = 0, shadowEvents = 0)
        )
        assertEquals(
            "All modalities confirming empty should return neutral",
            ConfidenceCalibrator.PRESENCE_NONE, result, 0.001f
        )
    }

    @Test
    fun `presence - VLM person count contributes to vision`() {
        val result = calibrator.computePresenceAgreement(
            bundle(faceCount = 0, personCount = 0, vlmPersonCount = 3, audioOccupancy = 5, shadowEvents = 1)
        )
        assertEquals(
            "VLM person count should count as vision present",
            ConfidenceCalibrator.PRESENCE_STRONG, result, 0.001f
        )
    }

    // ==========================================
    // Signal Count
    // ==========================================

    @Test
    fun `signal count - minimum signals`() {
        val count = calibrator.countNonNullSignals(bundle())
        assertEquals("Base faceCount + personCount should give 2", 2, count)
    }

    @Test
    fun `signal count - all signals present`() {
        val count = calibrator.countNonNullSignals(
            bundle(
                faceCount = 3,
                personCount = 2,
                vlmFaceCount = 3,
                vlmPersonCount = 2,
                shadowEvents = 1,
                audioOccupancy = 5,
                ferEmotion = "happy",
                vlmActivity = "browsing",
                gazeAttention = 0.8f,
                bodyEngagement = 0.7f
            )
        )
        assertEquals("All signals present should give 10", 10, count)
    }

    @Test
    fun `signal count - partial signals`() {
        val count = calibrator.countNonNullSignals(
            bundle(
                faceCount = 3,
                personCount = 2,
                vlmFaceCount = 3,
                ferEmotion = "happy",
                gazeAttention = 0.5f
            )
        )
        // 2 base + vlmFaceCount + ferEmotion + gazeAttention = 5
        assertEquals("Partial signals should count correctly", 5, count)
    }

    // ==========================================
    // Valence Categorization
    // ==========================================

    @Test
    fun `valence - positive emotions`() {
        assertEquals("positive", calibrator.categorizeValence("happy"))
        assertEquals("positive", calibrator.categorizeValence("surprised"))
        assertEquals("positive", calibrator.categorizeValence("engaged"))
        assertEquals("positive", calibrator.categorizeValence("browsing"))
    }

    @Test
    fun `valence - negative emotions`() {
        assertEquals("negative", calibrator.categorizeValence("angry"))
        assertEquals("negative", calibrator.categorizeValence("sad"))
        assertEquals("negative", calibrator.categorizeValence("frustrated"))
        assertEquals("negative", calibrator.categorizeValence("bored"))
    }

    @Test
    fun `valence - neutral emotions`() {
        assertEquals("neutral", calibrator.categorizeValence("neutral"))
        assertEquals("neutral", calibrator.categorizeValence("calm"))
        assertEquals("neutral", calibrator.categorizeValence("idle"))
    }

    @Test
    fun `valence - null input`() {
        assertEquals(null, calibrator.categorizeValence(null))
    }

    @Test
    fun `valence - unrecognized input`() {
        assertEquals(null, calibrator.categorizeValence("xyzzy"))
    }

    @Test
    fun `valence - case insensitive`() {
        assertEquals("positive", calibrator.categorizeValence("Happy"))
        assertEquals("negative", calibrator.categorizeValence("ANGRY"))
        assertEquals("neutral", calibrator.categorizeValence("Neutral"))
    }

    // ==========================================
    // Overall Calibration (Integration)
    // ==========================================

    @Test
    fun `calibrate - high agreement across all signals`() {
        val result = calibrator.calibrate(
            bundle(
                faceCount = 3,
                personCount = 3,
                vlmFaceCount = 3,
                vlmPersonCount = 3,
                shadowEvents = 2,
                audioOccupancy = 5,
                ferEmotion = "happy",
                vlmActivity = "engaged",
                gazeAttention = 0.8f,
                bodyEngagement = 0.7f
            )
        )
        assertTrue(
            "Full agreement should produce overall > 0.80",
            result.overall > 0.80f
        )
        assertEquals(ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result.faceCountConfidence, 0.001f)
        assertEquals(0.85f, result.emotionConfidence, 0.001f)
        assertEquals(ConfidenceCalibrator.ENGAGEMENT_CORRELATED, result.engagementConfidence, 0.001f)
        assertEquals(ConfidenceCalibrator.PRESENCE_STRONG, result.presenceConfidence, 0.001f)
        assertEquals(10, result.signalCount)
    }

    @Test
    fun `calibrate - poor agreement across signals`() {
        val result = calibrator.calibrate(
            bundle(
                faceCount = 1,
                personCount = 10,
                vlmFaceCount = 20,
                shadowEvents = 0,
                audioOccupancy = 0,
                ferEmotion = "happy",
                vlmActivity = "bored",
                gazeAttention = 0.8f,
                bodyEngagement = 0.1f
            )
        )
        assertTrue(
            "Disagreement should produce overall < 0.55",
            result.overall < 0.55f
        )
        assertEquals(ConfidenceCalibrator.FACE_AGREEMENT_WEAK, result.faceCountConfidence, 0.001f)
        assertEquals(0.3f, result.emotionConfidence, 0.001f)
        assertEquals(ConfidenceCalibrator.ENGAGEMENT_MIXED, result.engagementConfidence, 0.001f)
    }

    @Test
    fun `calibrate - minimal signals (no VLM no audio no emotion)`() {
        val result = calibrator.calibrate(
            bundle(
                faceCount = 2,
                personCount = 3
            )
        )
        // With minimal signals, most sub-scores default to 0.5
        assertTrue(
            "Minimal signals should produce moderate overall (0.4-0.7)",
            result.overall in 0.4f..0.7f
        )
        assertEquals(2, result.signalCount)
    }

    @Test
    fun `calibrate - overall is bounded 0 to 1`() {
        // Even with extreme disagreement, overall should never go below 0
        val lowResult = calibrator.calibrate(
            bundle(
                faceCount = 0,
                personCount = 100,
                vlmFaceCount = 50,
                ferEmotion = "happy",
                vlmActivity = "angry",
                gazeAttention = 0.9f,
                bodyEngagement = 0.0f
            )
        )
        assertTrue("Overall should be >= 0", lowResult.overall >= 0f)
        assertTrue("Overall should be <= 1", lowResult.overall <= 1f)
    }

    @Test
    fun `calibrate - weights sum to 1`() {
        val sum = ConfidenceCalibrator.WEIGHT_FACE_COUNT +
            ConfidenceCalibrator.WEIGHT_EMOTION +
            ConfidenceCalibrator.WEIGHT_ENGAGEMENT +
            ConfidenceCalibrator.WEIGHT_PRESENCE
        assertEquals("Weights must sum to 1.0", 1.0f, sum, 0.001f)
    }

    @Test
    fun `calibrate - empty scene full agreement`() {
        val result = calibrator.calibrate(
            bundle(
                faceCount = 0,
                personCount = 0,
                vlmFaceCount = 0,
                vlmPersonCount = 0,
                shadowEvents = 0,
                audioOccupancy = 0,
                ferEmotion = "neutral",
                vlmActivity = "idle",
                gazeAttention = 0f,
                bodyEngagement = 0f
            )
        )
        // All signals agree the scene is empty — this should be high confidence
        assertTrue(
            "Empty scene with full agreement should have overall >= 0.60",
            result.overall >= 0.60f
        )
        assertEquals(ConfidenceCalibrator.FACE_AGREEMENT_STRONG, result.faceCountConfidence, 0.001f)
    }

    // ==========================================
    // InferenceOutput effective confidence
    // ==========================================

    @Test
    fun `InferenceOutput effectiveConfidence returns semanticConfidence when set`() {
        val output = com.trillboards.ctv.core.inference.InferenceOutput(
            modelId = "test",
            fields = emptyMap(),
            latencyMs = 100,
            confidence = 0.95f,
            semanticConfidence = 0.72f
        )
        assertEquals(
            "effectiveConfidence should prefer semanticConfidence",
            0.72f, output.effectiveConfidence, 0.001f
        )
    }

    @Test
    fun `InferenceOutput effectiveConfidence falls back to confidence when semantic is null`() {
        val output = com.trillboards.ctv.core.inference.InferenceOutput(
            modelId = "test",
            fields = emptyMap(),
            latencyMs = 100,
            confidence = 0.95f,
            semanticConfidence = null
        )
        assertEquals(
            "effectiveConfidence should fall back to confidence",
            0.95f, output.effectiveConfidence, 0.001f
        )
    }

    @Test
    fun `InferenceOutput backwards compatible - default semanticConfidence is null`() {
        val output = com.trillboards.ctv.core.inference.InferenceOutput(
            modelId = "test",
            fields = emptyMap(),
            latencyMs = 100,
            confidence = 0.90f
        )
        assertEquals(null, output.semanticConfidence)
        assertEquals(0.90f, output.effectiveConfidence, 0.001f)
    }
}
