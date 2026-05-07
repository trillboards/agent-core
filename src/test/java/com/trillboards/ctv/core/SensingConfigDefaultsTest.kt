package com.trillboards.ctv.core

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Verifies that SensingConfig default values match the original hardcoded constants.
 * This ensures the Phase 1 refactor (moving magic numbers into SensingConfig)
 * introduces ZERO behavioral change.
 *
 * Each test group corresponds to a processor file that was refactored.
 */
class SensingConfigDefaultsTest {

    private val cfg = SensingConfig.Config()
    private val delta = 0.0001f  // Float comparison tolerance

    // ── DetectionConfig (AudienceAnalyzer magic numbers) ─────────────────

    @Test
    fun `detection minFaceSize matches original 0_15`() {
        assertEquals(0.15f, cfg.detection.minFaceSize, delta)
    }

    @Test
    fun `detection iouThreshold matches original 0_3`() {
        assertEquals(0.3f, cfg.detection.iouThreshold, delta)
    }

    @Test
    fun `detection centroidDistanceThreshold matches original 0_5`() {
        assertEquals(0.5f, cfg.detection.centroidDistanceThreshold, delta)
    }

    @Test
    fun `detection staleFaceRemovalMs matches original 1000`() {
        assertEquals(1000L, cfg.detection.staleFaceRemovalMs)
    }

    @Test
    fun `detection maxBufferedMetrics matches original 50`() {
        assertEquals(50, cfg.detection.maxBufferedMetrics)
    }

    @Test
    fun `detection minPoseBufferConfidence matches original 0_3`() {
        assertEquals(0.3f, cfg.detection.minPoseBufferConfidence, delta)
    }

    @Test
    fun `detection smoothingAlpha matches original 0_8`() {
        assertEquals(0.8f, cfg.detection.smoothingAlpha, delta)
    }

    @Test
    fun `detection engagement reaction thresholds match originals`() {
        assertEquals(0.7f, cfg.detection.highlyEngagedThreshold, delta)
        assertEquals(0.5f, cfg.detection.interestedThreshold, delta)
        assertEquals(0.3f, cfg.detection.neutralThreshold, delta)
    }

    // ── GazeConfig (GazeTrackingProcessor magic numbers) ─────────────────

    @Test
    fun `gaze headYawThreshold matches original 20`() {
        assertEquals(20f, cfg.gaze.headYawThreshold, delta)
    }

    @Test
    fun `gaze headPitchThreshold matches original 15`() {
        assertEquals(15f, cfg.gaze.headPitchThreshold, delta)
    }

    @Test
    fun `gaze minFaceConfidence matches original 0_5`() {
        assertEquals(0.5f, cfg.gaze.minFaceConfidence, delta)
    }

    @Test
    fun `gaze landmarkAvailableConfidence matches original 0_9`() {
        assertEquals(0.9f, cfg.gaze.landmarkAvailableConfidence, delta)
    }

    @Test
    fun `gaze landmarkFallbackConfidence matches original 0_5`() {
        assertEquals(0.5f, cfg.gaze.landmarkFallbackConfidence, delta)
    }

    @Test
    fun `gaze eyeFallbackYFactor matches original 0_4`() {
        assertEquals(0.4f, cfg.gaze.eyeFallbackYFactor, delta)
    }

    @Test
    fun `gaze yawAngleDivisor matches original 90`() {
        assertEquals(90f, cfg.gaze.yawAngleDivisor, delta)
    }

    @Test
    fun `gaze yawMaxAdjustment matches original 0_3`() {
        assertEquals(0.3f, cfg.gaze.yawMaxAdjustment, delta)
    }

    @Test
    fun `gaze pitchAngleDivisor matches original 45`() {
        assertEquals(45f, cfg.gaze.pitchAngleDivisor, delta)
    }

    @Test
    fun `gaze pitchMaxAdjustment matches original 0_2`() {
        assertEquals(0.2f, cfg.gaze.pitchMaxAdjustment, delta)
    }

    @Test
    fun `gaze offScreen confidence scaling matches originals`() {
        assertEquals(0.5f, cfg.gaze.offScreenConfidenceScale, delta)
        assertEquals(0.1f, cfg.gaze.offScreenMinConfidence, delta)
        assertEquals(0.4f, cfg.gaze.offScreenMaxConfidence, delta)
    }

    @Test
    fun `gaze defaultStability matches original 0_5`() {
        assertEquals(0.5f, cfg.gaze.defaultStability, delta)
    }

    @Test
    fun `gaze notLookingAttentionScore matches original 0_1`() {
        assertEquals(0.1f, cfg.gaze.notLookingAttentionScore, delta)
    }

    // ── BodyConfig (PoseEngagementProcessor magic numbers) ───────────────

    @Test
    fun `body facingAngleThreshold matches original 30`() {
        assertEquals(30f, cfg.body.facingAngleThreshold, delta)
    }

    @Test
    fun `body leanMagnitudeThreshold matches original 0_1`() {
        assertEquals(0.1f, cfg.body.leanMagnitudeThreshold, delta)
    }

    @Test
    fun `body movementSpeedThreshold matches original 0_05`() {
        assertEquals(0.05f, cfg.body.movementSpeedThreshold, delta)
    }

    @Test
    fun `body minShoulderWidth matches original 0_05`() {
        assertEquals(0.05f, cfg.body.minShoulderWidth, delta)
    }

    @Test
    fun `body leanNeutralScore matches original 0_15`() {
        assertEquals(0.15f, cfg.body.leanNeutralScore, delta)
    }

    @Test
    fun `body leanBackScore matches original 0_05`() {
        assertEquals(0.05f, cfg.body.leanBackScore, delta)
    }

    @Test
    fun `body leanUnknownScore matches original 0_1`() {
        assertEquals(0.1f, cfg.body.leanUnknownScore, delta)
    }

    @Test
    fun `body mediapipe confidence thresholds match original 0_5`() {
        assertEquals(0.5f, cfg.body.minPoseDetectionConfidence, delta)
        assertEquals(0.5f, cfg.body.minPosePresenceConfidence, delta)
        assertEquals(0.5f, cfg.body.minTrackingConfidence, delta)
    }

    @Test
    fun `body walkingSlowSpeedMultiplier matches original 3`() {
        assertEquals(3f, cfg.body.walkingSlowSpeedMultiplier, delta)
    }

    @Test
    fun `body missingLandmarksConfidence matches original 0_3`() {
        assertEquals(0.3f, cfg.body.missingLandmarksConfidence, delta)
    }

    @Test
    fun `body engagedStationaryHipThreshold default 0_08`() {
        // Originally added 2026-05-02 — gates upper-body micro-motion
        // (typing/sipping/gesturing) from being mis-classified as WALKING_FAST
        // when faceCount >= 1 + screenEngaged. P2.1 fix (audit 2026-05-03)
        // generalized the gate from `faceCount == 1` to `faceCount >= 1` so
        // multi-face engaged scenes (couple, coworkers) get the same treatment.
        // 0.08 sits between movementSpeedThreshold (0.05 = STOPPED upper bound) and
        // movementSpeedThreshold * walkingSlowSpeedMultiplier (0.15 = WALKING_SLOW upper
        // bound), giving a reasonable margin for upper-body sway.
        assertEquals(0.08f, cfg.body.engagedStationaryHipThreshold, delta)
    }

    // ── FootTrafficConfig (FootTrafficEstimator magic numbers) ───────────

    @Test
    fun `footTraffic normalization scales match originals`() {
        assertEquals(20f, cfg.footTraffic.faceNormScale, delta)
        assertEquals(65f, cfg.footTraffic.audioNormScale, delta)
        assertEquals(30f, cfg.footTraffic.shadowNormScale, delta)
    }

    @Test
    fun `footTraffic full signal weights match originals`() {
        assertEquals(0.50f, cfg.footTraffic.faceWeightFull, delta)
        assertEquals(0.20f, cfg.footTraffic.audioWeightFull, delta)
        assertEquals(0.30f, cfg.footTraffic.shadowWeightFull, delta)
    }

    @Test
    fun `footTraffic noShadow weights match originals`() {
        assertEquals(0.70f, cfg.footTraffic.faceWeightNoShadow, delta)
        assertEquals(0.30f, cfg.footTraffic.audioWeightNoShadow, delta)
    }

    @Test
    fun `footTraffic confidence breakpoints match originals`() {
        assertEquals(0.3f, cfg.footTraffic.lowSignalConfidence, delta)
        assertEquals(0.3f, cfg.footTraffic.twoSignalAgreementThreshold, delta)
        assertEquals(0.7f, cfg.footTraffic.twoSignalHighConfidence, delta)
        assertEquals(0.5f, cfg.footTraffic.twoSignalLowConfidence, delta)
        assertEquals(0.2f, cfg.footTraffic.threeSignalTightSpread, delta)
        assertEquals(0.9f, cfg.footTraffic.threeSignalTightConfidence, delta)
        assertEquals(0.4f, cfg.footTraffic.threeSignalMediumSpread, delta)
        assertEquals(0.7f, cfg.footTraffic.threeSignalMediumConfidence, delta)
        assertEquals(0.5f, cfg.footTraffic.threeSignalWideConfidence, delta)
    }

    // ── AudioConfig (AudioClassificationProcessor magic numbers) ─────────

    @Test
    fun `audio scoreThreshold matches original 0_2`() {
        assertEquals(0.2f, cfg.audio.scoreThreshold, delta)
    }

    @Test
    fun `audio defaultReceptivity matches original 0_5`() {
        assertEquals(0.5f, cfg.audio.defaultReceptivity, delta)
    }

    @Test
    fun `audio SNR estimates match originals`() {
        assertEquals(30f, cfg.audio.snrQuiet, delta)
        assertEquals(20f, cfg.audio.snrLow, delta)
        assertEquals(10f, cfg.audio.snrModerate, delta)
        assertEquals(5f, cfg.audio.snrElevated, delta)
        assertEquals(0f, cfg.audio.snrHigh, delta)
        assertEquals(-5f, cfg.audio.snrVeryHigh, delta)
        assertEquals(10f, cfg.audio.snrDefault, delta)
    }

    @Test
    fun `audio minAudioLevel matches current speech gate default`() {
        assertEquals(180, cfg.audio.minAudioLevel)
    }

    // ── SpeechProcessingConfig (SpeechIntelligenceProcessor magic numbers) ─

    @Test
    fun `speech onDeviceConfidenceThreshold matches original 0_70`() {
        assertEquals(0.70f, cfg.speech.onDeviceConfidenceThreshold, delta)
    }

    @Test
    fun `speech noisyGeminiConfidenceThreshold matches original 0_30`() {
        assertEquals(0.30f, cfg.speech.noisyGeminiConfidenceThreshold, delta)
    }

    @Test
    fun `speech high-pass cutoff frequencies match originals`() {
        assertEquals(300, cfg.speech.hpCutoffMusic)
        assertEquals(500, cfg.speech.hpCutoffCrowd)
        assertEquals(200, cfg.speech.hpCutoffSpeech)
        assertEquals(150, cfg.speech.hpCutoffAmbient)
    }

    @Test
    fun `speech low-pass cutoff for music matches original 3000`() {
        assertEquals(3000, cfg.speech.lpCutoffMusic)
    }

    // ── Pre-existing configs still have correct defaults ─────────────────

    @Test
    fun `engagement config defaults unchanged`() {
        assertEquals(0.35f, cfg.engagement.poseWeight, delta)
        assertEquals(0.40f, cfg.engagement.emotionWeight, delta)
        assertEquals(0.25f, cfg.engagement.gazeWeight, delta)
        assertEquals(0.7f, cfg.engagement.highEngagementThreshold, delta)
        assertEquals(0.3f, cfg.engagement.negativeReactionThreshold, delta)
    }

    @Test
    fun `pre-existing gaze config defaults unchanged`() {
        assertEquals(0.05f, cfg.gaze.gazeStabilityThreshold, delta)
        assertEquals(0.3f, cfg.gaze.confidenceTrackingWeight, delta)
        assertEquals(0.4f, cfg.gaze.confidenceAngleWeight, delta)
        assertEquals(0.3f, cfg.gaze.confidenceLandmarkWeight, delta)
        assertEquals(0.1f, cfg.gaze.minGazeConfidence, delta)
        assertEquals(0.85f, cfg.gaze.maxGazeConfidence, delta)
        assertEquals(0.15f, cfg.gaze.stabilityDivisor, delta)
        assertEquals(0.4f, cfg.gaze.centerFocusScore, delta)
        assertEquals(0.3f, cfg.gaze.adjacentFocusScore, delta)
        assertEquals(0.2f, cfg.gaze.cornerFocusScore, delta)
        assertEquals(0.3f, cfg.gaze.stabilityWeight, delta)
        assertEquals(0.3f, cfg.gaze.headDirectnessWeight, delta)
        assertEquals(45f, cfg.gaze.headDirectnessDivisor, delta)
    }

    @Test
    fun `pre-existing body config defaults unchanged`() {
        assertEquals(0.15f, cfg.body.leanBaseline, delta)
        assertEquals(0.2f, cfg.body.leanMaxMagnitude, delta)
        assertEquals(-0.03f, cfg.body.leanInMinDeviation, delta)
        assertEquals(0.03f, cfg.body.leanBackMinDeviation, delta)
        assertEquals(0.3f, cfg.body.facingWeight, delta)
        assertEquals(0.25f, cfg.body.leanWeight, delta)
        assertEquals(0.45f, cfg.body.stoppedScore, delta)
        assertEquals(0.25f, cfg.body.walkingSlowScore, delta)
        assertEquals(0.05f, cfg.body.walkingFastScore, delta)
        assertEquals(0.15f, cfg.body.unknownMovementScore, delta)
    }

    @Test
    fun `pre-existing audio config defaults unchanged`() {
        assertEquals(0.3f, cfg.audio.minClassificationConfidence, delta)
        assertEquals(0.7f, cfg.audio.audienceSizeCrowdedMusic, delta)
        assertEquals(0.9f, cfg.audio.audienceSizeCrowded, delta)
        assertEquals(0.6f, cfg.audio.audienceSizeSpeech, delta)
        assertEquals(0.3f, cfg.audio.audienceSizeQuiet, delta)
        assertEquals(0.5f, cfg.audio.audienceSizeDefault, delta)
        assertEquals(0.6f, cfg.audio.receptivityAudienceWeight, delta)
        assertEquals(0.4f, cfg.audio.receptivityAttentionWeight, delta)
        assertEquals(0.15f, cfg.audio.activeViewerBoost, delta)
    }

    @Test
    fun `pre-existing detection personConfidenceThreshold unchanged`() {
        assertEquals(0.4f, cfg.detection.personConfidenceThreshold, delta)
    }
}
