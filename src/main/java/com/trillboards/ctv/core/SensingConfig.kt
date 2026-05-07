package com.trillboards.ctv.core

import android.util.Log
import androidx.annotation.VisibleForTesting
import com.trillboards.ctv.core.calibration.DeploymentConfig
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Thread-safe singleton holding server-tunable sensing thresholds and weights.
 *
 * All 70+ hardcoded constants from CTV processors are centralized here.
 * Default values match the original hardcoded values, so behavior is identical
 * until the server pushes new config via config.push ("sensing_config_update").
 *
 * Usage:
 *   val cfg = SensingConfig.get()
 *   val score = value * cfg.audio.receptivityAudienceWeight
 *
 * Server update:
 *   SensingConfig.updateFromJson(payload.optJSONObject("config"))
 */
object SensingConfig {
    private const val TAG = "SensingConfig"

    private val current = AtomicReference(Config())
    private val hasServerConfig = AtomicBoolean(false)

    /**
     * Tracks which config sections and fields have been pushed by the server.
     * Thread-safe via AtomicReference; replaced atomically on each updateFromJson call.
     *
     * Structure: section -> set of field names that were present in the server push.
     * e.g. {"gaze" -> ["headYawThreshold", "headPitchThreshold"], "detection" -> ["personConfidenceThreshold"]}
     */
    private val serverPushedFields = AtomicReference<Map<String, Set<String>>>(emptyMap())

    /**
     * Tracks which config sections and fields have been auto-calibrated (by LuxCalibrator,
     * DeploymentCalibration, etc.) rather than coming from server push.
     */
    private val autoCalibratedFields = AtomicReference<Map<String, Set<String>>>(emptyMap())

    /**
     * Returns the complete map of server-pushed field names per section.
     */
    fun getServerPushedFields(): Map<String, Set<String>> = serverPushedFields.get()

    /**
     * Returns the complete map of auto-calibrated field names per section.
     */
    fun getAutoCalibratedFields(): Map<String, Set<String>> = autoCalibratedFields.get()

    /**
     * Check if a specific field was overridden by the server.
     * @param section The config section name (e.g. "gaze", "detection")
     * @param field The field name within that section (e.g. "headYawThreshold")
     */
    fun wasServerPushed(section: String, field: String): Boolean {
        return serverPushedFields.get()[section]?.contains(field) == true
    }

    /**
     * Check if a specific field was auto-calibrated on-device.
     * @param section The config section name (e.g. "viewability")
     * @param field The field name within that section (e.g. "darkLuxThreshold")
     */
    fun wasAutoCalibrated(section: String, field: String): Boolean {
        return autoCalibratedFields.get()[section]?.contains(field) == true
    }

    /**
     * Determine the source of a specific config field value.
     * @return "server" if server-pushed, "calibrated" if auto-calibrated, "default" otherwise
     */
    fun getFieldSource(section: String, field: String): String {
        return when {
            wasServerPushed(section, field) -> "server"
            wasAutoCalibrated(section, field) -> "calibrated"
            else -> "default"
        }
    }

    /**
     * Mark fields as auto-calibrated. Called by LuxCalibrator, DeploymentCalibration, etc.
     * @param section The config section name
     * @param fields The field names that were auto-calibrated
     */
    fun markAutoCalibrated(section: String, fields: Set<String>) {
        val updated = autoCalibratedFields.get().toMutableMap()
        val existing = updated[section]?.toMutableSet() ?: mutableSetOf()
        existing.addAll(fields)
        updated[section] = existing
        autoCalibratedFields.set(updated)
    }

    fun get(): Config = current.get()
    fun hasServerConfig(): Boolean = hasServerConfig.get()

    /**
     * Reset the singleton to its default state. Intended for unit tests so each
     * test starts from a clean slate; without this, server-pushed and
     * auto-calibrated tracking from one test leaks into the next and breaks
     * the "default" / "calibrated" / "server" provenance assertions.
     *
     * NOT for production use — server config and calibration data are durable
     * across the agent lifetime by design.
     */
    @VisibleForTesting
    fun resetForTesting() {
        current.set(Config())
        hasServerConfig.set(false)
        serverPushedFields.set(emptyMap())
        autoCalibratedFields.set(emptyMap())
    }

    fun updateFromJson(json: JSONObject) {
        val existing = current.get()
        val updated = existing.copy(
            audio = existing.audio.mergeWith(json.optJSONObject("audio")),
            attention = existing.attention.mergeWith(json.optJSONObject("attention")),
            engagement = existing.engagement.mergeWith(json.optJSONObject("engagement")),
            gaze = existing.gaze.mergeWith(json.optJSONObject("gaze")),
            body = existing.body.mergeWith(json.optJSONObject("body")),
            emotion = existing.emotion.mergeWith(json.optJSONObject("emotion")),
            detection = existing.detection.mergeWith(json.optJSONObject("detection")),
            shadow = existing.shadow.mergeWith(json.optJSONObject("shadow")),
            viewability = existing.viewability.mergeWith(json.optJSONObject("viewability")),
            insight = existing.insight.mergeWith(json.optJSONObject("insight")),
            capture = existing.capture.mergeWith(json.optJSONObject("capture")),
            vlm = existing.vlm.mergeWith(json.optJSONObject("vlm")),
            circuitBreaker = existing.circuitBreaker.mergeWith(json.optJSONObject("circuitBreaker")),
            perception = existing.perception.mergeWith(json.optJSONObject("perception")),
            hardware = existing.hardware.mergeWith(json.optJSONObject("hardware")),
            memory = existing.memory.mergeWith(json.optJSONObject("memory")),
            recovery = existing.recovery.mergeWith(json.optJSONObject("recovery")),
            rendering = existing.rendering.mergeWith(json.optJSONObject("rendering")),
            identity = existing.identity.mergeWith(json.optJSONObject("identity")),
            scene = existing.scene.mergeWith(json.optJSONObject("scene")),
            footTraffic = existing.footTraffic.mergeWith(json.optJSONObject("footTraffic")),
            speech = existing.speech.mergeWith(json.optJSONObject("speech")),
            deployment = existing.deployment.mergeWith(json.optJSONObject("deployment"))
        )
        current.set(updated)
        hasServerConfig.set(true)

        // Track which specific fields the server pushed per section
        val newPushed = serverPushedFields.get().toMutableMap()
        val sections = json.keys()
        while (sections.hasNext()) {
            val sectionName = sections.next()
            val sectionObj = json.optJSONObject(sectionName) ?: continue
            val existingFields = newPushed[sectionName]?.toMutableSet() ?: mutableSetOf()
            val fieldKeys = sectionObj.keys()
            while (fieldKeys.hasNext()) {
                existingFields.add(fieldKeys.next())
            }
            newPushed[sectionName] = existingFields
        }
        serverPushedFields.set(newPushed)

        Log.i(TAG, "SensingConfig updated from server: ${json.keys().asSequence().toList()}")
    }

    // ── Top-level config ────────────────────────────────────────────────────

    data class Config(
        val audio: AudioConfig = AudioConfig(),
        val attention: AttentionConfig = AttentionConfig(),
        val engagement: EngagementConfig = EngagementConfig(),
        val gaze: GazeConfig = GazeConfig(),
        val body: BodyConfig = BodyConfig(),
        val emotion: EmotionConfig = EmotionConfig(),
        val detection: DetectionConfig = DetectionConfig(),
        val shadow: ShadowConfig = ShadowConfig(),
        val viewability: ViewabilityConfig = ViewabilityConfig(),
        val insight: InsightConfig = InsightConfig(),
        val capture: CaptureConfig = CaptureConfig(),
        val vlm: VlmConfig = VlmConfig(),
        val circuitBreaker: CircuitBreakerConfig = CircuitBreakerConfig(),
        val perception: PerceptionConfig = PerceptionConfig(),
        val hardware: HardwareConfig = HardwareConfig(),
        val memory: MemoryConfig = MemoryConfig(),
        val recovery: RecoveryConfig = RecoveryConfig(),
        val rendering: RenderingConfig = RenderingConfig(),
        val identity: IdentityConfig = IdentityConfig(),
        val scene: SceneConfig = SceneConfig(),
        val footTraffic: FootTrafficConfig = FootTrafficConfig(),
        val speech: SpeechProcessingConfig = SpeechProcessingConfig(),
        val deployment: DeploymentConfig = DeploymentConfig()
    )

    // ── AudioClassificationProcessor ────────────────────────────────────────

    data class AudioConfig(
        val minClassificationConfidence: Float = 0.3f,
        // Ad receptivity: audience size factors
        val audienceSizeCrowdedMusic: Float = 0.7f,
        val audienceSizeCrowded: Float = 0.9f,
        val audienceSizeSpeech: Float = 0.6f,
        val audienceSizeQuiet: Float = 0.3f,
        val audienceSizeDefault: Float = 0.5f,
        // Ad receptivity: attention factors
        val attentionQuiet: Float = 0.9f,
        val attentionSpeechOnly: Float = 0.7f,
        val attentionMusicOnly: Float = 0.6f,
        val attentionCrowdOnly: Float = 0.5f,
        val attentionCrowdMusic: Float = 0.3f,
        val attentionDefault: Float = 0.5f,
        // Ad receptivity: composite weights
        val receptivityAudienceWeight: Float = 0.6f,
        val receptivityAttentionWeight: Float = 0.4f,
        // NEW: viewer presence boost (breaks 2-factor compression)
        val activeViewerBoost: Float = 0.15f,
        // AudioConfig scoreThreshold (minimum confidence for YAMNet classification)
        val scoreThreshold: Float = 0.2f,
        // Default ad receptivity score when no classification is available
        val defaultReceptivity: Float = 0.5f,
        // SNR estimates per noise tier (0-5 ordinal)
        val snrQuiet: Float = 30f,
        val snrLow: Float = 20f,
        val snrModerate: Float = 10f,
        val snrElevated: Float = 5f,
        val snrHigh: Float = 0f,
        val snrVeryHigh: Float = -5f,
        val snrDefault: Float = 10f,
        // Minimum RMS audio level to trigger speech processing.
        // The original 500 gate was too aggressive for field tablet mics and
        // suppressed real conversational speech before ASR even ran.
        val minAudioLevel: Int = 180
    ) {
        fun mergeWith(json: JSONObject?): AudioConfig {
            if (json == null) return this
            return copy(
                minClassificationConfidence = json.optDouble("minClassificationConfidence", minClassificationConfidence.toDouble()).toFloat(),
                audienceSizeCrowdedMusic = json.optDouble("audienceSizeCrowdedMusic", audienceSizeCrowdedMusic.toDouble()).toFloat(),
                audienceSizeCrowded = json.optDouble("audienceSizeCrowded", audienceSizeCrowded.toDouble()).toFloat(),
                audienceSizeSpeech = json.optDouble("audienceSizeSpeech", audienceSizeSpeech.toDouble()).toFloat(),
                audienceSizeQuiet = json.optDouble("audienceSizeQuiet", audienceSizeQuiet.toDouble()).toFloat(),
                audienceSizeDefault = json.optDouble("audienceSizeDefault", audienceSizeDefault.toDouble()).toFloat(),
                attentionQuiet = json.optDouble("attentionQuiet", attentionQuiet.toDouble()).toFloat(),
                attentionSpeechOnly = json.optDouble("attentionSpeechOnly", attentionSpeechOnly.toDouble()).toFloat(),
                attentionMusicOnly = json.optDouble("attentionMusicOnly", attentionMusicOnly.toDouble()).toFloat(),
                attentionCrowdOnly = json.optDouble("attentionCrowdOnly", attentionCrowdOnly.toDouble()).toFloat(),
                attentionCrowdMusic = json.optDouble("attentionCrowdMusic", attentionCrowdMusic.toDouble()).toFloat(),
                attentionDefault = json.optDouble("attentionDefault", attentionDefault.toDouble()).toFloat(),
                receptivityAudienceWeight = json.optDouble("receptivityAudienceWeight", receptivityAudienceWeight.toDouble()).toFloat(),
                receptivityAttentionWeight = json.optDouble("receptivityAttentionWeight", receptivityAttentionWeight.toDouble()).toFloat(),
                activeViewerBoost = json.optDouble("activeViewerBoost", activeViewerBoost.toDouble()).toFloat(),
                scoreThreshold = json.optDouble("scoreThreshold", scoreThreshold.toDouble()).toFloat(),
                defaultReceptivity = json.optDouble("defaultReceptivity", defaultReceptivity.toDouble()).toFloat(),
                snrQuiet = json.optDouble("snrQuiet", snrQuiet.toDouble()).toFloat(),
                snrLow = json.optDouble("snrLow", snrLow.toDouble()).toFloat(),
                snrModerate = json.optDouble("snrModerate", snrModerate.toDouble()).toFloat(),
                snrElevated = json.optDouble("snrElevated", snrElevated.toDouble()).toFloat(),
                snrHigh = json.optDouble("snrHigh", snrHigh.toDouble()).toFloat(),
                snrVeryHigh = json.optDouble("snrVeryHigh", snrVeryHigh.toDouble()).toFloat(),
                snrDefault = json.optDouble("snrDefault", snrDefault.toDouble()).toFloat(),
                minAudioLevel = json.optInt("minAudioLevel", minAudioLevel)
            )
        }
    }

    // ── AudienceMetrics (attention scoring) ─────────────────────────────────

    data class AttentionConfig(
        val yawDivisor: Float = 45f,
        val pitchDivisor: Float = 30f,
        val yawWeight: Float = 0.4f,
        val pitchWeight: Float = 0.3f,
        val eyeWeight: Float = 0.3f
    ) {
        fun mergeWith(json: JSONObject?): AttentionConfig {
            if (json == null) return this
            return copy(
                yawDivisor = json.optDouble("yawDivisor", yawDivisor.toDouble()).toFloat(),
                pitchDivisor = json.optDouble("pitchDivisor", pitchDivisor.toDouble()).toFloat(),
                yawWeight = json.optDouble("yawWeight", yawWeight.toDouble()).toFloat(),
                pitchWeight = json.optDouble("pitchWeight", pitchWeight.toDouble()).toFloat(),
                eyeWeight = json.optDouble("eyeWeight", eyeWeight.toDouble()).toFloat()
            )
        }
    }

    // ── EmotionalEngagementMetrics ──────────────────────────────────────────

    data class EngagementConfig(
        val poseWeight: Float = 0.35f,
        val emotionWeight: Float = 0.40f,
        val gazeWeight: Float = 0.25f,
        val highEngagementThreshold: Float = 0.7f,
        val negativeReactionThreshold: Float = 0.3f
    ) {
        fun mergeWith(json: JSONObject?): EngagementConfig {
            if (json == null) return this
            return copy(
                poseWeight = json.optDouble("poseWeight", poseWeight.toDouble()).toFloat(),
                emotionWeight = json.optDouble("emotionWeight", emotionWeight.toDouble()).toFloat(),
                gazeWeight = json.optDouble("gazeWeight", gazeWeight.toDouble()).toFloat(),
                highEngagementThreshold = json.optDouble("highEngagementThreshold", highEngagementThreshold.toDouble()).toFloat(),
                negativeReactionThreshold = json.optDouble("negativeReactionThreshold", negativeReactionThreshold.toDouble()).toFloat()
            )
        }
    }

    // ── GazeTrackingProcessor ───────────────────────────────────────────────

    data class GazeConfig(
        val headYawThreshold: Float = 20f,
        val headPitchThreshold: Float = 15f,
        val gazeStabilityThreshold: Float = 0.05f,
        val minFaceConfidence: Float = 0.5f,
        val confidenceTrackingWeight: Float = 0.3f,
        val confidenceAngleWeight: Float = 0.4f,
        val confidenceLandmarkWeight: Float = 0.3f,
        val minGazeConfidence: Float = 0.1f,
        val maxGazeConfidence: Float = 0.85f,
        val stabilityDivisor: Float = 0.15f,
        val centerFocusScore: Float = 0.4f,
        val adjacentFocusScore: Float = 0.3f,
        val cornerFocusScore: Float = 0.2f,
        val stabilityWeight: Float = 0.3f,
        val headDirectnessWeight: Float = 0.3f,
        val headDirectnessDivisor: Float = 45f,
        // Gaze point estimation: landmark confidence when both eyes detected
        val landmarkAvailableConfidence: Float = 0.9f,
        // Gaze point estimation: landmark confidence when eyes missing
        val landmarkFallbackConfidence: Float = 0.5f,
        // Gaze point estimation: eye position Y-offset when no eye landmarks
        val eyeFallbackYFactor: Float = 0.4f,
        // Gaze point estimation: yaw angle divisor and max adjustment
        val yawAngleDivisor: Float = 90f,
        val yawMaxAdjustment: Float = 0.3f,
        // Gaze point estimation: pitch angle divisor and max adjustment
        val pitchAngleDivisor: Float = 45f,
        val pitchMaxAdjustment: Float = 0.2f,
        // Not-looking-at-screen confidence scaling
        val offScreenConfidenceScale: Float = 0.5f,
        val offScreenMinConfidence: Float = 0.1f,
        val offScreenMaxConfidence: Float = 0.4f,
        // Default gaze stability when no previous data
        val defaultStability: Float = 0.5f,
        // Not-looking minimum attention score
        val notLookingAttentionScore: Float = 0.1f
    ) {
        fun mergeWith(json: JSONObject?): GazeConfig {
            if (json == null) return this
            return copy(
                headYawThreshold = json.optDouble("headYawThreshold", headYawThreshold.toDouble()).toFloat(),
                headPitchThreshold = json.optDouble("headPitchThreshold", headPitchThreshold.toDouble()).toFloat(),
                gazeStabilityThreshold = json.optDouble("gazeStabilityThreshold", gazeStabilityThreshold.toDouble()).toFloat(),
                minFaceConfidence = json.optDouble("minFaceConfidence", minFaceConfidence.toDouble()).toFloat(),
                confidenceTrackingWeight = json.optDouble("confidenceTrackingWeight", confidenceTrackingWeight.toDouble()).toFloat(),
                confidenceAngleWeight = json.optDouble("confidenceAngleWeight", confidenceAngleWeight.toDouble()).toFloat(),
                confidenceLandmarkWeight = json.optDouble("confidenceLandmarkWeight", confidenceLandmarkWeight.toDouble()).toFloat(),
                minGazeConfidence = json.optDouble("minGazeConfidence", minGazeConfidence.toDouble()).toFloat(),
                maxGazeConfidence = json.optDouble("maxGazeConfidence", maxGazeConfidence.toDouble()).toFloat(),
                stabilityDivisor = json.optDouble("stabilityDivisor", stabilityDivisor.toDouble()).toFloat(),
                centerFocusScore = json.optDouble("centerFocusScore", centerFocusScore.toDouble()).toFloat(),
                adjacentFocusScore = json.optDouble("adjacentFocusScore", adjacentFocusScore.toDouble()).toFloat(),
                cornerFocusScore = json.optDouble("cornerFocusScore", cornerFocusScore.toDouble()).toFloat(),
                stabilityWeight = json.optDouble("stabilityWeight", stabilityWeight.toDouble()).toFloat(),
                headDirectnessWeight = json.optDouble("headDirectnessWeight", headDirectnessWeight.toDouble()).toFloat(),
                headDirectnessDivisor = json.optDouble("headDirectnessDivisor", headDirectnessDivisor.toDouble()).toFloat(),
                landmarkAvailableConfidence = json.optDouble("landmarkAvailableConfidence", landmarkAvailableConfidence.toDouble()).toFloat(),
                landmarkFallbackConfidence = json.optDouble("landmarkFallbackConfidence", landmarkFallbackConfidence.toDouble()).toFloat(),
                eyeFallbackYFactor = json.optDouble("eyeFallbackYFactor", eyeFallbackYFactor.toDouble()).toFloat(),
                yawAngleDivisor = json.optDouble("yawAngleDivisor", yawAngleDivisor.toDouble()).toFloat(),
                yawMaxAdjustment = json.optDouble("yawMaxAdjustment", yawMaxAdjustment.toDouble()).toFloat(),
                pitchAngleDivisor = json.optDouble("pitchAngleDivisor", pitchAngleDivisor.toDouble()).toFloat(),
                pitchMaxAdjustment = json.optDouble("pitchMaxAdjustment", pitchMaxAdjustment.toDouble()).toFloat(),
                offScreenConfidenceScale = json.optDouble("offScreenConfidenceScale", offScreenConfidenceScale.toDouble()).toFloat(),
                offScreenMinConfidence = json.optDouble("offScreenMinConfidence", offScreenMinConfidence.toDouble()).toFloat(),
                offScreenMaxConfidence = json.optDouble("offScreenMaxConfidence", offScreenMaxConfidence.toDouble()).toFloat(),
                defaultStability = json.optDouble("defaultStability", defaultStability.toDouble()).toFloat(),
                notLookingAttentionScore = json.optDouble("notLookingAttentionScore", notLookingAttentionScore.toDouble()).toFloat()
            )
        }
    }

    // ── PoseEngagementProcessor ─────────────────────────────────────────────

    data class BodyConfig(
        val facingAngleThreshold: Float = 30f,
        val leanMagnitudeThreshold: Float = 0.1f,
        val movementSpeedThreshold: Float = 0.05f,
        val leanBaseline: Float = 0.15f,
        val leanMaxMagnitude: Float = 0.2f,
        val leanInMinDeviation: Float = -0.03f,
        val leanBackMinDeviation: Float = 0.03f,
        val facingWeight: Float = 0.3f,
        val leanWeight: Float = 0.25f,
        val stoppedScore: Float = 0.45f,
        val walkingSlowScore: Float = 0.25f,
        val walkingFastScore: Float = 0.05f,
        val unknownMovementScore: Float = 0.15f,
        // Shoulder width threshold below which body is considered turned ~90 degrees
        val minShoulderWidth: Float = 0.05f,
        // Lean direction engagement scores
        val leanNeutralScore: Float = 0.15f,
        val leanBackScore: Float = 0.05f,
        val leanUnknownScore: Float = 0.1f,
        // MediaPipe pose detection confidence thresholds
        val minPoseDetectionConfidence: Float = 0.5f,
        val minPosePresenceConfidence: Float = 0.5f,
        val minTrackingConfidence: Float = 0.5f,
        // Walking slow speed multiplier (speed < threshold * this = walking slow)
        val walkingSlowSpeedMultiplier: Float = 3f,
        // Missing key landmarks confidence fallback
        val missingLandmarksConfidence: Float = 0.3f,
        // Engagement-gated stationary override: when faceCount>=1 AND screenEngaged,
        // hip Euclidean displacement (normalized 0..1 coords) below this threshold
        // is forced to STOPPED regardless of computed speed. Prevents typing/sipping/
        // gesturing micro-motion from being classified as WALKING_FAST when the user
        // (or any of multiple users) is clearly stationary at the screen. Default
        // 0.08 sits between the "stopped" speed threshold (0.05) and the walking-
        // slow upper bound (0.15 = 0.05 * walkingSlowSpeedMultiplier), giving a
        // reasonable margin for upper-body sway while still letting truly-walking-
        // toward-screen (hip displacement >> 0.08) through to WALKING_SLOW/FAST.
        // P2.1 fix (audit 2026-05-03): the gate fires for any faceCount >= 1 so
        // multi-face engaged scenes (couple, two coworkers) are no longer
        // mis-classified as WALKING_FAST. screenEngaged uses max-face engagement
        // upstream so the highest-engagement face dominates.
        val engagedStationaryHipThreshold: Float = 0.08f
    ) {
        fun mergeWith(json: JSONObject?): BodyConfig {
            if (json == null) return this
            return copy(
                facingAngleThreshold = json.optDouble("facingAngleThreshold", facingAngleThreshold.toDouble()).toFloat(),
                leanMagnitudeThreshold = json.optDouble("leanMagnitudeThreshold", leanMagnitudeThreshold.toDouble()).toFloat(),
                movementSpeedThreshold = json.optDouble("movementSpeedThreshold", movementSpeedThreshold.toDouble()).toFloat(),
                leanBaseline = json.optDouble("leanBaseline", leanBaseline.toDouble()).toFloat(),
                leanMaxMagnitude = json.optDouble("leanMaxMagnitude", leanMaxMagnitude.toDouble()).toFloat(),
                leanInMinDeviation = json.optDouble("leanInMinDeviation", leanInMinDeviation.toDouble()).toFloat(),
                leanBackMinDeviation = json.optDouble("leanBackMinDeviation", leanBackMinDeviation.toDouble()).toFloat(),
                facingWeight = json.optDouble("facingWeight", facingWeight.toDouble()).toFloat(),
                leanWeight = json.optDouble("leanWeight", leanWeight.toDouble()).toFloat(),
                stoppedScore = json.optDouble("stoppedScore", stoppedScore.toDouble()).toFloat(),
                walkingSlowScore = json.optDouble("walkingSlowScore", walkingSlowScore.toDouble()).toFloat(),
                walkingFastScore = json.optDouble("walkingFastScore", walkingFastScore.toDouble()).toFloat(),
                unknownMovementScore = json.optDouble("unknownMovementScore", unknownMovementScore.toDouble()).toFloat(),
                minShoulderWidth = json.optDouble("minShoulderWidth", minShoulderWidth.toDouble()).toFloat(),
                leanNeutralScore = json.optDouble("leanNeutralScore", leanNeutralScore.toDouble()).toFloat(),
                leanBackScore = json.optDouble("leanBackScore", leanBackScore.toDouble()).toFloat(),
                leanUnknownScore = json.optDouble("leanUnknownScore", leanUnknownScore.toDouble()).toFloat(),
                minPoseDetectionConfidence = json.optDouble("minPoseDetectionConfidence", minPoseDetectionConfidence.toDouble()).toFloat(),
                minPosePresenceConfidence = json.optDouble("minPosePresenceConfidence", minPosePresenceConfidence.toDouble()).toFloat(),
                minTrackingConfidence = json.optDouble("minTrackingConfidence", minTrackingConfidence.toDouble()).toFloat(),
                walkingSlowSpeedMultiplier = json.optDouble("walkingSlowSpeedMultiplier", walkingSlowSpeedMultiplier.toDouble()).toFloat(),
                missingLandmarksConfidence = json.optDouble("missingLandmarksConfidence", missingLandmarksConfidence.toDouble()).toFloat(),
                engagedStationaryHipThreshold = json.optDouble("engagedStationaryHipThreshold", engagedStationaryHipThreshold.toDouble()).toFloat()
            )
        }
    }

    // ── EmotionClassificationProcessor ──────────────────────────────────────

    data class EmotionConfig(
        val minConfidence: Float = 0.3f,
        val positiveReactionThreshold: Float = 0.5f,
        val negativeReactionThreshold: Float = 0.5f,
        val happySmilingThreshold: Float = 0.7f,
        val surprisedSmilingThreshold: Float = 0.5f,
        val surprisedEyeOpenThreshold: Float = 0.7f,
        val neutralEngagementScore: Float = 0.3f,
        val neutralEngagementFactor: Float = 0.7f,
        val heuristicConfidenceFactor: Float = 0.6f
    ) {
        fun mergeWith(json: JSONObject?): EmotionConfig {
            if (json == null) return this
            return copy(
                minConfidence = json.optDouble("minConfidence", minConfidence.toDouble()).toFloat(),
                positiveReactionThreshold = json.optDouble("positiveReactionThreshold", positiveReactionThreshold.toDouble()).toFloat(),
                negativeReactionThreshold = json.optDouble("negativeReactionThreshold", negativeReactionThreshold.toDouble()).toFloat(),
                happySmilingThreshold = json.optDouble("happySmilingThreshold", happySmilingThreshold.toDouble()).toFloat(),
                surprisedSmilingThreshold = json.optDouble("surprisedSmilingThreshold", surprisedSmilingThreshold.toDouble()).toFloat(),
                surprisedEyeOpenThreshold = json.optDouble("surprisedEyeOpenThreshold", surprisedEyeOpenThreshold.toDouble()).toFloat(),
                neutralEngagementScore = json.optDouble("neutralEngagementScore", neutralEngagementScore.toDouble()).toFloat(),
                neutralEngagementFactor = json.optDouble("neutralEngagementFactor", neutralEngagementFactor.toDouble()).toFloat(),
                heuristicConfidenceFactor = json.optDouble("heuristicConfidenceFactor", heuristicConfidenceFactor.toDouble()).toFloat()
            )
        }
    }

    // ── PersonDetectionProcessor ────────────────────────────────────────────

    data class DetectionConfig(
        val personConfidenceThreshold: Float = 0.4f,
        // AudienceAnalyzer: ML Kit face detector min face size relative to image
        val minFaceSize: Float = 0.15f,
        // AudienceAnalyzer: IoU threshold for face deduplication
        val iouThreshold: Float = 0.3f,
        // AudienceAnalyzer: centroid distance threshold (fraction of avg face size)
        val centroidDistanceThreshold: Float = 0.5f,
        // AudienceAnalyzer: time before removing a face that hasn't been seen (ms)
        val staleFaceRemovalMs: Long = 1000L,
        // AudienceAnalyzer: max buffered metrics per processor before eviction
        val maxBufferedMetrics: Int = 50,
        // AudienceAnalyzer: minimum pose confidence to buffer metrics
        val minPoseBufferConfidence: Float = 0.3f,
        // AudienceAnalyzer: exponential smoothing alpha for rolling inference time
        val smoothingAlpha: Float = 0.8f,
        // AudienceAnalyzer: engagement reaction thresholds
        val highlyEngagedThreshold: Float = 0.7f,
        val interestedThreshold: Float = 0.5f,
        val neutralThreshold: Float = 0.3f
    ) {
        fun mergeWith(json: JSONObject?): DetectionConfig {
            if (json == null) return this
            return copy(
                personConfidenceThreshold = json.optDouble("personConfidenceThreshold", personConfidenceThreshold.toDouble()).toFloat(),
                minFaceSize = json.optDouble("minFaceSize", minFaceSize.toDouble()).toFloat(),
                iouThreshold = json.optDouble("iouThreshold", iouThreshold.toDouble()).toFloat(),
                centroidDistanceThreshold = json.optDouble("centroidDistanceThreshold", centroidDistanceThreshold.toDouble()).toFloat(),
                staleFaceRemovalMs = json.optLong("staleFaceRemovalMs", staleFaceRemovalMs),
                maxBufferedMetrics = json.optInt("maxBufferedMetrics", maxBufferedMetrics),
                minPoseBufferConfidence = json.optDouble("minPoseBufferConfidence", minPoseBufferConfidence.toDouble()).toFloat(),
                smoothingAlpha = json.optDouble("smoothingAlpha", smoothingAlpha.toDouble()).toFloat(),
                highlyEngagedThreshold = json.optDouble("highlyEngagedThreshold", highlyEngagedThreshold.toDouble()).toFloat(),
                interestedThreshold = json.optDouble("interestedThreshold", interestedThreshold.toDouble()).toFloat(),
                neutralThreshold = json.optDouble("neutralThreshold", neutralThreshold.toDouble()).toFloat()
            )
        }
    }

    // ── SensorCollector (shadow detection) ──────────────────────────────────

    data class ShadowConfig(
        val luxDropRatio: Float = 0.3f,
        val minLuxDrop: Float = 5f,
        val cooldownMs: Long = 500L,
        val baselineUpdateAlpha: Float = 0.01f,
        val baselineMinSamples: Int = 10,
        val baselineNonShadowRatio: Float = 0.7f,
        val tamperingThreshold: Float = 2.0f
    ) {
        fun mergeWith(json: JSONObject?): ShadowConfig {
            if (json == null) return this
            return copy(
                luxDropRatio = json.optDouble("luxDropRatio", luxDropRatio.toDouble()).toFloat(),
                minLuxDrop = json.optDouble("minLuxDrop", minLuxDrop.toDouble()).toFloat(),
                cooldownMs = json.optLong("cooldownMs", cooldownMs),
                baselineUpdateAlpha = json.optDouble("baselineUpdateAlpha", baselineUpdateAlpha.toDouble()).toFloat(),
                baselineMinSamples = json.optInt("baselineMinSamples", baselineMinSamples),
                baselineNonShadowRatio = json.optDouble("baselineNonShadowRatio", baselineNonShadowRatio.toDouble()).toFloat(),
                tamperingThreshold = json.optDouble("tamperingThreshold", tamperingThreshold.toDouble()).toFloat()
            )
        }
    }

    // ── SensorCollector (viewability / lux thresholds) ──────────────────────

    data class ViewabilityConfig(
        val minLux: Float = 5f,
        val darkLuxThreshold: Float = 5f,
        val darkViewability: Float = 0.3f,
        val dimLuxThreshold: Float = 50f,
        val dimViewability: Float = 0.7f,
        val goodLuxThreshold: Float = 500f,
        val goodViewability: Float = 1.0f,
        val brightLuxThreshold: Float = 10000f,
        val brightViewability: Float = 0.8f,
        val overbrightViewability: Float = 0.5f
    ) {
        fun mergeWith(json: JSONObject?): ViewabilityConfig {
            if (json == null) return this
            return copy(
                minLux = json.optDouble("minLux", minLux.toDouble()).toFloat(),
                darkLuxThreshold = json.optDouble("darkLuxThreshold", darkLuxThreshold.toDouble()).toFloat(),
                darkViewability = json.optDouble("darkViewability", darkViewability.toDouble()).toFloat(),
                dimLuxThreshold = json.optDouble("dimLuxThreshold", dimLuxThreshold.toDouble()).toFloat(),
                dimViewability = json.optDouble("dimViewability", dimViewability.toDouble()).toFloat(),
                goodLuxThreshold = json.optDouble("goodLuxThreshold", goodLuxThreshold.toDouble()).toFloat(),
                goodViewability = json.optDouble("goodViewability", goodViewability.toDouble()).toFloat(),
                brightLuxThreshold = json.optDouble("brightLuxThreshold", brightLuxThreshold.toDouble()).toFloat(),
                brightViewability = json.optDouble("brightViewability", brightViewability.toDouble()).toFloat(),
                overbrightViewability = json.optDouble("overbrightViewability", overbrightViewability.toDouble()).toFloat()
            )
        }
    }

    // ── InsightExtractor ────────────────────────────────────────────────────

    data class InsightConfig(
        val baselineConfidence: Float = 0.2f,
        val brandBoost: Float = 0.1f,
        val productBoost: Float = 0.1f,
        val interestBoost: Float = 0.1f,
        val readyToBuyBoost: Float = 0.35f,
        val comparingBoost: Float = 0.25f,
        val consideringBoost: Float = 0.2f,
        val browsingBoost: Float = 0.1f,
        val maxConfidence: Float = 0.95f,
        val sentimentImprovementThreshold: Float = 0.3f,
        val sentimentDeclineThreshold: Float = 0.3f,
        val skepticalConvincedThreshold: Float = 0.3f
    ) {
        fun mergeWith(json: JSONObject?): InsightConfig {
            if (json == null) return this
            return copy(
                baselineConfidence = json.optDouble("baselineConfidence", baselineConfidence.toDouble()).toFloat(),
                brandBoost = json.optDouble("brandBoost", brandBoost.toDouble()).toFloat(),
                productBoost = json.optDouble("productBoost", productBoost.toDouble()).toFloat(),
                interestBoost = json.optDouble("interestBoost", interestBoost.toDouble()).toFloat(),
                readyToBuyBoost = json.optDouble("readyToBuyBoost", readyToBuyBoost.toDouble()).toFloat(),
                comparingBoost = json.optDouble("comparingBoost", comparingBoost.toDouble()).toFloat(),
                consideringBoost = json.optDouble("consideringBoost", consideringBoost.toDouble()).toFloat(),
                browsingBoost = json.optDouble("browsingBoost", browsingBoost.toDouble()).toFloat(),
                maxConfidence = json.optDouble("maxConfidence", maxConfidence.toDouble()).toFloat(),
                sentimentImprovementThreshold = json.optDouble("sentimentImprovementThreshold", sentimentImprovementThreshold.toDouble()).toFloat(),
                sentimentDeclineThreshold = json.optDouble("sentimentDeclineThreshold", sentimentDeclineThreshold.toDouble()).toFloat(),
                skepticalConvincedThreshold = json.optDouble("skepticalConvincedThreshold", skepticalConvincedThreshold.toDouble()).toFloat()
            )
        }
    }

    // ── FrameCaptureManager ─────────────────────────────────────────────────

    data class CaptureConfig(
        val periodicIntervalMs: Long = 5 * 60 * 1000L,   // 5 minutes
        val faceTriggeredIntervalMs: Long = 30 * 1000L,   // 30 seconds
        val jpegQuality: Int = 70,
        // FrameCaptureManager — rate limiting & warmup
        val minCaptureIntervalMs: Long = 30 * 1000L,     // 30 seconds between captures
        val cameraWarmupMs: Long = 10_000L,               // 10 seconds before first capture
        val diagnosticCaptureIntervalMs: Long = 90_000L,  // 90 seconds between diagnostic captures
        // FrameCaptureManager — image dimensions
        val maxImageWidth: Int = 640,
        val maxImageHeight: Int = 480,
        // FrameCaptureManager — network timeouts
        val connectTimeoutMs: Int = 10_000,
        val readTimeoutMs: Int = 30_000,
        // FrameThrottler — FPS bounds
        val minFps: Int = 1,
        val maxFps: Int = 15,
        val initialTargetFps: Int = 6,
        val cameraFps: Int = 30
    ) {
        fun mergeWith(json: JSONObject?): CaptureConfig {
            if (json == null) return this
            return copy(
                periodicIntervalMs = json.optLong("periodicIntervalMs", periodicIntervalMs),
                faceTriggeredIntervalMs = json.optLong("faceTriggeredIntervalMs", faceTriggeredIntervalMs),
                jpegQuality = json.optInt("jpegQuality", jpegQuality),
                minCaptureIntervalMs = json.optLong("minCaptureIntervalMs", minCaptureIntervalMs),
                cameraWarmupMs = json.optLong("cameraWarmupMs", cameraWarmupMs),
                diagnosticCaptureIntervalMs = json.optLong("diagnosticCaptureIntervalMs", diagnosticCaptureIntervalMs),
                maxImageWidth = json.optInt("maxImageWidth", maxImageWidth),
                maxImageHeight = json.optInt("maxImageHeight", maxImageHeight),
                connectTimeoutMs = json.optInt("connectTimeoutMs", connectTimeoutMs),
                readTimeoutMs = json.optInt("readTimeoutMs", readTimeoutMs),
                minFps = json.optInt("minFps", minFps),
                maxFps = json.optInt("maxFps", maxFps),
                initialTargetFps = json.optInt("initialTargetFps", initialTargetFps),
                cameraFps = json.optInt("cameraFps", cameraFps)
            )
        }
    }

    // ── Rendering (ContentScheduler, NativeAdPlayer, NativeImageCarousel, ImpressionTracker, ContentApiClient) ──

    data class RenderingConfig(
        // ContentScheduler
        val contentPollIntervalMs: Long = 30_000L,
        val defaultAdIntervalMs: Long = 60_000L,
        val waterfallSourceTimeoutMs: Long = 5_000L,
        val adIntervalMinMs: Long = 15_000L,
        val contentRetryDelayMs: Long = 5_000L,
        val waterfallCompletionTimeoutMs: Long = 30_000L,
        val playbackPollIntervalMs: Long = 500L,
        // NativeAdPlayer
        val cacheSizeBytes: Long = 100L * 1024 * 1024,     // 100 MB LRU
        val stuckBufferingTimeoutMs: Long = 120_000L,
        val targetBufferBytes: Int = 512 * 1024,            // 512 KB
        // NativeImageCarousel
        val defaultImageDurationMs: Long = 8000L,
        val crossfadeDurationMs: Int = 300,
        // ImpressionTracker
        val maxPendingImpressions: Int = 1000,
        val impressionBatchSize: Int = 20,
        val maxDedupIds: Int = 5000,
        val impressionPersistInterval: Int = 10,
        val impressionConnectTimeoutS: Long = 10L,
        val impressionReadTimeoutS: Long = 15L,
        // ContentApiClient
        val contentConnectTimeoutS: Long = 10L,
        val contentReadTimeoutS: Long = 15L
    ) {
        fun mergeWith(json: JSONObject?): RenderingConfig {
            if (json == null) return this
            return copy(
                contentPollIntervalMs = json.optLong("contentPollIntervalMs", contentPollIntervalMs),
                defaultAdIntervalMs = json.optLong("defaultAdIntervalMs", defaultAdIntervalMs),
                waterfallSourceTimeoutMs = json.optLong("waterfallSourceTimeoutMs", waterfallSourceTimeoutMs),
                adIntervalMinMs = json.optLong("adIntervalMinMs", adIntervalMinMs),
                contentRetryDelayMs = json.optLong("contentRetryDelayMs", contentRetryDelayMs),
                waterfallCompletionTimeoutMs = json.optLong("waterfallCompletionTimeoutMs", waterfallCompletionTimeoutMs),
                playbackPollIntervalMs = json.optLong("playbackPollIntervalMs", playbackPollIntervalMs),
                cacheSizeBytes = json.optLong("cacheSizeBytes", cacheSizeBytes),
                stuckBufferingTimeoutMs = json.optLong("stuckBufferingTimeoutMs", stuckBufferingTimeoutMs),
                targetBufferBytes = json.optInt("targetBufferBytes", targetBufferBytes),
                defaultImageDurationMs = json.optLong("defaultImageDurationMs", defaultImageDurationMs),
                crossfadeDurationMs = json.optInt("crossfadeDurationMs", crossfadeDurationMs),
                maxPendingImpressions = json.optInt("maxPendingImpressions", maxPendingImpressions),
                impressionBatchSize = json.optInt("impressionBatchSize", impressionBatchSize),
                maxDedupIds = json.optInt("maxDedupIds", maxDedupIds),
                impressionPersistInterval = json.optInt("impressionPersistInterval", impressionPersistInterval),
                impressionConnectTimeoutS = json.optLong("impressionConnectTimeoutS", impressionConnectTimeoutS),
                impressionReadTimeoutS = json.optLong("impressionReadTimeoutS", impressionReadTimeoutS),
                contentConnectTimeoutS = json.optLong("contentConnectTimeoutS", contentConnectTimeoutS),
                contentReadTimeoutS = json.optLong("contentReadTimeoutS", contentReadTimeoutS)
            )
        }
    }

    // ── Identity (AdvertisingIdCollector, SensorDataCollector, BleBeaconScanner, MdnsDiscovery) ──

    data class IdentityConfig(
        // AdvertisingIdCollector
        val advertisingIdTimeoutMs: Long = 5000L,
        val advertisingIdCacheTtlMs: Long = 3600_000L,      // 1 hour
        // SensorDataCollector
        val sensorReadingTimeoutMs: Long = 2000L,
        // BleBeaconScanner
        val bleDefaultScanDurationMs: Long = 5000L,
        val bleMaxScanDurationMs: Long = 10000L,
        val bleMinScanDurationMs: Long = 1000L,
        val bleMaxDevices: Int = 100,
        // MdnsDiscovery
        val mdnsDefaultDiscoveryDurationMs: Long = 8000L,
        val mdnsMaxDiscoveryDurationMs: Long = 15000L,
        val mdnsMinDiscoveryDurationMs: Long = 3000L,
        val mdnsMaxDevices: Int = 30
    ) {
        fun mergeWith(json: JSONObject?): IdentityConfig {
            if (json == null) return this
            return copy(
                advertisingIdTimeoutMs = json.optLong("advertisingIdTimeoutMs", advertisingIdTimeoutMs),
                advertisingIdCacheTtlMs = json.optLong("advertisingIdCacheTtlMs", advertisingIdCacheTtlMs),
                sensorReadingTimeoutMs = json.optLong("sensorReadingTimeoutMs", sensorReadingTimeoutMs),
                bleDefaultScanDurationMs = json.optLong("bleDefaultScanDurationMs", bleDefaultScanDurationMs),
                bleMaxScanDurationMs = json.optLong("bleMaxScanDurationMs", bleMaxScanDurationMs),
                bleMinScanDurationMs = json.optLong("bleMinScanDurationMs", bleMinScanDurationMs),
                bleMaxDevices = json.optInt("bleMaxDevices", bleMaxDevices),
                mdnsDefaultDiscoveryDurationMs = json.optLong("mdnsDefaultDiscoveryDurationMs", mdnsDefaultDiscoveryDurationMs),
                mdnsMaxDiscoveryDurationMs = json.optLong("mdnsMaxDiscoveryDurationMs", mdnsMaxDiscoveryDurationMs),
                mdnsMinDiscoveryDurationMs = json.optLong("mdnsMinDiscoveryDurationMs", mdnsMinDiscoveryDurationMs),
                mdnsMaxDevices = json.optInt("mdnsMaxDevices", mdnsMaxDevices)
            )
        }
    }

    // ── Scene (SceneChangeDetector) ────────────────────────────────────────

    data class SceneConfig(
        val changeThreshold: Float = 0.3f,
        val minChangeIntervalMs: Long = 60_000L,
        val histogramBins: Int = 64,
        val samplePixels: Int = 10000
    ) {
        fun mergeWith(json: JSONObject?): SceneConfig {
            if (json == null) return this
            return copy(
                changeThreshold = json.optDouble("changeThreshold", changeThreshold.toDouble()).toFloat(),
                minChangeIntervalMs = json.optLong("minChangeIntervalMs", minChangeIntervalMs),
                histogramBins = json.optInt("histogramBins", histogramBins),
                samplePixels = json.optInt("samplePixels", samplePixels)
            )
        }
    }

    // ── FootTrafficEstimator ───────────────────────────────────────────────

    data class FootTrafficConfig(
        // Normalization scales: signal_value / scale, clamped to 0-1
        val faceNormScale: Float = 20f,       // 0 faces = 0.0, 20+ = 1.0
        val audioNormScale: Float = 65f,      // 0 occupancy = 0.0, 65+ = 1.0
        val shadowNormScale: Float = 30f,     // 0 events = 0.0, 30+ = 1.0
        // Signal weights when all 3 signals available
        val faceWeightFull: Float = 0.50f,
        val audioWeightFull: Float = 0.20f,
        val shadowWeightFull: Float = 0.30f,
        // Signal weights when only face + audio (no shadow)
        val faceWeightNoShadow: Float = 0.70f,
        val audioWeightNoShadow: Float = 0.30f,
        // Confidence: fewer than 2 signals
        val lowSignalConfidence: Float = 0.3f,
        // Confidence: 2 signals — threshold for high vs medium agreement
        val twoSignalAgreementThreshold: Float = 0.3f,
        val twoSignalHighConfidence: Float = 0.7f,
        val twoSignalLowConfidence: Float = 0.5f,
        // Confidence: 3 signals — spread thresholds
        val threeSignalTightSpread: Float = 0.2f,
        val threeSignalTightConfidence: Float = 0.9f,
        val threeSignalMediumSpread: Float = 0.4f,
        val threeSignalMediumConfidence: Float = 0.7f,
        val threeSignalWideConfidence: Float = 0.5f
    ) {
        fun mergeWith(json: JSONObject?): FootTrafficConfig {
            if (json == null) return this
            return copy(
                faceNormScale = json.optDouble("faceNormScale", faceNormScale.toDouble()).toFloat(),
                audioNormScale = json.optDouble("audioNormScale", audioNormScale.toDouble()).toFloat(),
                shadowNormScale = json.optDouble("shadowNormScale", shadowNormScale.toDouble()).toFloat(),
                faceWeightFull = json.optDouble("faceWeightFull", faceWeightFull.toDouble()).toFloat(),
                audioWeightFull = json.optDouble("audioWeightFull", audioWeightFull.toDouble()).toFloat(),
                shadowWeightFull = json.optDouble("shadowWeightFull", shadowWeightFull.toDouble()).toFloat(),
                faceWeightNoShadow = json.optDouble("faceWeightNoShadow", faceWeightNoShadow.toDouble()).toFloat(),
                audioWeightNoShadow = json.optDouble("audioWeightNoShadow", audioWeightNoShadow.toDouble()).toFloat(),
                lowSignalConfidence = json.optDouble("lowSignalConfidence", lowSignalConfidence.toDouble()).toFloat(),
                twoSignalAgreementThreshold = json.optDouble("twoSignalAgreementThreshold", twoSignalAgreementThreshold.toDouble()).toFloat(),
                twoSignalHighConfidence = json.optDouble("twoSignalHighConfidence", twoSignalHighConfidence.toDouble()).toFloat(),
                twoSignalLowConfidence = json.optDouble("twoSignalLowConfidence", twoSignalLowConfidence.toDouble()).toFloat(),
                threeSignalTightSpread = json.optDouble("threeSignalTightSpread", threeSignalTightSpread.toDouble()).toFloat(),
                threeSignalTightConfidence = json.optDouble("threeSignalTightConfidence", threeSignalTightConfidence.toDouble()).toFloat(),
                threeSignalMediumSpread = json.optDouble("threeSignalMediumSpread", threeSignalMediumSpread.toDouble()).toFloat(),
                threeSignalMediumConfidence = json.optDouble("threeSignalMediumConfidence", threeSignalMediumConfidence.toDouble()).toFloat(),
                threeSignalWideConfidence = json.optDouble("threeSignalWideConfidence", threeSignalWideConfidence.toDouble()).toFloat()
            )
        }
    }

    // ── SpeechIntelligenceProcessor ────────────────────────────────────────

    data class SpeechProcessingConfig(
        // Hybrid classification confidence thresholds
        val onDeviceConfidenceThreshold: Float = 0.70f,
        val noisyGeminiConfidenceThreshold: Float = 0.30f,
        // Noise-robust preprocessing: high-pass filter cutoff frequencies (Hz)
        val hpCutoffMusic: Int = 300,
        val hpCutoffCrowd: Int = 500,
        val hpCutoffSpeech: Int = 200,
        val hpCutoffAmbient: Int = 150,
        // Low-pass filter for music noise (removes harmonics above this Hz)
        val lpCutoffMusic: Int = 3000
    ) {
        fun mergeWith(json: JSONObject?): SpeechProcessingConfig {
            if (json == null) return this
            return copy(
                onDeviceConfidenceThreshold = json.optDouble("onDeviceConfidenceThreshold", onDeviceConfidenceThreshold.toDouble()).toFloat(),
                noisyGeminiConfidenceThreshold = json.optDouble("noisyGeminiConfidenceThreshold", noisyGeminiConfidenceThreshold.toDouble()).toFloat(),
                hpCutoffMusic = json.optInt("hpCutoffMusic", hpCutoffMusic),
                hpCutoffCrowd = json.optInt("hpCutoffCrowd", hpCutoffCrowd),
                hpCutoffSpeech = json.optInt("hpCutoffSpeech", hpCutoffSpeech),
                hpCutoffAmbient = json.optInt("hpCutoffAmbient", hpCutoffAmbient),
                lpCutoffMusic = json.optInt("lpCutoffMusic", lpCutoffMusic)
            )
        }
    }

    // ── VLM Inference / Engine ─────────────────────────────────────────────

    data class VlmConfig(
        val defaultSamplingIntervalMs: Long = 5_000L,
        val defaultInferenceTimeoutMs: Long = 45_000L,
        val samplingIntervalMinMs: Long = 1_000L,
        val samplingIntervalMaxMs: Long = 60_000L,
        val timeoutMinMs: Long = 2_000L,
        val timeoutMaxMs: Long = 60_000L,
        val bitmapSize: Int = 512,
        val baseMemoryMb: Float = 2.0f,
        val maxConsecutiveFailuresBeforeWarn: Int = 5,
        val gpuFallbackAfterFailures: Int = 3,
        val confidenceDirectJson: Float = 0.95f,
        val confidenceCodeFence: Float = 0.90f,
        val confidenceBraceExtraction: Float = 0.80f,
        val confidenceRegexFallback: Float = 0.50f,
        val confidenceNone: Float = 0.0f,
        val temporalAgreementBase: Float = 0.5f,
        val temporalAgreementBoost: Float = 0.3f,
        val temporalAgreementPenalty: Float = -0.2f,
        val temporalFaceCountThreshold: Float = 2f,
        val temporalPeakFaceCount: Float = 5f,
        val temporalPeakDeltaThreshold: Float = 2f,
        val maxTokens: Int = 512,
        val temperature: Float = 0.1f,
        val numThreads: Int = 4,
        val topP: Float = 0.95f,
        val topK: Int = 40,
        val liteRtMaxImageDim: Int = 512,
        val onnxMaxImageDim: Int = 384,
        val execuTorchMaxImageDim: Int = 448,
        val imageJpegQuality: Int = 85,
        val liteRtMemoryOverheadMultiplier: Float = 1.5f,
        val onnxMemoryOverheadMultiplier: Float = 1.4f,
        val execuTorchMemoryOverheadMultiplier: Float = 1.6f,
        val watchdogThresholdModelFraction: Float = 0.15f,
        val watchdogThresholdMinMb: Float = 100f,
        val watchdogConsecutiveThreshold: Int = 3,
        val watchdogReportInterval: Int = 100
    ) {
        fun mergeWith(json: JSONObject?): VlmConfig {
            if (json == null) return this
            return copy(
                defaultSamplingIntervalMs = json.optLong("defaultSamplingIntervalMs", defaultSamplingIntervalMs),
                defaultInferenceTimeoutMs = json.optLong("defaultInferenceTimeoutMs", defaultInferenceTimeoutMs),
                samplingIntervalMinMs = json.optLong("samplingIntervalMinMs", samplingIntervalMinMs),
                samplingIntervalMaxMs = json.optLong("samplingIntervalMaxMs", samplingIntervalMaxMs),
                timeoutMinMs = json.optLong("timeoutMinMs", timeoutMinMs),
                timeoutMaxMs = json.optLong("timeoutMaxMs", timeoutMaxMs),
                bitmapSize = json.optInt("bitmapSize", bitmapSize),
                baseMemoryMb = json.optDouble("baseMemoryMb", baseMemoryMb.toDouble()).toFloat(),
                maxConsecutiveFailuresBeforeWarn = json.optInt("maxConsecutiveFailuresBeforeWarn", maxConsecutiveFailuresBeforeWarn),
                gpuFallbackAfterFailures = json.optInt("gpuFallbackAfterFailures", gpuFallbackAfterFailures),
                confidenceDirectJson = json.optDouble("confidenceDirectJson", confidenceDirectJson.toDouble()).toFloat(),
                confidenceCodeFence = json.optDouble("confidenceCodeFence", confidenceCodeFence.toDouble()).toFloat(),
                confidenceBraceExtraction = json.optDouble("confidenceBraceExtraction", confidenceBraceExtraction.toDouble()).toFloat(),
                confidenceRegexFallback = json.optDouble("confidenceRegexFallback", confidenceRegexFallback.toDouble()).toFloat(),
                confidenceNone = json.optDouble("confidenceNone", confidenceNone.toDouble()).toFloat(),
                temporalAgreementBase = json.optDouble("temporalAgreementBase", temporalAgreementBase.toDouble()).toFloat(),
                temporalAgreementBoost = json.optDouble("temporalAgreementBoost", temporalAgreementBoost.toDouble()).toFloat(),
                temporalAgreementPenalty = json.optDouble("temporalAgreementPenalty", temporalAgreementPenalty.toDouble()).toFloat(),
                temporalFaceCountThreshold = json.optDouble("temporalFaceCountThreshold", temporalFaceCountThreshold.toDouble()).toFloat(),
                temporalPeakFaceCount = json.optDouble("temporalPeakFaceCount", temporalPeakFaceCount.toDouble()).toFloat(),
                temporalPeakDeltaThreshold = json.optDouble("temporalPeakDeltaThreshold", temporalPeakDeltaThreshold.toDouble()).toFloat(),
                maxTokens = json.optInt("maxTokens", maxTokens),
                temperature = json.optDouble("temperature", temperature.toDouble()).toFloat(),
                numThreads = json.optInt("numThreads", numThreads),
                topP = json.optDouble("topP", topP.toDouble()).toFloat(),
                topK = json.optInt("topK", topK),
                liteRtMaxImageDim = json.optInt("liteRtMaxImageDim", liteRtMaxImageDim),
                onnxMaxImageDim = json.optInt("onnxMaxImageDim", onnxMaxImageDim),
                execuTorchMaxImageDim = json.optInt("execuTorchMaxImageDim", execuTorchMaxImageDim),
                imageJpegQuality = json.optInt("imageJpegQuality", imageJpegQuality),
                liteRtMemoryOverheadMultiplier = json.optDouble("liteRtMemoryOverheadMultiplier", liteRtMemoryOverheadMultiplier.toDouble()).toFloat(),
                onnxMemoryOverheadMultiplier = json.optDouble("onnxMemoryOverheadMultiplier", onnxMemoryOverheadMultiplier.toDouble()).toFloat(),
                execuTorchMemoryOverheadMultiplier = json.optDouble("execuTorchMemoryOverheadMultiplier", execuTorchMemoryOverheadMultiplier.toDouble()).toFloat(),
                watchdogThresholdModelFraction = json.optDouble("watchdogThresholdModelFraction", watchdogThresholdModelFraction.toDouble()).toFloat(),
                watchdogThresholdMinMb = json.optDouble("watchdogThresholdMinMb", watchdogThresholdMinMb.toDouble()).toFloat(),
                watchdogConsecutiveThreshold = json.optInt("watchdogConsecutiveThreshold", watchdogConsecutiveThreshold),
                watchdogReportInterval = json.optInt("watchdogReportInterval", watchdogReportInterval)
            )
        }
    }

    data class CircuitBreakerConfig(
        val failureThreshold: Int = 5,
        val halfOpenRetryIntervalMs: Long = 5L * 60L * 1000L,
        val halfOpenSuccessThreshold: Int = 2
    ) {
        fun mergeWith(json: JSONObject?): CircuitBreakerConfig {
            if (json == null) return this
            return copy(
                failureThreshold = json.optInt("failureThreshold", failureThreshold),
                halfOpenRetryIntervalMs = json.optLong("halfOpenRetryIntervalMs", halfOpenRetryIntervalMs),
                halfOpenSuccessThreshold = json.optInt("halfOpenSuccessThreshold", halfOpenSuccessThreshold)
            )
        }
    }

    data class PerceptionConfig(
        val minTriggerIntervalMs: Long = 5_000L,
        val maxTriggerIntervalMs: Long = 30_000L,
        val noiseDeltaThreshold: Float = 0.3f,
        val minTriggerIntervalBoundMinMs: Long = 1_000L,
        val minTriggerIntervalBoundMaxMs: Long = 60_000L,
        val maxTriggerIntervalBoundMinMs: Long = 5_000L,
        val maxTriggerIntervalBoundMaxMs: Long = 120_000L
    ) {
        fun mergeWith(json: JSONObject?): PerceptionConfig {
            if (json == null) return this
            return copy(
                minTriggerIntervalMs = json.optLong("minTriggerIntervalMs", minTriggerIntervalMs),
                maxTriggerIntervalMs = json.optLong("maxTriggerIntervalMs", maxTriggerIntervalMs),
                noiseDeltaThreshold = json.optDouble("noiseDeltaThreshold", noiseDeltaThreshold.toDouble()).toFloat(),
                minTriggerIntervalBoundMinMs = json.optLong("minTriggerIntervalBoundMinMs", minTriggerIntervalBoundMinMs),
                minTriggerIntervalBoundMaxMs = json.optLong("minTriggerIntervalBoundMaxMs", minTriggerIntervalBoundMaxMs),
                maxTriggerIntervalBoundMinMs = json.optLong("maxTriggerIntervalBoundMinMs", maxTriggerIntervalBoundMinMs),
                maxTriggerIntervalBoundMaxMs = json.optLong("maxTriggerIntervalBoundMaxMs", maxTriggerIntervalBoundMaxMs)
            )
        }
    }

    data class HardwareConfig(
        val osReservationMb: Int = 1500,
        val vlmMemoryFraction: Double = 0.5,
        val floorTierMaxRamMb: Int = 4096,
        val standardTierMaxRamMb: Int = 8192
    ) {
        fun mergeWith(json: JSONObject?): HardwareConfig {
            if (json == null) return this
            return copy(
                osReservationMb = json.optInt("osReservationMb", osReservationMb),
                vlmMemoryFraction = json.optDouble("vlmMemoryFraction", vlmMemoryFraction),
                floorTierMaxRamMb = json.optInt("floorTierMaxRamMb", floorTierMaxRamMb),
                standardTierMaxRamMb = json.optInt("standardTierMaxRamMb", standardTierMaxRamMb)
            )
        }
    }

    data class MemoryConfig(
        val checkIntervalMs: Long = 15_000L,
        val criticalEscalationThreshold: Float = 0.15f,
        val highEscalationThreshold: Float = 0.25f,
        val mediumEscalationThreshold: Float = 0.40f,
        val criticalDeescalationThreshold: Float = 0.20f,
        val highDeescalationThreshold: Float = 0.30f,
        val mediumDeescalationThreshold: Float = 0.45f,
        val criticalFps: Int = 1,
        val highFps: Int = 2,
        val mediumFpsFraction: Float = 0.6f,
        val mediumFpsMin: Int = 2
    ) {
        fun mergeWith(json: JSONObject?): MemoryConfig {
            if (json == null) return this
            return copy(
                checkIntervalMs = json.optLong("checkIntervalMs", checkIntervalMs),
                criticalEscalationThreshold = json.optDouble("criticalEscalationThreshold", criticalEscalationThreshold.toDouble()).toFloat(),
                highEscalationThreshold = json.optDouble("highEscalationThreshold", highEscalationThreshold.toDouble()).toFloat(),
                mediumEscalationThreshold = json.optDouble("mediumEscalationThreshold", mediumEscalationThreshold.toDouble()).toFloat(),
                criticalDeescalationThreshold = json.optDouble("criticalDeescalationThreshold", criticalDeescalationThreshold.toDouble()).toFloat(),
                highDeescalationThreshold = json.optDouble("highDeescalationThreshold", highDeescalationThreshold.toDouble()).toFloat(),
                mediumDeescalationThreshold = json.optDouble("mediumDeescalationThreshold", mediumDeescalationThreshold.toDouble()).toFloat(),
                criticalFps = json.optInt("criticalFps", criticalFps),
                highFps = json.optInt("highFps", highFps),
                mediumFpsFraction = json.optDouble("mediumFpsFraction", mediumFpsFraction.toDouble()).toFloat(),
                mediumFpsMin = json.optInt("mediumFpsMin", mediumFpsMin)
            )
        }
    }

    data class RecoveryConfig(
        val maxRecoveryLevel: Int = 6,
        val maxServiceRestartsPerDay: Int = 3,
        val socketDisconnectAlertMs: Long = 5L * 60L * 1000L,
        val playerReloadCooldownMs: Long = 60_000L,
        val playerRecreateCooldownMs: Long = 120_000L,
        val playerReloadMaxRetries: Int = 3,
        val playerRecreateMaxRetries: Int = 5,
        val cameraMicCooldownMs: Long = 30_000L,
        val cameraMicMaxRetries: Int = 3,
        val maxDailyAttenuationTransitions: Int = 20,
        val escalationCooldownMs: Long = 60_000L,
        val deescalationCooldownMs: Long = 120_000L,
        val restorationHoldTimeMs: Long = 120_000L
    ) {
        fun mergeWith(json: JSONObject?): RecoveryConfig {
            if (json == null) return this
            return copy(
                maxRecoveryLevel = json.optInt("maxRecoveryLevel", maxRecoveryLevel),
                maxServiceRestartsPerDay = json.optInt("maxServiceRestartsPerDay", maxServiceRestartsPerDay),
                socketDisconnectAlertMs = json.optLong("socketDisconnectAlertMs", socketDisconnectAlertMs),
                playerReloadCooldownMs = json.optLong("playerReloadCooldownMs", playerReloadCooldownMs),
                playerRecreateCooldownMs = json.optLong("playerRecreateCooldownMs", playerRecreateCooldownMs),
                playerReloadMaxRetries = json.optInt("playerReloadMaxRetries", playerReloadMaxRetries),
                playerRecreateMaxRetries = json.optInt("playerRecreateMaxRetries", playerRecreateMaxRetries),
                cameraMicCooldownMs = json.optLong("cameraMicCooldownMs", cameraMicCooldownMs),
                cameraMicMaxRetries = json.optInt("cameraMicMaxRetries", cameraMicMaxRetries),
                maxDailyAttenuationTransitions = json.optInt("maxDailyAttenuationTransitions", maxDailyAttenuationTransitions),
                escalationCooldownMs = json.optLong("escalationCooldownMs", escalationCooldownMs),
                deescalationCooldownMs = json.optLong("deescalationCooldownMs", deescalationCooldownMs),
                restorationHoldTimeMs = json.optLong("restorationHoldTimeMs", restorationHoldTimeMs)
            )
        }
    }

}
