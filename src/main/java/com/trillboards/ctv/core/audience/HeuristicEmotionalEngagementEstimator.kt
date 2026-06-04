package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.SensingConfig
import kotlin.math.abs

/**
 * Conservative fallback when dedicated pose/emotion/gaze processors produce no samples.
 *
 * This preserves a body-language block for live, face-backed windows using only ML Kit
 * head pose, eye openness, smile probability, and dwell time. It should never pretend to
 * be as strong as the full engagement stack, so confidence is intentionally capped.
 */
object HeuristicEmotionalEngagementEstimator {

    fun fromFaces(
        faces: List<DetectedFace>,
        timestamp: Long = System.currentTimeMillis()
    ): EmotionalEngagement? {
        if (faces.isEmpty()) return null

        val gazeCfg = SensingConfig.get().gaze
        val detCfg = SensingConfig.get().detection
        val emotionCfg = SensingConfig.get().emotion
        val engagementCfg = EmotionalEngagementConfig()

        val windowStart = faces.minOfOrNull { it.firstSeenTimestamp } ?: timestamp
        val windowEnd = faces.maxOfOrNull { it.lastSeenTimestamp } ?: timestamp

        data class FaceHeuristic(
            val attention: Float,
            val facingScreen: Boolean,
            val leanDirection: LeanDirection,
            val leanMagnitude: Float,
            val smileScore: Float,
            val dwellSeconds: Float,
            val region: Int
        )

        val heuristics = faces.map { face ->
            val attention = face.calculateAttentionScore().coerceIn(0f, 1f)
            val facingScreen = abs(face.headEulerAngleY) < gazeCfg.headYawThreshold &&
                abs(face.headEulerAngleX) < gazeCfg.headPitchThreshold
            val normalizedPitch = (abs(face.headEulerAngleX) / (gazeCfg.headPitchThreshold * 2)).coerceIn(0f, 1f)
            val leanDirection = when {
                face.headEulerAngleX <= -gazeCfg.headPitchThreshold / 2f -> LeanDirection.LEANING_IN
                face.headEulerAngleX >= gazeCfg.headPitchThreshold / 2f -> LeanDirection.LEANING_BACK
                else -> LeanDirection.NEUTRAL
            }
            val smileScore = (face.smilingProbability ?: 0f).coerceIn(0f, 1f)
            val region = approximateFocusRegion(
                headPitch = face.headEulerAngleX,
                headYaw = face.headEulerAngleY,
                yawThreshold = gazeCfg.headYawThreshold,
                pitchThreshold = gazeCfg.headPitchThreshold
            )

            FaceHeuristic(
                attention = attention,
                facingScreen = facingScreen,
                leanDirection = leanDirection,
                leanMagnitude = normalizedPitch,
                smileScore = smileScore,
                dwellSeconds = face.getDwellTimeSeconds(),
                region = region
            )
        }

        val facingScreenPct = heuristics.count { it.facingScreen }.toFloat() / heuristics.size
        val avgAttention = heuristics.map { it.attention }.average().toFloat().coerceIn(0f, 1f)
        val smilingFaces = heuristics.filter { it.smileScore >= emotionCfg.happySmilingThreshold }
        val avgSmileScore = heuristics.map { it.smileScore }.average().toFloat().coerceIn(0f, 1f)
        val lingerCount = heuristics.count { it.dwellSeconds >= 1.5f }

        val pose = AggregatedPoseMetrics(
            windowStart = windowStart,
            windowEnd = windowEnd,
            sampleCount = heuristics.size,
            avgFacingAngle = faces.map { abs(it.headEulerAngleY) }.average().toFloat(),
            facingScreenPct = facingScreenPct,
            leaningInCount = heuristics.count { it.leanDirection == LeanDirection.LEANING_IN },
            leaningBackCount = heuristics.count { it.leanDirection == LeanDirection.LEANING_BACK },
            avgLeanMagnitude = heuristics.map { it.leanMagnitude }.average().toFloat().coerceIn(0f, 1f),
            // Heuristic estimator (no pose model) only knows lingering-face count;
            // emit a single-state distribution. When pose model loads, AudienceAnalyzer's
            // groupingBy.eachCount() takes over and surfaces all movement states.
            movementDistribution = if (lingerCount > 0) mapOf(MovementState.STOPPED to lingerCount) else emptyMap(),
            avgMovementSpeed = 0f,
            bodyEngagementScore = ((avgAttention * 0.8f) + (facingScreenPct * 0.2f)).coerceIn(0f, 1f)
        )

        val emotion = AggregatedEmotionMetrics(
            windowStart = windowStart,
            windowEnd = windowEnd,
            sampleCount = heuristics.size,
            dominantEmotion = if (smilingFaces.isNotEmpty()) EmotionType.HAPPY else EmotionType.NEUTRAL,
            positiveReactionCount = smilingFaces.size,
            neutralCount = heuristics.size - smilingFaces.size,
            negativeReactionCount = 0,
            happyCount = smilingFaces.size,
            surprisedCount = 0,
            confusedCount = 0,
            avgEmotionIntensity = avgSmileScore,
            emotionalEngagementScore = ((avgSmileScore * 0.7f) + (avgAttention * 0.3f)).coerceIn(0f, 1f)
        )

        val regionCounts = heuristics.groupingBy { it.region }.eachCount()
        val primaryFocusRegion = regionCounts.maxByOrNull { it.value }?.key ?: 4
        val gaze = AggregatedGazeMetrics(
            windowStart = windowStart,
            windowEnd = windowEnd,
            sampleCount = heuristics.size,
            primaryFocusRegion = primaryFocusRegion,
            lookingAtScreenPct = facingScreenPct,
            avgGazeStability = gazeCfg.defaultStability,
            regionHeatmap = regionCounts.mapValues { (_, count) -> count.toFloat() / heuristics.size },
            gazeAttentionScore = avgAttention
        )

        val totalWeight = engagementCfg.poseWeight + engagementCfg.emotionWeight + engagementCfg.gazeWeight
        val overallScore = if (totalWeight > 0f) {
            (
                pose.bodyEngagementScore * engagementCfg.poseWeight +
                    emotion.emotionalEngagementScore * engagementCfg.emotionWeight +
                    gaze.gazeAttentionScore * engagementCfg.gazeWeight
                ) / totalWeight
        } else {
            0f
        }.coerceIn(0f, 1f)

        val audienceReaction = when {
            overallScore > detCfg.highlyEngagedThreshold -> AudienceReaction.HIGHLY_ENGAGED
            overallScore > detCfg.interestedThreshold -> AudienceReaction.INTERESTED
            overallScore > detCfg.neutralThreshold -> AudienceReaction.NEUTRAL
            else -> AudienceReaction.DISINTERESTED
        }

        val confidence = (0.35f + (0.1f * heuristics.size.coerceAtMost(2))).coerceAtMost(0.55f)

        return EmotionalEngagement(
            timestamp = timestamp,
            pose = pose,
            emotion = emotion,
            gaze = gaze,
            overallEngagementScore = overallScore,
            audienceReaction = audienceReaction,
            isHighEngagement = overallScore > engagementCfg.highEngagementThreshold,
            isNegativeReaction = false,
            confidence = confidence
        )
    }

    private fun approximateFocusRegion(
        headPitch: Float,
        headYaw: Float,
        yawThreshold: Float,
        pitchThreshold: Float
    ): Int {
        val col = when {
            headYaw <= -yawThreshold -> 2
            headYaw >= yawThreshold -> 0
            else -> 1
        }
        val row = when {
            headPitch <= -pitchThreshold -> 0
            headPitch >= pitchThreshold -> 2
            else -> 1
        }
        return row * 3 + col
    }
}
