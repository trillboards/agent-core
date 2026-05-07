package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.SensingConfig
import org.json.JSONObject

/**
 * Emotional Engagement Metrics - Data classes for pose, emotion, and gaze analysis.
 *
 * These metrics go beyond basic face detection to measure what actually drives conversion:
 * - Body posture: Are they stopping to look or walking past?
 * - Emotional response: Are they engaged, interested, or indifferent?
 * - Gaze direction: Which part of the ad are they looking at?
 *
 * Combined, these provide a much richer picture of audience engagement than
 * simple view counting (like Quividi).
 */

// =========================================================================
// POSE / BODY ENGAGEMENT
// =========================================================================

/**
 * Body posture engagement metrics from a single detection.
 */
data class PoseMetrics(
    val timestamp: Long = System.currentTimeMillis(),

    // Body orientation
    val facingAngle: Float = 0f,              // Degrees from screen-facing (0 = facing screen, 90 = sideways)
    val isFacingScreen: Boolean = true,        // Is body oriented toward screen?

    // Engagement posture
    val leanDirection: LeanDirection = LeanDirection.NEUTRAL,
    val leanMagnitude: Float = 0f,             // 0.0 to 1.0 - how much they're leaning

    // Movement state
    val movementState: MovementState = MovementState.UNKNOWN,
    val movementSpeed: Float = 0f,             // Normalized 0.0 to 1.0

    // Body engagement score (composite)
    val bodyEngagementScore: Float = 0f,       // 0.0 to 1.0

    // Detection confidence
    val confidence: Float = 0f
)

/**
 * Direction the person is leaning.
 */
enum class LeanDirection {
    LEANING_IN,    // Leaning toward the screen (high engagement)
    NEUTRAL,       // Upright, neutral posture
    LEANING_BACK,  // Leaning away (disengagement or skepticism)
    UNKNOWN
}

/**
 * Movement state of the person.
 */
enum class MovementState {
    STOPPED,       // Stationary, viewing the screen
    WALKING_SLOW,  // Walking slowly past (some interest)
    WALKING_FAST,  // Walking quickly past (low/no interest)
    @Deprecated("Requires depth (Z-axis) which 2D pose doesn't provide. Never emitted by PoseEngagementProcessor.")
    APPROACHING,   // Moving toward the screen — dead: requires depth sensor
    @Deprecated("Requires depth (Z-axis) which 2D pose doesn't provide. Never emitted by PoseEngagementProcessor.")
    DEPARTING,     // Moving away from the screen — dead: requires depth sensor
    UNKNOWN
}

/**
 * Aggregated pose metrics for a time window.
 */
data class AggregatedPoseMetrics(
    val windowStart: Long = 0,
    val windowEnd: Long = 0,
    val sampleCount: Int = 0,

    // Body orientation aggregates
    val avgFacingAngle: Float = 0f,
    val facingScreenPct: Float = 0f,           // % of time body was facing screen

    // Engagement posture aggregates
    val leaningInCount: Int = 0,
    val leaningBackCount: Int = 0,
    val avgLeanMagnitude: Float = 0f,

    // Movement aggregates
    val stoppedCount: Int = 0,                  // Count of people who stopped
    val walkingPastCount: Int = 0,              // Count of people walking past
    val approachingCount: Int = 0,              // Count of people approaching
    val avgMovementSpeed: Float = 0f,

    // Overall body engagement score
    val bodyEngagementScore: Float = 0f        // 0.0 to 1.0
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("avgFacingAngle", avgFacingAngle.sanitize())
        put("facingScreenPct", facingScreenPct.sanitize())
        put("leaningInCount", leaningInCount)
        put("leaningBackCount", leaningBackCount)
        put("stoppedCount", stoppedCount)
        put("walkingPastCount", walkingPastCount)
        put("approachingCount", approachingCount)
        put("bodyEngagementScore", bodyEngagementScore.sanitize())
    }

    // Convert NaN/Infinity to 0 for JSON compatibility
    private fun Float.sanitize(): Float = if (this.isNaN() || this.isInfinite()) 0f else this
}

// =========================================================================
// EMOTION / FACIAL EXPRESSION
// =========================================================================

/**
 * Facial emotion metrics from a single detection.
 */
data class EmotionMetrics(
    val timestamp: Long = System.currentTimeMillis(),

    // Detected emotion
    val dominantEmotion: EmotionType = EmotionType.NEUTRAL,
    val emotionConfidence: Float = 0f,

    // Individual emotion scores (0.0 to 1.0)
    val happyScore: Float = 0f,
    val surprisedScore: Float = 0f,
    val neutralScore: Float = 0f,
    val sadScore: Float = 0f,
    val angryScore: Float = 0f,
    val fearScore: Float = 0f,
    val disgustScore: Float = 0f,

    // Derived engagement metrics
    val isPositiveReaction: Boolean = false,    // Happy, surprised
    val isNegativeReaction: Boolean = false,    // Angry, sad, fear, disgust
    val emotionalEngagementScore: Float = 0f,   // 0.0 to 1.0 (neutral = 0, strong emotion = 1)

    // Detection confidence
    val confidence: Float = 0f
)

/**
 * Type of detected emotion.
 */
enum class EmotionType {
    HAPPY,          // Positive engagement
    SURPRISED,      // Attention captured
    NEUTRAL,        // Passive viewing
    SAD,            // Negative (rare in ad context)
    ANGRY,          // Negative (frustration)
    FEARFUL,        // Negative (rare)
    DISGUSTED,      // Negative (dislike)
    CONTEMPT,       // Negative (dismissive/skeptical) — FER+ index 7
    CONFUSED,       // Interest but unsure
    UNKNOWN
}

/**
 * Aggregated emotion metrics for a time window.
 */
data class AggregatedEmotionMetrics(
    val windowStart: Long = 0,
    val windowEnd: Long = 0,
    val sampleCount: Int = 0,

    // Dominant emotion in the window
    val dominantEmotion: EmotionType = EmotionType.NEUTRAL,

    // Reaction counts
    val positiveReactionCount: Int = 0,         // Happy, surprised
    val neutralCount: Int = 0,                   // Neutral viewers
    val negativeReactionCount: Int = 0,          // Angry, sad, etc.

    // Emotion distribution (optional, for detailed analytics)
    val happyCount: Int = 0,
    val surprisedCount: Int = 0,
    val confusedCount: Int = 0,

    // Overall emotional engagement
    val avgEmotionIntensity: Float = 0f,         // How strong are the emotions?
    val emotionalEngagementScore: Float = 0f     // 0.0 to 1.0
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("dominantEmotion", dominantEmotion.name)
        put("positiveReactionCount", positiveReactionCount)
        put("neutralCount", neutralCount)
        put("negativeReactionCount", negativeReactionCount)
        put("happyCount", happyCount)
        put("surprisedCount", surprisedCount)
        put("confusedCount", confusedCount)
        put("emotionalEngagementScore", emotionalEngagementScore.sanitize())
    }

    // Convert NaN/Infinity to 0 for JSON compatibility
    private fun Float.sanitize(): Float = if (this.isNaN() || this.isInfinite()) 0f else this
}

// =========================================================================
// GAZE / ATTENTION DIRECTION
// =========================================================================

/**
 * Gaze tracking metrics from a single detection.
 *
 * Uses ML Kit face landmarks to estimate gaze direction.
 * Screen is divided into a 3x3 grid:
 *
 *   0 | 1 | 2
 *   ---------
 *   3 | 4 | 5
 *   ---------
 *   6 | 7 | 8
 *
 * Region 4 is center (ideal ad placement zone).
 */
data class GazeMetrics(
    val timestamp: Long = System.currentTimeMillis(),

    // Where they're looking (3x3 grid region)
    val focusRegion: Int = 4,                   // 0-8, center = 4
    val isLookingAtScreen: Boolean = true,      // Are eyes directed at screen?

    // Gaze stability (smooth focus vs darting eyes)
    val gazeStability: Float = 0f,              // 0.0 = erratic, 1.0 = stable focus

    // Head pose (from ML Kit)
    val headRotationX: Float = 0f,              // Pitch (looking up/down)
    val headRotationY: Float = 0f,              // Yaw (looking left/right)
    val headRotationZ: Float = 0f,              // Roll (tilting head)

    // Derived attention metrics
    val attentionScore: Float = 0f,             // 0.0 to 1.0

    // Detection confidence
    val confidence: Float = 0f
)

/**
 * Aggregated gaze metrics for a time window.
 */
data class AggregatedGazeMetrics(
    val windowStart: Long = 0,
    val windowEnd: Long = 0,
    val sampleCount: Int = 0,

    // Primary focus region in the window
    val primaryFocusRegion: Int = 4,

    // Looking at screen percentage
    val lookingAtScreenPct: Float = 0f,

    // Gaze stability (average)
    val avgGazeStability: Float = 0f,

    // Region heatmap (which areas got most attention)
    val regionHeatmap: Map<Int, Float> = emptyMap(),  // Region -> attention percentage

    // Overall gaze engagement
    val gazeAttentionScore: Float = 0f          // 0.0 to 1.0
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("primaryFocusRegion", primaryFocusRegion)
        put("lookingAtScreenPct", lookingAtScreenPct.sanitize())
        put("gazeStability", avgGazeStability.sanitize())
        put("heatmap", JSONObject(regionHeatmap.mapValues { it.value.sanitize() }.mapKeys { it.key.toString() }))
        put("gazeAttentionScore", gazeAttentionScore.sanitize())
    }

    // Convert NaN/Infinity to 0 for JSON compatibility
    private fun Float.sanitize(): Float = if (this.isNaN() || this.isInfinite()) 0f else this
}

// =========================================================================
// UNIFIED EMOTIONAL ENGAGEMENT
// =========================================================================

/**
 * Combined emotional engagement metrics - the full picture.
 *
 * This is what differentiates us from Quividi:
 * - Not just "how many people saw it"
 * - But "how did they react, where did they look, did they stop or walk past"
 */
data class EmotionalEngagement(
    val timestamp: Long = System.currentTimeMillis(),

    // Individual component metrics
    val pose: AggregatedPoseMetrics? = null,
    val emotion: AggregatedEmotionMetrics? = null,
    val gaze: AggregatedGazeMetrics? = null,

    // Overall engagement score (weighted composite)
    val overallEngagementScore: Float = 0f,

    // High-level insights
    val audienceReaction: AudienceReaction = AudienceReaction.NEUTRAL,
    val isHighEngagement: Boolean = false,       // Score > 0.7
    val isNegativeReaction: Boolean = false,     // Negative emotion or walking away

    // Confidence in the overall analysis
    val confidence: Float = 0f
) {
    fun toJson(): JSONObject = JSONObject().apply {
        pose?.let { put("pose", it.toJson()) }
        emotion?.let { put("emotion", it.toJson()) }
        gaze?.let { put("gaze", it.toJson()) }
        put("overallEngagementScore", overallEngagementScore.sanitize())
        put("audienceReaction", audienceReaction.name)
        put("isHighEngagement", isHighEngagement)
        put("isNegativeReaction", isNegativeReaction)
        put("confidence", confidence.sanitize())
    }

    // Convert NaN/Infinity to 0 for JSON compatibility
    private fun Float.sanitize(): Float = if (this.isNaN() || this.isInfinite()) 0f else this
}

/**
 * High-level audience reaction classification.
 */
enum class AudienceReaction {
    HIGHLY_ENGAGED,     // Stopped, leaning in, positive emotion, focused gaze
    INTERESTED,         // Some engagement signals
    NEUTRAL,            // Passive viewing
    DISINTERESTED,      // Walking past, looking away
    NEGATIVE            // Negative emotion, walking away
}

// =========================================================================
// CONFIGURATION
// =========================================================================

/**
 * Configuration for emotional engagement processing.
 *
 * Weights and thresholds are read LIVE from SensingConfig on each access
 * so that server-pushed updates (via sensing_config_update) take effect
 * immediately without restarting the sensing service.
 */
data class EmotionalEngagementConfig(
    val enabled: Boolean = true,

    // Enable individual processors
    val enablePoseDetection: Boolean = true,
    val enableEmotionDetection: Boolean = true,
    val enableGazeTracking: Boolean = true,

    // Processing intervals (ms)
    val poseProcessingIntervalMs: Long = 500,    // 2 FPS for pose
    val emotionProcessingIntervalMs: Long = 500, // 2 FPS for emotion
    val gazeProcessingIntervalMs: Long = 200     // 5 FPS for gaze (faster for stability)
) {
    // Engagement thresholds — read live from SensingConfig so server pushes take effect immediately
    val highEngagementThreshold: Float get() = SensingConfig.get().engagement.highEngagementThreshold
    val negativeReactionThreshold: Float get() = SensingConfig.get().engagement.negativeReactionThreshold

    // Scoring weights — read live from SensingConfig so server-pushed learned weights take effect immediately
    val poseWeight: Float get() = SensingConfig.get().engagement.poseWeight
    val emotionWeight: Float get() = SensingConfig.get().engagement.emotionWeight
    val gazeWeight: Float get() = SensingConfig.get().engagement.gazeWeight
}
