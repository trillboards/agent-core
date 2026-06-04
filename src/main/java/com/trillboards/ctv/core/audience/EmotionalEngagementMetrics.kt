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
    val confidence: Float = 0f,

    // Phase 1c — Stable ML Kit face track ID this pose belongs to. Pose is
    // body-level (not per-face), but when a single dominant face is bound
    // PoseEngagementProcessor outputs can flow into the per-(face × ad)
    // accumulator. Null when no face was bound (legacy path OR pose
    // detected without a clear face).
    val faceId: Int? = null
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

    // Movement aggregates — every MovementState enum value carries its own count
    // via a single Map so adding a new state (e.g. depth-aware APPROACHING /
    // DEPARTING when pose-tracking gains depth) requires only updating the enum,
    // never this data class or downstream consumers. Older code had 3 separate
    // Int fields and silently dropped WALKING_SLOW between them — exactly the kind
    // of leaky enum-as-N-fields shape that the Map representation eliminates.
    val movementDistribution: Map<MovementState, Int> = emptyMap(),
    val avgMovementSpeed: Float = 0f,

    // Overall body engagement score
    val bodyEngagementScore: Float = 0f        // 0.0 to 1.0
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("avgFacingAngle", avgFacingAngle.sanitize())
        put("facingScreenPct", facingScreenPct.sanitize())
        put("leaningInCount", leaningInCount)
        put("leaningBackCount", leaningBackCount)
        put("movementDistribution", JSONObject().apply {
            movementDistribution.forEach { (state, count) -> put(state.name, count) }
        })
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
    val confidence: Float = 0f,

    // Phase 1c — Stable ML Kit face track ID this emotion sample belongs to.
    // Bound from the FaceLandmarker output via box-IoU match against the
    // ML Kit faces list (AudienceAnalyzer:1547 — the first of four collapse
    // points fixed in P1c). Null on the legacy fallback path where the
    // FaceLandmarker output cannot be matched to an ML Kit face.
    val faceId: Int? = null
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
    val confidence: Float = 0f,

    // Phase 1c — Stable ML Kit face track ID. Bound from the FaceLandmarker
    // output via box-IoU match against ML Kit's tracked faces (the same
    // findOverlappingFace at AudienceAnalyzer:1889 that the dedup uses).
    val faceId: Int? = null
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

// =========================================================================
// PHASE 1c — PER-(FACE × AD) ATTENTION ACCUMULATOR
// =========================================================================

/**
 * Per-(face_track_id × ad_id) attention accumulator state.
 *
 * Phase 1c (FEIN) — eliminates the four collapse points where per-face
 * structure is discarded before the cloud sees anything:
 *
 *   1. AudienceAnalyzer:1547 — FaceLandmarker emotions zipped by list-index
 *      (now bound by faceId via box-IoU match)
 *   2. AudienceAnalyzer:1714 — pose movement histogram bucketing
 *   3. AudienceSensingService:2104-2110 — blazeface scalar averages
 *   4. AgeGenderProcessor:410-419 — per-face age/gender histogram bins
 *
 * Each `PerFaceAttention` instance accumulates over a window for ONE
 * (adId, faceTrackId) pair. At window flush time (AGGREGATION_WINDOW_MS)
 * the AudienceAnalyzer drains the map and AudienceSensingService packs the
 * result into `audienceSignals.per_face_observations[]` for emission.
 *
 * The accumulator is purely additive — it produces a new wire field but
 * never replaces existing histogram fields. Older edge APKs (without P1c)
 * continue to emit aggregate histograms; the server treats
 * `per_face_observations` as optional and falls back to the histogram path
 * when absent.
 */
data class PerFaceAttention(
    // Per-face dwell and gaze accumulation (seconds, summed over window)
    var dwellSecondsTotal: Float = 0f,
    var gazeSecondsTotal: Float = 0f,
    // Attention score distribution (p50 = median; max = window-max)
    var attentionP50: Float = 0f,
    var attentionMax: Float = 0f,
    // Emotion state — dominant_emotion is the modal label across the window,
    // emotion_confidence_max is the strongest single-sample confidence.
    var dominantEmotion: EmotionType = EmotionType.UNKNOWN,
    var emotionConfidenceMax: Float = 0f,
    // Gaze-on-screen percentage over the window (0..1)
    var gazePctOnScreen: Float = 0f,
    // Primary 3x3 focus region (0..8, center=4) — modal across samples
    var primaryFocusRegion: Int = 4,
    // Optional demographics (only when on-device FaceXFormer ran AND the
    // face was tracked across enough samples; null otherwise)
    var ageBucket: String? = null,
    var gender: String? = null,
    // Track lifetime within the window
    var firstSeenMs: Long = 0L,
    var lastSeenMs: Long = 0L,
    // Sample counter exposed for tests / observability
    var sampleCount: Int = 0,
) {
    // Internal accumulators (private — finalize() consumes them into the
    // emitted fields above). Kept outside the primary constructor so the
    // data class shape remains "what gets emitted on the wire".
    private val attentionSamples: MutableList<Float> = mutableListOf()
    private val emotionCounts: MutableMap<EmotionType, Int> = mutableMapOf()
    private val focusRegionCounts: IntArray = IntArray(9)
    private var gazeOnScreenSamples: Int = 0
    private var gazeTotalSamples: Int = 0

    /**
     * Accumulate a single per-face frame sample into this bucket.
     *
     * Called every frame the (adId, faceId) pair is observed. Updates
     * running counts; the final percentile / mode / max are computed on
     * drain via [finalize].
     */
    fun addSample(
        nowMs: Long,
        attentionScore: Float,
        gazeAttentionScore: Float,
        isLookingAtScreen: Boolean,
        focusRegion: Int,
        emotion: EmotionType,
        emotionConfidence: Float,
        frameIntervalMs: Long,
    ) {
        if (firstSeenMs == 0L) firstSeenMs = nowMs
        lastSeenMs = nowMs
        sampleCount += 1

        // Dwell = total time this face was tracked in this bucket (every
        // sample contributes its frame interval).
        dwellSecondsTotal += frameIntervalMs / 1000f

        // Gaze-on-screen seconds = time where isLookingAtScreen was true.
        if (isLookingAtScreen) {
            gazeSecondsTotal += frameIntervalMs / 1000f
            gazeOnScreenSamples += 1
        }
        gazeTotalSamples += 1

        attentionSamples.add(attentionScore)
        if (attentionScore > attentionMax) attentionMax = attentionScore

        if (emotion != EmotionType.UNKNOWN) {
            emotionCounts[emotion] = (emotionCounts[emotion] ?: 0) + 1
        }
        if (emotionConfidence > emotionConfidenceMax) emotionConfidenceMax = emotionConfidence

        if (focusRegion in 0..8) {
            focusRegionCounts[focusRegion] += 1
        }
    }

    /**
     * Finalize the bucket — compute p50 / mode / pct fields from
     * accumulated samples. Idempotent; safe to call multiple times.
     * Returns this for fluent use.
     */
    fun finalize(): PerFaceAttention {
        if (attentionSamples.isNotEmpty()) {
            val sorted = attentionSamples.sorted()
            attentionP50 = sorted[sorted.size / 2]
        }
        if (emotionCounts.isNotEmpty()) {
            dominantEmotion = emotionCounts.maxByOrNull { it.value }?.key ?: EmotionType.UNKNOWN
        }
        var maxIdx = 4
        var maxCount = -1
        for (i in 0..8) {
            if (focusRegionCounts[i] > maxCount) {
                maxCount = focusRegionCounts[i]
                maxIdx = i
            }
        }
        primaryFocusRegion = maxIdx
        gazePctOnScreen = if (gazeTotalSamples > 0) {
            gazeOnScreenSamples.toFloat() / gazeTotalSamples.toFloat()
        } else 0f
        return this
    }
}
