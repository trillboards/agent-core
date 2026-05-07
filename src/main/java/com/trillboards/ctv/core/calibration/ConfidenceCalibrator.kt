package com.trillboards.ctv.core.calibration

import android.util.Log

/**
 * Aggregation of all processor outputs for a single sensing window.
 *
 * Each field corresponds to one signal source. Nullable fields indicate that
 * the processor was not active during this window (memory attenuation, hardware
 * missing, or model not loaded).
 *
 * @param faceCount Face count from ML Kit BlazeFace detector.
 * @param personCount Person count from EfficientDet object detector.
 * @param vlmFaceCount Face count extracted from VLM structured output (may differ from ML Kit).
 * @param vlmPersonCount Person count extracted from VLM structured output.
 * @param shadowEvents Motion events from shadow detector (IR/light sensor).
 * @param audioOccupancy Estimated occupancy count from audio classification (YAMNet).
 * @param ferEmotion Dominant emotion string from FER+ model (e.g., "happy", "neutral").
 * @param vlmActivity Activity/mood string from VLM output (e.g., "browsing", "engaged").
 * @param gazeAttention Gaze attention score from gaze tracking processor (0.0 to 1.0).
 * @param bodyEngagement Body engagement score from pose processor (0.0 to 1.0).
 */
data class SignalBundle(
    val faceCount: Int,
    val personCount: Int,
    val vlmFaceCount: Int?,
    val vlmPersonCount: Int?,
    val shadowEvents: Int,
    val audioOccupancy: Int?,
    val ferEmotion: String?,
    val vlmActivity: String?,
    val gazeAttention: Float,
    val bodyEngagement: Float
)

/**
 * Result of multi-signal confidence calibration.
 *
 * Unlike parse-based confidence (which is always 0.95 for valid JSON), calibrated
 * confidence reflects P(observation correct | sensor data) by measuring agreement
 * across independent signal sources.
 *
 * @param overall Weighted average of all per-signal confidences (0.0 to 1.0).
 * @param faceCountConfidence Agreement score for face/person count across detectors.
 * @param emotionConfidence Agreement score for emotion/activity between FER+ and VLM.
 * @param engagementConfidence Agreement score for engagement signals (gaze + pose).
 * @param presenceConfidence Agreement score for human presence (vision + audio + shadow).
 * @param signalCount Number of non-null signals that contributed to calibration.
 */
data class CalibratedConfidence(
    val overall: Float,
    val faceCountConfidence: Float,
    val emotionConfidence: Float,
    val engagementConfidence: Float,
    val presenceConfidence: Float,
    val signalCount: Int
)

/**
 * Computes calibrated confidence scores from multi-signal agreement.
 *
 * The core insight: real confidence = P(observation correct | sensor data), measured
 * by how well independent signal sources agree. A face count that matches across
 * BlazeFace, EfficientDet, and VLM is much more trustworthy than a single detector
 * reporting 0.95 parse confidence.
 *
 * ## Signal Groups
 *
 * 1. **Face Count** — BlazeFace (faceCount) vs EfficientDet (personCount) vs VLM (vlmFaceCount/vlmPersonCount).
 *    Agreement measured by spread (max - min) across available signals.
 *
 * 2. **Emotion** — FER+ (ferEmotion) vs VLM activity (vlmActivity).
 *    Mapped to valence buckets (positive/negative/neutral) for cross-modal comparison.
 *
 * 3. **Engagement** — Gaze attention vs body engagement from pose processor.
 *    Correlated signals: someone looking at the screen should show body orientation.
 *
 * 4. **Presence** — Vision-based person detection vs audio occupancy vs shadow events.
 *    Multi-modal presence confirmation prevents false zeros and false crowds.
 *
 * ## Weights
 *
 * Face count agreement gets the highest weight (0.40) because it's the primary
 * metric used for billing and audience measurement. Engagement and presence get
 * moderate weights (0.20 each). Emotion gets the lowest weight (0.20) because
 * it's supplementary and the FER+/VLM mapping is lossy.
 */
class ConfidenceCalibrator {

    companion object {
        private const val TAG = "ConfidenceCalibrator"

        // Weights for overall score computation (must sum to 1.0)
        internal const val WEIGHT_FACE_COUNT = 0.40f
        internal const val WEIGHT_EMOTION = 0.20f
        internal const val WEIGHT_ENGAGEMENT = 0.20f
        internal const val WEIGHT_PRESENCE = 0.20f

        // Face count agreement thresholds
        internal const val FACE_SPREAD_STRONG = 1    // spread <= 1 = strong agreement
        internal const val FACE_SPREAD_MODERATE = 3  // spread <= 3 = moderate agreement

        // Face count agreement scores
        internal const val FACE_AGREEMENT_STRONG = 0.9f
        internal const val FACE_AGREEMENT_MODERATE = 0.7f
        internal const val FACE_AGREEMENT_WEAK = 0.4f
        internal const val FACE_AGREEMENT_INSUFFICIENT = 0.5f  // only 1 signal

        // Engagement correlation threshold
        internal const val ENGAGEMENT_CORRELATED_THRESHOLD = 0.3f

        // Engagement agreement scores
        internal const val ENGAGEMENT_CORRELATED = 0.85f
        internal const val ENGAGEMENT_MIXED = 0.55f
        internal const val ENGAGEMENT_SINGLE = 0.5f

        // Presence agreement scores
        internal const val PRESENCE_STRONG = 0.9f
        internal const val PRESENCE_MODERATE = 0.7f
        internal const val PRESENCE_SINGLE = 0.5f
        internal const val PRESENCE_CONFLICT = 0.35f
        internal const val PRESENCE_NONE = 0.5f

        // Emotion valence buckets for cross-modal comparison
        private val POSITIVE_EMOTIONS = setOf("happy", "surprised", "excited", "engaged", "browsing", "interested")
        private val NEGATIVE_EMOTIONS = setOf("angry", "sad", "disgusted", "fearful", "frustrated", "bored", "anxious")
        private val NEUTRAL_EMOTIONS = setOf("neutral", "contempt", "calm", "idle", "passing_through")
    }

    /**
     * Compute calibrated confidence from a bundle of all processor outputs.
     *
     * Each signal group is scored independently based on inter-signal agreement,
     * then combined with fixed weights into an overall score.
     *
     * @param bundle All processor outputs for this aggregation window.
     * @return Calibrated confidence with overall and per-signal scores.
     */
    fun calibrate(bundle: SignalBundle): CalibratedConfidence {
        val faceAgreement = computeFaceCountAgreement(bundle)
        val emotionAgreement = computeEmotionAgreement(bundle)
        val engagementAgreement = computeEngagementAgreement(bundle)
        val presenceAgreement = computePresenceAgreement(bundle)

        val signalCount = countNonNullSignals(bundle)

        val overall = (
            faceAgreement * WEIGHT_FACE_COUNT +
            emotionAgreement * WEIGHT_EMOTION +
            engagementAgreement * WEIGHT_ENGAGEMENT +
            presenceAgreement * WEIGHT_PRESENCE
        ).coerceIn(0f, 1f)

        Log.d(TAG, "Calibrated: overall=${String.format("%.3f", overall)}, " +
            "face=${String.format("%.2f", faceAgreement)}, " +
            "emotion=${String.format("%.2f", emotionAgreement)}, " +
            "engagement=${String.format("%.2f", engagementAgreement)}, " +
            "presence=${String.format("%.2f", presenceAgreement)}, " +
            "signals=$signalCount")

        return CalibratedConfidence(
            overall = overall,
            faceCountConfidence = faceAgreement,
            emotionConfidence = emotionAgreement,
            engagementConfidence = engagementAgreement,
            presenceConfidence = presenceAgreement,
            signalCount = signalCount
        )
    }

    /**
     * Face count agreement: compare BlazeFace vs EfficientDet vs VLM counts.
     *
     * Collects all available face/person count signals, computes the spread
     * (max - min), and maps it to a confidence score. Smaller spread = higher
     * agreement = higher confidence.
     *
     * With only 1 signal, returns [FACE_AGREEMENT_INSUFFICIENT] (0.5) — we can't
     * measure agreement with a single data point.
     */
    internal fun computeFaceCountAgreement(bundle: SignalBundle): Float {
        val signals = mutableListOf(bundle.faceCount, bundle.personCount)
        bundle.vlmFaceCount?.let { signals.add(it) }
        bundle.vlmPersonCount?.let { signals.add(it) }

        // Remove duplicate zero-only entries — if face and person are both 0 and
        // no VLM data, that's genuine agreement, not just "no data"
        val nonZeroCount = signals.count { it > 0 }
        val distinctSignals = if (nonZeroCount == 0 && signals.size >= 2) {
            // All zeros with multiple sources = genuine agreement on empty scene
            return FACE_AGREEMENT_STRONG
        } else {
            signals
        }

        if (distinctSignals.size < 2) return FACE_AGREEMENT_INSUFFICIENT

        val spread = distinctSignals.max() - distinctSignals.min()
        return when {
            spread <= FACE_SPREAD_STRONG -> FACE_AGREEMENT_STRONG
            spread <= FACE_SPREAD_MODERATE -> FACE_AGREEMENT_MODERATE
            else -> FACE_AGREEMENT_WEAK
        }
    }

    /**
     * Emotion agreement: compare FER+ dominant emotion with VLM activity/mood.
     *
     * Maps both signals to valence buckets (positive, negative, neutral) and
     * checks if they agree. Direct string comparison would be too strict — "happy"
     * from FER+ and "browsing" from VLM are both positive signals.
     *
     * Returns 0.5 (neutral) when only one or zero emotion signals are available.
     */
    internal fun computeEmotionAgreement(bundle: SignalBundle): Float {
        val ferValence = categorizeValence(bundle.ferEmotion)
        val vlmValence = categorizeValence(bundle.vlmActivity)

        if (ferValence == null && vlmValence == null) return 0.5f
        if (ferValence == null || vlmValence == null) return 0.5f

        return when {
            ferValence == vlmValence -> 0.85f            // Same valence bucket
            ferValence == "neutral" || vlmValence == "neutral" -> 0.6f  // One neutral, one polarized
            else -> 0.3f                                 // Opposite valence
        }
    }

    /**
     * Engagement agreement: correlate gaze attention with body engagement.
     *
     * When both signals are available, checks if they're directionally correlated:
     * high gaze + high body = correlated (0.85), low gaze + low body = correlated,
     * high one + low other = mixed signals (0.55).
     *
     * Returns 0.5 when only one signal source contributes meaningful data.
     */
    internal fun computeEngagementAgreement(bundle: SignalBundle): Float {
        val gazeHigh = bundle.gazeAttention > ENGAGEMENT_CORRELATED_THRESHOLD
        val bodyHigh = bundle.bodyEngagement > ENGAGEMENT_CORRELATED_THRESHOLD

        // Both signals are effectively zero/unavailable
        if (bundle.gazeAttention <= 0f && bundle.bodyEngagement <= 0f) {
            return ENGAGEMENT_SINGLE
        }

        // Only one signal active
        if (bundle.gazeAttention <= 0f || bundle.bodyEngagement <= 0f) {
            return ENGAGEMENT_SINGLE
        }

        // Both active — check directional correlation
        return if (gazeHigh == bodyHigh) {
            ENGAGEMENT_CORRELATED
        } else {
            ENGAGEMENT_MIXED
        }
    }

    /**
     * Presence agreement: fuse vision, audio, and shadow signals.
     *
     * Determines if multiple independent modalities agree on human presence.
     * Vision (person count > 0), audio (occupancy > 0), and shadow (events > 0)
     * are each boolean presence indicators.
     *
     * - 3/3 agree: 0.9 (strong multi-modal confirmation)
     * - 2/3 agree: 0.7 (moderate — one modality may be unreliable)
     * - 1/3 present: 0.5 (single signal — cannot confirm)
     * - 0/3 present: 0.5 (no data — absence is not evidence of absence in sparse sensing)
     * - Conflict (vision says yes, audio says no, or vice versa): 0.35
     */
    internal fun computePresenceAgreement(bundle: SignalBundle): Float {
        val visionPresent = bundle.faceCount > 0 || bundle.personCount > 0 ||
            (bundle.vlmPersonCount ?: 0) > 0
        val audioPresent = bundle.audioOccupancy != null && bundle.audioOccupancy > 0
        val shadowPresent = bundle.shadowEvents > 0

        val presentCount = listOf(visionPresent, audioPresent, shadowPresent).count { it }
        val availableCount = listOf(
            true,  // vision always available (may report 0)
            bundle.audioOccupancy != null,
            true   // shadow always available (may report 0)
        ).count { it }

        // No signals available at all
        if (availableCount == 0) return PRESENCE_NONE

        return when (presentCount) {
            3 -> PRESENCE_STRONG
            2 -> PRESENCE_MODERATE
            1 -> {
                // Check for conflict: one modality says present, another definitively says absent
                if (visionPresent && bundle.audioOccupancy != null && bundle.audioOccupancy == 0) {
                    PRESENCE_CONFLICT
                } else if (audioPresent && !visionPresent && bundle.faceCount == 0 && bundle.personCount == 0) {
                    PRESENCE_CONFLICT
                } else {
                    PRESENCE_SINGLE
                }
            }
            0 -> PRESENCE_NONE
            else -> PRESENCE_NONE
        }
    }

    /**
     * Count the number of non-null/non-default signal sources in the bundle.
     * Used as a quality indicator — more signals = more trustworthy calibration.
     */
    internal fun countNonNullSignals(bundle: SignalBundle): Int {
        var count = 2  // faceCount and personCount are always present
        if (bundle.vlmFaceCount != null) count++
        if (bundle.vlmPersonCount != null) count++
        if (bundle.shadowEvents > 0) count++
        if (bundle.audioOccupancy != null) count++
        if (bundle.ferEmotion != null) count++
        if (bundle.vlmActivity != null) count++
        if (bundle.gazeAttention > 0f) count++
        if (bundle.bodyEngagement > 0f) count++
        return count
    }

    /**
     * Map an emotion/activity string to a valence bucket for cross-modal comparison.
     * Returns null for null/unrecognized strings.
     */
    internal fun categorizeValence(emotion: String?): String? {
        if (emotion == null) return null
        val lower = emotion.lowercase()
        return when {
            lower in POSITIVE_EMOTIONS -> "positive"
            lower in NEGATIVE_EMOTIONS -> "negative"
            lower in NEUTRAL_EMOTIONS -> "neutral"
            else -> null  // Unrecognized — don't assume
        }
    }
}
