package com.trillboards.ctv.core.audience

import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import org.json.JSONObject

/**
 * Estimates foot traffic past the screen by fusing multiple signal sources.
 *
 * Inputs:
 * - Shadow events from SensorCollector (light sensor fluctuations from passersby)
 * - Max face count from AudienceAnalyzer (engaged viewers looking at screen)
 * - Audio estimated occupancy from AudioClassificationProcessor
 *
 * Heuristic:
 *   Foot traffic is always >= face count (not everyone who walks by looks at the screen)
 *   Shadow events capture passersby who don't look
 *   Audio occupancy captures crowded environments even without camera visibility
 *
 * Output: estimated count, confidence, and passerby ratio (walk-by / look-at)
 */
class FootTrafficEstimator {

    companion object {
        private const val TAG = "FootTraffic"

        // Occupancy string → numeric range midpoint mapping
        private val OCCUPANCY_MAP = mapOf(
            "0-5" to 3,
            "5-20" to 12,
            "20-50" to 35,
            "50+" to 65
        )
    }

    /**
     * Estimate foot traffic for a 10-second sensing window.
     *
     * @param shadowEvents Number of light sensor shadow events (people passing between screen and light)
     * @param maxFaceCount Maximum faces detected in any single frame during the window
     * @param estimatedOccupancy Audio-derived occupancy string ("0-5", "5-20", "20-50", "50+").
     *                            cosmic-brewing-bear C1 deleted this signal source from
     *                            AudioMetrics; null is the new normal and falls back to
     *                            the previous "0-5" baseline weighting (3 people).
     * @param luxVariance Variance in ambient light readings (higher = more movement)
     * @return FootTrafficMetrics with estimated count and confidence
     */
    fun estimate(
        shadowEvents: Int,
        maxFaceCount: Int,
        estimatedOccupancy: String?,
        luxVariance: Float = 0f
    ): FootTrafficMetrics {

        // Null occupancy → fall back to "0-5" (3) so historical estimator
        // behavior is preserved when the audio bucket signal is absent.
        val audioOccupancy = estimatedOccupancy?.let { OCCUPANCY_MAP[it] } ?: 3
        val ftCfg = SensingConfig.get().footTraffic

        // Normalize each signal to 0-1 range before fusion
        val faceSignal = (maxFaceCount / ftCfg.faceNormScale).coerceIn(0f, 1f)
        val audioSignal = (audioOccupancy / ftCfg.audioNormScale).coerceIn(0f, 1f)
        val shadowSignal = (shadowEvents / ftCfg.shadowNormScale).coerceIn(0f, 1f)

        // Determine weights based on which signals are actually available
        val hasShadow = shadowEvents > 0
        val (faceWeight, audioWeight, shadowWeight) = if (hasShadow) {
            Triple(ftCfg.faceWeightFull, ftCfg.audioWeightFull, ftCfg.shadowWeightFull)
        } else {
            Triple(ftCfg.faceWeightNoShadow, ftCfg.audioWeightNoShadow, 0.00f)
        }

        // Fused estimate: weighted normalized signals scaled to absolute count
        // Use audio occupancy as the scale anchor (it's the broadest estimate)
        val fusedNormalized = faceSignal * faceWeight + audioSignal * audioWeight + shadowSignal * shadowWeight
        val estimatedCount = (fusedNormalized * audioOccupancy.toFloat())
            .toInt()
            .coerceAtLeast(maxFaceCount) // Never report less than observed faces

        // Confidence: based on signal count and agreement
        val activeSignals = listOfNotNull(
            if (maxFaceCount > 0) faceSignal else null,
            if (audioOccupancy > 0) audioSignal else null,
            if (shadowEvents > 0) shadowSignal else null
        )
        val confidence = when {
            activeSignals.size < 2 -> ftCfg.lowSignalConfidence
            activeSignals.size == 2 -> {
                val diff = kotlin.math.abs(activeSignals[0] - activeSignals[1])
                if (diff < ftCfg.twoSignalAgreementThreshold) ftCfg.twoSignalHighConfidence else ftCfg.twoSignalLowConfidence
            }
            else -> {
                val spread = activeSignals.max() - activeSignals.min()
                when {
                    spread < ftCfg.threeSignalTightSpread -> ftCfg.threeSignalTightConfidence
                    spread < ftCfg.threeSignalMediumSpread -> ftCfg.threeSignalMediumConfidence
                    else -> ftCfg.threeSignalWideConfidence
                }
            }
        }

        // Determine which method contributed most
        val method = when {
            shadowEvents >= maxFaceCount && shadowEvents >= audioOccupancy -> "shadow_events"
            maxFaceCount >= audioOccupancy -> "face_extrapolation"
            else -> "audio_occupancy"
        }

        // Passerby ratio: how many walk by vs. look at the screen
        val passerbyRatio = if (maxFaceCount > 0) {
            estimatedCount.toFloat() / maxFaceCount.toFloat()
        } else 0f

        Log.d(TAG, "Estimate: count=$estimatedCount (shadow=$shadowEvents, face=$maxFaceCount, " +
                "audio=$audioOccupancy), method=$method, confidence=$confidence, ratio=$passerbyRatio")

        return FootTrafficMetrics(
            estimatedCount = estimatedCount,
            confidence = confidence,
            method = method,
            passerbyRatio = passerbyRatio
        )
    }

    /**
     * Convert metrics to JSON for the audience signal payload.
     */
    fun toJson(metrics: FootTrafficMetrics): JSONObject {
        return JSONObject().apply {
            put("estimatedCount", metrics.estimatedCount)
            put("confidence", metrics.confidence)
            put("method", metrics.method)
            put("passerbyRatio", metrics.passerbyRatio)
        }
    }
}

/**
 * Foot traffic estimation result for a 10-second window.
 */
data class FootTrafficMetrics(
    val estimatedCount: Int = 0,
    val confidence: Float = 0f,
    val method: String = "none",       // shadow_events, face_extrapolation, audio_occupancy
    val passerbyRatio: Float = 0f      // foot_traffic / face_count (>1 means many walk by without looking)
)
