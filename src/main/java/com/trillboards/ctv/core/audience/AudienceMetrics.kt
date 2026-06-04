package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.SensingConfig

/**
 * Data classes for audience measurement metrics.
 * All data is anonymous and aggregated - no images are stored or transmitted.
 */

/**
 * Snapshot of audience metrics at a point in time.
 */
data class AudienceSnapshot(
    val timestamp: Long = System.currentTimeMillis(),
    val viewerCount: Int = 0,
    val personCount: Int = 0,        // TFLite person detection (PoC)
    val attentionScore: Float = 0f,  // 0.0 to 1.0
    val demographics: Demographics = Demographics(),
    val dwellTime: DwellTimeMetrics = DwellTimeMetrics(),
    val environment: EnvironmentMetrics = EnvironmentMetrics()
)

/**
 * Anonymous demographic aggregates.
 * Uses ranges to preserve privacy.
 */
data class Demographics(
    val ageRanges: Map<String, Int> = emptyMap(),  // e.g., "18-24": 2, "25-34": 1
    val genderEstimates: Map<String, Int> = emptyMap()  // e.g., "male": 2, "female": 1
) {
    companion object {
        val AGE_RANGES = listOf("0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+")
        val GENDERS = listOf("male", "female")

        fun ageRangeFromValue(age: Int): String {
            return when {
                age < 18 -> "0-17"
                age < 25 -> "18-24"
                age < 35 -> "25-34"
                age < 45 -> "35-44"
                age < 55 -> "45-54"
                age < 65 -> "55-64"
                else -> "65+"
            }
        }
    }
}

/**
 * How long viewers spend watching.
 */
data class DwellTimeMetrics(
    val averageSeconds: Float = 0f,
    val maxSeconds: Float = 0f,
    val minSeconds: Float = 0f,
    val totalViewerSeconds: Float = 0f
)

/**
 * Environmental sensor data.
 */
data class EnvironmentMetrics(
    val ambientLightLux: Float = -1f,  // -1 = not available
    val deviceTemperatureC: Float = -1f,  // -1 = not available
    val orientation: String = "unknown"  // "portrait", "landscape", "unknown"
)

/**
 * Individual face detection result (ephemeral, not stored).
 */
data class DetectedFace(
    val id: Int,  // Tracking ID within session
    val boundingBox: FaceRect,
    val headEulerAngleX: Float,  // Pitch (looking up/down)
    val headEulerAngleY: Float,  // Yaw (looking left/right)
    val headEulerAngleZ: Float,  // Roll (tilting head)
    val smilingProbability: Float?,
    val leftEyeOpenProbability: Float?,
    val rightEyeOpenProbability: Float?,
    val estimatedAge: Int?,  // ML Kit doesn't provide this, may need separate model
    val estimatedGender: String?,  // ML Kit doesn't provide this, may need separate model
    val firstSeenTimestamp: Long,
    val lastSeenTimestamp: Long
) {
    /**
     * Calculate attention score based on head pose.
     * Looking directly at screen = higher score.
     */
    fun calculateAttentionScore(): Float {
        val cfg = SensingConfig.get().attention
        // Head facing forward (Y near 0) = looking at screen
        val yawScore = 1f - (kotlin.math.abs(headEulerAngleY) / cfg.yawDivisor).coerceIn(0f, 1f)
        // Head level (X near 0) = engaged
        val pitchScore = 1f - (kotlin.math.abs(headEulerAngleX) / cfg.pitchDivisor).coerceIn(0f, 1f)
        // Eyes open = engaged
        val eyeScore = ((leftEyeOpenProbability ?: 0.5f) + (rightEyeOpenProbability ?: 0.5f)) / 2f

        return (yawScore * cfg.yawWeight + pitchScore * cfg.pitchWeight + eyeScore * cfg.eyeWeight)
    }

    /**
     * Get dwell time in seconds.
     */
    fun getDwellTimeSeconds(): Float {
        return (lastSeenTimestamp - firstSeenTimestamp) / 1000f
    }
}

data class FaceRect(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int
)

/**
 * Configuration for audience measurement.
 */
data class AudienceConfig(
    val enabled: Boolean = false,
    val captureIntervalMs: Long = 1000,  // How often to analyze camera frames
    val reportIntervalMs: Long = 10000,  // How often to report to API (10s for responsive live dashboard)
    val minFaceConfidence: Float = 0.7f,  // Minimum confidence to count a face
    val enableDemographics: Boolean = false,  // Requires additional models
    val enableAttentionTracking: Boolean = true,
    val enableEnvironmentSensors: Boolean = true
)

