package com.trillboards.ctv.core.calibration

import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import org.json.JSONObject
import kotlin.math.atan
import kotlin.math.atan2

/**
 * Derives physical gaze/pose thresholds from screen geometry and camera placement.
 *
 * The core insight: a gaze yaw threshold is not a free parameter -- it is trigonometry.
 * Given a screen of known width at a known viewing distance, the maximum angle at which
 * a viewer can still see the screen is:
 *
 *   derivedYawThreshold = atan(screenWidthCm / 2 / viewingDistanceCm) * (180 / PI)
 *
 * This eliminates the need for manual threshold tuning per deployment. The server sends
 * deployment_config (screen dimensions, viewing distance, camera mount height) and this
 * class computes the correct thresholds automatically.
 *
 * The derived thresholds are applied to SensingConfig.gaze and SensingConfig.body via
 * the existing updateFromJson mechanism, so all downstream processors (FaceLandmarkerProcessor
 * gaze consumers, PoseEngagementProcessor) pick them up transparently.
 */
class DeploymentCalibration private constructor() {

    companion object {
        private const val TAG = "DeploymentCalibration"

        /**
         * Compute derived sensing thresholds from physical deployment geometry.
         *
         * @param config Deployment configuration describing the physical setup
         * @return DerivedThresholds containing all computed values
         */
        fun computeThresholds(config: DeploymentConfig): DerivedThresholds {
            // Convert all Float config values to Double for trig precision
            val widthCm = config.screenWidthCm.toDouble()
            val heightCm = config.screenHeightCm.toDouble()
            val distCm = config.viewingDistanceCm.toDouble()
            val fovDeg = config.cameraHorizontalFovDeg.toDouble()

            // Yaw threshold: half the horizontal subtended angle of the screen
            // atan(halfWidth / distance) gives the angle from center to screen edge
            val halfWidthCm = widthCm / 2.0
            val derivedYawDeg = Math.toDegrees(atan(halfWidthCm / distCm))

            // Pitch threshold: half the vertical subtended angle
            val halfHeightCm = heightCm / 2.0
            val derivedPitchDeg = Math.toDegrees(atan(halfHeightCm / distCm))

            // Facing angle: accounts for camera being offset from screen center vertically.
            // If the camera is mounted above the screen center, viewers looking straight at
            // the screen will appear to have a slight downward pitch from the camera's POV.
            // The facing angle threshold should accommodate this offset.
            // Camera is typically at the top of the screen; screen center is half a screen height below
            val cameraToScreenCenterOffsetCm = heightCm / 2.0
            val facingAngleDeg = Math.toDegrees(
                atan2(cameraToScreenCenterOffsetCm, distCm)
            )
            // The body facing threshold should be at least as wide as the yaw threshold,
            // plus some margin for body orientation lag vs head orientation
            val derivedFacingAngleDeg = (derivedYawDeg * 1.5).coerceAtLeast(facingAngleDeg)

            // Pixel-per-cm at viewing distance: useful for downstream face-size validation
            // This uses the horizontal FOV of the camera to estimate how many pixels
            // correspond to 1 cm at the viewing distance
            val cameraHalfFovRad = Math.toRadians(fovDeg / 2.0)
            val visibleWidthAtDistanceCm = 2.0 * distCm * kotlin.math.tan(cameraHalfFovRad)

            return DerivedThresholds(
                yawThresholdDeg = derivedYawDeg.toFloat(),
                pitchThresholdDeg = derivedPitchDeg.toFloat(),
                facingAngleThresholdDeg = derivedFacingAngleDeg.toFloat(),
                cameraVerticalOffsetDeg = facingAngleDeg.toFloat(),
                visibleWidthAtDistanceCm = visibleWidthAtDistanceCm.toFloat(),
                sourceConfig = config
            )
        }

        /**
         * Compute thresholds from a deployment config and apply them to SensingConfig.
         * This is the main entry point used by SensingProfileManager when it receives
         * a profile with deployment_config.
         *
         * @param config Deployment configuration
         * @return DerivedThresholds that were applied
         */
        fun computeAndApply(config: DeploymentConfig): DerivedThresholds {
            val thresholds = computeThresholds(config)

            // Build a JSON payload that maps to SensingConfig's mergeWith structure
            val configJson = JSONObject().apply {
                put("gaze", JSONObject().apply {
                    put("headYawThreshold", thresholds.yawThresholdDeg)
                    put("headPitchThreshold", thresholds.pitchThresholdDeg)
                })
                put("body", JSONObject().apply {
                    put("facingAngleThreshold", thresholds.facingAngleThresholdDeg)
                })
                put("attention", JSONObject().apply {
                    // Scale divisors proportionally to the thresholds
                    // The yaw divisor is the angle at which attention contribution is halved
                    put("yawDivisor", (thresholds.yawThresholdDeg * 2.25f))
                    put("pitchDivisor", (thresholds.pitchThresholdDeg * 2.0f))
                })
            }

            SensingConfig.updateFromJson(configJson)
            // Mark these fields as auto-calibrated (derived from physical geometry, not
            // server-pushed or hardcoded defaults) so CalibrationTelemetry can report the source
            SensingConfig.markAutoCalibrated("gaze", setOf("headYawThreshold", "headPitchThreshold"))
            SensingConfig.markAutoCalibrated("body", setOf("facingAngleThreshold"))
            SensingConfig.markAutoCalibrated("attention", setOf("yawDivisor", "pitchDivisor"))

            Log.i(TAG, "Applied deployment calibration: yaw=${thresholds.yawThresholdDeg}deg, " +
                "pitch=${thresholds.pitchThresholdDeg}deg, facing=${thresholds.facingAngleThresholdDeg}deg " +
                "(screen=${config.screenWidthCm}x${config.screenHeightCm}cm, " +
                "dist=${config.viewingDistanceCm}cm)")

            return thresholds
        }

        /**
         * Parse a DeploymentConfig from a JSON object (typically from the profile's
         * thresholds.deployment_config key) and compute + apply thresholds.
         *
         * @param json JSON object with deployment config fields
         * @return DerivedThresholds that were applied, or null if JSON was null/empty
         */
        fun fromJsonAndApply(json: JSONObject?): DerivedThresholds? {
            if (json == null || json.length() == 0) return null
            val config = DeploymentConfig().mergeWith(json)
            return computeAndApply(config)
        }
    }
}

/**
 * Physical deployment configuration describing screen geometry and camera placement.
 * Defaults represent a ~10-inch tablet at ~3 feet viewing distance -- the most common
 * Trillboards deployment for small retail screens.
 */
data class DeploymentConfig(
    val screenWidthCm: Float = 25.4f,           // ~10-inch tablet width
    val screenHeightCm: Float = 15.24f,         // ~10-inch tablet height
    val viewingDistanceCm: Float = 91.44f,      // ~3 feet
    val cameraMountHeightCm: Float = 160f,      // ~5'3" typical eye-level mount
    val cameraHorizontalFovDeg: Float = 78f     // Typical tablet front camera FOV
) {
    /**
     * Merge with a JSON object, using current values as defaults for any missing keys.
     * This follows the same pattern as all SensingConfig sub-configs.
     */
    fun mergeWith(json: JSONObject?): DeploymentConfig {
        if (json == null) return this
        return copy(
            screenWidthCm = json.optDouble("screenWidthCm", screenWidthCm.toDouble()).toFloat(),
            screenHeightCm = json.optDouble("screenHeightCm", screenHeightCm.toDouble()).toFloat(),
            viewingDistanceCm = json.optDouble("viewingDistanceCm", viewingDistanceCm.toDouble()).toFloat(),
            cameraMountHeightCm = json.optDouble("cameraMountHeightCm", cameraMountHeightCm.toDouble()).toFloat(),
            cameraHorizontalFovDeg = json.optDouble("cameraHorizontalFovDeg", cameraHorizontalFovDeg.toDouble()).toFloat()
        )
    }

    /** Validate that all dimensions are physically plausible. */
    fun isValid(): Boolean {
        return screenWidthCm > 0f &&
            screenHeightCm > 0f &&
            viewingDistanceCm > 0f &&
            cameraMountHeightCm > 0f &&
            cameraHorizontalFovDeg > 0f &&
            cameraHorizontalFovDeg < 180f
    }
}

/**
 * Thresholds derived from physical screen geometry via trigonometry.
 * These are the actual values applied to SensingConfig.
 */
data class DerivedThresholds(
    /** Maximum yaw angle (degrees) at which the screen edge is still visible */
    val yawThresholdDeg: Float,
    /** Maximum pitch angle (degrees) at which the screen top/bottom is still visible */
    val pitchThresholdDeg: Float,
    /** Maximum body facing angle (degrees) for "oriented toward screen" classification */
    val facingAngleThresholdDeg: Float,
    /** Vertical angle offset (degrees) from camera to screen center */
    val cameraVerticalOffsetDeg: Float,
    /** Visible horizontal width (cm) of the camera FOV at the viewing distance */
    val visibleWidthAtDistanceCm: Float,
    /** The deployment config that produced these thresholds */
    val sourceConfig: DeploymentConfig
) {
    /**
     * Serialize to JSON for heartbeat metadata reporting.
     */
    fun toJson(): JSONObject = JSONObject().apply {
        put("yawThresholdDeg", yawThresholdDeg)
        put("pitchThresholdDeg", pitchThresholdDeg)
        put("facingAngleThresholdDeg", facingAngleThresholdDeg)
        put("cameraVerticalOffsetDeg", cameraVerticalOffsetDeg)
        put("visibleWidthAtDistanceCm", visibleWidthAtDistanceCm)
        put("screenWidthCm", sourceConfig.screenWidthCm)
        put("screenHeightCm", sourceConfig.screenHeightCm)
        put("viewingDistanceCm", sourceConfig.viewingDistanceCm)
    }
}
