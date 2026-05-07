package com.trillboards.ctv.core.calibration

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.atan

/**
 * Unit tests for DeploymentCalibration trigonometry.
 *
 * These tests verify that known screen geometries produce correct gaze/pitch thresholds.
 * The math is straightforward: derivedYawThreshold = atan(screenWidthCm/2/viewingDistanceCm) * 180/PI
 * but getting it wrong silently degrades all downstream attention scoring, so we pin
 * expected values for common screen sizes.
 */
class DeploymentCalibrationTest {

    // ── Core trigonometry tests ─────────────────────────────────────────────

    @Test
    fun `default config produces expected thresholds for 10-inch tablet at 3 feet`() {
        val config = DeploymentConfig() // defaults: 25.4cm wide, 91.44cm distance
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // atan(12.7 / 91.44) * 180/PI = atan(0.1389) * 57.296 = 7.91 degrees
        val expectedYaw = Math.toDegrees(atan(12.7 / 91.44)).toFloat()
        assertEquals(expectedYaw, thresholds.yawThresholdDeg, 0.01f)

        // atan(7.62 / 91.44) * 180/PI = atan(0.0833) * 57.296 = 4.76 degrees
        val expectedPitch = Math.toDegrees(atan(7.62 / 91.44)).toFloat()
        assertEquals(expectedPitch, thresholds.pitchThresholdDeg, 0.01f)

        // Verify the values are in a reasonable range for a small screen
        assertTrue("Yaw should be < 15 deg for a 10-inch tablet",
            thresholds.yawThresholdDeg < 15f)
        assertTrue("Pitch should be < 10 deg for a 10-inch tablet",
            thresholds.pitchThresholdDeg < 10f)
    }

    @Test
    fun `55-inch TV at 2 meters produces wide yaw threshold`() {
        // 55-inch diagonal, 16:9 => width ~121.7cm, height ~68.5cm
        val config = DeploymentConfig(
            screenWidthCm = 121.7f,
            screenHeightCm = 68.5f,
            viewingDistanceCm = 200f,
            cameraMountHeightCm = 170f,
            cameraHorizontalFovDeg = 78f
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // atan(60.85 / 200) * 180/PI = atan(0.30425) * 57.296 = 16.92 degrees
        val expectedYaw = Math.toDegrees(atan(60.85 / 200.0)).toFloat()
        assertEquals(expectedYaw, thresholds.yawThresholdDeg, 0.01f)

        // atan(34.25 / 200) * 180/PI = atan(0.17125) * 57.296 = 9.71 degrees
        val expectedPitch = Math.toDegrees(atan(34.25 / 200.0)).toFloat()
        assertEquals(expectedPitch, thresholds.pitchThresholdDeg, 0.01f)

        // 55-inch TV at 2m should have yaw around 17 degrees
        assertTrue("Yaw should be 15-20 deg for a 55-inch TV at 2m",
            thresholds.yawThresholdDeg in 15f..20f)
    }

    @Test
    fun `75-inch TV at 3 meters produces expected thresholds`() {
        // 75-inch diagonal, 16:9 => width ~166cm, height ~93.5cm
        val config = DeploymentConfig(
            screenWidthCm = 166f,
            screenHeightCm = 93.5f,
            viewingDistanceCm = 300f,
            cameraMountHeightCm = 200f,
            cameraHorizontalFovDeg = 78f
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // atan(83 / 300) * 180/PI = atan(0.2767) * 57.296 = 15.47 degrees
        val expectedYaw = Math.toDegrees(atan(83.0 / 300.0)).toFloat()
        assertEquals(expectedYaw, thresholds.yawThresholdDeg, 0.01f)
    }

    @Test
    fun `32-inch kiosk display at 1 meter`() {
        // 32-inch diagonal, 16:9 => width ~70.8cm, height ~39.9cm
        val config = DeploymentConfig(
            screenWidthCm = 70.8f,
            screenHeightCm = 39.9f,
            viewingDistanceCm = 100f,
            cameraMountHeightCm = 150f,
            cameraHorizontalFovDeg = 78f
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // atan(35.4 / 100) * 180/PI = 19.49 degrees
        val expectedYaw = Math.toDegrees(atan(35.4 / 100.0)).toFloat()
        assertEquals(expectedYaw, thresholds.yawThresholdDeg, 0.01f)

        // Close viewing distance on a moderately wide screen => ~20 deg yaw
        assertTrue("Yaw should be ~19 deg for 32-inch at 1m",
            thresholds.yawThresholdDeg in 18f..21f)
    }

    // ── Facing angle tests ──────────────────────────────────────────────────

    @Test
    fun `facing angle is at least 1_5x yaw threshold`() {
        val config = DeploymentConfig()
        val thresholds = DeploymentCalibration.computeThresholds(config)

        assertTrue("Facing angle should be >= 1.5x yaw threshold",
            thresholds.facingAngleThresholdDeg >= thresholds.yawThresholdDeg * 1.5f)
    }

    @Test
    fun `facing angle accounts for camera vertical offset`() {
        val config = DeploymentConfig(
            screenWidthCm = 100f,
            screenHeightCm = 60f,
            viewingDistanceCm = 200f,
            cameraMountHeightCm = 200f,
            cameraHorizontalFovDeg = 78f
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // Camera vertical offset: atan(30 / 200) * 180/PI = 8.53 deg
        val expectedVerticalOffset = Math.toDegrees(kotlin.math.atan2(30.0, 200.0)).toFloat()
        assertEquals(expectedVerticalOffset, thresholds.cameraVerticalOffsetDeg, 0.01f)
    }

    // ── Visible width at distance ───────────────────────────────────────────

    @Test
    fun `visible width at distance scales with FOV and distance`() {
        val config1 = DeploymentConfig(viewingDistanceCm = 100f, cameraHorizontalFovDeg = 78f)
        val config2 = DeploymentConfig(viewingDistanceCm = 200f, cameraHorizontalFovDeg = 78f)

        val t1 = DeploymentCalibration.computeThresholds(config1)
        val t2 = DeploymentCalibration.computeThresholds(config2)

        // Visible width should roughly double when distance doubles
        assertEquals(t2.visibleWidthAtDistanceCm, t1.visibleWidthAtDistanceCm * 2f, 1f)
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    @Test
    fun `very close viewing distance produces wide angles`() {
        val config = DeploymentConfig(
            screenWidthCm = 50f,
            screenHeightCm = 30f,
            viewingDistanceCm = 30f // Very close
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // atan(25/30) = 39.8 degrees -- very wide
        assertTrue("Very close viewing should produce wide yaw threshold",
            thresholds.yawThresholdDeg > 30f)
    }

    @Test
    fun `very far viewing distance produces narrow angles`() {
        val config = DeploymentConfig(
            screenWidthCm = 50f,
            screenHeightCm = 30f,
            viewingDistanceCm = 1000f // 10 meters
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)

        // atan(25/1000) = 1.43 degrees -- very narrow
        assertTrue("Far viewing distance should produce narrow yaw threshold",
            thresholds.yawThresholdDeg < 3f)
    }

    // ── Monotonicity: wider screens or closer viewing = wider angles ────────

    @Test
    fun `wider screen produces wider yaw threshold at same distance`() {
        val narrow = DeploymentCalibration.computeThresholds(
            DeploymentConfig(screenWidthCm = 30f, viewingDistanceCm = 100f)
        )
        val wide = DeploymentCalibration.computeThresholds(
            DeploymentConfig(screenWidthCm = 120f, viewingDistanceCm = 100f)
        )

        assertTrue("Wider screen should produce wider yaw threshold",
            wide.yawThresholdDeg > narrow.yawThresholdDeg)
    }

    @Test
    fun `closer viewing distance produces wider yaw threshold for same screen`() {
        val far = DeploymentCalibration.computeThresholds(
            DeploymentConfig(screenWidthCm = 60f, viewingDistanceCm = 300f)
        )
        val close = DeploymentCalibration.computeThresholds(
            DeploymentConfig(screenWidthCm = 60f, viewingDistanceCm = 100f)
        )

        assertTrue("Closer viewing should produce wider yaw threshold",
            close.yawThresholdDeg > far.yawThresholdDeg)
    }

    @Test
    fun `taller screen produces wider pitch threshold at same distance`() {
        val short = DeploymentCalibration.computeThresholds(
            DeploymentConfig(screenHeightCm = 20f, viewingDistanceCm = 100f)
        )
        val tall = DeploymentCalibration.computeThresholds(
            DeploymentConfig(screenHeightCm = 80f, viewingDistanceCm = 100f)
        )

        assertTrue("Taller screen should produce wider pitch threshold",
            tall.pitchThresholdDeg > short.pitchThresholdDeg)
    }

    // ── DeploymentConfig tests ──────────────────────────────────────────────

    @Test
    fun `DeploymentConfig default values are valid`() {
        val config = DeploymentConfig()
        assertTrue("Default config should be valid", config.isValid())
    }

    @Test
    fun `DeploymentConfig rejects zero width`() {
        val config = DeploymentConfig(screenWidthCm = 0f)
        assertTrue("Zero width should be invalid", !config.isValid())
    }

    @Test
    fun `DeploymentConfig rejects negative distance`() {
        val config = DeploymentConfig(viewingDistanceCm = -1f)
        assertTrue("Negative distance should be invalid", !config.isValid())
    }

    @Test
    fun `DeploymentConfig rejects 180-degree FOV`() {
        val config = DeploymentConfig(cameraHorizontalFovDeg = 180f)
        assertTrue("180-degree FOV should be invalid", !config.isValid())
    }

    @Test
    fun `DeploymentConfig mergeWith applies JSON overrides`() {
        val json = JSONObject().apply {
            put("screenWidthCm", 120.0)
            put("screenHeightCm", 68.0)
            put("viewingDistanceCm", 200.0)
        }
        val merged = DeploymentConfig().mergeWith(json)

        assertEquals(120f, merged.screenWidthCm, 0.01f)
        assertEquals(68f, merged.screenHeightCm, 0.01f)
        assertEquals(200f, merged.viewingDistanceCm, 0.01f)
        // Un-specified fields keep defaults
        assertEquals(160f, merged.cameraMountHeightCm, 0.01f)
        assertEquals(78f, merged.cameraHorizontalFovDeg, 0.01f)
    }

    @Test
    fun `DeploymentConfig mergeWith null returns self`() {
        val original = DeploymentConfig(screenWidthCm = 42f)
        val merged = original.mergeWith(null)
        assertEquals(original, merged)
    }

    // ── DerivedThresholds serialization ─────────────────────────────────────

    @Test
    fun `DerivedThresholds toJson round-trips key fields`() {
        val config = DeploymentConfig(
            screenWidthCm = 121.7f,
            screenHeightCm = 68.5f,
            viewingDistanceCm = 200f
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)
        val json = thresholds.toJson()

        assertEquals(thresholds.yawThresholdDeg.toDouble(),
            json.getDouble("yawThresholdDeg"), 0.01)
        assertEquals(thresholds.pitchThresholdDeg.toDouble(),
            json.getDouble("pitchThresholdDeg"), 0.01)
        assertEquals(121.7, json.getDouble("screenWidthCm"), 0.01)
        assertEquals(200.0, json.getDouble("viewingDistanceCm"), 0.01)
    }

    // ── fromJsonAndApply ────────────────────────────────────────────────────

    @Test
    fun `fromJsonAndApply returns null for null JSON`() {
        val result = DeploymentCalibration.fromJsonAndApply(null)
        assertNull(result)
    }

    @Test
    fun `fromJsonAndApply returns null for empty JSON`() {
        val result = DeploymentCalibration.fromJsonAndApply(JSONObject())
        assertNull(result)
    }

    @Test
    fun `fromJsonAndApply computes thresholds from JSON`() {
        val json = JSONObject().apply {
            put("screenWidthCm", 121.7)
            put("screenHeightCm", 68.5)
            put("viewingDistanceCm", 200.0)
        }
        val result = DeploymentCalibration.fromJsonAndApply(json)
        assertNotNull(result)

        val expectedYaw = Math.toDegrees(atan(60.85 / 200.0)).toFloat()
        assertEquals(expectedYaw, result!!.yawThresholdDeg, 0.01f)
    }

    // ── Known-value regression tests ────────────────────────────────────────
    // Pin specific values so any accidental formula change is caught.

    @Test
    fun `regression - exact threshold for 25_4cm at 91_44cm`() {
        // This is the default config. Pin the exact value.
        val t = DeploymentCalibration.computeThresholds(DeploymentConfig())

        // atan(12.7 / 91.44) = 0.13813 rad = 7.916 deg
        assertEquals(7.916f, t.yawThresholdDeg, 0.01f)

        // atan(7.62 / 91.44) = 0.08318 rad = 4.765 deg
        assertEquals(4.765f, t.pitchThresholdDeg, 0.01f)
    }

    @Test
    fun `regression - exact threshold for 100cm at 100cm`() {
        // Simple geometry: 1:1 ratio at half-width = 50cm at 100cm
        val config = DeploymentConfig(
            screenWidthCm = 100f,
            screenHeightCm = 60f,
            viewingDistanceCm = 100f
        )
        val t = DeploymentCalibration.computeThresholds(config)

        // atan(50/100) = atan(0.5) = 26.565 deg
        assertEquals(26.565f, t.yawThresholdDeg, 0.01f)

        // atan(30/100) = atan(0.3) = 16.699 deg
        assertEquals(16.699f, t.pitchThresholdDeg, 0.01f)
    }

    @Test
    fun `regression - exact threshold for square screen`() {
        // Square screen should have equal yaw and pitch
        val config = DeploymentConfig(
            screenWidthCm = 80f,
            screenHeightCm = 80f,
            viewingDistanceCm = 100f
        )
        val t = DeploymentCalibration.computeThresholds(config)

        assertEquals(t.yawThresholdDeg, t.pitchThresholdDeg, 0.001f)
    }
}
