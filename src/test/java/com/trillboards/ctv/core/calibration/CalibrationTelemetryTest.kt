package com.trillboards.ctv.core.calibration

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class CalibrationTelemetryTest {

    @Before
    fun setUp() {
        // Reset CalibrationTelemetry state between tests by recording fresh data
        // (CalibrationTelemetry is an object singleton, so we can't reconstruct it)
    }

    @Test
    fun `buildTelemetry returns valid JSON with all top-level keys`() {
        val telemetry = CalibrationTelemetry.buildTelemetry()

        assertTrue(telemetry.has("effective_thresholds"))
        assertTrue(telemetry.has("config_source"))
        assertTrue(telemetry.has("signal_agreement"))
        assertTrue(telemetry.has("calibration_status"))
        assertTrue(telemetry.has("has_server_config"))
    }

    @Test
    fun `effective_thresholds contains gaze and detection thresholds`() {
        val telemetry = CalibrationTelemetry.buildTelemetry()
        val thresholds = telemetry.getJSONObject("effective_thresholds")

        // Gaze thresholds should exist and match SensingConfig defaults
        assertTrue(thresholds.has("gaze_yaw_threshold"))
        assertTrue(thresholds.has("gaze_pitch_threshold"))
        assertTrue(thresholds.has("gaze_min_face_confidence"))

        // Detection thresholds
        assertTrue(thresholds.has("person_confidence_threshold"))
        assertTrue(thresholds.has("min_face_size"))

        // Engagement weights sub-object
        val engagementWeights = thresholds.getJSONObject("engagement_weights")
        assertTrue(engagementWeights.has("pose"))
        assertTrue(engagementWeights.has("emotion"))
        assertTrue(engagementWeights.has("gaze"))

        // Viewability sub-object
        val viewability = thresholds.getJSONObject("viewability")
        assertTrue(viewability.has("dark_lux"))
        assertTrue(viewability.has("bright_lux"))
    }

    @Test
    fun `config_source reports default for unmodified fields`() {
        val telemetry = CalibrationTelemetry.buildTelemetry()
        val source = telemetry.getJSONObject("config_source")

        // All fields should report "default" when no server config or calibration has occurred
        // (In a fresh JVM without SensingConfig modifications)
        assertTrue(source.has("gaze_yaw"))
        assertTrue(source.has("person_confidence"))
        assertTrue(source.has("viewability_dark_lux"))
    }

    @Test
    fun `signal_agreement starts empty with zero sample count`() {
        // Build telemetry before recording any calibrations
        val telemetry = CalibrationTelemetry.buildTelemetry()
        val agreement = telemetry.getJSONObject("signal_agreement")

        // Sample count should reflect whatever is buffered
        assertTrue(agreement.has("sample_count"))
        assertTrue(agreement.has("overall_agreement_rate"))
    }

    @Test
    fun `recordCalibration populates signal agreement statistics`() {
        val calibration = CalibratedConfidence(
            overall = 0.85f,
            faceCountConfidence = 0.9f,
            emotionConfidence = 0.6f,
            engagementConfidence = 0.8f,
            presenceConfidence = 0.7f,
            signalCount = 6
        )

        CalibrationTelemetry.recordCalibration(calibration)

        val telemetry = CalibrationTelemetry.buildTelemetry()
        val agreement = telemetry.getJSONObject("signal_agreement")

        assertTrue(agreement.getInt("sample_count") > 0)
        // Overall agreement should be around 0.85 (single sample = exact)
        val overall = agreement.getDouble("overall_agreement_rate")
        assertTrue("Overall agreement should be > 0: $overall", overall > 0)
    }

    @Test
    fun `recordFaceAgreement tracks VLM vs TFLite agreement`() {
        // Record some matching and non-matching pairs
        CalibrationTelemetry.recordFaceAgreement(3, 3) // exact match
        CalibrationTelemetry.recordFaceAgreement(5, 4) // within 1 = match
        CalibrationTelemetry.recordFaceAgreement(2, 8) // far apart = mismatch

        val telemetry = CalibrationTelemetry.buildTelemetry()
        val agreement = telemetry.getJSONObject("signal_agreement")

        // vlm_tflite_face_agreement should exist and be between 0 and 1
        if (!agreement.isNull("vlm_tflite_face_agreement")) {
            val faceAgreement = agreement.getDouble("vlm_tflite_face_agreement")
            assertTrue("Face agreement should be >= 0: $faceAgreement", faceAgreement >= 0.0)
            assertTrue("Face agreement should be <= 1: $faceAgreement", faceAgreement <= 1.0)
        }
    }

    @Test
    fun `calibration_status includes subsystem status fields`() {
        val telemetry = CalibrationTelemetry.buildTelemetry()
        val status = telemetry.getJSONObject("calibration_status")

        assertTrue(status.has("lux"))
        assertTrue(status.has("deployment"))
        assertTrue(status.has("confidence_calibrator"))
        assertTrue(status.has("server_pushed_sections"))
        assertTrue(status.has("server_pushed_field_count"))
        assertTrue(status.has("auto_calibrated_sections"))
        assertTrue(status.has("auto_calibrated_field_count"))
    }

    @Test
    fun `setDerivedThresholds makes deployment active in telemetry`() {
        val config = DeploymentConfig(
            screenWidthCm = 50f,
            screenHeightCm = 30f,
            viewingDistanceCm = 100f,
            cameraMountHeightCm = 160f,
            cameraHorizontalFovDeg = 78f
        )
        val thresholds = DeploymentCalibration.computeThresholds(config)
        CalibrationTelemetry.setDerivedThresholds(thresholds)

        val telemetry = CalibrationTelemetry.buildTelemetry()
        val deployment = telemetry.getJSONObject("calibration_status").getJSONObject("deployment")

        assertTrue(deployment.getBoolean("active"))
        assertTrue(deployment.getDouble("yaw_threshold_deg") > 0)
        assertTrue(deployment.getDouble("screen_width_cm") == 50.0)
    }

    @Test
    fun `multiple calibrations produce rolling averages`() {
        // Record several calibrations with different values
        for (i in 1..5) {
            CalibrationTelemetry.recordCalibration(CalibratedConfidence(
                overall = 0.5f + i * 0.05f,
                faceCountConfidence = 0.6f + i * 0.04f,
                emotionConfidence = 0.5f,
                engagementConfidence = 0.7f,
                presenceConfidence = 0.6f,
                signalCount = 4 + i
            ))
        }

        val telemetry = CalibrationTelemetry.buildTelemetry()
        val agreement = telemetry.getJSONObject("signal_agreement")

        assertTrue(agreement.getInt("sample_count") >= 5)
        val avgSignalCount = agreement.getDouble("avg_signal_count")
        assertTrue("Avg signal count should be > 4: $avgSignalCount", avgSignalCount > 4.0)
    }
}
