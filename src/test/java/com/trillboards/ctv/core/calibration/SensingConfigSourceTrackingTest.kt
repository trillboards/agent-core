package com.trillboards.ctv.core.calibration

import com.trillboards.ctv.core.SensingConfig
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class SensingConfigSourceTrackingTest {

    // SensingConfig is a Kotlin object (singleton). Without explicit reset
    // hooks, server-pushed and auto-calibrated tracking from one test leaks
    // into the next: e.g. test #2 pushes "gaze.headYawThreshold" from server,
    // then test #6 expects "calibrated" provenance for the same field but
    // sees "server" because the prior push survived. Always start tests from
    // a clean singleton state.
    @Before
    fun setUp() {
        SensingConfig.resetForTesting()
    }

    @After
    fun tearDown() {
        SensingConfig.resetForTesting()
    }

    @Test
    fun `wasServerPushed returns false for fields not pushed`() {
        // Before any server push, all fields should be default
        assertFalse(SensingConfig.wasServerPushed("nonexistent", "field"))
    }

    @Test
    fun `updateFromJson tracks pushed field names per section`() {
        val json = JSONObject().apply {
            put("gaze", JSONObject().apply {
                put("headYawThreshold", 25.0)
                put("headPitchThreshold", 18.0)
            })
            put("detection", JSONObject().apply {
                put("personConfidenceThreshold", 0.5)
            })
        }

        SensingConfig.updateFromJson(json)

        assertTrue(SensingConfig.wasServerPushed("gaze", "headYawThreshold"))
        assertTrue(SensingConfig.wasServerPushed("gaze", "headPitchThreshold"))
        assertTrue(SensingConfig.wasServerPushed("detection", "personConfidenceThreshold"))
        // Field not in the push should not be tracked
        assertFalse(SensingConfig.wasServerPushed("gaze", "gazeStabilityThreshold"))
        assertFalse(SensingConfig.wasServerPushed("detection", "minFaceSize"))
    }

    @Test
    fun `getFieldSource returns correct source for each field type`() {
        // Push some fields from server
        SensingConfig.updateFromJson(JSONObject().apply {
            put("emotion", JSONObject().apply {
                put("minConfidence", 0.4)
            })
        })

        // Mark some fields as auto-calibrated
        SensingConfig.markAutoCalibrated("viewability", setOf("darkLuxThreshold"))

        assertEquals("server", SensingConfig.getFieldSource("emotion", "minConfidence"))
        assertEquals("calibrated", SensingConfig.getFieldSource("viewability", "darkLuxThreshold"))
        assertEquals("default", SensingConfig.getFieldSource("body", "facingAngleThreshold"))
    }

    @Test
    fun `markAutoCalibrated accumulates across multiple calls`() {
        SensingConfig.markAutoCalibrated("viewability", setOf("darkLuxThreshold"))
        SensingConfig.markAutoCalibrated("viewability", setOf("dimLuxThreshold"))

        assertTrue(SensingConfig.wasAutoCalibrated("viewability", "darkLuxThreshold"))
        assertTrue(SensingConfig.wasAutoCalibrated("viewability", "dimLuxThreshold"))
    }

    @Test
    fun `getServerPushedFields returns complete map`() {
        SensingConfig.updateFromJson(JSONObject().apply {
            put("audio", JSONObject().apply {
                put("minClassificationConfidence", 0.4)
            })
        })

        val pushed = SensingConfig.getServerPushedFields()
        assertTrue(pushed.containsKey("audio"))
        assertTrue(pushed["audio"]!!.contains("minClassificationConfidence"))
    }

    @Test
    fun `getAutoCalibratedFields returns complete map`() {
        SensingConfig.markAutoCalibrated("gaze", setOf("headYawThreshold"))

        val calibrated = SensingConfig.getAutoCalibratedFields()
        assertTrue(calibrated.containsKey("gaze"))
        assertTrue(calibrated["gaze"]!!.contains("headYawThreshold"))
    }

    @Test
    fun `server push trumps auto-calibrated for same field in getFieldSource`() {
        // First auto-calibrate a field
        SensingConfig.markAutoCalibrated("gaze", setOf("headYawThreshold"))
        assertEquals("calibrated", SensingConfig.getFieldSource("gaze", "headYawThreshold"))

        // Then server pushes the same field
        SensingConfig.updateFromJson(JSONObject().apply {
            put("gaze", JSONObject().apply {
                put("headYawThreshold", 30.0)
            })
        })

        // Server push should take precedence
        assertEquals("server", SensingConfig.getFieldSource("gaze", "headYawThreshold"))
    }
}
