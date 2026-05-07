package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ObservationSignalClassifierTest {

    @Test
    fun `face backed windows stay decisionable`() {
        val result = ObservationSignalClassifier.classify(
            avgFaceCountWindow = 1.3,
            currentFaceCount = 1,
            avgPersonCountWindow = 1.3,
            currentPersonCount = 1,
            speechSignalsPresent = false,
            proxySignalsPresent = false,
            proxySignalMismatch = false,
            captureMode = "face_triggered"
        )

        assertEquals("screen_audience", result.observationFamily)
        assertEquals("face_backed", result.evidenceGrade)
        assertEquals("decisionable", result.decisionability)
        assertTrue(result.decisionBlockReasons.isEmpty())
    }

    @Test
    fun `speech only windows are ambient and blocked`() {
        val result = ObservationSignalClassifier.classify(
            avgFaceCountWindow = 0.0,
            currentFaceCount = 0,
            avgPersonCountWindow = 0.0,
            currentPersonCount = 0,
            speechSignalsPresent = true,
            proxySignalsPresent = true,
            proxySignalMismatch = false,
            captureMode = "periodic_scene"
        )

        assertEquals("ambient_commerce_intent", result.observationFamily)
        assertEquals("speech_only", result.evidenceGrade)
        assertEquals("non_decisionable", result.decisionability)
        assertTrue(result.decisionBlockReasons.contains("no_visual_confirmation"))
    }

    @Test
    fun `person backed windows are treated as screen audience`() {
        val result = ObservationSignalClassifier.classify(
            avgFaceCountWindow = 0.0,
            currentFaceCount = 0,
            avgPersonCountWindow = 1.2,
            currentPersonCount = 1,
            speechSignalsPresent = true,
            proxySignalsPresent = true,
            proxySignalMismatch = false,
            captureMode = "periodic_scene"
        )

        assertEquals("screen_audience", result.observationFamily)
        assertEquals("person_backed", result.evidenceGrade)
        assertEquals("decisionable", result.decisionability)
        assertEquals("person_backed", result.measurementQuality)
        assertTrue(result.decisionBlockReasons.isEmpty())
    }

    @Test
    fun `diagnostic captures remain diagnostic only`() {
        val result = ObservationSignalClassifier.classify(
            avgFaceCountWindow = 0.1,
            currentFaceCount = 0,
            avgPersonCountWindow = 1.2,
            currentPersonCount = 1,
            speechSignalsPresent = false,
            proxySignalsPresent = true,
            proxySignalMismatch = true,
            captureMode = "signal_mismatch_probe"
        )

        assertEquals("diagnostic_probe", result.observationFamily)
        assertEquals("proxy_only", result.evidenceGrade)
        assertEquals("non_decisionable", result.decisionability)
        assertTrue(result.decisionBlockReasons.contains("diagnostic_capture"))
    }
}
