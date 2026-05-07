package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeechAudioGateTest {

    @Test
    fun `configured gate is clamped to tablet-safe ceiling`() {
        val decision = SpeechAudioGate.evaluate(audioLevel = 245, configuredMinAudioLevel = 500)

        assertEquals(180, decision.effectiveMinAudioLevel)
        assertTrue(decision.shouldProcess)
    }

    @Test
    fun `silent windows still get skipped`() {
        val decision = SpeechAudioGate.evaluate(audioLevel = 0, configuredMinAudioLevel = 500)

        assertEquals(180, decision.effectiveMinAudioLevel)
        assertFalse(decision.shouldProcess)
    }

    @Test
    fun `lower server-tuned thresholds are respected`() {
        val decision = SpeechAudioGate.evaluate(audioLevel = 150, configuredMinAudioLevel = 120)

        assertEquals(120, decision.effectiveMinAudioLevel)
        assertTrue(decision.shouldProcess)
    }
}
