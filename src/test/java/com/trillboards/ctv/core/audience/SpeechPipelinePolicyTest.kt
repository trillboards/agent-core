package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Test

class SpeechPipelinePolicyTest {

    @Test
    fun `speech stays active when microphone is available even if profile omits whisper`() {
        val decision = SpeechPipelinePolicy.decide(
            appliedModels = listOf("blazeface", "movenet"),
            hasMicrophoneHardware = true,
            hasMicrophonePermission = true
        )

        assertEquals(SpeechPipelinePolicy.Decision.KEEP_OR_START, decision)
    }

    @Test
    fun `speech stops when microphone permission is missing`() {
        val decision = SpeechPipelinePolicy.decide(
            appliedModels = listOf("whisper_tiny", "blazeface"),
            hasMicrophoneHardware = true,
            hasMicrophonePermission = false
        )

        assertEquals(SpeechPipelinePolicy.Decision.STOP, decision)
    }
}
