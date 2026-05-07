package com.trillboards.ctv.core.audience

/**
 * Speech intelligence is part of the base microphone telemetry plane, not an
 * expensive profile-only model. Profiles can still influence higher-cost audio
 * inference, but we should not churn speech on and off just because a profile
 * omitted `whisper_tiny`.
 */
object SpeechPipelinePolicy {
    enum class Decision {
        KEEP_OR_START,
        STOP
    }

    fun decide(
        appliedModels: List<String>,
        hasMicrophoneHardware: Boolean,
        hasMicrophonePermission: Boolean
    ): Decision {
        return if (hasMicrophoneHardware && hasMicrophonePermission) {
            Decision.KEEP_OR_START
        } else {
            Decision.STOP
        }
    }
}
