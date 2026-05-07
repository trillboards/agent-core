package com.trillboards.ctv.core.audience

/**
 * Clamps the configurable speech gate to a tablet-safe ceiling before ASR.
 *
 * The original gate of 500 came from lab tuning and is above the RMS values we
 * see from live Galaxy tablet mics in-field. That caused real speech windows to
 * be discarded before Moonshine ran, leaving only stale speech rows in backend
 * stores. Lower server-provided values still win; only overly aggressive gates
 * are reduced.
 */
object SpeechAudioGate {
    private const val MAX_TABLET_SAFE_MIN_AUDIO_LEVEL = 180

    data class Decision(
        val shouldProcess: Boolean,
        val effectiveMinAudioLevel: Int
    )

    fun evaluate(audioLevel: Int, configuredMinAudioLevel: Int): Decision {
        val effectiveMinAudioLevel = configuredMinAudioLevel.coerceAtMost(MAX_TABLET_SAFE_MIN_AUDIO_LEVEL)
        return Decision(
            shouldProcess = audioLevel >= effectiveMinAudioLevel,
            effectiveMinAudioLevel = effectiveMinAudioLevel
        )
    }
}
