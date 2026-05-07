package com.trillboards.ctv.core.audience

data class ObservationSignalClassification(
    val observationFamily: String,
    val evidenceGrade: String,
    val decisionability: String,
    val decisionBlockReasons: List<String>,
    val measurementQuality: String
)

object ObservationSignalClassifier {
    const val SCREEN_AUDIENCE = "screen_audience"
    const val AMBIENT_COMMERCE_INTENT = "ambient_commerce_intent"
    const val DIAGNOSTIC_PROBE = "diagnostic_probe"
    const val EMPTY_SCENE = "empty_scene"

    const val FACE_BACKED = "face_backed"
    const val PERSON_BACKED = "person_backed"
    const val SPEECH_ONLY = "speech_only"
    const val PROXY_ONLY = "proxy_only"
    const val EMPTY = "empty"

    const val DECISIONABLE = "decisionable"
    const val NON_DECISIONABLE = "non_decisionable"

    fun classify(
        avgFaceCountWindow: Double,
        currentFaceCount: Int,
        avgPersonCountWindow: Double,
        currentPersonCount: Int,
        speechSignalsPresent: Boolean,
        proxySignalsPresent: Boolean,
        proxySignalMismatch: Boolean,
        captureMode: String? = null
    ): ObservationSignalClassification {
        val baseEvidenceGrade = when {
            currentFaceCount > 0 || avgFaceCountWindow >= 1.0 -> FACE_BACKED
            currentPersonCount > 0 || avgPersonCountWindow >= 0.75 -> PERSON_BACKED
            speechSignalsPresent -> SPEECH_ONLY
            proxySignalsPresent || proxySignalMismatch -> PROXY_ONLY
            else -> EMPTY
        }
        val evidenceGrade = if (captureMode == "signal_mismatch_probe" && baseEvidenceGrade != FACE_BACKED) {
            PROXY_ONLY
        } else {
            baseEvidenceGrade
        }

        val measurementQuality = when (evidenceGrade) {
            FACE_BACKED -> "face_backed"
            PERSON_BACKED -> if (proxySignalMismatch) "proxy_only" else "person_backed"
            SPEECH_ONLY, PROXY_ONLY -> if (proxySignalMismatch) "proxy_only" else "ambient_proxy_only"
            else -> "empty_scene"
        }

        val observationFamily = when {
            captureMode == "signal_mismatch_probe" -> DIAGNOSTIC_PROBE
            evidenceGrade == FACE_BACKED || evidenceGrade == PERSON_BACKED -> SCREEN_AUDIENCE
            evidenceGrade == EMPTY -> EMPTY_SCENE
            else -> AMBIENT_COMMERCE_INTENT
        }

        val decisionability = if (
            observationFamily == SCREEN_AUDIENCE
                && (evidenceGrade == FACE_BACKED || evidenceGrade == PERSON_BACKED)
        ) {
            DECISIONABLE
        } else {
            NON_DECISIONABLE
        }

        val reasons = linkedSetOf<String>()
        if (observationFamily == DIAGNOSTIC_PROBE) {
            reasons += "diagnostic_capture"
        }
        if (decisionability == NON_DECISIONABLE) {
            when (evidenceGrade) {
                PERSON_BACKED -> reasons += "no_face_confirmation"
                SPEECH_ONLY -> reasons += "no_visual_confirmation"
                PROXY_ONLY -> reasons += "proxy_signal_only"
                EMPTY -> reasons += "empty_scene"
            }
        }
        if (proxySignalMismatch) {
            reasons += "proxy_signal_mismatch"
        }

        return ObservationSignalClassification(
            observationFamily = observationFamily,
            evidenceGrade = evidenceGrade,
            decisionability = decisionability,
            decisionBlockReasons = reasons.toList(),
            measurementQuality = measurementQuality
        )
    }

    fun applyCaptureModeOverride(
        edgeQuality: EdgeQualityTelemetry,
        captureMode: String?
    ): EdgeQualityTelemetry {
        if (captureMode != "signal_mismatch_probe") {
            return edgeQuality
        }

        val reasons = linkedSetOf<String>().apply {
            addAll(edgeQuality.decisionBlockReasons)
            add("diagnostic_capture")
        }

        return edgeQuality.copy(
            observationFamily = DIAGNOSTIC_PROBE,
            decisionability = NON_DECISIONABLE,
            decisionBlockReasons = reasons.toList()
        )
    }
}
