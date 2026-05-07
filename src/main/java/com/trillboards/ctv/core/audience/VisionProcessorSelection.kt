package com.trillboards.ctv.core.audience

/**
 * Profile-driven enablement for the camera-side processors that sit behind
 * AudienceAnalyzer. Keeping this mapping pure makes it easy to test and avoids
 * accidentally running heavyweight models that the active sensing profile did
 * not request.
 */
data class VisionProcessorSelection(
    val wantsPose: Boolean,
    val wantsEmotion: Boolean,
    val wantsAgeGender: Boolean,
    val wantsPersonDetection: Boolean,
    val wantsGaze: Boolean
) {
    companion object {
        fun fromModels(
            appliedModels: List<String>,
            enableCameraPresenceBaseline: Boolean = true
        ): VisionProcessorSelection {
            val modelSet = appliedModels.toSet()
            val wantsCameraPresence = enableCameraPresenceBaseline && (
                "blazeface" in modelSet
                || modelSet.any { modelId -> modelId.startsWith("gemma_") || modelId.contains("vlm") }
            )
            // Camera/VLM profiles still need the engagement stack. Without these
            // processors the aggregation window always emits `engagement=none`,
            // which starves the body-language path even when a person is in
            // frame. Explicit model ids still work, but presence profiles should
            // inherit the baseline automatically.
            val wantsPose = "movenet" in modelSet || wantsCameraPresence
            val wantsEmotion = "fer_plus" in modelSet || wantsCameraPresence
            val wantsAgeGender = "age_gender" in modelSet
            // Person detection is part of the visual corroboration baseline for
            // audience measurement. Profiles can still request it explicitly via
            // efficientdet, but camera/VLM presence profiles should not lose full
            // body confirmation just because they omitted the legacy model id.
            val wantsPersonDetection = "efficientdet" in modelSet || wantsCameraPresence

            // Gaze is part of the engagement stack.
            val wantsGaze = wantsPose || wantsEmotion

            return VisionProcessorSelection(
                wantsPose = wantsPose,
                wantsEmotion = wantsEmotion,
                wantsAgeGender = wantsAgeGender,
                wantsPersonDetection = wantsPersonDetection,
                wantsGaze = wantsGaze
            )
        }
    }

    fun activeModelIds(): List<String> {
        val modelIds = mutableListOf<String>()
        if (wantsEmotion) modelIds += "fer_plus"
        if (wantsPose) modelIds += "movenet"
        if (wantsAgeGender) modelIds += "age_gender"
        if (wantsPersonDetection) modelIds += "efficientdet"
        if (wantsGaze) modelIds += "gaze"
        return modelIds
    }
}
