package com.trillboards.ctv.core.stability

import com.trillboards.ctv.core.stability.SubsystemHealthRegistry.SubsystemStatus

private val AUDIO_PROFILE_MODELS = setOf("yamnet", "whisper_tiny")
private val CAMERA_PROFILE_MODELS = setOf(
    "blazeface",
    "age_gender",
    "fer_plus",
    "movenet",
    "efficientdet",
    "yolov8_nano",
    "gemma_4_e2b",
    "moondream_05b",
    "smolvlm_256m"
)

data class SubsystemExpectation(
    val camera: SubsystemStatus,
    val microphone: SubsystemStatus
)

fun resolveProfileSubsystemExpectation(
    profileModels: List<String>,
    cameraAvailable: Boolean,
    microphoneAvailable: Boolean
): SubsystemExpectation {
    val wantsCamera = profileModels.any(CAMERA_PROFILE_MODELS::contains)
    val wantsMicrophone = profileModels.any(AUDIO_PROFILE_MODELS::contains)

    return SubsystemExpectation(
        camera = when {
            !wantsCamera -> SubsystemStatus.DISABLED
            !cameraAvailable -> SubsystemStatus.PERMISSION_DENIED
            else -> SubsystemStatus.ACTIVE
        },
        microphone = when {
            !wantsMicrophone -> SubsystemStatus.DISABLED
            !microphoneAvailable -> SubsystemStatus.PERMISSION_DENIED
            else -> SubsystemStatus.ACTIVE
        }
    )
}
