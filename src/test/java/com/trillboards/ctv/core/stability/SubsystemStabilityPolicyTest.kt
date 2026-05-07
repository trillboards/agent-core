package com.trillboards.ctv.core.stability

import com.trillboards.ctv.core.stability.SubsystemHealthRegistry.SubsystemStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SubsystemStabilityPolicyTest {

    @Test
    fun `disabled subsystem is not treated as failed even when stale`() {
        assertFalse(
            shouldTreatSubsystemHealthAsFailed(
                status = SubsystemStatus.DISABLED,
                isStale = true
            )
        )
    }

    @Test
    fun `permission denied subsystem is not treated as failed even when stale`() {
        assertFalse(
            shouldTreatSubsystemHealthAsFailed(
                status = SubsystemStatus.PERMISSION_DENIED,
                isStale = true
            )
        )
    }

    @Test
    fun `active subsystem is treated as failed when stale`() {
        assertTrue(
            shouldTreatSubsystemHealthAsFailed(
                status = SubsystemStatus.ACTIVE,
                isStale = true
            )
        )
    }

    @Test
    fun `profile expectations disable microphone when audio models are absent`() {
        val expectation = resolveProfileSubsystemExpectation(
            profileModels = listOf("blazeface", "movenet"),
            cameraAvailable = true,
            microphoneAvailable = true
        )

        assertEquals(SubsystemStatus.ACTIVE, expectation.camera)
        assertEquals(SubsystemStatus.DISABLED, expectation.microphone)
    }

    @Test
    fun `profile expectations disable camera when only audio models are present`() {
        val expectation = resolveProfileSubsystemExpectation(
            profileModels = listOf("yamnet", "whisper_tiny"),
            cameraAvailable = true,
            microphoneAvailable = true
        )

        assertEquals(SubsystemStatus.DISABLED, expectation.camera)
        assertEquals(SubsystemStatus.ACTIVE, expectation.microphone)
    }

    @Test
    fun `profile expectations respect unavailable hardware for requested subsystems`() {
        val expectation = resolveProfileSubsystemExpectation(
            profileModels = listOf("gemma_4_e2b", "yamnet"),
            cameraAvailable = false,
            microphoneAvailable = false
        )

        assertEquals(SubsystemStatus.PERMISSION_DENIED, expectation.camera)
        assertEquals(SubsystemStatus.PERMISSION_DENIED, expectation.microphone)
    }
}
