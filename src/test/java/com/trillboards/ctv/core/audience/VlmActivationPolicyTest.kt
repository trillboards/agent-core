package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.stability.MemoryAttenuationManager
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VlmActivationPolicyTest {

    @Test
    fun `blocks VLM when foreground UI is absent`() {
        val decision = VlmActivationPolicy.decide(
            baseInput(hasForegroundUi = false)
        )

        assertFalse(decision.allowed)
        assertTrue(decision.reason == "ui_unavailable")
    }

    @Test
    fun `blocks VLM when available RAM is below required runtime headroom`() {
        val decision = VlmActivationPolicy.decide(
            baseInput(availableRamMb = 4300)
        )

        assertFalse(decision.allowed)
        assertTrue(decision.reason == "insufficient_available_ram")
    }

    @Test
    fun `blocks VLM when native heap is already hot`() {
        val decision = VlmActivationPolicy.decide(
            baseInput(nativeHeapMb = 1400)
        )

        assertFalse(decision.allowed)
        assertTrue(decision.reason == "native_heap_hot")
    }

    @Test
    fun `blocks VLM under high attenuation`() {
        val decision = VlmActivationPolicy.decide(
            baseInput(attenuationTier = MemoryAttenuationManager.AttenuationTier.HIGH)
        )

        assertFalse(decision.allowed)
        assertTrue(decision.reason == "attenuation_high")
    }

    @Test
    fun `allows gemma 4 on healthy premium device state`() {
        val decision = VlmActivationPolicy.decide(
            baseInput(
                totalRamMb = 11000,
                availableRamMb = 6200,
                nativeHeapMb = 320,
                pressure = ResourceMonitor.MemoryPressure.LOW,
                maxVlmSizeMb = 5400,
                hasForegroundUi = true
            )
        )

        assertTrue(decision.allowed)
        assertTrue(decision.reason == "ok")
    }

    private fun baseInput(
        modelId: String = "gemma_4_e2b",
        totalRamMb: Int = 8192,
        availableRamMb: Int = 6000,
        nativeHeapMb: Int = 256,
        lowMemory: Boolean = false,
        pressure: ResourceMonitor.MemoryPressure = ResourceMonitor.MemoryPressure.MEDIUM,
        attenuationTier: MemoryAttenuationManager.AttenuationTier = MemoryAttenuationManager.AttenuationTier.NORMAL,
        maxVlmSizeMb: Int = 3346,
        hasForegroundUi: Boolean = true,
        modelFileSizeMb: Int? = 2580
    ): VlmActivationInput {
        return VlmActivationInput(
            modelId = modelId,
            totalRamMb = totalRamMb,
            availableRamMb = availableRamMb,
            nativeHeapMb = nativeHeapMb,
            lowMemory = lowMemory,
            pressure = pressure,
            attenuationTier = attenuationTier,
            maxVlmSizeMb = maxVlmSizeMb,
            hasForegroundUi = hasForegroundUi,
            modelFileSizeMb = modelFileSizeMb
        )
    }
}
