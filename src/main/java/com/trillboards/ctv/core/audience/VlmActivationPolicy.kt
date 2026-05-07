package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.stability.MemoryAttenuationManager
import kotlin.math.ceil

internal data class VlmActivationInput(
    val modelId: String,
    val totalRamMb: Int,
    val availableRamMb: Int,
    val nativeHeapMb: Int,
    val lowMemory: Boolean,
    val pressure: ResourceMonitor.MemoryPressure,
    val attenuationTier: MemoryAttenuationManager.AttenuationTier,
    val maxVlmSizeMb: Int,
    val hasForegroundUi: Boolean,
    val modelFileSizeMb: Int? = null
)

internal data class VlmActivationDecision(
    val allowed: Boolean,
    val reason: String,
    val estimatedRuntimeFootprintMb: Int,
    val requiredAvailableRamMb: Int
)

internal object VlmActivationPolicy {
    private const val MIN_DEVICE_RAM_MB = 2048
    private const val MIN_AVAILABLE_HEADROOM_MB = 1536
    private const val MAX_NATIVE_HEAP_BEFORE_VLM_MB = 1024
    private const val LITERT_RUNTIME_MULTIPLIER = 1.5

    private val defaultModelSizeMb = mapOf(
        "gemma_4_e2b" to 2580,
        "gemma_3n_e2b" to 3660
    )

    fun decide(input: VlmActivationInput): VlmActivationDecision {
        val modelSizeMb = input.modelFileSizeMb ?: defaultModelSizeMb[input.modelId] ?: 0
        val estimatedRuntimeFootprintMb = ceil(modelSizeMb * LITERT_RUNTIME_MULTIPLIER).toInt()
        val requiredAvailableRamMb = estimatedRuntimeFootprintMb + MIN_AVAILABLE_HEADROOM_MB

        if (!input.hasForegroundUi) {
            return blocked("ui_unavailable", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.totalRamMb < MIN_DEVICE_RAM_MB) {
            return blocked("device_ram_floor", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.maxVlmSizeMb > 0 && modelSizeMb > input.maxVlmSizeMb) {
            return blocked("manifest_capacity_exceeded", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.lowMemory) {
            return blocked("system_low_memory", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.attenuationTier.ordinal >= MemoryAttenuationManager.AttenuationTier.HIGH.ordinal) {
            return blocked("attenuation_high", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.pressure.ordinal >= ResourceMonitor.MemoryPressure.HIGH.ordinal) {
            return blocked("memory_pressure_high", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.nativeHeapMb >= MAX_NATIVE_HEAP_BEFORE_VLM_MB) {
            return blocked("native_heap_hot", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.availableRamMb < requiredAvailableRamMb) {
            return blocked("insufficient_available_ram", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        return VlmActivationDecision(
            allowed = true,
            reason = "ok",
            estimatedRuntimeFootprintMb = estimatedRuntimeFootprintMb,
            requiredAvailableRamMb = requiredAvailableRamMb
        )
    }

    private fun blocked(
        reason: String,
        estimatedRuntimeFootprintMb: Int,
        requiredAvailableRamMb: Int
    ): VlmActivationDecision {
        return VlmActivationDecision(
            allowed = false,
            reason = reason,
            estimatedRuntimeFootprintMb = estimatedRuntimeFootprintMb,
            requiredAvailableRamMb = requiredAvailableRamMb
        )
    }
}
