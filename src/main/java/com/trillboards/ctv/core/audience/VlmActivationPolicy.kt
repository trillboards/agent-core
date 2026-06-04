package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.SensingConfig
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

/**
 * Decides whether a VLM model can be safely loaded and run on the current device.
 *
 * All numeric thresholds are read from [SensingConfig.get().vlm.activation] so
 * they can be adjusted via remote config without an APK ship.
 *
 * Phase 6 PR A: recalibrated defaults from empirical RSS measurements (Galaxy
 * Tab S11 / SM-X730, Dimensity 9400, 2026-05-27). See [SensingConfig.VlmActivationConfig]
 * for the full derivation.
 *
 * NOTE — competing formula: [com.trillboards.ctv.core.inference.HardwareManifest]
 * contains a separate `calculateMaxVlmSize` that also expresses a "does VLM fit?"
 * answer. That function and its [VlmActivationInput.maxVlmSizeMb] plumbing are the
 * target of Phase 6 PR B (delete / make derived from this policy). They are left
 * intact here to keep this PR narrowly scoped.
 */
internal object VlmActivationPolicy {

    fun decide(input: VlmActivationInput): VlmActivationDecision {
        val cfg = SensingConfig.get().vlm.activation

        val modelSizeMb = input.modelFileSizeMb
            ?: cfg.defaultModelSizeMb[input.modelId]
            ?: 0
        // Empirical: Gemma 4 E2B peak RSS 1.65 GB / 2.58 GB declared = 0.64 multiplier
        // on Galaxy Tab S11. cfg.liteRtRuntimeMultiplier = 0.75 adds 0.11 safety margin.
        val estimatedRuntimeFootprintMb = ceil(modelSizeMb * cfg.liteRtRuntimeMultiplier).toInt()
        val requiredAvailableRamMb = estimatedRuntimeFootprintMb + cfg.minAvailableHeadroomMb

        if (!input.hasForegroundUi) {
            return blocked("ui_unavailable", estimatedRuntimeFootprintMb, requiredAvailableRamMb)
        }

        if (input.totalRamMb < cfg.minDeviceRamMb) {
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

        if (input.nativeHeapMb >= cfg.maxNativeHeapBeforeVlmMb) {
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
