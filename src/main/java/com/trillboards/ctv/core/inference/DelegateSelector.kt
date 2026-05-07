package com.trillboards.ctv.core.inference

import android.util.Log
import com.google.mediapipe.tasks.core.Delegate
import com.trillboards.ctv.core.audience.DeviceProfile.ChipsetVendor

/**
 * Dynamic delegate selection for MediaPipe/TFLite ML inference.
 *
 * Replaces the hardcoded `Delegate.CPU` across all processors with chipset-aware
 * delegate selection that tries GPU/NNAPI where hardware supports it, with
 * automatic CPU fallback on failure.
 *
 * MediaPipe's BaseOptions supports two delegates: CPU and GPU.
 * GPU delegate internally uses OpenGL ES (works with Adreno, Mali, PowerVR).
 * NNAPI is not directly exposed through MediaPipe's Delegate enum, but the GPU
 * delegate path often leverages hardware accelerators on supported chipsets.
 *
 * Selection logic per chipset vendor:
 * - **Qualcomm**: GPU delegate for vision models (Adreno GPUs are well-tested with MediaPipe)
 * - **MediaTek**: GPU delegate for Dimensity series (Mali GPUs), CPU for older Helio
 * - **Samsung Exynos**: GPU delegate (Mali GPU, well-supported)
 * - **Rockchip**: CPU only (NPU driver quality varies too much, GPU drivers unreliable)
 * - **Amlogic**: CPU only (no reliable ML acceleration on TV box chipsets)
 * - **Unknown**: CPU only (safe default)
 *
 * For VLM models: always try GPU first regardless of vendor, since VLMs benefit
 * most from parallelism and have the largest memory footprint.
 */
object DelegateSelector {
    private const val TAG = "DelegateSelector"

    /**
     * The underlying accelerator type being targeted.
     * This is informational -- the actual MediaPipe delegate will be CPU or GPU.
     */
    enum class AcceleratorType {
        CPU,    // Pure CPU inference
        GPU,    // OpenGL ES GPU delegate (Adreno, Mali, PowerVR)
        NNAPI,  // Android NNAPI (used via GPU delegate path on supported chipsets)
        DSP,    // Digital Signal Processor (Qualcomm Hexagon, informational only)
        NPU     // Neural Processing Unit (Rockchip, MediaTek APU, informational only)
    }

    /**
     * Recommendation from the delegate selector including primary/fallback delegates
     * and the reason for the selection.
     */
    data class DelegateRecommendation(
        val primary: Delegate,
        val fallback: Delegate,
        val acceleratorType: AcceleratorType,
        val reason: String
    )

    /**
     * Returns the best delegate for this device based on chipset vendor and model type.
     *
     * @param vendor The detected chipset vendor from DeviceProfile
     * @param modelType The type of ML model: "vision", "audio", "pose", "vlm", etc.
     * @return DelegateRecommendation with primary delegate, fallback, and reasoning
     */
    fun recommend(vendor: ChipsetVendor, modelType: String): DelegateRecommendation {
        // VLM models always try GPU first regardless of vendor -- they need the parallelism
        if (modelType == "vlm") {
            return DelegateRecommendation(
                primary = Delegate.GPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.GPU,
                reason = "VLM models benefit most from GPU parallelism; GPU first for all vendors"
            )
        }

        return when (vendor) {
            ChipsetVendor.QUALCOMM -> recommendForQualcomm(modelType)
            ChipsetVendor.MEDIATEK -> recommendForMediaTek(modelType)
            ChipsetVendor.EXYNOS -> recommendForExynos(modelType)
            ChipsetVendor.ROCKCHIP -> recommendForRockchip(modelType)
            ChipsetVendor.AMLOGIC -> recommendForAmlogic(modelType)
            ChipsetVendor.UNKNOWN -> recommendForUnknown(modelType)
        }
    }

    /**
     * Qualcomm: GPU delegate for vision models (Adreno is reliable with MediaPipe),
     * CPU for audio models (audio processing is sequential, no GPU benefit).
     */
    private fun recommendForQualcomm(modelType: String): DelegateRecommendation {
        return when (modelType) {
            "audio" -> DelegateRecommendation(
                primary = Delegate.CPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.CPU,
                reason = "Qualcomm: audio inference is sequential, CPU is optimal"
            )
            else -> DelegateRecommendation(
                primary = Delegate.GPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.GPU,
                reason = "Qualcomm Adreno GPU: well-tested with MediaPipe vision models"
            )
        }
    }

    /**
     * MediaTek: GPU delegate for vision models on Dimensity series (Mali GPUs),
     * CPU for audio. MediaTek's APU (AI Processing Unit) is accessed via NNAPI
     * which the GPU delegate can sometimes leverage, but direct GPU is more reliable.
     */
    private fun recommendForMediaTek(modelType: String): DelegateRecommendation {
        return when (modelType) {
            "audio" -> DelegateRecommendation(
                primary = Delegate.CPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.CPU,
                reason = "MediaTek: audio inference is sequential, CPU is optimal"
            )
            else -> DelegateRecommendation(
                primary = Delegate.GPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.GPU,
                reason = "MediaTek Mali GPU: GPU delegate for vision; APU access via NNAPI path"
            )
        }
    }

    /**
     * Samsung Exynos: GPU delegate for vision models (Mali GPU, well-supported).
     */
    private fun recommendForExynos(modelType: String): DelegateRecommendation {
        return when (modelType) {
            "audio" -> DelegateRecommendation(
                primary = Delegate.CPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.CPU,
                reason = "Exynos: audio inference is sequential, CPU is optimal"
            )
            else -> DelegateRecommendation(
                primary = Delegate.GPU,
                fallback = Delegate.CPU,
                acceleratorType = AcceleratorType.GPU,
                reason = "Exynos Mali GPU: well-supported for MediaPipe vision models"
            )
        }
    }

    /**
     * Rockchip: CPU only. Despite having NPU (6 TOPS on RK3588), the NPU drivers
     * are not accessible via MediaPipe's delegate system, and GPU drivers on Rockchip
     * boards are unreliable for ML inference.
     */
    private fun recommendForRockchip(modelType: String): DelegateRecommendation {
        return DelegateRecommendation(
            primary = Delegate.CPU,
            fallback = Delegate.CPU,
            acceleratorType = AcceleratorType.CPU,
            reason = "Rockchip: NPU driver quality varies, GPU unreliable for ML; CPU only"
        )
    }

    /**
     * Amlogic: CPU only. TV box chipsets (S905, S912) have no reliable ML acceleration
     * through MediaPipe's delegate system.
     */
    private fun recommendForAmlogic(modelType: String): DelegateRecommendation {
        return DelegateRecommendation(
            primary = Delegate.CPU,
            fallback = Delegate.CPU,
            acceleratorType = AcceleratorType.CPU,
            reason = "Amlogic: no reliable ML acceleration on TV box chipsets; CPU only"
        )
    }

    /**
     * Unknown vendor: CPU only as the safe default.
     */
    private fun recommendForUnknown(modelType: String): DelegateRecommendation {
        return DelegateRecommendation(
            primary = Delegate.CPU,
            fallback = Delegate.CPU,
            acceleratorType = AcceleratorType.CPU,
            reason = "Unknown chipset: CPU is the safe default"
        )
    }

    /**
     * Execute a block with the primary delegate, falling back to the fallback delegate
     * if the primary fails. This handles the common pattern where GPU initialization
     * might fail on some devices due to driver issues.
     *
     * @param primary The preferred delegate to try first
     * @param fallback The fallback delegate if primary fails (defaults to CPU)
     * @param block Lambda that receives the delegate to use. Called once with primary,
     *              and again with fallback if the first call throws.
     */
    fun withFallback(
        primary: Delegate,
        fallback: Delegate = Delegate.CPU,
        block: (Delegate) -> Unit
    ) {
        try {
            block(primary)
            Log.i(TAG, "Successfully initialized with delegate: $primary")
        } catch (e: Exception) {
            if (primary == fallback) {
                // Primary and fallback are the same (both CPU), just rethrow
                throw e
            }
            Log.w(TAG, "Failed to initialize with $primary delegate, falling back to $fallback: ${e.message}")
            try {
                block(fallback)
                Log.i(TAG, "Successfully initialized with fallback delegate: $fallback")
            } catch (fallbackError: Exception) {
                Log.e(TAG, "Failed to initialize with fallback delegate $fallback", fallbackError)
                throw fallbackError
            }
        }
    }

    /**
     * Convenience method: recommend and log the decision.
     */
    fun recommendAndLog(vendor: ChipsetVendor, modelType: String): DelegateRecommendation {
        val recommendation = recommend(vendor, modelType)
        Log.i(TAG, "Delegate recommendation for $modelType on $vendor: " +
            "primary=${recommendation.primary}, fallback=${recommendation.fallback}, " +
            "accelerator=${recommendation.acceleratorType}, reason=${recommendation.reason}")
        return recommendation
    }
}
