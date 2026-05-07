package com.trillboards.ctv.core.inference.vlm

import android.content.Context
import android.util.Log

/**
 * Factory for creating [VLMEngine] instances based on model format strings.
 *
 * The `model_format` field in the model registry determines which engine runtime
 * to use for a given model. This factory maps format strings to concrete engine
 * implementations, enabling new model formats to be supported without changing
 * [VLMInferenceProcessor] or any other consumer code.
 *
 * ## Supported Formats
 *
 * | Format         | Engine              | Example Models                     |
 * |----------------|---------------------|------------------------------------|
 * | `litert-lm`    | [LiteRTLMEngine]    | Gemma 3n, Gemma 4 (LiteRT native)  |
 * | `onnx-vlm`     | [OnnxVLMEngine]     | SmolVLM 256M, Phi-3-vision (ONNX)  |
 * | `executorch`   | [ExecuTorchEngine]  | Llama 3.2 Vision, SmolVLM (Meta)   |
 *
 * Note: GGUF/llama.cpp support was removed. LiteRT-LM is the sole VLM runtime.
 * No device with <8 GB RAM should run a VLM, so small GGUF models (SmolVLM 175MB,
 * Moondream 375MB) are unnecessary.
 *
 * ## Graceful Degradation
 * If an engine's underlying SDK is not bundled in the APK, [createEngine] still
 * returns the engine instance. The engine's [VLMEngine.loadModel] will return
 * false when it detects the missing SDK at runtime, allowing the processor to
 * fall back gracefully.
 *
 * ## Extensibility
 * To add a new model format:
 * 1. Create a new [VLMEngine] implementation (e.g., `MediaPipeLLMEngine`)
 * 2. Add the format string to the `when` block in [createEngine]
 * 3. Register the format in the model registry (`model_format` column)
 *
 * No changes to [VLMInferenceProcessor] or any caller are needed.
 */
object VLMEngineFactory {

    private const val TAG = "VLMEngineFactory"

    /**
     * All model format strings recognized by this factory.
     * Useful for validation and UI/telemetry.
     */
    val supportedFormats: Set<String> = setOf(
        "litert-lm",
        "onnx-vlm",
        "executorch"
    )

    /**
     * Create a [VLMEngine] for the given model format.
     *
     * @param context Android context required by engine implementations for SDK
     *   initialization. Nullable to support JVM unit testing — engines defer
     *   context usage to [VLMEngine.loadModel] time, not construction.
     * @param modelFormat The model format string from the model registry
     *   (e.g., "litert-lm", "onnx-vlm", "executorch").
     * @return A [VLMEngine] instance for the format, or null if the format
     *   is not recognized. Note: even a non-null return may fail at
     *   [VLMEngine.loadModel] time if the SDK is not available.
     */
    fun createEngine(context: Context?, modelFormat: String): VLMEngine? {
        return when (modelFormat) {
            "litert-lm" -> {
                Log.d(TAG, "Creating LiteRTLMEngine for format: $modelFormat")
                // LiteRTLMEngine requires non-null Context (unlike other engines that accept Context?).
                // Return null when context is null — consistent with createEngine's nullable return type.
                if (context != null) LiteRTLMEngine(context) else {
                    Log.w(TAG, "Cannot create LiteRTLMEngine: context is null")
                    null
                }
            }
            "onnx-vlm" -> {
                Log.d(TAG, "Creating OnnxVLMEngine for format: $modelFormat")
                OnnxVLMEngine(context)
            }
            "executorch" -> {
                Log.d(TAG, "Creating ExecuTorchEngine for format: $modelFormat")
                ExecuTorchEngine(context)
            }
            else -> {
                Log.w(TAG, "Unknown model format: $modelFormat. " +
                    "Supported formats: ${supportedFormats.joinToString()}")
                null
            }
        }
    }

    /**
     * Check if a model format is supported by this factory.
     *
     * @param modelFormat The model format string to check.
     * @return true if [createEngine] will return a non-null engine for this format.
     */
    fun isFormatSupported(modelFormat: String): Boolean {
        return modelFormat in supportedFormats
    }
}
