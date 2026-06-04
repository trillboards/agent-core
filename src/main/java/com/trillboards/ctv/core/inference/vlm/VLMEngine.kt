package com.trillboards.ctv.core.inference.vlm

import android.graphics.Bitmap
import com.trillboards.ctv.core.SensingConfig

/**
 * Abstraction over on-device Vision-Language Model runtimes.
 *
 * Implementations wrap vendor-specific SDKs (LiteRT-LM, MediaPipe LLM Inference,
 * ONNX GenAI, etc.) behind a uniform interface so that [VLMInferenceProcessor]
 * doesn't couple to any single runtime. This enables:
 *
 * 1. **Runtime swap without processor change** — switch from LiteRT-LM to
 *    MediaPipe LLM Inference by swapping the engine factory.
 * 2. **Testability** — unit tests inject a fake engine.
 * 3. **Graceful degradation** — if a vendor SDK isn't bundled in the APK,
 *    [loadModel] returns false and the processor falls back to last-known-good output.
 *
 * ## Thread Safety
 * [generate] is a suspend function and may be called from Dispatchers.Default.
 * Implementations must be internally synchronized or single-session-safe.
 *
 * ## Memory Lifecycle
 * Callers MUST call [unload] when the engine is no longer needed. Engines hold
 * native memory (model weights, KV cache) that is not tracked by the Kotlin GC.
 */
interface VLMEngine {

    /** Human-readable engine name for logging/telemetry, e.g. "LiteRT-LM", "MediaPipe-LLM". */
    val engineName: String

    /** True if a model is currently loaded and ready for inference. */
    val isLoaded: Boolean

    /**
     * Load a model from the filesystem into the inference runtime.
     *
     * @param modelPath Absolute path to the model binary on disk (e.g. from [ModelDownloadManager]).
     * @param config Engine configuration (max tokens, temperature, GPU, threads).
     * @return true if the model loaded successfully and [isLoaded] is now true.
     */
    fun loadModel(modelPath: String, config: VLMConfig = VLMConfig()): Boolean

    /**
     * Run VLM inference with an optional image.
     *
     * When [image] is non-null the engine builds a multi-modal prompt (text + image).
     * When null, the engine runs text-only inference.
     *
     * @param prompt The text prompt (typically a structured JSON-output instruction).
     * @param image Optional camera frame Bitmap for visual analysis.
     * @return [VLMResponse] containing the raw text output and timing metadata.
     */
    suspend fun generate(prompt: String, image: Bitmap? = null): VLMResponse

    /**
     * Unload the model and free all native resources (weights, KV cache, session).
     * After this call [isLoaded] must return false. Safe to call multiple times.
     */
    fun unload()

    /**
     * Current estimated memory usage of the loaded model in megabytes.
     * Returns 0 if no model is loaded.
     */
    fun getMemoryUsageMb(): Float
}

/**
 * Configuration for VLM inference.
 *
 * @param maxTokens Maximum number of tokens to generate per inference call.
 *   Larger values increase latency linearly. 512 is sufficient for structured
 *   JSON output with 10-15 fields.
 * @param temperature Sampling temperature. 0.0 = greedy (deterministic),
 *   1.0 = maximum randomness. Low values (0.1) preferred for structured JSON.
 * @param useGpu Whether to request GPU acceleration. Falls back to CPU
 *   if the device/runtime doesn't support GPU inference.
 * @param numThreads Number of CPU threads for inference. Only used when
 *   [useGpu] is false or GPU acceleration is unavailable.
 * @param systemPrompt Optional system-level prompt for the VLM engine.
 *   When set, engines use this instead of their default system instruction.
 *   Currently used by direct engine callers; VLMInferenceProcessor passes
 *   the server-generated metricsPrompt via buildPrompt() in the user message.
 * @param toolsJsonSchema Optional JSON Schema string for an [OpenApiTool] to inject
 *   into the [ConversationConfig]. When non-null the engine creates one tool named
 *   "EmitSpeechInsights" with this schema and passes [automaticToolCalling] to
 *   control whether the SDK invokes the tool automatically. Defaults preserve the
 *   existing [gemma_4_e2b] VLM path (schema=null, automaticToolCalling=true).
 * @param automaticToolCalling When [toolsJsonSchema] is non-null, controls whether
 *   the SDK auto-invokes the tool (true) or returns the raw tool-call JSON in the
 *   response text (false). For [OnDeviceLlmInsightExtractor] use false so the JSON
 *   is available for parsing. Has no effect when [toolsJsonSchema] is null.
 */
data class VLMConfig(
    val maxTokens: Int = SensingConfig.get().vlm.maxTokens,
    val temperature: Float = SensingConfig.get().vlm.temperature,
    val useGpu: Boolean = true,
    val numThreads: Int = SensingConfig.get().vlm.numThreads,
    val systemPrompt: String? = null,
    val toolsJsonSchema: String? = null,
    val automaticToolCalling: Boolean = true
)

/**
 * Response from VLM inference.
 *
 * @param text Raw text output from the model (may contain JSON, markdown fences, etc.).
 * @param latencyMs Wall-clock inference time in milliseconds.
 * @param tokensGenerated Number of output tokens produced (0 if not tracked by the runtime).
 * @param parseSuccess Whether [VLMResponseParser] was able to extract valid JSON fields
 *   from [text]. Set by the caller after parsing, not by the engine itself.
 * @param parsedFields Extracted key-value fields from the JSON response.
 *   Populated by the caller after parsing, not by the engine.
 */
data class VLMResponse(
    val text: String,
    val latencyMs: Long,
    val tokensGenerated: Int = 0,
    val parseSuccess: Boolean = false,
    val parsedFields: Map<String, Any> = emptyMap()
)
