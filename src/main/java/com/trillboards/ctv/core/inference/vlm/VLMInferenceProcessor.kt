package com.trillboards.ctv.core.inference.vlm

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.inference.HardwareRequirement
import com.trillboards.ctv.core.inference.InferenceInput
import com.trillboards.ctv.core.inference.InferenceOutput
import com.trillboards.ctv.core.inference.InferenceProcessor
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

/**
 * InferenceProcessor implementation that runs a Vision-Language Model on-device
 * to extract structured metrics from camera frames.
 *
 * Unlike traditional ML processors (face detection, pose estimation) that produce
 * fixed-schema output, VLMInferenceProcessor produces dynamic key-value fields
 * defined by a server-side metricsPrompt. This enables "Sense Anything" — the
 * server tells the device what to observe, and the VLM extracts those observations
 * from camera frames as structured JSON.
 *
 * ## Architecture
 * ```
 *   CameraX frame → VLMInferenceProcessor → VLMEngine → VLMResponseParser → InferenceOutput
 *                         ↑                      ↑
 *                    metricsPrompt           LiteRTLMEngine
 *                   (from server)           (or any VLMEngine)
 * ```
 *
 * ## Frame Sampling
 * VLMs are orders of magnitude slower than traditional ML models (seconds vs
 * milliseconds). To avoid blocking the camera pipeline:
 * - Frames are sampled at [samplingIntervalMs] intervals (default 5 seconds)
 * - Between samples, the processor returns [lastKnownGoodOutput]
 * - VLM inference has a hard [inferenceTimeoutMs] timeout (default 10 seconds)
 * - On timeout, the last-known-good output is returned (not null)
 *
 * ## Bitmap Lifecycle
 * The VLM processor allocates its own bitmap buffer via [vlmBitmap]. It does NOT
 * share the Bitmap used by face/pose processors — this prevents contention on
 * the CameraX analysis pipeline. The bitmap is copied from the input frame during
 * [process] to avoid holding a reference to the CameraX ImageProxy backing buffer.
 *
 * ## Memory Attenuation
 * VLMs are the largest memory consumers in the sensing pipeline (1-4 GB model weights
 * plus KV cache). The processor should be registered with MemoryAttenuationManager at
 * SHED_AT_HIGH tier — it will be released before face/pose/audio processors during
 * memory pressure events.
 *
 * @param context Android context for engine initialization.
 * @param modelId Unique processor ID, must match the model_registry entry.
 * @param metricsPrompt Server-generated prompt describing which metrics to extract.
 *   Generated from the metricsSchema definition and injected via SensingConfig.
 * @param modelPath Absolute path to the model binary on disk.
 * @param modelFormat Model format string from the model registry (e.g., "litert-lm",
 *   "onnx-vlm", "executorch"). Used by [VLMEngineFactory] to select the
 *   correct engine runtime. Defaults to "litert-lm" for backwards compatibility.
 * @param engineFactory Factory for creating the VLM engine. Default uses
 *   [VLMEngineFactory.createEngine] with [modelFormat] to select the appropriate
 *   engine. Override in tests to inject a fake engine.
 * @param vlmConfig Engine configuration (max tokens, temperature, GPU, threads).
 * @param samplingIntervalMs Minimum interval between VLM inference calls in milliseconds.
 * @param inferenceTimeoutMs Maximum time to wait for a single VLM inference call.
 */
class VLMInferenceProcessor(
    private val context: Context?,
    override val modelId: String = "gemma_3n_e2b",
    private val metricsPrompt: String,
    private val modelPath: String,
    private val modelFormat: String = "litert-lm",
    private val engineFactory: (() -> VLMEngine)? = null,
    private val vlmConfig: VLMConfig = VLMConfig(),
    @Volatile private var samplingIntervalMs: Long = DEFAULT_SAMPLING_INTERVAL_MS,
    @Volatile private var inferenceTimeoutMs: Long = DEFAULT_INFERENCE_TIMEOUT_MS,
    private val maxStalenessMs: Long = MAX_STALENESS_MS,
    private val elapsedRealtimeProvider: () -> Long = { SystemClock.elapsedRealtime() }
) : InferenceProcessor {

    /**
     * Snapshot of TFLite sensor readings at a point in time.
     * Stored in a ring buffer by AudienceSensingService and provided to the
     * VLM via [contextProvider] so the model can reason about trends.
     */
    data class TFLiteSnapshot(
        val timestampMs: Long,
        val faceCount: Float,
        val avgAttention: Float,
        val dominantEmotion: String,
        val dominantAmbience: String
    )

    companion object {
        private const val TAG = "VLMInferenceProcessor"

        /** Default frame sampling interval: 5 seconds between VLM inference calls. */
        val DEFAULT_SAMPLING_INTERVAL_MS: Long get() = SensingConfig.get().vlm.defaultSamplingIntervalMs

        /** Default inference timeout: 45s — Gemma 4 on premium tablets regularly lands in the 30s under load. */
        val DEFAULT_INFERENCE_TIMEOUT_MS: Long get() = SensingConfig.get().vlm.defaultInferenceTimeoutMs

        /** Estimated base memory for the processor infrastructure (buffers, parser state). */
        private val BASE_MEMORY_MB: Float get() = SensingConfig.get().vlm.baseMemoryMb

        /** Bitmap copy size for 512x512 ARGB_8888 = 1 MB. */
        private val VLM_BITMAP_SIZE: Int get() = SensingConfig.get().vlm.bitmapSize

        /**
         * Maximum age of a last-known-good output before it's considered stale.
         * Stale outputs are discarded rather than returned as fallback, preventing
         * the system from reporting minute-old observations as current state.
         */
        const val MAX_STALENESS_MS: Long = 60_000L

        // Phase 5.2 — token-budget guardrail constants. Char→token heuristic
        // matches server-side audienceVisionService.CHARS_PER_TOKEN. Hard
        // ceiling and warn threshold mirror server defaults so a single env
        // BONUS_FIELD_PROMPT_HARD_CEILING flip on the server changes the
        // server-applied trim threshold; the device defaults do not currently
        // read SSM, so they are constants here.
        private const val CHARS_PER_TOKEN = 4
        const val PROMPT_TOKEN_WARN_THRESHOLD = 4000
        const val PROMPT_TOKEN_HARD_CEILING = 5000

        /**
         * Approximate-tokens helper using the same chars/4 heuristic as the
         * Node.js server (see audienceVisionService.countTokensApprox and
         * sensingIntentService.countTokensApprox). Visible for tests.
         */
        fun countTokensApprox(prompt: String?): Int {
            if (prompt.isNullOrEmpty()) return 0
            // ceil(len / 4) — matches Math.ceil(prompt.length / 4) on the server.
            return (prompt.length + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
        }
    }

    override val hardwareRequirement = HardwareRequirement.CAMERA

    /** The VLM engine instance. Created during [initialize], null before/after. */
    private var engine: VLMEngine? = null

    /**
     * Own pre-allocated bitmap buffer for VLM input, separate from the shared
     * bitmap used by face/pose processors. Allocated during [initialize],
     * recycled during [release].
     */
    private var vlmBitmap: Bitmap? = null

    /** Mutex to serialize process() calls — prevents concurrent VLM inference. */
    private val inferenceMutex = Mutex()
    @Volatile private var releaseRequested: Boolean = false

    /** Timestamp (elapsedRealtime) of last successful VLM inference. */
    @Volatile
    private var lastProcessedMs: Long = 0L

    /**
     * Last successful inference output. Returned between sampling intervals
     * and during VLM timeouts. Null until the first successful inference.
     * Subject to staleness expiry — see [maxStalenessMs].
     */
    @Volatile
    private var lastKnownGoodOutput: InferenceOutput? = null

    /** SystemClock.elapsedRealtime() when lastKnownGoodOutput was set. */
    @Volatile
    private var lastKnownGoodTimestampMs: Long = 0L

    /**
     * External provider of temporal context from TFLite sensor readings.
     * When set, buildPrompt() injects recent sensor snapshots into the VLM
     * prompt so the model has temporal awareness (trend direction, velocity).
     * Wired by AudienceSensingService from its TFLite snapshot ring buffer.
     */
    var contextProvider: (() -> String)? = null

    /** Tracks initialization state. */
    @Volatile
    private var ready: Boolean = false

    /** Count of consecutive inference failures for health monitoring. */
    private var consecutiveFailures: Int = 0

    /** Maximum consecutive failures before logging a warning. */
    private val maxConsecutiveFailuresBeforeWarn: Int get() = SensingConfig.get().vlm.maxConsecutiveFailuresBeforeWarn

    /** Per-model circuit breaker — blocks inference after repeated failures. */
    private val circuitBreaker = ModelCircuitBreaker(modelId = modelId)

    /** Count of consecutive inference failures specifically for GPU fallback logic. */
    private var consecutiveInferenceFailures: Int = 0

    /** Whether GPU→CPU fallback has already been attempted since last init/release. */
    private var gpuFallbackAttempted: Boolean = false

    /** Total successful inferences since initialization, for telemetry. */
    private var inferenceCount: Int = 0

    /** Total failed inferences since initialization, for telemetry. */
    private var errorCount: Int = 0

    override fun hasModel(): Boolean {
        val modelFile = File(modelPath)
        return modelFile.exists() && modelFile.length() > 0
    }

    override fun initialize(): Boolean {
        if (ready) {
            Log.d(TAG, "Already initialized, returning true")
            return true
        }

        if (!hasModel()) {
            Log.w(TAG, "Cannot initialize: model not found at $modelPath")
            return false
        }

        return try {
            // Create the VLM engine using the explicit modelFormat field.
            // Previously, the default engineFactory lambda captured modelFormat from the
            // constructor parameter scope, which could mismatch the actual field value
            // due to Kotlin default-parameter evaluation order.
            Log.i(TAG, "Creating VLM engine: modelFormat=$modelFormat, modelPath=$modelPath")
            val newEngine = if (engineFactory != null) {
                engineFactory.invoke()
            } else {
                VLMEngineFactory.createEngine(context, modelFormat)
                    ?: throw IllegalStateException(
                        "Unsupported model format: $modelFormat " +
                        "(context=${if (context != null) "present" else "null"}). " +
                        "Supported: ${VLMEngineFactory.supportedFormats.joinToString()}"
                    )
            }

            // Load the model
            val loaded = newEngine.loadModel(modelPath, vlmConfig)
            if (!loaded) {
                Log.e(TAG, "Engine failed to load model: $modelPath")
                return false
            }

            // Pre-allocate VLM bitmap buffer (512x512 ARGB_8888)
            vlmBitmap = Bitmap.createBitmap(
                VLM_BITMAP_SIZE,
                VLM_BITMAP_SIZE,
                Bitmap.Config.ARGB_8888
            )

            engine = newEngine
            ready = true
            consecutiveFailures = 0
            consecutiveInferenceFailures = 0
            gpuFallbackAttempted = false
            inferenceCount = 0
            errorCount = 0
            circuitBreaker.reset()
            lastProcessedMs = 0L
            lastKnownGoodOutput = null
            lastKnownGoodTimestampMs = 0L
            releaseRequested = false

            Log.i(TAG, "VLM processor initialized: modelId=$modelId, " +
                "engine=${newEngine.engineName}, modelPath=$modelPath, " +
                "sampling=${samplingIntervalMs}ms, timeout=${inferenceTimeoutMs}ms, " +
                "memoryEstimate=${getMemoryFootprintMb()}MB")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize VLM processor: ${e.message}", e)
            release()
            false
        }
    }

    override fun isReady(): Boolean = ready

    override suspend fun process(input: InferenceInput): InferenceOutput? {
        if (!ready || engine == null || releaseRequested) {
            return null
        }

        // Circuit breaker check — skip inference if breaker is OPEN
        if (!circuitBreaker.isAllowed) {
            Log.w(TAG, "Circuit breaker OPEN for $modelId — skipping inference")
            return getLastKnownGoodOutput()
        }

        val now = elapsedRealtimeProvider()

        // Frame sampling: skip if not enough time has elapsed since last inference
        if (now - lastProcessedMs < samplingIntervalMs) {
            return getLastKnownGoodOutput()
        }

        // For audio-only input, VLM has nothing to process
        if (input is InferenceInput.AudioBuffer) {
            return getLastKnownGoodOutput()
        }

        // Serialize inference calls — if another coroutine is already running inference,
        // return the last known good output instead of queueing up.
        if (!inferenceMutex.tryLock()) {
            Log.w(TAG, "VLM inference skipped for $modelId — inference mutex busy")
            return getLastKnownGoodOutput()
        }

        try {
            lastProcessedMs = now

            // Extract bitmap from the input (may be null for text-only MultiModal)
            val frame = extractBitmap(input)

            // Build the full prompt with metrics instruction
            val fullPrompt = buildPrompt(metricsPrompt)
            Log.d(TAG, "VLM inference start: modelId=$modelId, image=${frame != null}, promptChars=${fullPrompt.length}")

            // Run VLM inference with timeout
            val response = withTimeoutOrNull(inferenceTimeoutMs) {
                if (releaseRequested || !ready) {
                    null
                } else {
                    val activeEngine = engine ?: return@withTimeoutOrNull null
                    val vlmFrame = if (frame != null) copyToVlmBitmap(frame) else null
                    try {
                        activeEngine.generate(fullPrompt, vlmFrame)
                    } finally {
                        if (vlmFrame != null && vlmFrame !== vlmBitmap && !vlmFrame.isRecycled) {
                            vlmFrame.recycle()
                        }
                    }
                }
            }

            if (response != null && response.text.isNotBlank()) {
                // Parse the response into structured fields
                val parseResult = VLMResponseParser.parse(response.text)

                if (parseResult.success && parseResult.fields.isNotEmpty()) {
                    circuitBreaker.recordSuccess()
                    consecutiveFailures = 0
                    consecutiveInferenceFailures = 0
                    inferenceCount++

                    // Compute temporal agreement if contextProvider is wired
                    // (indicates TFLite snapshots are available for cross-validation)
                    val enrichedFields = if (contextProvider != null) {
                        // Agreement score is computed by the caller (AudienceSensingService)
                        // via computeTemporalAgreement() — here we just mark that temporal
                        // context was available so the server knows the prompt included it.
                        parseResult.fields + ("temporal_context_available" to true)
                    } else {
                        parseResult.fields
                    }

                    val output = InferenceOutput(
                        modelId = modelId,
                        fields = enrichedFields,
                        latencyMs = response.latencyMs,
                        confidence = calculateParseConfidence(parseResult)
                    )
                    lastKnownGoodOutput = output
                    lastKnownGoodTimestampMs = elapsedRealtimeProvider()

                    Log.d(TAG, "VLM inference success: latency=${response.latencyMs}ms, " +
                        "fields=${parseResult.fields.size}, strategy=${parseResult.strategy}, " +
                        "confidence=${output.confidence}")
                    return output
                } else {
                    // Parsed but failed — count as failure
                    circuitBreaker.recordFailure()
                    consecutiveFailures++
                    consecutiveInferenceFailures++
                    errorCount++
                    Log.w(TAG, "VLM response parse failed (strategy=${parseResult.strategy}), " +
                        "failure #$consecutiveFailures: ${response.text.take(200)}")
                    attemptGpuFallbackIfNeeded()
                    return getLastKnownGoodOutput()
                }
            } else {
                // Timeout or engine returned null / empty
                circuitBreaker.recordFailure()
                consecutiveFailures++
                consecutiveInferenceFailures++
                errorCount++
                if (consecutiveFailures <= maxConsecutiveFailuresBeforeWarn ||
                    consecutiveFailures % 10 == 0) {
                    Log.w(TAG, "VLM inference timeout/failure #$consecutiveFailures " +
                        "(timeout=${inferenceTimeoutMs}ms), returning last known good")
                }
                attemptGpuFallbackIfNeeded()
                return getLastKnownGoodOutput()
            }
        } finally {
            inferenceMutex.unlock()
        }
    }

    /**
     * After 3+ consecutive inference failures with GPU enabled, attempt CPU-only
     * re-initialization. This handles cases where the GPU runtime crashes but
     * the CPU path would still work (e.g., driver bugs, OOM on GPU memory).
     * Uses an idempotent flag to ensure fallback is attempted at most once per
     * initialize/release cycle.
     */
    private fun attemptGpuFallbackIfNeeded() {
        if (consecutiveInferenceFailures >= SensingConfig.get().vlm.gpuFallbackAfterFailures && vlmConfig.useGpu && !gpuFallbackAttempted) {
            gpuFallbackAttempted = true
            Log.w(TAG, "3+ consecutive failures with GPU — attempting CPU-only re-initialization")
            try {
                engine?.unload()
                val cpuConfig = vlmConfig.copy(useGpu = false)
                val reloaded = engine?.loadModel(modelPath, cpuConfig) ?: false
                if (reloaded) {
                    Log.i(TAG, "CPU-only re-initialization succeeded for $modelId")
                } else {
                    Log.e(TAG, "CPU-only re-initialization returned false for $modelId")
                }
            } catch (e: Exception) {
                Log.e(TAG, "CPU-only re-initialization failed: ${e.message}")
            }
        }
    }

    override fun release() {
        releaseRequested = true
        ready = false

        runBlocking {
            inferenceMutex.withLock {
                engine?.let { eng ->
                    try {
                        eng.unload()
                        Log.i(TAG, "VLM engine unloaded")
                    } catch (e: Exception) {
                        Log.e(TAG, "Error unloading VLM engine: ${e.message}", e)
                    }
                }
                engine = null

                vlmBitmap?.let { bmp ->
                    if (!bmp.isRecycled) {
                        bmp.recycle()
                    }
                }
                vlmBitmap = null
            }
        }

        lastKnownGoodOutput = null
        lastKnownGoodTimestampMs = 0L
        lastProcessedMs = 0L
        consecutiveFailures = 0
        consecutiveInferenceFailures = 0
        gpuFallbackAttempted = false
        inferenceCount = 0
        errorCount = 0
        circuitBreaker.reset()
        releaseRequested = false

        Log.i(TAG, "VLM processor released: modelId=$modelId")
    }

    override fun getMemoryFootprintMb(): Float {
        if (!ready) return 0f
        val engineMemory = engine?.getMemoryUsageMb() ?: 0f
        // Bitmap memory: 512*512*4 bytes = 1 MB
        val bitmapMemory = (VLM_BITMAP_SIZE * VLM_BITMAP_SIZE * 4f) / (1024f * 1024f)
        return BASE_MEMORY_MB + engineMemory + bitmapMemory
    }

    // --- Internal helpers ---

    /**
     * Extract a Bitmap from the inference input.
     * Supports CameraFrame (primary use case) and MultiModal inputs.
     *
     * Returns null when no image is available or the input contains
     * a null reference (e.g., in unit tests where Bitmap stubs return null).
     */
    private fun extractBitmap(input: InferenceInput): Bitmap? {
        return try {
            when (input) {
                is InferenceInput.CameraFrame -> input.bitmap
                is InferenceInput.MultiModal -> input.image
                is InferenceInput.AudioBuffer -> null
            }
        } catch (e: NullPointerException) {
            // CameraFrame.bitmap is non-null in Kotlin type system but may be
            // null at JVM level in unit tests with isReturnDefaultValues = true
            null
        }
    }

    /**
     * Copy the input frame to the VLM's own bitmap buffer.
     * This releases the reference to the CameraX-backed bitmap, allowing
     * the camera pipeline to reclaim its ImageProxy buffer.
     *
     * If the pre-allocated bitmap dimensions don't match, creates a scaled copy.
     */
    private fun copyToVlmBitmap(source: Bitmap): Bitmap {
        val target = vlmBitmap
        if (target != null && !target.isRecycled &&
            target.width == VLM_BITMAP_SIZE && target.height == VLM_BITMAP_SIZE) {
            // Scale source into pre-allocated buffer
            val canvas = android.graphics.Canvas(target)
            val srcRect = android.graphics.Rect(0, 0, source.width, source.height)
            val dstRect = android.graphics.Rect(0, 0, target.width, target.height)
            canvas.drawBitmap(source, srcRect, dstRect, null)
            return target
        }

        // Fallback: create a new scaled bitmap
        return Bitmap.createScaledBitmap(source, VLM_BITMAP_SIZE, VLM_BITMAP_SIZE, true)
    }

    /**
     * Build the complete VLM prompt from the server-provided metricsPrompt.
     *
     * The prompt instructs the model to analyze the camera frame and return
     * structured JSON with the fields defined in metricsPrompt. When a
     * [contextProvider] is wired, recent TFLite sensor readings are injected
     * so the model can reason about temporal trends (crowd building, winding
     * down, peak, idle).
     *
     * Phase 5.2 — token-budget guardrail. The server already enforces the hard
     * ceiling (drop-lowest-priority bonus fields) before sending the prompt;
     * this is a defensive logcat-warn so an oversized prompt at the edge can
     * be diagnosed via `adb logcat`. Edge cannot emit CloudWatch directly.
     */
    private fun buildPrompt(metricsPrompt: String): String {
        val temporalContext = contextProvider?.invoke() ?: ""
        val finalPrompt = """You are a visual analysis system for digital signage with temporal awareness.
${if (temporalContext.isNotEmpty()) "$temporalContext\n" else ""}Analyze the current camera frame and return ONLY a JSON object with these fields:
$metricsPrompt
Return ONLY valid JSON, no explanation."""

        // Phase 5.2 — token approximation (chars / 4, English heuristic). Mirrors
        // server-side audienceVisionService.countTokensApprox + sensingIntentService
        // so the same threshold values mean the same thing across the boundary.
        val approxTokens = countTokensApprox(finalPrompt)
        if (approxTokens > PROMPT_TOKEN_HARD_CEILING) {
            Log.w(TAG, "VLM prompt exceeds hard ceiling: tokens=$approxTokens ceiling=$PROMPT_TOKEN_HARD_CEILING chars=${finalPrompt.length}. " +
                "Server-side guardrail should have trimmed bonus fields before push — investigate sensing profile state.")
        } else if (approxTokens > PROMPT_TOKEN_WARN_THRESHOLD) {
            Log.w(TAG, "VLM prompt exceeds warn threshold: tokens=$approxTokens threshold=$PROMPT_TOKEN_WARN_THRESHOLD chars=${finalPrompt.length}.")
        }

        return finalPrompt
    }

    /**
     * Calculate parse confidence score based on how cleanly JSON was extracted.
     *
     * This measures P(parse correct | raw text), NOT P(observation correct | sensor data).
     * For semantic confidence based on multi-signal agreement, see
     * [com.trillboards.ctv.core.calibration.ConfidenceCalibrator].
     *
     * - Direct JSON parse: 0.95 (highest reliability)
     * - Code fence extraction: 0.90 (model wrapped JSON in markdown)
     * - Brace extraction: 0.80 (JSON found among other text)
     * - Regex fallback: 0.50 (partial extraction, low confidence)
     * - No parse: 0.0 (should not reach here — caller checks success)
     */
    private fun calculateParseConfidence(parseResult: VLMResponseParser.ParseResult): Float {
        val cfg = SensingConfig.get().vlm
        return when (parseResult.strategy) {
            VLMResponseParser.ParseStrategy.DIRECT_JSON -> cfg.confidenceDirectJson
            VLMResponseParser.ParseStrategy.CODE_FENCE -> cfg.confidenceCodeFence
            VLMResponseParser.ParseStrategy.BRACE_EXTRACTION -> cfg.confidenceBraceExtraction
            VLMResponseParser.ParseStrategy.REGEX_FALLBACK -> cfg.confidenceRegexFallback
            VLMResponseParser.ParseStrategy.NONE -> cfg.confidenceNone
        }
    }

    /**
     * Update the sampling interval at runtime (e.g., from dynamic config).
     * Clamps to a minimum of 1 second and maximum of 60 seconds.
     *
     * @param intervalMs New sampling interval in milliseconds.
     */
    fun updateSamplingInterval(intervalMs: Long) {
        val cfg = SensingConfig.get().vlm
        samplingIntervalMs = intervalMs.coerceIn(cfg.samplingIntervalMinMs, cfg.samplingIntervalMaxMs)
        Log.i(TAG, "Sampling interval updated to ${samplingIntervalMs}ms")
    }

    fun getSamplingIntervalMs(): Long = samplingIntervalMs

    fun getInferenceTimeoutMs(): Long = inferenceTimeoutMs

    /**
     * Update the inference timeout at runtime.
     * Clamps to a minimum of 2 seconds and maximum of 60 seconds.
     *
     * @param timeoutMs New inference timeout in milliseconds.
     */
    fun updateInferenceTimeout(timeoutMs: Long) {
        val cfg = SensingConfig.get().vlm
        inferenceTimeoutMs = timeoutMs.coerceIn(cfg.timeoutMinMs, cfg.timeoutMaxMs)
        Log.i(TAG, "Inference timeout updated to ${inferenceTimeoutMs}ms")
    }

    /**
     * Get the current count of consecutive inference failures.
     * Useful for health monitoring and adaptive sampling.
     */
    fun getConsecutiveFailures(): Int = consecutiveFailures

    /**
     * Get the last known good output without running inference.
     * Returns null if no successful inference has occurred yet, or if the
     * output has exceeded [maxStalenessMs] age. Stale outputs are discarded
     * to prevent reporting old observations as current state.
     */
    fun getLastKnownGoodOutput(): InferenceOutput? {
        val output = lastKnownGoodOutput ?: return null
        val age = elapsedRealtimeProvider() - lastKnownGoodTimestampMs
        return if (age <= maxStalenessMs) output else null
    }

    /**
     * Request that the next [process] call runs VLM inference immediately,
     * bypassing the sampling interval timer.
     *
     * Used by [PerceptionTrigger] to trigger VLM inference on meaningful
     * scene state changes detected by the perception plane (face detection,
     * object detection, audio classification), instead of waiting for
     * the fixed sampling interval to elapse.
     */
    fun requestImmediateInference() {
        lastProcessedMs = 0L
        Log.d(TAG, "Immediate inference requested — next process() will run VLM")
    }

    /**
     * Get model telemetry for heartbeat reporting.
     * Includes circuit breaker state, inference/error counts, and model metadata.
     * This is sent to the server via heartbeat metadata for fleet health monitoring.
     */
    fun getModelTelemetry(): Map<String, Any> = mapOf(
        "activeModel" to modelId,
        "modelFormat" to (engine?.engineName ?: "unknown"),
        "circuitBreakerState" to circuitBreaker.state.name,
        "inferenceCount" to inferenceCount,
        "errorCount" to errorCount
    )

    /**
     * Compute how well the VLM's qualitative event_phase label agrees with
     * the quantitative trend visible in recent TFLite snapshots.
     *
     * Returns a score in [0, 1]:
     *   0.8  = strong agreement (VLM says "building", TFLite confirms rising faces)
     *   0.5  = neutral / insufficient data
     *   0.3  = disagreement (VLM says "building" but faces are dropping)
     *
     * The score is attached to the VLM output fields as "temporal_agreement"
     * so the server can weight or discard VLM observations accordingly.
     */
    fun computeTemporalAgreement(
        vlmOutput: Map<String, Any>,
        snapshots: List<TFLiteSnapshot>
    ): Float {
        val cfg = SensingConfig.get().vlm
        if (snapshots.size < 2) return cfg.temporalAgreementBase

        val latest = snapshots.last()
        val oldest = snapshots.first()
        val faceDelta = latest.faceCount - oldest.faceCount
        var score = cfg.temporalAgreementBase

        when (vlmOutput["event_phase"] as? String) {
            "building" -> score += if (faceDelta > cfg.temporalFaceCountThreshold) cfg.temporalAgreementBoost else cfg.temporalAgreementPenalty
            "winding_down" -> score += if (faceDelta < -cfg.temporalFaceCountThreshold) cfg.temporalAgreementBoost else cfg.temporalAgreementPenalty
            "peak" -> score += if (latest.faceCount > cfg.temporalPeakFaceCount && kotlin.math.abs(faceDelta) < cfg.temporalPeakDeltaThreshold) cfg.temporalAgreementBoost else 0f
            "idle" -> score += if (latest.faceCount < cfg.temporalFaceCountThreshold) cfg.temporalAgreementBoost else cfg.temporalAgreementPenalty
        }

        return score.coerceIn(0f, 1f)
    }
}
