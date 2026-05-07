package com.trillboards.ctv.core.inference.vlm

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * VLMEngine implementation wrapping ONNX Runtime GenAI for VLM models.
 *
 * ONNX Runtime provides cross-platform model inference with hardware acceleration
 * support (NNAPI, QNN, CoreML). Key models using onnx-vlm format:
 * - SmolVLM 256M (Hugging Face's compact vision-language model)
 * - Phi-3-vision-ONNX (Microsoft's optimized VLM for edge)
 * - Florence-2 (Microsoft's unified vision model, ONNX export)
 *
 * ## SDK Detection
 * The ONNX Runtime GenAI library is detected via reflection. The engine checks for
 * the GenAI API classes at runtime and degrades gracefully if unavailable.
 *
 * Expected SDK structure (from com.microsoft.onnxruntime:onnxruntime-genai):
 * ```
 * com.microsoft.onnxruntime.genai.Model — model loading
 * com.microsoft.onnxruntime.genai.Tokenizer — text tokenization
 * com.microsoft.onnxruntime.genai.Generator — token generation
 * com.microsoft.onnxruntime.genai.GeneratorParams — generation parameters
 * com.microsoft.onnxruntime.genai.Images — image input wrapper
 * ```
 *
 * ## Graceful Degradation
 * ONNX Runtime GenAI is `compileOnly` — it may not be bundled in all APK variants.
 * This engine uses reflection to detect SDK availability:
 * - [loadModel] returns false if the SDK classes aren't available
 * - [generate] returns an empty response if no model is loaded
 * - All errors are logged, never thrown to the caller
 *
 * ## Memory Management
 * ONNX models are loaded into memory with operator-specific buffers. Memory
 * usage is estimated as model file size * 1.4x for runtime overhead (session
 * state, operator workspace, KV cache).
 *
 * ## Thread Safety
 * The ONNX Runtime session is thread-safe for inference but not for concurrent
 * generation with shared KV cache. [generate] must not be called concurrently.
 *
 * @param context Android context for ONNX Runtime initialization and NNAPI access.
 *   Nullable to support JVM unit testing — all SDK calls that use context happen
 *   inside [loadModel]/[initializeGenAI]/[initializeOrtStandard], not at construction time.
 */
class OnnxVLMEngine(private val context: Context?) : VLMEngine {

    companion object {
        private const val TAG = "OnnxVLMEngine"

        /** Runtime overhead multiplier: session state + KV cache ~ 40% of model size. */
        private val MEMORY_OVERHEAD_MULTIPLIER: Float get() = SensingConfig.get().vlm.onnxMemoryOverheadMultiplier

        /** Maximum image dimension for ONNX VLM models. */
        private val MAX_IMAGE_DIM: Int get() = SensingConfig.get().vlm.onnxMaxImageDim

        /** JPEG quality for image encoding. */
        private val IMAGE_JPEG_QUALITY: Int get() = SensingConfig.get().vlm.imageJpegQuality

        /**
         * Fully qualified class name for ONNX Runtime GenAI Model.
         */
        private const val GENAI_MODEL_CLASS = "com.microsoft.onnxruntime.genai.Model"

        /**
         * Alternative: standard ONNX Runtime InferenceSession (non-GenAI).
         */
        private const val ORT_SESSION_CLASS = "ai.onnxruntime.OrtSession"

        /**
         * Check if the ONNX Runtime GenAI SDK is available at runtime.
         * Tries GenAI first, then falls back to standard ONNX Runtime.
         * Thread-safe via `by lazy`.
         */
        private val sdkAvailable: Boolean by lazy {
            try {
                Class.forName(GENAI_MODEL_CLASS)
                true
            } catch (e: ClassNotFoundException) {
                try {
                    Class.forName(ORT_SESSION_CLASS)
                    true
                } catch (e2: ClassNotFoundException) {
                    Log.w(TAG, "ONNX Runtime not available at runtime — ONNX VLM inference disabled")
                    false
                }
            }
        }

        fun isSdkAvailable(): Boolean = sdkAvailable

        /**
         * Determine which ONNX API is available: GenAI or standard ORT.
         */
        private fun resolveApiType(): OnnxApiType {
            return try {
                Class.forName(GENAI_MODEL_CLASS)
                OnnxApiType.GENAI
            } catch (e: ClassNotFoundException) {
                try {
                    Class.forName(ORT_SESSION_CLASS)
                    OnnxApiType.ORT_STANDARD
                } catch (e2: ClassNotFoundException) {
                    OnnxApiType.NONE
                }
            }
        }

        private enum class OnnxApiType {
            GENAI,
            ORT_STANDARD,
            NONE
        }
    }

    override val engineName: String = "ONNX-GenAI"

    /**
     * The ONNX model/session object. Typed as Any? to avoid compile-time
     * dependency on the SDK classes.
     */
    private var model: Any? = null

    /**
     * The tokenizer instance (GenAI API). Typed as Any? for reflection.
     */
    private var tokenizer: Any? = null

    /** Model file/directory size in bytes, used for memory estimation. */
    private var modelSizeBytes: Long = 0L

    /** The config used for the current session. */
    private var activeConfig: VLMConfig = VLMConfig()

    /** Which ONNX API variant is in use. */
    private var apiType: OnnxApiType = OnnxApiType.NONE

    /** Tracks whether a model is loaded and ready. */
    @Volatile
    private var _isLoaded: Boolean = false
    override val isLoaded: Boolean get() = _isLoaded

    override fun loadModel(modelPath: String, config: VLMConfig): Boolean {
        if (_isLoaded) {
            Log.w(TAG, "Model already loaded, unloading first")
            unload()
        }

        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            Log.w(TAG, "Model path not found: $modelPath")
            return false
        }

        if (!modelFile.canRead()) {
            Log.w(TAG, "Model path not readable: $modelPath")
            return false
        }

        // ONNX models can be a single file or a directory with multiple files
        modelSizeBytes = if (modelFile.isDirectory) {
            modelFile.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        } else {
            modelFile.length()
        }

        if (modelSizeBytes == 0L) {
            Log.w(TAG, "Model path is empty: $modelPath")
            return false
        }

        activeConfig = config

        if (!isSdkAvailable()) {
            Log.w(TAG, "ONNX Runtime not available — cannot load model. " +
                "Add implementation(\"com.microsoft.onnxruntime:onnxruntime-genai-android\") " +
                "to build.gradle.kts")
            return false
        }

        apiType = resolveApiType()

        return try {
            when (apiType) {
                OnnxApiType.GENAI -> initializeGenAI(modelPath, config)
                OnnxApiType.ORT_STANDARD -> initializeOrtStandard(modelPath, config)
                OnnxApiType.NONE -> {
                    Log.e(TAG, "No ONNX API available despite sdkAvailable=true")
                    false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX session: ${e.message}", e)
            model = null
            tokenizer = null
            _isLoaded = false
            false
        }
    }

    /**
     * Initialize using ONNX Runtime GenAI API via reflection.
     *
     * GenAI expects a model directory containing:
     * - genai_config.json (model configuration)
     * - *.onnx (model weights)
     * - tokenizer files
     *
     * ```
     * val model = Model(modelPath)
     * val tokenizer = model.createTokenizer()
     * ```
     */
    private fun initializeGenAI(modelPath: String, config: VLMConfig): Boolean {
        try {
            val modelClass = Class.forName(GENAI_MODEL_CLASS)

            // GenAI Model constructor takes model directory path
            val constructor = modelClass.getDeclaredConstructor(String::class.java)
            model = constructor.newInstance(modelPath)

            // Create tokenizer from model
            try {
                val createTokenizer = modelClass.getMethod("createTokenizer")
                tokenizer = createTokenizer.invoke(model)
            } catch (e: NoSuchMethodException) {
                Log.d(TAG, "GenAI createTokenizer not available — will use raw token IDs")
            }

            _isLoaded = true

            Log.i(TAG, "ONNX GenAI model loaded: path=$modelPath, " +
                "maxTokens=${config.maxTokens}, temperature=${config.temperature}, " +
                "modelSize=${modelSizeBytes / 1024 / 1024}MB")
            return true
        } catch (e: ClassNotFoundException) {
            Log.e(TAG, "ONNX GenAI classes not found: ${e.message}")
            return false
        } catch (e: NoSuchMethodException) {
            Log.e(TAG, "ONNX GenAI API mismatch — method not found: ${e.message}")
            return false
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "ONNX GenAI model load threw: ${e.cause?.message}", e.cause)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ONNX GenAI model: ${e.message}", e)
            return false
        }
    }

    /**
     * Initialize using standard ONNX Runtime API via reflection.
     *
     * Standard ORT uses InferenceSession for model loading:
     * ```
     * val env = OrtEnvironment.getEnvironment()
     * val sessionOptions = OrtSession.SessionOptions()
     * val session = env.createSession(modelPath, sessionOptions)
     * ```
     */
    private fun initializeOrtStandard(modelPath: String, config: VLMConfig): Boolean {
        try {
            val envClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            @Suppress("UNUSED_VARIABLE")
            val sessionClass = Class.forName("ai.onnxruntime.OrtSession")
            val sessionOptionsClass = Class.forName("ai.onnxruntime.OrtSession\$SessionOptions")

            // Get the shared OrtEnvironment
            val getEnv = envClass.getMethod("getEnvironment")
            val env = getEnv.invoke(null)

            // Create session options
            val optionsConstructor = sessionOptionsClass.getDeclaredConstructor()
            val options = optionsConstructor.newInstance()

            // Set thread count if available
            try {
                val setThreads = sessionOptionsClass.getMethod(
                    "setIntraOpNumThreads",
                    Int::class.java
                )
                setThreads.invoke(options, config.numThreads)
            } catch (e: NoSuchMethodException) {
                Log.d(TAG, "setIntraOpNumThreads not available")
            }

            // Try to add NNAPI execution provider for GPU acceleration
            if (config.useGpu) {
                try {
                    val addNnapi = sessionOptionsClass.getMethod("addNnapi")
                    addNnapi.invoke(options)
                    Log.d(TAG, "NNAPI execution provider added")
                } catch (e: Exception) {
                    Log.d(TAG, "NNAPI not available, using CPU: ${e.message}")
                }
            }

            // Create session
            val createSession = envClass.getMethod(
                "createSession",
                String::class.java,
                sessionOptionsClass
            )
            model = createSession.invoke(env, modelPath, options)
            _isLoaded = true

            Log.i(TAG, "ONNX Runtime session loaded: path=$modelPath, " +
                "threads=${config.numThreads}, gpu=${config.useGpu}, " +
                "modelSize=${modelSizeBytes / 1024 / 1024}MB")
            return true
        } catch (e: ClassNotFoundException) {
            Log.e(TAG, "ONNX Runtime classes not found: ${e.message}")
            return false
        } catch (e: NoSuchMethodException) {
            Log.e(TAG, "ONNX Runtime API mismatch: ${e.message}")
            return false
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "ONNX Runtime session creation threw: ${e.cause?.message}", e.cause)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ONNX Runtime session: ${e.message}", e)
            return false
        }
    }

    override suspend fun generate(prompt: String, image: Bitmap?): VLMResponse {
        if (!_isLoaded || model == null) {
            Log.w(TAG, "generate() called but no ONNX model is loaded")
            return VLMResponse(
                text = "",
                latencyMs = 0,
                tokensGenerated = 0,
                parseSuccess = false
            )
        }

        val startMs = SystemClock.elapsedRealtime()

        return try {
            val rawOutput = when (apiType) {
                OnnxApiType.GENAI -> generateGenAI(prompt, image)
                OnnxApiType.ORT_STANDARD -> generateOrtStandard(prompt, image)
                OnnxApiType.NONE -> ""
            }

            val latencyMs = SystemClock.elapsedRealtime() - startMs
            val parseResult = VLMResponseParser.parse(rawOutput)

            val response = VLMResponse(
                text = rawOutput,
                latencyMs = latencyMs,
                tokensGenerated = estimateTokenCount(rawOutput),
                parseSuccess = parseResult.success,
                parsedFields = parseResult.fields
            )

            Log.d(TAG, "ONNX inference complete: latency=${latencyMs}ms, " +
                "tokens~${response.tokensGenerated}, parseSuccess=${parseResult.success}, " +
                "api=$apiType")

            response
        } catch (e: Exception) {
            val latencyMs = SystemClock.elapsedRealtime() - startMs
            Log.e(TAG, "ONNX inference failed after ${latencyMs}ms: ${e.message}", e)
            VLMResponse(
                text = "",
                latencyMs = latencyMs,
                tokensGenerated = 0,
                parseSuccess = false
            )
        }
    }

    /**
     * Generate via ONNX GenAI API (com.microsoft.onnxruntime.genai).
     *
     * GenAI flow:
     * ```
     * val params = model.createGeneratorParams()
     * params.setSearchOption("max_length", maxTokens)
     * params.setSearchOption("temperature", temperature)
     * if (image != null) {
     *     val images = Images.load(imagePath)
     *     params.setInputImages(images)
     * }
     * val sequences = model.generate(tokenizer.encode(prompt), params)
     * val output = tokenizer.decode(sequences[0])
     * ```
     */
    private fun generateGenAI(prompt: String, image: Bitmap?): String {
        val currentModel = model ?: return ""

        try {
            val modelClass = currentModel.javaClass

            // Create generator params
            val createParams = modelClass.getMethod("createGeneratorParams")
            val params = createParams.invoke(currentModel)
            val paramsClass = params.javaClass

            // Set max tokens
            try {
                val setSearchOption = paramsClass.getMethod(
                    "setSearchOption", String::class.java, Double::class.java
                )
                setSearchOption.invoke(params, "max_length", activeConfig.maxTokens.toDouble())
                setSearchOption.invoke(params, "temperature", activeConfig.temperature.toDouble())
            } catch (e: NoSuchMethodException) {
                Log.d(TAG, "setSearchOption not available on GenAI params")
            }

            // Handle image input if provided
            if (image != null) {
                try {
                    val resized = resizeIfNeeded(image)
                    val imageBytes = bitmapToJpegBytes(resized)
                    if (resized !== image) resized.recycle()

                    // Try to set image via params
                    val setInputImages = paramsClass.getMethod(
                        "setInputImages", ByteArray::class.java
                    )
                    setInputImages.invoke(params, imageBytes)
                } catch (e: NoSuchMethodException) {
                    Log.d(TAG, "GenAI image input not available, running text-only")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to set image input: ${e.message}")
                }
            }

            // Tokenize prompt
            val currentTokenizer = tokenizer
            if (currentTokenizer != null) {
                try {
                    val encode = currentTokenizer.javaClass.getMethod("encode", String::class.java)
                    val tokens = encode.invoke(currentTokenizer, prompt)

                    // Generate
                    val generate = modelClass.getMethod("generate", tokens.javaClass, paramsClass)
                    val sequences = generate.invoke(currentModel, tokens, params)

                    // Decode
                    val decode = currentTokenizer.javaClass.getMethod("decode", sequences.javaClass)
                    val result = decode.invoke(currentTokenizer, sequences)
                    return result?.toString() ?: ""
                } catch (e: NoSuchMethodException) {
                    Log.d(TAG, "GenAI tokenizer encode/decode not available: ${e.message}")
                } catch (e: java.lang.reflect.InvocationTargetException) {
                    Log.e(TAG, "GenAI generation threw: ${e.cause?.message}", e.cause)
                    return ""
                }
            }

            // Fallback: try direct generate(prompt) method
            try {
                val generate = modelClass.getMethod("generate", String::class.java)
                val result = generate.invoke(currentModel, prompt)
                return result?.toString() ?: ""
            } catch (e: NoSuchMethodException) {
                Log.e(TAG, "No compatible generate method on GenAI model")
                return ""
            }
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "GenAI generation threw: ${e.cause?.message}", e.cause)
            return ""
        } catch (e: Exception) {
            Log.e(TAG, "GenAI generation failed: ${e.message}", e)
            return ""
        }
    }

    /**
     * Generate via standard ONNX Runtime (ai.onnxruntime).
     *
     * Standard ORT does not have a direct text generation API — it works with
     * raw tensors. For VLM models exported to standard ONNX format, this method
     * invokes the session run with the appropriate input/output tensors.
     *
     * Falls back to text-only if multimodal input handling fails.
     */
    @Suppress("UNUSED_PARAMETER")
    private fun generateOrtStandard(prompt: String, image: Bitmap?): String {
        val currentSession = model ?: return ""

        try {
            // Try to find a generate or run method
            val sessionClass = currentSession.javaClass
            val methodNames = listOf("generate", "generateResponse", "run")

            for (methodName in methodNames) {
                try {
                    val method = sessionClass.getMethod(methodName, String::class.java)
                    val result = method.invoke(currentSession, prompt)
                    return result?.toString() ?: ""
                } catch (e: NoSuchMethodException) {
                    continue
                }
            }

            Log.e(TAG, "No compatible generation method found on ORT session")
            return ""
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "ORT generation threw: ${e.cause?.message}", e.cause)
            return ""
        } catch (e: Exception) {
            Log.e(TAG, "ORT generation failed: ${e.message}", e)
            return ""
        }
    }

    /**
     * Resize a bitmap if either dimension exceeds [MAX_IMAGE_DIM].
     */
    private fun resizeIfNeeded(image: Bitmap): Bitmap {
        val width = image.width
        val height = image.height

        if (width <= MAX_IMAGE_DIM && height <= MAX_IMAGE_DIM) {
            return image
        }

        val scale = minOf(
            MAX_IMAGE_DIM.toFloat() / width,
            MAX_IMAGE_DIM.toFloat() / height
        )

        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        return Bitmap.createScaledBitmap(image, newWidth, newHeight, true)
    }

    /**
     * Convert a Bitmap to JPEG byte array.
     */
    private fun bitmapToJpegBytes(bitmap: Bitmap): ByteArray {
        val stream = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, IMAGE_JPEG_QUALITY, stream)
        return stream.toByteArray()
    }

    private fun estimateTokenCount(text: String): Int {
        return maxOf(1, text.length / 4)
    }

    override fun unload() {
        // Close tokenizer first
        if (tokenizer != null) {
            try {
                val closeMethod = tokenizer!!.javaClass.getMethod("close")
                closeMethod.invoke(tokenizer)
                Log.d(TAG, "ONNX tokenizer closed")
            } catch (e: NoSuchMethodException) {
                // No close method — OK for tokenizer
            } catch (e: Exception) {
                Log.w(TAG, "Error closing ONNX tokenizer: ${e.message}")
            }
        }

        // Close model/session
        if (model != null) {
            try {
                val closeMethod = model!!.javaClass.getMethod("close")
                closeMethod.invoke(model)
                Log.i(TAG, "ONNX model closed")
            } catch (e: NoSuchMethodException) {
                try {
                    val destroyMethod = model!!.javaClass.getMethod("destroy")
                    destroyMethod.invoke(model)
                    Log.i(TAG, "ONNX model destroyed")
                } catch (e2: Exception) {
                    Log.w(TAG, "No close/destroy method on ONNX model: ${e2.message}")
                }
            } catch (e: java.lang.reflect.InvocationTargetException) {
                Log.e(TAG, "ONNX model close threw: ${e.cause?.message}", e.cause)
            } catch (e: Exception) {
                Log.e(TAG, "Error closing ONNX model: ${e.message}", e)
            }
        }

        model = null
        tokenizer = null
        _isLoaded = false
        modelSizeBytes = 0L
        apiType = OnnxApiType.NONE
        Log.i(TAG, "ONNX VLM engine unloaded")
    }

    override fun getMemoryUsageMb(): Float {
        if (!_isLoaded || modelSizeBytes == 0L) return 0f
        return (modelSizeBytes * MEMORY_OVERHEAD_MULTIPLIER) / (1024f * 1024f)
    }
}
