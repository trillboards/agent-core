package com.trillboards.ctv.core.inference.vlm

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import java.io.File

/**
 * VLMEngine implementation wrapping Meta's ExecuTorch framework for edge VLM inference.
 *
 * ExecuTorch is Meta's inference runtime optimized for mobile/edge deployment of
 * PyTorch models. Key models targeting this format:
 * - SmolVLM (Hugging Face, via ExecuTorch export)
 * - Llama 3.2 Vision (Meta's multimodal Llama)
 * - Llama 3.2 1B/3B (text-only variants with vision adapters)
 *
 * ## SDK Detection
 * The ExecuTorch Android library is loaded via reflection. The engine checks for
 * the Java bindings at runtime and degrades gracefully if unavailable.
 *
 * Expected SDK structure (from org.pytorch:executorch-android):
 * ```
 * org.pytorch.executorch.Module — model loading and execution
 * org.pytorch.executorch.EValue — input/output tensor wrapper
 * org.pytorch.executorch.Tensor — tensor data
 * ```
 *
 * Alternative SDK structure (from com.facebook.executorch):
 * ```
 * com.facebook.executorch.LlamaModule — LLM-specific module
 * com.facebook.executorch.LlamaCallback — streaming token callback
 * ```
 *
 * ## Graceful Degradation
 * ExecuTorch is `compileOnly` — it may not be bundled in all APK variants.
 * This engine uses reflection to detect SDK availability:
 * - [loadModel] returns false if the native library or Java classes aren't available
 * - [generate] returns an empty response if no model is loaded
 * - All errors are logged, never thrown to the caller
 *
 * ## Memory Management
 * ExecuTorch models use a dedicated memory allocator (MemoryAllocator) with
 * configurable planned memory. Memory is estimated as model file size * 1.6x
 * to account for KV cache, attention buffers, and ExecuTorch runtime overhead.
 *
 * ## Thread Safety
 * ExecuTorch Module is single-threaded for inference. The caller
 * ([VLMInferenceProcessor]) enforces serialization via coroutine confinement.
 *
 * @param context Android context for native library loading and asset access.
 *   Nullable to support JVM unit testing — all SDK calls that use context happen
 *   inside [loadModel]/[initializeModule], not at construction time.
 */
class ExecuTorchEngine(private val context: Context?) : VLMEngine {

    companion object {
        private const val TAG = "ExecuTorchEngine"

        /** Runtime overhead multiplier: KV cache + ExecuTorch runtime ~ 60% of model size. */
        private val MEMORY_OVERHEAD_MULTIPLIER: Float get() = SensingConfig.get().vlm.execuTorchMemoryOverheadMultiplier

        /** Maximum image dimension for ExecuTorch VLM models. */
        private val MAX_IMAGE_DIM: Int get() = SensingConfig.get().vlm.execuTorchMaxImageDim

        /** JPEG quality for image encoding when passing to multimodal prompt. */
        private val IMAGE_JPEG_QUALITY: Int get() = SensingConfig.get().vlm.imageJpegQuality

        /**
         * ExecuTorch Module class — general-purpose model execution.
         */
        private const val ET_MODULE_CLASS = "org.pytorch.executorch.Module"

        /**
         * ExecuTorch LlamaModule — LLM/VLM-specific module with generate API.
         */
        private const val ET_LLAMA_MODULE_CLASS = "com.facebook.executorch.LlamaModule"

        /**
         * Alternative ExecuTorch module class (newer SDK versions).
         */
        private const val ET_ALT_MODULE_CLASS = "org.executorch.Module"

        /**
         * Check if any ExecuTorch SDK variant is available at runtime.
         * Thread-safe via `by lazy`.
         */
        private val sdkAvailable: Boolean by lazy {
            resolveModuleClass() != null
        }

        fun isSdkAvailable(): Boolean = sdkAvailable

        /**
         * Resolve the ExecuTorch module class, trying all known variants.
         */
        private fun resolveModuleClass(): Class<*>? {
            val classNames = listOf(
                ET_LLAMA_MODULE_CLASS,  // Prefer LlamaModule for LLM/VLM use case
                ET_MODULE_CLASS,
                ET_ALT_MODULE_CLASS
            )

            for (className in classNames) {
                try {
                    return Class.forName(className)
                } catch (e: ClassNotFoundException) {
                    continue
                }
            }

            Log.w(TAG, "ExecuTorch SDK not available at runtime — ExecuTorch inference disabled")
            return null
        }
    }

    override val engineName: String = "ExecuTorch"

    /**
     * The ExecuTorch module instance. Typed as Any? to avoid compile-time
     * dependency on the SDK classes.
     */
    private var module: Any? = null

    /**
     * Which module class was loaded. Used to dispatch generate calls correctly.
     */
    private var moduleClassName: String = ""

    /** Model file size in bytes, used for memory estimation. */
    private var modelSizeBytes: Long = 0L

    /** The config used for the current session. */
    private var activeConfig: VLMConfig = VLMConfig()

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
            Log.w(TAG, "Model file not found: $modelPath")
            return false
        }

        if (!modelFile.canRead()) {
            Log.w(TAG, "Model file not readable: $modelPath")
            return false
        }

        modelSizeBytes = modelFile.length()
        if (modelSizeBytes == 0L) {
            Log.w(TAG, "Model file is empty: $modelPath")
            return false
        }

        activeConfig = config

        if (!isSdkAvailable()) {
            Log.w(TAG, "ExecuTorch SDK not available — cannot load model. " +
                "Add ExecuTorch Android dependency to build.gradle.kts")
            return false
        }

        return try {
            initializeModule(modelPath, config)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ExecuTorch module: ${e.message}", e)
            module = null
            _isLoaded = false
            false
        }
    }

    /**
     * Initialize the ExecuTorch module via reflection.
     *
     * Tries LlamaModule first (has generate API), then general Module.
     *
     * LlamaModule:
     * ```
     * val module = LlamaModule(modelPath, tokenizerPath, temperature)
     * ```
     *
     * General Module:
     * ```
     * val module = Module.load(modelPath)
     * ```
     */
    private fun initializeModule(modelPath: String, config: VLMConfig): Boolean {
        // Try LlamaModule first — has built-in generate/tokenize
        try {
            val llamaModuleClass = Class.forName(ET_LLAMA_MODULE_CLASS)

            // LlamaModule(modelPath: String, tokenizerPath: String, temperature: Float)
            // The tokenizer is typically co-located with the model
            val tokenizerPath = resolveTokenizerPath(modelPath)

            val constructor = llamaModuleClass.getDeclaredConstructor(
                String::class.java,
                String::class.java,
                Float::class.java
            )
            module = constructor.newInstance(modelPath, tokenizerPath, config.temperature)
            moduleClassName = ET_LLAMA_MODULE_CLASS
            _isLoaded = true

            Log.i(TAG, "ExecuTorch LlamaModule loaded: model=$modelPath, " +
                "tokenizer=$tokenizerPath, temperature=${config.temperature}, " +
                "modelSize=${modelSizeBytes / 1024 / 1024}MB")
            return true
        } catch (e: ClassNotFoundException) {
            Log.d(TAG, "LlamaModule not available, trying general Module")
        } catch (e: NoSuchMethodException) {
            Log.d(TAG, "LlamaModule constructor mismatch, trying general Module")
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "LlamaModule init threw: ${e.cause?.message}", e.cause)
            // Don't return — try general Module
        } catch (e: Exception) {
            Log.d(TAG, "LlamaModule init failed: ${e.message}, trying general Module")
        }

        // Try general ExecuTorch Module
        val moduleClassNames = listOf(ET_MODULE_CLASS, ET_ALT_MODULE_CLASS)
        for (className in moduleClassNames) {
            try {
                val moduleClass = Class.forName(className)

                // Try Module.load(modelPath) static method
                try {
                    val loadMethod = moduleClass.getMethod("load", String::class.java)
                    module = loadMethod.invoke(null, modelPath)
                    moduleClassName = className
                    _isLoaded = true

                    Log.i(TAG, "ExecuTorch Module loaded via static load: model=$modelPath, " +
                        "class=$className, modelSize=${modelSizeBytes / 1024 / 1024}MB")
                    return true
                } catch (e: NoSuchMethodException) {
                    // Try constructor instead
                }

                // Try Module(modelPath) constructor
                try {
                    val constructor = moduleClass.getDeclaredConstructor(String::class.java)
                    module = constructor.newInstance(modelPath)
                    moduleClassName = className
                    _isLoaded = true

                    Log.i(TAG, "ExecuTorch Module loaded via constructor: model=$modelPath, " +
                        "class=$className, modelSize=${modelSizeBytes / 1024 / 1024}MB")
                    return true
                } catch (e: NoSuchMethodException) {
                    continue
                }
            } catch (e: ClassNotFoundException) {
                continue
            } catch (e: java.lang.reflect.InvocationTargetException) {
                Log.e(TAG, "ExecuTorch Module init threw ($className): ${e.cause?.message}", e.cause)
            } catch (e: Exception) {
                Log.e(TAG, "ExecuTorch Module init failed ($className): ${e.message}", e)
            }
        }

        Log.e(TAG, "No compatible ExecuTorch module class could load the model")
        return false
    }

    /**
     * Resolve the tokenizer path for LlamaModule.
     * Looks for common tokenizer filenames in the same directory as the model.
     */
    private fun resolveTokenizerPath(modelPath: String): String {
        val modelDir = File(modelPath).parentFile ?: return modelPath
        val tokenizerNames = listOf(
            "tokenizer.bin",
            "tokenizer.model",
            "tokenizer.json",
            "sentencepiece.bpe.model"
        )

        for (name in tokenizerNames) {
            val tokenizerFile = File(modelDir, name)
            if (tokenizerFile.exists()) {
                return tokenizerFile.absolutePath
            }
        }

        // Fallback: return model path (some LlamaModule versions handle tokenizer internally)
        return modelPath
    }

    override suspend fun generate(prompt: String, image: Bitmap?): VLMResponse {
        if (!_isLoaded || module == null) {
            Log.w(TAG, "generate() called but no ExecuTorch model is loaded")
            return VLMResponse(
                text = "",
                latencyMs = 0,
                tokensGenerated = 0,
                parseSuccess = false
            )
        }

        val startMs = SystemClock.elapsedRealtime()

        return try {
            val rawOutput = if (moduleClassName == ET_LLAMA_MODULE_CLASS) {
                generateLlama(prompt, image)
            } else {
                generateGeneral(prompt, image)
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

            Log.d(TAG, "ExecuTorch inference complete: latency=${latencyMs}ms, " +
                "tokens~${response.tokensGenerated}, parseSuccess=${parseResult.success}, " +
                "moduleType=$moduleClassName")

            response
        } catch (e: Exception) {
            val latencyMs = SystemClock.elapsedRealtime() - startMs
            Log.e(TAG, "ExecuTorch inference failed after ${latencyMs}ms: ${e.message}", e)
            VLMResponse(
                text = "",
                latencyMs = latencyMs,
                tokensGenerated = 0,
                parseSuccess = false
            )
        }
    }

    /**
     * Generate via LlamaModule's generate API.
     *
     * LlamaModule provides a high-level generate method:
     * ```
     * module.generate(prompt, maxTokens) -> String
     * // or with streaming callback:
     * module.generate(prompt, maxTokens) { token -> ... }
     * ```
     */
    private fun generateLlama(prompt: String, image: Bitmap?): String {
        val currentModule = module ?: return ""

        // If image is provided, try multimodal prompt construction
        val fullPrompt = if (image != null) {
            buildMultiModalPrompt(prompt, image)
        } else {
            prompt
        }

        val moduleClass = currentModule.javaClass

        // Try generate(String, Int) — returns full text
        try {
            val generateMethod = moduleClass.getMethod(
                "generate",
                String::class.java,
                Int::class.java
            )
            val result = generateMethod.invoke(currentModule, fullPrompt, activeConfig.maxTokens)
            return result?.toString() ?: ""
        } catch (e: NoSuchMethodException) {
            // Try generate(String) without maxTokens
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "LlamaModule generate threw: ${e.cause?.message}", e.cause)
            return ""
        }

        // Try generate(String)
        try {
            val generateMethod = moduleClass.getMethod("generate", String::class.java)
            val result = generateMethod.invoke(currentModule, fullPrompt)
            return result?.toString() ?: ""
        } catch (e: NoSuchMethodException) {
            Log.e(TAG, "No compatible generate method on LlamaModule")
            return ""
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "LlamaModule generate threw: ${e.cause?.message}", e.cause)
            return ""
        } catch (e: Exception) {
            Log.e(TAG, "LlamaModule generation failed: ${e.message}", e)
            return ""
        }
    }

    /**
     * Generate via general ExecuTorch Module.
     *
     * General Module uses a forward() method with EValue tensors:
     * ```
     * val inputs = arrayOf(EValue.from(tensor))
     * val outputs = module.forward(inputs)
     * ```
     *
     * For text generation, tries common method names.
     */
    private fun generateGeneral(prompt: String, image: Bitmap?): String {
        val currentModule = module ?: return ""
        val moduleClass = currentModule.javaClass

        val fullPrompt = if (image != null) {
            buildMultiModalPrompt(prompt, image)
        } else {
            prompt
        }

        // Try various generate methods
        val methodNames = listOf("generate", "generateResponse", "forward")

        for (methodName in methodNames) {
            try {
                val method = moduleClass.getMethod(methodName, String::class.java)
                val result = method.invoke(currentModule, fullPrompt)
                return result?.toString() ?: ""
            } catch (e: NoSuchMethodException) {
                continue
            } catch (e: java.lang.reflect.InvocationTargetException) {
                Log.e(TAG, "ExecuTorch $methodName threw: ${e.cause?.message}", e.cause)
                return ""
            } catch (e: Exception) {
                Log.e(TAG, "ExecuTorch $methodName failed: ${e.message}", e)
                return ""
            }
        }

        Log.e(TAG, "No compatible generation method found on ExecuTorch module")
        return ""
    }

    /**
     * Build a multimodal prompt for models that accept image descriptions.
     * ExecuTorch VLM models that support native image input use a separate
     * prefill step, but for reflection-based usage we encode as text reference.
     */
    private fun buildMultiModalPrompt(prompt: String, image: Bitmap): String {
        // For ExecuTorch models with native multimodal support, the image
        // would be passed as a tensor. Since we use reflection, we provide
        // context about the image presence in the prompt.
        return "$prompt\n[Image provided: ${image.width}x${image.height} frame for visual analysis]"
    }

    private fun estimateTokenCount(text: String): Int {
        return maxOf(1, text.length / 4)
    }

    override fun unload() {
        if (module != null) {
            // Try close/destroy/stop in order
            val cleanupMethods = listOf("close", "destroy", "stop")

            for (methodName in cleanupMethods) {
                try {
                    val method = module!!.javaClass.getMethod(methodName)
                    method.invoke(module)
                    Log.i(TAG, "ExecuTorch module $methodName() called")
                    break
                } catch (e: NoSuchMethodException) {
                    continue
                } catch (e: java.lang.reflect.InvocationTargetException) {
                    Log.e(TAG, "ExecuTorch $methodName threw: ${e.cause?.message}", e.cause)
                    break
                } catch (e: Exception) {
                    Log.w(TAG, "ExecuTorch $methodName failed: ${e.message}")
                    continue
                }
            }
        }

        module = null
        moduleClassName = ""
        _isLoaded = false
        modelSizeBytes = 0L
        Log.i(TAG, "ExecuTorch engine unloaded")
    }

    override fun getMemoryUsageMb(): Float {
        if (!_isLoaded || modelSizeBytes == 0L) return 0f
        return (modelSizeBytes * MEMORY_OVERHEAD_MULTIPLIER) / (1024f * 1024f)
    }
}
