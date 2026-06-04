package com.trillboards.ctv.core.inference.vlm

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.CancellationException
import java.io.File
import java.io.FileOutputStream

/**
 * VLMEngine implementation wrapping the real LiteRT-LM SDK
 * (`com.google.ai.edge.litertlm:litertlm-android:0.9.0`) via reflection.
 *
 * ## Why Reflection?
 * The LiteRT-LM SDK 0.9.0 is compiled with Kotlin 2.3 / Java 21 (class version 65.0),
 * but this project uses Kotlin 1.9.24 / Java 17. Direct imports would break kapt
 * (Room annotation processor) with "wrong class version" errors. The SDK is declared
 * as `runtimeOnly` in build.gradle.kts, so its classes + native libs (liblitertlm_jni.so)
 * are bundled in the APK but not visible to the compiler. This engine uses reflection to
 * invoke the real SDK classes at runtime.
 *
 * When this project upgrades to Kotlin 2.3+, replace reflection with direct imports:
 *   import com.google.ai.edge.litertlm.Engine
 *   import com.google.ai.edge.litertlm.EngineConfig
 *   import com.google.ai.edge.litertlm.Conversation
 *   import com.google.ai.edge.litertlm.ConversationConfig
 *   import com.google.ai.edge.litertlm.Content
 *   import com.google.ai.edge.litertlm.Contents
 *   import com.google.ai.edge.litertlm.SamplerConfig
 *   import com.google.ai.edge.litertlm.Backend
 *   import com.google.ai.edge.litertlm.Message
 *
 * ## SDK API (reflected)
 * - [Engine] manages model loading and hardware acceleration
 * - [Conversation] handles stateful message exchange with the model
 * - [Contents.of] builds multi-modal prompts from text, images, and audio
 * - [Message.text] property contains the model's text response
 * - [Backend.CPU] / [Backend.GPU] select inference backend
 *
 * ## Memory Management
 * The engine tracks approximate memory usage based on the model file size and a
 * 1.5x multiplier for runtime overhead (KV cache, attention buffers). Both
 * Engine and Conversation implement AutoCloseable.
 *
 * ## Thread Safety
 * The LiteRT-LM conversation is single-threaded. [generate] must not be called
 * concurrently -- the caller ([VLMInferenceProcessor]) enforces this via frame
 * sampling and coroutine confinement.
 */
class LiteRTLMEngine(private val context: Context) : VLMEngine {

    companion object {
        private const val TAG = "LiteRTLMEngine"

        /** Runtime overhead multiplier: KV cache + attention buffers ~ 50% of model size. */
        private val MEMORY_OVERHEAD_MULTIPLIER: Float get() = SensingConfig.get().vlm.liteRtMemoryOverheadMultiplier

        /** Maximum image dimension before resize (LiteRT-LM multimodal input limit). */
        private val MAX_IMAGE_DIM: Int get() = SensingConfig.get().vlm.liteRtMaxImageDim

        /** JPEG quality for image encoding when saving temp file for multimodal prompt. */
        private val IMAGE_JPEG_QUALITY: Int get() = SensingConfig.get().vlm.imageJpegQuality

        /** Default system instruction used when VLMConfig.systemPrompt is null. */
        private const val DEFAULT_SYSTEM_PROMPT =
            "You are a visual analysis model for a digital signage system. " +
            "Analyze what you see and respond with structured JSON. " +
            "Focus on: audience demographics, engagement level, environmental context."

        // Real SDK class names for reflection
        private const val CLASS_ENGINE = "com.google.ai.edge.litertlm.Engine"
        private const val CLASS_ENGINE_CONFIG = "com.google.ai.edge.litertlm.EngineConfig"
        private const val CLASS_CONVERSATION = "com.google.ai.edge.litertlm.Conversation"
        private const val CLASS_CONVERSATION_CONFIG = "com.google.ai.edge.litertlm.ConversationConfig"
        private const val CLASS_SAMPLER_CONFIG = "com.google.ai.edge.litertlm.SamplerConfig"
        private const val CLASS_BACKEND = "com.google.ai.edge.litertlm.Backend"
        private const val CLASS_BACKEND_CPU = "com.google.ai.edge.litertlm.Backend\$CPU"
        private const val CLASS_BACKEND_GPU = "com.google.ai.edge.litertlm.Backend\$GPU"
        private const val CLASS_CONTENT = "com.google.ai.edge.litertlm.Content"
        private const val CLASS_CONTENT_TEXT = "com.google.ai.edge.litertlm.Content\$Text"
        private const val CLASS_CONTENT_IMAGE_FILE = "com.google.ai.edge.litertlm.Content\$ImageFile"
        private const val CLASS_CONTENTS = "com.google.ai.edge.litertlm.Contents"
        private const val CLASS_MESSAGE = "com.google.ai.edge.litertlm.Message"

        // OpenApiTool class name — available in LiteRT-LM v0.12.0 for function-calling
        private const val CLASS_OPEN_API_TOOL = "com.google.ai.edge.litertlm.OpenApiTool"

        /**
         * Check if the LiteRT-LM SDK is available at runtime.
         * With runtimeOnly dependency, the classes are in the APK classloader
         * but not visible at compile time. This check confirms they loaded.
         */
        private val sdkAvailable: Boolean by lazy {
            try {
                Class.forName(CLASS_ENGINE)
                Class.forName(CLASS_CONVERSATION)
                Log.i(TAG, "LiteRT-LM SDK detected at runtime (Engine + Conversation classes found)")
                true
            } catch (e: ClassNotFoundException) {
                Log.w(TAG, "LiteRT-LM SDK not available at runtime -- VLM inference disabled")
                false
            }
        }

        fun isSdkAvailable(): Boolean = sdkAvailable
    }

    override val engineName: String = "LiteRT-LM"

    /** The LiteRT-LM Engine instance (reflected). */
    private var engine: Any? = null

    /** The active Conversation session (reflected). */
    private var conversation: Any? = null

    /** Model file size in bytes, used for memory estimation. */
    private var modelSizeBytes: Long = 0L

    /** The config used for the current session. */
    private var activeConfig: VLMConfig = VLMConfig()

    /** Cached ConversationConfig for creating fresh conversations per inference. */
    private var cachedConvConfig: Any? = null

    /** Cached ConversationConfig class for reflection calls. */
    private var cachedConvConfigClass: Class<*>? = null

    /** Temporary directory for image files passed to multimodal prompts. */
    private val tempDir: File by lazy {
        File(context.cacheDir, "vlm_images").also { it.mkdirs() }
    }

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
            Log.w(TAG, "LiteRT-LM SDK not available -- cannot load model. " +
                "Ensure runtimeOnly(\"com.google.ai.edge.litertlm:litertlm-android:0.9.0\") " +
                "is in build.gradle.kts")
            return false
        }

        return try {
            initializeEngine(modelPath, config)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LiteRT-LM engine: ${e.message}", e)
            cleanupResources()
            false
        }
    }

    /**
     * Initialize the LiteRT-LM Engine and create a Conversation via reflection.
     *
     * Reflects the real SDK API:
     * ```kotlin
     * val backend = Backend.CPU() // or Backend.GPU()
     * val engineConfig = EngineConfig(
     *     modelPath = modelPath,
     *     backend = backend,
     *     visionBackend = Backend.GPU(),
     *     cacheDir = context.cacheDir.absolutePath
     * )
     * val engine = Engine(engineConfig)
     * engine.initialize()
     * val samplerConfig = SamplerConfig(temperature = 0.1f, topK = 40, topP = 0.95f)
     * val conversationConfig = ConversationConfig(
     *     systemInstruction = Contents.of(Content.Text("...")),
     *     samplerConfig = samplerConfig
     * )
     * val conversation = engine.createConversation(conversationConfig)
     * ```
     */
    private fun initializeEngine(modelPath: String, config: VLMConfig): Boolean {
        try {
            // Load SDK classes
            val engineClass = Class.forName(CLASS_ENGINE)
            val engineConfigClass = Class.forName(CLASS_ENGINE_CONFIG)
            val backendClass = Class.forName(CLASS_BACKEND)
            val backendCpuClass = Class.forName(CLASS_BACKEND_CPU)
            val backendGpuClass = Class.forName(CLASS_BACKEND_GPU)
            val conversationConfigClass = Class.forName(CLASS_CONVERSATION_CONFIG)
            val samplerConfigClass = Class.forName(CLASS_SAMPLER_CONFIG)
            val contentClass = Class.forName(CLASS_CONTENT)
            val contentTextClass = Class.forName(CLASS_CONTENT_TEXT)
            val contentsClass = Class.forName(CLASS_CONTENTS)

            // Create backend: try GPU first, fall back to CPU
            val backend = if (config.useGpu) {
                try {
                    backendGpuClass.getDeclaredConstructor().newInstance()
                } catch (e: Exception) {
                    Log.w(TAG, "GPU backend init failed, falling back to CPU: ${e.message}")
                    backendCpuClass.getDeclaredConstructor().newInstance()
                }
            } else {
                backendCpuClass.getDeclaredConstructor().newInstance()
            }

            // Create vision backend (GPU preferred for image processing).
            //
            // SKIP for text-only LLM paths. A non-null visionBackend makes the
            // litertlm runtime probe the model for TF_LITE_VISION_ENCODER, which
            // text-only models (FunctionGemma 270M, EmbeddingGemma 300M, etc.)
            // do not contain — the engine init then throws:
            //
            //   NOT_FOUND: TF_LITE_VISION_ENCODER not found in the model.
            //
            // Use config.toolsJsonSchema as the proxy: callers that pass a tool
            // schema are the OnDeviceLlmInsightExtractor (text-only) path; the
            // VLM path (gemma_4_e2b) leaves it null and gets the GPU vision
            // backend. If/when a VLM also wants tools, split into an explicit
            // VLMConfig.hasVisionBackend flag.
            val visionBackend: Any? = if (config.toolsJsonSchema != null) {
                Log.i(TAG, "Text-only LLM path (toolsJsonSchema set) — skipping visionBackend")
                null
            } else try {
                backendGpuClass.getDeclaredConstructor().newInstance()
            } catch (e: Exception) {
                Log.w(TAG, "GPU vision backend init failed, using CPU: ${e.message}")
                backendCpuClass.getDeclaredConstructor().newInstance()
            }

            // Build EngineConfig. Signature varies by litertlm-android SDK version:
            //   v0.12.0: 7-arg public ctor (String, Backend, Backend, Backend, Integer, Integer, String)
            //            = (modelPath, backend, visionBackend, audioBackend,
            //               maxNumTokens, maxNumImages, cacheDir)
            //            Verified 2026-05-29 via javap on
            //            litertlm-android-0.12.0.aar/classes.jar/EngineConfig.class.
            //   v0.11.x: 6-arg ctor — same fields minus maxNumImages.
            //   v0.9.x:  4-arg ctor — (modelPath, backend, visionBackend, cacheDir).
            //   pre-0.9: 2-arg ctor — (modelPath, backend).
            // Reflection cascades highest → lowest. We declare litertlm 0.12.0 in
            // build.gradle.kts (PR L1, #6352) so 7-arg should always win in prod;
            // the lower-arg fallbacks remain so an SDK downgrade doesn't crash on
            // boot. The previous code stopped at 6-arg and silently failed when
            // FunctionGemma 270M tried to initialize against the 0.12 SDK on a
            // live Tab S11 (2026-05-29 logcat: "Array contains no element matching
            // the predicate" → OnDeviceLlmInsightExtractor.loadModel failed → speech
            // pipeline fell through to regex/Gemini for every transcript).
            val cacheDir = File(context.cacheDir, "litertlm_cache").also { it.mkdirs() }
            val engineConfig = try {
                // 7-arg constructor for litertlm 0.12+
                engineConfigClass.constructors.first { it.parameterCount == 7 }
                    .newInstance(
                        modelPath,                  // modelPath: String
                        backend,                    // backend: Backend
                        visionBackend,              // visionBackend: Backend
                        null,                       // audioBackend: Backend? (not needed)
                        null,                       // maxNumTokens: Integer? (use default)
                        null,                       // maxNumImages: Integer? (use default)
                        cacheDir.absolutePath        // cacheDir: String?
                    )
            } catch (e: Exception) {
                Log.w(TAG, "7-arg EngineConfig failed, trying 6-arg: ${e.message}")
                try {
                    // 6-arg constructor for litertlm 0.11.x
                    engineConfigClass.constructors.first { it.parameterCount == 6 }
                        .newInstance(
                            modelPath,                  // modelPath: String
                            backend,                    // backend: Backend
                            visionBackend,              // visionBackend: Backend?
                            null,                       // audioBackend: Backend? (not needed)
                            null,                       // contextLength: Integer? (use default)
                            cacheDir.absolutePath        // cacheDir: String?
                        )
                } catch (e2: Exception) {
                    Log.w(TAG, "6-arg EngineConfig failed, trying 4-arg: ${e2.message}")
                    // Fallback: try (modelPath, backend, visionBackend, cacheDir)
                    try {
                        engineConfigClass.constructors.first { it.parameterCount == 4 }
                            .newInstance(modelPath, backend, visionBackend, cacheDir.absolutePath)
                    } catch (e3: Exception) {
                        Log.w(TAG, "4-arg also failed, trying 2-arg: ${e3.message}")
                        // Minimal: (modelPath, backend)
                        engineConfigClass.constructors.first { it.parameterCount == 2 }
                            .newInstance(modelPath, backend)
                    }
                }
            }

            // Create Engine and initialize
            val newEngine = engineClass.getDeclaredConstructor(engineConfigClass)
                .newInstance(engineConfig)
            engineClass.getMethod("initialize").invoke(newEngine)
            engine = newEngine

            // Create SamplerConfig. litertlm 0.12 ctor (verified 2026-05-29 via
            // javap on the cached .aar): (int topK, double topP, double temperature, int seed).
            // 4-arg primitive — NOT the prior assumed 3-arg with mixed types.
            // The pre-0.12 fallback (3-arg) stays for SDK downgrade safety.
            val samplerConfig = try {
                // 4-arg (int, double, double, int) for litertlm 0.12+
                val ctor = samplerConfigClass.constructors.first { it.parameterCount == 4 }
                ctor.newInstance(
                    SensingConfig.get().vlm.topK,                  // Int
                    SensingConfig.get().vlm.topP.toDouble(),       // Double (was Float)
                    config.temperature.toDouble(),                  // Double (was Float)
                    0                                               // Int seed (0 = SDK default)
                )
            } catch (e: Exception) {
                Log.w(TAG, "4-arg SamplerConfig failed, trying 3-arg: ${e.message}")
                try {
                    val ctor = samplerConfigClass.constructors.first { it.parameterCount == 3 }
                    ctor.newInstance(SensingConfig.get().vlm.topK, SensingConfig.get().vlm.topP, config.temperature)
                } catch (e2: Exception) {
                    Log.w(TAG, "3-arg SamplerConfig failed, trying default: ${e2.message}")
                    try {
                        samplerConfigClass.getDeclaredConstructor().newInstance()
                    } catch (e3: Exception) {
                        Log.w(TAG, "Default SamplerConfig also failed: ${e3.message}")
                        null
                    }
                }
            }

            // Create system instruction: Contents.of(Content.Text("..."))
            // Use configurable system prompt (from VLMConfig.systemPrompt) or the default
            val systemPromptText = config.systemPrompt ?: DEFAULT_SYSTEM_PROMPT
            val systemText = contentTextClass.getDeclaredConstructor(String::class.java)
                .newInstance(systemPromptText)
            // Use varargs-compatible invocation for Contents.of(Content...)
            val systemInstruction = try {
                val ofMethod = contentsClass.methods.first { m ->
                    m.name == "of" && m.parameterTypes.isNotEmpty()
                }
                val contentArray = java.lang.reflect.Array.newInstance(contentClass, 1)
                java.lang.reflect.Array.set(contentArray, 0, systemText)
                ofMethod.invoke(null, contentArray)
            } catch (e: Exception) {
                Log.w(TAG, "Could not create system instruction via Contents.of: ${e.message}")
                null
            }

            // Optionally build an OpenApiTool for constrained function-calling.
            // Used by OnDeviceLlmInsightExtractor (FunctionGemma 270M, automaticToolCalling=false).
            // When toolsJsonSchema is null the existing VLM path (gemma_4_e2b) is unaffected.
            val toolsList: List<Any>? = if (config.toolsJsonSchema != null) {
                buildOpenApiToolList(config.toolsJsonSchema)
            } else {
                null
            }
            val automaticToolCalling = if (config.toolsJsonSchema != null) {
                config.automaticToolCalling
            } else {
                true // default — preserve existing behaviour
            }

            // Create ConversationConfig
            val convConfig = if (systemInstruction != null) {
                try {
                    // Try full constructor: (systemInstruction, initialMessages, samplerConfig, tools, automaticToolCalling)
                    // Simpler: find a constructor that takes Contents + SamplerConfig
                    conversationConfigClass.constructors.firstOrNull { ctor ->
                        ctor.parameterCount >= 2
                    }?.let { ctor ->
                        // Use named-parameter-style reflection for Kotlin data class
                        // ConversationConfig(systemInstruction=..., samplerConfig=..., tools=..., automaticToolCalling=...)
                        conversationConfigClass.getDeclaredConstructor(
                            systemInstruction.javaClass,  // systemInstruction: Contents?
                            List::class.java,              // initialMessages: List<Message>?
                            samplerConfigClass,            // samplerConfig: SamplerConfig?
                            List::class.java,              // tools: List<Tool>?
                            Boolean::class.java            // automaticToolCalling: Boolean
                        ).newInstance(systemInstruction, null, samplerConfig, toolsList, automaticToolCalling)
                    } ?: run {
                        // Fallback: just SamplerConfig
                        conversationConfigClass.getDeclaredConstructor(samplerConfigClass)
                            .newInstance(samplerConfig)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Full ConversationConfig construction failed: ${e.message}")
                    try {
                        conversationConfigClass.getDeclaredConstructor().newInstance()
                    } catch (e2: Exception) {
                        Log.w(TAG, "Default ConversationConfig also failed: ${e2.message}")
                        null
                    }
                }
            } else {
                try {
                    conversationConfigClass.getDeclaredConstructor(samplerConfigClass)
                        .newInstance(samplerConfig)
                } catch (e: Exception) {
                    try {
                        conversationConfigClass.getDeclaredConstructor().newInstance()
                    } catch (e2: Exception) {
                        null
                    }
                }
            }

            // Cache conversation config + class for createFreshConversation()
            cachedConvConfig = convConfig
            cachedConvConfigClass = conversationConfigClass

            // Create initial conversation
            conversation = createFreshConversation()

            _isLoaded = true

            Log.i(TAG, "LiteRT-LM engine initialized via reflection: model=$modelPath, " +
                "backend=${if (config.useGpu) "GPU" else "CPU"}, " +
                "temperature=${config.temperature}, " +
                "modelSize=${modelSizeBytes / 1024 / 1024}MB")
            return true
        } catch (e: ClassNotFoundException) {
            Log.e(TAG, "LiteRT-LM SDK class not found: ${e.message}")
            return false
        } catch (e: NoSuchMethodException) {
            Log.e(TAG, "LiteRT-LM API mismatch -- method not found: ${e.message}", e)
            return false
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "LiteRT-LM engine init threw: ${e.cause?.message}", e.cause)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create LiteRT-LM engine: ${e.message}", e)
            return false
        }
    }

    /**
     * Build a single-element [List] wrapping an [OpenApiTool] instance, constructed
     * via reflection from [toolsJsonSchema].
     *
     * The [OpenApiTool] is injected as the 4th positional arg to [ConversationConfig]
     * (the `tools: List<Tool>?` parameter). When [config.automaticToolCalling] is
     * false the SDK returns the tool-call JSON as the raw response text, which
     * [OnDeviceLlmInsightExtractor] then parses.
     *
     * Falls back to null (no tools) if the [OpenApiTool] class is not found —
     * e.g. on an older LiteRT-LM SDK version — and logs a warning.
     *
     * @param schemaJson JSON Schema string for the EmitSpeechInsights tool parameters.
     * @return A [List] containing one [OpenApiTool], or null on reflection failure.
     */
    private fun buildOpenApiToolList(schemaJson: String): List<Any>? {
        return try {
            val toolClass = Class.forName(CLASS_OPEN_API_TOOL)
            // litertlm 0.12: OpenApiTool is an INTERFACE (verified 2026-05-29
            // via javap on the cached .aar):
            //
            //   public interface com.google.ai.edge.litertlm.OpenApiTool {
            //     public abstract String getToolDescriptionJsonString();
            //     public abstract String execute(String);
            //   }
            //
            // The previous reflection used getDeclaredConstructor(String, String, String)
            // which threw NoSuchMethodException because interfaces have no constructors.
            // Build a Proxy that implements the interface. The SDK calls
            // getToolDescriptionJsonString() to build the prompt grammar; execute()
            // is called only if automaticToolCalling=true (we set false in
            // OnDeviceLlmInsightExtractor so the raw tool-call JSON arrives in the
            // response text for our own parser).
            val descriptionJson = """
                {
                  "name": "EmitSpeechInsights",
                  "description": "Emit structured speech insights extracted from a retail conversation transcript.",
                  "parameters": $schemaJson
                }
            """.trimIndent()
            val proxy = java.lang.reflect.Proxy.newProxyInstance(
                toolClass.classLoader,
                arrayOf(toolClass)
            ) { _, method, args ->
                when (method.name) {
                    "getToolDescriptionJsonString" -> descriptionJson
                    // execute(String) — return empty result string; SDK will fold
                    // this back into the model context if automaticToolCalling=true.
                    // OnDeviceLlmInsightExtractor sets automaticToolCalling=false so
                    // execute() is never invoked in our flow; keep the impl safe.
                    "execute" -> ""
                    // Object methods (toString/hashCode/equals) — sensible defaults.
                    "toString" -> "OpenApiTool(EmitSpeechInsights)"
                    "hashCode" -> System.identityHashCode(args)
                    "equals" -> (args?.getOrNull(0) === this)
                    else -> null
                }
            }

            // CRITICAL: wrap the OpenApiTool with ToolKt.tool(...) to get a ToolProvider.
            // ConversationConfig.tools is declared `List<? extends ToolProvider>`. The
            // SDK's native bridge resolves tools via ToolProvider.provideTools$...() →
            // InternalJsonTool. Passing a raw OpenApiTool directly silently produces
            // tools=[] at the native layer because of Kotlin's covariant List type-erasure.
            //
            // Concrete symptom verified on Tab S11 (2026-05-29) BEFORE this fix:
            // FunctionGemma 270M emitted "<start_function_call>call EmitSpeechInsights"
            // (44 chars, ~11 tokens) and stopped — the model knew to call the tool but
            // the runtime had no registered tool to bridge to, so generation halted.
            //
            // The wrapper is the top-level Kotlin function
            //   public static ToolProvider ToolKt.tool(OpenApiTool openApiTool)
            // verified via javap on litertlm-android-0.12.0.aar/classes.jar/ToolKt.class.
            val toolKtClass = Class.forName("com.google.ai.edge.litertlm.ToolKt")
            val toolFactoryMethod = toolKtClass.methods.first { m ->
                m.name == "tool" && m.parameterTypes.size == 1 &&
                    m.parameterTypes[0].name == "com.google.ai.edge.litertlm.OpenApiTool"
            }
            val toolProvider = toolFactoryMethod.invoke(null, proxy)
            Log.i(TAG, "Wrapped OpenApiTool proxy in ToolProvider: ${toolProvider?.javaClass?.simpleName}")
            listOf(toolProvider)
        } catch (e: ClassNotFoundException) {
            Log.w(TAG, "OpenApiTool class not found — LiteRT-LM SDK may be older than v0.12.0. " +
                "Falling back to unguided generation (no tool injection).")
            null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to construct OpenApiTool proxy: ${e.message}. " +
                "Falling back to unguided generation.")
            null
        }
    }

    /**
     * Create a fresh Conversation from the cached ConversationConfig.
     *
     * LiteRT-LM conversations accumulate KV cache state across sendMessage calls.
     * Without clearing, this causes SIGSEGV after 3-5 inferences due to KV cache
     * overflow. Creating a fresh conversation per inference clears the KV cache.
     *
     * @return A new Conversation instance, or null if creation fails.
     */
    private fun createFreshConversation(): Any? {
        val currentEngine = engine ?: return null
        val engineClass = currentEngine.javaClass
        val convConfigClass = cachedConvConfigClass ?: return null

        return try {
            val convConfig = cachedConvConfig
            if (convConfig != null) {
                val createConvMethod = engineClass.getMethod(
                    "createConversation",
                    convConfigClass
                )
                createConvMethod.invoke(currentEngine, convConfig)
            } else {
                // Try no-arg createConversation
                val createConvMethod = engineClass.getMethod("createConversation")
                createConvMethod.invoke(currentEngine)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create fresh conversation: ${e.message}", e)
            null
        }
    }

    override suspend fun generate(prompt: String, image: Bitmap?): VLMResponse {
        if (!_isLoaded || engine == null) {
            Log.w(TAG, "generate() called but no model is loaded")
            return VLMResponse(
                text = "",
                latencyMs = 0,
                tokensGenerated = 0,
                parseSuccess = false
            )
        }

        // Close old conversation to release KV cache — prevents SIGSEGV after 3-5 inferences.
        conversation?.let { old ->
            try {
                old.javaClass.getMethod("close").invoke(old)
            } catch (e: Exception) {
                Log.d(TAG, "Previous conversation close failed (non-fatal): ${e.message}")
            }
        }
        // Create fresh conversation (releases KV cache, prevents SIGSEGV)
        conversation = createFreshConversation()
        if (conversation == null) {
            Log.e(TAG, "Failed to create fresh conversation — cannot run inference")
            return VLMResponse(text = "", latencyMs = 0, tokensGenerated = 0, parseSuccess = false)
        }

        val startMs = SystemClock.elapsedRealtime()

        return try {
            Log.d(TAG, "VLM generate start: image=${image != null}, promptChars=${prompt.length}")
            val rawOutput = runBlockingInference {
                if (image != null) {
                    generateMultiModal(prompt, image)
                } else {
                    generateTextOnly(prompt)
                }
            }

            val latencyMs = SystemClock.elapsedRealtime() - startMs

            // Parse the response to extract structured fields
            val parseResult = VLMResponseParser.parse(rawOutput)

            val response = VLMResponse(
                text = rawOutput,
                latencyMs = latencyMs,
                tokensGenerated = estimateTokenCount(rawOutput),
                parseSuccess = parseResult.success,
                parsedFields = parseResult.fields
            )

            Log.d(TAG, "VLM inference complete: latency=${latencyMs}ms, " +
                "tokens~${response.tokensGenerated}, parseSuccess=${parseResult.success}, " +
                "strategy=${parseResult.strategy}, fields=${parseResult.fields.size}")

            response
        } catch (e: CancellationException) {
            val latencyMs = SystemClock.elapsedRealtime() - startMs
            Log.w(TAG, "VLM inference cancelled after ${latencyMs}ms")
            throw e
        } catch (e: Exception) {
            val latencyMs = SystemClock.elapsedRealtime() - startMs
            Log.e(TAG, "VLM inference failed after ${latencyMs}ms: ${e.message}", e)
            VLMResponse(
                text = "",
                latencyMs = latencyMs,
                tokensGenerated = 0,
                parseSuccess = false
            )
        }
    }

    /**
     * Run text-only inference via reflection on the real SDK:
     * ```kotlin
     * val response: Message = conversation.sendMessage("prompt")
     * response.text
     * ```
     */
    private fun generateTextOnly(prompt: String): String {
        val conv = conversation ?: return ""
        return try {
            // Use sendMessage(String, Map) directly — the SDK Conversation class
            // accepts String prompts natively (no need to wrap in Contents)
            val sendMethod = conv.javaClass.methods.first { m ->
                m.name == "sendMessage" && m.parameterCount == 2 &&
                    m.parameterTypes[0] == String::class.java
            }
            Log.d(TAG, "Calling sendMessage(${sendMethod.parameterTypes.map { it.simpleName }.joinToString(",")}) with prompt (${prompt.length} chars)")
            val message = sendMethod.invoke(conv, prompt, emptyMap<String, Any>()) ?: return ""
            Log.d(TAG, "Got Message: ${message.javaClass.name}")

            // ── Tool-call extraction (FunctionGemma path) ─────────────────────
            //
            // FunctionGemma 270M emits the function-call sentinel
            // (`<start_function_call>call <name>`) as a *stop sequence* — the
            // SDK halts generation there and surfaces the parsed call via
            // Message.getToolCalls() (List<ToolCall>) where each ToolCall has
            // getName() and getArguments() : Map<String, Any>. Verified via
            // javap on litertlm-android-0.12.0.aar/classes.jar/Message.class.
            //
            // The raw `getText()` for a tool-emitting model therefore returns
            // only `<start_function_call>call EmitSpeechInsights` (~44 chars,
            // 11 tokens) with no args body — that's the prefix BEFORE the
            // halt. The args body lives in toolCalls[0].arguments.
            //
            // Live Tab S11 verified this 2026-05-29: every speech window
            // logged `VLM raw response (44 chars): <start_function_call>...`
            // and `parseSuccess=false, fields=0` UNTIL this branch was added.
            //
            // Strategy: when toolCalls is non-empty, serialize the first
            // call's arguments map to a JSON object string and return THAT as
            // the response text. VLMResponseParser then runs JSON.parse on it
            // and the existing OnDeviceLlmInsightExtractor → SpeechInsights
            // mapping picks up every field. When toolCalls is empty (non-tool
            // model path, e.g. gemma_4_e2b VLM) we fall back to getText() —
            // zero behaviour change for the VLM camera-frame pipeline.
            val getToolCalls = message.javaClass.methods.firstOrNull { it.name == "getToolCalls" }
            val toolCalls = getToolCalls?.invoke(message) as? List<*>
            if (!toolCalls.isNullOrEmpty()) {
                val firstCall = toolCalls.first()!!
                val getName = firstCall.javaClass.methods.firstOrNull { it.name == "getName" }
                val getArguments = firstCall.javaClass.methods.firstOrNull { it.name == "getArguments" }
                val name = getName?.invoke(firstCall)?.toString() ?: "?"
                @Suppress("UNCHECKED_CAST")
                val args = getArguments?.invoke(firstCall) as? Map<String, Any?> ?: emptyMap()
                // Serialise args -> JSON. Map<String, Any> handles nested Maps,
                // Lists, primitives via org.json's JSONObject(map) ctor.
                val argsJson = org.json.JSONObject(args).toString()
                Log.d(TAG, "Tool-call extracted: name=$name, argsKeys=${args.keys}, argsJsonLen=${argsJson.length}")
                Log.d(TAG, "VLM tool-call response (${argsJson.length} chars): ${argsJson.take(300)}")
                return argsJson
            }

            // Extract text from Message (existing path for non-tool models)
            val getText = message.javaClass.methods.firstOrNull { it.name == "getText" }
            val text = getText?.invoke(message)?.toString() ?: message.toString()
            Log.d(TAG, "VLM raw response (${text.length} chars): ${text.take(300)}")
            text
        } catch (e: java.lang.reflect.InvocationTargetException) {
            Log.e(TAG, "Text-only generation threw: ${e.cause?.message}", e.cause)
            ""
        } catch (e: Exception) {
            Log.e(TAG, "Text-only generation failed: ${e.message}", e)
            ""
        }
    }

    /**
     * Run multi-modal inference (text + image) via reflection on the real SDK.
     *
     * The LiteRT-LM multi-modal API uses Contents.of(Content.ImageFile, Content.Text)
     * and conversation.sendMessage(Contents). We save the bitmap to a temp file and
     * pass it as an image content.
     *
     * ```kotlin
     * val response = conversation.sendMessage(
     *     Contents.of(
     *         Content.ImageFile("/path/to/image.jpg"),
     *         Content.Text("Describe this image.")
     *     )
     * )
     * ```
     */
    private fun generateMultiModal(prompt: String, image: Bitmap): String {
        val conv = conversation ?: return ""

        // Resize image if too large
        val resizedImage = resizeIfNeeded(image)

        // Save bitmap to a temp file for LiteRT-LM ImageFile input
        val tempImageFile = File(tempDir, "vlm_frame_${SystemClock.elapsedRealtime()}.jpg")
        var savedSuccessfully = false
        try {
            FileOutputStream(tempImageFile).use { fos ->
                resizedImage.compress(Bitmap.CompressFormat.JPEG, IMAGE_JPEG_QUALITY, fos)
            }
            savedSuccessfully = true

            // Build Contents.of(Content.ImageFile(path), Content.Text(prompt))
            val contentClass = Class.forName(CLASS_CONTENT)
            val contentTextClass = Class.forName(CLASS_CONTENT_TEXT)
            val contentImageFileClass = Class.forName(CLASS_CONTENT_IMAGE_FILE)
            val contentsClass = Class.forName(CLASS_CONTENTS)

            val imageContent = contentImageFileClass
                .getDeclaredConstructor(String::class.java)
                .newInstance(tempImageFile.absolutePath)
            val textContent = contentTextClass
                .getDeclaredConstructor(String::class.java)
                .newInstance(prompt)

            // Contents.of(vararg Content) -- build array
            val contentArray = java.lang.reflect.Array.newInstance(contentClass, 2)
            java.lang.reflect.Array.set(contentArray, 0, imageContent)
            java.lang.reflect.Array.set(contentArray, 1, textContent)

            // Build Contents from List<Content>.
            // LiteRT-LM 0.9.0 Kotlin API: Contents has a synthetic constructor
            // ctor(List<Content>, DefaultConstructorMarker) — the DefaultConstructorMarker
            // is Kotlin's internal sentinel for default params, pass null for it.
            val contentList = listOf(imageContent, textContent)
            val contents = try {
                // Strategy 1: Contents.of(vararg Content) — present in some SDK versions
                val ofMethod = contentsClass.methods.firstOrNull { m ->
                    m.name == "of" && m.parameterTypes.isNotEmpty()
                }
                if (ofMethod != null) {
                    ofMethod.invoke(null, contentArray)
                } else {
                    // Strategy 2: Kotlin synthetic constructor ctor(List, DefaultConstructorMarker)
                    val listCtor = contentsClass.constructors.firstOrNull { c ->
                        c.parameterCount == 2 &&
                            c.parameterTypes[0] == List::class.java
                    }
                    if (listCtor != null) {
                        listCtor.isAccessible = true
                        listCtor.newInstance(contentList, null)
                    } else {
                        // Strategy 3: Single-arg List constructor
                        val singleListCtor = contentsClass.constructors.firstOrNull { c ->
                            c.parameterCount == 1 && c.parameterTypes[0] == List::class.java
                        }
                        singleListCtor?.newInstance(contentList)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to build Contents: ${e.message}")
                null
            }
            if (contents != null) {
                // conversation.sendMessage(Contents, Map) — 0.9.0 requires the Map param
                val sendMethod = conv.javaClass.methods.firstOrNull { m ->
                    m.name == "sendMessage" && m.parameterCount == 2 &&
                        m.parameterTypes[0] != String::class.java &&
                        m.parameterTypes[1] == Map::class.java
                } ?: conv.javaClass.methods.firstOrNull { m ->
                    m.name == "sendMessage" && m.parameterCount == 1 &&
                        m.parameterTypes[0] != String::class.java
                }
                if (sendMethod != null) {
                    Log.d(TAG, "Calling multimodal sendMessage(${sendMethod.parameterTypes.map { it.simpleName }.joinToString(",")}) with image=${tempImageFile.length()} bytes and prompt (${prompt.length} chars)")
                    val message = if (sendMethod.parameterCount == 2) {
                        sendMethod.invoke(conv, contents, emptyMap<String, Any>())
                    } else {
                        sendMethod.invoke(conv, contents)
                    } ?: return ""
                    val getText = message.javaClass.methods.firstOrNull { it.name == "getText" }
                    val text = getText?.invoke(message)?.toString() ?: message.toString()
                    Log.d(TAG, "VLM multimodal response (${text.length} chars): ${text.take(300)}")
                    return text
                }
            }

            // Multimodal Contents couldn't be built — fall back to text-only
            Log.d(TAG, "Multimodal Contents build failed, falling back to text-only sendMessage")
            return generateTextOnly(prompt)
        } catch (e: Exception) {
            Log.e(TAG, "Multi-modal generation failed: ${e.message}", e)

            // Fallback: text-only with a note about the image
            return try {
                generateTextOnly(
                    "$prompt\n[Note: Image was provided but multi-modal processing failed]"
                )
            } catch (e2: Exception) {
                Log.e(TAG, "Text-only fallback also failed: ${e2.message}", e2)
                ""
            }
        } finally {
            if (resizedImage !== image) {
                resizedImage.recycle()
            }
            if (savedSuccessfully) {
                tempImageFile.delete()
            }
        }
    }

    /**
     * Resize a bitmap if either dimension exceeds [MAX_IMAGE_DIM].
     * Maintains aspect ratio. Returns the original bitmap if no resize is needed.
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
     * Rough token estimate: ~4 characters per token for English text.
     */
    private fun estimateTokenCount(text: String): Int {
        return maxOf(1, text.length / 4)
    }

    override fun unload() {
        cleanupResources()
        Log.i(TAG, "VLM engine unloaded")
    }

    /**
     * Clean up all SDK resources. Both Engine and Conversation implement AutoCloseable.
     */
    private fun cleanupResources() {
        // Close conversation first (it references the engine)
        conversation?.let { conv ->
            try {
                conv.javaClass.getMethod("close").invoke(conv)
                Log.d(TAG, "LiteRT-LM conversation closed")
            } catch (e: Exception) {
                Log.w(TAG, "Error closing conversation: ${e.message}")
            }
        }
        conversation = null

        // Then close the engine
        engine?.let { eng ->
            try {
                eng.javaClass.getMethod("close").invoke(eng)
                Log.d(TAG, "LiteRT-LM engine closed")
            } catch (e: Exception) {
                Log.w(TAG, "Error closing engine: ${e.message}")
            }
        }
        engine = null

        _isLoaded = false
        modelSizeBytes = 0L
        cachedConvConfig = null
        cachedConvConfigClass = null

        // Clean up temp image files
        try {
            tempDir.listFiles()?.forEach { it.delete() }
        } catch (e: Exception) {
            Log.w(TAG, "Error cleaning temp files: ${e.message}")
        }
    }

    override fun getMemoryUsageMb(): Float {
        if (!_isLoaded || modelSizeBytes == 0L) return 0f
        // Model file size * overhead multiplier (KV cache, attention buffers, activations)
        return (modelSizeBytes * MEMORY_OVERHEAD_MULTIPLIER) / (1024f * 1024f)
    }
}
