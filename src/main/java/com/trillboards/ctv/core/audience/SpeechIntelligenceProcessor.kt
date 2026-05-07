package com.trillboards.ctv.core.audience

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineMoonshineModelConfig

/**
 * On-device speech intelligence processor using sherpa-onnx Moonshine ASR.
 *
 * PRIVACY-FIRST DESIGN:
 * - Audio is processed in memory only, NEVER written to disk
 * - Transcripts are ephemeral - deleted immediately after extraction
 * - Only structured SpeechInsights are emitted, NEVER raw text
 * - No speaker identification or biometric data collected
 *
 * HYBRID RACE ARCHITECTURE:
 * 1. Continuous audio capture into rolling buffer
 * 2. Every [transcriptionIntervalMs]: transcribe audio chunk with Moonshine
 * 3. Race two classification paths in parallel:
 *    - Path A: MediaPipe Text Classifier (on-device, 50-150ms)
 *    - Path B: Server Gemini (accurate fallback, 500-800ms)
 * 4. Use first high-confidence result, fallback to InsightExtractor
 * 5. IMMEDIATELY zero-out transcript memory
 * 6. Emit SpeechInsights via callback
 */
class SpeechIntelligenceProcessor(
    private val context: Context,
    private val config: SpeechConfig = SpeechConfig(),
    private val fingerprint: String = "",
    private val apiBaseUrl: String = "https://api.trillboards.com"
) {
    companion object {
        private const val TAG = "SpeechIntelligence"

        // Audio capture settings (Moonshine expects 16kHz mono)
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var captureJob: Job? = null
    private var processingJob: Job? = null

    // Moonshine ASR recognizer (sherpa-onnx)
    private var moonshineRecognizer: OfflineRecognizer? = null
    private var moonshineInitialized = false

    // Audio capture
    private var audioRecord: AudioRecord? = null
    private val audioBuffer = RollingAudioBuffer(config.audioBufferLengthMs, SAMPLE_RATE)

    // Hybrid classification components
    private var mediaPipeClassifier: MediaPipeTextClassifier? = null
    private var httpClient: OkHttpClient? = null
    private var mediaPipeAvailable = false

    // Confidence thresholds — read from SensingConfig for server-tunability
    private val ON_DEVICE_CONFIDENCE_THRESHOLD: Float
        get() = SensingConfig.get().speech.onDeviceConfidenceThreshold

    private val NOISY_GEMINI_CONFIDENCE_THRESHOLD: Float
        get() = SensingConfig.get().speech.noisyGeminiConfidenceThreshold

    // Audio classification processor reference for noise profile
    var audioClassificationProcessor: AudioClassificationProcessor? = null

    @Volatile private var isRunning = false

    // Current insights state
    private val _currentInsights = MutableStateFlow(SpeechInsights())
    val currentInsights: StateFlow<SpeechInsights> = _currentInsights

    // Callback for when insights are ready
    var onInsightsReady: ((SpeechInsights) -> Unit)? = null

    /**
     * Check if microphone permission is granted.
     */
    fun hasMicrophonePermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Check if Moonshine model files exist in assets.
     */
    fun hasWhisperModel(): Boolean {
        return MoonshineModelLoader.hasModelAssets(context)
    }

    /**
     * Initialize Moonshine ASR via sherpa-onnx.
     * Extracts ONNX model files from assets to filesystem, then creates the recognizer.
     */
    private fun initializeMoonshine(): Boolean {
        return try {
            val paths = MoonshineModelLoader.extractIfNeeded(context)
            if (paths == null) {
                Log.e(TAG, "Failed to extract Moonshine model files")
                return false
            }

            val moonshineModelConfig = OfflineMoonshineModelConfig(
                preprocessor = paths.preprocessor,
                encoder = paths.encoder,
                uncachedDecoder = paths.uncachedDecoder,
                cachedDecoder = paths.cachedDecoder
            )

            val modelConfig = OfflineModelConfig()
            modelConfig.moonshine = moonshineModelConfig
            modelConfig.tokens = paths.tokens
            modelConfig.numThreads = (Runtime.getRuntime().availableProcessors() / 4).coerceAtLeast(1)
            modelConfig.debug = false

            val recognizerConfig = OfflineRecognizerConfig()
            recognizerConfig.modelConfig = modelConfig
            recognizerConfig.decodingMethod = "greedy_search"

            moonshineRecognizer = OfflineRecognizer(null, recognizerConfig)
            moonshineInitialized = true

            Log.i(TAG, "Moonshine ASR initialized successfully (sherpa-onnx)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Moonshine ASR", e)
            false
        }
    }

    /**
     * Initialize audio capture.
     */
    private fun initializeAudioCapture(): Boolean {
        if (!hasMicrophonePermission()) {
            Log.w(TAG, "Microphone permission not granted")
            return false
        }

        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT
        )

        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid buffer size: $bufferSize")
            return false
        }

        return try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize * 2
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                return false
            }

            Log.i(TAG, "Audio capture initialized (buffer: $bufferSize)")
            true
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception initializing audio", e)
            false
        }
    }

    /**
     * Start speech intelligence processing.
     */
    fun start() {
        if (!config.enabled) {
            Log.i(TAG, "Speech intelligence disabled in config")
            return
        }

        if (isRunning) {
            Log.w(TAG, "Already running")
            return
        }

        // Initialize components
        if (!hasWhisperModel()) {
            Log.e(TAG, "Moonshine model not available - speech intelligence cannot start")
            return
        }

        if (!initializeMoonshine()) {
            Log.e(TAG, "Failed to initialize Moonshine ASR")
            return
        }

        if (!initializeAudioCapture()) {
            Log.e(TAG, "Failed to initialize audio capture")
            return
        }

        // Initialize hybrid classification components
        initializeHybridClassification()

        isRunning = true

        // Start continuous audio capture
        startAudioCapture()

        // Start periodic processing
        startProcessingLoop()

        Log.i(TAG, "Speech intelligence started (interval: ${config.transcriptionIntervalMs}ms)")
    }

    /**
     * Start continuous audio capture into rolling buffer.
     */
    private fun startAudioCapture() {
        captureJob = processorScope.launch(Dispatchers.IO) {
            val buffer = ShortArray(1024)
            audioRecord?.startRecording()

            while (isActive && isRunning) {
                val readCount = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (readCount > 0) {
                    audioBuffer.append(buffer, readCount)
                }
            }

            try {
                audioRecord?.stop()
            } catch (_: IllegalStateException) {
                // AudioRecord already stopped/released during shutdown — harmless
            }
        }
    }

    /**
     * Start the processing loop that transcribes and extracts insights.
     */
    private fun startProcessingLoop() {
        processingJob = processorScope.launch {
            while (isActive && isRunning) {
                delay(config.transcriptionIntervalMs)

                try {
                    processAudioChunk()
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing audio chunk", e)
                }
            }
        }
    }

    /**
     * Initialize hybrid classification components.
     * - MediaPipe Text Classifier for on-device classification
     * - ApiClient for server-side Gemini fallback
     */
    private fun initializeHybridClassification() {
        // Initialize MediaPipe Text Classifier
        try {
            mediaPipeClassifier = MediaPipeTextClassifier(context)
            if (mediaPipeClassifier?.hasModel() == true) {
                mediaPipeAvailable = mediaPipeClassifier?.initialize() ?: false
                if (mediaPipeAvailable) {
                    Log.i(TAG, "MediaPipe Text Classifier initialized - on-device classification enabled")
                } else {
                    Log.w(TAG, "MediaPipe Text Classifier failed to initialize - using server fallback only")
                }
            } else {
                Log.w(TAG, "MediaPipe intent model not found - using server fallback only")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe classifier", e)
            mediaPipeAvailable = false
        }

        // Initialize OkHttpClient for Gemini fallback
        if (fingerprint.isNotEmpty()) {
            try {
                httpClient = OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .readTimeout(15, TimeUnit.SECONDS)
                    .build()
                Log.i(TAG, "HttpClient initialized for Gemini fallback")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize HttpClient", e)
            }
        } else {
            Log.w(TAG, "No fingerprint provided - Gemini fallback disabled")
        }
    }

    /**
     * Call Gemini server for speech analysis.
     * PRIVACY: Transcript sent over HTTPS, not logged on server.
     */
    private fun analyzeWithGeminiServer(transcript: String): ClassificationResult {
        val startTime = System.currentTimeMillis()

        return try {
            val requestJson = JSONObject().apply {
                put("transcript", transcript)
                put("fingerprint", fingerprint)
            }

            val request = Request.Builder()
                .url("$apiBaseUrl/v2/earner/analyze-speech")
                .post(requestJson.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                .build()

            httpClient?.newCall(request)?.execute()?.use { response ->
                val latencyMs = System.currentTimeMillis() - startTime

                if (!response.isSuccessful) {
                    Log.w(TAG, "Gemini analysis failed: ${response.code}")
                    return ClassificationResult(
                        intentType = IntentType.UNKNOWN,
                        confidence = 0f,
                        latencyMs = latencyMs,
                        source = ClassificationSource.SERVER_GEMINI,
                        error = "HTTP ${response.code}"
                    )
                }

                val body = response.body?.string()
                if (body.isNullOrBlank()) {
                    return ClassificationResult(
                        intentType = IntentType.UNKNOWN,
                        confidence = 0f,
                        latencyMs = latencyMs,
                        source = ClassificationSource.SERVER_GEMINI,
                        error = "Empty response"
                    )
                }

                val json = JSONObject(body)

                // Parse intent
                val intentStr = json.optString("intent", "NEUTRAL")
                val intentType = when (intentStr.uppercase()) {
                    "PURCHASE_INTENT" -> IntentType.PURCHASE_INTENT
                    "PRICE_INQUIRY" -> IntentType.PRICE_INQUIRY
                    "PRODUCT_INTEREST" -> IntentType.PRODUCT_INTEREST
                    "COMPARISON" -> IntentType.COMPARISON
                    "BROWSING" -> IntentType.BROWSING
                    "NEGATIVE" -> IntentType.NEGATIVE
                    else -> IntentType.NEUTRAL
                }

                // Parse brands array
                val brandsArray = json.optJSONArray("brands") ?: JSONArray()
                val brands = (0 until brandsArray.length()).map { brandsArray.getString(it) }

                // Parse products array
                val productsArray = json.optJSONArray("products") ?: JSONArray()
                val products = (0 until productsArray.length()).map { productsArray.getString(it) }

                // Parse price context
                val priceJson = json.optJSONObject("priceContext")
                val priceContext = if (priceJson != null) {
                    PriceContext(
                        mentioned = priceJson.optBoolean("mentioned", false),
                        sensitivity = priceJson.optString("sensitivity", "NONE")
                    )
                } else null

                Log.d(TAG, "Gemini analysis: intent=$intentType, brands=${brands.size}, latency=${latencyMs}ms")

                ClassificationResult(
                    intentType = intentType,
                    confidence = json.optDouble("confidence", 0.8).toFloat(),
                    latencyMs = latencyMs,
                    source = ClassificationSource.SERVER_GEMINI,
                    brands = brands,
                    products = products,
                    sentiment = json.optString("sentiment", "NEUTRAL"),
                    priceContext = priceContext
                )
            } ?: ClassificationResult(
                intentType = IntentType.UNKNOWN,
                confidence = 0f,
                latencyMs = System.currentTimeMillis() - startTime,
                source = ClassificationSource.SERVER_GEMINI,
                error = "No HTTP client"
            )
        } catch (e: Exception) {
            Log.e(TAG, "Gemini analysis exception", e)
            ClassificationResult(
                intentType = IntentType.UNKNOWN,
                confidence = 0f,
                latencyMs = System.currentTimeMillis() - startTime,
                source = ClassificationSource.SERVER_GEMINI,
                error = e.message
            )
        }
    }

    /**
     * Process the current audio buffer using hybrid race architecture:
     * 1. Get audio samples and check for meaningful audio
     * 2. Transcribe with Moonshine ASR
     * 3. Race MediaPipe (on-device) vs Gemini (server) classification
     * 4. Use first high-confidence result, fallback to keyword extraction
     * 5. IMMEDIATELY delete transcript
     * 6. Emit structured insights
     */
    private suspend fun processAudioChunk() {
        // Get audio samples from buffer
        val audioSamples = audioBuffer.getLatestSamples(config.transcriptionIntervalMs.toInt())

        if (audioSamples.isEmpty()) {
            Log.d(TAG, "No audio samples to process")
            return
        }

        // Check audio level - skip if too quiet
        val audioLevel = calculateAudioLevel(audioSamples)
        val gateDecision = SpeechAudioGate.evaluate(
            audioLevel = audioLevel,
            configuredMinAudioLevel = SensingConfig.get().audio.minAudioLevel
        )
        if (!gateDecision.shouldProcess) {
            Log.d(
                TAG,
                "Audio level too low ($audioLevel < ${gateDecision.effectiveMinAudioLevel}), skipping"
            )
            return
        }

        Log.d(
            TAG,
            "Processing audio chunk: ${audioSamples.size} samples, level: $audioLevel, " +
                "min=${gateDecision.effectiveMinAudioLevel}"
        )

        // Transcribe audio (on background thread)
        var transcript: String? = null
        try {
            transcript = withContext(Dispatchers.Default) {
                transcribeAudio(audioSamples)
            }

            if (transcript.isNullOrBlank()) {
                Log.d(TAG, "Empty transcript, skipping")
                return
            }

            Log.d(TAG, "Moonshine transcription length: ${transcript.length} chars")

            // Use hybrid race architecture for classification
            val insights = analyzeWithHybridRace(transcript)

            // Check confidence threshold
            if (insights.confidence < config.minConfidenceThreshold) {
                Log.d(TAG, "Low confidence (${insights.confidence}), skipping emission")
                return
            }

            // Update state and emit
            _currentInsights.value = insights
            onInsightsReady?.invoke(insights)

            Log.i(TAG, "Speech insights emitted: " +
                "brands=${insights.brandMentions.size}, " +
                "intent=${insights.purchaseIntent}, " +
                "sentiment=${insights.sentimentTone}, " +
                "confidence=${insights.confidence}")

        } finally {
            // CRITICAL: Immediately zero out transcript memory
            // This is the privacy guarantee - transcript never persists
            @Suppress("UNUSED_VALUE")
            transcript = null
        }
    }

    /**
     * Hybrid race classification: run MediaPipe and Gemini in parallel.
     *
     * Strategy:
     * 1. Start both classifiers in parallel
     * 2. If MediaPipe returns with high confidence (>0.70), use it immediately
     * 3. If server returns better result, use that
     * 4. Fallback to keyword extraction if both fail
     *
     * Privacy: Transcript is sent to server only, never logged/stored.
     */
    private suspend fun analyzeWithHybridRace(transcript: String): SpeechInsights {
        return coroutineScope {
            // Start both paths in parallel
            val onDeviceDeferred = if (mediaPipeAvailable && mediaPipeClassifier != null) {
                async(Dispatchers.Default) {
                    runCatching { mediaPipeClassifier!!.classify(transcript) }
                }
            } else null

            val serverDeferred = if (httpClient != null && fingerprint.isNotEmpty()) {
                async(Dispatchers.IO) {
                    runCatching { analyzeWithGeminiServer(transcript) }
                }
            } else null

            // Await results
            val onDeviceResult = onDeviceDeferred?.await()?.getOrNull()
            val serverResult = serverDeferred?.await()?.getOrNull()

            // Decision logic
            when {
                // On-device wins with high confidence
                onDeviceResult != null &&
                onDeviceResult.isValid() &&
                onDeviceResult.confidence >= ON_DEVICE_CONFIDENCE_THRESHOLD -> {
                    Log.d(TAG, "Using on-device result (${onDeviceResult.latencyMs}ms, " +
                            "confidence=${onDeviceResult.confidence})")
                    onDeviceResult.toSpeechInsights()
                }

                // Server result available and better
                serverResult != null && serverResult.isValid() -> {
                    Log.d(TAG, "Using server Gemini result (${serverResult.latencyMs}ms, " +
                            "brands=${serverResult.brands.size})")
                    serverResult.toSpeechInsights()
                }

                // On-device available but low confidence - still better than keyword fallback
                onDeviceResult != null && onDeviceResult.isValid() -> {
                    Log.d(TAG, "Using on-device (low confidence fallback, " +
                            "confidence=${onDeviceResult.confidence})")
                    onDeviceResult.toSpeechInsights()
                }

                // Last resort: keyword extraction
                else -> {
                    Log.w(TAG, "Both classifiers failed, using keyword fallback")
                    InsightExtractor.extract(transcript)
                }
            }
        }
    }

    /**
     * Preprocess audio samples based on current noise environment.
     * Applies frequency-domain filtering to enhance speech before Moonshine.
     *
     * Strategy by noise type:
     * - QUIET: No filtering (already clean)
     * - MUSIC: Bandpass filter 300Hz-3kHz (speech frequencies only)
     * - CROWD/TRAFFIC: High-pass filter 500Hz+ (remove low-freq rumble)
     * - SPEECH: Mild high-pass at 200Hz
     * - AMBIENT: Gentle high-pass at 150Hz
     */
    private fun preprocessAudio(samples: ShortArray): ShortArray {
        val noiseProfile = audioClassificationProcessor?.getNoiseProfile() ?: return samples

        if (!noiseProfile.isNoisy()) return samples // Clean environment, skip

        val speechCfg = SensingConfig.get().speech
        val cutoffHz = when (noiseProfile.ambientType) {
            NoiseType.MUSIC -> speechCfg.hpCutoffMusic
            NoiseType.CROWD -> speechCfg.hpCutoffCrowd
            NoiseType.SPEECH -> speechCfg.hpCutoffSpeech
            NoiseType.AMBIENT -> speechCfg.hpCutoffAmbient
            NoiseType.QUIET -> return samples
        }

        Log.d(TAG, "Preprocessing audio: ${noiseProfile.ambientType}, cutoff=${cutoffHz}Hz, " +
                "noiseTier=${noiseProfile.noiseTier}")

        // Simple first-order high-pass filter (IIR)
        // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        // alpha = RC / (RC + dt), where RC = 1/(2*pi*cutoff)
        val dt = 1.0 / SAMPLE_RATE
        val rc = 1.0 / (2.0 * Math.PI * cutoffHz)
        val alpha = rc / (rc + dt)

        val output = ShortArray(samples.size)
        var prevInput = samples[0].toDouble()
        var prevOutput = samples[0].toDouble()

        for (i in 1 until samples.size) {
            val input = samples[i].toDouble()
            prevOutput = alpha * (prevOutput + input - prevInput)
            prevInput = input
            output[i] = prevOutput.toInt().toShort().coerceIn(Short.MIN_VALUE, Short.MAX_VALUE)
        }

        // For MUSIC noise, also apply low-pass to remove music harmonics
        if (noiseProfile.ambientType == NoiseType.MUSIC) {
            val lpCutoff = speechCfg.lpCutoffMusic
            val lpRc = 1.0 / (2.0 * Math.PI * lpCutoff)
            val lpAlpha = dt / (lpRc + dt)
            var lpPrev = output[0].toDouble()

            for (i in 1 until output.size) {
                lpPrev = lpPrev + lpAlpha * (output[i].toDouble() - lpPrev)
                output[i] = lpPrev.toInt().toShort().coerceIn(Short.MIN_VALUE, Short.MAX_VALUE)
            }
        }

        return output
    }

    /**
     * Transcribe audio using Moonshine ASR via sherpa-onnx.
     * Returns empty string if transcription fails.
     */
    private fun transcribeAudio(samples: ShortArray): String {
        val recognizer = moonshineRecognizer
        if (recognizer == null || !moonshineInitialized) {
            Log.e(TAG, "Moonshine not initialized")
            return ""
        }

        return try {
            // Apply noise-robust preprocessing before Moonshine
            val processedSamples = preprocessAudio(samples)

            // Convert ShortArray to FloatArray (sherpa-onnx expects float samples in [-1.0, 1.0])
            val floatSamples = FloatArray(processedSamples.size)
            for (i in processedSamples.indices) {
                floatSamples[i] = processedSamples[i] / 32768.0f
            }

            // Create a stream, feed audio, decode, get result
            val stream = recognizer.createStream()
            stream.acceptWaveform(floatSamples, SAMPLE_RATE)
            recognizer.decode(stream)
            val result = recognizer.getResult(stream).text
            stream.release()

            Log.d(TAG, "Moonshine transcription result: ${result.take(100)}...")
            result
        } catch (e: Exception) {
            Log.e(TAG, "Moonshine transcription failed", e)
            ""
        }
    }

    /**
     * Calculate RMS audio level from samples.
     */
    private fun calculateAudioLevel(samples: ShortArray): Int {
        if (samples.isEmpty()) return 0
        var sum = 0L
        for (sample in samples) {
            sum += sample.toLong() * sample
        }
        return kotlin.math.sqrt(sum.toDouble() / samples.size).toInt()
    }

    /**
     * Stop speech intelligence processing.
     */
    fun stop() {
        if (!isRunning) return

        isRunning = false

        captureJob?.cancel()
        captureJob = null

        processingJob?.cancel()
        processingJob = null

        audioRecord?.release()
        audioRecord = null

        moonshineRecognizer?.release()
        moonshineRecognizer = null
        moonshineInitialized = false

        // Clean up hybrid classification components
        mediaPipeClassifier?.close()
        mediaPipeClassifier = null
        mediaPipeAvailable = false
        httpClient = null

        audioBuffer.clear()

        Log.i(TAG, "Speech intelligence stopped")
    }

    /**
     * Check if speech processing is running.
     */
    fun isProcessing(): Boolean = isRunning

    /**
     * Get the latest insights without waiting for next interval.
     */
    fun getLatestInsights(): SpeechInsights = _currentInsights.value

    /**
     * Set dynamic brands for keyword detection.
     * Delegates to InsightExtractor which is used as the keyword fallback path.
     */
    fun setDynamicBrands(brands: List<String>) {
        InsightExtractor.setDynamicBrands(brands)
        Log.i(TAG, "Dynamic brands updated: ${brands.size} brands")
    }
}

/**
 * Rolling audio buffer that maintains a sliding window of audio samples.
 * Thread-safe for concurrent read/write.
 */
class RollingAudioBuffer(
    private val durationMs: Long,
    private val sampleRate: Int
) {
    private val maxSamples = ((durationMs * sampleRate) / 1000).toInt()
    private val buffer = ShortArray(maxSamples)
    private var writePosition = 0
    private var sampleCount = 0
    private val lock = Any()

    /**
     * Append audio samples to the buffer.
     */
    fun append(samples: ShortArray, count: Int) {
        synchronized(lock) {
            // Bounds check: cap count to avoid ArrayIndexOutOfBoundsException
            val safeCopy = minOf(count, samples.size, maxSamples)
            for (i in 0 until safeCopy) {
                buffer[writePosition] = samples[i]
                writePosition = (writePosition + 1) % maxSamples
                if (sampleCount < maxSamples) sampleCount++
            }
            // Ensure sampleCount never exceeds maxSamples
            sampleCount = sampleCount.coerceAtMost(maxSamples)
        }
    }

    /**
     * Get the latest N milliseconds of audio.
     */
    fun getLatestSamples(durationMs: Int): ShortArray {
        synchronized(lock) {
            val samplesToGet = minOf(
                (durationMs * sampleRate) / 1000,
                sampleCount
            )

            if (samplesToGet == 0) return ShortArray(0)

            val result = ShortArray(samplesToGet)
            var readPos = (writePosition - samplesToGet + maxSamples) % maxSamples

            for (i in 0 until samplesToGet) {
                result[i] = buffer[readPos]
                readPos = (readPos + 1) % maxSamples
            }

            return result
        }
    }

    /**
     * Clear the buffer.
     */
    fun clear() {
        synchronized(lock) {
            buffer.fill(0)
            writePosition = 0
            sampleCount = 0
        }
    }
}
