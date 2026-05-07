package com.trillboards.ctv.core.audience

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.audio.audioclassifier.AudioClassifier
import com.google.mediapipe.tasks.audio.audioclassifier.AudioClassifierResult
import com.google.mediapipe.tasks.audio.core.RunningMode
import com.google.mediapipe.tasks.components.containers.AudioData
import com.google.mediapipe.tasks.core.BaseOptions
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.label.Category

/**
 * Processes ambient audio using TensorFlow Lite to classify the environment.
 * Uses YAMNet model to detect crowd sounds, music, speech, etc.
 *
 * Privacy-first design:
 * - Only classifies ambient sound type (not content)
 * - No audio is recorded or transmitted
 * - Only aggregate classifications are collected
 */
@OptIn(ExperimentalCoroutinesApi::class)
class AudioClassificationProcessor(
    private val context: Context,
    private val config: AudioConfig = AudioConfig()
) {
    companion object {
        private const val TAG = "AudioClassification"
        private const val MODEL_FILE = "yamnet.tflite"
        private const val MAX_RETRIES = 3
        private const val RETRY_DELAY_MS = 2000L

        // Ordinal noise tiers (0-5 scale) — YAMNet classifies sound type, not volume
        // These replace the previous fabricated dB constants that implied false precision
        private const val NOISE_TIER_QUIET = 0       // Silence/white noise
        private const val NOISE_TIER_LOW = 1          // Faint ambient
        private const val NOISE_TIER_MODERATE = 2     // Normal conversation
        private const val NOISE_TIER_ELEVATED = 3     // Background music
        private const val NOISE_TIER_HIGH = 4         // Crowded space
        private const val NOISE_TIER_VERY_HIGH = 5    // Crowded venue with music

        // YAMNet class IDs for relevant audio categories
        private val CROWD_CLASSES = setOf(
            "crowd", "cheering", "applause", "chatter", "hubbub", "speech", "babble",
            "conversation", "crowd noise", "ambient music"
        )
        private val MUSIC_CLASSES = setOf(
            "music", "pop music", "rock music", "hip hop music", "jazz", "classical music",
            "electronic music", "singing", "song", "background music"
        )
        private val SPEECH_CLASSES = setOf(
            "speech", "narration", "conversation", "male speech", "female speech",
            "child speech", "whispering"
        )
        private val QUIET_CLASSES = setOf(
            "silence", "quiet", "white noise", "pink noise", "static"
        )
    }

    private val classificationScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val controlScope = CoroutineScope(SupervisorJob() + Dispatchers.Default.limitedParallelism(1))
    private var classificationJob: Job? = null

    private var classifier: AudioClassifier? = null

    // YAMNet model contract: 16 kHz mono float32 PCM, 0.975-second window
    // (15600 samples per inference). MediaPipe AudioClassifier in
    // RunningMode.AUDIO_CLIPS expects an `AudioData` populated with these
    // samples per `classify()` call. Hard-coded — yamnet.tflite ships with
    // these dims fixed; bumping requires a new model file.
    private val sampleRateHz = 16000
    private val samplesPerWindow = 15600
    private var audioRecord: AudioRecord? = null

    @Volatile private var isRunning = false
    @Volatile private var restartInProgress = false

    // Current audio metrics state
    private val _currentMetrics = MutableStateFlow(AudioMetrics())
    val currentMetrics: StateFlow<AudioMetrics> = _currentMetrics

    // Callback for when metrics are ready
    var onMetricsReady: ((AudioMetrics) -> Unit)? = null

    /**
     * Phase 4 PR 8: optional callback delivering raw YAMNet category outputs
     * (label, score) per classification cycle, alongside the wall-clock
     * timestamp the cycle produced. Wired by AudienceSensingService into
     * YamnetHistogramExtractor for per-class binned counts in
     * audienceSignals.audio_class_histogram.
     *
     * Distinct from onMetricsReady because the histogram needs the FULL
     * category list (not just the dominant class) to preserve multi-class
     * signal — see YamnetHistogramExtractor for rationale.
     */
    var onCategoriesReady: ((List<org.tensorflow.lite.support.label.Category>, Long) -> Unit)? = null

    // Callbacks for audio state changes (matching camera pattern)
    var onAudioReady: (() -> Unit)? = null
    var onAudioFailed: ((String) -> Unit)? = null

    // Retry tracking
    @Volatile private var retryCount = 0
    @Volatile private var isIntentionalStop = false

    /**
     * Check if microphone hardware is available.
     */
    fun hasMicrophone(): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_MICROPHONE)
    }

    /**
     * Check if the model file exists in assets.
     */
    fun hasModel(): Boolean {
        return try {
            context.assets.open(MODEL_FILE).close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "YAMNet model not found in assets: $MODEL_FILE")
            false
        }
    }

    /**
     * Initialize the audio classifier.
     */
    private fun initialize(): Boolean {
        if (!hasMicrophone()) {
            Log.w(TAG, "No microphone available")
            return false
        }

        if (!hasModel()) {
            Log.w(TAG, "YAMNet model not available")
            return false
        }

        return try {
            // MediaPipe AudioClassifier replaces the deprecated TFLite Task
            // Library `AudioClassifier`. Same YAMNet .tflite asset works
            // without conversion. RunningMode.AUDIO_CLIPS gives synchronous
            // `classify()` calls — matching the existing polling loop —
            // rather than the async `classifyAsync()` of AUDIO_STREAM.
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_FILE)
                .build()

            val options = AudioClassifier.AudioClassifierOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.AUDIO_CLIPS)
                .setMaxResults(config.maxResults)
                .setScoreThreshold(config.scoreThreshold)
                .build()

            classifier = AudioClassifier.createFromOptions(context, options)

            Log.i(TAG, "Audio classifier initialized successfully (MediaPipe ${MODEL_FILE})")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize audio classifier", e)
            false
        }
    }

    /**
     * Build a 16 kHz mono PCM_FLOAT [AudioRecord] sized for one YAMNet
     * inference window. The TFLite Task Library used to auto-create this
     * via `classifier.createAudioRecord()`; MediaPipe leaves audio capture
     * to the caller. Returns null on permission denial / hardware failure
     * (start() handles the retry path).
     */
    private fun buildAudioRecord(): AudioRecord? {
        val hasPerm = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (!hasPerm) {
            Log.w(TAG, "RECORD_AUDIO permission not granted")
            return null
        }
        return try {
            val minBuf = AudioRecord.getMinBufferSize(
                sampleRateHz,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT
            )
            val bufBytes = maxOf(minBuf, samplesPerWindow * java.lang.Float.BYTES * 4)
            @Suppress("MissingPermission")
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRateHz,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
                bufBytes
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to construct AudioRecord", e)
            null
        }
    }

    /**
     * Start audio classification with retry support.
     */
    fun start() {
        controlScope.launch {
            startInternal()
        }
    }

    private suspend fun startInternal() {
        if (!config.enabled) {
            Log.i(TAG, "Audio classification disabled in config")
            return
        }

        if (isRunning || classificationJob?.isActive == true) {
            Log.w(TAG, "Already running")
            return
        }

        if (!initialize()) {
            val errorMsg = "Failed to initialize audio classifier"
            Log.e(TAG, errorMsg)
            onAudioFailed?.invoke(errorMsg)
            scheduleRetry()
            return
        }

        isRunning = true

        classificationJob = classificationScope.launch {
            var record: AudioRecord? = null
            try {
                record = buildAudioRecord()
                if (record == null) {
                    val errorMsg = "Failed to create AudioRecord"
                    Log.e(TAG, errorMsg)
                    isRunning = false
                    onAudioFailed?.invoke(errorMsg)
                    scheduleRetry()
                    return@launch
                }
                audioRecord = record

                // Allocate the MediaPipe AudioData wrapper + the underlying
                // FloatArray once per loop start; we reload samples each
                // iteration via `audioData.load(samples)`. Format must match
                // YAMNet's published contract (16 kHz mono).
                val audioFormat = AudioData.AudioDataFormat.builder()
                    .setNumOfChannels(1)
                    .setSampleRate(sampleRateHz.toFloat())
                    .build()
                val audioData = AudioData.create(audioFormat, samplesPerWindow)
                val samples = FloatArray(samplesPerWindow)

                record.startRecording()

                // Verify recording actually started
                if (record.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                    val errorMsg = "AudioRecord failed to start (state: ${record.recordingState})"
                    Log.e(TAG, errorMsg)
                    isRunning = false
                    record.release()
                    audioRecord = null
                    onAudioFailed?.invoke(errorMsg)
                    scheduleRetry()
                    return@launch
                }

                // Success! Reset retry count and notify
                retryCount = 0
                Log.i(TAG, ">>> Audio recording started successfully")
                onAudioReady?.invoke()

                while (isActive && isRunning) {
                    try {
                        // Read one full YAMNet window (15600 samples). On
                        // most devices `read` blocks until the buffer is
                        // full when given a positive sample count + the
                        // BLOCKING flag; if we get a partial read we just
                        // pass what we got — MediaPipe pads internally.
                        val readSamples = record.read(
                            samples, 0, samples.size, AudioRecord.READ_BLOCKING
                        )
                        if (readSamples <= 0) {
                            // ERROR_INVALID_OPERATION / ERROR_BAD_VALUE / 0 — try again next tick
                            delay(config.classificationIntervalMs)
                            continue
                        }
                        if (readSamples < samples.size) {
                            // Zero out the tail so leftover state from the
                            // previous cycle doesn't bleed into the model.
                            for (i in readSamples until samples.size) samples[i] = 0f
                        }

                        audioData.load(samples)

                        // MediaPipe AUDIO_CLIPS-mode classify() returns
                        // synchronously. Result shape (per AAR introspection
                        // 2026-05-04):
                        //   result.classificationResults()  // List<ClassificationResult>
                        //     .first().classifications()    // List<Classifications>
                        //     .first().categories()         // List<Category>
                        val result: AudioClassifierResult? = classifier?.classify(audioData)
                        val mpResults = result?.classificationResults().orEmpty()
                        val mpClassifications = mpResults.firstOrNull()?.classifications().orEmpty()
                        val mpCategories = mpClassifications.firstOrNull()?.categories().orEmpty()

                        if (mpCategories.isNotEmpty()) {
                            // Convert MediaPipe Categories to TFLite-Support
                            // Category so downstream consumers
                            // (analyzeClassifications, YamnetHistogramExtractor)
                            // keep working unchanged. tensorflow-lite-support
                            // dep stays in the build (pure JVM, no native libs,
                            // no 16 KB-alignment concern). Category.create()
                            // signature: (label, displayName, score, index).
                            val categories: List<Category> = mpCategories.map { c ->
                                Category.create(
                                    c.categoryName() ?: "",
                                    c.displayName() ?: "",
                                    c.score(),
                                    c.index()
                                )
                            }

                            // Analyze classifications
                            val metrics = analyzeClassifications(categories)
                            _currentMetrics.value = metrics

                            // Invoke callback
                            onMetricsReady?.invoke(metrics)

                            // Phase 4 PR 8: deliver raw categories to the
                            // histogram extractor (timestamp = metrics.timestamp
                            // so bin alignment matches the same cycle).
                            try {
                                onCategoriesReady?.invoke(categories, metrics.timestamp)
                            } catch (_: Exception) {
                                // Best-effort — never fail the classification loop.
                            }
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Classification error", e)
                    }

                    delay(config.classificationIntervalMs)
                }
            } catch (e: CancellationException) {
                if (isIntentionalStop) {
                    Log.d(TAG, "Audio stopped intentionally, not reporting as failure")
                } else {
                    Log.d(TAG, "Audio classification loop cancelled")
                }
                throw e
            } catch (e: Exception) {
                // Don't report failure or retry if this was an intentional stop (restart scenario)
                if (isIntentionalStop) {
                    Log.d(TAG, "Audio stopped intentionally, not reporting as failure")
                } else {
                    Log.e(TAG, "Audio classification loop error", e)
                    isRunning = false
                    onAudioFailed?.invoke("Audio loop error: ${e.message}")
                    scheduleRetry()
                }
            } finally {
                record?.let { audioInput ->
                    try {
                        if (audioInput.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                            audioInput.stop()
                        }
                    } catch (_: IllegalStateException) {
                        // Already stopped/released during shutdown
                    }
                    try {
                        audioInput.release()
                    } catch (_: Exception) {
                        // Best-effort cleanup
                    }
                }
            }
        }
    }

    /**
     * Schedule a retry if we haven't exceeded max retries.
     */
    private fun scheduleRetry() {
        if (retryCount < MAX_RETRIES) {
            retryCount++
            val delayMs = RETRY_DELAY_MS * retryCount  // Exponential backoff
            Log.i(TAG, "Scheduling audio retry $retryCount/$MAX_RETRIES in ${delayMs}ms")

            classificationScope.launch {
                delay(delayMs)
                if (!isRunning) {
                    Log.i(TAG, "Retrying audio initialization (attempt $retryCount)")
                    start()
                }
            }
        } else {
            Log.e(TAG, "Max retries ($MAX_RETRIES) exceeded - audio classification unavailable")
            onAudioFailed?.invoke("Max retries exceeded")
        }
    }

    /**
     * Restart audio classification (for use after app resume).
     */
    fun restart() {
        controlScope.launch {
            if (restartInProgress) {
                Log.w(TAG, "Restart already in progress, skipping duplicate request")
                return@launch
            }

            restartInProgress = true
            try {
                Log.i(TAG, "Restarting audio classification...")
                stopInternal(intentional = true)
                retryCount = 0  // Reset retry count on manual restart
                startInternal()
            } finally {
                restartInProgress = false
            }
        }
    }

    /**
     * Stop audio classification.
     */
    fun stop() {
        controlScope.launch {
            stopInternal(intentional = true)
        }
    }

    private suspend fun stopInternal(intentional: Boolean) {
        val jobToJoin = classificationJob

        if (!isRunning && jobToJoin == null && classifier == null) {
            return
        }

        if (intentional) {
            isIntentionalStop = true
        }

        isRunning = false
        classificationJob = null
        jobToJoin?.cancel()
        jobToJoin?.join()

        try {
            classifier?.close()
        } catch (_: Exception) {
            // Best-effort cleanup
        }
        classifier = null
        audioRecord = null
        isIntentionalStop = false

        Log.i(TAG, "Audio classification stopped")
    }

    /**
     * Analyze classification results to extract meaningful metrics.
     */
    private fun analyzeClassifications(
        categories: List<org.tensorflow.lite.support.label.Category>
    ): AudioMetrics {
        val timestamp = System.currentTimeMillis()

        // Get top classification
        val topCategory = categories.firstOrNull()
        val dominantClass = topCategory?.label ?: "unknown"
        val confidence = topCategory?.score ?: 0f

        // Check for specific audio types
        var isCrowded = false
        var isMusicPlaying = false
        var hasSpeech = false
        var isQuiet = false

        for (category in categories) {
            val label = category.label.lowercase()
            val score = category.score

            if (score < SensingConfig.get().audio.minClassificationConfidence) continue

            when {
                CROWD_CLASSES.any { it in label } -> isCrowded = true
                MUSIC_CLASSES.any { it in label } -> isMusicPlaying = true
                SPEECH_CLASSES.any { it in label } -> hasSpeech = true
                QUIET_CLASSES.any { it in label } -> isQuiet = true
            }
        }

        // Classify noise tier based on detected audio patterns
        val ambientNoiseTier = when {
            isQuiet -> NOISE_TIER_QUIET
            isCrowded && isMusicPlaying -> NOISE_TIER_VERY_HIGH
            isCrowded -> NOISE_TIER_HIGH
            isMusicPlaying -> NOISE_TIER_ELEVATED
            hasSpeech -> NOISE_TIER_MODERATE
            else -> NOISE_TIER_LOW
        }

        // Determine dominant ambience type
        val ambienceType = when {
            isCrowded && isMusicPlaying -> "crowded_with_music"
            isCrowded -> "crowded"
            isMusicPlaying -> "music_playing"
            hasSpeech -> "conversational"
            isQuiet -> "quiet"
            else -> "ambient"
        }

        // Derived signal: Estimate occupancy from audio patterns
        val estimatedOccupancy = when {
            ambientNoiseTier >= NOISE_TIER_VERY_HIGH && isCrowded -> "50+"
            ambientNoiseTier >= NOISE_TIER_HIGH && isCrowded -> "20-50"
            ambientNoiseTier >= NOISE_TIER_MODERATE -> "5-20"
            else -> "0-5"
        }

        // Ad receptivity = weighted(audience_size, attention, viewer_presence)
        // Server-tunable via SensingConfig — breaks old 2-factor compression (3 collapsed values)
        val cfg = SensingConfig.get().audio
        val audienceSizeFactor = when {
            isCrowded && isMusicPlaying -> cfg.audienceSizeCrowdedMusic
            isCrowded -> cfg.audienceSizeCrowded
            hasSpeech -> cfg.audienceSizeSpeech
            isQuiet -> cfg.audienceSizeQuiet
            else -> cfg.audienceSizeDefault
        }
        val attentionFactor = when {
            isQuiet -> cfg.attentionQuiet
            hasSpeech && !isCrowded -> cfg.attentionSpeechOnly
            isMusicPlaying && !isCrowded -> cfg.attentionMusicOnly
            isCrowded && !isMusicPlaying -> cfg.attentionCrowdOnly
            isCrowded && isMusicPlaying -> cfg.attentionCrowdMusic
            else -> cfg.attentionDefault
        }
        val adReceptivityScore = (audienceSizeFactor * cfg.receptivityAudienceWeight
            + attentionFactor * cfg.receptivityAttentionWeight).coerceIn(0f, 1f)

        // Derived signal: Infer venue type from audio patterns
        val inferredVenueType = when {
            isMusicPlaying && isCrowded -> "entertainment"  // Bar, club, concert
            isMusicPlaying && !isCrowded -> "retail"  // Store with background music
            hasSpeech && isCrowded -> "restaurant"  // Many conversations
            hasSpeech && !isCrowded -> "office"  // Some conversation, not crowded
            isQuiet -> "waiting_area"  // Lobby, clinic, transit
            else -> "unknown"
        }

        return AudioMetrics(
            timestamp = timestamp,
            dominantClass = dominantClass,
            confidence = confidence,
            ambientNoiseLevel = ambientNoiseTier.toFloat(),
            isCrowded = isCrowded,
            isMusicPlaying = isMusicPlaying,
            hasSpeech = hasSpeech,
            ambienceType = ambienceType,
            estimatedOccupancy = estimatedOccupancy,
            adReceptivityScore = adReceptivityScore,
            inferredVenueType = inferredVenueType
        )
    }

    /**
     * Check if audio classification is currently running.
     */
    fun isClassifying(): Boolean = isRunning

    /**
     * Get the current noise profile for speech preprocessing.
     * Used by SpeechIntelligenceProcessor to adapt noise filtering.
     */
    fun getNoiseProfile(): NoiseProfile {
        val metrics = _currentMetrics.value
        val ambientType = when {
            metrics.isMusicPlaying -> NoiseType.MUSIC
            metrics.isCrowded -> NoiseType.CROWD
            metrics.ambienceType == "quiet" -> NoiseType.QUIET
            metrics.hasSpeech -> NoiseType.SPEECH
            else -> NoiseType.AMBIENT
        }
        return NoiseProfile(
            ambientType = ambientType,
            noiseTier = metrics.ambientNoiseLevel.toInt(),
            snrEstimate = estimateSnr(metrics),
            isMusicPlaying = metrics.isMusicPlaying,
            isCrowded = metrics.isCrowded
        )
    }

    /**
     * Estimate signal-to-noise ratio based on current audio metrics.
     */
    private fun estimateSnr(metrics: AudioMetrics): Float {
        // SNR estimate based on noise tier instead of fake dB values
        val cfg = SensingConfig.get().audio
        return when (metrics.ambientNoiseLevel.toInt()) {
            NOISE_TIER_QUIET -> cfg.snrQuiet
            NOISE_TIER_LOW -> cfg.snrLow
            NOISE_TIER_MODERATE -> cfg.snrModerate
            NOISE_TIER_ELEVATED -> cfg.snrElevated
            NOISE_TIER_HIGH -> cfg.snrHigh
            NOISE_TIER_VERY_HIGH -> cfg.snrVeryHigh
            else -> cfg.snrDefault
        }
    }
}

/**
 * Noise environment classification for speech preprocessing.
 */
enum class NoiseType {
    QUIET,    // Low noise, no preprocessing needed
    SPEECH,   // Conversational background
    MUSIC,    // Background music — bandpass filter speech frequencies
    CROWD,    // Crowd noise — spectral subtraction
    AMBIENT   // General ambient — mild filtering
}

/**
 * Noise profile snapshot for speech preprocessing decisions.
 */
data class NoiseProfile(
    val ambientType: NoiseType = NoiseType.AMBIENT,
    val noiseTier: Int = 1,         // 0-5 ordinal (was fake dB)
    val snrEstimate: Float = 15f,
    val isMusicPlaying: Boolean = false,
    val isCrowded: Boolean = false
) {
    fun isNoisy(): Boolean = noiseTier >= 3 || isMusicPlaying || isCrowded
}

/**
 * Configuration for audio classification.
 */
data class AudioConfig(
    val enabled: Boolean = true,
    val classificationIntervalMs: Long = 1000,  // Classify every 1 second
    val maxResults: Int = 5,  // Top 5 classifications
    val scoreThreshold: Float = SensingConfig.get().audio.scoreThreshold  // Minimum confidence threshold
)

/**
 * Audio classification metrics with derived signals for retail media intelligence.
 */
data class AudioMetrics(
    val timestamp: Long = System.currentTimeMillis(),
    val dominantClass: String = "unknown",
    val confidence: Float = 0f,
    val ambientNoiseLevel: Float = 45f,  // Estimated dB level
    val isCrowded: Boolean = false,
    val isMusicPlaying: Boolean = false,
    val hasSpeech: Boolean = false,
    val ambienceType: String = "unknown",  // crowded, quiet, music_playing, conversational, ambient

    // Derived signals for retail media network
    val estimatedOccupancy: String = "0-5",  // "0-5", "5-20", "20-50", "50+"
    val adReceptivityScore: Float = SensingConfig.get().audio.defaultReceptivity,  // 0-1 (higher = more receptive to ads)
    val inferredVenueType: String = "unknown"  // retail, restaurant, office, entertainment, waiting_area
) {
    /**
     * Compute ad receptivity with viewer presence boost.
     * Breaks the 2-factor compression by adding a third axis (face count).
     * Called by the orchestrator when face count is available.
     */
    fun adReceptivityWithViewers(faceCount: Int): Float {
        val cfg = SensingConfig.get().audio
        val viewerBonus = if (faceCount > 0) cfg.activeViewerBoost else 0f
        return (adReceptivityScore + viewerBonus).coerceIn(0f, 1f)
    }
}
