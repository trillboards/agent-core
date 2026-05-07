package com.trillboards.ctv.core.audience

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.google.mlkit.vision.face.Face
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import com.trillboards.ctv.core.SensingConfig
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Facial emotion classification processor using FER TFLite model.
 *
 * Classifies emotions from face images:
 * - Happy (positive engagement)
 * - Surprised (attention captured)
 * - Neutral (passive viewing)
 * - Sad, Angry, Fearful, Disgusted (negative reactions)
 *
 * Model: fer_emotion.tflite (~300KB, FER-2013 trained)
 * Inference: ~20ms on mid-range devices
 *
 * Input: 48x48 grayscale face image
 * Output: 7 emotion probabilities
 *
 * Note: Uses ML Kit face detection output (Face bounding box) to crop
 * the face region before classification.
 */
class EmotionClassificationProcessor(
    private val context: Context,
    private val config: EmotionalEngagementConfig = EmotionalEngagementConfig()
) {
    companion object {
        private const val TAG = "EmotionClassification"

        // Model file
        private const val EMOTION_MODEL_FILE = "fer_emotion.tflite"

        // FER model expects 48x48 grayscale input
        private const val INPUT_SIZE = 48
        private const val PIXEL_SIZE = 1  // Grayscale
        private const val NUM_CLASSES = 8  // FER+ has 8 classes

        // Emotion labels (FER+ order - different from FER-2013)
        private val EMOTION_LABELS = arrayOf(
            EmotionType.NEUTRAL,    // 0
            EmotionType.HAPPY,      // 1 (happiness)
            EmotionType.SURPRISED,  // 2
            EmotionType.SAD,        // 3 (sadness)
            EmotionType.ANGRY,      // 4 (anger)
            EmotionType.DISGUSTED,  // 5 (disgust)
            EmotionType.FEARFUL,    // 6 (fear)
            EmotionType.CONTEMPT    // 7 (contempt — distinct from disgust)
        )

        // Minimum confidence to report emotion (fallback; runtime uses SensingConfig)
        private const val MIN_CONFIDENCE = 0.3f

        /**
         * Apply softmax to convert raw logits to probabilities (0-1 range, sum to 1).
         * FER+ model outputs raw logits, not probabilities.
         */
        private fun softmax(logits: FloatArray): FloatArray {
            val maxLogit = logits.maxOrNull() ?: 0f
            val expValues = FloatArray(logits.size) { i ->
                kotlin.math.exp((logits[i] - maxLogit).toDouble()).toFloat()
            }
            val sumExp = expValues.sum()
            return FloatArray(expValues.size) { i -> expValues[i] / sumExp }
        }
    }

    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    // Pre-allocated buffers
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: Array<FloatArray>? = null
    // Pre-allocated pixel buffer for face preprocessing
    private var pixelBuffer: IntArray? = null

    // Current metrics state
    private val _currentMetrics = MutableStateFlow(EmotionMetrics())
    val currentMetrics: StateFlow<EmotionMetrics> = _currentMetrics

    // Callback for when metrics are ready
    var onMetricsReady: ((EmotionMetrics) -> Unit)? = null

    /**
     * Check if the emotion model is available.
     */
    fun hasModel(): Boolean {
        return try {
            context.assets.open(EMOTION_MODEL_FILE).close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Emotion model not found: ${e.message}")
            false
        }
    }

    /**
     * Initialize the emotion classifier.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        if (!hasModel()) {
            Log.e(TAG, "Emotion model not available - cannot initialize")
            return false
        }

        return try {
            val model = loadModelFile(EMOTION_MODEL_FILE)

            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }

            interpreter = Interpreter(model, options)

            // Allocate input buffer (48 * 48 * 1 * 4 bytes for float)
            inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE * 4)
            inputBuffer?.order(ByteOrder.nativeOrder())

            // Allocate output buffer
            outputBuffer = Array(1) { FloatArray(NUM_CLASSES) }
            pixelBuffer = IntArray(INPUT_SIZE * INPUT_SIZE)

            isInitialized = true
            Log.i(TAG, "Emotion classifier initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize emotion classifier", e)
            false
        }
    }

    /**
     * Load TFLite model from assets.
     */
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Process a face from a bitmap and classify emotion.
     *
     * @param bitmap Full camera frame
     * @param face ML Kit Face with bounding box
     * @return EmotionMetrics with classification results
     */
    fun process(bitmap: Bitmap, face: Face): EmotionMetrics? {
        if (!isInitialized) {
            Log.w(TAG, "Not initialized - call initialize() first")
            return null
        }

        return try {
            // Crop face region from bitmap
            val faceBitmap = cropFace(bitmap, face) ?: return null

            // Preprocess to model input format
            preprocessFace(faceBitmap)

            // Run inference
            interpreter?.run(inputBuffer, outputBuffer)

            // Parse results
            parseResults(outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Emotion classification error", e)
            null
        }
    }

    /**
     * Process multiple faces and return the dominant emotion.
     */
    fun processMultiple(bitmap: Bitmap, faces: List<Face>): List<EmotionMetrics> {
        if (!isInitialized || faces.isEmpty()) return emptyList()

        return faces.mapNotNull { face ->
            process(bitmap, face)
        }
    }

    /**
     * Zero-allocation emotion classification using a pre-allocated crop bitmap.
     *
     * Uses Canvas.drawBitmap to crop+scale the face region directly into the
     * pre-allocated 48x48 bitmap — no Bitmap.createBitmap or createScaledBitmap.
     *
     * @param sourceBitmap Full camera frame (the shared bitmap)
     * @param face ML Kit Face with bounding box
     * @param preallocatedCrop Pre-allocated 48x48 ARGB_8888 bitmap (owned by caller, reused across frames)
     * @return EmotionMetrics with classification results, or null on failure
     */
    fun processWithPreallocatedCrop(sourceBitmap: Bitmap, face: Face, preallocatedCrop: Bitmap): EmotionMetrics? {
        if (!isInitialized) return null

        return try {
            val bounds = face.boundingBox

            // Add 20% padding around face
            val padding = (bounds.width() * 0.2f).toInt()
            val left = (bounds.left - padding).coerceAtLeast(0)
            val top = (bounds.top - padding).coerceAtLeast(0)
            val right = (bounds.right + padding).coerceAtMost(sourceBitmap.width)
            val bottom = (bounds.bottom + padding).coerceAtMost(sourceBitmap.height)

            val width = right - left
            val height = bottom - top
            if (width <= 0 || height <= 0) return null

            // Crop + scale into pre-allocated bitmap in one Canvas operation — zero allocation
            val srcRect = Rect(left, top, right, bottom)
            val dstRect = Rect(0, 0, INPUT_SIZE, INPUT_SIZE)
            val canvas = Canvas(preallocatedCrop)
            canvas.drawBitmap(sourceBitmap, srcRect, dstRect, null)

            // Preprocess the pre-allocated crop (already 48x48)
            preprocessFace(preallocatedCrop)

            // Run inference
            interpreter?.run(inputBuffer, outputBuffer)

            // Parse results
            parseResults(outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Emotion classification (preallocated) error", e)
            null
        }
    }

    /**
     * Zero-allocation multi-face emotion classification.
     *
     * @param sourceBitmap Full camera frame
     * @param faces List of ML Kit faces
     * @param preallocatedCrop Pre-allocated 48x48 bitmap (reused for each face sequentially)
     * @return List of EmotionMetrics for each face
     */
    fun processMultipleWithPreallocatedCrop(
        sourceBitmap: Bitmap,
        faces: List<Face>,
        preallocatedCrop: Bitmap
    ): List<EmotionMetrics> {
        if (!isInitialized || faces.isEmpty()) return emptyList()

        return faces.mapNotNull { face ->
            processWithPreallocatedCrop(sourceBitmap, face, preallocatedCrop)
        }
    }

    /**
     * Process asynchronously and emit results via callback.
     */
    fun processAsync(bitmap: Bitmap, faces: List<Face>) {
        if (!isInitialized || faces.isEmpty()) return

        processorScope.launch {
            val results = processMultiple(bitmap, faces)
            if (results.isNotEmpty()) {
                // Use the first face's emotion (primary person)
                _currentMetrics.value = results[0]
                onMetricsReady?.invoke(results[0])
            }
        }
    }

    /**
     * Crop the face region from the bitmap using ML Kit Face bounds.
     */
    private fun cropFace(bitmap: Bitmap, face: Face): Bitmap? {
        val bounds = face.boundingBox

        // Add padding around face (20%)
        val padding = (bounds.width() * 0.2f).toInt()
        val left = (bounds.left - padding).coerceAtLeast(0)
        val top = (bounds.top - padding).coerceAtLeast(0)
        val right = (bounds.right + padding).coerceAtMost(bitmap.width)
        val bottom = (bounds.bottom + padding).coerceAtMost(bitmap.height)

        val width = right - left
        val height = bottom - top

        if (width <= 0 || height <= 0) {
            Log.w(TAG, "Invalid face bounds")
            return null
        }

        return try {
            val cropped = Bitmap.createBitmap(bitmap, left, top, width, height)
            val scaled = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)
            if (cropped != scaled) cropped.recycle()
            scaled
        } catch (e: Exception) {
            Log.e(TAG, "Error cropping face", e)
            null
        }
    }

    /**
     * Preprocess face bitmap to model input format.
     * Converts to grayscale and normalizes to [-1, 1].
     */
    private fun preprocessFace(bitmap: Bitmap) {
        val buffer = inputBuffer ?: return
        buffer.rewind()

        // Convert bitmap to grayscale float array
        val pixels = pixelBuffer ?: return
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Convert RGB to grayscale
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            val gray = (0.299f * r + 0.587f * g + 0.114f * b)

            // Normalize to [-1, 1]
            val normalized = (gray / 127.5f) - 1f
            buffer.putFloat(normalized)
        }
    }

    /**
     * Parse model output into EmotionMetrics.
     */
    private fun parseResults(output: Array<FloatArray>?): EmotionMetrics {
        if (output == null || output.isEmpty()) {
            return EmotionMetrics(confidence = 0f)
        }

        // FER+ model outputs raw logits - convert to probabilities using softmax
        val rawLogits = output[0]
        val probabilities = softmax(rawLogits)

        // Find dominant emotion
        var maxIndex = 0
        var maxProb = 0f
        for (i in probabilities.indices) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIndex = i
            }
        }

        val dominantEmotion = if (maxIndex < EMOTION_LABELS.size) {
            EMOTION_LABELS[maxIndex]
        } else {
            EmotionType.UNKNOWN
        }

        // Get individual emotion scores (FER+ order)
        val neutralScore = probabilities.getOrElse(0) { 0f }
        val happyScore = probabilities.getOrElse(1) { 0f }
        val surprisedScore = probabilities.getOrElse(2) { 0f }
        val sadScore = probabilities.getOrElse(3) { 0f }
        val angryScore = probabilities.getOrElse(4) { 0f }
        val disgustScore = probabilities.getOrElse(5) { 0f }  // contempt (index 7) mapped via labels, not inflated here
        val fearScore = probabilities.getOrElse(6) { 0f }

        // Calculate positive/negative reaction
        val cfg = SensingConfig.get().emotion
        val positiveSum = happyScore + surprisedScore
        val negativeSum = angryScore + disgustScore + fearScore + sadScore

        val isPositiveReaction = positiveSum > cfg.positiveReactionThreshold && positiveSum > negativeSum
        val isNegativeReaction = negativeSum > cfg.negativeReactionThreshold && negativeSum > positiveSum

        // Calculate emotional engagement score
        // Higher emotion intensity = higher engagement (even negative)
        // Neutral = low engagement
        val emotionalIntensity = 1f - neutralScore
        val emotionalEngagementScore = emotionalIntensity * maxProb

        val metrics = EmotionMetrics(
            dominantEmotion = dominantEmotion,
            emotionConfidence = maxProb,
            happyScore = happyScore,
            surprisedScore = surprisedScore,
            neutralScore = neutralScore,
            sadScore = sadScore,
            angryScore = angryScore,
            fearScore = fearScore,
            disgustScore = disgustScore,
            isPositiveReaction = isPositiveReaction,
            isNegativeReaction = isNegativeReaction,
            emotionalEngagementScore = emotionalEngagementScore,
            confidence = maxProb
        )

        Log.d(TAG, "Emotion: $dominantEmotion (${String.format("%.2f", maxProb)}), " +
                "engagement=${String.format("%.2f", emotionalEngagementScore)}")

        return metrics
    }

    /**
     * Classify emotion from ML Kit smiling/eyes open probabilities (fallback).
     *
     * When FER model is not available, use ML Kit's built-in probabilities.
     */
    fun classifyFromMLKit(face: Face): EmotionMetrics {
        val smilingProb = face.smilingProbability ?: 0f
        val leftEyeOpenProb = face.leftEyeOpenProbability ?: 1f
        val rightEyeOpenProb = face.rightEyeOpenProbability ?: 1f
        val avgEyeOpen = (leftEyeOpenProb + rightEyeOpenProb) / 2f

        // Heuristic classification — conservative: default to NEUTRAL for ambiguous states
        // Previous version incorrectly classified resting faces as SAD or ANGRY
        val eCfg = SensingConfig.get().emotion
        val (emotion, confidence) = when {
            smilingProb > eCfg.happySmilingThreshold -> EmotionType.HAPPY to smilingProb
            smilingProb > eCfg.surprisedSmilingThreshold && avgEyeOpen > eCfg.surprisedEyeOpenThreshold -> EmotionType.SURPRISED to (smilingProb * 0.7f + avgEyeOpen * 0.3f)
            // Removed: avgEyeOpen < 0.3f -> SAD (blinking/squinting is not sadness)
            // Removed: smilingProb < 0.2f && avgEyeOpen > 0.9f -> ANGRY (resting face is not anger)
            else -> EmotionType.NEUTRAL to 0.6f  // Default to neutral for ambiguous states
        }

        val isPositive = emotion == EmotionType.HAPPY || emotion == EmotionType.SURPRISED
        val isNegative = emotion == EmotionType.SAD || emotion == EmotionType.ANGRY

        return EmotionMetrics(
            dominantEmotion = emotion,
            emotionConfidence = confidence,
            happyScore = smilingProb,
            neutralScore = 1f - smilingProb,
            isPositiveReaction = isPositive,
            isNegativeReaction = isNegative,
            emotionalEngagementScore = if (emotion == EmotionType.NEUTRAL) eCfg.neutralEngagementScore else confidence * eCfg.neutralEngagementFactor,
            confidence = confidence * eCfg.heuristicConfidenceFactor  // Lower confidence for heuristic-based classification
        )
    }

    /**
     * Release resources.
     */
    fun release() {
        try {
            interpreter?.close()
            interpreter = null
            inputBuffer = null
            outputBuffer = null
            pixelBuffer = null
            isInitialized = false
            Log.i(TAG, "Emotion classifier released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing emotion classifier", e)
        }
    }

    /**
     * Check if processor is ready.
     */
    fun isReady(): Boolean = isInitialized
}
