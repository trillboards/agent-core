package com.trillboards.ctv.core.audience

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.text.textclassifier.TextClassifier
import com.google.mediapipe.tasks.text.textclassifier.TextClassifierResult

/**
 * On-device text classifier using MediaPipe.
 *
 * Classifies transcripts into retail intent categories:
 * - PURCHASE_INTENT: "I want to buy this", "I'll take it"
 * - PRICE_INQUIRY: "How much is this?", "What's the price?"
 * - PRODUCT_INTEREST: "Tell me about this", "What does it do?"
 * - COMPARISON: "Which one is better?", "Difference between"
 * - BROWSING: "Just looking", "Looking around"
 * - NEGATIVE: "Too expensive", "Not interested"
 * - NEUTRAL: General conversation, no clear intent
 *
 * Runs 100% on-device with 50-150ms latency.
 * Privacy: No transcript data leaves the device.
 */
class MediaPipeTextClassifier(private val context: Context) {
    companion object {
        private const val TAG = "MediaPipeTextClassifier"

        // Model file in assets
        private const val INTENT_MODEL_FILE = "intent_classifier.tflite"

        // Minimum confidence to consider a classification valid
        private const val MIN_CONFIDENCE = 0.4f

        // Intent label mappings (match training labels)
        private val INTENT_LABELS = mapOf(
            "PURCHASE_INTENT" to IntentType.PURCHASE_INTENT,
            "PRICE_INQUIRY" to IntentType.PRICE_INQUIRY,
            "PRODUCT_INTEREST" to IntentType.PRODUCT_INTEREST,
            "COMPARISON" to IntentType.COMPARISON,
            "BROWSING" to IntentType.BROWSING,
            "NEGATIVE" to IntentType.NEGATIVE,
            "NEUTRAL" to IntentType.NEUTRAL
        )
    }

    private var textClassifier: TextClassifier? = null
    private var isInitialized = false

    /**
     * Check if the intent classifier model is available.
     */
    fun hasModel(): Boolean {
        return try {
            context.assets.open(INTENT_MODEL_FILE).close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Intent classifier model not found: ${e.message}")
            false
        }
    }

    /**
     * Initialize the MediaPipe text classifier.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        if (!hasModel()) {
            Log.e(TAG, "Intent classifier model not available")
            return false
        }

        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(INTENT_MODEL_FILE)
                .build()

            val options = TextClassifier.TextClassifierOptions.builder()
                .setBaseOptions(baseOptions)
                .setMaxResults(3) // Get top 3 results for fallback
                .build()

            textClassifier = TextClassifier.createFromOptions(context, options)
            isInitialized = true

            Log.i(TAG, "MediaPipe text classifier initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe text classifier", e)
            false
        }
    }

    /**
     * Classify a transcript and return structured result.
     *
     * @param transcript The speech transcript to classify
     * @return ClassificationResult with intent and confidence
     */
    fun classify(transcript: String): ClassificationResult {
        val startTime = System.currentTimeMillis()

        if (!isInitialized) {
            Log.w(TAG, "Classifier not initialized")
            return ClassificationResult(
                intentType = IntentType.UNKNOWN,
                confidence = 0f,
                latencyMs = 0,
                source = ClassificationSource.ON_DEVICE
            )
        }

        val normalizedText = transcript.trim().lowercase()
        if (normalizedText.isEmpty() || normalizedText.length < 3) {
            return ClassificationResult(
                intentType = IntentType.NEUTRAL,
                confidence = 0.5f,
                latencyMs = System.currentTimeMillis() - startTime,
                source = ClassificationSource.ON_DEVICE
            )
        }

        return try {
            val result: TextClassifierResult = textClassifier!!.classify(normalizedText)
            val latencyMs = System.currentTimeMillis() - startTime

            val topCategory = result.classificationResult()
                .classifications()
                .firstOrNull()
                ?.categories()
                ?.maxByOrNull { it.score() }

            if (topCategory == null || topCategory.score() < MIN_CONFIDENCE) {
                Log.d(TAG, "Low confidence or no result: ${topCategory?.score()}")
                return ClassificationResult(
                    intentType = IntentType.NEUTRAL,
                    confidence = topCategory?.score() ?: 0f,
                    latencyMs = latencyMs,
                    source = ClassificationSource.ON_DEVICE
                )
            }

            val intentType = INTENT_LABELS[topCategory.categoryName()] ?: IntentType.NEUTRAL

            Log.d(TAG, "Classification: ${topCategory.categoryName()} " +
                    "(${String.format("%.2f", topCategory.score())}) in ${latencyMs}ms")

            ClassificationResult(
                intentType = intentType,
                confidence = topCategory.score(),
                latencyMs = latencyMs,
                source = ClassificationSource.ON_DEVICE,
                rawLabel = topCategory.categoryName()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Classification failed", e)
            ClassificationResult(
                intentType = IntentType.UNKNOWN,
                confidence = 0f,
                latencyMs = System.currentTimeMillis() - startTime,
                source = ClassificationSource.ON_DEVICE,
                error = e.message
            )
        }
    }

    /**
     * Close the classifier and release resources.
     */
    fun close() {
        try {
            textClassifier?.close()
            textClassifier = null
            isInitialized = false
            Log.i(TAG, "MediaPipe text classifier closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing classifier", e)
        }
    }

    /**
     * Check if classifier is ready.
     */
    fun isReady(): Boolean = isInitialized
}

/**
 * Intent type from classification.
 */
enum class IntentType {
    PURCHASE_INTENT,    // Strong buying signal
    PRICE_INQUIRY,      // Asking about price
    PRODUCT_INTEREST,   // Interested in learning more
    COMPARISON,         // Comparing options
    BROWSING,           // Just looking
    NEGATIVE,           // Disinterest or rejection
    NEUTRAL,            // General conversation
    UNKNOWN             // Classification failed
}

/**
 * Source of classification result.
 */
enum class ClassificationSource {
    ON_DEVICE,      // MediaPipe on-device
    SERVER_GEMINI,  // Server-side Gemini
    KEYWORD_FALLBACK // Legacy keyword matching
}

/**
 * Result of intent classification.
 */
data class ClassificationResult(
    val intentType: IntentType,
    val confidence: Float,
    val latencyMs: Long,
    val source: ClassificationSource,
    val rawLabel: String? = null,
    val error: String? = null,

    // Extended fields for Gemini results
    val brands: List<String> = emptyList(),
    val products: List<String> = emptyList(),
    val sentiment: String = "NEUTRAL",
    val priceContext: PriceContext? = null
) {
    /**
     * Convert to SpeechInsights for downstream consumption.
     */
    fun toSpeechInsights(): SpeechInsights {
        val purchaseIntent = when (intentType) {
            IntentType.PURCHASE_INTENT -> PurchaseIntent.READY_TO_BUY
            IntentType.COMPARISON -> PurchaseIntent.COMPARING
            IntentType.PRODUCT_INTEREST -> PurchaseIntent.CONSIDERING
            IntentType.BROWSING -> PurchaseIntent.BROWSING
            IntentType.PRICE_INQUIRY -> PurchaseIntent.CONSIDERING
            else -> PurchaseIntent.NONE
        }

        val sentimentTone = when (sentiment.uppercase()) {
            "POSITIVE" -> SentimentTone.POSITIVE
            "NEGATIVE" -> SentimentTone.NEGATIVE
            "EXCITED" -> SentimentTone.EXCITED
            else -> SentimentTone.NEUTRAL
        }

        val objections = if (intentType == IntentType.NEGATIVE && priceContext?.sensitivity == "HIGH") {
            listOf(Objection.TOO_EXPENSIVE)
        } else {
            emptyList()
        }

        val conversationType = when (intentType) {
            IntentType.PURCHASE_INTENT -> ConversationType.PURCHASE
            IntentType.PRICE_INQUIRY -> ConversationType.PRICE_NEGOTIATION
            IntentType.PRODUCT_INTEREST -> ConversationType.PRODUCT_INQUIRY
            IntentType.COMPARISON -> ConversationType.PRODUCT_INQUIRY
            IntentType.BROWSING -> ConversationType.CASUAL
            IntentType.NEGATIVE -> ConversationType.COMPLAINT
            else -> ConversationType.UNKNOWN
        }

        val speechSemantics = SpeechSemantics(
            entities = brands.distinct().map { brand ->
                SpeechSemanticEntity(
                    name = brand,
                    type = "brand",
                    role = purchaseIntent.toSpeechSemanticRole()
                )
            },
            questions = buildList {
                if (intentType == IntentType.PRICE_INQUIRY || priceContext?.mentioned == true) {
                    add(SpeechSemanticQuestion(type = "price_inquiry", text = "price inquiry"))
                }
            },
            journeyState = when (purchaseIntent) {
                PurchaseIntent.READY_TO_BUY -> "ready_to_buy"
                PurchaseIntent.COMPARING -> "active_comparison"
                PurchaseIntent.CONSIDERING -> "active_consideration"
                PurchaseIntent.BROWSING -> "casual_browsing"
                PurchaseIntent.NONE -> null
            },
            confidence = confidence
        ).takeIf { it.hasSignals() }

        return SpeechInsights(
            brandMentions = brands,
            productCategories = products,
            priceInquiry = intentType == IntentType.PRICE_INQUIRY || priceContext?.mentioned == true,
            purchaseIntent = purchaseIntent,
            mentionedBuyingToday = intentType == IntentType.PURCHASE_INTENT,
            objections = objections,
            sentimentTone = sentimentTone,
            conversationType = conversationType,
            classificationSource = source.name.lowercase(),
            confidence = confidence,
            speechSemantics = speechSemantics
        )
    }

    /**
     * Check if this is a valid, usable result.
     */
    fun isValid(): Boolean = intentType != IntentType.UNKNOWN && error == null
}

/**
 * Price context from Gemini analysis.
 */
data class PriceContext(
    val mentioned: Boolean = false,
    val sensitivity: String = "NONE" // HIGH, MEDIUM, LOW, NONE
)
