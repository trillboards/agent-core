package com.trillboards.ctv.core.audience

/**
 * Structured insights extracted from speech transcription.
 *
 * Privacy-first design:
 * - Raw transcript is NEVER stored or transmitted
 * - Only these structured signals are extracted and emitted
 * - Extraction happens on-device immediately after transcription
 * - Transcript is deleted within milliseconds of processing
 */
data class SpeechInsights(
    val timestamp: Long = System.currentTimeMillis(),

    // Product/Brand intelligence
    val brandMentions: List<String> = emptyList(),
    val brandMatches: List<BrandMatch> = emptyList(),
    val productCategories: List<String> = emptyList(),
    val shoppingContexts: List<String> = emptyList(),
    val competitorMentions: List<String> = emptyList(),

    // Purchase signals
    val priceInquiry: Boolean = false,
    val availabilityInquiry: Boolean = false,
    val purchaseJourney: PurchaseJourney = PurchaseJourney(),
    val mentionedBuyingToday: Boolean = false,

    // Objections & interests
    val objections: List<Objection> = emptyList(),
    val objectionInsights: List<ObjectionInsight> = emptyList(),
    val interests: List<String> = emptyList(),

    // Sentiment
    val sentimentTone: SentimentTone = SentimentTone.NEUTRAL,
    val emotionalArc: EmotionalArc = EmotionalArc.STABLE,

    // Conversation context
    val conversationType: ConversationType = ConversationType.UNKNOWN,
    val estimatedSpeakerCount: Int = 0,
    val classificationSource: String = "unknown",

    // Confidence score (0-1)
    val confidence: Float = 0f,

    // Canonical speech semantics for cross-platform "Sense Anything" contracts
    val speechSemantics: SpeechSemantics? = null
) {
    /**
     * Check if this contains any actionable purchase signals.
     *
     * Edge no longer infers purchase intent — that's cloud-Gemini territory
     * (`speech_purchase_intent` enum on the audience-metrics row). Edge still
     * surfaces price/availability/journey/buy-today signals from keyword
     * detection; the intent enum itself is read from the cloud response.
     */
    fun hasActionablePurchaseSignals(): Boolean {
        return priceInquiry ||
               availabilityInquiry ||
               mentionedBuyingToday ||
               purchaseJourney.stage != PurchaseJourney.NONE_STAGE
    }

    /**
     * Check if any brands were mentioned
     */
    fun hasBrandMentions(): Boolean = brandMentions.isNotEmpty()

    /**
     * Get the primary product category being discussed
     */
    fun primaryProductCategory(): String? = productCategories.firstOrNull()
}

data class SpeechSemantics(
    val entities: List<SpeechSemanticEntity> = emptyList(),
    val reasons: List<SpeechSemanticReason> = emptyList(),
    val questions: List<SpeechSemanticQuestion> = emptyList(),
    val journeyState: String? = null,
    val confidence: Float? = null,
    val evidencePhrases: List<String> = emptyList(),
    val brandMatches: List<BrandMatch> = emptyList(),
    val shoppingContexts: List<String> = emptyList(),
    val purchaseJourney: PurchaseJourney? = null,
    val objectionInsights: List<ObjectionInsight> = emptyList()
) {
    fun hasSignals(): Boolean {
        return entities.isNotEmpty() ||
            reasons.isNotEmpty() ||
            questions.isNotEmpty() ||
            !journeyState.isNullOrBlank() ||
            evidencePhrases.isNotEmpty() ||
            brandMatches.isNotEmpty() ||
            shoppingContexts.isNotEmpty() ||
            purchaseJourney != null ||
            objectionInsights.isNotEmpty()
    }
}

data class SpeechSemanticEntity(
    val name: String,
    val type: String? = null,
    val role: String? = null,
    val polarity: String? = null
)

data class SpeechSemanticReason(
    val target: String? = null,
    val type: String? = null,
    val detail: String? = null
)

data class SpeechSemanticQuestion(
    val type: String? = null,
    val text: String? = null
)

data class BrandMatch(
    val canonicalName: String,
    val observedText: String,
    val confidence: Float = 1f,
    val matchType: String = "exact"
)

data class PurchaseJourney(
    val stage: String = NONE_STAGE,
    val urgency: String = "none",
    val journeySignals: List<String> = emptyList(),
    val evidencePhrases: List<String> = emptyList()
) {
    companion object {
        const val NONE_STAGE = "none"
    }
}

data class ObjectionInsight(
    val theme: String,
    val summaryLabel: String? = null,
    val evidencePhrase: String? = null,
    val confidence: Float = 0.5f
)

// PurchaseIntent enum + toSpeechSemanticRole extension removed —
// edge no longer infers intent (regex/keyword heuristics deleted in
// cosmic-brewing-bear task #26). The cloud Gemini response carries the
// canonical 7-state speech_purchase_intent enum (BROWSING / RESEARCHING /
// COMPARING / READY_TO_BUY / POST_PURCHASE / COMPLAINING / NONE) per
// audienceVisionService.js#_assembleProfilePrompt.

/**
 * Sentiment detected from conversation tone
 */
enum class SentimentTone {
    NEGATIVE,   // Frustration, disappointment, anger
    NEUTRAL,    // Normal conversation
    POSITIVE,   // Happy, satisfied
    EXCITED     // High energy, enthusiasm
}

/**
 * Emotional arc across the conversation window
 */
enum class EmotionalArc {
    DECLINING,              // Getting more negative over time
    STABLE,                 // No significant change
    IMPROVING,              // Getting more positive
    SKEPTICAL_TO_CONVINCED  // Started doubtful, now positive
}

/**
 * Type of conversation detected
 */
enum class ConversationType {
    UNKNOWN,            // Can't determine
    CASUAL,             // General chat, not shopping-related
    PRODUCT_INQUIRY,    // Asking about products
    PRICE_NEGOTIATION,  // Discussing price/deals
    SALES_PITCH,        // Sales associate explaining features
    COMPLAINT,          // Customer expressing issues
    PURCHASE            // Actively making a purchase
}

/**
 * Customer objections detected in conversation
 */
enum class Objection(val keywords: List<String>) {
    TOO_EXPENSIVE(listOf("too expensive", "too much", "can't afford", "out of budget", "cheaper")),
    WRONG_COLOR(listOf("different color", "other color", "don't like the color")),
    WRONG_SIZE(listOf("different size", "too big", "too small", "doesn't fit")),
    NEED_TO_THINK(listOf("think about it", "come back", "not sure yet", "let me think")),
    COMPARING_ELSEWHERE(listOf("check other", "look elsewhere", "amazon", "online")),
    QUALITY_CONCERN(listOf("is it good", "will it last", "quality", "durable")),
    WARRANTY_CONCERN(listOf("warranty", "return policy", "guarantee")),
    NOT_NOW(listOf("not today", "another time", "just browsing", "just looking"))
}

/**
 * Configuration for speech intelligence processing
 */
data class SpeechConfig(
    val enabled: Boolean = true,
    val transcriptionIntervalMs: Long = 10_000,  // Transcribe every 10 seconds
    val audioBufferLengthMs: Long = 30_000,      // Keep 30 seconds of audio
    val minConfidenceThreshold: Float = 0.3f,    // Minimum confidence to emit
    val language: String = "en"                   // Language for transcription
)
