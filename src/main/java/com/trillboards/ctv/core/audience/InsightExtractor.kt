package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.SensingConfig

/**
 * On-device extraction of structured insights from transcripts.
 *
 * This runs entirely on-device - no transcript data is transmitted.
 * Uses keyword matching, pattern detection, and heuristics to extract
 * actionable signals from conversations.
 */
object InsightExtractor {
    private data class PhraseMatch(val label: String, val phrase: String)

    private data class ObjectionThemePattern(
        val theme: String,
        val summary: Objection?,
        val patterns: List<String>
    )

    // ========== BRAND PATTERNS ==========
    // Popular brands that might be mentioned in retail contexts
    // This list can be dynamically extended per-venue from server
    private val BRAND_PATTERNS = setOf(
        // Electronics
        "apple", "samsung", "sony", "lg", "bose", "beats", "jbl", "google", "microsoft",
        "dell", "hp", "lenovo", "asus", "acer", "nvidia", "intel", "amd",
        // Fashion
        "nike", "adidas", "puma", "reebok", "under armour", "lululemon", "north face",
        "patagonia", "zara", "h&m", "uniqlo", "gap", "levi's", "gucci", "louis vuitton",
        // Food & Beverage
        "starbucks", "mcdonald's", "chipotle", "subway", "dunkin", "coca-cola", "pepsi",
        // Retail
        "amazon", "walmart", "target", "costco", "best buy", "home depot", "ikea",
        // Beauty
        "sephora", "ulta", "mac", "maybelline", "l'oreal", "revlon"
    )

    // ========== PRODUCT CATEGORY PATTERNS ==========
    private val PRODUCT_PATTERNS = mapOf(
        "electronics" to setOf(
            "phone", "iphone", "android", "laptop", "computer", "tablet", "ipad",
            "tv", "television", "headphones", "earbuds", "airpods", "speaker",
            "camera", "watch", "smartwatch", "gaming", "console", "playstation", "xbox"
        ),
        "fashion" to setOf(
            "shirt", "t-shirt", "pants", "jeans", "dress", "skirt", "jacket", "coat",
            "shoes", "sneakers", "boots", "sandals", "hat", "cap", "bag", "purse",
            "wallet", "sunglasses", "jewelry", "watch", "bracelet", "necklace"
        ),
        "food" to setOf(
            "coffee", "latte", "espresso", "tea", "smoothie", "juice",
            "sandwich", "salad", "pizza", "burger", "fries", "chicken",
            "breakfast", "lunch", "dinner", "snack", "dessert"
        ),
        "beauty" to setOf(
            "makeup", "lipstick", "foundation", "mascara", "eyeshadow",
            "skincare", "moisturizer", "serum", "cleanser", "sunscreen",
            "perfume", "cologne", "shampoo", "conditioner"
        ),
        "home" to setOf(
            "furniture", "sofa", "couch", "chair", "table", "desk", "bed",
            "mattress", "pillow", "sheets", "towel", "rug", "lamp", "decor"
        ),
        "sports" to setOf(
            "running", "gym", "workout", "yoga", "fitness", "weights",
            "bike", "bicycle", "golf", "tennis", "basketball", "football"
        )
    )

    private val SHOPPING_CONTEXT_PATTERNS = mapOf(
        "gift_mission" to listOf(
            "gift for", "present for", "birthday gift", "need a gift", "gift idea"
        ),
        "occasion_birthday" to listOf(
            "birthday", "birthday gift", "birthday present"
        ),
        "recipient_daughter" to listOf(
            "for my daughter", "my daughter"
        ),
        "recipient_son" to listOf(
            "for my son", "my son"
        ),
        "social_approval" to listOf(
            "my wife would kill me",
            "my husband would kill me",
            "my partner would kill me",
            "need to ask my wife",
            "need to ask my husband",
            "need to ask my partner",
            "need to check with my wife",
            "need to check with my husband",
            "need to check with my partner"
        ),
        "repeat_consideration" to listOf(
            "third time i've looked",
            "third time i have looked",
            "keep coming back",
            "came back for this",
            "looked at this again",
            "still thinking about this"
        ),
        "budget_guardrail" to listOf(
            "out of budget",
            "can't afford",
            "too expensive",
            "more than i want to spend",
            "more than i should spend"
        )
    )

    // ========== PURCHASE INTENT SIGNALS ==========
    private val STRONG_BUYING_SIGNALS = listOf(
        "i'll take", "we'll take", "i want this", "i'll get", "let me get",
        "ring this up", "i'm buying", "add to cart", "check out",
        "want to buy", "gonna buy", "going to buy", "need to buy",
        "should buy", "will buy", "buying this", "purchase"
    )

    private val MEDIUM_BUYING_SIGNALS = listOf(
        "how much", "what's the price", "price on this", "cost",
        "do you have", "is this available", "in stock", "can i get",
        "looking for", "i need", "i'm interested", "want to get",
        "should get", "thinking about", "considering", "might get",
        "could use", "need new", "looking at", "checking out"
    )

    private val COMPARING_SIGNALS = listOf(
        "which one", "what's the difference", "compare", "versus",
        "better than", "or this one", "both of them", "between these"
    )

    private val BROWSING_SIGNALS = listOf(
        "just looking", "just browsing", "looking around", "checking out"
    )

    // ========== INTEREST PATTERNS ==========
    private val INTEREST_PATTERNS = mapOf(
        "battery_life" to listOf("battery", "charge", "how long does it last"),
        "warranty" to listOf("warranty", "guarantee", "return policy", "if it breaks"),
        "durability" to listOf("durable", "sturdy", "will it last", "quality"),
        "compatibility" to listOf("compatible", "work with", "connect to"),
        "features" to listOf("what can it do", "features", "does it have"),
        "reviews" to listOf("reviews", "ratings", "what do people say"),
        "deals" to listOf("deal", "discount", "sale", "coupon", "promo")
    )

    private val OBJECTION_THEME_PATTERNS = listOf(
        ObjectionThemePattern(
            theme = "budget_pressure",
            summary = Objection.TOO_EXPENSIVE,
            patterns = listOf(
                "too expensive", "out of budget", "can't afford", "more than i want to spend"
            )
        ),
        ObjectionThemePattern(
            theme = "social_approval",
            summary = Objection.NEED_TO_THINK,
            patterns = SHOPPING_CONTEXT_PATTERNS.getValue("social_approval")
        ),
        ObjectionThemePattern(
            theme = "decision_deferral",
            summary = Objection.NEED_TO_THINK,
            patterns = listOf(
                "sleep on it", "need to think about it", "let me think", "not sure yet"
            )
        )
    )

    // ========== SENTIMENT PATTERNS ==========
    private val POSITIVE_PATTERNS = listOf(
        "love", "amazing", "great", "perfect", "exactly what", "beautiful",
        "awesome", "fantastic", "excellent", "wonderful"
    )

    private val NEGATIVE_PATTERNS = listOf(
        "hate", "terrible", "awful", "disappointing", "waste", "broken",
        "doesn't work", "not working", "frustrated", "annoyed"
    )

    private val EXCITED_PATTERNS = listOf(
        "oh my god", "omg", "wow", "can't believe", "so excited",
        "finally", "yes!", "i've been waiting"
    )

    // Additional brand list that can be loaded from server
    private var dynamicBrands: Set<String> = emptySet()

    // Competitor brands for this venue (from venues.competitor_brands JSONB)
    private var competitorBrands: Set<String> = emptySet()
    private var competitorPatterns: Map<String, Regex>? = null

    // Emotional arc tracking across multiple aggregation windows
    private val emotionHistory = ArrayDeque<SentimentTone>(5) // last 3-5 windows

    /**
     * Set additional brands to detect (loaded from server per-venue)
     */
    fun setDynamicBrands(brands: List<String>) {
        dynamicBrands = brands.map { it.lowercase() }.toSet()
    }

    /**
     * Set competitor brands for this venue.
     * Called when venue config is received from server (from venues.competitor_brands JSONB).
     */
    fun setCompetitorBrands(brands: List<String>) {
        competitorBrands = brands.map { it.lowercase() }.toSet()
        // Pre-compile regex patterns for competitor matching
        competitorPatterns = competitorBrands.associateWith { brand ->
            Regex("\\b${Regex.escape(brand)}\\b", RegexOption.IGNORE_CASE)
        }
    }

    /**
     * Extract competitor mentions from transcript using the venue's competitor list.
     */
    private fun extractCompetitorMentions(text: String): List<String> {
        val patterns = competitorPatterns ?: return emptyList()
        return patterns.entries
            .filter { (_, pattern) -> pattern.containsMatchIn(text) }
            .map { (brand, _) -> brand }
    }

    /**
     * Determine emotional arc across recent aggregation windows.
     * Compares sentiment progression: neutral→happy = IMPROVING, happy→sad = DECLINING, etc.
     */
    private fun computeEmotionalArc(currentSentiment: SentimentTone): EmotionalArc {
        emotionHistory.addLast(currentSentiment)
        while (emotionHistory.size > 5) emotionHistory.removeFirst()

        if (emotionHistory.size < 2) return EmotionalArc.STABLE

        val sentimentValues = emotionHistory.map { sentimentToValue(it) }
        val first = sentimentValues.first()
        val last = sentimentValues.last()
        val delta = last - first

        val cfg = SensingConfig.get().insight
        return when {
            // Started negative/neutral, now positive
            first <= 0 && last > cfg.skepticalConvincedThreshold -> EmotionalArc.SKEPTICAL_TO_CONVINCED
            // Trending up
            delta > cfg.sentimentImprovementThreshold -> EmotionalArc.IMPROVING
            // Trending down
            delta < -cfg.sentimentDeclineThreshold -> EmotionalArc.DECLINING
            // No significant change
            else -> EmotionalArc.STABLE
        }
    }

    private fun sentimentToValue(tone: SentimentTone): Float = when (tone) {
        SentimentTone.NEGATIVE -> -1f
        SentimentTone.NEUTRAL -> 0f
        SentimentTone.POSITIVE -> 0.5f
        SentimentTone.EXCITED -> 1f
    }

    /**
     * Extract all insights from a transcript.
     * This is the main entry point - call this with the raw transcript,
     * then IMMEDIATELY delete the transcript.
     */
    fun extract(transcript: String): SpeechInsights {
        val normalizedText = transcript.lowercase().trim()

        if (normalizedText.isEmpty()) {
            return SpeechInsights(confidence = 0f)
        }

        // Extract all signals
        val brandMatches = extractBrandMatches(normalizedText)
        val brands = brandMatches.map { it.canonicalName }.distinct()
        val products = extractProductCategories(normalizedText)
        val shoppingContextHits = extractShoppingContextHits(normalizedText)
        val shoppingContexts = shoppingContextHits.map { it.label }.distinct()
        val baseIntent = detectPurchaseIntent(normalizedText)
        val objectionInsights = extractObjectionInsights(normalizedText, shoppingContextHits)
        val intent = enrichPurchaseIntent(baseIntent, shoppingContexts, objectionInsights)
        val objections = extractObjections(normalizedText, objectionInsights)
        val interests = extractInterests(normalizedText)
        val sentiment = detectSentiment(normalizedText)
        val conversationType = classifyConversation(normalizedText, intent, objections)
        val speakerCount = estimateSpeakerCount(normalizedText)
        val purchaseJourney = buildPurchaseJourney(
            text = normalizedText,
            intent = intent,
            contextHits = shoppingContextHits,
            objectionInsights = objectionInsights
        )

        // Calculate confidence based on signal strength
        val confidence = calculateConfidence(
            brands, products, intent, objections, interests
        )
        val speechSemantics = buildSpeechSemantics(
            intent = intent,
            brandMatches = brandMatches,
            shoppingContexts = shoppingContexts,
            purchaseJourney = purchaseJourney,
            objectionInsights = objectionInsights,
            confidence = confidence,
            priceInquiry = containsAny(normalizedText, listOf("how much", "price", "cost")),
            availabilityInquiry = containsAny(normalizedText, listOf("do you have", "in stock", "available"))
        )

        return SpeechInsights(
            brandMentions = brands,
            brandMatches = brandMatches,
            productCategories = products,
            shoppingContexts = shoppingContexts,
            competitorMentions = extractCompetitorMentions(normalizedText),
            priceInquiry = containsAny(normalizedText, listOf("how much", "price", "cost")),
            availabilityInquiry = containsAny(normalizedText, listOf("do you have", "in stock", "available")),
            purchaseIntent = intent,
            purchaseJourney = purchaseJourney,
            mentionedBuyingToday = containsAny(normalizedText, listOf("today", "right now", "i'll take")),
            objections = objections,
            objectionInsights = objectionInsights,
            interests = interests,
            sentimentTone = sentiment.first,
            emotionalArc = computeEmotionalArc(sentiment.first),
            conversationType = conversationType,
            estimatedSpeakerCount = speakerCount,
            classificationSource = ClassificationSource.KEYWORD_FALLBACK.name.lowercase(),
            confidence = confidence,
            speechSemantics = speechSemantics
        )
    }

    /**
     * Extract brand mentions from transcript
     */
    // Pre-compiled word boundary patterns for brand matching (avoids substring false positives)
    private var brandPatterns: Map<String, Regex>? = null
    private var lastBrandSet: Set<String>? = null

    private fun getBrandPatterns(allBrands: Set<String>): Map<String, Regex> {
        // Rebuild patterns only when brand set changes
        if (allBrands == lastBrandSet && brandPatterns != null) {
            return brandPatterns!!
        }
        lastBrandSet = allBrands
        brandPatterns = allBrands.associateWith { brand ->
            Regex("\\b${Regex.escape(brand)}\\b", RegexOption.IGNORE_CASE)
        }
        return brandPatterns!!
    }

    private fun extractBrandMatches(text: String): List<BrandMatch> {
        val allBrands = BRAND_PATTERNS + dynamicBrands
        val patterns = getBrandPatterns(allBrands)
        val exactMatches = patterns.entries.flatMap { (brand, pattern) ->
            pattern.findAll(text).map { match ->
                BrandMatch(
                    canonicalName = brand,
                    observedText = match.value,
                    confidence = 1f,
                    matchType = "exact"
                )
            }.toList()
        }

        val exactBrands = exactMatches.map { it.canonicalName }.toSet()
        val fuzzyMatches = allBrands
            .asSequence()
            .filterNot { it in exactBrands }
            .mapNotNull { brand -> findFuzzyBrandMatch(text, brand) }
            .toList()

        return (exactMatches + fuzzyMatches)
            .distinctBy { "${it.canonicalName}:${it.observedText}:${it.matchType}" }
    }

    private fun findFuzzyBrandMatch(text: String, brand: String): BrandMatch? {
        val brandWords = normalizeBrandKey(brand).split(" ").filter { it.isNotBlank() }
        if (brandWords.isEmpty()) return null

        val candidates = buildCandidatePhrases(text, brandWords.size)
        val brandCompact = brandWords.joinToString("")
        val brandSoundex = soundex(brandCompact)
        var bestMatch: BrandMatch? = null

        for (candidate in candidates) {
            val candidateCompact = normalizeBrandCompact(candidate)
            if (candidateCompact.length < 4) continue
            if (brandCompact.firstOrNull() != candidateCompact.firstOrNull()) continue
            if (kotlin.math.abs(candidateCompact.length - brandCompact.length) > 3) continue

            val similarity = similarityScore(brandCompact, candidateCompact)
            val candidateSoundex = soundex(candidateCompact)
            val requiredScore = if (brand in dynamicBrands) 0.64f else 0.7f
            val soundexCompatible = brandSoundex.isNotEmpty() && brandSoundex == candidateSoundex
            val accepted = similarity >= requiredScore || (soundexCompatible && similarity >= 0.58f)
            if (!accepted) continue

            val matchConfidence = (if (soundexCompatible) similarity + 0.08f else similarity)
                .coerceAtMost(0.95f)

            if (bestMatch == null || matchConfidence > bestMatch.confidence) {
                bestMatch = BrandMatch(
                    canonicalName = brand,
                    observedText = candidate,
                    confidence = matchConfidence,
                    matchType = "fuzzy"
                )
            }
        }

        return bestMatch
    }

    private fun buildCandidatePhrases(text: String, targetWordCount: Int): List<String> {
        val tokens = Regex("[a-z0-9']+").findAll(text).map { it.value }.toList()
        if (tokens.isEmpty()) return emptyList()

        val windowSizes = listOf(
            targetWordCount,
            (targetWordCount - 1).coerceAtLeast(1),
            (targetWordCount + 1).coerceAtMost(3)
        ).distinct()

        val candidates = LinkedHashSet<String>()
        for (windowSize in windowSizes) {
            if (windowSize <= 0 || windowSize > tokens.size) continue
            for (start in 0..tokens.size - windowSize) {
                candidates += tokens.subList(start, start + windowSize).joinToString(" ")
            }
        }
        return candidates.toList()
    }

    private fun normalizeBrandKey(value: String): String {
        return value.lowercase()
            .replace("&", "and")
            .replace(Regex("[^a-z0-9]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")
    }

    private fun normalizeBrandCompact(value: String): String {
        return normalizeBrandKey(value).replace(" ", "")
    }

    private fun similarityScore(left: String, right: String): Float {
        if (left == right) return 1f
        val maxLength = maxOf(left.length, right.length).coerceAtLeast(1)
        val distance = levenshteinDistance(left, right)
        return 1f - (distance.toFloat() / maxLength.toFloat())
    }

    private fun levenshteinDistance(left: String, right: String): Int {
        if (left == right) return 0
        if (left.isEmpty()) return right.length
        if (right.isEmpty()) return left.length

        val costs = IntArray(right.length + 1) { it }
        for (i in left.indices) {
            var previous = i
            costs[0] = i + 1
            for (j in right.indices) {
                val current = costs[j + 1]
                val substitution = if (left[i] == right[j]) previous else previous + 1
                costs[j + 1] = minOf(
                    costs[j + 1] + 1,
                    costs[j] + 1,
                    substitution
                )
                previous = current
            }
        }
        return costs[right.length]
    }

    private fun soundex(value: String): String {
        if (value.isBlank()) return ""

        val letters = value.filter { it.isLetter() }.uppercase()
        if (letters.isEmpty()) return ""

        val codes = mapOf(
            'B' to '1', 'F' to '1', 'P' to '1', 'V' to '1',
            'C' to '2', 'G' to '2', 'J' to '2', 'K' to '2', 'Q' to '2', 'S' to '2', 'X' to '2', 'Z' to '2',
            'D' to '3', 'T' to '3',
            'L' to '4',
            'M' to '5', 'N' to '5',
            'R' to '6'
        )

        val encoded = StringBuilder().append(letters.first())
        var previousCode = codes[letters.first()]

        for (char in letters.drop(1)) {
            val currentCode = codes[char]
            if (currentCode != null && currentCode != previousCode) {
                encoded.append(currentCode)
            }
            previousCode = currentCode
        }

        return encoded.toString().padEnd(4, '0').take(4)
    }

    /**
     * Extract product categories being discussed
     */
    private fun extractProductCategories(text: String): List<String> {
        return PRODUCT_PATTERNS.entries
            .filter { (_, keywords) -> keywords.any { text.contains(it) } }
            .map { it.key }
    }

    private fun extractShoppingContextHits(text: String): List<PhraseMatch> {
        return SHOPPING_CONTEXT_PATTERNS.entries.flatMap { (label, patterns) ->
            patterns.filter { text.contains(it) }.map { phrase ->
                PhraseMatch(label = label, phrase = phrase)
            }
        }
    }

    /**
     * Check if a buying signal is negated by words like "not", "don't", "won't", "never"
     * within 3 words before the signal keyword.
     */
    private fun isNegated(text: String, signalPhrase: String): Boolean {
        val negationWords = listOf("not", "don't", "doesn't", "won't", "never", "no", "neither", "nor", "can't", "couldn't", "wouldn't", "shouldn't")
        val phraseIndex = text.indexOf(signalPhrase)
        if (phraseIndex < 0) return false

        // Get the 30 characters before the signal phrase (roughly 3-5 words)
        val contextStart = maxOf(0, phraseIndex - 30)
        val contextBefore = text.substring(contextStart, phraseIndex)
        val wordsBefore = contextBefore.trim().split("\\s+".toRegex())

        // Check last 3 words for negation
        val recentWords = wordsBefore.takeLast(3)
        return recentWords.any { word -> negationWords.any { neg -> word.contains(neg) } }
    }

    /**
     * Detect purchase intent level
     */
    private fun detectPurchaseIntent(text: String): PurchaseIntent {
        // Check strong signals first, but verify they're not negated
        val hasStrongSignal = STRONG_BUYING_SIGNALS.any { text.contains(it) }
        val strongNegated = hasStrongSignal && STRONG_BUYING_SIGNALS.filter { text.contains(it) }
            .all { isNegated(text, it) }

        val hasMediumSignal = MEDIUM_BUYING_SIGNALS.any { text.contains(it) }
        val mediumNegated = hasMediumSignal && MEDIUM_BUYING_SIGNALS.filter { text.contains(it) }
            .all { isNegated(text, it) }

        return when {
            hasStrongSignal && !strongNegated -> PurchaseIntent.READY_TO_BUY
            hasStrongSignal && strongNegated -> PurchaseIntent.CONSIDERING // Negated strong → downgrade
            containsAny(text, COMPARING_SIGNALS) -> PurchaseIntent.COMPARING
            hasMediumSignal && !mediumNegated -> PurchaseIntent.CONSIDERING
            hasMediumSignal && mediumNegated -> PurchaseIntent.BROWSING // Negated medium → downgrade
            containsAny(text, BROWSING_SIGNALS) -> PurchaseIntent.BROWSING
            else -> PurchaseIntent.NONE
        }
    }

    private fun enrichPurchaseIntent(
        baseIntent: PurchaseIntent,
        shoppingContexts: List<String>,
        objectionInsights: List<ObjectionInsight>
    ): PurchaseIntent {
        if (baseIntent >= PurchaseIntent.CONSIDERING) return baseIntent

        val hasDeliberateMission = shoppingContexts.any {
            it in setOf("gift_mission", "repeat_consideration")
        }
        val hasApprovalFriction = objectionInsights.any { it.theme == "social_approval" }

        return when {
            hasDeliberateMission || hasApprovalFriction -> PurchaseIntent.CONSIDERING
            else -> baseIntent
        }
    }

    /**
     * Extract customer objections
     */
    private fun extractObjections(
        text: String,
        objectionInsights: List<ObjectionInsight>
    ): List<Objection> {
        val compatibilitySummaries = objectionInsights.mapNotNull { insight ->
            insight.summaryLabel?.let { label ->
                Objection.values().firstOrNull { it.name == label }
            }
        }

        val fixedSummaries = Objection.values().filter { objection ->
            containsAny(text, objection.keywords)
        }

        return (compatibilitySummaries + fixedSummaries).distinct()
    }

    private fun extractObjectionInsights(
        text: String,
        shoppingContextHits: List<PhraseMatch>
    ): List<ObjectionInsight> {
        val insights = mutableListOf<ObjectionInsight>()

        for (objection in Objection.values()) {
            objection.keywords.firstOrNull { text.contains(it) }?.let { phrase ->
                insights += ObjectionInsight(
                    theme = defaultObjectionTheme(objection),
                    summaryLabel = objection.name,
                    evidencePhrase = phrase,
                    confidence = 0.7f
                )
            }
        }

        for (themePattern in OBJECTION_THEME_PATTERNS) {
            themePattern.patterns.firstOrNull { text.contains(it) }?.let { phrase ->
                insights += ObjectionInsight(
                    theme = themePattern.theme,
                    summaryLabel = themePattern.summary?.name,
                    evidencePhrase = phrase,
                    confidence = 0.72f
                )
            }
        }

        if (shoppingContextHits.any { it.label == "social_approval" }) {
            shoppingContextHits
                .firstOrNull { it.label == "social_approval" }
                ?.let { match ->
                    insights += ObjectionInsight(
                        theme = "social_approval",
                        summaryLabel = Objection.NEED_TO_THINK.name,
                        evidencePhrase = match.phrase,
                        confidence = 0.78f
                    )
                }
        }

        return insights.distinctBy { "${it.theme}:${it.summaryLabel}:${it.evidencePhrase}" }
    }

    private fun defaultObjectionTheme(objection: Objection): String = when (objection) {
        Objection.TOO_EXPENSIVE -> "budget_pressure"
        Objection.WRONG_COLOR -> "style_mismatch"
        Objection.WRONG_SIZE -> "fit_mismatch"
        Objection.NEED_TO_THINK -> "decision_deferral"
        Objection.COMPARING_ELSEWHERE -> "external_comparison"
        Objection.QUALITY_CONCERN -> "quality_risk"
        Objection.WARRANTY_CONCERN -> "warranty_risk"
        Objection.NOT_NOW -> "timing_deferral"
    }

    private fun buildPurchaseJourney(
        text: String,
        intent: PurchaseIntent,
        contextHits: List<PhraseMatch>,
        objectionInsights: List<ObjectionInsight>
    ): PurchaseJourney {
        val contextLabels = contextHits.map { it.label }.toSet()
        val journeySignals = linkedSetOf<String>()
        val evidencePhrases = linkedSetOf<String>()

        if (contextLabels.contains("gift_mission")) {
            journeySignals += "mission_driven"
            contextHits.filter { it.label == "gift_mission" }.forEach { evidencePhrases += it.phrase }
        }

        if (contextLabels.contains("repeat_consideration")) {
            journeySignals += "repeat_consideration"
            contextHits.filter { it.label == "repeat_consideration" }.forEach { evidencePhrases += it.phrase }
        }

        if (objectionInsights.any { it.theme == "social_approval" }) {
            journeySignals += "approval_blocked"
            objectionInsights
                .filter { it.theme == "social_approval" }
                .mapNotNullTo(evidencePhrases) { it.evidencePhrase }
        }

        val stage = when {
            intent == PurchaseIntent.READY_TO_BUY -> "ready_to_buy"
            contextLabels.contains("repeat_consideration") -> "repeat_consideration"
            intent == PurchaseIntent.COMPARING -> "active_comparison"
            intent == PurchaseIntent.CONSIDERING -> "active_consideration"
            intent == PurchaseIntent.BROWSING -> "casual_browsing"
            else -> PurchaseJourney.NONE_STAGE
        }

        val urgency = when {
            containsAny(text, listOf("today", "right now", "tonight")) -> "immediate"
            intent == PurchaseIntent.READY_TO_BUY -> "high"
            intent in listOf(PurchaseIntent.COMPARING, PurchaseIntent.CONSIDERING) -> "medium"
            intent == PurchaseIntent.BROWSING -> "low"
            else -> "none"
        }

        return PurchaseJourney(
            stage = stage,
            urgency = urgency,
            journeySignals = journeySignals.toList(),
            evidencePhrases = evidencePhrases.toList()
        )
    }

    private fun buildSpeechSemantics(
        intent: PurchaseIntent,
        brandMatches: List<BrandMatch>,
        shoppingContexts: List<String>,
        purchaseJourney: PurchaseJourney,
        objectionInsights: List<ObjectionInsight>,
        confidence: Float,
        priceInquiry: Boolean,
        availabilityInquiry: Boolean
    ): SpeechSemantics? {
        val entities = brandMatches.map { match ->
            SpeechSemanticEntity(
                name = match.canonicalName,
                type = "brand",
                role = intent.toSpeechSemanticRole()
            )
        }.distinctBy { "${it.name}:${it.type}:${it.role}:${it.polarity}" }

        val reasons = objectionInsights.map { objection ->
            SpeechSemanticReason(
                target = null,
                type = objection.theme,
                detail = objection.evidencePhrase ?: objection.summaryLabel
            )
        }.toMutableList()

        val questions = mutableListOf<SpeechSemanticQuestion>()
        if (priceInquiry) {
            questions += SpeechSemanticQuestion(type = "price_inquiry", text = "price inquiry")
        }
        if (availabilityInquiry) {
            questions += SpeechSemanticQuestion(type = "availability_inquiry", text = "availability inquiry")
        }

        val evidencePhrases = linkedSetOf<String>()
        purchaseJourney.evidencePhrases.forEach { evidencePhrases += it }
        objectionInsights.mapNotNullTo(evidencePhrases) { it.evidencePhrase }

        val speechSemantics = SpeechSemantics(
            entities = entities,
            reasons = reasons.distinctBy { "${it.target}:${it.type}:${it.detail}" },
            questions = questions.distinctBy { "${it.type}:${it.text}" },
            journeyState = purchaseJourney.stage.takeUnless { it == PurchaseJourney.NONE_STAGE },
            confidence = confidence,
            evidencePhrases = evidencePhrases.toList(),
            brandMatches = brandMatches,
            shoppingContexts = shoppingContexts,
            purchaseJourney = purchaseJourney,
            objectionInsights = objectionInsights
        )

        return speechSemantics.takeIf { it.hasSignals() }
    }

    /**
     * Extract customer interests/concerns
     */
    private fun extractInterests(text: String): List<String> {
        return INTEREST_PATTERNS.entries
            .filter { (_, patterns) -> containsAny(text, patterns) }
            .map { it.key }
    }

    /**
     * Detect sentiment from transcript
     */
    private fun detectSentiment(text: String): Pair<SentimentTone, Float> {
        val positiveCount = POSITIVE_PATTERNS.count { text.contains(it) }
        val negativeCount = NEGATIVE_PATTERNS.count { text.contains(it) }
        val excitedCount = EXCITED_PATTERNS.count { text.contains(it) }

        return when {
            excitedCount > 0 -> SentimentTone.EXCITED to 0.9f
            positiveCount > negativeCount + 1 -> SentimentTone.POSITIVE to 0.7f
            negativeCount > positiveCount + 1 -> SentimentTone.NEGATIVE to 0.7f
            else -> SentimentTone.NEUTRAL to 0.5f
        }
    }

    /**
     * Classify the type of conversation
     */
    private fun classifyConversation(
        text: String,
        intent: PurchaseIntent,
        objections: List<Objection>
    ): ConversationType {
        return when {
            intent == PurchaseIntent.READY_TO_BUY -> ConversationType.PURCHASE
            objections.contains(Objection.TOO_EXPENSIVE) ||
                containsAny(text, listOf("discount", "deal", "negotiate")) -> ConversationType.PRICE_NEGOTIATION
            containsAny(text, listOf("tell me about", "explain", "how does", "what is")) -> ConversationType.SALES_PITCH
            containsAny(text, listOf("problem", "issue", "broken", "return", "refund")) -> ConversationType.COMPLAINT
            intent in listOf(PurchaseIntent.CONSIDERING, PurchaseIntent.COMPARING) -> ConversationType.PRODUCT_INQUIRY
            intent == PurchaseIntent.BROWSING -> ConversationType.CASUAL
            else -> ConversationType.UNKNOWN
        }
    }

    /**
     * Estimate number of speakers (very rough heuristic)
     */
    private fun estimateSpeakerCount(text: String): Int {
        // Look for conversational markers suggesting multiple speakers
        val dialogueMarkers = listOf("?", "yes", "no", "okay", "sure", "thanks", "please")
        val markerCount = dialogueMarkers.count { text.contains(it) }

        return when {
            markerCount >= 4 -> 2  // Likely conversation
            markerCount >= 1 -> 1  // Monologue with some interaction
            else -> 0  // Can't determine
        }
    }

    /**
     * Calculate confidence score based on signal strength.
     *
     * AGGRESSIVE MODE: Since transcripts are deleted immediately, we want to
     * capture as much insight as possible. Any real conversation has value.
     */
    private fun calculateConfidence(
        brands: List<String>,
        products: List<String>,
        intent: PurchaseIntent,
        objections: List<Objection>,
        interests: List<String>
    ): Float {
        val cfg = SensingConfig.get().insight
        var score = cfg.baselineConfidence  // Baseline: any valid transcript has value

        // Brands mentioned = boost
        score += brands.size.coerceAtMost(3) * cfg.brandBoost

        // Products mentioned = boost
        score += products.size.coerceAtMost(3) * cfg.productBoost

        // Strong intent = major boost
        score += when (intent) {
            PurchaseIntent.READY_TO_BUY -> cfg.readyToBuyBoost
            PurchaseIntent.COMPARING -> cfg.comparingBoost
            PurchaseIntent.CONSIDERING -> cfg.consideringBoost
            PurchaseIntent.BROWSING -> cfg.browsingBoost
            PurchaseIntent.NONE -> 0f
        }

        // Objections detected = boost (we know what they're thinking)
        score += objections.size.coerceAtMost(2) * cfg.interestBoost

        // Interests detected = boost
        score += interests.size.coerceAtMost(2) * cfg.interestBoost

        return minOf(score, cfg.maxConfidence).coerceAtLeast(0f)
    }

    /**
     * Helper: Check if text contains any of the patterns
     */
    private fun containsAny(text: String, patterns: List<String>): Boolean {
        return patterns.any { text.contains(it) }
    }
}
