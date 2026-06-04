package com.trillboards.ctv.core.audience

/**
 * Interface for providing current content state to AudienceSensingService.
 *
 * The CTV agent knows what content is currently displayed (ads, streams, idle).
 * This interface allows the sensing service to correlate audience metrics with
 * the content that was playing at the time — enabling ad-audience attribution.
 *
 * Implement in the tablet-agent's service layer where WebView content state is tracked.
 */
interface ContentStateProvider {
    /**
     * Get the current content state from the WebView.
     * Called during each aggregation window (every 30 seconds).
     *
     * @return Current content state with ad ID and content type
     */
    fun getCurrentContentState(): ContentState
}

/**
 * Represents what content is currently playing on the screen.
 *
 * @param adId IMA programmatic ad ID if an ad is currently playing, null otherwise.
 *             Kept for back-compat with consumers that only attribute against programmatic ads.
 * @param contentType Type of content: "ad", "default_stream", "self_promo", "sponsored", "idle"
 * @param streamId Stream/loop item ID if a stream is playing (self_promo, default_stream, sponsored)
 * @param impressionId Impression ID (e.g. "imp_1708000000_abc123") for VAS correlation, null if unavailable
 * @param creativeId UNIFIED identifier of whatever creative is on screen RIGHT NOW —
 *                   IMA ad ID, self_promo item ID, default_stream item ID, or sponsored item ID.
 *                   Non-null whenever ANY content is playing. THE attribution key for FEIN
 *                   per-face × creative learning across all content sources, not just IMA.
 * @param creativeSource Discriminator for `creativeId`'s origin:
 *                   "ima_programmatic" | "self_promo" | "default_stream" | "sponsored" | null (idle).
 *                   Lets downstream consumers distinguish a $5 CPM programmatic ad from an
 *                   earner-uploaded self_promo without overloading creativeId's meaning.
 */
data class ContentState(
    val adId: String? = null,
    val contentType: String = "idle",
    val streamId: String? = null,
    val impressionId: String? = null,
    val creativeId: String? = null,
    val creativeSource: String? = null
)
