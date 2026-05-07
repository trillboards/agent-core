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
 * @param adId MongoDB advertisement ID if an ad is currently playing, null otherwise
 * @param contentType Type of content: "ad", "default_stream", "self_promo", "idle"
 * @param streamId Optional stream ID if a stream is playing
 * @param impressionId Impression ID (e.g. "imp_1708000000_abc123") for VAS correlation, null if unavailable
 */
data class ContentState(
    val adId: String? = null,
    val contentType: String = "idle",
    val streamId: String? = null,
    val impressionId: String? = null
)
