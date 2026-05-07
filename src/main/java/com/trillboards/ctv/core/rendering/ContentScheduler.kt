package com.trillboards.ctv.core.rendering

import android.util.Log
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.rendering.models.AdItem
import com.trillboards.ctv.core.rendering.models.ContentResponse
import com.trillboards.ctv.core.rendering.models.ImageItem
import com.trillboards.ctv.core.rendering.models.PlaybackEvent
import com.trillboards.ctv.core.rendering.models.StreamItem
import com.trillboards.ctv.core.rendering.models.WaterfallSource
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class ContentScheduler(
    private val scope: CoroutineScope,
    private val config: AgentConfig,
    private val adPlayer: NativeAdPlayer,
    private val imageCarousel: NativeImageCarousel,
    private val contentApiClient: ContentApiClient,
    private val impressionTracker: ImpressionTracker,
    private val listener: Listener
) {
    companion object {
        private const val TAG = "ContentScheduler"
        private val renderCfg get() = SensingConfig.get().rendering
        private val CONTENT_POLL_INTERVAL_MS get() = renderCfg.contentPollIntervalMs
        private val DEFAULT_AD_INTERVAL_MS get() = renderCfg.defaultAdIntervalMs
        private val WATERFALL_SOURCE_TIMEOUT_MS get() = renderCfg.waterfallSourceTimeoutMs
    }

    interface Listener {
        fun onContentLoaded(response: ContentResponse)
        fun onAdSlotStart()
        fun onAdSlotEnd()
        fun onStreamStart(stream: StreamItem)
        fun onStreamEnd(stream: StreamItem)
        fun onNeedsWebView(stream: StreamItem)
        fun onPlaybackEvent(event: PlaybackEvent)
        fun onError(error: String)
    }

    private var contentPollJob: Job? = null
    private var rotationJob: Job? = null
    private var currentContent: ContentResponse? = null
    private var currentEtag: String? = null
    private var currentAdIndex = 0
    private var currentStreamIndex = 0
    private var screenId: String = ""
    private var fingerprint: String = ""
    @Volatile private var isRunning = false

    // Mutex to prevent concurrent ad playback (rotation vs instant ad)
    private val adPlaybackMutex = Mutex()

    fun start(screenId: String, fingerprint: String) {
        if (isRunning) stop()
        this.screenId = screenId
        this.fingerprint = fingerprint
        this.isRunning = true

        // Start content polling
        contentPollJob = scope.launch {
            pollContentLoop()
        }

        Log.i(TAG, "Started for screen=$screenId")
    }

    fun stop() {
        isRunning = false
        contentPollJob?.cancel()
        rotationJob?.cancel()
        contentPollJob = null
        rotationJob = null
        adPlayer.stop()
        imageCarousel.stop()
        Log.i(TAG, "Stopped")
    }

    fun handleInstantAd(ad: AdItem) {
        Log.i(TAG, "Instant ad received: ${ad.id}")
        scope.launch {
            adPlaybackMutex.withLock {
                // Stop current playback before instant ad
                adPlayer.stop()
                imageCarousel.stop()
                playAd(ad)
            }
        }
    }

    fun refresh() {
        scope.launch {
            val content = contentApiClient.fetchContent(screenId) ?: return@launch
            currentContent = content
            currentEtag = content.etag
            listener.onContentLoaded(content)
            Log.i(TAG, "Refreshed: ${content.ads.size} ads, ${content.streams.size} streams")
        }
    }

    private suspend fun pollContentLoop() {
        // Initial fetch
        val initialContent = contentApiClient.fetchContent(screenId)
        if (initialContent != null) {
            currentContent = initialContent
            currentEtag = initialContent.etag
            listener.onContentLoaded(initialContent)
            startRotation()
        } else {
            listener.onError("Failed to fetch initial content")
        }

        // Polling loop
        while (isRunning) {
            delay(CONTENT_POLL_INTERVAL_MS)
            try {
                val updated = contentApiClient.fetchContentIfChanged(screenId, currentEtag)
                if (updated != null) {
                    currentContent = updated
                    currentEtag = updated.etag
                    listener.onContentLoaded(updated)
                    Log.i(TAG, "Content updated: ${updated.ads.size} ads, ${updated.streams.size} streams")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Content poll error", e)
            }
        }
    }

    private fun startRotation() {
        rotationJob?.cancel()
        rotationJob = scope.launch {
            rotationLoop()
        }
    }

    private suspend fun rotationLoop() {
        while (isRunning) {
            val content = currentContent
            if (content == null) {
                delay(renderCfg.contentRetryDelayMs)
                continue
            }

            // Play background stream if available
            var currentStream: StreamItem? = null
            if (content.streams.isNotEmpty()) {
                val stream = content.streams[currentStreamIndex % content.streams.size]
                currentStreamIndex++

                if (needsWebView(stream)) {
                    listener.onNeedsWebView(stream)
                } else {
                    currentStream = stream
                    playStream(stream)
                }
            }

            // Wait for ad interval (stream plays during this time)
            val adInterval = content.displayConfig.adIntervalMs.coerceAtLeast(renderCfg.adIntervalMinMs)
            delay(adInterval)

            // Notify stream ended before ad slot starts
            if (currentStream != null) {
                adPlayer.stop()
                listener.onStreamEnd(currentStream)
            }

            // Play ad slot (mutex prevents instant ad from conflicting)
            if (content.ads.isNotEmpty()) {
                adPlaybackMutex.withLock {
                    listener.onAdSlotStart()
                    val ad = content.ads[currentAdIndex % content.ads.size]
                    currentAdIndex++
                    playAd(ad)
                    listener.onAdSlotEnd()
                }
            }
        }
    }

    private suspend fun playAd(ad: AdItem) {
        val content = currentContent

        // Try VAST waterfall first if available
        val waterfallConfig = content?.waterfallConfig
        if (waterfallConfig != null && waterfallConfig.sources.isNotEmpty()) {
            val filled = tryWaterfall(waterfallConfig.sources, ad.id)
            if (filled) return
        }

        // Try ad's own VAST tag
        if (!ad.vastTag.isNullOrBlank()) {
            emitEvent(PlaybackEvent.TYPE_VAST_REQUEST, ad.id)
            adPlayer.playVastAd(ad.vastTag, ad.id)
            waitForPlaybackCompletion(ad.duration.toLong() * 1000 + 5000)
            return
        }

        // Direct video fallback
        if (ad.mediaType == "video" && !ad.videoUrl.isNullOrBlank()) {
            adPlayer.playDirectVideo(ad.videoUrl, "mp4")
            trackImpression(ad)
            waitForPlaybackCompletion(ad.duration.toLong() * 1000 + 5000)
            return
        }

        // Image ad fallback
        if (ad.imageUrls.isNotEmpty()) {
            val items = ad.imageUrls.map { ImageItem(it, ad.duration.toLong() * 1000) }
            imageCarousel.showImages(items, ad.duration.toLong() * 1000)
            trackImpression(ad)
            delay(ad.duration.toLong() * 1000 * ad.imageUrls.size)
            imageCarousel.stop()
            return
        }

        Log.w(TAG, "Ad ${ad.id} has no playable content")
    }

    private suspend fun tryWaterfall(sources: List<WaterfallSource>, adId: String?): Boolean {
        for (source in sources) {
            if (!isRunning) return false
            if (source.vastUrl.isBlank()) continue

            emitEvent(PlaybackEvent.TYPE_VAST_REQUEST, adId, mapOf("source" to source.name))
            try {
                adPlayer.playVastAd(source.vastUrl, adId)
                // Give the VAST source time to respond
                val timeout = source.timeoutMs.toLong().coerceAtLeast(WATERFALL_SOURCE_TIMEOUT_MS)
                delay(timeout)

                if (adPlayer.isPlaying()) {
                    emitEvent(PlaybackEvent.TYPE_VAST_FILL, adId, mapOf("source" to source.name))
                    // Wait for ad to complete
                    waitForPlaybackCompletion(renderCfg.waterfallCompletionTimeoutMs)
                    return true
                } else {
                    emitEvent(PlaybackEvent.TYPE_VAST_NO_FILL, adId, mapOf("source" to source.name))
                }
            } catch (e: Exception) {
                emitEvent(PlaybackEvent.TYPE_VAST_ERROR, adId, mapOf(
                    "source" to source.name,
                    "error" to (e.message ?: "unknown")
                ))
            }
        }
        return false
    }

    private fun playStream(stream: StreamItem) {
        listener.onStreamStart(stream)

        when {
            !stream.contentUrl.isNullOrBlank() -> {
                val mediaType = when {
                    stream.contentUrl.contains(".m3u8", ignoreCase = true) -> "hls"
                    stream.contentUrl.contains(".mpd", ignoreCase = true) -> "dash"
                    else -> "mp4"
                }
                adPlayer.playDirectVideo(stream.contentUrl, mediaType)
            }
            else -> {
                Log.w(TAG, "Stream ${stream.id} has no direct URL — needs WebView")
                listener.onNeedsWebView(stream)
            }
        }
    }

    private fun needsWebView(stream: StreamItem): Boolean {
        return stream.youtubeId != null ||
                stream.twitchId != null ||
                stream.streamType == "youtube" ||
                stream.streamType == "twitch" ||
                stream.streamType == "slides" ||
                stream.streamType == "canva" ||
                stream.streamType == "iframe"
    }

    private suspend fun waitForPlaybackCompletion(maxWaitMs: Long) {
        val startTime = System.currentTimeMillis()
        while (isRunning && adPlayer.isPlaying() &&
            (System.currentTimeMillis() - startTime) < maxWaitMs
        ) {
            delay(renderCfg.playbackPollIntervalMs)
        }
    }

    private fun trackImpression(ad: AdItem) {
        if (ad.advertisementId != null) {
            impressionTracker.trackImpression(
                adId = ad.advertisementId,
                impressionId = ad.impressionId,
                screenId = screenId,
                fingerprint = fingerprint
            )
        }
    }

    private fun emitEvent(type: String, adId: String?, metadata: Map<String, Any> = emptyMap()) {
        listener.onPlaybackEvent(
            PlaybackEvent(
                type = type,
                adId = adId,
                impressionId = null,
                screenId = screenId,
                fingerprint = fingerprint,
                renderingMode = "native",
                metadata = metadata
            )
        )
    }
}
