package com.trillboards.ctv.core.rendering

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import com.trillboards.ctv.core.SensingConfig
import android.util.Log
import androidx.annotation.OptIn
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaItem.AdsConfiguration
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultDataSource
import androidx.media3.datasource.cache.CacheDataSource
import androidx.media3.datasource.cache.LeastRecentlyUsedCacheEvictor
import androidx.media3.datasource.cache.SimpleCache
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.ima.ImaAdsLoader
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.PlayerView
import com.trillboards.ctv.core.rendering.models.PlaybackEvent
import java.io.File

@OptIn(UnstableApi::class)
class NativeAdPlayer(
    private val context: Context,
    private val listener: Listener
) {
    companion object {
        private const val TAG = "NativeAdPlayer"
        private const val CACHE_DIR = "exo_media_cache"
        private val renderCfg get() = SensingConfig.get().rendering
        private val CACHE_SIZE_BYTES get() = renderCfg.cacheSizeBytes
        private val STUCK_BUFFERING_TIMEOUT_MS get() = renderCfg.stuckBufferingTimeoutMs
        private val TARGET_BUFFER_BYTES get() = renderCfg.targetBufferBytes
    }

    interface Listener {
        fun onAdStarted(adId: String?)
        fun onAdCompleted(adId: String?)
        fun onAdError(adId: String?, error: String)
        fun onVideoStarted(url: String)
        fun onVideoCompleted(url: String)
        fun onVideoError(url: String, error: String)
        fun onPlayerStuck()
        fun onPlayerRecovered()
    }

    private var player: ExoPlayer? = null
    private var imaAdsLoader: ImaAdsLoader? = null
    private var playerView: PlayerView? = null
    private var mediaCache: SimpleCache? = null
    @Volatile private var currentAdId: String? = null
    @Volatile private var currentVideoUrl: String? = null
    @Volatile private var isInitialized = false
    @Volatile private var isStuck = false
    private var hasEmittedStartForCurrentMedia = false

    // Reuse a single Handler + Runnable for stuck detection — cancel previous before posting new
    private val stuckDetectionHandler = Handler(Looper.getMainLooper())
    private val stuckDetectionRunnable = Runnable {
        if (player?.playbackState == Player.STATE_BUFFERING && !isStuck) {
            isStuck = true
            listener.onPlayerStuck()
        }
    }

    fun initialize(playerView: PlayerView) {
        if (isInitialized) {
            Log.w(TAG, "Already initialized — releasing first")
            release()
        }
        this.playerView = playerView

        // Set up media cache
        val cacheDir = File(context.cacheDir, CACHE_DIR)
        if (!cacheDir.exists()) cacheDir.mkdirs()
        mediaCache = SimpleCache(
            cacheDir,
            LeastRecentlyUsedCacheEvictor(CACHE_SIZE_BYTES),
            androidx.media3.database.StandaloneDatabaseProvider(context)
        )

        // Build IMA ads loader for VAST
        imaAdsLoader = ImaAdsLoader.Builder(context).build()

        // Build cache-enabled data source
        val upstreamFactory = DefaultDataSource.Factory(context)
        val cacheDataSourceFactory = CacheDataSource.Factory()
            .setCache(mediaCache!!)
            .setUpstreamDataSourceFactory(upstreamFactory)
            .setFlags(CacheDataSource.FLAG_IGNORE_CACHE_ON_ERROR)

        // Build media source factory with IMA
        val mediaSourceFactory = DefaultMediaSourceFactory(cacheDataSourceFactory)
            .setLocalAdInsertionComponents({ imaAdsLoader!! }, playerView)

        // Build ExoPlayer
        player = ExoPlayer.Builder(context)
            .setMediaSourceFactory(mediaSourceFactory)
            .build()
            .also { exoPlayer ->
                playerView.player = exoPlayer
                imaAdsLoader?.setPlayer(exoPlayer)
                exoPlayer.addListener(playerEventListener)
                exoPlayer.playWhenReady = true
            }

        isInitialized = true
        Log.i(TAG, "Initialized with IMA + 100MB cache")
    }

    fun playVastAd(vastTagUrl: String, adId: String?) {
        val exoPlayer = player ?: run {
            listener.onAdError(adId, "Player not initialized")
            return
        }
        currentAdId = adId
        currentVideoUrl = null
        hasEmittedStartForCurrentMedia = false

        val adTagUri = Uri.parse(vastTagUrl)
        val mediaItem = MediaItem.Builder()
            .setUri(Uri.EMPTY)
            .setAdsConfiguration(AdsConfiguration.Builder(adTagUri).build())
            .build()

        exoPlayer.stop()
        exoPlayer.setMediaItem(mediaItem)
        exoPlayer.prepare()
        exoPlayer.playWhenReady = true
        Log.i(TAG, "Playing VAST ad: adId=$adId")
    }

    fun playDirectVideo(videoUrl: String, mediaType: String) {
        val exoPlayer = player ?: run {
            listener.onVideoError(videoUrl, "Player not initialized")
            return
        }
        currentAdId = null
        currentVideoUrl = videoUrl
        hasEmittedStartForCurrentMedia = false

        val uri = Uri.parse(videoUrl)
        val builder = MediaItem.Builder().setUri(uri)

        // Set MIME type hint for HLS/DASH
        when {
            videoUrl.contains(".m3u8", ignoreCase = true) || mediaType == "hls" ->
                builder.setMimeType("application/x-mpegURL")
            videoUrl.contains(".mpd", ignoreCase = true) || mediaType == "dash" ->
                builder.setMimeType("application/dash+xml")
        }

        val mediaItem = builder.build()
        exoPlayer.stop()
        exoPlayer.setMediaItem(mediaItem)
        exoPlayer.prepare()
        exoPlayer.playWhenReady = true
        Log.i(TAG, "Playing direct video: $videoUrl (type=$mediaType)")
    }

    fun stop() {
        stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
        player?.stop()
        currentAdId = null
        currentVideoUrl = null
    }

    fun release() {
        isInitialized = false
        stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
        player?.removeListener(playerEventListener)
        player?.release()
        player = null
        imaAdsLoader?.release()
        imaAdsLoader = null
        mediaCache?.release()
        mediaCache = null
        playerView?.player = null
        playerView = null
        currentAdId = null
        currentVideoUrl = null
        Log.i(TAG, "Released")
    }

    fun isPlaying(): Boolean = player?.isPlaying == true

    fun isReady(): Boolean = isInitialized && player != null

    fun releaseBuffers() {
        player?.stop()
        Log.d(TAG, "Buffers released for memory pressure")
    }

    private val playerEventListener = object : Player.Listener {
        override fun onPlaybackStateChanged(playbackState: Int) {
            when (playbackState) {
                Player.STATE_READY -> {
                    if (isStuck) {
                        isStuck = false
                        listener.onPlayerRecovered()
                    }
                    // Cancel stuck detection — player recovered
                    stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
                }
                Player.STATE_BUFFERING -> {
                    // Cancel previous stuck timer, start a fresh one
                    stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
                    stuckDetectionHandler.postDelayed(stuckDetectionRunnable, STUCK_BUFFERING_TIMEOUT_MS)
                }
                Player.STATE_ENDED -> {
                    stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
                    // Capture to local vals to avoid race conditions
                    val adId = currentAdId
                    val videoUrl = currentVideoUrl
                    if (adId != null) {
                        listener.onAdCompleted(adId)
                        currentAdId = null
                    } else if (videoUrl != null) {
                        listener.onVideoCompleted(videoUrl)
                        currentVideoUrl = null
                    }
                }
                Player.STATE_IDLE -> {
                    stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
                }
            }
        }

        override fun onPlayerError(error: PlaybackException) {
            stuckDetectionHandler.removeCallbacks(stuckDetectionRunnable)
            val errorMsg = "${error.errorCodeName}: ${error.message}"
            Log.e(TAG, "Playback error: $errorMsg", error)
            // Capture to local vals to avoid race conditions
            val adId = currentAdId
            val videoUrl = currentVideoUrl
            if (adId != null) {
                listener.onAdError(adId, errorMsg)
            } else if (videoUrl != null) {
                listener.onVideoError(videoUrl, errorMsg)
            }
        }

        override fun onIsPlayingChanged(isPlaying: Boolean) {
            // Guard: only emit start once per media item to avoid duplicates
            // (IMA fires onIsPlayingChanged for both ad content and post-roll content)
            if (isPlaying && !hasEmittedStartForCurrentMedia) {
                hasEmittedStartForCurrentMedia = true
                val adId = currentAdId
                val videoUrl = currentVideoUrl
                if (adId != null) {
                    listener.onAdStarted(adId)
                } else if (videoUrl != null) {
                    listener.onVideoStarted(videoUrl)
                }
            }
        }
    }
}
