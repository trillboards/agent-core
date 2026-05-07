package com.trillboards.ctv.core.rendering.models

import org.json.JSONArray
import org.json.JSONObject

data class ContentResponse(
    val ads: List<AdItem>,
    val streams: List<StreamItem>,
    val displayConfig: DisplayConfig,
    val waterfallConfig: WaterfallConfig?,
    val etag: String?
)

data class AdItem(
    val id: String,
    val mediaType: String,
    val videoUrl: String?,
    val imageUrls: List<String>,
    val duration: Int,
    val campaignLink: String?,
    val selfPromo: Boolean,
    val instantAd: Boolean,
    val vastTag: String?,
    val advertisementId: String?,
    val impressionId: String?
)

data class StreamItem(
    val id: String,
    val streamType: String,
    val contentUrl: String?,
    val youtubeId: String?,
    val twitchId: String?,
    val title: String?,
    val duration: Int
)

data class DisplayConfig(
    val displayMode: String,
    val lBarEnabled: Boolean,
    val adIntervalMs: Long,
    val streamEnabled: Boolean
)

data class WaterfallConfig(
    val sources: List<WaterfallSource>,
    val globalTimeoutMs: Int
)

data class WaterfallSource(
    val name: String,
    val vastUrl: String,
    val timeoutMs: Int,
    val priority: Int,
    val cpm: Float
)

data class ImageItem(
    val url: String,
    val durationMs: Long = 8000
)

data class PlaybackEvent(
    val type: String,
    val adId: String?,
    val impressionId: String?,
    val screenId: String,
    val fingerprint: String,
    val renderingMode: String,
    val timestamp: Long = System.currentTimeMillis(),
    val metadata: Map<String, Any> = emptyMap()
) {
    companion object {
        const val TYPE_AD_START = "ad_start"
        const val TYPE_AD_COMPLETE = "ad_complete"
        const val TYPE_AD_ERROR = "ad_error"
        const val TYPE_AD_SKIP = "ad_skip"
        const val TYPE_VAST_REQUEST = "vast_request"
        const val TYPE_VAST_FILL = "vast_fill"
        const val TYPE_VAST_NO_FILL = "vast_no_fill"
        const val TYPE_VAST_ERROR = "vast_error"
        const val TYPE_STREAM_START = "stream_start"
        const val TYPE_STREAM_END = "stream_end"
        const val TYPE_IMAGE_DISPLAYED = "image_displayed"
        const val TYPE_CAROUSEL_COMPLETE = "carousel_complete"
        const val TYPE_PLAYER_STUCK = "player_stuck"
        const val TYPE_PLAYER_RECOVERED = "player_recovered"
    }
}

object ContentParser {

    fun parseContentResponse(json: JSONObject, etag: String?): ContentResponse {
        val dataNode = json.opt("data")
        return when (dataNode) {
            is JSONArray -> parseLegacyContentResponse(json, etag, dataNode)
            is JSONObject -> parseStructuredContentResponse(dataNode, etag)
            else -> parseLegacyContentResponse(json, etag, JSONArray())
        }
    }

    private fun parseAdItem(json: JSONObject): AdItem {
        val contentObj = json.optJSONObject("content") ?: json
        val imageUrls = mutableListOf<String>()
        val primaryImageArray = contentObj.optJSONArray("image") ?: contentObj.optJSONArray("images")
        if (primaryImageArray != null) {
            for (i in 0 until primaryImageArray.length()) {
                primaryImageArray.optString(i)?.takeIf { it.isNotBlank() }?.let { imageUrls.add(it) }
            }
        }
        val photosArray = contentObj.optJSONArray("photos")
        if (photosArray != null) {
            for (i in 0 until photosArray.length()) {
                photosArray.optString(i)?.takeIf { it.isNotBlank() && it !in imageUrls }?.let { imageUrls.add(it) }
            }
        }
        val singlePhoto = contentObj.optString("photo", "")
        if (singlePhoto.isNotBlank() && singlePhoto !in imageUrls) {
            imageUrls.add(0, singlePhoto)
        }

        val videoUrl = contentObj.optNonBlankString("video_url", "video")
        val vastTag = contentObj.optNonBlankString("vast_tag")
        val mediaType = when {
            contentObj.optNonBlankString("mediaType") != null -> contentObj.optString("mediaType")
            videoUrl != null -> "video"
            vastTag != null -> "video"
            imageUrls.isNotEmpty() -> "image"
            else -> "unknown"
        }

        return AdItem(
            id = contentObj.optNonBlankString("id", "_id")
                ?: json.optNonBlankString("id", "_id")
                ?: "",
            mediaType = mediaType,
            videoUrl = videoUrl,
            imageUrls = imageUrls,
            duration = contentObj.optInt("duration", contentObj.optInt("advertisement_duration", 15)),
            campaignLink = contentObj.optNonBlankString("campaign_link"),
            selfPromo = contentObj.optBooleanAny("selfPromo", "self_promo", default = false),
            instantAd = contentObj.optBooleanAny("instant_ad", default = false),
            vastTag = vastTag,
            advertisementId = contentObj.optNonBlankString("id", "_id")
                ?: json.optNonBlankString("advertisement_id", "_id"),
            impressionId = contentObj.optNonBlankString("allocation_id", "impression_id")
                ?: json.optNonBlankString("allocation_id", "impression_id")
        )
    }

    private fun parseStreamItem(json: JSONObject): StreamItem {
        val contentObj = json.optJSONObject("content") ?: json
        val youtubeUrl = contentObj.optNonBlankString("youtube_url")
        val twitchUrl = contentObj.optNonBlankString("twitch_url")
        val streamType = contentObj.optNonBlankString("stream_type") ?: "default"
        return StreamItem(
            id = contentObj.optNonBlankString("id", "_id") ?: "",
            streamType = streamType,
            contentUrl = contentObj.optNonBlankString(
                "content_url",
                "video_url",
                "local_file_path"
            ),
            youtubeId = contentObj.optNonBlankString("youtube_id")
                ?: extractVideoId(youtubeUrl),
            twitchId = contentObj.optNonBlankString("twitch_id")
                ?: extractTwitchId(twitchUrl),
            title = contentObj.optNonBlankString("title", "name", "description"),
            duration = contentObj.optInt("duration", contentObj.optInt("duration_seconds", 0))
        )
    }

    private fun parseLegacyContentResponse(
        json: JSONObject,
        etag: String?,
        dataArray: JSONArray
    ): ContentResponse {
        val ads = mutableListOf<AdItem>()
        val streams = mutableListOf<StreamItem>()

        for (i in 0 until dataArray.length()) {
            val item = dataArray.optJSONObject(i) ?: continue
            val type = item.optString("type", "ad")
            if (type == "stream" || type == "default_stream") {
                streams.add(parseStreamItem(item))
            } else {
                ads.add(parseAdItem(item))
            }
        }

        val configObj = json.optJSONObject("config") ?: JSONObject()
        val displayConfig = DisplayConfig(
            displayMode = configObj.optString("display_mode", "fullscreen"),
            lBarEnabled = configObj.optBoolean("l_bar_enabled", false),
            adIntervalMs = configObj.optLong("ad_interval_ms", 60_000),
            streamEnabled = configObj.optBoolean("stream_enabled", true)
        )

        val waterfallObj = json.optJSONObject("waterfall")
        val waterfallConfig = if (waterfallObj != null) parseWaterfallConfig(waterfallObj) else null

        return ContentResponse(
            ads = ads,
            streams = streams,
            displayConfig = displayConfig,
            waterfallConfig = waterfallConfig,
            etag = etag
        )
    }

    private fun parseStructuredContentResponse(
        dataObj: JSONObject,
        etag: String?
    ): ContentResponse {
        val ads = parseStructuredAds(dataObj)
        val streams = parseStructuredStreams(dataObj)

        val playbackPolicy = dataObj.optJSONObject("playback_policy") ?: JSONObject()
        val overlaySettings = dataObj.optJSONObject("overlay_settings") ?: JSONObject()
        val displayPreferences = dataObj.optJSONObject("display_preferences") ?: JSONObject()

        val displayMode = displayPreferences.optNonBlankString("display_mode")
            ?: overlaySettings.optNonBlankString("display_mode")
            ?: "fullscreen"
        val adIntervalMs = playbackPolicy.optLong("paid_interval_ms", 0L)
            .takeIf { it > 0 }
            ?: overlaySettings.optLong("overlay_interval", 0L).takeIf { it > 0 }
            ?: displayPreferences.optLong("ad_interval", 0L).takeIf { it > 0 }?.times(1000L)
            ?: 60_000L
        val lBarEnabled = displayPreferences.optBooleanAny("l_bar_ads_enabled", default = false)
            || dataObj.optJSONObject("l_bar_content")?.optBoolean("enabled", false) == true

        return ContentResponse(
            ads = ads,
            streams = streams,
            displayConfig = DisplayConfig(
                displayMode = displayMode,
                lBarEnabled = lBarEnabled,
                adIntervalMs = adIntervalMs,
                streamEnabled = streams.isNotEmpty() || dataObj.optJSONObject("background_stream") != null
            ),
            waterfallConfig = null,
            etag = etag
        )
    }

    private fun parseStructuredAds(dataObj: JSONObject): List<AdItem> {
        val adsArray = dataObj.optJSONArray("overlay_advertisements") ?: JSONArray()
        val ads = mutableListOf<AdItem>()
        for (i in 0 until adsArray.length()) {
            val item = adsArray.optJSONObject(i) ?: continue
            val parsed = parseAdItem(item)
            if (parsed.id.isNotBlank()) {
                ads.add(parsed)
            }
        }
        return ads
    }

    private fun parseStructuredStreams(dataObj: JSONObject): List<StreamItem> {
        val streamArray = dataObj.optJSONArray("all_default_streams") ?: JSONArray()
        val orderedStreamObjects = mutableListOf<JSONObject>()
        for (i in 0 until streamArray.length()) {
            streamArray.optJSONObject(i)?.let { orderedStreamObjects.add(it) }
        }

        orderedStreamObjects.sortWith(
            compareBy<JSONObject>(
                { it.optJSONObject("content")?.optInt("playback_order", Int.MAX_VALUE) ?: Int.MAX_VALUE },
                { it.optString("createdAt", "") },
                { it.optJSONObject("content")?.optString("id", "") ?: "" }
            )
        )

        val streamQueue = orderedStreamObjects.map(::parseStreamItem).filter { it.id.isNotBlank() }
        if (streamQueue.isEmpty()) {
            val background = dataObj.optJSONObject("background_stream")?.let(::parseStreamItem)
            return listOfNotNull(background?.takeIf { it.id.isNotBlank() })
        }

        val backgroundId = dataObj.optJSONObject("background_stream")
            ?.optJSONObject("content")
            ?.optNonBlankString("id", "_id")
        val playbackState = dataObj.optJSONObject("playback_state")
        val currentStreamId = backgroundId ?: playbackState?.optNonBlankString("stream_id")
        val queuePosition = playbackState?.optInt("queue_position", -1) ?: -1

        val currentIndex = when {
            queuePosition in streamQueue.indices && (
                currentStreamId == null || streamQueue[queuePosition].id == currentStreamId
            ) -> queuePosition
            currentStreamId != null -> streamQueue.indexOfFirst { it.id == currentStreamId }
            else -> 0
        }

        val rotated = if (currentIndex in streamQueue.indices) {
            rotate(streamQueue, currentIndex)
        } else {
            streamQueue
        }

        if (currentStreamId == null) {
            return rotated
        }

        val backgroundStream = dataObj.optJSONObject("background_stream")?.let(::parseStreamItem)
            ?.takeIf { it.id.isNotBlank() }
        return if (backgroundStream != null && rotated.none { it.id == backgroundStream.id }) {
            listOf(backgroundStream) + rotated
        } else {
            rotated
        }
    }

    private fun rotate(streams: List<StreamItem>, startIndex: Int): List<StreamItem> {
        if (streams.isEmpty() || startIndex !in streams.indices) {
            return streams
        }
        return streams.drop(startIndex) + streams.take(startIndex)
    }

    private fun parseWaterfallConfig(json: JSONObject): WaterfallConfig {
        val sourcesArray = json.optJSONArray("sources") ?: JSONArray()
        val sources = mutableListOf<WaterfallSource>()
        for (i in 0 until sourcesArray.length()) {
            val src = sourcesArray.optJSONObject(i) ?: continue
            sources.add(
                WaterfallSource(
                    name = src.optString("name", "unknown"),
                    vastUrl = src.optString("vast_url", ""),
                    timeoutMs = src.optInt("timeout_ms", 5000),
                    priority = src.optInt("priority", i),
                    cpm = src.optDouble("cpm", 0.0).toFloat()
                )
            )
        }
        return WaterfallConfig(
            sources = sources.sortedBy { it.priority },
            globalTimeoutMs = json.optInt("global_timeout_ms", 15000)
        )
    }

    private fun JSONObject.optNonBlankString(vararg keys: String): String? {
        keys.forEach { key ->
            val value = optString(key, "").trim()
            if (value.isNotEmpty() && !value.equals("null", ignoreCase = true)) {
                return value
            }
        }
        return null
    }

    private fun JSONObject.optBooleanAny(vararg keys: String, default: Boolean): Boolean {
        keys.forEach { key ->
            if (has(key) && !isNull(key)) {
                return optBoolean(key, default)
            }
        }
        return default
    }

    private fun extractVideoId(url: String?): String? {
        if (url.isNullOrBlank()) {
            return null
        }

        val regex = Regex("(?:v=|youtu\\.be/)([A-Za-z0-9_-]{6,})")
        return regex.find(url)?.groupValues?.getOrNull(1)
    }

    private fun extractTwitchId(url: String?): String? {
        if (url.isNullOrBlank()) {
            return null
        }

        val regex = Regex("twitch\\.tv/([^/?#]+)")
        return regex.find(url)?.groupValues?.getOrNull(1)
    }
}
