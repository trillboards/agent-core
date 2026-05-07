package com.trillboards.ctv.core.bridge

import org.json.JSONArray
import org.json.JSONObject

data class OverlayHealthSnapshot(
    val receivedAtMs: Long,
    val capturedAtMs: Long?,
    val reason: String?,
    val uptimeMs: Long?,
    val documentHidden: Boolean?,
    val currentStreamId: String?,
    val currentStreamType: String?,
    val queueLength: Int?,
    val prefetchedUrlCount: Int?,
    val jsHeapUsedMb: Int?,
    val jsHeapTotalMb: Int?,
    val blobUrlCount: Int?,
    val blobPinnedMb: Double?,
    val socketConnected: Boolean?,
    val socketDisconnectReason: String?,
    val youtubeRequestedQuality: String?,
    val youtubePlaybackQuality: String?,
    val youtubeObservedQuality: String?,
    val youtubeAvailableQualityLevels: List<String>,
    val youtubeLastActivityAgeMs: Long?,
    val youtubeKeepaliveReloadCount: Int?,
    val youtubePlayerState: Int?,
    val youtubeCurrentTimeSeconds: Double?,
    val youtubeProbeAgeMs: Long?
) {
    fun toHeartbeatMetadata(nowMs: Long = System.currentTimeMillis()): Map<String, Any?> = buildMap {
        put("overlayHealthAgeMs", nowMs - receivedAtMs)
        capturedAtMs?.let { put("overlayCapturedAtMs", it) }
        reason?.let { put("overlayHealthReason", it) }
        uptimeMs?.let { put("overlayUptimeMs", it) }
        documentHidden?.let { put("overlayDocumentHidden", it) }
        currentStreamId?.let { put("overlayCurrentStreamId", it) }
        currentStreamType?.let { put("overlayCurrentStreamType", it) }
        queueLength?.let { put("overlayQueueLength", it) }
        prefetchedUrlCount?.let { put("overlayPrefetchedUrlCount", it) }
        jsHeapUsedMb?.let { put("overlayJsHeapUsedMb", it) }
        jsHeapTotalMb?.let { put("overlayJsHeapTotalMb", it) }
        blobUrlCount?.let { put("overlayBlobUrlCount", it) }
        blobPinnedMb?.let { put("overlayBlobPinnedMb", it) }
        socketConnected?.let { put("overlaySocketConnected", it) }
        socketDisconnectReason?.let { put("overlaySocketDisconnectReason", it) }
        youtubeRequestedQuality?.let { put("overlayYouTubeRequestedQuality", it) }
        youtubePlaybackQuality?.let { put("overlayYouTubePlaybackQuality", it) }
        youtubeObservedQuality?.let { put("overlayYouTubeObservedQuality", it) }
        if (youtubeAvailableQualityLevels.isNotEmpty()) {
            put("overlayYouTubeAvailableQualityLevels", youtubeAvailableQualityLevels.joinToString(","))
        }
        youtubeLastActivityAgeMs?.let { put("overlayYouTubeLastActivityAgeMs", it) }
        youtubeKeepaliveReloadCount?.let { put("overlayYouTubeKeepaliveReloadCount", it) }
        youtubePlayerState?.let { put("overlayYouTubePlayerState", it) }
        youtubeCurrentTimeSeconds?.let { put("overlayYouTubeCurrentTimeSeconds", it) }
        youtubeProbeAgeMs?.let { put("overlayYouTubeProbeAgeMs", it) }
    }

    companion object {
        fun fromJson(raw: String?, receivedAtMs: Long = System.currentTimeMillis()): OverlayHealthSnapshot? {
            if (raw.isNullOrBlank()) return null

            return runCatching {
                val root = JSONObject(raw)
                val memory = root.optJSONObject("memory")
                val socket = root.optJSONObject("socket")
                val stream = root.optJSONObject("stream")
                val youtube = root.optJSONObject("youtube")

                OverlayHealthSnapshot(
                    receivedAtMs = receivedAtMs,
                    capturedAtMs = root.optLongOrNull("capturedAtMs"),
                    reason = root.optStringOrNull("reason"),
                    uptimeMs = root.optLongOrNull("uptimeMs"),
                    documentHidden = root.optBooleanOrNull("documentHidden"),
                    currentStreamId = stream.optStringOrNull("currentStreamId"),
                    currentStreamType = stream.optStringOrNull("currentStreamType"),
                    queueLength = stream.optIntOrNull("queueLength"),
                    prefetchedUrlCount = stream.optIntOrNull("prefetchedUrlCount"),
                    jsHeapUsedMb = memory.optIntOrNull("jsHeapUsedMb"),
                    jsHeapTotalMb = memory.optIntOrNull("jsHeapTotalMb"),
                    blobUrlCount = memory.optIntOrNull("blobUrlCount"),
                    blobPinnedMb = memory.optDoubleOrNull("blobPinnedMb"),
                    socketConnected = socket.optBooleanOrNull("connected"),
                    socketDisconnectReason = socket.optStringOrNull("disconnectReason"),
                    youtubeRequestedQuality = youtube.optStringOrNull("requestedQuality"),
                    youtubePlaybackQuality = youtube.optStringOrNull("playbackQuality"),
                    youtubeObservedQuality = youtube.optStringOrNull("observedQuality"),
                    youtubeAvailableQualityLevels = youtube.optStringList("availableQualityLevels"),
                    youtubeLastActivityAgeMs = youtube.optLongOrNull("lastActivityAgeMs"),
                    youtubeKeepaliveReloadCount = youtube.optIntOrNull("keepaliveReloadCount"),
                    youtubePlayerState = youtube.optIntOrNull("playerState"),
                    youtubeCurrentTimeSeconds = youtube.optDoubleOrNull("currentTimeSeconds"),
                    youtubeProbeAgeMs = youtube.optLongOrNull("probeAgeMs")
                )
            }.getOrNull()
        }

        private fun JSONObject?.optStringOrNull(key: String): String? {
            if (this == null || !has(key) || isNull(key)) return null
            return optString(key).takeIf { it.isNotBlank() }
        }

        private fun JSONObject?.optBooleanOrNull(key: String): Boolean? {
            if (this == null || !has(key) || isNull(key)) return null
            return optBoolean(key)
        }

        private fun JSONObject?.optIntOrNull(key: String): Int? {
            if (this == null || !has(key) || isNull(key)) return null
            return optInt(key)
        }

        private fun JSONObject?.optLongOrNull(key: String): Long? {
            if (this == null || !has(key) || isNull(key)) return null
            return when (val value = opt(key)) {
                is Number -> value.toLong()
                is String -> value.toLongOrNull()
                else -> null
            }
        }

        private fun JSONObject?.optDoubleOrNull(key: String): Double? {
            if (this == null || !has(key) || isNull(key)) return null
            return when (val value = opt(key)) {
                is Number -> value.toDouble()
                is String -> value.toDoubleOrNull()
                else -> null
            }
        }

        private fun JSONObject?.optStringList(key: String): List<String> {
            if (this == null || !has(key) || isNull(key)) return emptyList()
            val value = opt(key)
            return when (value) {
                is JSONArray -> buildList {
                    for (index in 0 until value.length()) {
                        value.optString(index).takeIf { it.isNotBlank() }?.let(::add)
                    }
                }
                is String -> value
                    .split(',')
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                else -> emptyList()
            }
        }
    }
}
