package com.trillboards.ctv.core.bridge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OverlayHealthSnapshotTest {

    @Test
    fun `parses nested overlay health json into heartbeat-friendly fields`() {
        val snapshot = OverlayHealthSnapshot.fromJson(
            """
            {
              "capturedAtMs": 1713920000123,
              "reason": "interval",
              "uptimeMs": 60000,
              "documentHidden": false,
              "memory": {
                "jsHeapUsedMb": 73,
                "jsHeapTotalMb": 128,
                "blobUrlCount": 4,
                "blobPinnedMb": 12.5
              },
              "socket": {
                "connected": true,
                "disconnectReason": "transport close"
              },
              "stream": {
                "currentStreamId": "stream-123",
                "currentStreamType": "youtube",
                "queueLength": 3,
                "prefetchedUrlCount": 2
              },
              "youtube": {
                "requestedQuality": "highres",
                "playbackQuality": "hd1080",
                "observedQuality": "hd1080",
                "availableQualityLevels": ["highres", "hd1080", "hd720"],
                "lastActivityAgeMs": 1200,
                "keepaliveReloadCount": 1,
                "playerState": 1,
                "currentTimeSeconds": 42.5,
                "probeAgeMs": 900
              }
            }
            """.trimIndent(),
            receivedAtMs = 1713920001123
        )

        requireNotNull(snapshot)
        assertEquals("interval", snapshot.reason)
        assertEquals("stream-123", snapshot.currentStreamId)
        assertEquals("youtube", snapshot.currentStreamType)
        assertEquals(73, snapshot.jsHeapUsedMb)
        assertEquals(128, snapshot.jsHeapTotalMb)
        assertEquals(4, snapshot.blobUrlCount)
        assertEquals(12.5, requireNotNull(snapshot.blobPinnedMb), 0.001)
        assertTrue(snapshot.socketConnected == true)
        assertEquals("transport close", snapshot.socketDisconnectReason)
        assertEquals("highres", snapshot.youtubeRequestedQuality)
        assertEquals("hd1080", snapshot.youtubePlaybackQuality)
        assertEquals("hd1080", snapshot.youtubeObservedQuality)
        assertEquals(listOf("highres", "hd1080", "hd720"), snapshot.youtubeAvailableQualityLevels)

        val metadata = snapshot.toHeartbeatMetadata(nowMs = 1713920005123)
        assertEquals(4000L, metadata["overlayHealthAgeMs"])
        assertEquals(73, metadata["overlayJsHeapUsedMb"])
        assertEquals("youtube", metadata["overlayCurrentStreamType"])
        assertEquals(true, metadata["overlaySocketConnected"])
        assertEquals("hd1080", metadata["overlayYouTubeObservedQuality"])
        assertEquals("highres,hd1080,hd720", metadata["overlayYouTubeAvailableQualityLevels"])
    }

    @Test
    fun `returns null for blank or malformed payloads`() {
        assertEquals(null, OverlayHealthSnapshot.fromJson(null))
        assertEquals(null, OverlayHealthSnapshot.fromJson("  "))
        assertEquals(null, OverlayHealthSnapshot.fromJson("{bad json"))
    }

    @Test
    fun `heartbeat metadata omits null-valued fields`() {
        val snapshot = OverlayHealthSnapshot(
            receivedAtMs = 1000L,
            capturedAtMs = null,
            reason = null,
            uptimeMs = 2000L,
            documentHidden = false,
            currentStreamId = null,
            currentStreamType = "uploaded",
            queueLength = null,
            prefetchedUrlCount = null,
            jsHeapUsedMb = null,
            jsHeapTotalMb = null,
            blobUrlCount = null,
            blobPinnedMb = null,
            socketConnected = null,
            socketDisconnectReason = null,
            youtubeRequestedQuality = null,
            youtubePlaybackQuality = null,
            youtubeObservedQuality = null,
            youtubeAvailableQualityLevels = emptyList(),
            youtubeLastActivityAgeMs = null,
            youtubeKeepaliveReloadCount = null,
            youtubePlayerState = null,
            youtubeCurrentTimeSeconds = null,
            youtubeProbeAgeMs = null
        )

        val metadata = snapshot.toHeartbeatMetadata(nowMs = 2500L)
        assertEquals(1500L, metadata["overlayHealthAgeMs"])
        assertEquals("uploaded", metadata["overlayCurrentStreamType"])
        assertEquals(false, metadata["overlayDocumentHidden"])
        assertFalse(metadata.containsKey("overlayJsHeapUsedMb"))
        assertFalse(metadata.containsKey("overlayYouTubeAvailableQualityLevels"))
    }
}
