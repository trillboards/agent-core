package com.trillboards.ctv.core.rendering.models

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContentParserTest {

    @Test
    fun `parses structured content manifests from the current overlay API`() {
        val json = JSONObject(
            """
            {
              "status": true,
              "code": 200,
              "message": "Content retrieved successfully for overlay system",
              "data": {
                "background_stream": {
                  "type": "default_stream",
                  "content": {
                    "id": "stream-2",
                    "name": "Current YouTube Stream",
                    "youtube_url": "https://www.youtube.com/watch?v=abc123xyz00",
                    "stream_type": "youtube",
                    "duration_seconds": 120,
                    "playback_order": 2
                  }
                },
                "overlay_advertisements": [
                  {
                    "type": "advertisement",
                    "priority": 1,
                    "content": {
                      "id": "ad-1",
                      "mediaType": "image",
                      "image": ["https://example.com/ad-1.jpg"],
                      "name": "Launch Ad",
                      "campaign_link": "https://example.com/campaign",
                      "selfPromo": false,
                      "allocation_id": "alloc-1",
                      "instant_ad": false
                    }
                  }
                ],
                "playback_policy": {
                  "paid_interval_ms": 30000
                },
                "display_preferences": {
                  "display_mode": "fullscreen",
                  "l_bar_ads_enabled": false
                },
                "all_default_streams": [
                  {
                    "type": "default_stream",
                    "content": {
                      "id": "stream-1",
                      "name": "Uploaded MP4",
                      "video_url": "https://cdn.example.com/one.mp4",
                      "stream_type": "uploaded",
                      "duration_seconds": 45,
                      "playback_order": 1
                    }
                  },
                  {
                    "type": "default_stream",
                    "content": {
                      "id": "stream-2",
                      "name": "Current YouTube Stream",
                      "youtube_url": "https://www.youtube.com/watch?v=abc123xyz00",
                      "stream_type": "youtube",
                      "duration_seconds": 120,
                      "playback_order": 2
                    }
                  },
                  {
                    "type": "default_stream",
                    "content": {
                      "id": "stream-3",
                      "name": "Slides Deck",
                      "google_slides_url": "https://docs.google.com/presentation/d/demo",
                      "stream_type": "slides",
                      "duration_seconds": 0,
                      "playback_order": 3
                    }
                  }
                ],
                "playback_state": {
                  "queue_position": 1,
                  "stream_id": "stream-2"
                }
              }
            }
            """.trimIndent()
        )

        val response = ContentParser.parseContentResponse(json, "etag-123")

        assertEquals("etag-123", response.etag)
        assertEquals(1, response.ads.size)
        assertEquals("ad-1", response.ads.first().id)
        assertEquals("ad-1", response.ads.first().advertisementId)
        assertEquals("alloc-1", response.ads.first().impressionId)
        assertEquals(listOf("https://example.com/ad-1.jpg"), response.ads.first().imageUrls)

        assertEquals(listOf("stream-2", "stream-3", "stream-1"), response.streams.map { it.id })
        assertEquals("youtube", response.streams[0].streamType)
        assertEquals("abc123xyz00", response.streams[0].youtubeId)
        assertNull(response.streams[0].contentUrl)
        assertEquals("slides", response.streams[1].streamType)
        assertEquals("https://cdn.example.com/one.mp4", response.streams[2].contentUrl)

        assertEquals("fullscreen", response.displayConfig.displayMode)
        assertEquals(30_000L, response.displayConfig.adIntervalMs)
        assertTrue(response.displayConfig.streamEnabled)
    }

    @Test
    fun `preserves legacy array manifest parsing`() {
        val json = JSONObject(
            """
            {
              "data": [
                {
                  "type": "default_stream",
                  "content": {
                    "id": "legacy-stream",
                    "content_url": "https://cdn.example.com/legacy.m3u8",
                    "stream_type": "uploaded",
                    "duration": 90
                  }
                },
                {
                  "type": "advertisement",
                  "id": "legacy-ad",
                  "photo": "https://example.com/legacy.jpg",
                  "duration": 15
                }
              ],
              "config": {
                "display_mode": "fullscreen",
                "l_bar_enabled": true,
                "ad_interval_ms": 45000,
                "stream_enabled": true
              },
              "waterfall": {
                "global_timeout_ms": 15000,
                "sources": [
                  {
                    "name": "primary",
                    "vast_url": "https://ads.example.com/vast.xml",
                    "timeout_ms": 5000,
                    "priority": 0,
                    "cpm": 1.5
                  }
                ]
              }
            }
            """.trimIndent()
        )

        val response = ContentParser.parseContentResponse(json, null)

        assertEquals(1, response.ads.size)
        assertEquals("legacy-ad", response.ads.first().id)
        assertEquals(listOf("https://example.com/legacy.jpg"), response.ads.first().imageUrls)

        assertEquals(1, response.streams.size)
        assertEquals("legacy-stream", response.streams.first().id)
        assertEquals("https://cdn.example.com/legacy.m3u8", response.streams.first().contentUrl)

        assertEquals("fullscreen", response.displayConfig.displayMode)
        assertEquals(45_000L, response.displayConfig.adIntervalMs)
        assertTrue(response.displayConfig.lBarEnabled)
        assertEquals(1, response.waterfallConfig?.sources?.size)
        assertEquals("primary", response.waterfallConfig?.sources?.first()?.name)
    }
}
