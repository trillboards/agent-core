package com.trillboards.ctv.core.rendering

import org.junit.Assert.assertEquals
import org.junit.Test

class OverlayUrlBuilderTest {

    @Test
    fun `build appends query params to clean base url`() {
        val url = OverlayUrlBuilder.build(
            baseUrl = "https://screen.trillboards.com",
            fingerprint = "fp-1",
            sessionId = "session-1",
            deviceToken = "token-1"
        )

        assertEquals(
            "https://screen.trillboards.com?fingerprint=fp-1&session=session-1&device_token=token-1",
            url
        )
    }

    @Test
    fun `build preserves existing query params`() {
        val url = OverlayUrlBuilder.build(
            baseUrl = "http://127.0.0.1:8001?apiBase=http://127.0.0.1:8001&appBase=http://127.0.0.1:8001&socketBase=https://chat.trillboards.com",
            fingerprint = "fp-1",
            sessionId = "session-1",
            deviceToken = "token-1"
        )

        assertEquals(
            "http://127.0.0.1:8001?apiBase=http://127.0.0.1:8001&appBase=http://127.0.0.1:8001&socketBase=https://chat.trillboards.com&fingerprint=fp-1&session=session-1&device_token=token-1",
            url
        )
    }

    @Test
    fun `build omits device_token param when null or blank`() {
        val urlNullToken = OverlayUrlBuilder.build(
            baseUrl = "https://screen.trillboards.com",
            fingerprint = "fp-1",
            sessionId = "session-1",
            deviceToken = null
        )
        val urlBlankToken = OverlayUrlBuilder.build(
            baseUrl = "https://screen.trillboards.com",
            fingerprint = "fp-1",
            sessionId = "session-1",
            deviceToken = ""
        )

        assertEquals("https://screen.trillboards.com?fingerprint=fp-1&session=session-1", urlNullToken)
        assertEquals("https://screen.trillboards.com?fingerprint=fp-1&session=session-1", urlBlankToken)
    }
}
