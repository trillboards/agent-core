package com.trillboards.ctv.core.rendering

import java.net.URLEncoder

/**
 * Builds the screen-overlay URL with fingerprint, session, and optional device-token query params.
 *
 * Originally lived at `tablet-agent/.../OverlayUrlBuilder.kt`. Moved into agent-core so every
 * platform module can render the overlay with identical query-param shape — required by
 * server-side correlation between MDM-bound screen identity and the live overlay session.
 *
 * Pure function; no Android dependencies.
 */
object OverlayUrlBuilder {
    fun build(
        baseUrl: String,
        fingerprint: String,
        sessionId: String,
        deviceToken: String?
    ): String {
        val separator = if (baseUrl.contains("?")) "&" else "?"
        val encodedFingerprint = URLEncoder.encode(fingerprint, "UTF-8")
        val encodedSessionId = URLEncoder.encode(sessionId, "UTF-8")
        val tokenParam = if (!deviceToken.isNullOrEmpty()) {
            "&device_token=${URLEncoder.encode(deviceToken, "UTF-8")}"
        } else {
            ""
        }

        return "${baseUrl}${separator}fingerprint=$encodedFingerprint&session=$encodedSessionId$tokenParam"
    }
}
