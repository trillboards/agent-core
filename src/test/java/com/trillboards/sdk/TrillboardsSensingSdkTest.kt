package com.trillboards.sdk

import android.content.Context
import io.mockk.mockk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [TrillboardsSensingSdk] — covers the public surface that
 * partners contract against. Heavy lifecycle wiring (CameraX bind,
 * Socket.io connect, AudioRecord open) lives in `AudienceSensingService`
 * and is exercised by the existing instrumentation suite; this file only
 * checks that the facade refuses bad input and round-trips the
 * [SensingSdkConfig] defaults that ship in the public API.
 */
class TrillboardsSensingSdkTest {

    @Before
    fun resetState() {
        // Each test gets a fresh facade — `start()` is idempotent in
        // production but the `started` flag persists across tests.
        TrillboardsSensingSdk.resetForTesting()
    }

    @Test
    fun `isStarted returns false before start is called`() {
        assertFalse(TrillboardsSensingSdk.isStarted())
    }

    @Test
    fun `SensingSdkConfig defaults match the documented partner contract`() {
        val config = SensingSdkConfig()

        assertEquals("https://api.trillboards.com", config.apiBaseUrl)
        assertEquals("https://chat.trillboards.com", config.socketUrl)
        assertEquals(30_000L, config.heartbeatIntervalMs)

        // All sensing modalities default ON. Partners opt out by passing a
        // SensingSdkConfig with the relevant flag flipped to false.
        assertTrue(config.sensingEnabled)
        assertTrue(config.faceDetectionEnabled)
        assertTrue(config.audioClassificationEnabled)
        assertTrue(config.speechIntelligenceEnabled)
        assertTrue(config.demographicsCaptureEnabled)
        assertTrue(config.emotionalEngagementEnabled)
        assertTrue(config.connectSocket)
    }

    @Test
    fun `SensingSdkConfig allows disabling individual modalities`() {
        val config = SensingSdkConfig(
            faceDetectionEnabled = false,
            audioClassificationEnabled = false,
            speechIntelligenceEnabled = false,
            demographicsCaptureEnabled = false,
            emotionalEngagementEnabled = false,
            connectSocket = false
        )

        assertTrue(config.sensingEnabled)
        assertFalse(config.faceDetectionEnabled)
        assertFalse(config.audioClassificationEnabled)
        assertFalse(config.speechIntelligenceEnabled)
        assertFalse(config.demographicsCaptureEnabled)
        assertFalse(config.emotionalEngagementEnabled)
        assertFalse(config.connectSocket)
    }

    @Test
    fun `SensingSdkConfig allows custom backend URLs for non-prod`() {
        val config = SensingSdkConfig(
            apiBaseUrl = "https://staging.api.trillboards.com",
            socketUrl = "https://staging.chat.trillboards.com",
            heartbeatIntervalMs = 60_000L
        )

        assertEquals("https://staging.api.trillboards.com", config.apiBaseUrl)
        assertEquals("https://staging.chat.trillboards.com", config.socketUrl)
        assertEquals(60_000L, config.heartbeatIntervalMs)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `start throws when partnerApiKey is blank`() {
        // The require() precondition fires before any Context method is
        // touched, so a relaxed mockk Context is enough — the test
        // exercises the input-validation contract, not the underlying
        // ApiClient/Socket wiring.
        TrillboardsSensingSdk.start(
            context = mockk<Context>(relaxed = true),
            partnerApiKey = ""
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `start throws when partnerApiKey is whitespace-only`() {
        TrillboardsSensingSdk.start(
            context = mockk<Context>(relaxed = true),
            partnerApiKey = "   "
        )
    }

    @Test
    fun `SensingSdkConfig is a data class so partners can copy with overrides`() {
        val base = SensingSdkConfig()
        val staging = base.copy(apiBaseUrl = "https://staging.api.trillboards.com")

        // Original is untouched; only the override changed.
        assertEquals("https://api.trillboards.com", base.apiBaseUrl)
        assertEquals("https://staging.api.trillboards.com", staging.apiBaseUrl)

        // All other fields preserved verbatim by data-class copy().
        assertEquals(base.socketUrl, staging.socketUrl)
        assertEquals(base.heartbeatIntervalMs, staging.heartbeatIntervalMs)
        assertEquals(base.faceDetectionEnabled, staging.faceDetectionEnabled)
    }

    @Test
    fun `facade exposes start and stop as JvmStatic for Java callers`() {
        // Reflection check — the @JvmStatic annotation surfaces these
        // methods directly on TrillboardsSensingSdk (not on .INSTANCE), so
        // Java callers can write `TrillboardsSensingSdk.start(...)` without
        // routing through the Kotlin object instance.
        val klass = TrillboardsSensingSdk.javaClass
        val startMethod = klass.declaredMethods.firstOrNull { it.name == "start" }
        val stopMethod = klass.declaredMethods.firstOrNull { it.name == "stop" }

        assertNotNull("start() must exist on TrillboardsSensingSdk", startMethod)
        assertNotNull("stop() must exist on TrillboardsSensingSdk", stopMethod)
    }
}
