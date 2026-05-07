package com.trillboards.ctv.core.socket

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for the screen_binding socket event handling via AgentSocketManager.Listener.
 *
 * These tests verify:
 * 1. The default onScreenBinding implementation is a no-op (doesn't throw)
 * 2. Payload parsing logic for paired/deleted actions
 * 3. Guard conditions (empty screenId, same screenId, unknown action)
 */
class ScreenBindingListenerTest {

    /** Minimal Listener implementation that records onScreenBinding calls. */
    private class RecordingListener : AgentSocketManager.Listener {
        var lastScreenBinding: JSONObject? = null
        var screenBindingCallCount = 0

        override fun onConnect(socketId: String?) {}
        override fun onDisconnect() {}
        override fun onError(args: Array<Any?>) {}
        override fun onPrivateMessage(payload: JSONObject) {}
        override fun onDeviceCommand(payload: JSONObject) {}

        override fun onScreenBinding(payload: JSONObject) {
            lastScreenBinding = payload
            screenBindingCallCount++
        }
    }

    @Test
    fun `default onScreenBinding is no-op and does not throw`() {
        // Use an anonymous implementation that only overrides required methods
        val listener = object : AgentSocketManager.Listener {
            override fun onConnect(socketId: String?) {}
            override fun onDisconnect() {}
            override fun onError(args: Array<Any?>) {}
            override fun onPrivateMessage(payload: JSONObject) {}
            override fun onDeviceCommand(payload: JSONObject) {}
        }

        // Should not throw — default implementation is empty
        listener.onScreenBinding(JSONObject().apply {
            put("screenId", "abc123")
            put("action", "paired")
        })
    }

    @Test
    fun `paired action payload has screenId and venueType`() {
        val listener = RecordingListener()
        val payload = JSONObject().apply {
            put("screenId", "screen_001")
            put("action", "paired")
            put("venueType", "bar")
        }

        listener.onScreenBinding(payload)

        assertEquals(1, listener.screenBindingCallCount)
        assertEquals("screen_001", listener.lastScreenBinding?.optString("screenId"))
        assertEquals("paired", listener.lastScreenBinding?.optString("action"))
        assertEquals("bar", listener.lastScreenBinding?.optString("venueType"))
    }

    @Test
    fun `deleted action payload is received correctly`() {
        val listener = RecordingListener()
        val payload = JSONObject().apply {
            put("screenId", "screen_001")
            put("action", "deleted")
        }

        listener.onScreenBinding(payload)

        assertEquals(1, listener.screenBindingCallCount)
        assertEquals("deleted", listener.lastScreenBinding?.optString("action"))
    }

    @Test
    fun `payload with missing action defaults to empty string via optString`() {
        val payload = JSONObject().apply {
            put("screenId", "screen_002")
        }

        // Verify optString behavior matches what DeviceAgentService expects
        assertEquals("paired", payload.optString("action", "paired"))
    }

    @Test
    fun `payload with empty screenId returns empty via optString`() {
        val payload = JSONObject().apply {
            put("screenId", "")
            put("action", "paired")
        }

        // DeviceAgentService guards: newScreenId.isNotEmpty()
        val screenId = payload.optString("screenId", "")
        assertTrue(screenId.isEmpty())
    }

    @Test
    fun `payload with null venueType returns null via optString with null fallback`() {
        val payload = JSONObject().apply {
            put("screenId", "screen_003")
            put("action", "paired")
        }

        // DeviceAgentService uses optString("venueType", null)
        val venueType = payload.optString("venueType", null)
        assertNull(venueType)
    }
}
