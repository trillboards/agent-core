package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Unit tests for AuracastDiscovery.
 *
 * Note: Full Auracast discovery (with BluetoothLeBroadcastAssistant) requires
 * Android instrumented tests and API 33+ device/emulator. These unit tests
 * verify data class contracts only.
 */
class AuracastDiscoveryTest {

    @Test
    fun auracastBroadcast_equalsWork() {
        val broadcast1 = AuracastBroadcast(
            broadcastId = 1,
            broadcastName = "Airport TV",
            publicBroadcastData = byteArrayOf(0x01.toByte(), 0x02.toByte()),
            sourceId = 10
        )

        val broadcast2 = AuracastBroadcast(
            broadcastId = 1,
            broadcastName = "Airport TV",
            publicBroadcastData = byteArrayOf(0x01.toByte(), 0x02.toByte()),
            sourceId = 10
        )

        assertEquals(broadcast1, broadcast2)
    }

    @Test
    fun auracastBroadcast_hashCodeWork() {
        val broadcast1 = AuracastBroadcast(
            broadcastId = 1,
            broadcastName = "Test Broadcast",
            publicBroadcastData = byteArrayOf(0xAA.toByte(), 0xBB.toByte()),
            sourceId = 5
        )

        val broadcast2 = AuracastBroadcast(
            broadcastId = 1,
            broadcastName = "Test Broadcast",
            publicBroadcastData = byteArrayOf(0xAA.toByte(), 0xBB.toByte()),
            sourceId = 5
        )

        assertEquals(broadcast1.hashCode(), broadcast2.hashCode())
    }

    @Test
    fun auracastBroadcast_notEqualWhenDifferent() {
        val broadcast1 = AuracastBroadcast(
            broadcastId = 1,
            broadcastName = "Broadcast A",
            publicBroadcastData = null,
            sourceId = 1
        )

        val broadcast2 = AuracastBroadcast(
            broadcastId = 2,
            broadcastName = "Broadcast B",
            publicBroadcastData = null,
            sourceId = 2
        )

        assert(broadcast1 != broadcast2)
    }

    @Test
    fun auracastSnapshot_createsWithDefaultValues() {
        val snapshot = AuracastSnapshot(
            broadcasts = emptyList(),
            broadcastCount = 0,
            discoveryDurationMs = 1000L,
            discoveryTimestampMs = System.currentTimeMillis(),
            skipReason = null
        )

        assertEquals(0, snapshot.broadcasts.size)
        assertEquals(0, snapshot.broadcastCount)
        assertNull(snapshot.skipReason)
    }

    @Test
    fun auracastSnapshot_createsWithSkipReason() {
        val snapshot = AuracastSnapshot(
            broadcasts = emptyList(),
            broadcastCount = 0,
            discoveryDurationMs = 500L,
            discoveryTimestampMs = System.currentTimeMillis(),
            skipReason = "permission_denied"
        )

        assertEquals("permission_denied", snapshot.skipReason)
    }
}
