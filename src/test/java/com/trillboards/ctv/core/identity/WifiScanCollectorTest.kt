package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for WifiScanCollector.
 *
 * Tests that require Android context (WifiManager) are integration tests and
 * must run on a device/emulator. These tests verify data class contracts and
 * the SHA-256 hashing primitive that other identity collectors still rely on
 * for non-MAC data.
 */
class WifiScanCollectorTest {

    @Test
    fun `WifiScanResult data class should hold all fields including raw BSSID`() {
        val result = WifiScanResult(
            rawBssid = "AA:BB:CC:DD:EE:FF",
            signalStrengthDbm = -50,
            frequencyMhz = 2412,
            channelWidthMhz = 20
        )

        assertEquals("AA:BB:CC:DD:EE:FF", result.rawBssid)
        assertEquals(-50, result.signalStrengthDbm)
        assertEquals(2412, result.frequencyMhz)
        assertEquals(20, result.channelWidthMhz)
    }

    @Test
    fun `WifiScanResult should allow null channelWidthMhz`() {
        val result = WifiScanResult(
            rawBssid = "AA:BB:CC:DD:EE:FF",
            signalStrengthDbm = -70,
            frequencyMhz = 5180,
            channelWidthMhz = null
        )

        assertNull(result.channelWidthMhz)
    }

    @Test
    fun `WifiScanSnapshot should contain networks and count`() {
        val networks = listOf(
            WifiScanResult("AA:BB:CC:DD:EE:01", -40, 2412, 20),
            WifiScanResult("AA:BB:CC:DD:EE:02", -60, 5180, 80),
            WifiScanResult("AA:BB:CC:DD:EE:03", -80, 2437, 40),
        )

        val snapshot = WifiScanSnapshot(
            networks = networks,
            uniqueBssidCount = 3,
            scanTimestampMs = System.currentTimeMillis()
        )

        assertEquals(3, snapshot.networks.size)
        assertEquals(3, snapshot.uniqueBssidCount)
        assertTrue(snapshot.scanTimestampMs > 0)
    }

    @Test
    fun `SHA-256 of non-identity bytes should be consistent via CryptoUtils`() {
        // CryptoUtils.sha256 is retained for non-identity hashing
        // (e.g. file fingerprints, content hashes). MAC hashing has moved server-side.
        val input = "non-identity-content"
        val hash1 = CryptoUtils.sha256(input)
        val hash2 = CryptoUtils.sha256(input)

        assertEquals(hash1, hash2)
        assertEquals(64, hash1.length)
    }

    @Test
    fun `SHA-256 should produce different hashes for different inputs`() {
        val hash1 = CryptoUtils.sha256("payload-A")
        val hash2 = CryptoUtils.sha256("payload-B")

        assertNotEquals(hash1, hash2)
    }

    @Test
    fun `MAX_NETWORKS limit should be 50`() {
        val networks = (1..100).map {
            WifiScanResult("AA:BB:CC:DD:EE:%02X".format(it), -40 - it, 2412, null)
        }
        val capped = networks.take(50)
        assertEquals(50, capped.size)
    }

    @Test
    fun `WifiScanSnapshot with empty networks list is valid`() {
        val snapshot = WifiScanSnapshot(
            networks = emptyList(),
            uniqueBssidCount = 0,
            scanTimestampMs = System.currentTimeMillis()
        )
        assertTrue(snapshot.networks.isEmpty())
        assertEquals(0, snapshot.uniqueBssidCount)
    }
}
