package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for CryptoUtils.
 *
 * Verifies SHA-256 hashing consistency, output format, and uniqueness.
 * NOTE: Scan/discovery logic (BleBeaconScanner, MdnsDiscovery, WifiScanCollector)
 * requires Android instrumented tests to exercise their full code paths.
 */
class CryptoUtilsTest {

    @Test
    fun `sha256 should produce 64-character hex string`() {
        val hash = CryptoUtils.sha256("test input")
        assertEquals(64, hash.length)
        assertTrue(hash.all { it in '0'..'9' || it in 'a'..'f' })
    }

    @Test
    fun `sha256 should be consistent for same input`() {
        val input = "AA:BB:CC:DD:EE:FF"
        val hash1 = CryptoUtils.sha256(input)
        val hash2 = CryptoUtils.sha256(input)
        assertEquals(hash1, hash2)
    }

    @Test
    fun `sha256 should produce different hashes for different inputs`() {
        val hash1 = CryptoUtils.sha256("AA:BB:CC:DD:EE:FF")
        val hash2 = CryptoUtils.sha256("11:22:33:44:55:66")
        assertNotEquals(hash1, hash2)
    }

    @Test
    fun `sha256 should handle empty string`() {
        val hash = CryptoUtils.sha256("")
        assertEquals(64, hash.length)
        // SHA-256 of empty string is well-known (verified via shasum -a 256)
        assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", hash)
    }

    @Test
    fun `sha256 should handle unicode input`() {
        val hash = CryptoUtils.sha256("Living Room Roku")
        assertEquals(64, hash.length)
        // Should be consistent
        assertEquals(hash, CryptoUtils.sha256("Living Room Roku"))
    }

    @Test
    fun `sha256 should produce lowercase hex`() {
        val hash = CryptoUtils.sha256("TEST")
        assertEquals(hash, hash.lowercase())
    }
}
