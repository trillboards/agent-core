package com.trillboards.ctv.core.identity

import java.security.MessageDigest

/**
 * Shared cryptographic utilities for identity collectors.
 *
 * Centralizes SHA-256 hashing used across BleBeaconScanner, MdnsDiscovery,
 * WifiScanCollector, and WiFiSignalCollector to eliminate duplication.
 */
object CryptoUtils {

    /**
     * SHA-256 hash of the input string, returned as a lowercase hex string.
     *
     * @param input The string to hash
     * @return 64-character lowercase hex-encoded SHA-256 digest
     */
    fun sha256(input: String): String {
        return MessageDigest.getInstance("SHA-256")
            .digest(input.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
    }
}
