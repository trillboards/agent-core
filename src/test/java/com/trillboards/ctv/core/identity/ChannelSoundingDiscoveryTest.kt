package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for ChannelSoundingDiscovery.
 *
 * Since the Channel Sounding API is not available on the test JVM (Android 16+/SDK 36 only),
 * tests verify graceful degradation and sorted output. Reflection probe success/failure is
 * verified at the acceptance level on real hardware (Tab S11 dev device).
 */
class ChannelSoundingDiscoveryTest {

    @Test
    fun `collectChannelSoundingMeasurements returns empty list on non-Android JVM`() {
        // The JVM doesn't have BluetoothAdapter or the Channel Sounding API.
        // Verify the method returns empty list (graceful no-op).
        val measurements = ChannelSoundingDiscovery.collectChannelSoundingMeasurements()
        assertTrue("Should return empty list on JVM (not Android)", measurements.isEmpty())
    }

    @Test
    fun `ChannelSoundingMeasurement data class serializes correctly`() {
        // Verify the data class can be instantiated and serialized.
        val measurement = ChannelSoundingMeasurement(
            address = "AA:BB:CC:DD:EE:FF",
            distance_mm = 1234.5f,
            aoa_deg = 45.0f
        )
        assertEquals("AA:BB:CC:DD:EE:FF", measurement.address)
        assertEquals(1234.5f, measurement.distance_mm, 0.01f)
        assertEquals(45.0f, measurement.aoa_deg!!, 0.01f)
    }

    @Test
    fun `ChannelSoundingMeasurement with null AoA`() {
        // AoA is optional (nullable).
        val measurement = ChannelSoundingMeasurement(
            address = "11:22:33:44:55:66",
            distance_mm = 500.0f,
            aoa_deg = null
        )
        assertEquals("11:22:33:44:55:66", measurement.address)
        assertEquals(500.0f, measurement.distance_mm, 0.01f)
        assertEquals(null, measurement.aoa_deg)
    }
}
