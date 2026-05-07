package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Unit tests for the post-2026-05-01-redesign NativeSensorCollector.
 *
 * The collector now harvests only TYPE_LIGHT + TYPE_PRESSURE — the
 * detectors for accelerometer / gyroscope / magnetometer / proximity were
 * retired alongside CH migration 059 (which drops their target columns).
 * The Android SensorManager binding is out of scope for these unit tests
 * (would require an instrumented test on a real device); we only verify
 * the data-class shape stays in lock-step with the wire contract enforced
 * by ApiClient.payloadToJson.
 */
class NativeSensorCollectorTest {

    @Test
    fun `NativeSensorSnapshot only carries ambient_light and barometer fields`() {
        val snap = NativeSensorSnapshot(
            ambientLightLux = 320.5f,
            barometerPressureHpa = 1013.25f,
            sensorTimestampMs = 1730_000_000_000L,
        )
        assertEquals(320.5f, snap.ambientLightLux!!, 0.01f)
        assertEquals(1013.25f, snap.barometerPressureHpa!!, 0.01f)
        assertEquals(1730_000_000_000L, snap.sensorTimestampMs)
    }

    @Test
    fun `NativeSensorSnapshot accepts null fields on devices missing the sensor`() {
        val snap = NativeSensorSnapshot(
            ambientLightLux = null,
            barometerPressureHpa = null,
            sensorTimestampMs = 0L,
        )
        assertNotNull(snap)
        assertNull(snap.ambientLightLux)
        assertNull(snap.barometerPressureHpa)
    }
}
