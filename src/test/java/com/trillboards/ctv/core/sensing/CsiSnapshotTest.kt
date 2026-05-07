package com.trillboards.ctv.core.sensing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

/**
 * Unit tests for CsiSnapshot data class (mirrors agent-core-lite test).
 */
class CsiSnapshotTest {

    @Test
    fun `CsiSnapshot should hold all fields`() {
        val snapshot = CsiSnapshot(
            nodeId = 1,
            occupantCount = 3,
            motionScore = 0.75f,
            signalQuality = 0.95f,
            subcarrierCount = 52,
            captureRateHz = 100.0f,
            avgRssiDbm = -45,
            framesProcessed = 1000,
            framesDropped = 50,
            windowStartMs = 1000L,
            windowEndMs = 11000L,
            hardwareType = "esp32-s3"
        )

        assertEquals(1, snapshot.nodeId)
        assertEquals(3, snapshot.occupantCount)
        assertEquals(0.75f, snapshot.motionScore, 0.001f)
        assertEquals(0.95f, snapshot.signalQuality, 0.001f)
        assertEquals(52, snapshot.subcarrierCount)
        assertEquals(100.0f, snapshot.captureRateHz, 0.001f)
        assertEquals(-45, snapshot.avgRssiDbm)
        assertEquals(1000, snapshot.framesProcessed)
        assertEquals(50, snapshot.framesDropped)
        assertEquals(1000L, snapshot.windowStartMs)
        assertEquals(11000L, snapshot.windowEndMs)
        assertEquals("esp32-s3", snapshot.hardwareType)
    }

    @Test
    fun `CsiSnapshot should use default hardware type`() {
        val snapshot = CsiSnapshot(
            nodeId = 0,
            occupantCount = 0,
            motionScore = 0.0f,
            signalQuality = 1.0f,
            subcarrierCount = 52,
            captureRateHz = 50.0f,
            avgRssiDbm = -60,
            framesProcessed = 500,
            framesDropped = 0,
            windowStartMs = 0L,
            windowEndMs = 10000L
        )

        assertEquals("esp32-s3", snapshot.hardwareType)
    }

    @Test
    fun `CsiSnapshot data class copy should preserve fields`() {
        val original = CsiSnapshot(
            nodeId = 1,
            occupantCount = 5,
            motionScore = 0.5f,
            signalQuality = 0.9f,
            subcarrierCount = 52,
            captureRateHz = 100.0f,
            avgRssiDbm = -40,
            framesProcessed = 900,
            framesDropped = 100,
            windowStartMs = 5000L,
            windowEndMs = 15000L
        )

        val copy = original.copy(occupantCount = 10)

        assertEquals(10, copy.occupantCount)
        assertEquals(1, copy.nodeId)
        assertEquals(0.5f, copy.motionScore, 0.001f)
    }

    @Test
    fun `CsiSnapshot equality should compare all fields`() {
        val a = CsiSnapshot(1, 3, 0.5f, 0.9f, 52, 100f, -40, 900, 100, 0L, 10000L)
        val b = CsiSnapshot(1, 3, 0.5f, 0.9f, 52, 100f, -40, 900, 100, 0L, 10000L)
        val c = CsiSnapshot(2, 3, 0.5f, 0.9f, 52, 100f, -40, 900, 100, 0L, 10000L)

        assertEquals(a, b)
        assertNotEquals(a, c)
    }
}
