package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VlmFrameAdmissionGateTest {

    @Test
    fun `tryAcquire accepts first frame`() {
        val gate = VlmFrameAdmissionGate()

        assertTrue(gate.tryAcquire(nowMs = 1_000L, minIntervalMs = 5_000L))
    }

    @Test
    fun `tryAcquire rejects frame while inference is in flight`() {
        val gate = VlmFrameAdmissionGate()

        assertTrue(gate.tryAcquire(nowMs = 1_000L, minIntervalMs = 5_000L))
        assertFalse(gate.tryAcquire(nowMs = 7_000L, minIntervalMs = 5_000L))
    }

    @Test
    fun `tryAcquire recovers stale in flight state`() {
        val gate = VlmFrameAdmissionGate()

        assertTrue(gate.tryAcquire(nowMs = 1_000L, minIntervalMs = 5_000L, maxInFlightMs = 10_000L))
        assertEquals(9_999L, gate.inFlightAgeMs(nowMs = 10_999L))
        assertFalse(gate.tryAcquire(nowMs = 10_999L, minIntervalMs = 5_000L, maxInFlightMs = 10_000L))

        assertTrue(gate.tryAcquire(nowMs = 11_001L, minIntervalMs = 5_000L, maxInFlightMs = 10_000L))
        assertEquals(0L, gate.inFlightAgeMs(nowMs = 11_001L))
    }

    @Test
    fun `release allows a new frame after the interval elapses`() {
        val gate = VlmFrameAdmissionGate()

        assertTrue(gate.tryAcquire(nowMs = 1_000L, minIntervalMs = 5_000L))
        assertEquals(2_500L, gate.inFlightAgeMs(nowMs = 3_500L))
        gate.release()
        assertNull(gate.inFlightAgeMs(nowMs = 3_501L))

        assertFalse(gate.tryAcquire(nowMs = 5_999L, minIntervalMs = 5_000L))
        assertTrue(gate.tryAcquire(nowMs = 6_000L, minIntervalMs = 5_000L))
    }

    @Test
    fun `reset clears in flight and timing state`() {
        val gate = VlmFrameAdmissionGate()

        assertTrue(gate.tryAcquire(nowMs = 1_000L, minIntervalMs = 5_000L))
        gate.reset()

        assertTrue(gate.tryAcquire(nowMs = 1_100L, minIntervalMs = 5_000L))
    }
}
