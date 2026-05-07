package com.trillboards.ctv.core.stability

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LatestWinsCommandGateTest {

    private data class TestCommand(
        val id: String,
        val orderKey: Long
    )

    @Test
    fun `newer pending command replaces older pending command`() {
        val gate = LatestWinsCommandGate<TestCommand> { it.orderKey }

        val first = TestCommand(id = "older", orderKey = 100L)
        val second = TestCommand(id = "newer", orderKey = 200L)

        val firstResult = gate.offer(first)
        val secondResult = gate.offer(second)

        assertEquals(LatestWinsCommandGate.OfferStatus.ENQUEUED, firstResult.status)
        assertEquals(LatestWinsCommandGate.OfferStatus.REPLACED_PENDING, secondResult.status)
        assertEquals(first, secondResult.replacedCommand)
        assertEquals(second, gate.takeNext())
        assertNull(gate.takeNext())
    }

    @Test
    fun `older command is rejected after newer command already applied`() {
        val gate = LatestWinsCommandGate<TestCommand> { it.orderKey }

        val latest = TestCommand(id = "latest", orderKey = 300L)
        val stale = TestCommand(id = "stale", orderKey = 250L)

        gate.offer(latest)
        gate.markApplied(gate.takeNext()!!)

        val staleResult = gate.offer(stale)

        assertEquals(LatestWinsCommandGate.OfferStatus.STALE_ALREADY_APPLIED, staleResult.status)
        assertNull(gate.takeNext())
    }

    @Test
    fun `newer command still enters after a previous command was applied`() {
        val gate = LatestWinsCommandGate<TestCommand> { it.orderKey }

        val applied = TestCommand(id = "applied", orderKey = 100L)
        val next = TestCommand(id = "next", orderKey = 101L)

        gate.offer(applied)
        gate.markApplied(gate.takeNext()!!)

        val result = gate.offer(next)

        assertEquals(LatestWinsCommandGate.OfferStatus.ENQUEUED, result.status)
        assertTrue(gate.takeNext() == next)
    }

    @Test
    fun `older replay is rejected while newer command is in flight`() {
        val gate = LatestWinsCommandGate<TestCommand> { it.orderKey }

        val newest = TestCommand(id = "newest", orderKey = 300L)
        val staleReplay = TestCommand(id = "stale-replay", orderKey = 250L)
        val evenNewer = TestCommand(id = "even-newer", orderKey = 350L)

        gate.offer(newest)
        val inFlight = gate.takeNext()

        assertEquals(newest, inFlight)
        assertEquals(
            LatestWinsCommandGate.OfferStatus.STALE_ALREADY_APPLIED,
            gate.offer(staleReplay).status
        )
        assertEquals(
            LatestWinsCommandGate.OfferStatus.ENQUEUED,
            gate.offer(evenNewer).status
        )
        assertEquals(evenNewer, gate.takeNext())
    }
}
