package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * JVM unit tests for [SkipReasonAggregator].
 *
 * The aggregator is the load-bearing piece that makes silent-skip
 * failures observable: every adapter calls [SkipReasonAggregator.record]
 * once per cycle with a possibly-null reason. The heartbeat builder
 * snapshots + resets per cycle and ships the delta on the wire as
 * `skip_reason_counts`.
 *
 * These tests cover:
 *   - Null-reason no-op (aggregator stays empty when adapters report success)
 *   - Per-source / per-reason counter accumulation
 *   - [SkipReasonAggregator.snapshotAndReset] empties state
 *   - Concurrent record() calls across threads land safely
 *   - Returned snapshot omits zero-count entries
 *   - Empty aggregator returns null (lets ApiClient skip the JSON field)
 */
class SkipReasonAggregatorTest {

    @Test
    fun `null reason is a no-op`() {
        val agg = SkipReasonAggregator()
        agg.record(SkipReasonAggregator.SOURCE_BLE, null)
        agg.record(SkipReasonAggregator.SOURCE_MDNS, null)
        assertTrue("Aggregator must stay empty after null records", agg.isEmpty())
        assertNull("snapshotAndReset returns null when nothing recorded", agg.snapshotAndReset())
    }

    @Test
    fun `single record produces one source one reason in snapshot`() {
        val agg = SkipReasonAggregator()
        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")

        val snap = agg.snapshotAndReset()
        assertNotNull(snap)
        assertEquals(1, snap!!.size)
        val ble = snap[SkipReasonAggregator.SOURCE_BLE]
        assertNotNull(ble)
        assertEquals(1, ble!!["permission_denied"])
    }

    @Test
    fun `multiple records on same source-reason pair accumulate`() {
        val agg = SkipReasonAggregator()
        repeat(7) { agg.record(SkipReasonAggregator.SOURCE_MDNS, "multicast_lock_failed") }

        val snap = agg.snapshotAndReset()
        assertNotNull(snap)
        assertEquals(7, snap!![SkipReasonAggregator.SOURCE_MDNS]!!["multicast_lock_failed"])
    }

    @Test
    fun `different reasons under same source land in same source bucket`() {
        val agg = SkipReasonAggregator()
        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")
        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")
        agg.record(SkipReasonAggregator.SOURCE_BLE, "scanner_unavailable")

        val snap = agg.snapshotAndReset()!!
        val ble = snap[SkipReasonAggregator.SOURCE_BLE]!!
        assertEquals(2, ble["permission_denied"])
        assertEquals(1, ble["scanner_unavailable"])
    }

    @Test
    fun `different sources stay isolated`() {
        val agg = SkipReasonAggregator()
        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")
        agg.record(SkipReasonAggregator.SOURCE_MDNS, "ffi_failure")
        agg.record(SkipReasonAggregator.SOURCE_SSDP, "multicast_lock_failed")
        agg.record(SkipReasonAggregator.SOURCE_HTTP_PROBE, "no_response")

        val snap = agg.snapshotAndReset()!!
        assertEquals(4, snap.size)
        assertEquals(1, snap["ble"]!!["permission_denied"])
        assertEquals(1, snap["mdns"]!!["ffi_failure"])
        assertEquals(1, snap["ssdp"]!!["multicast_lock_failed"])
        assertEquals(1, snap["http_probe"]!!["no_response"])
    }

    @Test
    fun `snapshotAndReset clears state for next cycle`() {
        val agg = SkipReasonAggregator()
        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")
        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")

        // First snapshot grabs the counts.
        val first = agg.snapshotAndReset()!!
        assertEquals(2, first[SkipReasonAggregator.SOURCE_BLE]!!["permission_denied"])

        // Second snapshot must be null — counters are reset.
        val second = agg.snapshotAndReset()
        assertNull("Second snapshot must be null after reset", second)
        assertTrue(agg.isEmpty())
    }

    @Test
    fun `snapshotAndReset isolates cycles`() {
        // Cycle 1: 3 BLE perm denied, 1 MDNS ffi failure
        val agg = SkipReasonAggregator()
        repeat(3) { agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied") }
        agg.record(SkipReasonAggregator.SOURCE_MDNS, "ffi_failure")
        val cycle1 = agg.snapshotAndReset()!!
        assertEquals(3, cycle1["ble"]!!["permission_denied"])
        assertEquals(1, cycle1["mdns"]!!["ffi_failure"])

        // Cycle 2: 1 BLE bluetooth_disabled. Must NOT carry over cycle 1's counts.
        agg.record(SkipReasonAggregator.SOURCE_BLE, "bluetooth_disabled")
        val cycle2 = agg.snapshotAndReset()!!
        assertEquals(1, cycle2["ble"]!!["bluetooth_disabled"])
        assertNull("Cycle 2 must NOT contain permission_denied (was cycle-1)", cycle2["ble"]!!["permission_denied"])
        assertNull("Cycle 2 must NOT contain mdns at all", cycle2["mdns"])
    }

    @Test
    fun `concurrent records across threads accumulate correctly`() {
        // Smoke test for thread safety: 10 threads × 100 records on the
        // same (source, reason) pair = 1000 expected.
        val agg = SkipReasonAggregator()
        val threads = 10
        val perThread = 100
        val pool = Executors.newFixedThreadPool(threads)
        val latch = CountDownLatch(threads)
        try {
            repeat(threads) {
                pool.submit {
                    repeat(perThread) {
                        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")
                    }
                    latch.countDown()
                }
            }
            assertTrue("All record threads must finish", latch.await(10, TimeUnit.SECONDS))
        } finally {
            pool.shutdown()
        }

        val snap = agg.snapshotAndReset()!!
        assertEquals(threads * perThread, snap["ble"]!!["permission_denied"])
    }

    @Test
    fun `isEmpty returns true initially and after reset`() {
        val agg = SkipReasonAggregator()
        assertTrue("Fresh aggregator is empty", agg.isEmpty())

        agg.record(SkipReasonAggregator.SOURCE_BLE, "permission_denied")
        assertFalse("Aggregator with one record is not empty", agg.isEmpty())

        agg.snapshotAndReset()
        assertTrue("Aggregator after snapshot+reset is empty", agg.isEmpty())
    }

    @Test
    fun `source name constants match wire-format keys`() {
        // Server-side SignalIngestService.normalize() expects exactly
        // these snake_case-but-shorter keys. Don't let a typo land in
        // the aggregator and silently produce wire-format-incompatible
        // keys.
        assertEquals("ble", SkipReasonAggregator.SOURCE_BLE)
        assertEquals("mdns", SkipReasonAggregator.SOURCE_MDNS)
        assertEquals("ssdp", SkipReasonAggregator.SOURCE_SSDP)
        assertEquals("http_probe", SkipReasonAggregator.SOURCE_HTTP_PROBE)
    }
}
