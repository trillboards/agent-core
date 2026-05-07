package com.trillboards.ctv.core.identity

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Per-source skip-reason aggregator.
 *
 * Phase 0 of the `redis-stream-scale-fix` plan extends the structured
 * skip-reason channel introduced for BLE in PR #4480 to mDNS, SSDP, and
 * HTTP probe. The heartbeat builder calls every adapter once per
 * heartbeat; each call may bail with a non-null `skipReason`. Without
 * this aggregator, the wire payload would carry only the LAST cycle's
 * skip reason — useless for spotting periodic flaps. With it, we
 * accumulate counts since the previous heartbeat and ship deltas, not
 * running totals (running totals would over-weight long-uptime screens).
 *
 * Wire shape produced by [snapshotAndReset]:
 *
 * ```
 * {
 *   "ble":        { "permission_denied": N, "scanner_unavailable": N, ... },
 *   "mdns":       { "multicast_lock_failed": N, "ffi_failure": N },
 *   "ssdp":       { ... },
 *   "http_probe": { "no_response": N, ... }
 * }
 * ```
 *
 * Server-side `SignalIngestService.normalize()` accepts this map and emits
 * `signal_skip_reason_total{source, reason}` CW counters.
 *
 * Class fix: the 2026-05-03 SM-X730 silent-BLE incident lost 27 hours of
 * data because every silent-failure mode (perm denied, lock unavailable,
 * scanner crashed, FFI threw) returned `null` and looked identical to "no
 * devices in range." This aggregator is the load-bearing piece that makes
 * those failures observable per-screen at server cron cadence.
 *
 * Thread-safety: uses [ConcurrentHashMap] + [AtomicInteger] so adapter
 * results from any thread can land safely. The heartbeat builder is the
 * only reader; it calls [snapshotAndReset] once per cycle on the IO
 * dispatcher.
 */
class SkipReasonAggregator {

    // source → reason → counter. ConcurrentHashMap is sufficient; we
    // never iterate while writing in a way that requires snapshot
    // semantics (snapshotAndReset takes a fresh map by .toMap()).
    private val counters: ConcurrentHashMap<String, ConcurrentHashMap<String, AtomicInteger>> =
        ConcurrentHashMap()

    /**
     * Record one skip event for [source] / [reason]. Idempotent w.r.t.
     * the (source, reason) key — repeated calls increment the same
     * counter.
     *
     * Pass `null` reason as a no-op (success path doesn't increment).
     * That's a deliberate ergonomic choice: callers can write
     * `aggregator.record("ble", snapshot.skipReason)` without a guard.
     */
    fun record(source: String, reason: String?) {
        if (reason == null) return
        val perSource = counters.computeIfAbsent(source) { ConcurrentHashMap() }
        val counter = perSource.computeIfAbsent(reason) { AtomicInteger(0) }
        counter.incrementAndGet()
    }

    /**
     * Atomically extract the current counts and reset state. Heartbeat
     * builder calls this once per cycle and ships the returned map on
     * the wire. Calling it without any intervening [record] returns null
     * so the caller can omit `skipReasonCounts` from the JSON payload
     * entirely — server-side normalize() short-circuits on null and
     * skips emitting counter rows.
     *
     * Concurrency note: a record() racing with snapshotAndReset() may
     * land in either the snapshot OR the next snapshot, never both. We
     * accept the "may slip into next cycle" race because the alternative
     * (a full lock around every record + snapshot) would burn CPU in the
     * heartbeat hot path for no observable benefit — the server-side
     * counter aggregates over hours of heartbeats anyway.
     */
    fun snapshotAndReset(): Map<String, Map<String, Int>>? {
        if (counters.isEmpty()) return null

        val out = mutableMapOf<String, Map<String, Int>>()
        for (source in counters.keys.toList()) {
            val perSource = counters.remove(source) ?: continue
            val rendered = mutableMapOf<String, Int>()
            for ((reason, counter) in perSource) {
                val n = counter.get()
                if (n > 0) rendered[reason] = n
            }
            if (rendered.isNotEmpty()) {
                out[source] = rendered
            }
        }
        return out.takeIf { it.isNotEmpty() }
    }

    /**
     * Returns true when no events have been recorded since the last
     * snapshot. Test-only — production code calls [snapshotAndReset]
     * directly.
     */
    fun isEmpty(): Boolean = counters.isEmpty() ||
        counters.values.all { perSource -> perSource.values.all { it.get() == 0 } }

    companion object {
        // Source-name constants — kept here so the heartbeat builder
        // can't accidentally typo "blueooth" or "ssdo" in three different
        // files. These map 1:1 to the wire-format keys the server's
        // SignalIngestService.normalize() expects.
        const val SOURCE_BLE: String = "ble"
        const val SOURCE_MDNS: String = "mdns"
        const val SOURCE_SSDP: String = "ssdp"
        const val SOURCE_HTTP_PROBE: String = "http_probe"
    }
}
