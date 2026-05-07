package com.trillboards.ctv.core.identity

import android.content.Context
import com.trillboards.discovery.MdnsAdapter

/**
 * Phase 1 — thin compatibility shim over the Rust-backed [MdnsAdapter].
 *
 * The historical `MdnsDiscovery.kt` in this fork (and the two sibling forks
 * `agent-core-lite` / `trillboards-measurement-sdk`) used Android's
 * [android.net.nsd.NsdManager] to discover smart devices on the local LAN.
 * That API is single-flight per service type at the C++ NSD daemon — listening
 * for >2-3 service types in parallel silently drops responses on most OEM
 * builds, and the per-fork copy-paste left three subtly different drift trails
 * to maintain. Phase 1 of the multi-protocol device discovery rewrite (plan
 * `reflective-whistling-rabin.md`) replaces those three files with a single
 * Rust core wrapping the `mdns-sd` 0.19.1 crate, surfaced in Kotlin via
 * UniFFI and the [com.trillboards.discovery] AAR.
 *
 * Wire format contract: this file preserves the legacy `NetworkDevice` and
 * `MdnsSnapshot` data classes BIT-FOR-BIT. `HeartbeatPayload.kt` and
 * `ApiClient.kt` are unchanged in Phase 1; the JSON shape on the wire
 * (`discovered_network_devices`, `network_device_count`) is byte-identical.
 *
 * What changed:
 *   - The `NsdManager` orchestration moved to Rust (one `ServiceDaemon`
 *     listening to N types).
 *   - The TXT-record extraction moved to Rust per-protocol schemas (replaces
 *     the deleted `MdnsTxtExtractor.kt`).
 *   - PII filtering moved to Rust BEFORE the FFI boundary — banned keys
 *     (`fn`, `name`, `id`, `srcid`, MAC patterns, emails) and personal-name
 *     instance labels never cross into Kotlin memory.
 *   - The `WifiManager.MulticastLock` is acquired inside `MdnsAdapter`
 *     (Kotlin), since the Rust core has no Android dependency.
 *
 * What did NOT change:
 *   - `discover(context, durationMs)` signature.
 *   - `NetworkDevice` field shape (`serviceType`, `instanceName`, `host`,
 *     `port`, `mdnsModel`, `mdnsVendor`, `mdnsSoftwareVersion`).
 *   - `MdnsSnapshot` field shape (`devices`, `deviceCount`,
 *     `discoveryDurationMs`, `discoveryTimestampMs`).
 *   - The list of service types listened on (the 8 Phase 0 types — `_roku`,
 *     `_airplay`, `_googlecast`, `_amzn-wplay`, `_androidtvremote2`, `_ipp`,
 *     `_http`, `_trillboards-csi`).
 *   - 30-row cap, 3000-15000 ms duration clamp.
 */
data class NetworkDevice(
    val serviceType: String,
    val instanceName: String,
    val host: String?,
    val port: Int?,
    val mdnsModel: String? = null,
    val mdnsVendor: String? = null,
    val mdnsSoftwareVersion: String? = null
)

data class MdnsSnapshot(
    val devices: List<NetworkDevice>,
    val deviceCount: Int,
    val discoveryDurationMs: Long,
    val discoveryTimestampMs: Long,
    /**
     * Surfaces the underlying [com.trillboards.discovery.MdnsAdapter]
     * skip-reason. Null when discovery ran successfully. See the
     * docstring on the discovery-core MdnsSnapshot for the catalog of
     * non-null values (`multicast_lock_failed`, `ffi_failure`).
     *
     * Mirrors `BleScanSnapshot.skipReason` (PR #4480). DeviceAgentService
     * heartbeat builder feeds this into [SkipReasonAggregator] so the
     * server-side classifier sees a per-screen counter for the failure
     * mode instead of "absent rows" indistinguishable from "no devices."
     */
    val skipReason: String? = null
)

object MdnsDiscovery {

    /**
     * Discover network devices via the Rust `mdns-sd`-backed core.
     *
     * @param context Application context (used by [MdnsAdapter] for the
     *   [android.net.wifi.WifiManager.MulticastLock] — Android's framework
     *   silently drops multicast packets at the netd / kernel layer without
     *   the lock, regardless of `INTERNET` permission).
     * @param discoveryDurationMs Wall-clock budget (clamped to [3000, 15000]
     *   ms inside both this layer and the Rust core).
     * @return [MdnsSnapshot] with per-device rows, or null if discovery is
     *   not possible (already running, FFI threw, .so missing, etc.).
     */
    suspend fun discover(
        context: Context,
        discoveryDurationMs: Long = 8_000L
    ): MdnsSnapshot? {
        val rust = MdnsAdapter.discover(context, discoveryDurationMs) ?: return null
        return MdnsSnapshot(
            devices = rust.devices.map { d ->
                NetworkDevice(
                    serviceType = d.serviceType,
                    instanceName = d.instanceName,
                    host = d.host,
                    port = d.port,
                    mdnsModel = d.mdnsModel,
                    mdnsVendor = d.mdnsVendor,
                    mdnsSoftwareVersion = d.mdnsSoftwareVersion
                )
            },
            deviceCount = rust.deviceCount,
            discoveryDurationMs = rust.discoveryDurationMs,
            discoveryTimestampMs = rust.discoveryTimestampMs,
            // Forward skip reason from the discovery-core adapter so the
            // heartbeat builder can aggregate per-source counters
            // without having to import discovery-core directly.
            skipReason = rust.skipReason
        )
    }
}
