package com.trillboards.ctv.core.models

import android.util.Log
import com.trillboards.api.types.BleScanResultData as WireBleScanResultData
import com.trillboards.api.types.HttpProbeResult as WireHttpProbeResult
import com.trillboards.api.types.MdnsNetworkDevice as WireMdnsNetworkDevice
import com.trillboards.api.types.PartnerDeviceHeartbeatBody
import com.trillboards.api.types.SsdpDeviceInfo as WireSsdpDeviceInfo
import com.trillboards.api.types.WifiScanResult as WireWifiScanResult

/**
 * Producer-side factory for the partner-facing wire-shape heartbeat
 * payload (`com.trillboards.api.types.PartnerDeviceHeartbeatBody`).
 *
 * The generated DTO carries the shape contract emitted from the merged
 * OpenAPI spec (`trillboard-api/docs/openapi/merged/partner-api.json`,
 * regenerated from the Zod registry by `scripts/build-openapi.js`). It
 * contains ONLY fields verified to ship in 30-day production telemetry —
 * the full list lives in `validation/deviceHeartbeatSchemas.js` along
 * with the audit comment.
 *
 * The producer-side caller still has the internal [HeartbeatPayload]
 * data class which carries every Kotlin-source field including the
 * Phase-6 / MDM / native-sensor / CSI stubs that don't yet flow to the
 * server. This factory's job is the wire-side projection — given the
 * subset of inputs the OpenAPI surface declares, build the DTO with
 * the truncation invariants enforced.
 *
 * # Truncation invariants
 *
 * Every "nearby N devices" list is bounded at construction by
 * [MAX_NEARBY_DEVICES_PER_LIST]. A misconfigured tablet shipping
 * unbounded lists would DoS Redis Cloud (12.38 GB peak, 226 hb/s
 * inflow at the 6,800-screen Dolphin Media launch). Cap is enforced
 * here so producer code that constructs the wire payload cannot
 * bypass it.
 *
 * - BLE/WiFi: RSSI-sorted, strongest first (less-negative = stronger).
 * - mDNS/SSDP/HTTP probes: original order (no per-entry signal
 *   strength to sort by — first-discovered wins).
 *
 * # Conformance contract
 *
 * Round-trip tests in
 * `trillboard-ctv/agent-core/src/test/.../HeartbeatPayloadConformanceTest.kt`
 * load fixtures from `trillboard-api/docs/openapi/fixtures/heartbeat-*.json`
 * and assert that:
 *   1. JSON → [PartnerDeviceHeartbeatBody] → JSON is byte-equivalent.
 *   2. [build] applied to a 51-element BLE input produces a 50-element
 *      output preserving the strongest 50 entries.
 */
object HeartbeatPayloadFactory {

    /**
     * Maximum number of entries the producer is allowed to ship per
     * "nearby N devices" list. Sized for Redis Cloud RAM budget at the
     * 6,800-screen Dolphin Media launch — at 226 heartbeats/sec inflow
     * with 12.38 GB peak Redis memory, one misconfigured tablet shipping
     * unbounded lists can DoS the entire telemetry pipeline. The cap
     * survives a malicious or buggy sensor.
     *
     * Raise only if signal density in the field grows. The corresponding
     * server-side ingest path (`SignalIngestService.normalize`) tolerates
     * any count, so this is purely a producer-side bound.
     *
     * Mirrors the exact value enforced server-side via the Zod
     * `deviceHeartbeatBodySchema` in
     * `trillboard-api/validation/deviceHeartbeatSchemas.js`.
     */
    const val MAX_NEARBY_DEVICES_PER_LIST: Int = 50

    private const val TAG = "HeartbeatPayloadFactory"

    /**
     * Truncate a BLE list to the strongest [MAX_NEARBY_DEVICES_PER_LIST]
     * entries by RSSI. Below the cap, the list passes through unchanged
     * in original order — no reorder when no truncation is needed.
     */
    fun capBleByRssi(devices: List<WireBleScanResultData>?): List<WireBleScanResultData>? {
        if (devices == null) return null
        if (devices.size <= MAX_NEARBY_DEVICES_PER_LIST) return devices
        Log.w(
            TAG,
            "nearbyBleDevices truncated: original_count=${devices.size} " +
                "truncated_count=$MAX_NEARBY_DEVICES_PER_LIST",
        )
        return devices.sortedByDescending { it.rssi }
            .take(MAX_NEARBY_DEVICES_PER_LIST)
    }

    /**
     * Truncate a WiFi list to the strongest [MAX_NEARBY_DEVICES_PER_LIST]
     * entries by signal strength dBm. Same strongest-first preference as
     * [capBleByRssi].
     */
    fun capWifiByRssi(networks: List<WireWifiScanResult>?): List<WireWifiScanResult>? {
        if (networks == null) return null
        if (networks.size <= MAX_NEARBY_DEVICES_PER_LIST) return networks
        Log.w(
            TAG,
            "nearbyWifiNetworks truncated: original_count=${networks.size} " +
                "truncated_count=$MAX_NEARBY_DEVICES_PER_LIST",
        )
        return networks.sortedByDescending { it.signalStrengthDbm }
            .take(MAX_NEARBY_DEVICES_PER_LIST)
    }

    /**
     * Truncate a list to the first [MAX_NEARBY_DEVICES_PER_LIST] entries
     * in original order. For sources with no per-entry signal strength
     * (mDNS / SSDP / HTTP probes), discovery order is the most meaningful
     * ranking — first-discovered wins.
     */
    private fun <T> capInOrder(items: List<T>?, fieldName: String): List<T>? {
        if (items == null) return null
        if (items.size <= MAX_NEARBY_DEVICES_PER_LIST) return items
        Log.w(
            TAG,
            "$fieldName truncated: original_count=${items.size} " +
                "truncated_count=$MAX_NEARBY_DEVICES_PER_LIST",
        )
        return items.take(MAX_NEARBY_DEVICES_PER_LIST)
    }

    /**
     * Build a [PartnerDeviceHeartbeatBody] from raw inputs with the
     * truncation invariants applied. This is the canonical producer
     * entry point for callers that want the wire DTO directly — the
     * generated DTO's primary constructor takes already-capped lists,
     * so external producers must route through here to guarantee the
     * cap is enforced before serialization.
     */
    @Suppress("LongParameterList")
    fun build(
        schemaVersion: Int? = 2,
        fingerprint: String? = null,
        screenId: String? = null,
        status: String? = null,
        metadata: Map<String, kotlinx.serialization.json.JsonElement>? = null,
        capabilities: com.trillboards.api.types.DeviceCapabilityPayload? = null,
        advertisingId: String? = null,
        advertisingIdType: PartnerDeviceHeartbeatBody.AdvertisingIdType? = null,
        limitAdTracking: Boolean? = null,
        wifiBssidHash: String? = null,
        wifiSsidHash: String? = null,
        gatewayIpHash: String? = null,
        nearbyWifiNetworks: List<WireWifiScanResult>? = null,
        wifiNetworkCount: Int? = null,
        wifiEnvironment: com.trillboards.api.types.WifiEnvironmentSnapshot? = null,
        nearbyBleDevices: List<WireBleScanResultData>? = null,
        bleDeviceCount: Int? = null,
        discoveredNetworkDevices: List<WireMdnsNetworkDevice>? = null,
        networkDeviceCount: Int? = null,
        ssdpDevices: List<WireSsdpDeviceInfo>? = null,
        httpProbes: List<WireHttpProbeResult>? = null,
        skipReasonCounts: Map<String, Map<String, Int>>? = null,
    ): PartnerDeviceHeartbeatBody {
        return PartnerDeviceHeartbeatBody(
            advertisingId = advertisingId,
            advertisingIdType = advertisingIdType,
            bleDeviceCount = bleDeviceCount,
            capabilities = capabilities,
            discoveredNetworkDevices = capInOrder(discoveredNetworkDevices, "discoveredNetworkDevices"),
            fingerprint = fingerprint,
            gatewayIpHash = gatewayIpHash,
            httpProbes = capInOrder(httpProbes, "httpProbes"),
            limitAdTracking = limitAdTracking,
            metadata = metadata,
            nearbyBleDevices = capBleByRssi(nearbyBleDevices),
            nearbyWifiNetworks = capWifiByRssi(nearbyWifiNetworks),
            networkDeviceCount = networkDeviceCount,
            schemaVersion = schemaVersion,
            screenId = screenId,
            skipReasonCounts = skipReasonCounts,
            ssdpDevices = capInOrder(ssdpDevices, "ssdpDevices"),
            status = status,
            wifiBssidHash = wifiBssidHash,
            wifiEnvironment = wifiEnvironment,
            wifiNetworkCount = wifiNetworkCount,
            wifiSsidHash = wifiSsidHash,
        )
    }

    /**
     * Re-apply the truncation invariants to an existing
     * [PartnerDeviceHeartbeatBody] (for example, one parsed from JSON
     * that has not yet been validated). Returns a copy of the input
     * with each "nearby N devices" list capped at
     * [MAX_NEARBY_DEVICES_PER_LIST].
     *
     * Conformance tests use this to verify that an oversized fixture
     * (51 BLE devices) → cap → 50 devices, RSSI-sorted strongest-first.
     */
    fun enforceCaps(body: PartnerDeviceHeartbeatBody): PartnerDeviceHeartbeatBody {
        return body.copy(
            nearbyBleDevices = capBleByRssi(body.nearbyBleDevices),
            nearbyWifiNetworks = capWifiByRssi(body.nearbyWifiNetworks),
            discoveredNetworkDevices = capInOrder(body.discoveredNetworkDevices, "discoveredNetworkDevices"),
            ssdpDevices = capInOrder(body.ssdpDevices, "ssdpDevices"),
            httpProbes = capInOrder(body.httpProbes, "httpProbes"),
        )
    }
}
