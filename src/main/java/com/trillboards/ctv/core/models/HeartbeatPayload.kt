package com.trillboards.ctv.core.models

import android.util.Log
import com.trillboards.ctv.core.identity.AuracastBroadcast
import com.trillboards.ctv.core.identity.BleScanResultData
import com.trillboards.ctv.core.identity.GattDeviceInfo
import com.trillboards.ctv.core.identity.NativeSensorSnapshot
import com.trillboards.ctv.core.identity.NetworkDevice
import com.trillboards.ctv.core.identity.SsdpDeviceInfo
import com.trillboards.ctv.core.identity.WifiEnvironmentSnapshot
import com.trillboards.ctv.core.identity.WifiScanResult
import com.trillboards.ctv.core.sensing.CsiSnapshot

/**
 * Producer-side payload bound for every "nearby N devices" list on a heartbeat.
 *
 * Sized for the Redis Cloud RAM budget at the 6,800-screen Dolphin Media
 * launch — at 226 heartbeats/sec inflow with 12.38 GB peak Redis memory,
 * one misconfigured tablet shipping unbounded lists can DoS the entire
 * telemetry pipeline. The cap survives a malicious or buggy sensor.
 *
 * Raise only if signal density in the field grows. The corresponding
 * server-side ingest path (`SignalIngestService.normalize`) tolerates any
 * count, so this is purely a producer-side bound.
 */
const val MAX_NEARBY_DEVICES_PER_LIST = 50

private const val TAG = "HeartbeatPayload"

/**
 * Truncate a BLE list to the strongest [MAX_NEARBY_DEVICES_PER_LIST] entries
 * by RSSI (less-negative = stronger). Below the cap, the list passes through
 * unchanged in original order — no reorder when no truncation is needed.
 */
private fun capBleByRssi(devices: List<BleScanResultData>?): List<BleScanResultData>? {
    if (devices == null) return null
    if (devices.size <= MAX_NEARBY_DEVICES_PER_LIST) return devices
    Log.w(
        TAG,
        "nearbyBleDevices truncated: original_count=${devices.size} truncated_count=$MAX_NEARBY_DEVICES_PER_LIST"
    )
    return devices.sortedByDescending { it.rssi }.take(MAX_NEARBY_DEVICES_PER_LIST)
}

/**
 * Truncate a WiFi list to the strongest [MAX_NEARBY_DEVICES_PER_LIST] entries
 * by RSSI. Same strongest-first preference as [capBleByRssi].
 */
private fun capWifiByRssi(networks: List<WifiScanResult>?): List<WifiScanResult>? {
    if (networks == null) return null
    if (networks.size <= MAX_NEARBY_DEVICES_PER_LIST) return networks
    Log.w(
        TAG,
        "nearbyWifiNetworks truncated: original_count=${networks.size} truncated_count=$MAX_NEARBY_DEVICES_PER_LIST"
    )
    return networks.sortedByDescending { it.signalStrengthDbm }.take(MAX_NEARBY_DEVICES_PER_LIST)
}

/**
 * Truncate a list to the first [MAX_NEARBY_DEVICES_PER_LIST] entries in
 * original order. For sources with no per-entry signal strength
 * (mDNS / SSDP / HTTP probes / GATT), the discovery order is the most
 * meaningful ranking — first-discovered wins.
 */
private fun <T> capInOrder(items: List<T>?, fieldName: String): List<T>? {
    if (items == null) return null
    if (items.size <= MAX_NEARBY_DEVICES_PER_LIST) return items
    Log.w(
        TAG,
        "$fieldName truncated: original_count=${items.size} truncated_count=$MAX_NEARBY_DEVICES_PER_LIST"
    )
    return items.take(MAX_NEARBY_DEVICES_PER_LIST)
}

/**
 * Heartbeat wire payload. Every "nearby N devices" list is bounded at
 * construction by [MAX_NEARBY_DEVICES_PER_LIST]. The cap is enforced via
 * a public `invoke()` factory + private primary constructor: external
 * callers writing `HeartbeatPayload(...)` route through the factory and
 * receive the truncated lists, while the data class generated `.copy()`
 * (which calls the private primary directly) preserves whatever lists
 * the original payload already held.
 */
data class HeartbeatPayload private constructor(
    /**
     * Heartbeat schema version. Server uses this to negotiate forward-compatible
     * payload shape:
     *   v1 — proximity-identity-graph baseline (BLE / WiFi / mDNS / gateway).
     *   v2 — `nativeSensorSnapshot` (footstep biometrics, magnetometer flux,
     *        proximity events, ambient light, barometer surfaced into
     *        `signal_observations` under `source='native_sensor'`).
     */
    val schemaVersion: Int,
    val fingerprint: String,
    val screenId: String?,
    val status: String,
    val metadata: Map<String, Any?>,
    val capabilities: DeviceCapabilityPayload,
    val advertisingId: String?,
    val advertisingIdType: String?,
    val limitAdTracking: Boolean?,
    val wifiBssidHash: String?,
    val wifiSsidHash: String?,
    val gatewayIpHash: String?,

    // WiFi scan results — all nearby networks (not just connected). Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] (RSSI-sorted, strongest first).
    val nearbyWifiNetworks: List<WifiScanResult>?,
    val wifiNetworkCount: Int?,

    // WiFi environment analysis — structured signals for unified intelligence pipeline
    val wifiEnvironment: WifiEnvironmentSnapshot?,

    // BLE scan results — nearby Bluetooth LE devices. Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] (RSSI-sorted, strongest first).
    val nearbyBleDevices: List<BleScanResultData>?,
    val bleDeviceCount: Int?,

    // BLE GATT 0x180A Device Information Service enrichment — opportunistic
    // post-scan reads of Manufacturer/Model/Hardware/Firmware on the strongest
    // BLE neighbors (Phase 5). One entry per address that returned at least
    // one populated DIS characteristic. Joined server-side to the matching
    // `nearbyBleDevices` row by `rawAddress`. Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] in original order.
    val bleGattDevices: List<GattDeviceInfo>?,

    // mDNS discovery results — smart devices on the local network. Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] in original (discovery) order.
    val discoveredNetworkDevices: List<NetworkDevice>?,
    val networkDeviceCount: Int?,

    // SSDP / UPnP M-SEARCH discovery results — TVs / NAS / routers /
    // smart-home hubs that don't broadcast on mDNS (Phase 2 of the
    // multi-protocol device discovery rewrite). Optional + additive — a
    // heartbeat without `ssdpDevices` keeps the existing wire shape.
    // Server-side `SignalIngestService.normalize()` HMAC-hashes friendlyName /
    // manufacturer / modelName / UDN under the daily pepper and emits one
    // row per LOCATION with `source='ssdp'` (CH migration 061 adds the four
    // dedicated `ssdp_*_hash` columns). Capped at [MAX_NEARBY_DEVICES_PER_LIST]
    // in original order.
    val ssdpDevices: List<SsdpDeviceInfo>?,
    // ── Phases 3+4 of the multi-protocol device discovery rewrite ──
    // ARP cache rows (Phase 3): one entry per resolvable LAN neighbor with
    // OUI vendor enrichment attached at the Rust boundary via the bundled
    // top-N IEEE OUI table. Optional + additive — heartbeats without this
    // field keep the existing wire shape unchanged.
    //
    // Each entry is `{ip, mac, iface, ouiVendor}`. Server-side
    // SignalIngestService.normalize() emits one signal_observations row per
    // entry with `source='arp'` and HMACs the MAC under the daily pepper.
    val arpDevices: List<Map<String, Any?>>?,

    // HTTP HEAD probe results (Phase 4): one entry per discovered LAN host
    // that returned a non-null `Server:` response header on port 80/8080.
    // The Rust `probe_http()` call sanitizes the header (drops empty /
    // oversize / email-shaped / control-char values) at the FFI boundary,
    // so by the time the row arrives here the `server` value is the clean
    // identity hint we want to HMAC.
    //
    // Each entry is `{host, port, server}`. Server-side normalize() emits
    // one signal_observations row per entry with `source='http_probe'` and
    // HMACs the host + server header under the daily pepper. Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] in original order.
    val httpProbes: List<Map<String, Any?>>?,

    // UWB (Ultra-WideBand) peer discovery and ranging (Phase 6).
    // One entry per detected UWB peer with ranging data.
    // Each entry is `{address, distance_mm, rssi_dbm}`.
    // Server-side SignalIngestService.normalize() emits one signal_observations
    // row per entry with `source='uwb'` and HMACs the address under the daily
    // pepper. Capped at [MAX_NEARBY_DEVICES_PER_LIST] in original order.
    val uwbPeers: List<Map<String, Any?>>?,

    // Auracast (LE Audio Broadcast) discovery results (Phase 6 bonus signal).
    // Active Bluetooth LE broadcasts enumerated via BluetoothLeBroadcastAssistant.
    // Optional + additive — heartbeats without this field keep the existing
    // wire shape. Server-side normalize() emits one signal_observations row
    // per broadcast with `source='auracast'` and HMACs the broadcastName +
    // publicBroadcastData under the daily pepper. Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] in original order.
    val auracastBroadcasts: List<Map<String, Any?>>?,

    // BLE Channel Sounding measurements (Phase 6): distance (mm) and angle-of-arrival (deg)
    // from nearby Bluetooth LE devices via the Android 16+ native Channel Sounding API.
    // Available only on SDK 36+ with hardware support; gracefully empty on older devices.
    // Serialized as `channel_sounding_measurements` JSON array. Server-side normalize()
    // emits one signal_observations row per measurement with `source='channel_sounding'`,
    // HMACs the BLE address, and stores distance + AoA as typed columns. Capped at
    // [MAX_NEARBY_DEVICES_PER_LIST] in original order.
    val channelSoundingMeasurements: List<Map<String, Any?>>?,

    // Enhanced environmental sensors (legacy single-shot — still emitted for
    // back-compat alongside the richer `nativeSensorSnapshot` below).
    val ambientLightLux: Float?,
    val barometerPressure: Float?,

    // Native Android sensor harvest — footstep biometrics, magnetometer flux,
    // proximity events, and the existing ambient light + barometer surfaced as
    // a single periodic row. Server emits `signal_observations` rows with
    // `source='native_sensor'` per heartbeat.
    val nativeSensorSnapshot: NativeSensorSnapshot?,

    // MDM location data
    val latitude: Double?,
    val longitude: Double?,
    val accuracyMeters: Float?,
    val locationSource: String?,

    // MDM compliance & enrollment
    val mdmCompliance: Map<String, Any?>?,
    val mdmEnrolled: Boolean,
    val mdmOrgId: String?,

    // CSI sensing snapshot from ESP32 WiFi CSI hardware
    val csiSnapshot: CsiSnapshot?,
    val csiNodeCount: Int?,

    // ── Phase 0 of redis-stream-scale-fix: structured skip-reason channel ──
    // Per-source skip-reason counts since the previous heartbeat. The
    // heartbeat builder snapshots + resets the [SkipReasonAggregator] every
    // cycle so the server sees deltas, not running totals (running totals
    // would over-weight long-running screens).
    //
    // Shape: `Map<source, Map<reason, count>>`.
    //   source ∈ {"ble", "mdns", "ssdp", "http_probe"}
    //   reason ∈ adapter-specific SKIP_* enum values (e.g. ble:
    //     "permission_denied", "scanner_unavailable", "bluetooth_disabled";
    //     mdns/ssdp: "multicast_lock_failed", "ffi_failure"; http_probe:
    //     "no_response", "ffi_failure")
    //
    // Wire format: serialized as `skip_reason_counts` snake_case to match
    // server-side `SignalIngestService.normalize()` convention. Server
    // emits `signal_skip_reason_total{source, reason}` CW counters; alarms
    // page when any reason='permission_denied' > 0 OR any reason fires on
    // > 5% of fleet for > 15 min.
    //
    // Class fix for the silent-zero-rows pattern (PR #4480 BLE; this PR
    // extends to mDNS/SSDP/HTTP probe). Wire-additive — older agents that
    // omit this field keep the existing payload shape.
    val skipReasonCounts: Map<String, Map<String, Int>>?
) {
    companion object {
        /**
         * Public factory. External `HeartbeatPayload(...)` call sites resolve
         * here (the primary constructor is private). Each "nearby N devices"
         * list is truncated to [MAX_NEARBY_DEVICES_PER_LIST] before the
         * private primary constructor is invoked, so the wire payload size
         * is bounded regardless of producer behavior.
         */
        operator fun invoke(
            schemaVersion: Int = 2,
            fingerprint: String,
            screenId: String?,
            status: String,
            metadata: Map<String, Any?>,
            capabilities: DeviceCapabilityPayload,
            advertisingId: String? = null,
            advertisingIdType: String? = null,
            limitAdTracking: Boolean? = null,
            wifiBssidHash: String? = null,
            wifiSsidHash: String? = null,
            gatewayIpHash: String? = null,
            nearbyWifiNetworks: List<WifiScanResult>? = null,
            wifiNetworkCount: Int? = null,
            wifiEnvironment: WifiEnvironmentSnapshot? = null,
            nearbyBleDevices: List<BleScanResultData>? = null,
            bleDeviceCount: Int? = null,
            bleGattDevices: List<GattDeviceInfo>? = null,
            discoveredNetworkDevices: List<NetworkDevice>? = null,
            networkDeviceCount: Int? = null,
            ssdpDevices: List<SsdpDeviceInfo>? = null,
            arpDevices: List<Map<String, Any?>>? = null,
            httpProbes: List<Map<String, Any?>>? = null,
            uwbPeers: List<Map<String, Any?>>? = null,
            auracastBroadcasts: List<Map<String, Any?>>? = null,
            channelSoundingMeasurements: List<Map<String, Any?>>? = null,
            ambientLightLux: Float? = null,
            barometerPressure: Float? = null,
            nativeSensorSnapshot: NativeSensorSnapshot? = null,
            latitude: Double? = null,
            longitude: Double? = null,
            accuracyMeters: Float? = null,
            locationSource: String? = null,
            mdmCompliance: Map<String, Any?>? = null,
            mdmEnrolled: Boolean = false,
            mdmOrgId: String? = null,
            csiSnapshot: CsiSnapshot? = null,
            csiNodeCount: Int? = null,
            skipReasonCounts: Map<String, Map<String, Int>>? = null
        ): HeartbeatPayload = HeartbeatPayload(
            schemaVersion = schemaVersion,
            fingerprint = fingerprint,
            screenId = screenId,
            status = status,
            metadata = metadata,
            capabilities = capabilities,
            advertisingId = advertisingId,
            advertisingIdType = advertisingIdType,
            limitAdTracking = limitAdTracking,
            wifiBssidHash = wifiBssidHash,
            wifiSsidHash = wifiSsidHash,
            gatewayIpHash = gatewayIpHash,
            nearbyWifiNetworks = capWifiByRssi(nearbyWifiNetworks),
            wifiNetworkCount = wifiNetworkCount,
            wifiEnvironment = wifiEnvironment,
            nearbyBleDevices = capBleByRssi(nearbyBleDevices),
            bleDeviceCount = bleDeviceCount,
            bleGattDevices = capInOrder(bleGattDevices, "bleGattDevices"),
            discoveredNetworkDevices = capInOrder(discoveredNetworkDevices, "discoveredNetworkDevices"),
            networkDeviceCount = networkDeviceCount,
            ssdpDevices = capInOrder(ssdpDevices, "ssdpDevices"),
            arpDevices = arpDevices,
            httpProbes = capInOrder(httpProbes, "httpProbes"),
            uwbPeers = capInOrder(uwbPeers, "uwbPeers"),
            auracastBroadcasts = capInOrder(auracastBroadcasts, "auracastBroadcasts"),
            channelSoundingMeasurements = capInOrder(channelSoundingMeasurements, "channelSoundingMeasurements"),
            ambientLightLux = ambientLightLux,
            barometerPressure = barometerPressure,
            nativeSensorSnapshot = nativeSensorSnapshot,
            latitude = latitude,
            longitude = longitude,
            accuracyMeters = accuracyMeters,
            locationSource = locationSource,
            mdmCompliance = mdmCompliance,
            mdmEnrolled = mdmEnrolled,
            mdmOrgId = mdmOrgId,
            csiSnapshot = csiSnapshot,
            csiNodeCount = csiNodeCount,
            skipReasonCounts = skipReasonCounts
        )
    }
}
