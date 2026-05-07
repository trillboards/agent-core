package com.trillboards.ctv.core.identity

/**
 * Phase 2 of the multi-protocol device discovery rewrite — wire-format
 * carrier for one SSDP / UPnP responder.
 *
 * Mirrors the Rust core's [`SsdpDevice`](../../../../discovery-core-rust/src/ssdp.rs)
 * UniFFI record but lives in `com.trillboards.ctv.core.identity` so
 * `HeartbeatPayload.kt` (which is shared with `agent-core-lite` and the
 * `trillboards-measurement-sdk`) doesn't have to import the
 * `com.trillboards.discovery.SsdpDevice` UniFFI type directly. The
 * `SsdpAdapter.discover()` call site is responsible for converting the
 * UniFFI `SsdpDevice` to this carrier (one-to-one field copy).
 *
 * Field semantics (see `discovery-core-rust/discovery-core.udl` for the
 * authoritative documentation):
 *   - `location` — required URL of description.xml. Used by the server as
 *     the unique key per LAN device (HMAC under the daily pepper before
 *     landing in CH `signal_observations.raw_signal_hash`).
 *   - `server` — RFC 2616 product token (e.g. "Linux/3.0 UPnP/1.0
 *     Sonos/72.6-31040"). Vendor / version hint. NOT hashed (low
 *     cardinality, useful as a low-fidelity vendor signal).
 *   - `friendlyName`, `manufacturer`, `modelName`, `udn` — populated by
 *     parsing description.xml. May be null (description fetch failed,
 *     device omitted the field, or the Rust core PII-stripped it). HMAC'd
 *     server-side under the daily pepper.
 *   - `st` — search target URN (e.g. "urn:schemas-upnp-org:device:ZonePlayer:1").
 *     NOT hashed (short, low-cardinality categorical).
 *
 * Wire serialization: `ApiClient.kt` emits the heartbeat field as
 * `ssdp_devices` JSONArray with snake_case keys (matches the existing
 * `nearby_ble_devices` / `discovered_network_devices` / `ble_gatt_devices`
 * pattern).
 */
data class SsdpDeviceInfo(
    val location: String,
    val server: String? = null,
    val friendlyName: String? = null,
    val manufacturer: String? = null,
    val modelName: String? = null,
    val udn: String? = null,
    val st: String,
)
