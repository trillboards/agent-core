package com.trillboards.ctv.core.identity

import android.content.Context
import android.content.pm.PackageManager
import android.util.Log

/**
 * UWB (Ultra-WideBand) peer enumeration and distance measurement.
 *
 * Gated on PackageManager.FEATURE_UWB — devices without a UWB chipset
 * (e.g., Tab S11 SM-X730) report isAvailable() = false and return empty lists.
 * Runtime failures (permission denied, adapter not initialized) are logged
 * but do not crash the heartbeat.
 *
 * Returns a list of Map<String, Any?> with entries:
 *   - address: String (UWB session controller address, raw hex format)
 *   - distance_mm: Int (ranging distance in millimeters)
 *   - rssi_dbm: Int (received signal strength indicator)
 *
 * Capped at MAX_NEARBY_DEVICES_PER_LIST per HeartbeatPayload.capInOrder().
 *
 * IMPLEMENTATION NOTE:
 * The androidx.core.uwb library is still in alpha/beta and not widely distributed.
 * This implementation uses a no-op stub that returns empty lists. Real UWB integration
 * will arrive when the AndroidX UWB API stabilizes and is available in standard repos.
 * For now, this allows the heartbeat payload to accept uwbPeers without forcing a
 * dependency on unstable libraries.
 */
class UwbDiscovery(private val context: Context) {
    private val tag = "UwbDiscovery"

    /**
     * Check if UWB is available on this device.
     */
    fun isAvailable(): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_UWB)
    }

    /**
     * Enumerate nearby UWB peers and their ranging data.
     *
     * Returns a list of maps, one per peer:
     *   { address: "...", distance_mm: 1234, rssi_dbm: -45 }
     *
     * CURRENT: Returns empty list. This is a stub implementation.
     * TODO: Integrate with androidx.core.uwb (or android.uwb.* system API) once
     * stable and widely available.
     *
     * On error or if UWB is unavailable, returns an empty list (not null).
     * The list is NOT pre-capped — HeartbeatPayload.capInOrder() handles size.
     */
    suspend fun discoverUwbPeers(): List<Map<String, Any?>> {
        if (!isAvailable()) {
            return emptyList()
        }

        return runCatching<List<Map<String, Any?>>> {
            // TODO: Real peer enumeration pending androidx.core.uwb stabilization.
            // For now, return empty. The heartbeat structure accepts the field;
            // server-side ingest (SignalIngestService) is ready for the data.
            Log.d(
                tag,
                "UWB peer enumeration stub — no peers discovered (awaiting androidx.core.uwb stable release)"
            )
            emptyList()
        }.onFailure { error ->
            Log.w(tag, "Exception in discoverUwbPeers", error)
        }.getOrElse { emptyList() }
    }
}
