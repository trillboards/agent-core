package com.trillboards.ctv.core.identity

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * Active WiFi scan collector — enumerates ALL nearby WiFi networks (not just connected).
 *
 * Unlike [WiFiSignalCollector] which reads only the currently-connected AP,
 * this collector runs WifiManager.startScan() to discover all visible BSSIDs.
 * Used for venue foot-traffic proxy (unique BSSID count) and WiFi co-location
 * identity edges.
 *
 * Privacy: raw BSSIDs are sent over TLS to the API and HMAC-hashed there with a
 * KMS-backed daily-rotating pepper. Edge no longer hashes — that broke server-side
 * rotation policy. SSID is dropped entirely (free-text PII risk).
 *
 * Permissions required: NEARBY_WIFI_DEVICES (Android 13+) or ACCESS_FINE_LOCATION (older).
 * Both are already declared in the tablet-agent-lite AndroidManifest.
 */
data class WifiScanResult(
    val rawBssid: String,
    val signalStrengthDbm: Int,
    val frequencyMhz: Int,
    val channelWidthMhz: Int?,
    // ── Venue-insights PR 8 (Cut E.4) — WiFi enrichment fields ──
    // `capabilities`: raw ScanResult.capabilities flag string (e.g.
    // "[WPA2-PSK-CCMP][RSN-PSK-CCMP][ESS]"). Used server-side to coarse-bucket
    // venue posture (open / WPA2 / WPA3 / Enterprise) for the audience-grade
    // ladder. The Android ScanResult contract guarantees a non-null string
    // (possibly empty). Default "" so empty-capabilities reads still serialize
    // without a null wire value.
    val capabilities: String = "",
    // `is80211mcResponder`: whether the AP supports IEEE 802.11mc FTM ranging
    // (responder side). API 28+ (Android 9.0). Default false on older OS
    // versions or when the radio doesn't report it. Stable per-AP attribute;
    // future cuts use this to identify ranging-capable infra (vs randomized
    // mobile hotspots), helping classify venue-fixed APs.
    val is80211mcResponder: Boolean = false,
    // `vendorOui2Byte`: the first 2 bytes of the BSSID as an unsigned 16-bit
    // int (0..65535). Cheap server-side join key against the IEEE OUI table
    // (the bundled Rust `discovery-core-rust/src/oui.rs` table is currently
    // ARP-only; the 2-byte coarse prefix lets the server bucket WiFi BSSIDs
    // by vendor without per-row 24-bit lookup on the hot path). Null when
    // the BSSID is malformed (non-hex / too short).
    val vendorOui2Byte: Int? = null
)

data class WifiScanSnapshot(
    val networks: List<WifiScanResult>,
    val uniqueBssidCount: Int,
    val scanTimestampMs: Long
)

/**
 * Structured WiFi environment signals derived from scan results + connection info.
 *
 * Feeds into the unified intelligence pipeline:
 * - Moment embedding → pgvector 768-D semantic search
 * - VAS formula as environmental context signal
 * - Evidence grade hierarchy as venue classification trust
 * - Attenuation ladder for graceful degradation
 */
data class WifiEnvironmentSnapshot(
    val networkCount: Int,                    // Visible APs — venue density proxy
    val connectedSignalDbm: Int,              // RSSI of connected AP (-120..0 dBm)
    val connectedFrequencyMhz: Int,           // Center frequency (2412-7115 MHz)
    val connectedChannelWidthMhz: Int?,       // 20/40/80/160 MHz (API 23+)
    val connectedLinkSpeedMbps: Int?,         // Link speed of connected AP
    val frequencyBand: String,                // "2.4ghz" | "5ghz" | "6ghz" | "unknown"
    val channelCongestionRatio: Float,        // APs on same channel / total (0.0-1.0)
    val rssiVariance: Float,                  // Variance over scan window — motion indicator
    val uniqueBssidCount: Int,                // Deduped AP count
    val medianSignalDbm: Int,                 // Median RSSI across all visible APs
    val signalSpreadDbm: Int,                 // Max-min RSSI — venue openness indicator
    val scanTimestampMs: Long
)

object WifiScanCollector {

    private const val TAG = "WifiScanCollector"
    private const val MAX_NETWORKS = 50

    /**
     * Collect a snapshot of nearby WiFi networks.
     *
     * @param context Application context
     * @return [WifiScanSnapshot] or null if scanning is not possible
     */
    fun collect(context: Context): WifiScanSnapshot? {
        return try {
            if (!hasPermission(context)) {
                Log.d(TAG, "Missing WiFi scan permission — skipping")
                return null
            }

            val wifiManager = context.applicationContext
                .getSystemService(Context.WIFI_SERVICE) as? WifiManager
                ?: return null

            if (!wifiManager.isWifiEnabled) {
                Log.d(TAG, "WiFi disabled — skipping scan")
                return null
            }

            // Trigger a fresh scan (best-effort — results may be cached by the OS)
            @Suppress("DEPRECATION")
            val scanStarted = wifiManager.startScan()
            if (!scanStarted) {
                Log.d(TAG, "WiFi scan throttled by OS — using cached results")
            }

            val scanResults = wifiManager.scanResults
            if (scanResults.isNullOrEmpty()) {
                Log.d(TAG, "No scan results available")
                return null
            }

            // Deduplicate by BSSID, keep strongest signal per BSSID
            val dedupedByBssid = scanResults
                .filter { it.BSSID != null && it.BSSID != "02:00:00:00:00:00" }
                .groupBy { it.BSSID }
                .mapValues { (_, results) -> results.maxByOrNull { it.level } ?: results.first() }
                .values
                .sortedByDescending { it.level } // Strongest first
                .take(MAX_NETWORKS)

            val networks = dedupedByBssid.map { result ->
                WifiScanResult(
                    rawBssid = result.BSSID,
                    signalStrengthDbm = result.level,
                    frequencyMhz = result.frequency,
                    channelWidthMhz = getChannelWidth(result),
                    capabilities = result.capabilities ?: "",
                    is80211mcResponder = is80211mcResponderSafe(result),
                    vendorOui2Byte = extractVendorOui2Byte(result.BSSID)
                )
            }

            val snapshot = WifiScanSnapshot(
                networks = networks,
                uniqueBssidCount = dedupedByBssid.size,
                scanTimestampMs = System.currentTimeMillis()
            )

            Log.d(TAG, "Collected ${networks.size} WiFi networks (${snapshot.uniqueBssidCount} unique BSSIDs)")
            snapshot
        } catch (e: Exception) {
            Log.w(TAG, "WiFi scan failed: ${e.message}")
            null
        }
    }

    private fun hasPermission(context: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+: NEARBY_WIFI_DEVICES
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.NEARBY_WIFI_DEVICES
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            // Older: ACCESS_FINE_LOCATION
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun getChannelWidth(result: ScanResult): Int? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                when (result.channelWidth) {
                    ScanResult.CHANNEL_WIDTH_20MHZ -> 20
                    ScanResult.CHANNEL_WIDTH_40MHZ -> 40
                    ScanResult.CHANNEL_WIDTH_80MHZ -> 80
                    ScanResult.CHANNEL_WIDTH_160MHZ -> 160
                    else -> null
                }
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Read [ScanResult.is80211mcResponder] safely.
     *
     * The method is available on API 28+ (Android 9.0 Pie). On older OS
     * versions we return `false` because the radio doesn't report 802.11mc
     * support. Wrapped in try/catch as a belt-and-suspenders guard against
     * vendor ROMs that strip the method symbol.
     */
    private fun is80211mcResponderSafe(result: ScanResult): Boolean {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                result.is80211mcResponder
            } else {
                false
            }
        } catch (e: Throwable) {
            false
        }
    }

    /**
     * Extract the first 2 bytes of a BSSID as a 16-bit unsigned integer.
     *
     * BSSID wire format examples: `"aa:bb:cc:dd:ee:ff"`, `"AA-BB-CC-DD-EE-FF"`,
     * `"aabb.ccdd.eeff"`. Strips separators, takes the first 4 hex chars, parses
     * as an unsigned 16-bit value (0..65535). Returns null when the BSSID has
     * fewer than 4 hex chars or contains non-hex characters in the prefix —
     * malformed BSSIDs shouldn't survive [WifiManager.scanResults] but the
     * defensive parse keeps us correct under buggy vendor ROMs.
     *
     * Server-side this value is used to coarse-bucket WiFi BSSIDs by vendor.
     * The IEEE OUI assignments are 24-bit (3 bytes); the 2-byte prefix is a
     * lossy compression (some vendors span multiple 2-byte buckets) but is
     * sufficient as a JOIN key for the existing OUI table (which lives in
     * `discovery-core-rust/src/oui.rs` and is keyed on the 24-bit prefix —
     * server can match by `LEFT(oui_key, 4) == hex(vendor_oui_2byte)`).
     */
    internal fun extractVendorOui2Byte(bssid: String?): Int? {
        if (bssid.isNullOrEmpty()) return null
        var value = 0
        var hexCount = 0
        for (ch in bssid) {
            if (ch == ':' || ch == '-' || ch == '.' || ch == ' ') continue
            val digit = when (ch) {
                in '0'..'9' -> ch.code - '0'.code
                in 'a'..'f' -> ch.code - 'a'.code + 10
                in 'A'..'F' -> ch.code - 'A'.code + 10
                else -> return null
            }
            value = (value shl 4) or digit
            hexCount++
            if (hexCount == 4) return value
        }
        return null
    }

}
