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
    val channelWidthMhz: Int?
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
                    channelWidthMhz = getChannelWidth(result)
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

}
