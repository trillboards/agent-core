package com.trillboards.ctv.core.identity

import android.content.Context
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import kotlin.math.abs

/**
 * Analyzes WiFi scan results to extract structured venue environment signals.
 *
 * These signals feed into the unified intelligence pipeline:
 * - Moment embedding (composeMomentDescription) for pgvector semantic search
 * - VAS formula as environmental context
 * - Evidence grade hierarchy as venue classification trust
 *
 * Does NOT do occupancy detection (that requires ESP32 CSI hardware).
 * Instead provides venue density, network quality, and motion indicators
 * from standard Android WiFi scan APIs available on all devices.
 */
object WifiEnvironmentAnalyzer {

    private const val TAG = "WifiEnvironmentAnalyzer"
    private const val RSSI_HISTORY_SIZE = 5

    // Rolling RSSI history for variance calculation (motion indicator)
    private val rssiHistory = ArrayDeque<Int>(RSSI_HISTORY_SIZE + 1)

    /**
     * Analyze a WiFi scan snapshot + current connection info to produce
     * structured environment signals.
     *
     * @param context Application context
     * @param scanSnapshot Result from WifiScanCollector.collect()
     * @return WifiEnvironmentSnapshot or null if analysis not possible
     */
    fun analyze(context: Context, scanSnapshot: WifiScanSnapshot?): WifiEnvironmentSnapshot? {
        if (scanSnapshot == null || scanSnapshot.networks.isEmpty()) return null

        return try {
            val wifiManager = context.applicationContext
                .getSystemService(Context.WIFI_SERVICE) as? WifiManager
                ?: return null

            // Current connection info
            @Suppress("DEPRECATION")
            val connectionInfo = wifiManager.connectionInfo
            val connectedRssi = connectionInfo?.rssi ?: -100
            val connectedFrequency = connectionInfo?.frequency ?: 0
            val connectedLinkSpeed = connectionInfo?.linkSpeed?.takeIf { it > 0 }

            // Track RSSI history for variance calculation
            if (connectedRssi > -120 && connectedRssi < 0) {
                rssiHistory.addLast(connectedRssi)
                if (rssiHistory.size > RSSI_HISTORY_SIZE) {
                    rssiHistory.removeFirst()
                }
            }

            // Frequency band classification
            val frequencyBand = classifyFrequencyBand(connectedFrequency)

            // Channel width of connected AP (API 21+ via scan results)
            val connectedChannelWidth = getConnectedChannelWidth(scanSnapshot, connectionInfo?.bssid)

            // Channel congestion: ratio of APs on same 20MHz primary channel
            val channelCongestion = computeChannelCongestion(scanSnapshot.networks, connectedFrequency)

            // RSSI variance across scan window (motion indicator)
            val rssiVariance = computeRssiVariance()

            // Signal spread across all visible APs (venue openness indicator)
            val signalLevels = scanSnapshot.networks.map { it.signalStrengthDbm }
            val medianSignal = signalLevels.sorted().let { sorted ->
                if (sorted.isEmpty()) -100
                else sorted[sorted.size / 2]
            }
            val signalSpread = if (signalLevels.size >= 2) {
                signalLevels.max() - signalLevels.min()
            } else 0

            val snapshot = WifiEnvironmentSnapshot(
                networkCount = scanSnapshot.uniqueBssidCount,
                connectedSignalDbm = connectedRssi,
                connectedFrequencyMhz = connectedFrequency,
                connectedChannelWidthMhz = connectedChannelWidth,
                connectedLinkSpeedMbps = connectedLinkSpeed,
                frequencyBand = frequencyBand,
                channelCongestionRatio = channelCongestion,
                rssiVariance = rssiVariance,
                uniqueBssidCount = scanSnapshot.uniqueBssidCount,
                medianSignalDbm = medianSignal,
                signalSpreadDbm = signalSpread,
                scanTimestampMs = scanSnapshot.scanTimestampMs
            )

            Log.d(TAG, "WiFi environment: ${snapshot.networkCount} APs, " +
                    "${snapshot.frequencyBand}, congestion=${String.format("%.2f", snapshot.channelCongestionRatio)}, " +
                    "variance=${String.format("%.1f", snapshot.rssiVariance)}")

            snapshot
        } catch (e: Exception) {
            Log.w(TAG, "WiFi environment analysis failed: ${e.message}")
            null
        }
    }

    private fun classifyFrequencyBand(frequencyMhz: Int): String = when {
        frequencyMhz in 2400..2500 -> "2.4ghz"
        frequencyMhz in 5000..5900 -> "5ghz"
        frequencyMhz in 5925..7125 -> "6ghz"
        else -> "unknown"
    }

    private fun getConnectedChannelWidth(
        scanSnapshot: WifiScanSnapshot,
        connectedBssid: String?
    ): Int? {
        if (connectedBssid == null) return null
        // We don't have raw BSSID in scan results (they're hashed), so
        // match by the strongest signal AP on the same frequency.
        // The connected AP is typically the strongest.
        return scanSnapshot.networks.firstOrNull()?.channelWidthMhz
    }

    /**
     * Channel congestion = fraction of visible APs whose primary 20MHz channel
     * overlaps with the connected AP's channel.
     *
     * High congestion (>0.5) = many APs competing = dense venue (mall, office).
     * Low congestion (<0.2) = few APs on this channel = isolated venue.
     */
    private fun computeChannelCongestion(
        networks: List<WifiScanResult>,
        connectedFrequencyMhz: Int
    ): Float {
        if (networks.isEmpty() || connectedFrequencyMhz == 0) return 0f

        // Two APs are on the "same channel" if their center frequencies
        // are within 10MHz (20MHz channel spacing)
        val sameChannel = networks.count { abs(it.frequencyMhz - connectedFrequencyMhz) <= 10 }
        return sameChannel.toFloat() / networks.size.toFloat()
    }

    /**
     * RSSI variance across the rolling history window.
     *
     * High variance = person/object moving near the AP = motion indicator.
     * Low variance = stable environment = no motion.
     *
     * This is NOT occupancy detection (that requires CSI subcarrier analysis),
     * but it's a useful ambient signal for the moment embedding pipeline.
     */
    private fun computeRssiVariance(): Float {
        if (rssiHistory.size < 2) return 0f

        val mean = rssiHistory.average()
        val variance = rssiHistory.sumOf { (it - mean) * (it - mean) } / rssiHistory.size
        return variance.toFloat()
    }
}
