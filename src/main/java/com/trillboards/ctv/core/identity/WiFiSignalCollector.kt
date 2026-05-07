package com.trillboards.ctv.core.identity

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.util.Log
import java.net.Inet4Address
import java.security.MessageDigest

data class WiFiSignalResult(
    val bssidHash: String,
    val ssidHash: String,
    val gatewayIpHash: String?,
    val signalStrengthDbm: Int?
)

object WiFiSignalCollector {

    private const val TAG = "WiFiSignalCollector"

    fun collect(context: Context): WiFiSignalResult? {
        return try {
            val wifiManager = context.applicationContext
                .getSystemService(Context.WIFI_SERVICE) as? WifiManager
                ?: return null

            val connectionInfo = wifiManager.connectionInfo ?: return null
            val bssid = connectionInfo.bssid

            // BSSID can be null, "02:00:00:00:00:00" (unknown), or a valid MAC
            if (bssid.isNullOrBlank() || bssid == "02:00:00:00:00:00") {
                Log.d(TAG, "No valid BSSID available")
                return null
            }

            val ssid = connectionInfo.ssid?.removeSurrounding("\"") ?: ""

            // SHA-256 hash on-device — raw values NEVER leave the device
            val bssidHash = sha256(bssid)
            val ssidHash = if (ssid.isNotBlank()) sha256(ssid) else ""

            val gatewayIpHash = resolveGatewayIp(context, wifiManager)?.let { sha256(it) }

            val rssi = connectionInfo.rssi
            val signalDbm = if (rssi != 0 && rssi != -127) rssi else null

            WiFiSignalResult(
                bssidHash = bssidHash,
                ssidHash = ssidHash,
                gatewayIpHash = gatewayIpHash,
                signalStrengthDbm = signalDbm
            ).also {
                Log.d(TAG, "Collected WiFi signals: BSSID=${bssidHash.take(12)}..., signal=${signalDbm}dBm, gateway=${if (gatewayIpHash != null) "set" else "missing"}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to collect WiFi signals: ${e.message}")
            null
        }
    }

    /**
     * Two-path gateway resolution. The legacy [WifiManager.getDhcpInfo] returns
     * `gateway=0` on Android 13+ for many OEM builds even when DHCP succeeded —
     * so we ALSO probe [ConnectivityManager.getLinkProperties], which carries
     * the resolved DHCP server / default-route gateway via the modern Network
     * API. First non-null wins.
     *
     * Returns the dotted-quad IPv4 string or null if no gateway is resolvable.
     *
     * Public so the discovery cycle in DeviceAgentService can reuse it as a
     * cleartext IP for HTTP-probe targeting (the multi-protocol discovery
     * heartbeat path can't read /proc/net/arp on Android targetSdk ≥ 32, so
     * it can't enumerate other LAN MACs — it falls back to "probe the
     * gateway + the mDNS-discovered hosts" instead).
     */
    fun resolveGatewayIp(context: Context): String? {
        val wifiManager = context.applicationContext
            .getSystemService(Context.WIFI_SERVICE) as? WifiManager
            ?: return null
        return resolveGatewayIp(context, wifiManager)
    }

    private fun resolveGatewayIp(context: Context, wifiManager: WifiManager): String? {
        // Path 1: legacy DhcpInfo (works on most pre-13 devices and some OEMs)
        try {
            val dhcpInfo = wifiManager.dhcpInfo
            if (dhcpInfo != null && dhcpInfo.gateway != 0) {
                return intToIp(dhcpInfo.gateway)
            }
        } catch (_: Exception) { /* fall through */ }

        // Path 2: modern LinkProperties.getRoutes() default-route lookup
        try {
            val cm = context.applicationContext
                .getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return null
            val activeNetwork = cm.activeNetwork ?: return null
            val caps = cm.getNetworkCapabilities(activeNetwork) ?: return null
            // Only honor a WIFI transport here — keeps cellular/USB-tether
            // gateways out of the WiFi-context observation.
            if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) return null
            val link = cm.getLinkProperties(activeNetwork) ?: return null
            for (route in link.routes) {
                if (!route.isDefaultRoute) continue
                val gw = route.gateway
                if (gw is Inet4Address && !gw.isAnyLocalAddress) {
                    return gw.hostAddress
                }
            }
            // Fallback: dhcpServerAddress (API 30+) is usually the LAN gateway
            // on consumer routers. Last resort before giving up.
            link.dhcpServerAddress?.let { server ->
                if (server is Inet4Address && !server.isAnyLocalAddress) {
                    return server.hostAddress
                }
            }
        } catch (_: Exception) { /* fall through */ }

        return null
    }

    private fun sha256(input: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(input.toByteArray(Charsets.UTF_8))
        return hash.joinToString("") { "%02x".format(it) }
    }

    private fun intToIp(ip: Int): String {
        return "${ip and 0xFF}.${ip shr 8 and 0xFF}.${ip shr 16 and 0xFF}.${ip shr 24 and 0xFF}"
    }
}
