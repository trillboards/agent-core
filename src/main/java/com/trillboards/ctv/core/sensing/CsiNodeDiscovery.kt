package com.trillboards.ctv.core.sensing

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Discovers ESP32 CSI sensing nodes on the local network via mDNS/NSD.
 *
 * Listens for `_trillboards-csi._udp.` service advertisements published by
 * ESP32-S3 nodes running the Trillboards CSI firmware. When a node is found,
 * its IP address and UDP port are extracted for [CsiUdpListener] to receive
 * raw CSI frames.
 *
 * Thread safety: discovered nodes are stored in a [CopyOnWriteArrayList].
 *
 * Lifecycle: call [startDiscovery] once at agent startup, [stopDiscovery] on shutdown.
 * The discovery listener runs continuously (unlike the time-bounded MdnsDiscovery)
 * because CSI nodes may come and go as they reboot or move between networks.
 */
data class CsiNode(
    val nodeId: Int,
    val hostAddress: String,
    val port: Int,
    val screenId: String?,
    val discoveredAtMs: Long
)

object CsiNodeDiscovery {

    private const val TAG = "CsiNodeDiscovery"
    private const val CSI_SERVICE_TYPE = "_trillboards-csi._udp."

    private val discoveredNodes = CopyOnWriteArrayList<CsiNode>()
    private val isDiscovering = AtomicBoolean(false)
    private var nsdManager: NsdManager? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null

    /**
     * Returns a snapshot of all currently discovered CSI nodes.
     */
    fun getDiscoveredNodes(): List<CsiNode> = discoveredNodes.toList()

    /**
     * Number of currently discovered CSI nodes.
     */
    fun getNodeCount(): Int = discoveredNodes.size

    /**
     * Start continuous mDNS discovery for ESP32 CSI nodes.
     *
     * @param context Application context
     * @return true if discovery was started, false if already running or NSD unavailable
     */
    fun startDiscovery(context: Context): Boolean {
        if (!isDiscovering.compareAndSet(false, true)) {
            Log.d(TAG, "CSI node discovery already running")
            return false
        }

        val manager = context.getSystemService(Context.NSD_SERVICE) as? NsdManager
        if (manager == null) {
            Log.w(TAG, "NsdManager not available — CSI discovery disabled")
            isDiscovering.set(false)
            return false
        }

        nsdManager = manager

        val listener = object : NsdManager.DiscoveryListener {
            override fun onStartDiscoveryFailed(serviceType: String?, errorCode: Int) {
                Log.w(TAG, "CSI discovery start failed: error=$errorCode")
                isDiscovering.set(false)
            }

            override fun onStopDiscoveryFailed(serviceType: String?, errorCode: Int) {
                Log.d(TAG, "CSI discovery stop failed: error=$errorCode")
            }

            override fun onDiscoveryStarted(serviceType: String?) {
                Log.d(TAG, "CSI node discovery started for $serviceType")
            }

            override fun onDiscoveryStopped(serviceType: String?) {
                Log.d(TAG, "CSI node discovery stopped")
            }

            override fun onServiceFound(serviceInfo: NsdServiceInfo?) {
                serviceInfo?.let { info ->
                    Log.d(TAG, "Found CSI node service: ${info.serviceName}")
                    resolveService(manager, info)
                }
            }

            override fun onServiceLost(serviceInfo: NsdServiceInfo?) {
                serviceInfo?.let { info ->
                    Log.d(TAG, "CSI node service lost: ${info.serviceName}")
                    removeNodeByServiceName(info.serviceName)
                }
            }
        }

        try {
            manager.discoverServices(CSI_SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, listener)
            discoveryListener = listener
            Log.d(TAG, "CSI node discovery initiated")
            return true
        } catch (e: Exception) {
            Log.w(TAG, "Failed to start CSI node discovery: ${e.message}")
            isDiscovering.set(false)
            return false
        }
    }

    /**
     * Stop CSI node discovery and clear discovered nodes.
     */
    fun stopDiscovery() {
        if (!isDiscovering.compareAndSet(true, false)) {
            return
        }

        discoveryListener?.let { listener ->
            try {
                nsdManager?.stopServiceDiscovery(listener)
            } catch (e: Exception) {
                Log.d(TAG, "Error stopping CSI discovery: ${e.message}")
            }
        }

        discoveryListener = null
        nsdManager = null
        discoveredNodes.clear()
        Log.d(TAG, "CSI node discovery stopped and nodes cleared")
    }

    /**
     * Resolve a discovered service to extract host address and port.
     */
    private fun resolveService(manager: NsdManager, serviceInfo: NsdServiceInfo) {
        val resolveListener = object : NsdManager.ResolveListener {
            override fun onResolveFailed(info: NsdServiceInfo?, errorCode: Int) {
                Log.d(TAG, "Failed to resolve CSI node: error=$errorCode")
            }

            override fun onServiceResolved(info: NsdServiceInfo?) {
                info ?: return
                val host = info.host?.hostAddress
                if (host == null) {
                    Log.d(TAG, "Resolved CSI node has no host address")
                    return
                }

                val port = info.port
                val nodeId = extractNodeId(info)
                val screenId = extractScreenId(info)

                val node = CsiNode(
                    nodeId = nodeId,
                    hostAddress = host,
                    port = port,
                    screenId = screenId,
                    discoveredAtMs = System.currentTimeMillis()
                )

                // Replace existing node with same nodeId, or add new
                discoveredNodes.removeAll { it.nodeId == nodeId }
                discoveredNodes.add(node)

                Log.d(TAG, "Resolved CSI node: id=$nodeId host=$host port=$port")
            }
        }

        try {
            manager.resolveService(serviceInfo, resolveListener)
        } catch (e: Exception) {
            Log.d(TAG, "Error resolving CSI node service: ${e.message}")
        }
    }

    /**
     * Extract node ID from service info.
     *
     * Convention: service name is "trillboards-csi-<nodeId>" or the node_id
     * is provided as a TXT record attribute.
     */
    private fun extractNodeId(info: NsdServiceInfo): Int {
        // Try TXT record first (Android 21+)
        try {
            val attrs = info.attributes
            attrs?.get("node_id")?.let { bytes ->
                String(bytes, Charsets.UTF_8).toIntOrNull()?.let { return it }
            }
        } catch (_: Exception) {
            // attributes may not be available on all API levels
        }

        // Fall back to parsing service name: "trillboards-csi-<N>"
        val name = info.serviceName ?: return 0
        val match = Regex("trillboards-csi-(\\d+)").find(name)
        return match?.groupValues?.getOrNull(1)?.toIntOrNull() ?: 0
    }

    /**
     * Extract screen ID from TXT record if the ESP32 is pre-paired to a screen.
     */
    private fun extractScreenId(info: NsdServiceInfo): String? {
        return try {
            val attrs = info.attributes
            attrs?.get("screen_id")?.let { bytes ->
                String(bytes, Charsets.UTF_8).takeIf { it.isNotBlank() }
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Remove a node when its mDNS service is lost.
     */
    private fun removeNodeByServiceName(serviceName: String?) {
        serviceName ?: return
        val match = Regex("trillboards-csi-(\\d+)").find(serviceName)
        val nodeId = match?.groupValues?.getOrNull(1)?.toIntOrNull()
        if (nodeId != null) {
            discoveredNodes.removeAll { it.nodeId == nodeId }
            Log.d(TAG, "Removed CSI node: id=$nodeId")
        }
    }
}
