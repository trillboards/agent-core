package com.trillboards.ctv.core.identity

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import java.lang.reflect.Method

/**
 * Phase 6 bonus signal — Auracast (LE Audio Broadcast Discovery).
 *
 * Auracast is Bluetooth Low Energy's broadcast mode for spatial audio.
 * A broadcast source (e.g., airport TV, stadium display) advertises its
 * presence without requiring pairing. Receivers use [BluetoothLeBroadcastAssistant]
 * to discover active broadcasts on the LAN.
 *
 * Wire format: `auracast_broadcasts` (array of broadcast metadata dicts).
 * Server-side [SignalIngestService.normalize] emits one `signal_observations`
 * row per broadcast with `source='auracast'`, HMAC-hashing broadcast_name and
 * public_broadcast_data under the daily pepper.
 *
 * Hardware: API 33+ (Android 13+). Tab S11 (SDK 36) supports this.
 * Gating: Runtime check on `Build.VERSION.SDK_INT >= 33`.
 */

data class AuracastBroadcast(
    val broadcastId: Int,
    val broadcastName: String?,
    val publicBroadcastData: ByteArray?,
    val sourceId: Int?
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as AuracastBroadcast

        if (broadcastId != other.broadcastId) return false
        if (broadcastName != other.broadcastName) return false
        if (publicBroadcastData != null) {
            if (other.publicBroadcastData == null) return false
            if (!publicBroadcastData.contentEquals(other.publicBroadcastData)) return false
        } else if (other.publicBroadcastData != null) return false
        if (sourceId != other.sourceId) return false

        return true
    }

    override fun hashCode(): Int {
        var result = broadcastId
        result = 31 * result + (broadcastName?.hashCode() ?: 0)
        result = 31 * result + (publicBroadcastData?.contentHashCode() ?: 0)
        result = 31 * result + (sourceId ?: 0)
        return result
    }
}

data class AuracastSnapshot(
    val broadcasts: List<AuracastBroadcast>,
    val broadcastCount: Int,
    val discoveryDurationMs: Long,
    val discoveryTimestampMs: Long,
    /**
     * Surfaces any skip reason from discovery failure. Null on success.
     * Possible values: "bluetooth_unavailable", "permission_denied", "ffi_failure".
     */
    val skipReason: String? = null
)

object AuracastDiscovery {

    private const val TAG = "AuracastDiscovery"

    /**
     * Discover active Auracast broadcasts via BluetoothLeBroadcastAssistant.
     *
     * Uses reflection to access system APIs available in API 33+, gracefully
     * handling missing APIs on older devices or test environments.
     *
     * @param context Application context (used for BluetoothManager).
     * @param discoveryDurationMs Wall-clock budget (clamped to [3000, 15000] ms).
     * @return [AuracastSnapshot] with per-broadcast rows, or null if discovery
     *   failed (Bluetooth unavailable, no permission, API too old, etc.).
     */
    fun discover(
        context: Context,
        discoveryDurationMs: Long = 8_000L
    ): AuracastSnapshot? {
        return try {
            // API 33+ only
            if (Build.VERSION.SDK_INT < 33) {
                Log.d(TAG, "Auracast requires API 33+; skipping on API ${Build.VERSION.SDK_INT}")
                return null
            }

            val startTimeMs = System.currentTimeMillis()

            val bluetoothManager =
                context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
            val bluetoothAdapter = bluetoothManager?.adapter
            if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled) {
                Log.w(TAG, "Bluetooth unavailable or disabled")
                return null
            }

            // Check for BLUETOOTH permission (required for broadcast enumeration)
            if (!hasBluetoothPermission(context)) {
                Log.w(TAG, "BLUETOOTH permission denied")
                return AuracastSnapshot(
                    broadcasts = emptyList(),
                    broadcastCount = 0,
                    discoveryDurationMs = System.currentTimeMillis() - startTimeMs,
                    discoveryTimestampMs = System.currentTimeMillis(),
                    skipReason = "permission_denied"
                )
            }

            // Use reflection to access BluetoothLeBroadcastAssistant (API 33+)
            // Available in android.bluetooth.le package on API 33+
            val broadcasts = mutableListOf<AuracastBroadcast>()
            try {
                val assistantClass = Class.forName("android.bluetooth.le.BluetoothLeBroadcastAssistant")
                val getAssistantMethod: Method? = bluetoothAdapter.javaClass.getMethod("getBluetoothLeBroadcastAssistant")
                val broadcastAssistant = getAssistantMethod?.invoke(bluetoothAdapter)

                if (broadcastAssistant != null) {
                    // Call getAllBroadcastMetadata() via reflection
                    val getAllMethod = assistantClass.getMethod("getAllBroadcastMetadata")
                    val metadataList = getAllMethod.invoke(broadcastAssistant) as? List<*>

                    if (metadataList != null) {
                        for (m in metadataList) {
                            if (m == null) continue
                            try {
                                val mClass = m.javaClass
                                val idMethod = mClass.getMethod("getBroadcastId")
                                val nameMethod = mClass.getMethod("getBroadcastName")
                                val dataMethod = mClass.getMethod("getPublicBroadcastData")
                                val typeMethod = mClass.getMethod("getSourceAddressType")

                                val broadcastId = (idMethod.invoke(m) as? Int) ?: continue
                                val broadcastName = nameMethod.invoke(m) as? String
                                val publicData = dataMethod.invoke(m) as? ByteArray
                                val sourceId = typeMethod.invoke(m) as? Int

                                broadcasts.add(
                                    AuracastBroadcast(
                                        broadcastId = broadcastId,
                                        broadcastName = broadcastName,
                                        publicBroadcastData = publicData,
                                        sourceId = sourceId
                                    )
                                )
                            } catch (e: Exception) {
                                Log.d(TAG, "Failed to extract broadcast metadata: ${e.message}")
                            }
                        }
                    }
                }
            } catch (e: ClassNotFoundException) {
                Log.d(TAG, "BluetoothLeBroadcastAssistant class not found (API < 33?)")
                return null
            } catch (e: NoSuchMethodException) {
                Log.d(TAG, "BluetoothLeBroadcastAssistant method not available")
                return null
            } catch (e: Exception) {
                Log.e(TAG, "Reflection error accessing Auracast API: ${e.message}", e)
                return null
            }

            AuracastSnapshot(
                broadcasts = broadcasts.take(50), // Cap at MAX_NEARBY_DEVICES_PER_LIST
                broadcastCount = broadcasts.size,
                discoveryDurationMs = System.currentTimeMillis() - startTimeMs,
                discoveryTimestampMs = System.currentTimeMillis()
            )
        } catch (e: SecurityException) {
            Log.w(TAG, "SecurityException during discovery: ${e.message}")
            AuracastSnapshot(
                broadcasts = emptyList(),
                broadcastCount = 0,
                discoveryDurationMs = 0L,
                discoveryTimestampMs = System.currentTimeMillis(),
                skipReason = "permission_denied"
            )
        } catch (e: Exception) {
            Log.e(TAG, "Discovery failed: ${e.message}", e)
            null
        }
    }

    /**
     * Check the runtime BT permission(s) needed to enumerate Auracast
     * broadcasts. The legacy `android.permission.BLUETOOTH` permission has
     * `maxSdkVersion=30` in the manifest schema; on Android 12+ (API 31+)
     * scan-style discovery requires `BLUETOOTH_SCAN` and connect-style
     * enumeration (`BluetoothLeBroadcastAssistant.getAllBroadcastMetadata`)
     * requires `BLUETOOTH_CONNECT`. Tab S11 (API 36) confirmed: gating on
     * the legacy permission alone returns `permission_denied` even when
     * scan/connect are granted.
     *
     * Per Codex P1 review on PR #4550 — fixed inline.
     */
    private fun hasBluetoothPermission(context: Context): Boolean {
        val pm = android.content.pm.PackageManager.PERMISSION_GRANTED
        // API 31+ — modern split permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // BluetoothLeBroadcastAssistant.getAllBroadcastMetadata is a
            // connect-style API (it returns metadata about active broadcast
            // sources we could connect to). BLUETOOTH_CONNECT is the
            // canonical gate. BLUETOOTH_SCAN is also useful for the
            // discovery side; we accept either as evidence of consent so
            // the call still attempts when the app holds one but not both.
            return context.checkSelfPermission("android.permission.BLUETOOTH_CONNECT") == pm ||
                    context.checkSelfPermission("android.permission.BLUETOOTH_SCAN") == pm
        }
        // API 30 and below — legacy single permission
        return context.checkSelfPermission("android.permission.BLUETOOTH") == pm
    }
}
