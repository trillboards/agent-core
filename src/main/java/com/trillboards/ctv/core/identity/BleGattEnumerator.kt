package com.trillboards.ctv.core.identity

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.util.UUID

/**
 * BLE GATT enumerator — opportunistically reads public unauthenticated
 * characteristics from the **Device Information Service** (UUID 0x180A) on BLE
 * devices the existing [BleBeaconScanner] saw at least N times in a discovery
 * window.
 *
 * The four characteristics we read are device-class metadata, NOT user
 * identifiers:
 *   - Manufacturer Name (0x2A29) → `mdns_vendor`-equivalent for BLE
 *   - Model Number       (0x2A24) → `mdns_model`-equivalent for BLE
 *   - Hardware Revision  (0x2A27) → device-generation signal
 *   - Firmware Revision  (0x2A26) → version signal
 *
 * 60-70% of consumer BLE devices (AirPods, Bluetooth speakers, smart-home
 * sensors, fitness trackers) expose these without authentication. The current
 * advert-only scanner ignores them, so we add a free-signal layer on top.
 *
 * **Concurrency:** Android BLE GATT can hold ~7 simultaneous connections
 * globally and the connect → discoverServices → readCharacteristic state
 * machine is touchy under contention. This enumerator processes addresses
 * **serialized**, not in parallel.
 *
 * **Privacy:** server-side hashes Manufacturer/Model/Firmware under the
 * daily KMS-backed pepper (`RotatingPepperService`) — same path as today's
 * mDNS values. The agent does NOT hash. PII like device names, AirPods owner
 * tag, etc. live OUTSIDE the Device Information Service and are NEVER read by
 * this enumerator.
 *
 * Permission required: `BLUETOOTH_CONNECT` (Android 12+) — already declared in
 * all 3 forks; no manifest change needed.
 */
data class GattDeviceInfo(
    val rawAddress: String,
    val manufacturer: String?,
    val modelNumber: String?,
    val hardwareRevision: String?,
    val firmwareRevision: String?,
    val durationMs: Long
)

object BleGattEnumerator {

    private const val TAG = "BleGattEnumerator"

    /** SIG-allocated 0x180A Device Information Service. */
    private val DIS_SERVICE_UUID: UUID = UUID.fromString("0000180a-0000-1000-8000-00805f9b34fb")

    /** Manufacturer Name String, GATT char 0x2A29. */
    private val CHAR_MANUFACTURER_NAME: UUID = UUID.fromString("00002a29-0000-1000-8000-00805f9b34fb")

    /** Model Number String, GATT char 0x2A24. */
    private val CHAR_MODEL_NUMBER: UUID = UUID.fromString("00002a24-0000-1000-8000-00805f9b34fb")

    /** Hardware Revision String, GATT char 0x2A27. */
    private val CHAR_HARDWARE_REVISION: UUID = UUID.fromString("00002a27-0000-1000-8000-00805f9b34fb")

    /** Firmware Revision String, GATT char 0x2A26. */
    private val CHAR_FIRMWARE_REVISION: UUID = UUID.fromString("00002a26-0000-1000-8000-00805f9b34fb")

    private const val PER_DEVICE_TIMEOUT_MS = 1500L
    private const val DEFAULT_BUDGET_MS = 8000L

    /**
     * Local kill-switch — flip to true at runtime to disable enumeration entirely
     * (parallel to SSM `BLE_GATT_ENUMERATION_DISABLED` server-side filtering).
     */
    @Volatile
    var disabled: Boolean = false

    /**
     * Read DIS characteristics from each address in [addresses], serialized,
     * within the total [budgetMs]. Returns one [GattDeviceInfo] per address that
     * produced any non-null DIS field; addresses that failed (no DIS, GATT
     * timeout, adapter disabled mid-scan) are silently dropped — the vast
     * majority of BLE devices don't expose 0x180A and that's expected.
     *
     * @param context Application context
     * @param addresses BLE MAC addresses (top-N by RSSI from [BleBeaconScanner])
     * @param budgetMs Total budget across all reads (default 8000ms)
     * @return [GattDeviceInfo] entries — possibly empty
     */
    suspend fun enumerate(
        context: Context,
        addresses: List<String>,
        budgetMs: Long = DEFAULT_BUDGET_MS
    ): List<GattDeviceInfo> = withContext(Dispatchers.IO) {
        if (disabled) {
            Log.d(TAG, "BleGattEnumerator disabled — skipping")
            return@withContext emptyList<GattDeviceInfo>()
        }
        if (addresses.isEmpty()) return@withContext emptyList<GattDeviceInfo>()

        if (!hasConnectPermission(context)) {
            Log.d(TAG, "Missing BLUETOOTH_CONNECT permission — skipping")
            return@withContext emptyList<GattDeviceInfo>()
        }

        if (!context.packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)) {
            Log.d(TAG, "BLE not supported — skipping")
            return@withContext emptyList<GattDeviceInfo>()
        }

        val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
            ?: return@withContext emptyList<GattDeviceInfo>()

        val adapter: BluetoothAdapter = bluetoothManager.adapter
            ?: run {
                Log.d(TAG, "BluetoothAdapter not available — skipping")
                return@withContext emptyList<GattDeviceInfo>()
            }

        if (!adapter.isEnabled) {
            Log.d(TAG, "Bluetooth disabled — skipping")
            return@withContext emptyList<GattDeviceInfo>()
        }

        val results = mutableListOf<GattDeviceInfo>()
        val startedAt = System.currentTimeMillis()

        for (address in addresses) {
            val elapsed = System.currentTimeMillis() - startedAt
            val remaining = budgetMs - elapsed
            if (remaining <= 0L) {
                Log.d(TAG, "Budget exhausted at ${results.size}/${addresses.size}")
                break
            }
            val perDeviceTimeout = minOf(PER_DEVICE_TIMEOUT_MS, remaining)

            val info = readDeviceInfo(context, adapter, address, perDeviceTimeout)
            if (info != null && hasAnyValue(info)) {
                results += info
            }
        }

        Log.d(
            TAG,
            "GATT enumeration: ${results.size}/${addresses.size} devices populated " +
                "in ${System.currentTimeMillis() - startedAt}ms"
        )
        results
    }

    private fun hasAnyValue(info: GattDeviceInfo): Boolean =
        info.manufacturer != null
            || info.modelNumber != null
            || info.hardwareRevision != null
            || info.firmwareRevision != null

    /**
     * Connect, discover services, read each DIS char, disconnect/close.
     * Always closes the GATT — leaking connections degrades the radio and
     * eventually bricks BLE for the whole device until reboot.
     */
    private suspend fun readDeviceInfo(
        context: Context,
        adapter: BluetoothAdapter,
        address: String,
        timeoutMs: Long
    ): GattDeviceInfo? {
        val started = System.currentTimeMillis()
        val device: BluetoothDevice = try {
            adapter.getRemoteDevice(address)
        } catch (e: IllegalArgumentException) {
            Log.d(TAG, "Invalid BLE address '$address': ${e.message}")
            return null
        }

        val connected = CompletableDeferred<Boolean>()
        val servicesDiscovered = CompletableDeferred<Boolean>()
        val readQueue = ArrayDeque<UUID>()
        val readResults = mutableMapOf<UUID, String?>()
        val readCompleted = CompletableDeferred<Unit>()

        var gattRef: BluetoothGatt? = null

        val callback = object : BluetoothGattCallback() {
            override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
                if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                    connected.complete(true)
                    try {
                        @Suppress("MissingPermission")
                        gatt?.discoverServices()
                    } catch (e: SecurityException) {
                        Log.d(TAG, "discoverServices SecurityException: ${e.message}")
                        servicesDiscovered.complete(false)
                    }
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    if (!connected.isCompleted) connected.complete(false)
                    if (!servicesDiscovered.isCompleted) servicesDiscovered.complete(false)
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                }
            }

            override fun onServicesDiscovered(gatt: BluetoothGatt?, status: Int) {
                val ok = status == BluetoothGatt.GATT_SUCCESS && gatt != null
                servicesDiscovered.complete(ok)
                if (!ok) {
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                    return
                }
                val service = gatt!!.getService(DIS_SERVICE_UUID)
                if (service == null) {
                    // Most BLE devices do not expose 0x180A — expected.
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                    return
                }
                listOf(
                    CHAR_MANUFACTURER_NAME,
                    CHAR_MODEL_NUMBER,
                    CHAR_HARDWARE_REVISION,
                    CHAR_FIRMWARE_REVISION
                ).forEach { uuid ->
                    if (service.getCharacteristic(uuid) != null) {
                        readQueue.addLast(uuid)
                    }
                }
                if (readQueue.isEmpty()) {
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                    return
                }
                triggerNextRead(gatt, service)
            }

            @Suppress("DEPRECATION")
            override fun onCharacteristicRead(
                gatt: BluetoothGatt?,
                characteristic: BluetoothGattCharacteristic?,
                status: Int
            ) {
                val uuid = characteristic?.uuid
                val value = if (status == BluetoothGatt.GATT_SUCCESS) {
                    runCatching {
                        characteristic?.getStringValue(0)?.takeIf { it.isNotEmpty() }
                    }.getOrNull()
                } else null
                if (uuid != null) readResults[uuid] = value

                val service = gatt?.getService(DIS_SERVICE_UUID)
                if (gatt == null || service == null) {
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                    return
                }
                triggerNextRead(gatt, service)
            }

            // API 33+ overload: forwards the byte array directly. We use the
            // legacy getStringValue() above which still works on every API
            // level; this just prevents the new-API call from being silently
            // dropped on Android 13+.
            override fun onCharacteristicRead(
                gatt: BluetoothGatt,
                characteristic: BluetoothGattCharacteristic,
                value: ByteArray,
                status: Int
            ) {
                val uuid = characteristic.uuid
                val parsed = if (status == BluetoothGatt.GATT_SUCCESS) {
                    runCatching { String(value, Charsets.UTF_8).trim(' ').takeIf { it.isNotEmpty() } }
                        .getOrNull()
                } else null
                readResults[uuid] = parsed
                val service = gatt.getService(DIS_SERVICE_UUID) ?: run {
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                    return
                }
                triggerNextRead(gatt, service)
            }

            private fun triggerNextRead(gatt: BluetoothGatt, service: android.bluetooth.BluetoothGattService) {
                val next = readQueue.removeFirstOrNull()
                if (next == null) {
                    if (!readCompleted.isCompleted) readCompleted.complete(Unit)
                    return
                }
                val char = service.getCharacteristic(next)
                if (char == null) {
                    triggerNextRead(gatt, service)
                    return
                }
                val ok = try {
                    @Suppress("MissingPermission")
                    gatt.readCharacteristic(char)
                } catch (e: SecurityException) {
                    Log.d(TAG, "readCharacteristic SecurityException: ${e.message}")
                    false
                }
                if (!ok) {
                    // readCharacteristic returns false synchronously on busy /
                    // invalid char — skip and proceed.
                    triggerNextRead(gatt, service)
                }
            }
        }

        try {
            gattRef = try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    @Suppress("MissingPermission")
                    device.connectGatt(context, /* autoConnect = */ false, callback, BluetoothDevice.TRANSPORT_LE)
                } else {
                    @Suppress("MissingPermission")
                    device.connectGatt(context, /* autoConnect = */ false, callback)
                }
            } catch (e: SecurityException) {
                Log.d(TAG, "connectGatt SecurityException for $address: ${e.message}")
                return null
            }

            if (gattRef == null) {
                Log.d(TAG, "connectGatt returned null for $address")
                return null
            }

            val didReachReadDone = withTimeoutOrNull(timeoutMs) {
                // Wait for callbacks to drain. If the device never connects or
                // never reaches readCompleted, the timeout fires and we exit
                // with whatever readResults we accumulated (likely empty).
                readCompleted.await()
                true
            } ?: false

            if (!didReachReadDone) {
                Log.d(TAG, "GATT timeout for $address after ${timeoutMs}ms")
            }
        } catch (e: TimeoutCancellationException) {
            Log.d(TAG, "GATT timeout (cancellation) for $address: ${e.message}")
        } catch (e: Exception) {
            Log.d(TAG, "GATT error for $address: ${e.message}")
        } finally {
            try {
                @Suppress("MissingPermission")
                gattRef?.disconnect()
            } catch (_: Exception) { /* swallow */ }
            try {
                @Suppress("MissingPermission")
                gattRef?.close()
            } catch (_: Exception) { /* swallow */ }
        }

        val duration = System.currentTimeMillis() - started
        return GattDeviceInfo(
            rawAddress = address,
            manufacturer = readResults[CHAR_MANUFACTURER_NAME],
            modelNumber = readResults[CHAR_MODEL_NUMBER],
            hardwareRevision = readResults[CHAR_HARDWARE_REVISION],
            firmwareRevision = readResults[CHAR_FIRMWARE_REVISION],
            durationMs = duration
        )
    }

    private fun hasConnectPermission(context: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            // Pre-Android 12: legacy BLUETOOTH permission was install-time and
            // implicitly granted; ACCESS_FINE_LOCATION also covered scan access.
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        }
    }
}
