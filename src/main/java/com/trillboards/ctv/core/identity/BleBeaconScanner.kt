package com.trillboards.ctv.core.identity

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean

/**
 * BLE beacon scanner — discovers Bluetooth Low Energy devices within range (~100m).
 *
 * Scans for a short window (configurable, default 5 seconds) to conserve battery,
 * then returns a snapshot of discovered devices. RSSI (signal strength) is used as
 * a proximity estimator for identity edge confidence.
 *
 * Privacy: raw MAC addresses are sent over TLS to the API and HMAC-hashed there
 * with a KMS-backed daily-rotating pepper. Edge no longer hashes — that broke
 * server-side rotation policy. Device names are dropped (free-text PII risk).
 *
 * Each scan record is also fed through [BleAdvertisementParser] which extracts
 * RPA-stable manufacturer payloads (Apple Continuity sub-types, Microsoft Swift
 * Pair, Google Fast Pair, iBeacon, Eddystone) — these survive MAC rotation and
 * power the resolved-device clustering done in ClickHouse.
 *
 * Permission required: BLUETOOTH_SCAN (Android 12+, declared with neverForUser).
 */
data class BleScanResultData(
    val rawAddress: String,
    val rssi: Int,
    val deviceType: Int,  // BluetoothDevice.DEVICE_TYPE_*
    // RPA-stable parsed manufacturer/Continuity fields. All nullable so older
    // agents emitting only (rawAddress, rssi, deviceType) keep working.
    val manufacturerCompanyId: Int? = null,
    val appleContinuitySubtype: Int? = null,
    val stableManufacturerPayloadHex: String? = null,
    val serviceUuids: List<String>? = null,
    val txPowerDbm: Int? = null,
    val ibeaconUuid: String? = null,
    val ibeaconMajor: Int? = null,
    val ibeaconMinor: Int? = null
)

data class BleScanSnapshot(
    val devices: List<BleScanResultData>,
    val deviceCount: Int,
    val scanDurationMs: Long,
    val scanTimestampMs: Long,
    /**
     * Why this scan produced zero rows, when zero rows is unexpected.
     * Null = scan ran successfully (rows may be 0 because no beacons in
     * range — observable noise). Non-null = best-effort path bailed and
     * the heartbeat builder should surface the reason on the payload so
     * a server-side cron can detect the per-screen drop and alert.
     *
     * Class fix for the 2026-05-03 SM-X730 silent-BLE incident: a missing
     * runtime perm produced zero CH rows for >24h with no observable
     * signal in any dashboard. Skip reasons now travel as first-class
     * heartbeat fields.
     */
    val skipReason: String? = null
)

object BleBeaconScanner {

    private const val TAG = "BleBeaconScanner"
    private const val DEFAULT_SCAN_DURATION_MS = 5000L

    // Skip-reason enum — kept as plain strings so the heartbeat-payload
    // pass-through stays primitive and the server-side classifier can
    // group on them without a Kotlin/JS shared schema dance.
    const val SKIP_PERMISSION_DENIED = "permission_denied"
    const val SKIP_BLE_NOT_SUPPORTED = "ble_not_supported"
    const val SKIP_ADAPTER_UNAVAILABLE = "adapter_unavailable"
    const val SKIP_BLUETOOTH_DISABLED = "bluetooth_disabled"
    const val SKIP_SCANNER_UNAVAILABLE = "scanner_unavailable"
    const val SKIP_ALREADY_IN_PROGRESS = "already_in_progress"
    private const val MAX_SCAN_DURATION_MS = 10000L
    private const val MAX_DEVICES = 100

    private val scanInProgress = AtomicBoolean(false)

    /**
     * Scan for BLE devices for a bounded time window.
     *
     * @param context Application context
     * @param scanDurationMs Duration to scan in milliseconds (default 5000, max 10000)
     * @return [BleScanSnapshot] or null if BLE is not available/permitted
     */
    suspend fun scan(
        context: Context,
        scanDurationMs: Long = DEFAULT_SCAN_DURATION_MS
    ): BleScanSnapshot? = withContext(Dispatchers.IO) {
        if (!scanInProgress.compareAndSet(false, true)) {
            // Already-in-progress is the one skip we deliberately do NOT
            // surface upward — it's a benign duplicate-call guard, not a
            // misconfiguration. Caller's previous in-flight scan will
            // emit the snapshot. Log at debug so it stays out of warn-
            // level dashboards.
            Log.d(TAG, "Scan already in progress — skipping")
            return@withContext null
        }

        try {
            val effectiveDuration = scanDurationMs.coerceIn(1000L, MAX_SCAN_DURATION_MS)
            val skipNow = System.currentTimeMillis()

            if (!hasPermission(context)) {
                // Promoted to W: a kiosk that's been deployed without the
                // runtime BLUETOOTH_SCAN grant produces ZERO ble rows
                // forever; this used to debug-log silently. Now every
                // scan-cycle skip prints once at warn-level AND the
                // skipReason flows up through BleScanSnapshot to the
                // heartbeat payload (see SKIP_PERMISSION_DENIED handling
                // in DeviceAgentService heartbeat builder).
                Log.w(TAG, "Missing BLUETOOTH_SCAN permission — skipping (no perm grant)")
                return@withContext emptySnapshotWithReason(SKIP_PERMISSION_DENIED, scanStart = skipNow)
            }

            if (!context.packageManager.hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)) {
                Log.w(TAG, "BLE not supported on this device — skipping")
                return@withContext emptySnapshotWithReason(SKIP_BLE_NOT_SUPPORTED, scanStart = skipNow)
            }

            val bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
                ?: return@withContext emptySnapshotWithReason(SKIP_ADAPTER_UNAVAILABLE, scanStart = skipNow)

            val adapter: BluetoothAdapter = bluetoothManager.adapter
                ?: run {
                    Log.w(TAG, "BluetoothAdapter not available — skipping")
                    return@withContext emptySnapshotWithReason(SKIP_ADAPTER_UNAVAILABLE, scanStart = skipNow)
                }

            if (!adapter.isEnabled) {
                Log.w(TAG, "Bluetooth disabled — skipping")
                return@withContext emptySnapshotWithReason(SKIP_BLUETOOTH_DISABLED, scanStart = skipNow)
            }

            val scanner: BluetoothLeScanner = adapter.bluetoothLeScanner
                ?: run {
                    Log.w(TAG, "BluetoothLeScanner not available — skipping")
                    return@withContext emptySnapshotWithReason(SKIP_SCANNER_UNAVAILABLE, scanStart = skipNow)
                }

            val scanStart = System.currentTimeMillis()
            val discoveredDevices = CopyOnWriteArrayList<ScanResult>()

            // Use low-power scan mode to conserve battery on always-on CTV devices
            val settings = ScanSettings.Builder()
                .setScanMode(ScanSettings.SCAN_MODE_LOW_POWER)
                .setReportDelay(0) // Immediate results
                .build()

            val callback = object : ScanCallback() {
                override fun onScanResult(callbackType: Int, result: ScanResult?) {
                    result?.let {
                        if (discoveredDevices.size < MAX_DEVICES * 2) {
                            discoveredDevices.add(it)
                        }
                    }
                }

                override fun onBatchScanResults(results: MutableList<ScanResult>?) {
                    results?.let {
                        for (r in it) {
                            if (discoveredDevices.size < MAX_DEVICES * 2) {
                                discoveredDevices.add(r)
                            }
                        }
                    }
                }

                override fun onScanFailed(errorCode: Int) {
                    Log.w(TAG, "BLE scan failed with error code: $errorCode")
                }
            }

            try {
                scanner.startScan(emptyList<ScanFilter>(), settings, callback)
            } catch (e: SecurityException) {
                Log.w(TAG, "SecurityException starting BLE scan: ${e.message}")
                return@withContext null
            }

            // Wait for the scan duration
            delay(effectiveDuration)

            // Stop the scan
            try {
                scanner.stopScan(callback)
            } catch (_: Exception) {}

            val scanDuration = System.currentTimeMillis() - scanStart

            if (discoveredDevices.isEmpty()) {
                Log.d(TAG, "No BLE devices discovered")
                return@withContext BleScanSnapshot(
                    devices = emptyList(),
                    deviceCount = 0,
                    scanDurationMs = scanDuration,
                    scanTimestampMs = scanStart
                )
            }

            // Deduplicate by device address, keep strongest RSSI per address
            val dedupedByAddress = discoveredDevices
                .filter { it.device?.address != null }
                .groupBy { it.device.address }
                .mapValues { (_, scans) -> scans.maxByOrNull { it.rssi } ?: scans.first() }
                .values
                .sortedByDescending { it.rssi } // Strongest (closest) first
                .take(MAX_DEVICES)

            val devices = dedupedByAddress.map { result ->
                buildResultData(
                    rawAddress = result.device.address,
                    rssi = result.rssi,
                    deviceType = try { result.device.type } catch (_: Exception) { 0 },
                    scanRecordBytes = try { result.scanRecord?.bytes } catch (_: Exception) { null }
                )
            }

            val snapshot = BleScanSnapshot(
                devices = devices,
                deviceCount = devices.size,
                scanDurationMs = scanDuration,
                scanTimestampMs = scanStart
            )

            Log.d(TAG, "BLE scan complete: ${devices.size} devices in ${scanDuration}ms")
            snapshot
        } catch (e: Exception) {
            Log.w(TAG, "BLE scan failed: ${e.message}")
            null
        } finally {
            scanInProgress.set(false)
        }
    }

    /**
     * Build an empty snapshot tagged with a [skipReason] so the heartbeat
     * builder can surface the reason on the wire instead of treating
     * scanner unavailability as observable noise (zero rows in CH).
     *
     * Class fix for the silent-zero-rows bug: every `null` return used to
     * mean "scanner couldn't run — produce no observable signal", which
     * dovetailed with the heartbeat builder treating absent BLE rows as
     * "no beacons in range." Now every recoverable failure ships the
     * reason upward.
     */
    private fun emptySnapshotWithReason(reason: String, scanStart: Long): BleScanSnapshot =
        BleScanSnapshot(
            devices = emptyList(),
            deviceCount = 0,
            scanDurationMs = 0,
            scanTimestampMs = scanStart,
            skipReason = reason
        )

    /**
     * Pure-domain builder used by [scan] and exposed for JVM unit tests so the
     * parser-wiring path can be exercised without an Android emulator.
     */
    @JvmStatic
    fun buildResultData(
        rawAddress: String,
        rssi: Int,
        deviceType: Int,
        scanRecordBytes: ByteArray?
    ): BleScanResultData {
        val parsed = BleAdvertisementParser.parse(scanRecordBytes ?: byteArrayOf())
        return BleScanResultData(
            rawAddress = rawAddress,
            rssi = rssi,
            deviceType = deviceType,
            manufacturerCompanyId = parsed.manufacturerCompanyId,
            appleContinuitySubtype = parsed.appleContinuitySubtype,
            stableManufacturerPayloadHex = parsed.stableManufacturerPayload?.let {
                // Reuse parser's lookup-table hex helper (~3x faster than
                // joinToString + "%02x".format on the ~33K call/sec hot path).
                BleAdvertisementParser.bytesToHex(it)
            },
            serviceUuids = parsed.serviceUuids.takeIf { it.isNotEmpty() },
            txPowerDbm = parsed.txPowerDbm,
            ibeaconUuid = parsed.ibeaconUuid,
            ibeaconMajor = parsed.ibeaconMajor,
            ibeaconMinor = parsed.ibeaconMinor
        )
    }

    private fun hasPermission(context: Context): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Android 12+: BLUETOOTH_SCAN
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.BLUETOOTH_SCAN
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            // Older: ACCESS_FINE_LOCATION covers BLE scanning
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

}
