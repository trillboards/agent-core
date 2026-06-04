package com.trillboards.ctv.core.identity

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
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
import com.trillboards.ctv.core.audience.DeviceProfile
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
    val ibeaconMinor: Int? = null,
    // Venue-insights PR 7 (E.2): per-device address type from
    // `BluetoothDevice.getAddressType()` (API 31+). One of:
    //   "public"    — burned-in MAC (infrastructure, fitness trackers, smart
    //                 speakers) — strong L1-clustering stability hint
    //   "random"    — RPA / privacy-rotated address (most modern phones)
    //   "anonymous" — anonymous LE address (rarely used)
    //   "unknown"   — pre-API-31 device, or getAddressType() returned the
    //                 ADDRESS_TYPE_UNKNOWN sentinel
    // Null on older agents that don't capture this; server normalizes to
    // 'unknown' via the `address_type LowCardinality(String) DEFAULT 'unknown'`
    // column.
    val addrType: String? = null,
    // Venue-insights PR 7 (E.3): paired-device flag from
    // `BluetoothDevice.getBondState()` — true when the device is `BOND_BONDED`,
    // false otherwise (`BOND_NONE` or `BOND_BONDING`). Paired devices (the
    // venue owner's iPad, the operator's phone) get auto-classified as
    // stationary at L0 ingest. Null on older agents; server defaults to false.
    val isPaired: Boolean? = null
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
    // Per-cycle scan window. Bumped 5s → 30s as part of venue-insights PR 7
    // (E.1): the previous 5-second window under-sampled Samsung devices
    // broadcasting at ~1–2 Hz (the dashboard showed Samsung at 0.4% of the
    // mix while Apple Continuity at ~10 Hz dominated). At LOW_LATENCY duty
    // cycle 30s captures multiple ADV intervals from slower advertisers
    // without thrashing the radio across heartbeat cycles.
    private const val DEFAULT_SCAN_DURATION_MS = 30000L

    // Skip-reason enum — kept as plain strings so the heartbeat-payload
    // pass-through stays primitive and the server-side classifier can
    // group on them without a Kotlin/JS shared schema dance.
    const val SKIP_PERMISSION_DENIED = "permission_denied"
    const val SKIP_BLE_NOT_SUPPORTED = "ble_not_supported"
    const val SKIP_ADAPTER_UNAVAILABLE = "adapter_unavailable"
    const val SKIP_BLUETOOTH_DISABLED = "bluetooth_disabled"
    const val SKIP_SCANNER_UNAVAILABLE = "scanner_unavailable"
    const val SKIP_ALREADY_IN_PROGRESS = "already_in_progress"
    // Bumped 10s → 60s to allow the 30s default + headroom for callers that
    // explicitly request a longer window (still bounded to keep heartbeat
    // cadence sane).
    private const val MAX_SCAN_DURATION_MS = 60000L
    private const val MAX_DEVICES = 100

    // Venue-insights PR 7 (E.2/E.3): the addressType constants moved from
    // BluetoothDevice.ADDRESS_TYPE_* (added in API 31, but values intentional
    // duplicates) to public Int values. Mirror them here so we can map without
    // pulling the Android constant on older SDKs.
    //   PUBLIC    = 0  — burned-in MAC
    //   RANDOM    = 1  — RPA / privacy-rotated
    //   ANONYMOUS = 0xff — anonymous LE address (rare)
    //   UNKNOWN   = 0xfe — getAddressType() unable to determine
    private const val BT_ADDR_TYPE_PUBLIC = 0
    private const val BT_ADDR_TYPE_RANDOM = 1
    private const val BT_ADDR_TYPE_ANONYMOUS = 0xff
    private const val BT_ADDR_TYPE_UNKNOWN_SENTINEL = 0xfe

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

            // Venue-insights PR 7 (E.1): hardware-gated SCAN_MODE_LOW_LATENCY.
            //
            // The OS-level ScanController on Samsung Tab S11 is firing per-ADV
            // at multi-Hz with addressType / primaryPhy / secondaryPhy /
            // advertisingSid / txPower / rssi / eventType per packet. The data
            // is at the system surface; the scanner just wasn't requesting
            // the high-duty-cycle mode. Apple Continuity (~10 Hz) was over-
            // sampled and Samsung devices (~1–2 Hz ADV) were under-sampled
            // under LOW_POWER, hence the dashboard's 83% Apple / 0.4% Samsung
            // manufacturer mix.
            //
            // LOW_LATENCY on combo BT+WiFi radios (older Fire TV sticks,
            // MediaTek budget devices) is documented to degrade WiFi
            // throughput. Gate it on DeviceProfile.hasSeparateBtWifiRadios:
            // only Tier-1/2 devices on Qualcomm-flagship, RK3588, or Exynos
            // chipsets opt in. Everything else stays on LOW_POWER (the
            // conservative default).
            //
            // setLegacy(false) + setPhy(PHY_LE_ALL_SUPPORTED) opt into BLE 5
            // extended advertising — captures both the legacy 1M PHY and
            // the BLE 5 Coded/2M PHY broadcasters. No-op on older devices.
            val capability = try {
                DeviceProfile.detectCapabilityMatrix(context)
            } catch (e: Exception) {
                Log.w(TAG, "DeviceCapabilityMatrix detect failed; defaulting to LOW_POWER: ${e.message}")
                null
            }
            val useLowLatency = capability != null
                && DeviceProfile.hasSeparateBtWifiRadios(capability)
            val scanMode = if (useLowLatency) {
                ScanSettings.SCAN_MODE_LOW_LATENCY
            } else {
                ScanSettings.SCAN_MODE_LOW_POWER
            }
            val settingsBuilder = ScanSettings.Builder()
                .setScanMode(scanMode)
                .setReportDelay(0) // Immediate results
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                // BLE 5 extended advertising opt-in — gated to API 26+ where
                // the setters exist. Older devices fall through unchanged.
                settingsBuilder.setLegacy(false)
                settingsBuilder.setPhy(ScanSettings.PHY_LE_ALL_SUPPORTED)
            }
            val settings = settingsBuilder.build()

            Log.i(
                TAG,
                "[BleBeaconScanner] scanMode=" +
                    (if (useLowLatency) "LOW_LATENCY" else "LOW_POWER") +
                    ", window=${effectiveDuration}ms" +
                    ", setLegacy=false" +
                    ", phy=PHY_LE_ALL_SUPPORTED" +
                    ", chipset=${capability?.chipsetVendor}" +
                    ", tier=${capability?.tier}"
            )

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
                    scanRecordBytes = try { result.scanRecord?.bytes } catch (_: Exception) { null },
                    // Venue-insights PR 7 (E.2): address-type from
                    // BluetoothDevice.getAddressType() (API 31+). Samsung Tab
                    // S11 is API 36 so the call always succeeds here; older
                    // devices fall through with null and the server normalizes
                    // to 'unknown'.
                    addrType = readAddressType(result.device),
                    // Venue-insights PR 7 (E.3): bondState from
                    // BluetoothDevice.getBondState(). Available since API 18 so
                    // no SDK gate needed; wrapped in try/catch for the rare
                    // SecurityException path (BLUETOOTH_CONNECT not granted —
                    // BLUETOOTH_SCAN is granted but CONNECT is separate on
                    // Android 12+).
                    isPaired = readBondState(result.device)
                )
            }

            val snapshot = BleScanSnapshot(
                devices = devices,
                deviceCount = devices.size,
                scanDurationMs = scanDuration,
                scanTimestampMs = scanStart
            )

            Log.d(TAG, "BLE scan complete: ${devices.size} devices in ${scanDuration}ms")
            // Venue-insights PR 7 (E.2/E.3): log per-cycle distribution of the
            // new addr_type + is_paired captures so we can confirm wire-side
            // capture from logcat without an HTTP body interceptor. Single
            // line per cycle, INFO level, no PII.
            if (devices.isNotEmpty()) {
                val addrTypeHistogram = devices.groupingBy { it.addrType ?: "null" }.eachCount()
                val pairedCount = devices.count { it.isPaired == true }
                Log.i(
                    TAG,
                    "[BleBeaconScanner] capture summary: " +
                        "addr_type=$addrTypeHistogram, " +
                        "is_paired_count=$pairedCount/${devices.size}"
                )
            }
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
     *
     * [addrType] and [isPaired] are venue-insights PR 7 (E.2/E.3) additions —
     * captured at the scan-callback boundary in [scan] and threaded through
     * here. Default to null so JVM unit tests that don't supply them get the
     * pre-PR-7 behavior (server defaults to 'unknown' / false).
     */
    @JvmStatic
    @JvmOverloads
    fun buildResultData(
        rawAddress: String,
        rssi: Int,
        deviceType: Int,
        scanRecordBytes: ByteArray?,
        addrType: String? = null,
        isPaired: Boolean? = null
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
            ibeaconMinor = parsed.ibeaconMinor,
            addrType = addrType,
            isPaired = isPaired
        )
    }

    /**
     * Venue-insights PR 7 (E.2): read `BluetoothDevice.getAddressType()` and
     * normalize to a wire-friendly lowercase string. Available since API 31
     * (Build.VERSION_CODES.S). Returns null on older devices; server-side
     * column `address_type LowCardinality(String) DEFAULT 'unknown'` handles
     * the omission.
     *
     * The Android constants are:
     *   BluetoothDevice.ADDRESS_TYPE_PUBLIC    = 0
     *   BluetoothDevice.ADDRESS_TYPE_RANDOM    = 1
     *   BluetoothDevice.ADDRESS_TYPE_ANONYMOUS = 0xff (since API 34)
     *   BluetoothDevice.ADDRESS_TYPE_UNKNOWN   = 0xfe (since API 34)
     *
     * Wrapped in try/catch for the `IllegalStateException: address type cannot
     * be resolved` path some manufacturers hit when the device hasn't been
     * fully advertised. Caller treats null as "unknown".
     */
    private fun readAddressType(device: BluetoothDevice?): String? {
        if (device == null) return null
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return null
        return try {
            when (val raw = device.addressType) {
                BT_ADDR_TYPE_PUBLIC -> "public"
                BT_ADDR_TYPE_RANDOM -> "random"
                BT_ADDR_TYPE_ANONYMOUS -> "anonymous"
                BT_ADDR_TYPE_UNKNOWN_SENTINEL -> "unknown"
                else -> {
                    // Future-proofing — Android may add new constants.
                    Log.d(TAG, "Unknown BluetoothDevice.addressType raw=$raw")
                    "unknown"
                }
            }
        } catch (e: SecurityException) {
            // BLUETOOTH_CONNECT not granted — addressType requires it on some
            // OEM builds even when BLUETOOTH_SCAN is fine. Caller falls
            // through to 'unknown'.
            null
        } catch (e: Exception) {
            // IllegalStateException / IllegalArgumentException etc — addressType
            // can throw "cannot be resolved" when the device hasn't been
            // fully advertised.
            null
        }
    }

    /**
     * Venue-insights PR 7 (E.3): read `BluetoothDevice.getBondState()` and
     * return true when the device is bonded (BOND_BONDED = 12). False for
     * BOND_NONE / BOND_BONDING. Available since API 18; no SDK gate needed.
     * Wrapped in try/catch for the rare SecurityException path
     * (BLUETOOTH_CONNECT required on Android 12+ for some OEM builds).
     */
    private fun readBondState(device: BluetoothDevice?): Boolean? {
        if (device == null) return null
        return try {
            device.bondState == BluetoothDevice.BOND_BONDED
        } catch (e: SecurityException) {
            null
        } catch (e: Exception) {
            null
        }
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
