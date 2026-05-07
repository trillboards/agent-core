package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for BleBeaconScanner.
 *
 * Tests requiring BluetoothLeScanner are integration tests that must run
 * on a device/emulator. These tests verify data class contracts, the
 * pure-domain `buildResultData` helper that wires [BleAdvertisementParser]
 * into the scan callback, and snapshot structure.
 */
class BleBeaconScannerTest {

    private fun hex(s: String): ByteArray {
        val clean = s.replace(" ", "")
        return ByteArray(clean.length / 2) {
            ((Character.digit(clean[it * 2], 16) shl 4) +
                Character.digit(clean[it * 2 + 1], 16)).toByte()
        }
    }

    @Test
    fun `BleScanResultData should hold raw address and default null parsed fields`() {
        val result = BleScanResultData(
            rawAddress = "AA:BB:CC:DD:EE:FF",
            rssi = -65,
            deviceType = 2
        )

        assertEquals("AA:BB:CC:DD:EE:FF", result.rawAddress)
        assertEquals(-65, result.rssi)
        assertEquals(2, result.deviceType)
        // Older agents emit only (rawAddress, rssi, deviceType); parsed fields default null.
        assertNull(result.manufacturerCompanyId)
        assertNull(result.appleContinuitySubtype)
        assertNull(result.stableManufacturerPayloadHex)
        assertNull(result.serviceUuids)
        assertNull(result.txPowerDbm)
        assertNull(result.ibeaconUuid)
    }

    @Test
    fun `buildResultData feeds scan record bytes through BleAdvertisementParser`() {
        val ibeaconBytes = hex(
            "1AFF4C000215E2C56DB5DFFB48D2B060D0F5A71096E0000100020C00"
        )
        val data = BleBeaconScanner.buildResultData(
            rawAddress = "11:22:33:44:55:66",
            rssi = -55,
            deviceType = 2,
            scanRecordBytes = ibeaconBytes
        )
        assertEquals("11:22:33:44:55:66", data.rawAddress)
        assertEquals(-55, data.rssi)
        assertEquals(0x004c, data.manufacturerCompanyId)
        assertEquals(1, data.ibeaconMajor)
        assertEquals(2, data.ibeaconMinor)
        // iBeacon UUID present, hex string lower-cased
        assertEquals("e2c56db5-dffb-48d2-b060-d0f5a71096e0", data.ibeaconUuid)
    }

    @Test
    fun `buildResultData with null scan record yields all-null parsed fields`() {
        val data = BleBeaconScanner.buildResultData(
            rawAddress = "11:22:33:44:55:66",
            rssi = -55,
            deviceType = 0,
            scanRecordBytes = null
        )
        assertNull(data.manufacturerCompanyId)
        assertNull(data.appleContinuitySubtype)
        assertNull(data.stableManufacturerPayloadHex)
    }

    @Test
    fun `buildResultData extracts Apple Continuity Nearby Action sub-type as flat fields`() {
        val raw = hex("0AFF4C0010052102018004020A0C")
        val data = BleBeaconScanner.buildResultData(
            rawAddress = "AA:BB:CC:DD:EE:FF", rssi = -50, deviceType = 2, scanRecordBytes = raw
        )
        assertEquals(0x004c, data.manufacturerCompanyId)
        assertEquals(0x10, data.appleContinuitySubtype)
        assertNotNull(data.stableManufacturerPayloadHex)
        // Hex string is hex-only lowercase
        assertTrue(data.stableManufacturerPayloadHex!!.all { it in '0'..'9' || it in 'a'..'f' })
        // Continuity 0x10 keeps only StatusFlags + ActionCode (2 bytes -> 4 hex chars).
        // Trailing AuthTag bytes rotate and MUST NOT appear in the stable payload.
        assertEquals("2102", data.stableManufacturerPayloadHex)
        assertEquals(12, data.txPowerDbm) // included via the trailing TX-power TLV
    }

    @Test
    fun `BleScanSnapshot should contain devices, count, and timing`() {
        val devices = listOf(
            BleScanResultData("AA:BB:CC:DD:EE:01", -40, 1),
            BleScanResultData("AA:BB:CC:DD:EE:02", -60, 2),
        )
        val snapshot = BleScanSnapshot(
            devices = devices,
            deviceCount = 2,
            scanDurationMs = 5000,
            scanTimestampMs = System.currentTimeMillis()
        )
        assertEquals(2, snapshot.devices.size)
        assertEquals(2, snapshot.deviceCount)
        assertEquals(5000L, snapshot.scanDurationMs)
        assertTrue(snapshot.scanTimestampMs > 0)
    }

    @Test
    fun `RSSI values should be negative (dBm)`() {
        val result = BleScanResultData("AA:BB:CC:DD:EE:FF", -55, 1)
        assertTrue(result.rssi < 0)
    }

    @Test
    fun `scan duration should be bounded`() {
        val min = 1000L
        val max = 10000L
        assertEquals(1000L, 500L.coerceIn(min, max))
        assertEquals(5000L, 5000L.coerceIn(min, max))
        assertEquals(10000L, 15000L.coerceIn(min, max))
    }
}
