package com.trillboards.ctv.core.identity

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for BleAdvertisementParser.
 *
 * Golden-file fixtures use hex-encoded sample BLE advertisements built from
 * public protocol specs (Apple Continuity, Microsoft Swift Pair, Google
 * Fast Pair, iBeacon, Eddystone-UID). Each fixture is verified against
 * expected manufacturer / sub-type / payload extraction so RPA-survival
 * fields stay stable under MAC rotation.
 *
 * RPA-rotation classification follows the public reverse-engineered
 * Continuity spec (furiousmac.com/handoff): only the listed STABLE sub-types
 * keep payload bytes; everything else gets `stableManufacturerPayload = null`.
 * Find My (0x12) and AirDrop (0x05) are EXPLICITLY rejected — they are the
 * categorical "rotating" identifiers this parser exists to defeat.
 */
class BleAdvertisementParserTest {

    private fun hex(s: String): ByteArray {
        val clean = s.replace(" ", "").replace("\n", "")
        require(clean.length % 2 == 0) { "Hex string length must be even: ${clean.length}" }
        return ByteArray(clean.length / 2) {
            ((Character.digit(clean[it * 2], 16) shl 4) +
                Character.digit(clean[it * 2 + 1], 16)).toByte()
        }
    }

    @Test
    fun `null or empty input returns all-null result`() {
        val empty = BleAdvertisementParser.parse(byteArrayOf())
        assertNull(empty.manufacturerCompanyId)
        assertNull(empty.appleContinuitySubtype)
        assertNull(empty.stableManufacturerPayload)
        assertNull(empty.txPowerDbm)
        assertEquals(emptyList<String>(), empty.serviceUuids)
        assertFalse(empty.isIbeacon)
        assertFalse(empty.isEddystone)
    }

    @Test
    fun `iBeacon advertisement is parsed correctly`() {
        val raw = hex(
            "1AFF4C0002 15 E2C56DB5DFFB48D2B060D0F5A71096E0 0001 0002 C5 00"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertTrue(parsed.isIbeacon)
        assertEquals("e2c56db5-dffb-48d2-b060-d0f5a71096e0", parsed.ibeaconUuid)
        assertEquals(1, parsed.ibeaconMajor)
        assertEquals(2, parsed.ibeaconMinor)
        assertEquals(-59, parsed.txPowerDbm)
        // iBeacon now contributes to the cross-format stable payload via the
        // synthetic SUBTYPE_IBEACON marker so server-side clustering picks it up.
        assertEquals(BleAdvertisementParser.SUBTYPE_IBEACON, parsed.appleContinuitySubtype)
        assertNotNull(parsed.stableManufacturerPayload)
        // Layout: [SUBTYPE_IBEACON || UUID(16) || major(2) || minor(2)] = 21 bytes.
        // TxPower (calibration data) is intentionally excluded.
        assertEquals(21, parsed.stableManufacturerPayload!!.size)
        assertEquals(BleAdvertisementParser.SUBTYPE_IBEACON.toByte(), parsed.stableManufacturerPayload!![0])
        // First UUID byte = 0xE2.
        assertEquals(0xE2.toByte(), parsed.stableManufacturerPayload!![1])
        // Major bytes 0x00, 0x01 land at indices 17, 18.
        assertEquals(0x00.toByte(), parsed.stableManufacturerPayload!![17])
        assertEquals(0x01.toByte(), parsed.stableManufacturerPayload!![18])
        // Minor bytes 0x00, 0x02 land at indices 19, 20.
        assertEquals(0x00.toByte(), parsed.stableManufacturerPayload!![19])
        assertEquals(0x02.toByte(), parsed.stableManufacturerPayload!![20])
    }

    @Test
    fun `iBeacon stable payload is identical for repeated UUID-Major-Minor packets`() {
        // Different rolling MAC + different TxPower bytes still produce the SAME
        // stable manufacturer payload — that is the whole point of the field.
        val a = hex("1AFF4C0002 15 11223344556677889900AABBCCDDEEFF 1234 5678 C5 00")
        val b = hex("1AFF4C0002 15 11223344556677889900AABBCCDDEEFF 1234 5678 B0 00")
        val pa = BleAdvertisementParser.parse(a)
        val pb = BleAdvertisementParser.parse(b)
        assertNotNull(pa.stableManufacturerPayload)
        assertNotNull(pb.stableManufacturerPayload)
        assertEquals(
            BleAdvertisementParser.bytesToHex(pa.stableManufacturerPayload!!),
            BleAdvertisementParser.bytesToHex(pb.stableManufacturerPayload!!)
        )
        // TxPower differs across packets.
        assertEquals(-59, pa.txPowerDbm)
        assertEquals(-80, pb.txPowerDbm)
    }

    @Test
    fun `iBeacon with subtype byte 02 but bad inner length is rejected (no phantom subtype)`() {
        // 0x02 outer subtype but inner length byte != 0x15 — must NOT fall through
        // to generic Continuity (would emit appleContinuitySubtype=0x02 phantoms).
        val raw = hex(
            "0BFF4C00 02 09 11223344556677 88"  // 9 bytes payload claimed but not iBeacon length
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertFalse("Bogus 0x02 must not be reported as iBeacon", parsed.isIbeacon)
        // Subtype 0x02 with non-iBeacon length must reject as ambiguous (no stable payload).
        assertNull(
            "Subtype 0x02 with non-iBeacon length must reject as ambiguous (no stable payload)",
            parsed.stableManufacturerPayload
        )
    }

    @Test
    fun `Apple AirDrop sub-type 05 is rejected as RPA-rotating`() {
        val raw = hex("0D FF 4C00 05 08 AABBCCDD11223344")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x05, parsed.appleContinuitySubtype)
        assertNull(parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Hey Siri sub-type 08 is rejected as RPA-rotating`() {
        // Hey Siri (0x08) carries a per-session RNG nonce / session ID — rotating.
        val raw = hex("0BFF4C00 08 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x08, parsed.appleContinuitySubtype)
        assertNull("Hey Siri 0x08 carries per-session nonce", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Magic Switch sub-type 0a is rejected as RPA-rotating`() {
        // Magic Switch / Tethering Target Presence (0x0a) — rotating Auth tag.
        val raw = hex("0BFF4C00 0A 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0a, parsed.appleContinuitySubtype)
        assertNull("Magic Switch 0x0a carries rotating Auth tag", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Watch Connectivity sub-type 0b is rejected as RPA-rotating`() {
        // Watch Connectivity (0x0b) — rotating session bytes between watch + phone.
        val raw = hex("0BFF4C00 0B 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0b, parsed.appleContinuitySubtype)
        assertNull("Watch Connectivity 0x0b carries rotating session bytes", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Watch Bridge sub-type 11 is rejected as RPA-rotating`() {
        // Watch Bridge (0x11) — rotating session token.
        val raw = hex("0BFF4C00 11 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x11, parsed.appleContinuitySubtype)
        assertNull("Watch Bridge 0x11 carries rotating session token", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Tethering Source sub-type 14 is rejected as RPA-rotating`() {
        // Tethering Source (0x14) — rotating session bytes.
        val raw = hex("0BFF4C00 14 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x14, parsed.appleContinuitySubtype)
        assertNull("Tethering Source 0x14 carries rotating session bytes", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple unknown future sub-type fails closed (no payload bytes)`() {
        // 0x18, 0x19, 0x1B etc. are sub-types Apple has shipped post-spec-publication.
        // The parser MUST NOT optimistically classify unknown subtypes as stable —
        // that would silently corrupt clusters when a future rotating subtype lands.
        // Instead the subtype is reported (so dashboards can flag it for triage)
        // but no bytes contribute to clustering until the subtype is researched
        // and explicitly added to APPLE_ROTATING_SUBTYPES or given a stable-prefix
        // case in parseApple.
        val raw = hex("0BFF4C00 1F 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x1F, parsed.appleContinuitySubtype)
        assertNull(
            "Unknown Continuity sub-types must fail CLOSED (drop payload)",
            parsed.stableManufacturerPayload
        )
    }

    @Test
    fun `Apple AirPlay Source sub-type 09 is rejected as RPA-rotating`() {
        // 0x09 carries a rotating Auth tag (Apple ID hash). Parser MUST drop the payload.
        val raw = hex("0BFF4C00 09 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x09, parsed.appleContinuitySubtype)
        assertNull("AirPlay Source 0x09 carries rotating Auth tag", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Handoff sub-type 0c is rejected as RPA-rotating`() {
        val raw = hex("0BFF4C00 0C 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0c, parsed.appleContinuitySubtype)
        assertNull("Handoff 0x0c carries encrypted rotating Auth tag", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Wifi Settings sub-type 0d is rejected as RPA-rotating`() {
        val raw = hex("0BFF4C00 0D 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0d, parsed.appleContinuitySubtype)
        assertNull(parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Instant Hotspot sub-type 0e is rejected as RPA-rotating`() {
        val raw = hex("0BFF4C00 0E 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0e, parsed.appleContinuitySubtype)
        assertNull(parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Wifi Join sub-type 0f is rejected as RPA-rotating`() {
        val raw = hex("0BFF4C00 0F 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0f, parsed.appleContinuitySubtype)
        assertNull(parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Find My sub-type 12 is rejected as RPA-rotating`() {
        // Find My (0x12) rotates every 15 minutes via secp224r1 key derivation.
        // This is the categorical example of a rotating identifier — must reject.
        val raw = hex("1CFF4C00 12 19 0102030405060708090A0B0C0D0E0F1011121314151617")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x12, parsed.appleContinuitySubtype)
        assertNull("Find My 0x12 rotates every 15 min — must NEVER be stable", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Proximity Pairing sub-type 16 is rejected as RPA-rotating`() {
        val raw = hex("0BFF4C00 16 06 010203040506")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x16, parsed.appleContinuitySubtype)
        assertNull("Proximity Pairing 0x16 carries rotating session token", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Apple Nearby Action sub-type 10 keeps only stable prefix bytes`() {
        // Sub-type 0x10 layout: [Status Flags 1B][Action Code 1B][AuthTag 3B][...]
        // Only the first 2 bytes (StatusFlags + ActionCode) are stable per device-class.
        val raw = hex(
            "02 01 06" +                    // flags AD field
            "0A FF 4C00 10 05 21 02 01 80 04" + // Continuity 0x10, sublen=5, body = 21 02 01 80 04
            "02 0A 0C"                       // TX power TLV
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x10, parsed.appleContinuitySubtype)
        assertNotNull(parsed.stableManufacturerPayload)
        // Only the first 2 bytes (StatusFlags=0x21, ActionCode=0x02) are stable.
        assertEquals(2, parsed.stableManufacturerPayload!!.size)
        assertEquals(0x21.toByte(), parsed.stableManufacturerPayload!![0])
        assertEquals(0x02.toByte(), parsed.stableManufacturerPayload!![1])
        assertEquals(12, parsed.txPowerDbm)
    }

    @Test
    fun `Apple AirPods sub-type 07 keeps only model-id prefix (max 9 bytes)`() {
        val raw = hex(
            "1E FF 4C00 07 19" +
            "01 20 75 AA BB CC DD EE FF" + // first 9 bytes: stable model id + state
            "11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF 00" // rotating tail
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x004c, parsed.manufacturerCompanyId)
        assertEquals(0x07, parsed.appleContinuitySubtype)
        assertNotNull(parsed.stableManufacturerPayload)
        assertTrue(
            "Stable payload should be <= 9 bytes (model id + state). got=${parsed.stableManufacturerPayload!!.size}",
            parsed.stableManufacturerPayload!!.size <= 9
        )
    }

    @Test
    fun `Microsoft Swift Pair beacon ID 03 keeps stable payload`() {
        // Microsoft 0x0006, beacon ID 0x03 = Swift Pair (stable model bytes).
        // TLV: [LEN=0x09][TYPE=0xFF][Company-LE=0600][BeaconID=03][Reserved=00][Model=12345678]
        val raw = hex("09 FF 0600 03 00 12345678")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0006, parsed.manufacturerCompanyId)
        assertNotNull(parsed.stableManufacturerPayload)
        assertTrue(parsed.stableManufacturerPayload!!.size >= 4)
    }

    @Test
    fun `Microsoft non-Swift-Pair beacon ID is rejected as untrusted`() {
        // Microsoft 0x0006 with beacon ID 0x01 (NOT Swift Pair). Could carry rotating
        // CDP / cross-device-promotion identifiers — must reject.
        val raw = hex("09 FF 0600 01 00 AABBCCDD")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0006, parsed.manufacturerCompanyId)
        assertNull(
            "Non-SwiftPair Microsoft beacon (id != 0x03) carries unknown identifiers",
            parsed.stableManufacturerPayload
        )
    }

    @Test
    fun `Google Fast Pair service data with FE2C extracts 24-bit BE model ID`() {
        // Google Fast Pair v1 model ID is 24 bits, BIG-ENDIAN, in the service-data
        // payload under 16-bit UUID 0xFE2C. Layout: [LEN=0x06][0x16][UUID-LE=2CFE][Model-3B-BE]
        val raw = hex(
            "06 16 2CFE 0A1B2C"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        // Service UUID list must include fe2c
        assertTrue(parsed.serviceUuids.any { it == "fe2c" })
        assertEquals(0x0A1B2C, parsed.googleFastPairModelId)
        assertNotNull(parsed.stableManufacturerPayload)
        // Stable payload is [SUBTYPE_FAST_PAIR || model_id_3B] = 4 bytes.
        assertEquals(4, parsed.stableManufacturerPayload!!.size)
        assertEquals(BleAdvertisementParser.SUBTYPE_FAST_PAIR.toByte(), parsed.stableManufacturerPayload!![0])
        assertEquals(0x0A.toByte(), parsed.stableManufacturerPayload!![1])
        assertEquals(0x1B.toByte(), parsed.stableManufacturerPayload!![2])
        assertEquals(0x2C.toByte(), parsed.stableManufacturerPayload!![3])
        // Subtype field carries the synthetic Fast Pair marker so the server
        // can demux without reparsing the UUID list.
        assertEquals(BleAdvertisementParser.SUBTYPE_FAST_PAIR, parsed.appleContinuitySubtype)
    }

    @Test
    fun `Eddystone-UID frame keeps namespace plus instance bytes`() {
        // Eddystone UID frame: type byte 0x00, then TX power, 10-byte namespace, 6-byte instance, 2 RFU.
        // Service-data TLV: [LEN=0x17][TYPE=0x16][UUID-LE=AAFE][FrameType=00][TxPower=EE][NS 10B][Instance 6B][RFU 2B]
        val raw = hex(
            "02 01 06" +
            "03 03 AAFE" +
            "17 16 AAFE 00 EE 00112233445566778899 AABBCCDDEEFF 0000"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.isEddystone)
        assertEquals(BleAdvertisementParser.EDDYSTONE_FRAME_UID, parsed.eddystoneFrameType)
        assertEquals("00112233445566778899aabbccddeeff", parsed.eddystoneIdHex)
        assertFalse(parsed.isIbeacon)
        // Eddystone UID now contributes to the cross-format stable payload via
        // [SUBTYPE_EDDYSTONE_UID || namespace(10) || instance(6)] = 17 bytes.
        assertNotNull(parsed.stableManufacturerPayload)
        assertEquals(17, parsed.stableManufacturerPayload!!.size)
        assertEquals(
            BleAdvertisementParser.SUBTYPE_EDDYSTONE_UID.toByte(),
            parsed.stableManufacturerPayload!![0]
        )
        assertEquals(BleAdvertisementParser.SUBTYPE_EDDYSTONE_UID, parsed.appleContinuitySubtype)
    }

    @Test
    fun `Eddystone-UID stable payload is identical across rolling TxPower frames`() {
        // Same beacon, different calibration TX power byte → stable payload bytes
        // remain identical so resolved-device clustering can collapse them.
        val a = hex("17 16 AAFE 00 EE 00112233445566778899 AABBCCDDEEFF 0000")
        val b = hex("17 16 AAFE 00 80 00112233445566778899 AABBCCDDEEFF 0000")
        val pa = BleAdvertisementParser.parse(a)
        val pb = BleAdvertisementParser.parse(b)
        assertNotNull(pa.stableManufacturerPayload)
        assertNotNull(pb.stableManufacturerPayload)
        assertEquals(
            BleAdvertisementParser.bytesToHex(pa.stableManufacturerPayload!!),
            BleAdvertisementParser.bytesToHex(pb.stableManufacturerPayload!!)
        )
    }

    @Test
    fun `Eddystone-URL captures URL bytes into stable payload`() {
        // Eddystone-URL spec: [frameType=0x10][TxPower 1B][URL Scheme 1B][Encoded URL...].
        // Encoded URL "ample.com" with prefix 0x02 (https://) → 0x02 'a' 'm' 'p' 'l' 'e' 0x07 (.com)
        // Outer service-data TLV body = type(1) + UUID(2) + frame(1) + tx(1) + scheme(1) + 'ample'(5) + 0x07(1) = 12
        val raw = hex(
            "0C 16 AAFE 10 EE 02 61 6D 70 6C 65 07"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.isEddystone)
        assertEquals(BleAdvertisementParser.EDDYSTONE_FRAME_URL, parsed.eddystoneFrameType)
        // URL host is stable per advertiser. Stable payload = [SUBTYPE_EDDYSTONE_URL || url_scheme(1) || url_bytes...].
        assertNotNull(parsed.stableManufacturerPayload)
        assertEquals(BleAdvertisementParser.SUBTYPE_EDDYSTONE_URL, parsed.appleContinuitySubtype)
        // marker(1) + scheme(1) + 'ample'(5) + 0x07(1) = 8
        assertEquals(8, parsed.stableManufacturerPayload!!.size)
        assertEquals(BleAdvertisementParser.SUBTYPE_EDDYSTONE_URL.toByte(), parsed.stableManufacturerPayload!![0])
        assertEquals(0x02.toByte(), parsed.stableManufacturerPayload!![1]) // https:// scheme
        assertEquals('a'.code.toByte(), parsed.stableManufacturerPayload!![2])
    }

    @Test
    fun `Eddystone-EID rotating frame is detected and identifier dropped`() {
        // Eddystone EID frame type 0x30 — rotates every 8-1024s via AES-CTR.
        val raw = hex(
            "0D 16 AAFE 30 EE 0102030405060708"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.isEddystone)
        assertEquals(BleAdvertisementParser.EDDYSTONE_FRAME_EID, parsed.eddystoneFrameType)
        assertNull("EID frame is rotating — eddystoneIdHex must be null", parsed.eddystoneIdHex)
    }

    @Test
    fun `Eddystone-TLM rotating telemetry frame is detected and identifier dropped`() {
        // Eddystone TLM frame type 0x20 — telemetry (rotates as device state changes).
        val raw = hex(
            "11 16 AAFE 20 00 0BB8 1F40 00112233 44556677"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.isEddystone)
        assertEquals(BleAdvertisementParser.EDDYSTONE_FRAME_TLM, parsed.eddystoneFrameType)
        assertNull("TLM frame is volatile — eddystoneIdHex must be null", parsed.eddystoneIdHex)
    }

    @Test
    fun `random unknown payload does not crash and returns minimal result`() {
        val raw = hex("0102030405DEADBEEFCAFEBABE")
        val parsed = BleAdvertisementParser.parse(raw)
        assertNotNull(parsed)
        assertNull(parsed.manufacturerCompanyId)
        assertFalse(parsed.isIbeacon)
        assertFalse(parsed.isEddystone)
    }

    @Test
    fun `truncated TLV does not crash`() {
        val raw = hex("06 FF 4C 00 10")
        val parsed = BleAdvertisementParser.parse(raw)
        assertNotNull(parsed)
    }

    @Test
    fun `tlv with length 1 (only type byte) does not crash`() {
        val raw = hex("01 02 03 03 0DFE")
        val parsed = BleAdvertisementParser.parse(raw)
        // Length=1 means TLV body is empty — should be a no-op
        assertNotNull(parsed)
        assertTrue("Length-1 TLV must not break later TLVs", parsed.serviceUuids.contains("fe0d"))
    }

    @Test
    fun `mid-stream zero-pad bytes are skipped`() {
        val raw = hex("00 00 02 0A 0C 00 00 03 03 0DFE 00")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(12, parsed.txPowerDbm)
        assertTrue(parsed.serviceUuids.contains("fe0d"))
    }

    @Test
    fun `manufacturer TLV with body too short for company ID does not crash`() {
        // length=2 but body has only 1 byte after type — can't read 2-byte company ID
        val raw = hex("02 FF 4C")
        val parsed = BleAdvertisementParser.parse(raw)
        assertNotNull(parsed)
        assertNull("Body too short to contain 2-byte company ID — must reject", parsed.manufacturerCompanyId)
    }

    @Test
    fun `apple sub-length lying about body size does not crash`() {
        // 0x10 sub-len=0xFF claims 255 bytes but body is only 5 bytes
        val raw = hex("09 FF 4C00 10 FF 0102 0304")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x10, parsed.appleContinuitySubtype)
        // Lying sub-length → no usable stable payload
        assertNull(parsed.stableManufacturerPayload)
    }

    @Test
    fun `multiple manufacturer TLVs - first wins`() {
        // Defensive: if we ever see two manufacturer TLVs in one packet, the first
        // (presumably the "real" one) wins. A second 0xff TLV must not overwrite.
        val raw = hex("0BFF4C00 10 06 010203040506 0BFF4C00 12 06 AABBCCDDEEFF")
        val parsed = BleAdvertisementParser.parse(raw)
        // First was 0x10 Nearby — kept (with first 2 bytes stable per fix).
        // Second was 0x12 Find My — would override if the parser took last-wins, but
        // we want first-wins so a malicious packet can't overwrite a legitimate one.
        assertEquals(0x10, parsed.appleContinuitySubtype)
    }

    @Test
    fun `txPower TLV is extracted regardless of manufacturer`() {
        val raw = hex(
            "02 01 06" +
            "02 0A F4"     // TX power -12 dBm
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(-12, parsed.txPowerDbm)
    }

    @Test
    fun `service UUID list 16-bit is parsed`() {
        val raw = hex("03 03 0DFE 03 02 12FD")
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.serviceUuids.contains("fe0d"))
        assertTrue(parsed.serviceUuids.contains("fd12"))
    }

    @Test
    fun `service UUID list 16-bit with odd-length body is bounded safely`() {
        // body len = 3 (odd) — should parse 1 UUID (0xFE0D) and ignore the trailing byte
        val raw = hex("04 03 0DFE FF")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(1, parsed.serviceUuids.size)
        assertTrue(parsed.serviceUuids.contains("fe0d"))
    }

    @Test
    fun `service UUID list 32-bit with non-multiple-of-4 body is bounded safely`() {
        // body len = 5 (not a multiple of 4)
        val raw = hex("06 05 78563412 FF")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(1, parsed.serviceUuids.size)
        assertTrue(parsed.serviceUuids.contains("12345678"))
    }

    @Test
    fun `service UUID list 128-bit is parsed`() {
        val raw = hex("11 07 0123456789ABCDEF 0123456789ABCDEF")
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(1, parsed.serviceUuids.size)
        assertTrue(parsed.serviceUuids[0].length == 36)
    }

    @Test
    fun `parser produces lowercase hex stable payload`() {
        // Spot-check: hex serialization is lowercase and zero-padded.
        val raw = hex("09 FF 0600 03 00 0A0B0C0D")
        val parsed = BleAdvertisementParser.parse(raw)
        assertNotNull(parsed.stableManufacturerPayload)
        // The stable bytes are exactly [0x03, 0x00, 0x0A, 0x0B, 0x0C, 0x0D] — Swift Pair body.
        val body = parsed.stableManufacturerPayload!!
        assertEquals(0x03.toByte(), body[0])
    }

    // -------- Newly widened parsers (2026-04 expansion) --------

    @Test
    fun `AltBeacon under canonical Radius Networks company ID is parsed`() {
        // AltBeacon spec: manufacturer payload = [BE AC][UUID 16B][Major 2B][Minor 2B][RefRSSI 1B][MfgRsvd 1B].
        // Company ID 0x0118 is Radius Networks, the AltBeacon-canonical assignment.
        // TLV length: 1 (type) + 2 (company) + 2 (BEAC) + 16 (UUID) + 2 (major) + 2 (minor) + 1 (refrssi) + 1 (mfgrsvd) = 27 bytes.
        // Outer TLV length byte = 1B type + 26B body = 0x1B.
        val raw = hex(
            "1B FF 1801 BEAC 00112233445566778899AABBCCDDEEFF 1234 5678 C5 00"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0118, parsed.manufacturerCompanyId)
        assertEquals(BleAdvertisementParser.SUBTYPE_ALTBEACON, parsed.appleContinuitySubtype)
        assertNotNull(parsed.stableManufacturerPayload)
        // Stable layout: [SUBTYPE_ALTBEACON || UUID(16) || major(2) || minor(2)] = 21 bytes.
        // RefRSSI (calibration) and MfgRsvd are excluded.
        assertEquals(21, parsed.stableManufacturerPayload!!.size)
        assertEquals(BleAdvertisementParser.SUBTYPE_ALTBEACON.toByte(), parsed.stableManufacturerPayload!![0])
        assertEquals(0x00.toByte(), parsed.stableManufacturerPayload!![1]) // first UUID byte
        assertEquals(0xFF.toByte(), parsed.stableManufacturerPayload!![16]) // last UUID byte
        assertEquals(0x12.toByte(), parsed.stableManufacturerPayload!![17]) // major hi
        assertEquals(0x34.toByte(), parsed.stableManufacturerPayload!![18]) // major lo
    }

    @Test
    fun `AltBeacon BEAC marker under non-canonical company ID is detected`() {
        // Some vendors ship AltBeacons under their own assigned company ID with
        // the BEAC marker as the first 2 payload bytes. The fallback path picks
        // those up so resolved-device clustering doesn't lose them.
        val raw = hex(
            "1B FF 5904 BEAC 00112233445566778899AABBCCDDEEFF 1234 5678 C5 00"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0459, parsed.manufacturerCompanyId)
        assertEquals(BleAdvertisementParser.SUBTYPE_ALTBEACON, parsed.appleContinuitySubtype)
        assertNotNull(parsed.stableManufacturerPayload)
        assertEquals(21, parsed.stableManufacturerPayload!!.size)
    }

    @Test
    fun `AltBeacon stable payload survives RefRSSI calibration drift`() {
        // Same beacon, different calibration RefRSSI byte → identical stable payload.
        val a = hex("1B FF 1801 BEAC 00112233445566778899AABBCCDDEEFF 1234 5678 C5 00")
        val b = hex("1B FF 1801 BEAC 00112233445566778899AABBCCDDEEFF 1234 5678 B0 FF")
        val pa = BleAdvertisementParser.parse(a)
        val pb = BleAdvertisementParser.parse(b)
        assertNotNull(pa.stableManufacturerPayload)
        assertNotNull(pb.stableManufacturerPayload)
        assertEquals(
            BleAdvertisementParser.bytesToHex(pa.stableManufacturerPayload!!),
            BleAdvertisementParser.bytesToHex(pb.stableManufacturerPayload!!)
        )
    }

    @Test
    fun `AltBeacon malformed - missing BEAC marker is rejected`() {
        // Radius Networks company ID but body starts with garbage rather than 0xBEAC.
        val raw = hex(
            "1B FF 1801 1234 00112233445566778899AABBCCDDEEFF 1234 5678 C5 00"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertEquals(0x0118, parsed.manufacturerCompanyId)
        // Without BEAC marker we have nothing trustworthy — fail closed.
        assertNull(parsed.stableManufacturerPayload)
    }

    @Test
    fun `Tile FEED service data extracts 16-byte tracker UID into stable payload`() {
        // Tile broadcasts service data on UUID 0xFEED. The first 16 bytes are
        // the static factory-assigned tracker UID; trailing bytes carry rotating
        // button-press / battery state which must NOT contribute to clustering.
        // Service-data TLV layout: [LEN][TYPE=0x16][UUID-LE=EDFE][Tracker UID 16B][State...]
        val raw = hex(
            "13 16 EDFE 00112233445566778899AABBCCDDEEFF 0102"
        )
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.serviceUuids.contains("feed"))
        assertEquals(BleAdvertisementParser.SUBTYPE_TILE, parsed.appleContinuitySubtype)
        assertNotNull(parsed.stableManufacturerPayload)
        // Stable payload = [SUBTYPE_TILE || tracker_uid_16B] = 17 bytes.
        // Trailing state bytes (button press / battery) are dropped.
        assertEquals(17, parsed.stableManufacturerPayload!!.size)
        assertEquals(BleAdvertisementParser.SUBTYPE_TILE.toByte(), parsed.stableManufacturerPayload!![0])
        assertEquals(0x00.toByte(), parsed.stableManufacturerPayload!![1])
        assertEquals(0xFF.toByte(), parsed.stableManufacturerPayload!![16])
    }

    @Test
    fun `Tile FEED stable payload ignores trailing state byte rotation`() {
        // Same tracker, different button/battery state bytes → same stable payload.
        val a = hex("13 16 EDFE 00112233445566778899AABBCCDDEEFF 0102")
        val b = hex("13 16 EDFE 00112233445566778899AABBCCDDEEFF FFEE")
        val pa = BleAdvertisementParser.parse(a)
        val pb = BleAdvertisementParser.parse(b)
        assertNotNull(pa.stableManufacturerPayload)
        assertNotNull(pb.stableManufacturerPayload)
        assertEquals(
            BleAdvertisementParser.bytesToHex(pa.stableManufacturerPayload!!),
            BleAdvertisementParser.bytesToHex(pb.stableManufacturerPayload!!)
        )
    }

    @Test
    fun `Tile FEED truncated under 16 tracker bytes is rejected`() {
        // Body too short to contain a complete tracker UID — fail closed.
        val raw = hex("0A 16 EDFE 0011223344556677")
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.serviceUuids.contains("feed"))
        assertNull("Truncated Tile UID must not produce a partial stable payload", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Eddystone-EID rotating frame produces no stable payload`() {
        // EID frames rotate every 8s-1024s via AES-CTR. The parser already
        // refuses to surface eddystoneIdHex — confirm it also refuses to write
        // stable_manufacturer_payload bytes (the field hashed server-side).
        val raw = hex("0D 16 AAFE 30 EE 0102030405060708")
        val parsed = BleAdvertisementParser.parse(raw)
        assertTrue(parsed.isEddystone)
        assertEquals(BleAdvertisementParser.EDDYSTONE_FRAME_EID, parsed.eddystoneFrameType)
        assertNull("EID is rotating — must NOT contribute stable bytes", parsed.stableManufacturerPayload)
    }

    @Test
    fun `Subtype constants are in the synthetic 0x80-plus reserved range`() {
        // Apple Continuity subtypes are 0x00..0x1F. Synthetic markers MUST live
        // at 0x80+ to keep the wire-level demux unambiguous (server reads
        // apple_continuity_subtype + maps to manufacturer_subtype). Lock that
        // invariant via a unit test so a future patch can't silently collide.
        assertTrue(BleAdvertisementParser.SUBTYPE_IBEACON >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_EDDYSTONE_UID >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_FAST_PAIR >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_TILE >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_AIRTAG >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_EDDYSTONE_URL >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_ALTBEACON >= 0x80)
        assertTrue(BleAdvertisementParser.SUBTYPE_MS_CDP >= 0x80)
        // All distinct.
        val codes = setOf(
            BleAdvertisementParser.SUBTYPE_IBEACON,
            BleAdvertisementParser.SUBTYPE_EDDYSTONE_UID,
            BleAdvertisementParser.SUBTYPE_FAST_PAIR,
            BleAdvertisementParser.SUBTYPE_TILE,
            BleAdvertisementParser.SUBTYPE_AIRTAG,
            BleAdvertisementParser.SUBTYPE_EDDYSTONE_URL,
            BleAdvertisementParser.SUBTYPE_ALTBEACON,
            BleAdvertisementParser.SUBTYPE_MS_CDP
        )
        assertEquals(8, codes.size)
    }
}
