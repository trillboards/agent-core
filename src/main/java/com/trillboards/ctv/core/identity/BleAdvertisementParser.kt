package com.trillboards.ctv.core.identity

/**
 * Parsed BLE advertisement scan record.
 *
 * `stableManufacturerPayload` holds ONLY sub-bytes that survive RPA (Resolvable
 * Private Address) MAC rotation. Apple Continuity sub-types known to carry
 * rotating Auth tags (AirDrop 0x05, Hey Siri 0x08, AirPlay Source 0x09,
 * Magic Switch 0x0a, Watch Connectivity 0x0b, Handoff 0x0c, Wifi Settings 0x0d,
 * Instant Hotspot 0x0e, Wifi Join 0x0f, Watch Bridge 0x11, Find My 0x12,
 * Tethering Source 0x14, Proximity Pairing 0x16) are explicitly dropped —
 * `stableManufacturerPayload` is null even though `appleContinuitySubtype`
 * is populated. Unknown sub-types fail CLOSED (drop payload) since Apple
 * routinely ships new rotating types and silent classification as stable
 * would corrupt resolved-device clusters.
 *
 * `appleContinuitySubtype` is a generalized "advert subtype" code: for Apple
 * Continuity it is the literal 1-byte Apple subtype (0x00..0x1F). For non-Apple
 * advert formats the parser uses synthetic markers in the 0x80+ range so the
 * server can demux on a single field without breaking the wire contract
 * (`apple_continuity_subtype` JSON key persists; only its value space widens).
 *
 * Spec references:
 *   - iBeacon Apple Tech Note QA1937
 *   - Apple Continuity reverse-engineered spec: furiousmac.com/handoff
 *   - Microsoft Swift Pair Beacon ID 0x03: docs.microsoft.com/windows-hardware
 *   - Google Fast Pair v1 Service Data 0xFE2C, 24-bit BE Model ID
 *   - Eddystone GitHub spec: google/eddystone (frame types 0x00 UID, 0x10 URL,
 *     0x20 TLM, 0x30 EID — only UID/URL are stable identifiers)
 *   - AltBeacon GitHub spec: AltBeacon/spec — manufacturer payload prefix 0xBEAC
 *   - Tile FEED 0xFEED service data — first 16 bytes are the static tracker UID
 */
data class ParsedBleAdvertisement(
    val manufacturerCompanyId: Int? = null,
    val appleContinuitySubtype: Int? = null,
    val stableManufacturerPayload: ByteArray? = null,
    val serviceUuids: List<String> = emptyList(),
    val txPowerDbm: Int? = null,
    val isIbeacon: Boolean = false,
    val ibeaconUuid: String? = null,
    val ibeaconMajor: Int? = null,
    val ibeaconMinor: Int? = null,
    val isEddystone: Boolean = false,
    val eddystoneFrameType: Int? = null,
    val eddystoneIdHex: String? = null,
    val googleFastPairModelId: Int? = null
)

/**
 * Pure-domain TLV parser for BLE advertisement scan records.
 *
 * JVM-testable, no Android deps. Designed to NEVER throw on adversarial input —
 * every byte read is bounds-checked. Hot path: ~33K calls/sec across the fleet,
 * so allocations are minimized (lookup-table hex, single mutable list, no
 * String.format invocations).
 */
object BleAdvertisementParser {

    // Manufacturer company IDs (Bluetooth SIG assigned numbers).
    private const val APPLE = 0x004c
    private const val MICROSOFT = 0x0006
    private const val GOOGLE = 0x00e0
    private const val RADIUS_NETWORKS = 0x0118 // AltBeacon canonical company ID

    // iBeacon constants (Apple Tech Note QA1937).
    private const val APPLE_SUBTYPE_IBEACON = 0x02
    private const val IBEACON_INNER_LENGTH = 0x15

    // Apple Continuity sub-types whose payloads carry rotating Auth tags or
    // ephemeral identifiers and MUST be dropped from stable clustering.
    // Source: https://furiousmac.com/handoff (reverse-engineered Continuity spec).
    private val APPLE_ROTATING_SUBTYPES = setOf(
        0x05, // AirDrop — rotating identifier
        0x08, // Hey Siri — rotating session-id / RNG nonce
        0x09, // AirPlay Source — rotating Apple ID hash
        0x0a, // Magic Switch (Tethering Target Presence) — rotating Auth tag
        0x0b, // Watch Connectivity — rotating session bytes
        0x0c, // Handoff — encrypted rotating Auth tag
        0x0d, // Wifi Settings — rotating Auth tag
        0x0e, // Instant Hotspot — rotating Auth tag
        0x0f, // Wifi Join Network — rotating Auth tag
        0x11, // Watch Bridge — rotating session token
        0x12, // Find My / Offline Finding — rotates every 15 min via secp224r1
        0x14, // Tethering Source — rotating session bytes
        0x16  // Proximity Pairing — rotating session token
    )

    // Apple Continuity sub-types where ONLY the leading prefix bytes are stable.
    // The remaining bytes rotate (state flags, action codes, AuthTags, etc.).
    private const val APPLE_SUBTYPE_AIRPODS = 0x07
    private const val APPLE_AIRPODS_STABLE_PREFIX_BYTES = 9
    private const val APPLE_SUBTYPE_NEARBY = 0x10
    private const val APPLE_NEARBY_STABLE_PREFIX_BYTES = 2 // StatusFlags + ActionCode only

    // Microsoft 0x0006 advertisement types — only Swift Pair (0x03) is stable.
    // Other beacon IDs (CDP, MDDM, Cross-Device Promotion) carry rotating data.
    private const val MICROSOFT_BEACON_SWIFT_PAIR = 0x03

    // Eddystone frame type identifiers.
    const val EDDYSTONE_FRAME_UID = 0x00
    const val EDDYSTONE_FRAME_URL = 0x10
    const val EDDYSTONE_FRAME_TLM = 0x20
    const val EDDYSTONE_FRAME_EID = 0x30
    private const val EDDYSTONE_UUID_HEX = "feaa"
    private const val FAST_PAIR_UUID_HEX = "fe2c"
    private const val TILE_UUID_HEX = "feed"
    private const val EDDYSTONE_UID_NAMESPACE_BYTES = 10
    private const val EDDYSTONE_UID_INSTANCE_BYTES = 6
    private const val TILE_TRACKER_UID_BYTES = 16

    // AltBeacon spec (https://github.com/AltBeacon/spec): manufacturer-specific
    // payload begins with the 16-bit "beacon code" 0xBEAC then 16-byte UUID +
    // 2-byte Major + 2-byte Minor + 1-byte RefRSSI + 1-byte MfgRsvd.
    private const val ALTBEACON_CODE_HI = 0xBE
    private const val ALTBEACON_CODE_LO = 0xAC

    // Synthetic advert-type subtype codes for non-Apple formats. Stored in the
    // `appleContinuitySubtype` field (which the wire serializes as
    // `apple_continuity_subtype` and the server consumes as `manufacturer_subtype`).
    // Apple Continuity uses 0x00..0x1F so 0x80+ is collision-free.
    const val SUBTYPE_IBEACON: Int = 0x80
    const val SUBTYPE_EDDYSTONE_UID: Int = 0x81
    const val SUBTYPE_FAST_PAIR: Int = 0x82
    const val SUBTYPE_TILE: Int = 0x83
    const val SUBTYPE_AIRTAG: Int = 0x84 // Reserved — Find My 0x12 rotates fully.
    const val SUBTYPE_EDDYSTONE_URL: Int = 0x85
    const val SUBTYPE_ALTBEACON: Int = 0x86
    const val SUBTYPE_MS_CDP: Int = 0x87 // Reserved — Microsoft non-SwiftPair CDP rotates.

    // BLE AD type identifiers (Core Specification Supplement Part A).
    private const val AD_TYPE_TX_POWER = 0x0a
    private const val AD_TYPE_INCOMPLETE_LIST_16_UUID = 0x02
    private const val AD_TYPE_COMPLETE_LIST_16_UUID = 0x03
    private const val AD_TYPE_INCOMPLETE_LIST_32_UUID = 0x04
    private const val AD_TYPE_COMPLETE_LIST_32_UUID = 0x05
    private const val AD_TYPE_INCOMPLETE_LIST_128_UUID = 0x06
    private const val AD_TYPE_COMPLETE_LIST_128_UUID = 0x07
    private const val AD_TYPE_SERVICE_DATA_16_UUID = 0x16
    private const val AD_TYPE_MANUFACTURER_SPECIFIC = 0xff

    // Lookup table for fast hex serialization (avoids String.format + Locale lookup
    // on the hot path — measured ~3x faster than "%02x".format).
    private val HEX_DIGITS = charArrayOf(
        '0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'
    )

    fun parse(b: ByteArray): ParsedBleAdvertisement {
        if (b.isEmpty()) return ParsedBleAdvertisement()

        var manufacturerCompanyId: Int? = null
        var appleContinuitySubtype: Int? = null
        var stableManufacturerPayload: ByteArray? = null
        val serviceUuids = ArrayList<String>(4)
        var txPowerDbm: Int? = null
        var isIbeacon = false
        var ibeaconUuid: String? = null
        var ibeaconMajor: Int? = null
        var ibeaconMinor: Int? = null
        var isEddystone = false
        var eddystoneFrameType: Int? = null
        var eddystoneIdHex: String? = null
        var googleFastPairModelId: Int? = null

        var i = 0
        while (i < b.size) {
            val tlvLength = b[i].toInt() and 0xff
            // Zero-length sentinel — Bluetooth spec allows AD list trailing pad.
            if (tlvLength == 0) { i++; continue }
            // TLV occupies bytes [i, i+tlvLength]; last data byte index = i+tlvLength.
            // If that's past the end, the TLV is truncated → stop parsing entirely.
            if (i + tlvLength >= b.size) break

            val adType = b[i + 1].toInt() and 0xff
            val bodyOffset = i + 2
            val bodyLength = tlvLength - 1
            // bodyEnd is exclusive; verified by the i+tlvLength check above.

            when (adType) {
                AD_TYPE_TX_POWER -> {
                    if (bodyLength >= 1) txPowerDbm = b[bodyOffset].toInt() // signed dBm
                }
                AD_TYPE_INCOMPLETE_LIST_16_UUID, AD_TYPE_COMPLETE_LIST_16_UUID -> {
                    parseUuid16List(b, bodyOffset, bodyLength, serviceUuids)
                }
                AD_TYPE_INCOMPLETE_LIST_32_UUID, AD_TYPE_COMPLETE_LIST_32_UUID -> {
                    parseUuid32List(b, bodyOffset, bodyLength, serviceUuids)
                }
                AD_TYPE_INCOMPLETE_LIST_128_UUID, AD_TYPE_COMPLETE_LIST_128_UUID -> {
                    parseUuid128List(b, bodyOffset, bodyLength, serviceUuids)
                }
                AD_TYPE_MANUFACTURER_SPECIFIC -> {
                    // Body must contain a 2-byte little-endian company ID.
                    if (bodyLength >= 2 && manufacturerCompanyId == null) {
                        // FIRST manufacturer TLV wins — defensive against duplicate
                        // 0xff TLVs that could overwrite legitimate identifiers.
                        val companyId = ((b[bodyOffset + 1].toInt() and 0xff) shl 8) or
                            (b[bodyOffset].toInt() and 0xff)
                        manufacturerCompanyId = companyId

                        when (companyId) {
                            APPLE -> {
                                val apple = parseApple(b, bodyOffset + 2, bodyLength - 2)
                                appleContinuitySubtype = apple.subtype
                                stableManufacturerPayload = apple.stable
                                if (apple.isIbeacon) {
                                    isIbeacon = true
                                    ibeaconUuid = apple.ibeaconUuid
                                    ibeaconMajor = apple.ibeaconMajor
                                    ibeaconMinor = apple.ibeaconMinor
                                    if (apple.ibeaconTxPower != null) {
                                        txPowerDbm = apple.ibeaconTxPower
                                    }
                                }
                            }
                            MICROSOFT -> {
                                stableManufacturerPayload = parseMicrosoft(
                                    b, bodyOffset + 2, bodyLength - 2
                                )
                            }
                            RADIUS_NETWORKS -> {
                                // AltBeacon (canonical company ID). Other vendors
                                // sometimes overload other company IDs with the
                                // 0xBEAC marker; those are matched as a fallback
                                // below.
                                val alt = parseAltBeacon(b, bodyOffset + 2, bodyLength - 2)
                                if (alt != null) {
                                    appleContinuitySubtype = SUBTYPE_ALTBEACON
                                    stableManufacturerPayload = alt
                                }
                            }
                            // GOOGLE (0x00e0) manufacturer-specific is NOT used for
                            // Fast Pair — Fast Pair uses Service Data UUID FE2C.
                            // Other Google 0x00e0 ads are heterogeneous; reject as
                            // untrusted to avoid clustering on rotating bytes.
                        }
                        // Fallback: AltBeacon is occasionally shipped under a
                        // non-Radius company ID with the 0xBEAC marker as the
                        // first 2 bytes of the payload. Detect that here so the
                        // beacon is still clustered correctly. Only triggers
                        // when no stable payload was already extracted above
                        // (i.e. unknown company ID + BEAC prefix).
                        if (
                            stableManufacturerPayload == null &&
                            companyId != APPLE && companyId != MICROSOFT &&
                            companyId != RADIUS_NETWORKS &&
                            bodyLength >= 4 &&
                            (b[bodyOffset + 2].toInt() and 0xff) == ALTBEACON_CODE_HI &&
                            (b[bodyOffset + 3].toInt() and 0xff) == ALTBEACON_CODE_LO
                        ) {
                            val alt = parseAltBeacon(b, bodyOffset + 2, bodyLength - 2)
                            if (alt != null) {
                                appleContinuitySubtype = SUBTYPE_ALTBEACON
                                stableManufacturerPayload = alt
                            }
                        }
                    }
                }
                AD_TYPE_SERVICE_DATA_16_UUID -> {
                    if (bodyLength >= 2) {
                        val uuid16 = formatUuid16Le(b, bodyOffset)
                        addUnique(serviceUuids, uuid16)
                        when (uuid16) {
                            EDDYSTONE_UUID_HEX -> {
                                isEddystone = true
                                val eddy = parseEddystone(b, bodyOffset + 2, bodyLength - 2)
                                eddystoneFrameType = eddy.frameType
                                eddystoneIdHex = eddy.idHex
                                if (eddy.txPower != null) txPowerDbm = eddy.txPower
                                // Promote stable Eddystone identifiers (UID/URL)
                                // into the cross-format stable payload + subtype
                                // so the server can cluster them alongside Apple
                                // Continuity / iBeacon / AltBeacon.
                                if (eddy.stable != null && stableManufacturerPayload == null) {
                                    stableManufacturerPayload = eddy.stable
                                    if (appleContinuitySubtype == null) {
                                        appleContinuitySubtype = eddy.subtypeCode
                                    }
                                }
                            }
                            FAST_PAIR_UUID_HEX -> {
                                // Google Fast Pair v1: 24-bit BIG-ENDIAN model ID.
                                if (bodyLength >= 5 && manufacturerCompanyId == null) {
                                    googleFastPairModelId =
                                        ((b[bodyOffset + 2].toInt() and 0xff) shl 16) or
                                        ((b[bodyOffset + 3].toInt() and 0xff) shl 8) or
                                        (b[bodyOffset + 4].toInt() and 0xff)
                                    manufacturerCompanyId = GOOGLE
                                    // Stable payload = [SUBTYPE_FAST_PAIR || model_id_3B].
                                    // Prefixing the synthetic subtype byte makes the
                                    // hash space disjoint from raw Apple Continuity
                                    // payloads and from other format families.
                                    val fp = ByteArray(4)
                                    fp[0] = SUBTYPE_FAST_PAIR.toByte()
                                    fp[1] = b[bodyOffset + 2]
                                    fp[2] = b[bodyOffset + 3]
                                    fp[3] = b[bodyOffset + 4]
                                    stableManufacturerPayload = fp
                                    appleContinuitySubtype = SUBTYPE_FAST_PAIR
                                }
                            }
                            TILE_UUID_HEX -> {
                                // Tile FEED service data: first 16 bytes after the
                                // UUID are the static, factory-assigned tracker UID.
                                // Trailing bytes carry rotating button-press / battery
                                // state — drop those.
                                if (bodyLength >= 2 + TILE_TRACKER_UID_BYTES &&
                                    stableManufacturerPayload == null
                                ) {
                                    val trackerStart = bodyOffset + 2
                                    val tile = ByteArray(1 + TILE_TRACKER_UID_BYTES)
                                    tile[0] = SUBTYPE_TILE.toByte()
                                    for (k in 0 until TILE_TRACKER_UID_BYTES) {
                                        tile[1 + k] = b[trackerStart + k]
                                    }
                                    stableManufacturerPayload = tile
                                    if (appleContinuitySubtype == null) {
                                        appleContinuitySubtype = SUBTYPE_TILE
                                    }
                                }
                            }
                        }
                    }
                }
            }
            i += tlvLength + 1
        }

        return ParsedBleAdvertisement(
            manufacturerCompanyId = manufacturerCompanyId,
            appleContinuitySubtype = appleContinuitySubtype,
            stableManufacturerPayload = stableManufacturerPayload,
            serviceUuids = if (serviceUuids.isEmpty()) emptyList() else serviceUuids,
            txPowerDbm = txPowerDbm,
            isIbeacon = isIbeacon,
            ibeaconUuid = ibeaconUuid,
            ibeaconMajor = ibeaconMajor,
            ibeaconMinor = ibeaconMinor,
            isEddystone = isEddystone,
            eddystoneFrameType = eddystoneFrameType,
            eddystoneIdHex = eddystoneIdHex,
            googleFastPairModelId = googleFastPairModelId
        )
    }

    // -------- Apple --------

    private data class AppleParse(
        val subtype: Int? = null,
        val stable: ByteArray? = null,
        val isIbeacon: Boolean = false,
        val ibeaconUuid: String? = null,
        val ibeaconMajor: Int? = null,
        val ibeaconMinor: Int? = null,
        val ibeaconTxPower: Int? = null
    )

    private fun parseApple(b: ByteArray, off: Int, len: Int): AppleParse {
        if (len < 1) return AppleParse()
        val subtype = b[off].toInt() and 0xff

        // iBeacon: subtype 0x02 + inner length 0x15. Layout (after 0x02 0x15):
        //   16B UUID + 2B major + 2B minor + 1B signed TxPower = 21 bytes.
        if (subtype == APPLE_SUBTYPE_IBEACON) {
            // Inner length must be exactly 0x15 AND there must be 21 bytes of payload.
            if (len >= 23 && (b[off + 1].toInt() and 0xff) == IBEACON_INNER_LENGTH) {
                val uuid = formatUuid128Be(b, off + 2)
                val major = ((b[off + 18].toInt() and 0xff) shl 8) or (b[off + 19].toInt() and 0xff)
                val minor = ((b[off + 20].toInt() and 0xff) shl 8) or (b[off + 21].toInt() and 0xff)
                val txPower = b[off + 22].toInt()
                // Stable payload = [SUBTYPE_IBEACON || UUID(16) || major(2) || minor(2)].
                // TxPower is excluded — it tracks calibration state, not identity.
                val stable = ByteArray(1 + 16 + 2 + 2)
                stable[0] = SUBTYPE_IBEACON.toByte()
                for (k in 0 until 16) stable[1 + k] = b[off + 2 + k]
                stable[17] = b[off + 18]
                stable[18] = b[off + 19]
                stable[19] = b[off + 20]
                stable[20] = b[off + 21]
                return AppleParse(
                    subtype = SUBTYPE_IBEACON,
                    stable = stable,
                    isIbeacon = true,
                    ibeaconUuid = uuid,
                    ibeaconMajor = major,
                    ibeaconMinor = minor,
                    ibeaconTxPower = txPower
                )
            }
            // Subtype 0x02 with non-iBeacon length is ambiguous — reject explicitly.
            // Returning the raw subtype as a phantom Continuity field would corrupt
            // resolved-device clusters with bogus 0x02 entries.
            return AppleParse(subtype = APPLE_SUBTYPE_IBEACON, stable = null)
        }

        // Generic Continuity TLV: [subtype][sublen][payload...].
        if (len < 2) return AppleParse(subtype = subtype, stable = null)
        val subLen = b[off + 1].toInt() and 0xff
        // Sub-length must fit within the remaining manufacturer body.
        if (subLen <= 0 || 2 + subLen > len) {
            return AppleParse(subtype = subtype, stable = null)
        }
        val payloadStart = off + 2
        // payloadEnd would be off + 2 + subLen — currently unused since both
        // documented stable subtypes (AirPods 0x07, Nearby 0x10) keep prefix
        // bytes and unknown subtypes fail closed. If a future subtype needs
        // the FULL body, reintroduce `val payloadEnd = off + 2 + subLen`.

        // Drop rotating sub-types entirely.
        if (subtype in APPLE_ROTATING_SUBTYPES) {
            return AppleParse(subtype = subtype, stable = null)
        }

        val stable: ByteArray? = when (subtype) {
            APPLE_SUBTYPE_AIRPODS -> {
                // Keep only the model-id + state prefix; trailing bytes rotate.
                val keep = minOf(APPLE_AIRPODS_STABLE_PREFIX_BYTES, subLen)
                if (keep > 0) b.copyOfRange(payloadStart, payloadStart + keep) else null
            }
            APPLE_SUBTYPE_NEARBY -> {
                // Keep only StatusFlags + ActionCode; AuthTag and tail rotate.
                val keep = minOf(APPLE_NEARBY_STABLE_PREFIX_BYTES, subLen)
                if (keep > 0) b.copyOfRange(payloadStart, payloadStart + keep) else null
            }
            else -> {
                // Unknown sub-types fail CLOSED. Apple routinely ships new
                // sub-types (0x18 Bluetooth Restoration, 0x19 PreBoot, 0x1B
                // Nearby Info, etc.) that often carry rotating data. Returning
                // the body would silently classify them as stable until manually
                // added to APPLE_ROTATING_SUBTYPES — privacy-unsafe default.
                // The subtype itself IS reported (so dashboards can flag new
                // unknown subtypes for review) but no payload bytes contribute
                // to clustering.
                null
            }
        }
        return AppleParse(subtype = subtype, stable = stable)
    }

    // -------- Microsoft --------

    private fun parseMicrosoft(b: ByteArray, off: Int, len: Int): ByteArray? {
        if (len < 1) return null
        val beaconId = b[off].toInt() and 0xff
        // Only Swift Pair (0x03) carries stable model bytes. Other beacon types
        // (CDP MDDM, Cross-Device Promotion) include rotating session bytes.
        if (beaconId != MICROSOFT_BEACON_SWIFT_PAIR) return null
        return b.copyOfRange(off, off + len)
    }

    // -------- AltBeacon --------

    /**
     * Parse an AltBeacon manufacturer payload. The wire layout is
     * `[BE AC][UUID 16B][Major 2B][Minor 2B][RefRSSI 1B][MfgRsvd 1B]` per the
     * AltBeacon spec. RefRSSI is calibration state and rotates; MfgRsvd is
     * vendor-specific and may rotate. Stable identity = `[SUBTYPE_ALTBEACON ||
     * UUID || major || minor]`.
     *
     * Returns null when the body does not begin with the BEAC marker or is
     * truncated.
     */
    private fun parseAltBeacon(b: ByteArray, off: Int, len: Int): ByteArray? {
        // Need: 2B beacon code + 16B UUID + 2B major + 2B minor = 22 bytes.
        if (len < 22) return null
        if ((b[off].toInt() and 0xff) != ALTBEACON_CODE_HI) return null
        if ((b[off + 1].toInt() and 0xff) != ALTBEACON_CODE_LO) return null
        val out = ByteArray(1 + 16 + 2 + 2)
        out[0] = SUBTYPE_ALTBEACON.toByte()
        for (k in 0 until 16) out[1 + k] = b[off + 2 + k]
        out[17] = b[off + 18]
        out[18] = b[off + 19]
        out[19] = b[off + 20]
        out[20] = b[off + 21]
        return out
    }

    // -------- Eddystone --------

    private data class EddyParse(
        val frameType: Int?,
        val idHex: String?,
        val txPower: Int?,
        val stable: ByteArray? = null,
        val subtypeCode: Int? = null
    )

    private fun parseEddystone(b: ByteArray, off: Int, len: Int): EddyParse {
        if (len < 1) return EddyParse(frameType = null, idHex = null, txPower = null)
        val frameType = b[off].toInt() and 0xff
        return when (frameType) {
            EDDYSTONE_FRAME_UID -> {
                // Layout: [frameType=0x00][TxPower 1B signed][Namespace 10B][Instance 6B][RFU 2B]
                val txPower = if (len >= 2) b[off + 1].toInt() else null
                val idStart = off + 2
                val idLen = EDDYSTONE_UID_NAMESPACE_BYTES + EDDYSTONE_UID_INSTANCE_BYTES
                if (len >= 2 + idLen) {
                    val idHex = bytesToHex(b, idStart, idLen)
                    // Stable payload = [SUBTYPE_EDDYSTONE_UID || namespace(10) || instance(6)].
                    // TxPower excluded — calibration data.
                    val stable = ByteArray(1 + idLen)
                    stable[0] = SUBTYPE_EDDYSTONE_UID.toByte()
                    for (k in 0 until idLen) stable[1 + k] = b[idStart + k]
                    EddyParse(
                        frameType = frameType,
                        idHex = idHex,
                        txPower = txPower,
                        stable = stable,
                        subtypeCode = SUBTYPE_EDDYSTONE_UID
                    )
                } else {
                    EddyParse(frameType = frameType, idHex = null, txPower = txPower)
                }
            }
            EDDYSTONE_FRAME_URL -> {
                // Layout: [frameType=0x10][TxPower 1B][URL Scheme 1B][Encoded URL...].
                // The scheme byte + encoded URL bytes are stable per advertiser
                // (advertisers don't rotate the URL they broadcast). We capture
                // them verbatim so the server can hash them into a stable
                // identifier; the on-device hot path does NOT decode the URL
                // string (no allocations beyond the byte copy).
                val txPower = if (len >= 2) b[off + 1].toInt() else null
                if (len >= 3) {
                    val urlStart = off + 2
                    val urlLen = len - 2
                    val stable = ByteArray(1 + urlLen)
                    stable[0] = SUBTYPE_EDDYSTONE_URL.toByte()
                    for (k in 0 until urlLen) stable[1 + k] = b[urlStart + k]
                    EddyParse(
                        frameType = frameType,
                        idHex = null,
                        txPower = txPower,
                        stable = stable,
                        subtypeCode = SUBTYPE_EDDYSTONE_URL
                    )
                } else {
                    EddyParse(frameType = frameType, idHex = null, txPower = txPower)
                }
            }
            EDDYSTONE_FRAME_TLM, EDDYSTONE_FRAME_EID -> {
                // TLM = volatile telemetry (battery, temp, count). EID = rotating
                // ephemeral identifier (8s-1024s rotation via AES-CTR). Both must
                // NOT contribute identifier bytes to clustering.
                EddyParse(frameType = frameType, idHex = null, txPower = null)
            }
            else -> EddyParse(frameType = frameType, idHex = null, txPower = null)
        }
    }

    // -------- UUID list parsers --------

    /** Append `s` to `out` if absent. Linear-scan dedupe is fine here: in practice
     *  each AD record holds <=4 service UUIDs, so a HashSet would cost more in
     *  allocations than the scan saves. */
    private fun addUnique(out: MutableList<String>, s: String) {
        // Reverse iteration: most duplicates appear adjacent (same service split
        // across Incomplete + Complete UUID list AD types).
        for (j in out.size - 1 downTo 0) if (out[j] == s) return
        out.add(s)
    }

    /** Parse 16-bit Service UUIDs (LE on the wire). Bounded by `len`, ignores trailing
     *  bytes that don't form a complete UUID. */
    private fun parseUuid16List(b: ByteArray, off: Int, len: Int, out: MutableList<String>) {
        var p = 0
        while (p + 2 <= len) {
            addUnique(out, formatUuid16Le(b, off + p))
            p += 2
        }
    }

    /** Parse 32-bit Service UUIDs (LE on the wire). */
    private fun parseUuid32List(b: ByteArray, off: Int, len: Int, out: MutableList<String>) {
        var p = 0
        while (p + 4 <= len) {
            val sb = StringBuilder(8)
            // Wire is little-endian; canonical form is big-endian → reverse.
            for (j in 3 downTo 0) appendHexByte(sb, b[off + p + j].toInt())
            addUnique(out, sb.toString())
            p += 4
        }
    }

    /** Parse 128-bit Service UUIDs (LE on the wire). Each UUID is 16 bytes,
     *  formatted as canonical 8-4-4-4-12 dashed lowercase. */
    private fun parseUuid128List(b: ByteArray, off: Int, len: Int, out: MutableList<String>) {
        var p = 0
        while (p + 16 <= len) {
            addUnique(out, formatUuid128Le(b, off + p))
            p += 16
        }
    }

    // -------- Hex helpers (allocation-conscious) --------

    /** 16-bit UUID at `off` interpreted as LITTLE-ENDIAN, returned as 4 lowercase hex. */
    private fun formatUuid16Le(b: ByteArray, off: Int): String {
        val sb = StringBuilder(4)
        appendHexByte(sb, b[off + 1].toInt())
        appendHexByte(sb, b[off].toInt())
        return sb.toString()
    }

    /** 16-byte UUID at `off` interpreted as BIG-ENDIAN (iBeacon canonical),
     *  formatted 8-4-4-4-12. */
    private fun formatUuid128Be(b: ByteArray, off: Int): String {
        val sb = StringBuilder(36)
        for (j in 0 until 16) {
            appendHexByte(sb, b[off + j].toInt())
            if (j == 3 || j == 5 || j == 7 || j == 9) sb.append('-')
        }
        return sb.toString()
    }

    /** 16-byte UUID at `off` interpreted as LITTLE-ENDIAN (BLE service-UUID list
     *  canonical), formatted 8-4-4-4-12. */
    private fun formatUuid128Le(b: ByteArray, off: Int): String {
        val sb = StringBuilder(36)
        for (j in 15 downTo 0) {
            appendHexByte(sb, b[off + j].toInt())
            val pos = 15 - j
            if (pos == 3 || pos == 5 || pos == 7 || pos == 9) sb.append('-')
        }
        return sb.toString()
    }

    /** Render `len` bytes of `b` from `off` as a lowercase hex string. */
    fun bytesToHex(b: ByteArray, off: Int, len: Int): String {
        val sb = StringBuilder(len * 2)
        for (j in 0 until len) appendHexByte(sb, b[off + j].toInt())
        return sb.toString()
    }

    /** Render an entire ByteArray as a lowercase hex string. Convenience wrapper
     *  around [bytesToHex] used by hot-path callers (e.g. BleBeaconScanner) so
     *  they don't fall back to slow `"%02x".format(b)`. */
    fun bytesToHex(b: ByteArray): String = bytesToHex(b, 0, b.size)

    /** Append 2-char lowercase hex of the low 8 bits of `v` to `sb`.
     *  Lookup-table version — measured ~3x faster than `"%02x".format(v)` due
     *  to no Locale lookup, no Formatter allocation, no varargs boxing. */
    internal fun appendHexByte(sb: StringBuilder, v: Int) {
        sb.append(HEX_DIGITS[(v ushr 4) and 0xf])
        sb.append(HEX_DIGITS[v and 0xf])
    }
}
