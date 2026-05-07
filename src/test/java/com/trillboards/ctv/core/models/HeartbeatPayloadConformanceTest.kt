package com.trillboards.ctv.core.models

// Conformance scope:
//   Round-trips: heartbeat-minimal, heartbeat-typical, heartbeat-get-response
//   Truncation: heartbeat-truncation (51 BLE devices in → 50 strongest out)
//
//   Intentionally ignores (not emitted by this SDK on this platform):
//     - audio_diarization (separate emit channel — not in heartbeat body)
//     - vertex_embedding (server-side only)
//     - Phase-6 stub fields (uwb_peers / auracast_broadcasts /
//       channel_sounding_measurements) — excluded from OpenAPI by design
//     - MDM enrollment / location, CSI, native sensors — excluded from
//       OpenAPI per audit (0% production population)

import com.trillboards.api.types.BleScanResultData
import com.trillboards.api.types.PartnerDeviceHeartbeatBody
import com.trillboards.api.types.PartnerHeartbeatQueryResponse
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.encodeToString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Conformance tests for the partner-facing wire shape of agent-core's
 * heartbeat payload.
 *
 * These verify that the generated OpenAPI DTOs at
 * `com.trillboards.api.types.*` round-trip the canonical fixtures without
 * loss, and that [HeartbeatPayloadFactory.MAX_NEARBY_DEVICES_PER_LIST] is
 * enforced when raw input lists exceed the cap.
 *
 * Fixtures live at `trillboard-api/docs/openapi/fixtures/heartbeat-*.json`
 * and are sourced from real production samples (anonymized). They are the
 * canonical contract used by every per-SDK conformance test in PR 4.
 */
class HeartbeatPayloadConformanceTest {

    /**
     * Locate the fixtures directory by walking up from the working
     * directory until we find `trillboard-api/docs/openapi/fixtures`.
     * Gradle runs tests with cwd = `trillboard-ctv/tablet-agent` so the
     * relative path is `../../trillboard-api/docs/openapi/fixtures`, but
     * the walk-up makes the test resilient to where the runner picks cwd.
     */
    private fun fixturesDir(): File {
        var dir: File? = File("").absoluteFile
        while (dir != null) {
            val candidate = File(dir, "trillboard-api/docs/openapi/fixtures")
            if (candidate.isDirectory) return candidate
            dir = dir.parentFile
        }
        error(
            "could not locate trillboard-api/docs/openapi/fixtures from cwd=" +
                File("").absolutePath,
        )
    }

    private fun readFixture(name: String): String =
        File(fixturesDir(), name).readText()

    /**
     * Tolerant JSON config — `passthrough()` on the Zod side and
     * `additionalProperties: true` on the OpenAPI side mean partner
     * agents may emit fields beyond the documented surface (legacy
     * telemetry blocks, future-compatible additions). The conformance
     * loop accepts these without 400ing the entire payload.
     */
    private val tolerantJson = Json {
        ignoreUnknownKeys = true
        explicitNulls = false
        prettyPrint = false
    }

    private val pretty = Json {
        ignoreUnknownKeys = true
        explicitNulls = false
        prettyPrint = true
        prettyPrintIndent = "  "
    }

    @Test
    fun `heartbeat-minimal round-trips through generated DTO`() {
        val raw = readFixture("heartbeat-minimal.json")
        val parsed = tolerantJson.decodeFromString<PartnerDeviceHeartbeatBody>(raw)

        assertEquals(2, parsed.schemaVersion)
        assertEquals("fingerprint-fixture-minimal-001", parsed.fingerprint)
        assertEquals("screen-fixture-minimal-001", parsed.screenId)
        assertEquals("00000000-0000-0000-0000-000000000000", parsed.advertisingId)
        assertEquals(
            PartnerDeviceHeartbeatBody.AdvertisingIdType.GAID,
            parsed.advertisingIdType,
        )
        assertNotNull(parsed.capabilities)
        assertNull(parsed.nearbyBleDevices)
        assertNull(parsed.nearbyWifiNetworks)

        // Round-trip — re-encode and re-parse; field values must be stable.
        val re = tolerantJson.encodeToString(parsed)
        val reparsed = tolerantJson.decodeFromString<PartnerDeviceHeartbeatBody>(re)
        assertEquals(parsed, reparsed)
    }

    @Test
    fun `heartbeat-typical round-trips with BLE WiFi mDNS SSDP populated`() {
        val raw = readFixture("heartbeat-typical.json")
        val parsed = tolerantJson.decodeFromString<PartnerDeviceHeartbeatBody>(raw)

        // Core
        assertEquals(2, parsed.schemaVersion)
        assertEquals("active", parsed.status)

        // Ad framework
        assertEquals(false, parsed.limitAdTracking)

        // BLE — verify count + strongest signal preserved
        assertNotNull(parsed.nearbyBleDevices)
        assertEquals(5, parsed.nearbyBleDevices!!.size)
        assertEquals(-50, parsed.nearbyBleDevices!![0].rssi)
        assertEquals(76, parsed.nearbyBleDevices!![0].manufacturerCompanyId)
        assertEquals(5, parsed.bleDeviceCount)

        // WiFi
        assertNotNull(parsed.nearbyWifiNetworks)
        assertEquals(3, parsed.nearbyWifiNetworks!!.size)
        assertEquals(-42, parsed.nearbyWifiNetworks!![0].signalStrengthDbm)
        assertEquals(5180, parsed.nearbyWifiNetworks!![0].frequencyMhz)
        assertEquals(3, parsed.wifiNetworkCount)

        // mDNS
        assertNotNull(parsed.discoveredNetworkDevices)
        assertEquals(3, parsed.discoveredNetworkDevices!!.size)
        assertEquals("_googlecast._tcp.", parsed.discoveredNetworkDevices!![0].serviceType)

        // SSDP
        assertNotNull(parsed.ssdpDevices)
        assertEquals(1, parsed.ssdpDevices!!.size)
        assertEquals("Linux/3.10 UPnP/1.0 GoogleTV", parsed.ssdpDevices!![0].server)

        // HTTP probes
        assertNotNull(parsed.httpProbes)
        assertEquals(1, parsed.httpProbes!!.size)
        assertEquals("Cisco-IOS", parsed.httpProbes!![0].server)

        // Skip-reason telemetry
        assertNotNull(parsed.skipReasonCounts)
        assertEquals(0, parsed.skipReasonCounts!!["ble"]!!["permission_denied"])

        // Round-trip stability
        val re = tolerantJson.encodeToString(parsed)
        val reparsed = tolerantJson.decodeFromString<PartnerDeviceHeartbeatBody>(re)
        assertEquals(parsed, reparsed)
    }

    @Test
    fun `heartbeat-truncation factory caps 51 BLE entries to 50 strongest`() {
        val raw = readFixture("heartbeat-truncation.json")
        // The truncation fixture wraps input/expected_output in a top-level
        // envelope so we extract the input body before deserializing.
        val envelope = tolerantJson.parseToJsonElement(raw).let { it as JsonObject }
        val input = envelope["input"]!!
        val parsedInput = tolerantJson.decodeFromString<PartnerDeviceHeartbeatBody>(input.toString())

        // Sanity: input has 51 entries (oversized).
        assertNotNull(parsedInput.nearbyBleDevices)
        assertEquals(51, parsedInput.nearbyBleDevices!!.size)

        // Apply factory invariants — truncate to 50, RSSI-sorted strongest first.
        val capped = HeartbeatPayloadFactory.enforceCaps(parsedInput)
        assertEquals(
            HeartbeatPayloadFactory.MAX_NEARBY_DEVICES_PER_LIST,
            capped.nearbyBleDevices!!.size,
        )

        val expectedOutput = (envelope["expected_output"]!! as JsonObject)
        val expectedStrongestRssi = expectedOutput["expected_strongest_rssi"]!!
            .toString().toInt()
        val expectedStrongestAddress = expectedOutput["expected_strongest_address"]!!
            .toString().trim('"')
        val expectedWeakestKept = expectedOutput["expected_weakest_rssi_kept"]!!
            .toString().toInt()
        val expectedDroppedAddress = expectedOutput["expected_dropped_address"]!!
            .toString().trim('"')

        assertEquals(expectedStrongestRssi, capped.nearbyBleDevices!![0].rssi)
        assertEquals(expectedStrongestAddress, capped.nearbyBleDevices!![0].rawAddress)
        assertEquals(expectedWeakestKept, capped.nearbyBleDevices!![49].rssi)

        // The dropped device must NOT appear in the truncated list.
        val keptAddresses = capped.nearbyBleDevices!!.map { it.rawAddress }
        assertTrue(
            "dropped address $expectedDroppedAddress should not be in capped list",
            !keptAddresses.contains(expectedDroppedAddress),
        )
    }

    @Test
    fun `heartbeat-get-response round-trips through generated DTO`() {
        val raw = readFixture("heartbeat-get-response.json")
        val parsed = tolerantJson.decodeFromString<PartnerHeartbeatQueryResponse>(raw)

        assertEquals(true, parsed.success)
        assertNotNull(parsed.`data`)
        val data = parsed.`data`!!
        assertEquals("device-fixture-get-response-001", data.deviceId)
        assertEquals("2026-05-04T12:34:56.789+00:00", data.lastHeartbeatAt)
        assertEquals(12.4, data.stalenessSeconds!!, 0.001)

        // Telemetry — verify partner-safe subset
        val telemetry = data.telemetry
        assertNotNull(telemetry)
        assertEquals("on", telemetry!!.powerState)
        assertEquals(
            com.trillboards.api.types.PartnerSafeHeartbeatTelemetry.OnlineStatus.ONLINE,
            telemetry.onlineStatus,
        )
        assertEquals(-52, telemetry.networkStrengthDbm)

        // Ad delivery profile
        val profile = data.adDeliveryProfile
        assertNotNull(profile)

        // Round-trip
        val re = tolerantJson.encodeToString(parsed)
        val reparsed = tolerantJson.decodeFromString<PartnerHeartbeatQueryResponse>(re)
        assertEquals(parsed, reparsed)
    }

    @Test
    fun `factory MAX_NEARBY_DEVICES_PER_LIST matches the OpenAPI cap`() {
        // The Zod schema sets `.max(MAX_NEARBY_DEVICES_PER_LIST)` on every
        // nearby-N list. The Kotlin producer must mirror the exact same
        // value or partner agents would emit oversized lists that fail
        // server-side validation.
        assertEquals(50, HeartbeatPayloadFactory.MAX_NEARBY_DEVICES_PER_LIST)
    }

    @Test
    fun `factory build produces identical wire DTO when invariants are already satisfied`() {
        // Construct via the factory with already-capped inputs — output
        // must equal direct DTO construction.
        val ble = listOf(
            BleScanResultData(rawAddress = "AA:BB:CC:11:22:33", rssi = -50),
            BleScanResultData(rawAddress = "AA:BB:CC:44:55:66", rssi = -60),
        )
        val viaFactory = HeartbeatPayloadFactory.build(
            schemaVersion = 2,
            fingerprint = "fp-fac",
            screenId = "screen-fac",
            status = "active",
            advertisingId = "ad-1",
            advertisingIdType = PartnerDeviceHeartbeatBody.AdvertisingIdType.GAID,
            nearbyBleDevices = ble,
            bleDeviceCount = 2,
        )
        val direct = PartnerDeviceHeartbeatBody(
            schemaVersion = 2,
            fingerprint = "fp-fac",
            screenId = "screen-fac",
            status = "active",
            advertisingId = "ad-1",
            advertisingIdType = PartnerDeviceHeartbeatBody.AdvertisingIdType.GAID,
            nearbyBleDevices = ble,
            bleDeviceCount = 2,
        )
        assertEquals(direct, viaFactory)
    }
}
// CI re-trigger 1778097655
