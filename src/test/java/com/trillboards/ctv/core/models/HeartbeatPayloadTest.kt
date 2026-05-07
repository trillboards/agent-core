package com.trillboards.ctv.core.models

import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.identity.BleScanResultData
import com.trillboards.ctv.core.identity.GattDeviceInfo
import com.trillboards.ctv.core.identity.NativeSensorSnapshot
import com.trillboards.ctv.core.identity.NetworkDevice
import com.trillboards.ctv.core.identity.SsdpDeviceInfo
import com.trillboards.ctv.core.identity.WifiEnvironmentSnapshot
import com.trillboards.ctv.core.identity.WifiScanResult
import com.trillboards.ctv.core.net.ApiClient
import com.trillboards.ctv.core.sensing.CsiSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for HeartbeatPayload with sensor enrichment fields.
 */
class HeartbeatPayloadTest {

    @Test
    fun `schemaVersion defaults to 2 and is preserved through copy`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload()
        )
        assertEquals(2, payload.schemaVersion)
        val copied = payload.copy(status = "standby")
        assertEquals(2, copied.schemaVersion)
    }

    @Test
    fun `payload should hold native sensor snapshot`() {
        val snap = NativeSensorSnapshot(
            ambientLightLux = 320.5f,
            barometerPressureHpa = 1013.2f,
            sensorTimestampMs = 1730_000_000_000L
        )
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "screen",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nativeSensorSnapshot = snap
        )
        assertNotNull(payload.nativeSensorSnapshot)
        assertEquals(320.5f, payload.nativeSensorSnapshot!!.ambientLightLux!!, 0.01f)
        assertEquals(1013.2f, payload.nativeSensorSnapshot!!.barometerPressureHpa!!, 0.01f)
    }

    @Test
    fun `payload should have default null values for new fields`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = mapOf("key" to "value"),
            capabilities = DeviceCapabilityPayload()
        )

        // All new fields should be null by default (backward compatible)
        assertNull(payload.nearbyWifiNetworks)
        assertNull(payload.wifiNetworkCount)
        assertNull(payload.nearbyBleDevices)
        assertNull(payload.bleDeviceCount)
        assertNull(payload.discoveredNetworkDevices)
        assertNull(payload.networkDeviceCount)
        assertNull(payload.wifiEnvironment)
        assertNull(payload.ambientLightLux)
        assertNull(payload.barometerPressure)
        assertNull(payload.nativeSensorSnapshot)

        // Original fields still work
        assertEquals("fp123", payload.fingerprint)
        assertEquals("screen123", payload.screenId)
        assertEquals("active", payload.status)
    }

    @Test
    fun `payload should hold WiFi scan results`() {
        val networks = listOf(
            WifiScanResult("AA:BB:CC:DD:EE:01", -40, 2412, 20),
            WifiScanResult("AA:BB:CC:DD:EE:02", -60, 5180, 80),
        )

        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyWifiNetworks = networks,
            wifiNetworkCount = 2
        )

        assertNotNull(payload.nearbyWifiNetworks)
        assertEquals(2, payload.nearbyWifiNetworks!!.size)
        assertEquals(2, payload.wifiNetworkCount)
        assertEquals("AA:BB:CC:DD:EE:01", payload.nearbyWifiNetworks!![0].rawBssid)
        assertEquals(-40, payload.nearbyWifiNetworks!![0].signalStrengthDbm)
    }

    @Test
    fun `payload should hold BLE scan results`() {
        val devices = listOf(
            BleScanResultData("AA:BB:CC:DD:EE:01", -55, 1),
            BleScanResultData("AA:BB:CC:DD:EE:02", -80, 2),
        )

        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = devices,
            bleDeviceCount = 2
        )

        assertNotNull(payload.nearbyBleDevices)
        assertEquals(2, payload.nearbyBleDevices!!.size)
        assertEquals(2, payload.bleDeviceCount)
        assertEquals(-55, payload.nearbyBleDevices!![0].rssi)
    }

    @Test
    fun `payload should hold mDNS discovery results`() {
        val networkDevices = listOf(
            NetworkDevice("_roku._tcp.", "Living Room Roku", "192.168.1.10", 8060,
                mdnsModel = "Roku Ultra"),
            NetworkDevice("_airplay._tcp.", "AppleTV", null, null,
                mdnsModel = "AppleTV6,2", mdnsSoftwareVersion = "390.7.1"),
        )

        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            discoveredNetworkDevices = networkDevices,
            networkDeviceCount = 2
        )

        assertNotNull(payload.discoveredNetworkDevices)
        assertEquals(2, payload.discoveredNetworkDevices!!.size)
        assertEquals("_roku._tcp.", payload.discoveredNetworkDevices!![0].serviceType)
    }

    @Test
    fun `payload should hold environmental sensor data`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            ambientLightLux = 250.5f,
            barometerPressure = 1013.25f
        )

        assertEquals(250.5f, payload.ambientLightLux!!, 0.01f)
        assertEquals(1013.25f, payload.barometerPressure!!, 0.01f)
    }

    @Test
    fun `payload should hold all enrichment fields simultaneously`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = mapOf("manufacturer" to "Samsung"),
            capabilities = DeviceCapabilityPayload(),
            // Original identity fields
            advertisingId = "ad123",
            advertisingIdType = "gaid",
            limitAdTracking = false,
            wifiBssidHash = "connected_bssid",
            wifiSsidHash = "connected_ssid",
            gatewayIpHash = "gateway_hash",
            // New sensor enrichment fields
            nearbyWifiNetworks = listOf(WifiScanResult("AA:BB:CC:DD:EE:01", -40, 2412, 20)),
            wifiNetworkCount = 1,
            nearbyBleDevices = listOf(BleScanResultData("AA:BB:CC:DD:EE:01", -50, 0)),
            bleDeviceCount = 1,
            discoveredNetworkDevices = listOf(NetworkDevice("_roku._tcp.", "Family Roku", null, null)),
            networkDeviceCount = 1,
            ambientLightLux = 100.0f,
            barometerPressure = 1015.0f
        )

        // All fields should be accessible
        assertEquals("ad123", payload.advertisingId)
        assertEquals("connected_bssid", payload.wifiBssidHash)
        assertEquals(1, payload.nearbyWifiNetworks!!.size)
        assertEquals(1, payload.nearbyBleDevices!!.size)
        assertEquals(1, payload.discoveredNetworkDevices!!.size)
        assertEquals(100.0f, payload.ambientLightLux!!, 0.01f)
    }

    @Test
    fun `payload should hold WiFi environment analysis`() {
        val wifiEnv = WifiEnvironmentSnapshot(
            networkCount = 23,
            connectedSignalDbm = -42,
            connectedFrequencyMhz = 5180,
            connectedChannelWidthMhz = 80,
            connectedLinkSpeedMbps = 866,
            frequencyBand = "5ghz",
            channelCongestionRatio = 0.35f,
            rssiVariance = 8.2f,
            uniqueBssidCount = 23,
            medianSignalDbm = -65,
            signalSpreadDbm = 45,
            scanTimestampMs = 1712418000000L
        )

        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            wifiEnvironment = wifiEnv
        )

        assertNotNull(payload.wifiEnvironment)
        assertEquals(23, payload.wifiEnvironment!!.networkCount)
        assertEquals(-42, payload.wifiEnvironment!!.connectedSignalDbm)
        assertEquals("5ghz", payload.wifiEnvironment!!.frequencyBand)
        assertEquals(0.35f, payload.wifiEnvironment!!.channelCongestionRatio, 0.01f)
        assertEquals(8.2f, payload.wifiEnvironment!!.rssiVariance, 0.1f)
        assertEquals(80, payload.wifiEnvironment!!.connectedChannelWidthMhz)
        assertEquals(866, payload.wifiEnvironment!!.connectedLinkSpeedMbps)
    }

    @Test
    fun `payload should have default values for MDM fields`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload()
        )

        // MDM location fields should be null by default
        assertNull(payload.latitude)
        assertNull(payload.longitude)
        assertNull(payload.accuracyMeters)
        assertNull(payload.locationSource)

        // MDM compliance & enrollment defaults
        assertNull(payload.mdmCompliance)
        assertEquals(false, payload.mdmEnrolled)
        assertNull(payload.mdmOrgId)
    }

    @Test
    fun `payload should hold MDM location data`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            latitude = 40.7128,
            longitude = -74.0060,
            accuracyMeters = 10.5f,
            locationSource = "fused"
        )

        assertEquals(40.7128, payload.latitude!!, 0.0001)
        assertEquals(-74.0060, payload.longitude!!, 0.0001)
        assertEquals(10.5f, payload.accuracyMeters!!, 0.01f)
        assertEquals("fused", payload.locationSource)
    }

    @Test
    fun `payload should hold MDM enrollment and compliance data`() {
        val compliance = mapOf<String, Any?>(
            "os_version" to "14",
            "security_patch" to "2026-01-05",
            "storage_free_pct" to 42,
            "battery_level" to 85,
            "encryption_active" to true
        )

        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            mdmCompliance = compliance,
            mdmEnrolled = true,
            mdmOrgId = "org_abc123"
        )

        assertEquals(true, payload.mdmEnrolled)
        assertEquals("org_abc123", payload.mdmOrgId)
        assertNotNull(payload.mdmCompliance)
        assertEquals("14", payload.mdmCompliance!!["os_version"])
        assertEquals(true, payload.mdmCompliance!!["encryption_active"])
    }

    @Test
    fun `payload data class copy should preserve new fields`() {
        val original = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s1",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            ambientLightLux = 300.0f
        )

        val copy = original.copy(status = "standby")

        assertEquals("standby", copy.status)
        assertEquals(300.0f, copy.ambientLightLux!!, 0.01f)
    }

    @Test
    fun `payload should have default null values for CSI fields`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload()
        )

        assertNull(payload.csiSnapshot)
        assertNull(payload.csiNodeCount)
    }

    @Test
    fun `payload should hold CSI snapshot data`() {
        val snapshot = CsiSnapshot(
            nodeId = 1,
            occupantCount = 3,
            motionScore = 0.75f,
            signalQuality = 0.95f,
            subcarrierCount = 52,
            captureRateHz = 100.0f,
            avgRssiDbm = -45,
            framesProcessed = 1000,
            framesDropped = 50,
            windowStartMs = 1000L,
            windowEndMs = 11000L
        )

        val payload = HeartbeatPayload(
            fingerprint = "fp123",
            screenId = "screen123",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            csiSnapshot = snapshot,
            csiNodeCount = 2
        )

        assertNotNull(payload.csiSnapshot)
        assertEquals(1, payload.csiSnapshot!!.nodeId)
        assertEquals(3, payload.csiSnapshot!!.occupantCount)
        assertEquals(0.75f, payload.csiSnapshot!!.motionScore, 0.001f)
        assertEquals(0.95f, payload.csiSnapshot!!.signalQuality, 0.001f)
        assertEquals(52, payload.csiSnapshot!!.subcarrierCount)
        assertEquals(2, payload.csiNodeCount)
    }

    @Test
    fun `payload copy should preserve CSI snapshot`() {
        val snapshot = CsiSnapshot(
            nodeId = 1,
            occupantCount = 5,
            motionScore = 0.5f,
            signalQuality = 0.9f,
            subcarrierCount = 52,
            captureRateHz = 100.0f,
            avgRssiDbm = -40,
            framesProcessed = 900,
            framesDropped = 100,
            windowStartMs = 0L,
            windowEndMs = 10000L
        )

        val original = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s1",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            csiSnapshot = snapshot,
            csiNodeCount = 1
        )

        val copy = original.copy(status = "standby")

        assertEquals("standby", copy.status)
        assertNotNull(copy.csiSnapshot)
        assertEquals(5, copy.csiSnapshot!!.occupantCount)
        assertEquals(1, copy.csiNodeCount)
    }

    @Test
    fun `JSON-roundtrip preserves all new identity fields`() {
        // Build a payload that exercises every new field added in Phase 2.
        val ble = BleScanResultData(
            rawAddress = "AA:BB:CC:DD:EE:FF",
            rssi = -55,
            deviceType = 2,
            manufacturerCompanyId = 0x004c,
            appleContinuitySubtype = 0x10,
            stableManufacturerPayloadHex = "2102018004",
            serviceUuids = listOf("fe0d", "fe2c"),
            txPowerDbm = 12,
            ibeaconUuid = "e2c56db5-dffb-48d2-b060-d0f5a71096e0",
            ibeaconMajor = 1,
            ibeaconMinor = 2
        )
        val mdns = NetworkDevice(
            serviceType = "_airplay._tcp.",
            instanceName = "AppleTV",
            host = "192.168.1.20",
            port = 7000,
            mdnsModel = "AppleTV6,2",
            mdnsVendor = null,
            mdnsSoftwareVersion = "390.7.1"
        )
        val payload = HeartbeatPayload(
            schemaVersion = 2,
            fingerprint = "fp",
            screenId = "screen-123",
            status = "active",
            metadata = mapOf("model" to "Samsung"),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = listOf(ble),
            bleDeviceCount = 1,
            discoveredNetworkDevices = listOf(mdns),
            networkDeviceCount = 1
        )

        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val client = ApiClient(cfg)
        val json = client.payloadToJson(payload)

        // Top-level
        assertEquals(2, json.getInt("schema_version"))

        // BLE
        val bleArr = json.getJSONArray("nearby_ble_devices")
        val b0 = bleArr.getJSONObject(0)
        assertEquals("AA:BB:CC:DD:EE:FF", b0.getString("raw_address"))
        assertEquals(-55, b0.getInt("rssi"))
        assertEquals(0x004c, b0.getInt("manufacturer_company_id"))
        assertEquals(0x10, b0.getInt("apple_continuity_subtype"))
        assertEquals("2102018004", b0.getString("stable_manufacturer_payload_hex"))
        assertTrue(b0.has("service_uuids"))
        assertEquals("fe0d", b0.getJSONArray("service_uuids").getString(0))
        assertEquals(12, b0.getInt("tx_power_dbm"))
        assertEquals("e2c56db5-dffb-48d2-b060-d0f5a71096e0", b0.getString("ibeacon_uuid"))
        assertEquals(1, b0.getInt("ibeacon_major"))
        assertEquals(2, b0.getInt("ibeacon_minor"))

        // mDNS
        val nd = json.getJSONArray("discovered_network_devices").getJSONObject(0)
        assertEquals("AppleTV", nd.getString("instance_name"))
        assertEquals("AppleTV6,2", nd.getString("mdns_model"))
        assertEquals("390.7.1", nd.getString("mdns_software_version"))
        assertFalse(nd.has("mdns_vendor"))
    }

    @Test
    fun `legacy payload (no new fields) JSON-roundtrip works`() {
        // Older agents may emit payloads without any of the new fields. The wire
        // contract must continue to serialize cleanly.
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "screen",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload()
        )
        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val client = ApiClient(cfg)
        val json = client.payloadToJson(payload)
        assertEquals(2, json.getInt("schema_version"))
        assertFalse(json.has("nearby_ble_devices"))
        assertFalse(json.has("discovered_network_devices"))
        assertFalse(json.has("native_sensor_snapshot"))
    }

    @Test
    fun `JSON-roundtrip serializes nativeSensorSnapshot under snake_case key`() {
        val snap = NativeSensorSnapshot(
            ambientLightLux = 320.5f,
            barometerPressureHpa = 1013.25f,
            sensorTimestampMs = 1730_000_000_000L
        )
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "screen",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nativeSensorSnapshot = snap
        )
        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val json = ApiClient(cfg).payloadToJson(payload)
        assertTrue(json.has("native_sensor_snapshot"))
        val sensor = json.getJSONObject("native_sensor_snapshot")
        assertEquals(320.5, sensor.getDouble("ambient_light_lux"), 0.01)
        assertEquals(1013.25, sensor.getDouble("barometer_pressure_hpa"), 0.01)
        // Wire contract (post 2026-05-01 redesign — CH migration 059):
        // ambient_light_lux + barometer_pressure_hpa only. The four dropped
        // fields (magnetic_flux_events_per_min, proximity_events_per_min,
        // footstep_count, gait_signature_hash) were retired with their CH
        // columns. `sensor_timestamp_ms` is also not serialized — there's
        // no CH column for it, only used in-process.
        assertFalse(sensor.has("magnetic_flux_events_per_min"))
        assertFalse(sensor.has("proximity_events_per_min"))
        assertFalse(sensor.has("footstep_count"))
        assertFalse(sensor.has("gait_signature_hash"))
        assertFalse(sensor.has("dominant_orientation_deg"))
        assertFalse(sensor.has("cadence_hz"))
        assertFalse(sensor.has("sensor_timestamp_ms"))
    }

    // ────────────────────────────────────────────────────────────────────
    // Producer-side payload bounds (Phase 3 of redis-stream-scale-fix.md).
    // A misconfigured tablet must not be able to ship arbitrary-size lists
    // and DoS the telemetry pipeline. Cap is enforced at the data class
    // construction boundary, sized for Redis Cloud RAM budget at the
    // 6,800-screen Dolphin Media launch.
    // ────────────────────────────────────────────────────────────────────

    @Test
    fun `MAX_NEARBY_DEVICES_PER_LIST is 50`() {
        // Locked at 50 — sized for Redis Cloud RAM budget at the
        // 6,800-screen Dolphin Media launch. Raise only if signal
        // density grows.
        assertEquals(50, MAX_NEARBY_DEVICES_PER_LIST)
    }

    @Test
    fun `nearbyBleDevices is capped to 50 by RSSI strongest-first`() {
        // Build 200 BLE devices with monotonically weakening RSSI. The cap
        // must keep the strongest 50 (-40 down to -89), not arbitrary 50.
        val devices = (0 until 200).map { i ->
            BleScanResultData(
                rawAddress = "AA:BB:CC:DD:E${(i / 16) % 16}:${"%02X".format(i % 256)}",
                rssi = -40 - i,  // -40, -41, -42, ..., -239
                deviceType = 1
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = devices
        )
        assertNotNull(payload.nearbyBleDevices)
        assertEquals(50, payload.nearbyBleDevices!!.size)
        // Strongest signal preserved
        assertEquals(-40, payload.nearbyBleDevices!![0].rssi)
        // 50th-strongest is -89 (RSSI is signed; less-negative = stronger)
        assertEquals(-89, payload.nearbyBleDevices!![49].rssi)
    }

    @Test
    fun `nearbyBleDevices below cap passes through unchanged in original order`() {
        val devices = listOf(
            BleScanResultData("AA:BB:CC:DD:EE:01", -90, 1),
            BleScanResultData("AA:BB:CC:DD:EE:02", -50, 1),
            BleScanResultData("AA:BB:CC:DD:EE:03", -70, 1)
        )
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = devices
        )
        // No truncation needed → preserve original order, do not RSSI-sort
        assertEquals(3, payload.nearbyBleDevices!!.size)
        assertEquals("AA:BB:CC:DD:EE:01", payload.nearbyBleDevices!![0].rawAddress)
        assertEquals("AA:BB:CC:DD:EE:02", payload.nearbyBleDevices!![1].rawAddress)
        assertEquals("AA:BB:CC:DD:EE:03", payload.nearbyBleDevices!![2].rawAddress)
    }

    @Test
    fun `nearbyWifiNetworks is capped to 50 by RSSI strongest-first`() {
        val networks = (0 until 120).map { i ->
            WifiScanResult(
                rawBssid = "AA:BB:CC:DD:E${(i / 16) % 16}:${"%02X".format(i % 256)}",
                signalStrengthDbm = -40 - i,
                frequencyMhz = 2412,
                channelWidthMhz = 20
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyWifiNetworks = networks
        )
        assertEquals(50, payload.nearbyWifiNetworks!!.size)
        assertEquals(-40, payload.nearbyWifiNetworks!![0].signalStrengthDbm)
        assertEquals(-89, payload.nearbyWifiNetworks!![49].signalStrengthDbm)
    }

    @Test
    fun `discoveredNetworkDevices (mDNS) is capped to 50 in original order`() {
        val devices = (0 until 100).map { i ->
            NetworkDevice(
                serviceType = "_roku._tcp.",
                instanceName = "Roku-$i",
                host = "192.168.1.${i % 256}",
                port = 8060
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            discoveredNetworkDevices = devices
        )
        assertEquals(50, payload.discoveredNetworkDevices!!.size)
        // Original order preserved (mDNS has no RSSI signal — take first 50)
        assertEquals("Roku-0", payload.discoveredNetworkDevices!![0].instanceName)
        assertEquals("Roku-49", payload.discoveredNetworkDevices!![49].instanceName)
    }

    @Test
    fun `ssdpDevices is capped to 50 in original order`() {
        val devices = (0 until 75).map { i ->
            SsdpDeviceInfo(
                location = "http://192.168.1.${i % 256}:1900/desc.xml",
                friendlyName = "Device-$i",
                st = "urn:schemas-upnp-org:device:MediaServer:1"
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            ssdpDevices = devices
        )
        assertEquals(50, payload.ssdpDevices!!.size)
        assertEquals("Device-0", payload.ssdpDevices!![0].friendlyName)
        assertEquals("Device-49", payload.ssdpDevices!![49].friendlyName)
    }

    @Test
    fun `httpProbes is capped to 50 in original order`() {
        val probes = (0 until 100).map { i ->
            mapOf<String, Any?>(
                "host" to "192.168.1.$i",
                "port" to 80,
                "server" to "Server-$i"
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            httpProbes = probes
        )
        assertEquals(50, payload.httpProbes!!.size)
        assertEquals("192.168.1.0", payload.httpProbes!![0]["host"])
        assertEquals("192.168.1.49", payload.httpProbes!![49]["host"])
    }

    @Test
    fun `bleGattDevices is capped to 50 in original order`() {
        val devices = (0 until 80).map { i ->
            GattDeviceInfo(
                rawAddress = "AA:BB:CC:DD:E${(i / 16) % 16}:${"%02X".format(i % 256)}",
                manufacturer = "Vendor-$i",
                modelNumber = "Model-$i",
                hardwareRevision = null,
                firmwareRevision = null,
                durationMs = 100L
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            bleGattDevices = devices
        )
        assertEquals(50, payload.bleGattDevices!!.size)
        assertEquals("Vendor-0", payload.bleGattDevices!![0].manufacturer)
        assertEquals("Vendor-49", payload.bleGattDevices!![49].manufacturer)
    }

    @Test
    fun `null lists are not affected by the cap`() {
        // Defensive: a null input must remain null, not become an empty list.
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = null,
            nearbyWifiNetworks = null,
            ssdpDevices = null,
            httpProbes = null,
            bleGattDevices = null,
            discoveredNetworkDevices = null
        )
        assertNull(payload.nearbyBleDevices)
        assertNull(payload.nearbyWifiNetworks)
        assertNull(payload.ssdpDevices)
        assertNull(payload.httpProbes)
        assertNull(payload.bleGattDevices)
        assertNull(payload.discoveredNetworkDevices)
    }

    @Test
    fun `empty lists pass through as empty lists`() {
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = emptyList(),
            nearbyWifiNetworks = emptyList()
        )
        assertNotNull(payload.nearbyBleDevices)
        assertEquals(0, payload.nearbyBleDevices!!.size)
        assertEquals(0, payload.nearbyWifiNetworks!!.size)
    }

    @Test
    fun `exactly 50 entries pass through unchanged in original order`() {
        // Boundary: cap of 50 means 50 entries are NOT truncated.
        val devices = (0 until 50).map { i ->
            BleScanResultData(
                rawAddress = "AA:BB:CC:DD:E${(i / 16) % 16}:${"%02X".format(i % 256)}",
                rssi = -90,  // identical RSSIs — order must not change
                deviceType = 1
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = devices
        )
        assertEquals(50, payload.nearbyBleDevices!!.size)
        // Original order preserved (no truncation triggered)
        assertEquals(devices[0].rawAddress, payload.nearbyBleDevices!![0].rawAddress)
        assertEquals(devices[49].rawAddress, payload.nearbyBleDevices!![49].rawAddress)
    }

    @Test
    fun `data class copy preserves the cap on the original list`() {
        // The cap is applied at construction. .copy() of an unrelated
        // field must not lose the already-capped state.
        val devices = (0 until 100).map { i ->
            BleScanResultData(
                rawAddress = "AA:BB:CC:DD:E${(i / 16) % 16}:${"%02X".format(i % 256)}",
                rssi = -40 - i,
                deviceType = 1
            )
        }
        val original = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = devices
        )
        assertEquals(50, original.nearbyBleDevices!!.size)
        val copied = original.copy(status = "standby")
        assertEquals(50, copied.nearbyBleDevices!!.size)
        assertEquals("standby", copied.status)
    }

    @Test
    fun `JSON wire format reflects the cap`() {
        // End-to-end: serialize a payload with 200 BLE devices and assert
        // the wire JSON has only 50 entries. This is the actual DoS
        // protection — the Redis stream entry size is bounded.
        val devices = (0 until 200).map { i ->
            BleScanResultData(
                rawAddress = "AA:BB:CC:DD:E${(i / 16) % 16}:${"%02X".format(i % 256)}",
                rssi = -40 - i,
                deviceType = 1
            )
        }
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "s",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(),
            nearbyBleDevices = devices
        )
        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val json = ApiClient(cfg).payloadToJson(payload)
        val arr = json.getJSONArray("nearby_ble_devices")
        assertEquals(50, arr.length())
        // Strongest signal first
        assertEquals(-40, arr.getJSONObject(0).getInt("rssi"))
        assertEquals(-89, arr.getJSONObject(49).getInt("rssi"))
    }

    // ── Phase 2 prereq PR 2 — wire format for new MLCapabilities fields ──

    @Test
    fun `wire format includes ABI and OS API and display geometry under camelCase mlCapabilities keys`() {
        // End-to-end: a populated MLCapabilities emits each new key under
        // capabilities.mlCapabilities exactly as the server expects in
        // services/deviceCommandSupport.js:buildCapabilityUpdateDoc.
        val capabilities = DeviceCapabilityPayload(
            mlCapabilities = DeviceCapabilityPayload.MLCapabilities(
                chipsetVendor = "qualcomm",
                chipsetName = "SM8650",
                totalRamMb = 12288,
                availableRamMb = 6144,
                gpuName = "Adreno 750",
                hasNpu = true,
                npuName = "Qualcomm Hexagon DSP",
                gpuDelegateSupported = true,
                nnapiSupported = true,
                recommendedModelTier = "PREMIUM",
                maxVlmSizeMb = 5394,
                cpuAbi = "arm64-v8a",
                osApiLevel = 34,
                screenWidthPx = 2800,
                screenHeightPx = 1752,
                densityDpi = 360
            )
        )
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "screen",
            status = "active",
            metadata = emptyMap(),
            capabilities = capabilities
        )
        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val json = ApiClient(cfg).payloadToJson(payload)
        val ml = json.getJSONObject("capabilities").getJSONObject("mlCapabilities")

        // Original 11 fields preserved
        assertEquals("qualcomm", ml.getString("chipsetVendor"))
        assertEquals(12288, ml.getInt("totalRamMb"))
        assertEquals(6144, ml.getInt("availableRamMb"))
        assertEquals("Qualcomm Hexagon DSP", ml.getString("npuName"))
        assertTrue(ml.getBoolean("nnapiSupported"))

        // 5 new fields (Phase 2 prereq PR 2)
        assertEquals("arm64-v8a", ml.getString("cpuAbi"))
        assertEquals(34, ml.getInt("osApiLevel"))
        assertEquals(2800, ml.getInt("screenWidthPx"))
        assertEquals(1752, ml.getInt("screenHeightPx"))
        assertEquals(360, ml.getInt("densityDpi"))
    }

    @Test
    fun `wire format emits cpuAbi as JSON null when not detected`() {
        // Defensive contract — empty Build.SUPPORTED_ABIS yields cpuAbi=null.
        // Wire JSON must serialize this as JSON null (not the string "null"
        // and not an absent key). Server-side `buildCapabilityUpdateDoc`
        // skips the column update when the value is null.
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "screen",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload(
                mlCapabilities = DeviceCapabilityPayload.MLCapabilities(cpuAbi = null)
            )
        )
        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val json = ApiClient(cfg).payloadToJson(payload)
        val ml = json.getJSONObject("capabilities").getJSONObject("mlCapabilities")
        assertTrue(ml.has("cpuAbi"))
        assertTrue(ml.isNull("cpuAbi"))
    }

    @Test
    fun `wire format default mlCapabilities emits JSON null for fail-zero sentinels`() {
        // Codex P1 (PR #4540): default ctor (HardwareManifest.detect() failed
        // or hasn't finished) MUST emit JSON null — NOT 0 — for the geometry
        // / API / RAM fields. Previously the wire emitted 0, and the server
        // wrote 0 to typed columns via COALESCE, overwriting previously valid
        // values whenever a transient detect failure happened. Server now
        // treats null as "skip column update"; 0 is no longer a valid sentinel.
        val payload = HeartbeatPayload(
            fingerprint = "fp",
            screenId = "screen",
            status = "active",
            metadata = emptyMap(),
            capabilities = DeviceCapabilityPayload()
        )
        val cfg = AgentConfig(
            apiBaseUrl = "https://example.test",
            heartbeatPath = "/h",
            sharedPrefsName = "test",
            overlayRefreshAction = "x.refresh",
            overlayBlackoutAction = "x.blackout"
        )
        val json = ApiClient(cfg).payloadToJson(payload)
        val ml = json.getJSONObject("capabilities").getJSONObject("mlCapabilities")

        // Every nullable Int field must serialize as JSON null when default-ctored.
        assertTrue(ml.isNull("cpuAbi"))
        assertTrue(ml.isNull("osApiLevel"))
        assertTrue(ml.isNull("screenWidthPx"))
        assertTrue(ml.isNull("screenHeightPx"))
        assertTrue(ml.isNull("densityDpi"))
        assertTrue(ml.isNull("totalRamMb"))
        assertTrue(ml.isNull("availableRamMb"))
        assertTrue(ml.isNull("maxVlmSizeMb"))
    }
}
