package com.trillboards.ctv.core.net

import android.util.Log
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.audience.AudienceMetricsPayload
import com.trillboards.ctv.core.audience.ClassificationResult
import com.trillboards.ctv.core.audience.ClassificationSource
import com.trillboards.ctv.core.audience.IntentType
import com.trillboards.ctv.core.audience.PriceContext
import com.trillboards.ctv.core.models.DeviceCapabilityPayload
import com.trillboards.ctv.core.models.HeartbeatPayload
import org.json.JSONArray
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.UUID

class ApiClient(
    private val config: AgentConfig,
    private val deviceTokenProvider: () -> String? = { null }
) {
    private val client: OkHttpClient = OkHttpClient.Builder()
        .addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC })
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    data class ScreenResolution(
        val screenId: String,
        val venueType: String?,
        val deviceToken: String?
    )

    private fun currentDeviceToken(): String? =
        deviceTokenProvider().orEmpty().trim().takeIf { it.isNotEmpty() }

    private fun authRequestBuilder(url: String): Request.Builder {
        val timestamp = System.currentTimeMillis().toString()
        val nonce = UUID.randomUUID().toString()
        val builder = Request.Builder()
            .url(url)
            .header("X-Device-Timestamp", timestamp)
            .header("X-Device-Nonce", nonce)

        currentDeviceToken()?.let { token ->
            builder.header("X-Device-Token", token)
            builder.header("X-Trillboard-Device-Token", token)
        }

        return builder
    }

    suspend fun fetchScreenId(fingerprint: String): String? = withContext(Dispatchers.IO) {
        fetchScreenResolution(fingerprint)?.screenId
    }

    /**
     * Fetch screen data including venue_type for VAS initialization.
     * Returns Pair(screenId, venueType) or null if not found.
     */
    suspend fun fetchScreenData(fingerprint: String): Pair<String, String?>? = withContext(Dispatchers.IO) {
        fetchScreenResolution(fingerprint)?.let { Pair(it.screenId, it.venueType) }
    }

    suspend fun fetchScreenResolution(fingerprint: String): ScreenResolution? = withContext(Dispatchers.IO) {
        runCatching {
            val request = authRequestBuilder("${config.apiBaseUrl}/v2/earner/check-screen/$fingerprint")
                .get()
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@use null
                val json = JSONObject(response.body?.string() ?: return@use null)
                if (!json.optBoolean("status")) return@use null
                val data = json.optJSONObject("data") ?: return@use null
                val screenId = data.optString("_id", "") .ifEmpty { data.optString("screenId", "") }.ifEmpty { return@use null }
                val venueType = data.optString("venue_type", "").ifEmpty { null }
                val deviceToken = data.optString("device_token", "").ifEmpty { null }
                ScreenResolution(
                    screenId = screenId,
                    venueType = venueType,
                    deviceToken = deviceToken
                )
            }
        }.onFailure { Log.w(TAG, "fetchScreenResolution failed", it) }.getOrNull()
    }

    data class HeartbeatResponse(
        val socketToken: String? = null,
        val screenId: String? = null,
        val commands: List<JSONObject>? = null
    )

    suspend fun sendHeartbeat(payload: HeartbeatPayload): HeartbeatResponse? = withContext(Dispatchers.IO) {
        runCatching {
            val json = payloadToJson(payload)
            val request = authRequestBuilder("${config.apiBaseUrl}${config.heartbeatPath}")
                .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@use null
                val body = response.body?.string() ?: return@use null
                val responseJson = JSONObject(body)
                // Parse commands from heartbeat response (Redis-signaled PG-native delivery)
                val commandsJson = responseJson.optJSONArray("commands")
                val parsedCommands = if (commandsJson != null && commandsJson.length() > 0) {
                    (0 until commandsJson.length()).map { commandsJson.getJSONObject(it) }
                } else null

                HeartbeatResponse(
                    socketToken = responseJson.optString("socket_token", "").takeIf { it.isNotBlank() },
                    screenId = responseJson.optString("screenId", "").takeIf { it.isNotBlank() },
                    commands = parsedCommands
                )
            }
        }.onFailure { Log.w(TAG, "sendHeartbeat failed", it) }.getOrNull()
    }

    /**
     * Pure-domain HeartbeatPayload → JSON serializer. Mirrors agent-core-lite's
     * helper so unit tests can assert wire contract without a network stub.
     */
    fun payloadToJson(payload: HeartbeatPayload): JSONObject = JSONObject().apply {
        put("schema_version", payload.schemaVersion)
        put("fingerprint", payload.fingerprint)
        payload.screenId?.let { put("screenId", it) }
        put("status", payload.status)
        put("metadata", JSONObject().apply {
            payload.metadata.forEach { (key, value) -> if (value != null) put(key, value) }
        })
        put("capabilities", capabilitiesJson(payload.capabilities))
        payload.advertisingId?.let { put("advertising_id", it) }
        payload.advertisingIdType?.let { put("advertising_id_type", it) }
        payload.limitAdTracking?.let { put("limit_ad_tracking", it) }
        payload.wifiBssidHash?.let { put("wifi_bssid_hash", it) }
        payload.wifiSsidHash?.let { put("wifi_ssid_hash", it) }
        payload.gatewayIpHash?.let { put("gateway_ip_hash", it) }
        payload.nearbyWifiNetworks?.let { networks ->
            put("nearby_wifi_networks", JSONArray().apply {
                for (network in networks) {
                    put(JSONObject().apply {
                        put("raw_bssid", network.rawBssid)
                        put("signal_dbm", network.signalStrengthDbm)
                        put("frequency_mhz", network.frequencyMhz)
                        network.channelWidthMhz?.let { put("channel_width_mhz", it) }
                    })
                }
            })
        }
        payload.wifiNetworkCount?.let { put("wifi_network_count", it) }
        payload.wifiEnvironment?.let { env ->
            put("wifi_environment", JSONObject().apply {
                put("network_count", env.networkCount)
                put("connected_signal_dbm", env.connectedSignalDbm)
                put("connected_frequency_mhz", env.connectedFrequencyMhz)
                env.connectedChannelWidthMhz?.let { put("connected_channel_width_mhz", it) }
                env.connectedLinkSpeedMbps?.let { put("connected_link_speed_mbps", it) }
                put("frequency_band", env.frequencyBand)
                put("channel_congestion_ratio", env.channelCongestionRatio.toDouble())
                put("rssi_variance", env.rssiVariance.toDouble())
                put("unique_bssid_count", env.uniqueBssidCount)
                put("median_signal_dbm", env.medianSignalDbm)
                put("signal_spread_dbm", env.signalSpreadDbm)
                put("scan_timestamp_ms", env.scanTimestampMs)
            })
        }
        payload.nearbyBleDevices?.let { devices ->
            put("nearby_ble_devices", JSONArray().apply {
                for (d in devices) {
                    put(JSONObject().apply {
                        put("raw_address", d.rawAddress)
                        put("rssi", d.rssi)
                        put("device_type", d.deviceType)
                        d.manufacturerCompanyId?.let { put("manufacturer_company_id", it) }
                        d.appleContinuitySubtype?.let { put("apple_continuity_subtype", it) }
                        d.stableManufacturerPayloadHex?.let { put("stable_manufacturer_payload_hex", it) }
                        d.serviceUuids?.let { put("service_uuids", JSONArray(it)) }
                        d.txPowerDbm?.let { put("tx_power_dbm", it) }
                        d.ibeaconUuid?.let { put("ibeacon_uuid", it) }
                        d.ibeaconMajor?.let { put("ibeacon_major", it) }
                        d.ibeaconMinor?.let { put("ibeacon_minor", it) }
                    })
                }
            })
        }
        payload.bleDeviceCount?.let { put("ble_device_count", it) }
        payload.bleGattDevices?.let { gattDevices ->
            put("ble_gatt_devices", JSONArray().apply {
                for (g in gattDevices) {
                    put(JSONObject().apply {
                        put("raw_address", g.rawAddress)
                        g.manufacturer?.let { put("manufacturer", it) }
                        g.modelNumber?.let { put("model_number", it) }
                        g.hardwareRevision?.let { put("hardware_revision", it) }
                        g.firmwareRevision?.let { put("firmware_revision", it) }
                    })
                }
            })
        }
        payload.discoveredNetworkDevices?.let { devices ->
            put("discovered_network_devices", JSONArray().apply {
                for (d in devices) {
                    put(JSONObject().apply {
                        put("service_type", d.serviceType)
                        put("instance_name", d.instanceName)
                        d.host?.let { put("host", it) }
                        d.port?.let { put("port", it) }
                        d.mdnsModel?.let { put("mdns_model", it) }
                        d.mdnsVendor?.let { put("mdns_vendor", it) }
                        d.mdnsSoftwareVersion?.let { put("mdns_software_version", it) }
                    })
                }
            })
        }
        payload.networkDeviceCount?.let { put("network_device_count", it) }
        // Phase 2 of the multi-protocol discovery rewrite — SSDP / UPnP M-SEARCH.
        // Captures TVs / NAS / routers / smart-home hubs that don't broadcast
        // on mDNS. One JSONObject per unique LOCATION with snake_case keys
        // matching the existing wire-format convention. Server-side
        // `SignalIngestService.normalize()` HMACs friendlyName / manufacturer /
        // modelName / UDN under the daily pepper. Wire-additive — older agents
        // that don't include `ssdp_devices` keep the existing row contract.
        payload.ssdpDevices?.let { devices ->
            put("ssdp_devices", JSONArray().apply {
                for (d in devices) {
                    put(JSONObject().apply {
                        put("location", d.location)
                        d.server?.let { put("server", it) }
                        d.friendlyName?.let { put("friendly_name", it) }
                        d.manufacturer?.let { put("manufacturer", it) }
                        d.modelName?.let { put("model_name", it) }
                        d.udn?.let { put("udn", it) }
                        put("st", d.st)
                    })
                }
            })
        }

        // ── Phase 3 — ARP cache rows ──
        // Each entry is a Map<String, Any?> from the Rust ArpEntry binding
        // (`ip`, `mac`, `iface`, `ouiVendor`). We re-emit them snake_case at
        // the wire layer to match the server's canonical contract
        // (oui_vendor not ouiVendor).
        payload.arpDevices?.let { entries ->
            put("arp_devices", JSONArray().apply {
                for (e in entries) {
                    put(JSONObject().apply {
                        e["ip"]?.let { put("ip", it.toString()) }
                        e["mac"]?.let { put("mac", it.toString()) }
                        e["iface"]?.let { put("iface", it.toString()) }
                        // Accept either case from the producer.
                        val vendor = e["oui_vendor"] ?: e["ouiVendor"]
                        vendor?.let { put("oui_vendor", it.toString()) }
                    })
                }
            })
        }

        // ── Phase 4 — HTTP probe rows ──
        // Each entry is `{host, port, server}` from the Rust probe_http()
        // wrapper. The Rust side already sanitized the `server` header so
        // we can ship it as-is.
        payload.httpProbes?.let { probes ->
            put("http_probes", JSONArray().apply {
                for (p in probes) {
                    put(JSONObject().apply {
                        p["host"]?.let { put("host", it.toString()) }
                        p["port"]?.let { put("port", (it as? Number)?.toInt() ?: it.toString().toInt()) }
                        p["server"]?.let { put("server", it.toString()) }
                    })
                }
            })
        }

        // ── Phase 6 — UWB peer ranging ──
        // Each entry is `{address, distance_mm, rssi_dbm}` from
        // UwbDiscovery.discoverUwbPeers(). Server-side normalize() HMACs
        // the address under the daily pepper.
        payload.uwbPeers?.let { peers ->
            put("uwb_peers", JSONArray().apply {
                for (peer in peers) {
                    put(JSONObject().apply {
                        peer["address"]?.let { put("address", it.toString()) }
                        peer["distance_mm"]?.let { put("distance_mm", (it as? Number)?.toInt() ?: it.toString().toInt()) }
                        peer["rssi_dbm"]?.let { put("rssi_dbm", (it as? Number)?.toInt() ?: it.toString().toInt()) }
                    })
                }
            })
        }

        // ── Phase 6 bonus — Auracast (LE Audio Broadcast) rows ──
        // Each entry is `{broadcastId, broadcastName, publicBroadcastData, sourceId}`.
        // The server-side normalize() will HMAC the broadcast name + data under
        // the daily pepper.
        payload.auracastBroadcasts?.let { broadcasts ->
            put("auracast_broadcasts", JSONArray().apply {
                for (b in broadcasts) {
                    put(JSONObject().apply {
                        b["broadcastId"]?.let { put("broadcast_id", (it as? Number)?.toInt() ?: it.toString().toInt()) }
                        b["broadcastName"]?.let { put("broadcast_name", it.toString()) }
                        b["publicBroadcastData"]?.let { put("public_broadcast_data", it.toString()) }
                        b["sourceId"]?.let { put("source_id", (it as? Number)?.toInt() ?: it.toString().toInt()) }
                    })
                }
            })
        }

        // ── Phase 6 — BLE Channel Sounding measurements ──
        // Distance (mm) and angle-of-arrival (deg) from nearby BLE devices via
        // the Android 16+ native Channel Sounding API. Each entry is
        // `{address, distance_mm, aoa_deg}`. Wire format mirrors httpProbes /
        // auracast_broadcasts: Map → JSONObject, pass through raw values.
        payload.channelSoundingMeasurements?.let { measurements ->
            put("channel_sounding_measurements", JSONArray().apply {
                for (m in measurements) {
                    put(JSONObject().apply {
                        m["address"]?.let { put("address", it.toString()) }
                        m["distance_mm"]?.let { put("distance_mm", (it as? Number)?.toFloat() ?: it.toString().toFloat()) }
                        m["aoa_deg"]?.let { put("aoa_deg", (it as? Number)?.toFloat() ?: it.toString().toFloat()) }
                    })
                }
            })
        }

        payload.csiSnapshot?.let { csi ->
            put("csi_snapshot", JSONObject().apply {
                put("node_id", csi.nodeId)
                put("occupant_count", csi.occupantCount)
                put("motion_score", csi.motionScore.toDouble())
                put("signal_quality", csi.signalQuality.toDouble())
                put("subcarrier_count", csi.subcarrierCount)
                put("capture_rate_hz", csi.captureRateHz.toDouble())
                put("avg_rssi_dbm", csi.avgRssiDbm)
                put("frames_processed", csi.framesProcessed)
                put("frames_dropped", csi.framesDropped)
                put("window_start_ms", csi.windowStartMs)
                put("window_end_ms", csi.windowEndMs)
                put("hardware_type", csi.hardwareType)
            })
        }
        payload.csiNodeCount?.let { put("csi_node_count", it) }
        payload.nativeSensorSnapshot?.let { sensor ->
            // Wire contract (post 2026-05-01 redesign — CH migration 059):
            // ambient_light_lux + barometer_pressure_hpa only. The four
            // dropped fields (magnetic_flux_events_per_min,
            // proximity_events_per_min, footstep_count, gait_signature_hash)
            // were retired with their CH columns — agent stops sampling
            // accelerometer / gyroscope / magnetometer / proximity entirely.
            // Server-side `SignalIngestService.normalize()` hoists these
            // two values onto every emitted BLE/WiFi/mDNS row instead of
            // creating a dedicated source='native_sensor' row.
            put("native_sensor_snapshot", JSONObject().apply {
                sensor.ambientLightLux?.let { put("ambient_light_lux", it.toDouble()) }
                sensor.barometerPressureHpa?.let { put("barometer_pressure_hpa", it.toDouble()) }
            })
        }

        // Phase 0 of redis-stream-scale-fix — structured skip-reason channel.
        // Per-source skip-reason counts emitted as `skip_reason_counts` on the
        // wire. Server-side `SignalIngestService.normalize()` accepts the
        // snake_case key and emits per-(source,reason) CW counters. Field is
        // omitted entirely when the aggregator had nothing to report.
        payload.skipReasonCounts?.takeIf { it.isNotEmpty() }?.let { perSource ->
            put("skip_reason_counts", JSONObject().apply {
                for ((source, reasons) in perSource) {
                    put(source, JSONObject().apply {
                        for ((reason, count) in reasons) {
                            put(reason, count)
                        }
                    })
                }
            })
        }
    }

    suspend fun sendCommandAck(
        commandId: String,
        status: String,
        result: JSONObject = JSONObject(),
        screenId: String? = null
    ): Boolean = withContext(Dispatchers.IO) {
        if (commandId.isBlank()) return@withContext false

        runCatching {
            val json = JSONObject().apply {
                put("command_id", commandId)
                put("status", status)
                put("result", result)
                screenId?.takeIf { it.isNotBlank() }?.let { put("screen_id", it) }
            }
            val request = authRequestBuilder("${config.apiBaseUrl}/openrtb/v1/command-ack")
                .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w(TAG, "sendCommandAck failed: ${response.code}")
                }
                response.isSuccessful
            }
        }.onFailure { Log.w(TAG, "sendCommandAck failed", it) }.getOrDefault(false)
    }

    /**
     * Send audience metrics to the API.
     */
    suspend fun sendAudienceMetrics(payload: AudienceMetricsPayload) = withContext(Dispatchers.IO) {
        runCatching {
            val json = JSONObject().apply {
                payload.screenId?.let { put("screenId", it) }
                put("fingerprint", payload.fingerprint)
                put("timestamp", payload.timestamp)
                put("intervalMs", payload.intervalMs)
                put("viewerCount", payload.viewerCount)
                put("peakViewerCount", payload.peakViewerCount)
                put("attentionScore", payload.attentionScore)
                put("demographics", JSONObject().apply {
                    put("ageRanges", JSONObject(payload.demographics.ageRanges))
                    put("genders", JSONObject(payload.demographics.genderEstimates))
                })
                put("dwellTime", JSONObject().apply {
                    put("avg", payload.dwellTime.averageSeconds)
                    put("max", payload.dwellTime.maxSeconds)
                    put("min", payload.dwellTime.minSeconds)
                    put("total", payload.dwellTime.totalViewerSeconds)
                })
                put("environment", JSONObject().apply {
                    put("ambientLight", payload.environment.ambientLightLux)
                    put("temperature", payload.environment.deviceTemperatureC)
                    put("orientation", payload.environment.orientation)
                })
            }
            val request = authRequestBuilder("${config.apiBaseUrl}/v2/earner/audience-metrics")
                .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w(TAG, "sendAudienceMetrics failed: ${response.code}")
                }
            }
        }.onFailure { Log.w(TAG, "sendAudienceMetrics failed", it) }
    }

    /**
     * Upload a batch of paired (camera-pose, CSI-window) training samples to the
     * WiFlow training pipeline. The collector batches samples client-side and
     * calls this method per batch.
     *
     * Endpoint contract: see
     * `agent-core/src/main/java/com/trillboards/ctv/core/sensing/PAIRED_TRAINING_API.md`
     *
     * @param payload Pre-built JSON body conforming to the contract
     * @return true on a 2xx response, false on any error or non-success status
     */
    suspend fun sendPairedTrainingSamples(payload: JSONObject): Boolean = withContext(Dispatchers.IO) {
        runCatching {
            val request = authRequestBuilder("${config.apiBaseUrl}/v2/sensing/paired-training-samples")
                .post(payload.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                .build()
            client.newCall(request).execute().use { response ->
                val ok = response.isSuccessful
                if (!ok) {
                    Log.w(TAG, "sendPairedTrainingSamples failed: ${response.code}")
                }
                ok
            }
        }.onFailure { Log.w(TAG, "sendPairedTrainingSamples failed", it) }
            .getOrDefault(false)
    }

    private fun capabilitiesJson(capabilities: DeviceCapabilityPayload): JSONObject = JSONObject().apply {
        put("powerControl", JSONObject().apply {
            put("cec", capabilities.powerControl.cec)
            put("wol", capabilities.powerControl.wol)
            put("softBlackout", capabilities.powerControl.softBlackout)
        })
        put("inputControl", JSONObject().apply {
            put("hdmi", capabilities.inputControl.hdmi)
            put("appSwitch", capabilities.inputControl.appSwitch)
        })
        put("audioControl", JSONObject().apply {
            put("volume", capabilities.audioControl.volume)
            put("mute", capabilities.audioControl.mute)
        })
        put("environmentSensors", JSONObject().apply {
            put("ambientLight", capabilities.environmentSensors.ambientLight)
            put("temperature", capabilities.environmentSensors.temperature)
        })
        put("audienceSensing", JSONObject().apply {
            put("cameraAvailable", capabilities.audienceSensing.cameraAvailable)
            put("cameraType", capabilities.audienceSensing.cameraType)
            put("cameraCount", capabilities.audienceSensing.cameraCount)
            put("microphoneAvailable", capabilities.audienceSensing.microphoneAvailable)
            put("sensingMode", capabilities.audienceSensing.sensingMode)
            // 2026-05-03 fix-ctv-camera-never-started-product-fix: surface
            // CameraHealthMonitor's per-device gauge to the heartbeat. Lets
            // the server distinguish "camera open delivering frames" from
            // "camera never started for 30 days" without inferring it from
            // missing vision_call_log rows.
            put("cameraHealth", JSONObject().apply {
                val ch = capabilities.audienceSensing.cameraHealth
                put("state", ch.state)
                put("lastFrameAtMs", ch.lastFrameAtMs)
                put("retryCount", ch.retryCount)
                ch.lastFailureClass?.let { put("lastFailureClass", it) } ?: put("lastFailureClass", JSONObject.NULL)
                ch.lastFailureMessage?.let { put("lastFailureMessage", it) } ?: put("lastFailureMessage", JSONObject.NULL)
            })
        })
        // MDM Management Capabilities
        put("managementCapabilities", JSONObject().apply {
            put("restart", capabilities.managementCapabilities.restart)
            put("reboot", capabilities.managementCapabilities.reboot)
            put("screenshot", capabilities.managementCapabilities.screenshot)
            put("clearCache", capabilities.managementCapabilities.clearCache)
            put("kioskMode", capabilities.managementCapabilities.kioskMode)
            put("installApk", capabilities.managementCapabilities.installApk)
            put("wipe", capabilities.managementCapabilities.wipe)
        })
        // ML hardware capabilities from HardwareManifest
        //
        // Codex P1 (PR #4540): nullable Int fields (totalRamMb / availableRamMb /
        // maxVlmSizeMb / osApiLevel / screenWidthPx / screenHeightPx / densityDpi)
        // emit JSONObject.NULL when the underlying probe failed (or
        // MLCapabilities was default-constructed). The server then writes
        // NULL via COALESCE, preserving prior valid values. The previous
        // wire shape emitted 0 for these defaults, which overwrote real
        // typed-column values via the dual-write upsert.
        put("mlCapabilities", JSONObject().apply {
            put("chipsetVendor", capabilities.mlCapabilities.chipsetVendor)
            put("chipsetName", capabilities.mlCapabilities.chipsetName)
            capabilities.mlCapabilities.totalRamMb?.let { put("totalRamMb", it) } ?: put("totalRamMb", JSONObject.NULL)
            capabilities.mlCapabilities.availableRamMb?.let { put("availableRamMb", it) } ?: put("availableRamMb", JSONObject.NULL)
            capabilities.mlCapabilities.gpuName?.let { put("gpuName", it) } ?: put("gpuName", JSONObject.NULL)
            put("hasNpu", capabilities.mlCapabilities.hasNpu)
            capabilities.mlCapabilities.npuName?.let { put("npuName", it) } ?: put("npuName", JSONObject.NULL)
            put("gpuDelegateSupported", capabilities.mlCapabilities.gpuDelegateSupported)
            put("nnapiSupported", capabilities.mlCapabilities.nnapiSupported)
            put("recommendedModelTier", capabilities.mlCapabilities.recommendedModelTier)
            capabilities.mlCapabilities.maxVlmSizeMb?.let { put("maxVlmSizeMb", it) } ?: put("maxVlmSizeMb", JSONObject.NULL)
            // Phase 2 prereq PR 2 — ABI / OS API / display geometry.
            // cpuAbi: JSONObject.NULL on the empty-array edge case; otherwise
            // emit the string directly. Server tolerates both snake_case +
            // camelCase keys.
            capabilities.mlCapabilities.cpuAbi?.let { put("cpuAbi", it) } ?: put("cpuAbi", JSONObject.NULL)
            capabilities.mlCapabilities.osApiLevel?.let { put("osApiLevel", it) } ?: put("osApiLevel", JSONObject.NULL)
            capabilities.mlCapabilities.screenWidthPx?.let { put("screenWidthPx", it) } ?: put("screenWidthPx", JSONObject.NULL)
            capabilities.mlCapabilities.screenHeightPx?.let { put("screenHeightPx", it) } ?: put("screenHeightPx", JSONObject.NULL)
            capabilities.mlCapabilities.densityDpi?.let { put("densityDpi", it) } ?: put("densityDpi", JSONObject.NULL)
        })
        // WiFi CSI sensing hardware capabilities (mirrors agent-core-lite)
        put("wifiCsiSensing", JSONObject().apply {
            put("csiAvailable", capabilities.wifiCsiSensing.csiAvailable)
            put("csiNodeCount", capabilities.wifiCsiSensing.csiNodeCount)
            put("csiHardwareType", capabilities.wifiCsiSensing.csiHardwareType)
        })
        put("isDeviceOwner", capabilities.isDeviceOwner)
        put("agentType", capabilities.agentType)
        put("agentVersion", capabilities.agentVersion)
        put("capabilitiesVersion", capabilities.capabilitiesVersion)
    }

    /**
     * Analyze transcript using server-side Gemini for accurate NLP.
     *
     * PRIVACY: Transcript is sent over HTTPS, not logged on server,
     * and deleted immediately after processing.
     *
     * @param transcript The speech transcript to analyze
     * @param fingerprint Device fingerprint for authentication
     * @return ClassificationResult with intent, brands, and sentiment
     */
    suspend fun analyzeWithGemini(
        transcript: String,
        fingerprint: String
    ): ClassificationResult = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()

        runCatching {
            val requestJson = JSONObject().apply {
                put("transcript", transcript)
                put("fingerprint", fingerprint)
            }

            val request = authRequestBuilder("${config.apiBaseUrl}/v2/earner/analyze-speech")
                .post(requestJson.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                .build()

            client.newCall(request).execute().use { response ->
                val latencyMs = System.currentTimeMillis() - startTime

                if (!response.isSuccessful) {
                    Log.w(TAG, "analyzeWithGemini failed: ${response.code}")
                    return@use ClassificationResult(
                        intentType = IntentType.UNKNOWN,
                        confidence = 0f,
                        latencyMs = latencyMs,
                        source = ClassificationSource.SERVER_GEMINI,
                        error = "HTTP ${response.code}"
                    )
                }

                val body = response.body?.string() ?: return@use ClassificationResult(
                    intentType = IntentType.UNKNOWN,
                    confidence = 0f,
                    latencyMs = latencyMs,
                    source = ClassificationSource.SERVER_GEMINI,
                    error = "Empty response"
                )

                val json = JSONObject(body)

                // Parse intent
                val intentStr = json.optString("intent", "NEUTRAL")
                val intentType = when (intentStr.uppercase()) {
                    "PURCHASE_INTENT" -> IntentType.PURCHASE_INTENT
                    "PRICE_INQUIRY" -> IntentType.PRICE_INQUIRY
                    "PRODUCT_INTEREST" -> IntentType.PRODUCT_INTEREST
                    "COMPARISON" -> IntentType.COMPARISON
                    "BROWSING" -> IntentType.BROWSING
                    "NEGATIVE" -> IntentType.NEGATIVE
                    else -> IntentType.NEUTRAL
                }

                // Parse brands array
                val brandsArray = json.optJSONArray("brands") ?: JSONArray()
                val brands = (0 until brandsArray.length()).map { brandsArray.getString(it) }

                // Parse products array
                val productsArray = json.optJSONArray("products") ?: JSONArray()
                val products = (0 until productsArray.length()).map { productsArray.getString(it) }

                // Parse price context
                val priceJson = json.optJSONObject("priceContext")
                val priceContext = if (priceJson != null) {
                    PriceContext(
                        mentioned = priceJson.optBoolean("mentioned", false),
                        sensitivity = priceJson.optString("sensitivity", "NONE")
                    )
                } else null

                Log.d(TAG, "Gemini analysis: intent=$intentType, " +
                        "brands=${brands.size}, latency=${latencyMs}ms")

                ClassificationResult(
                    intentType = intentType,
                    confidence = json.optDouble("confidence", 0.8).toFloat(),
                    latencyMs = latencyMs,
                    source = ClassificationSource.SERVER_GEMINI,
                    brands = brands,
                    products = products,
                    sentiment = json.optString("sentiment", "NEUTRAL"),
                    priceContext = priceContext
                )
            }
        }.getOrElse { e ->
            Log.e(TAG, "analyzeWithGemini exception", e)
            ClassificationResult(
                intentType = IntentType.UNKNOWN,
                confidence = 0f,
                latencyMs = System.currentTimeMillis() - startTime,
                source = ClassificationSource.SERVER_GEMINI,
                error = e.message
            )
        }
    }

    /**
     * Response from screenshot upload URL endpoint
     */
    data class ScreenshotUploadUrlResponse(
        val uploadUrl: String,
        val cdnUrl: String,
        val expiresIn: Int
    )

    /**
     * Get a presigned URL for uploading a screenshot to S3.
     * Called when processing device.screenshot MDM command.
     *
     * @param fingerprint Device fingerprint for authentication
     * @return ScreenshotUploadUrlResponse with upload and CDN URLs, or null on failure
     */
    suspend fun getScreenshotUploadUrl(fingerprint: String): ScreenshotUploadUrlResponse? =
        withContext(Dispatchers.IO) {
            runCatching {
                val json = JSONObject().apply {
                    put("fingerprint", fingerprint)
                }
                val request = authRequestBuilder("${config.apiBaseUrl}/v2/earner/screenshot-upload-url")
                    .post(json.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
                    .build()

                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.w(TAG, "getScreenshotUploadUrl failed: ${response.code}")
                        return@use null
                    }
                    val body = response.body?.string() ?: return@use null
                    val data = JSONObject(body)
                    val uploadUrl = data.optString("uploadUrl", "").takeIf { it.isNotBlank() } ?: return@use null
                    val cdnUrl = data.optString("cdnUrl", "").takeIf { it.isNotBlank() } ?: return@use null
                    ScreenshotUploadUrlResponse(
                        uploadUrl = uploadUrl,
                        cdnUrl = cdnUrl,
                        expiresIn = data.optInt("expiresIn", 300)
                    )
                }
            }.onFailure { Log.w(TAG, "getScreenshotUploadUrl failed", it) }.getOrNull()
        }

    /**
     * APK manifest entry for OTA updates
     */
    data class ApkManifestEntry(
        val version: String,
        val versionCode: Int,
        val url: String,
        val minSdkVersion: Int,
        val changelog: String
    )

    /**
     * Fetch APK manifest to check for available updates.
     * Called when processing device.checkUpdate MDM command.
     *
     * @param agentType Agent type (fire-tv-agent, android-tv-agent, tablet-agent)
     * @return ApkManifestEntry with version info and download URL, or null on failure
     */
    suspend fun fetchApkManifest(agentType: String): ApkManifestEntry? =
        withContext(Dispatchers.IO) {
            runCatching {
                val request = authRequestBuilder("${config.apiBaseUrl}/v2/earner/apk-manifest?agent=$agentType")
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.w(TAG, "fetchApkManifest failed: ${response.code}")
                        return@use null
                    }
                    val body = response.body?.string() ?: return@use null
                    val data = JSONObject(body)
                    ApkManifestEntry(
                        version = data.optString("version", ""),
                        versionCode = data.optInt("versionCode", 0),
                        url = data.optString("url", ""),
                        minSdkVersion = data.optInt("minSdkVersion", 21),
                        changelog = data.optString("changelog", "")
                    )
                }
            }.onFailure { Log.w(TAG, "fetchApkManifest failed", it) }.getOrNull()
        }

    companion object {
        private const val TAG = "AgentCoreApiClient"
    }
}
