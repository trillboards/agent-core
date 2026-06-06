package com.trillboards.sdk

import android.content.Context
import android.util.Log
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.DeviceIdentity
import com.trillboards.ctv.core.audience.AudienceSensingService
import com.trillboards.ctv.core.audience.SensingConfig as AudienceSensingConfig
import com.trillboards.ctv.core.audience.SensingResult
import com.trillboards.ctv.core.net.ApiClient
import com.trillboards.ctv.core.socket.AgentSocketManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Public partner-facing entry point for the Trillboards Sensing SDK.
 *
 * Three-line integration:
 *
 * ```
 * TrillboardsSensingSdk.start(
 *     context = applicationContext,
 *     partnerApiKey = "tb_ctv_xxx"
 * )
 * ```
 *
 * Scope is intentionally narrow: this is the **sensing-only** entry point.
 * Ad serving (VAST, IMA, programmatic events, proof-of-play) lives on the
 * five HTTP endpoints documented in
 * `docs/integrations/partner-native-ima-sdk.md` Sections 1–11. Partners drive
 * those endpoints from their existing IMA SDK setup; sensing runs alongside
 * over the same `partnerApiKey` and emits aggregate signals (face counts,
 * age bins, audio classifications, dwell estimates, BLE/WiFi co-viewing) to
 * the Trillboards backend.
 *
 * What `start()` actually wires up:
 *
 * 1. Builds an [AgentConfig] with partner-friendly defaults — no overlay
 *    intent action exposure, no shared-prefs collisions with partner code.
 * 2. Creates an [ApiClient] that authenticates every emit with the
 *    `partnerApiKey` (passed via the `X-Device-Token` /
 *    `X-Trillboard-Device-Token` headers).
 * 3. Connects an [AgentSocketManager] for real-time signal emission.
 * 4. Starts an [AudienceSensingService] whose CameraX binding follows the
 *    host-supplied [LifecycleOwner] when one is passed to [start], else
 *    `ProcessLifecycleOwner` (the default — a real lifecycle without forcing
 *    partners to declare a foreground Service). Headless hosts that sense from a
 *    `FOREGROUND_SERVICE_TYPE_CAMERA` Service with no visible Activity should
 *    pass their `LifecycleService` so the camera stays bound when backgrounded.
 *
 * Speech recognition (Moonshine ASR via sherpa-onnx) is opt-in: agent-core
 * marks sherpa-onnx as `compileOnly`, so partners who don't bundle the
 * `sherpa-onnx-1.12.26.aar` AAR + `moonshine-tiny/` model assets get a
 * clean degrade path — face / audio / BLE / WiFi sensing runs unaffected.
 * See `partner-native-ima-sdk.md` Section 12.5b for the BYO sherpa-onnx
 * instructions.
 */
object TrillboardsSensingSdk {
    private const val TAG = "TrillboardsSensingSdk"
    private const val SDK_PREFS_NAME = "trillboards_sdk"
    private const val OVERLAY_REFRESH_ACTION = "com.trillboards.sdk.OVERLAY_REFRESH"
    private const val OVERLAY_BLACKOUT_ACTION = "com.trillboards.sdk.OVERLAY_BLACKOUT"

    @Volatile private var apiClient: ApiClient? = null
    @Volatile private var socketManager: AgentSocketManager? = null
    @Volatile private var audienceSensing: AudienceSensingService? = null
    @Volatile private var heartbeatJob: Job? = null
    @Volatile private var started: Boolean = false

    /**
     * Start the Sensing SDK. Idempotent — calling twice is a no-op. Binds the
     * camera to `ProcessLifecycleOwner` (Activity-visibility scoped — correct for
     * the SDK embedded in the on-screen player). Headless hosts should use the
     * [start] overload that takes an explicit `lifecycleOwner`.
     *
     * @param context Any [Context] (Application or Activity). The SDK
     *   immediately calls `applicationContext` so the lifecycle is bound to
     *   the process, not the caller.
     * @param partnerApiKey The API key Trillboards issues you. The same key
     *   you use for the heartbeat / ads endpoints in
     *   `partner-native-ima-sdk.md` Sections 2–7. The SDK passes this as the
     *   device token on every backend emit.
     * @param config Optional [SensingSdkConfig] overrides. Defaults match
     *   production tablets at typical retail mounting distances.
     */
    @JvmStatic
    @JvmOverloads
    fun start(
        context: Context,
        partnerApiKey: String,
        config: SensingSdkConfig = SensingSdkConfig()
    ) {
        // Delegates to the 4-arg overload with the default ProcessLifecycleOwner.
        // Kept as a distinct function (NOT a defaulted 4th param on this one) so the
        // existing 2-/3-arg ABI — including Kotlin's synthetic start$default — stays
        // unchanged for callers compiled against an earlier agent-core AAR.
        start(context, partnerApiKey, config, null)
    }

    /**
     * Start the Sensing SDK, binding the camera to a host-supplied [LifecycleOwner].
     *
     * @param context Any [Context]; `applicationContext` is used internally.
     * @param partnerApiKey The API key Trillboards issues you (device token on emits).
     * @param config [SensingSdkConfig] overrides.
     * @param lifecycleOwner The [LifecycleOwner] the camera (CameraX) binds to, or
     *   `null` for the default `ProcessLifecycleOwner` (Activity-visibility scoped —
     *   correct when the SDK is embedded in the on-screen player). **Headless hosts**
     *   that run sensing from a `FOREGROUND_SERVICE_TYPE_CAMERA` Service with no
     *   visible Activity MUST pass their own owner (e.g. an androidx
     *   `LifecycleService` held at `STARTED` for the service's lifetime), otherwise
     *   CameraX unbinds the camera the moment the host is backgrounded. Audio,
     *   discovery, aggregation, and heartbeat are not lifecycle-bound and already
     *   run in the background, so this only governs the camera.
     */
    @JvmStatic
    fun start(
        context: Context,
        partnerApiKey: String,
        config: SensingSdkConfig,
        lifecycleOwner: LifecycleOwner?
    ) {
        synchronized(this) {
            if (started) {
                Log.w(TAG, "TrillboardsSensingSdk.start() called twice — ignoring second call")
                return
            }

            require(partnerApiKey.isNotBlank()) {
                "TrillboardsSensingSdk.start: partnerApiKey must not be blank"
            }

            val appContext = context.applicationContext
            // Identity: a partner who pre-registered this device via POST /v1/partner/device
            // passes the SAME stable identifier here as `config.deviceId` (mirrors
            // MeasurementConfig.deviceId in the CTV Measurement SDK). The backend keys the
            // device->screen binding off this fingerprint, so it MUST match what was registered.
            // When unset, fall back to the hardware-derived fingerprint.
            val fingerprint = config.deviceId?.takeIf { it.isNotBlank() }
                ?: DeviceIdentity.stableFingerprint(appContext, SDK_PREFS_NAME)

            Log.i(TAG, "Starting TrillboardsSensingSdk (fingerprint=${fingerprint.take(12)}…)")

            val agentConfig = AgentConfig(
                apiBaseUrl = config.apiBaseUrl,
                socketUrl = config.socketUrl,
                heartbeatIntervalMs = config.heartbeatIntervalMs,
                sharedPrefsName = SDK_PREFS_NAME,
                overlayRefreshAction = OVERLAY_REFRESH_ACTION,
                overlayBlackoutAction = OVERLAY_BLACKOUT_ACTION
            )

            // ApiClient routes the partnerApiKey through deviceTokenProvider so
            // every emit (heartbeat, audience-metrics, screenshot-upload-url,
            // etc.) carries `X-Device-Token: tb_ctv_…` automatically.
            val client = ApiClient(agentConfig) { partnerApiKey }
            apiClient = client

            // Real-time signal channel. Sensing emits Socket.io events
            // (audienceSignals, audienceDemographics, deviceCapabilities) over
            // this connection — the partner does not need to subscribe; the
            // backend consumes them for the audience-attention flywheel.
            val socket = AgentSocketManager(agentConfig)
            if (config.connectSocket) {
                socket.connect(
                    fingerprint = fingerprint,
                    listener = NoopSocketListener,
                    deviceToken = partnerApiKey
                )
            }
            socketManager = socket

            val sensing = AudienceSensingService(
                context = appContext,
                fingerprint = fingerprint,
                socketManager = socket,
                apiBaseUrl = config.apiBaseUrl,
                // Partner SDK consumers carry their device-token-equivalent as partnerApiKey
                // (line 124 above passes the same value to socket.connect); FrameCaptureManager
                // routes it through as X-Device-Token on the /v2/earner/audience-analyze POST.
                deviceTokenProvider = { partnerApiKey }
            )

            // Wire the optional host-app sensing callback (default null in
            // SensingSdkConfig, so existing integrations are unaffected).
            sensing.onSensingResult = config.onSensingResult

            if (config.sensingEnabled) {
                val audienceConfig = AudienceSensingConfig(
                    enableFaceDetection = config.faceDetectionEnabled,
                    enableAudioClassification = config.audioClassificationEnabled,
                    enableSpeechIntelligence = config.speechIntelligenceEnabled,
                    enableDemographicsCapture = config.demographicsCaptureEnabled,
                    enableEmotionalEngagement = config.emotionalEngagementEnabled
                )

                // Camera binds to the host-supplied lifecycleOwner when present,
                // else ProcessLifecycleOwner. ProcessLifecycleOwner only tracks
                // Activity visibility, so a headless host (foreground camera Service,
                // no visible Activity) MUST supply its own owner or CameraX unbinds
                // the camera when the app backgrounds. The chosen owner flows through
                // to startFaceDetection / initializeFrameCapture / rebindCamera.
                sensing.start(lifecycleOwner ?: ProcessLifecycleOwner.get(), audienceConfig)
            } else {
                Log.i(TAG, "sensingEnabled=false — SDK initialized but no sensors started")
            }

            audienceSensing = sensing

            // Resolve this device's screen binding the SAME way the first-party agent does
            // (BaseDeviceAgentService.resolveScreenId): GET /v2/earner/check-screen/<fingerprint>
            // returns the screenId for a device the partner already registered+bound via
            // POST /v1/partner/device. Without a screenId, audience-analyze emits unattributed
            // (screen_mongo_id=null) — this is what binds sensing to the partner's screen so
            // Moments/chips show in the Venue portal. Reuses ApiClient.fetchScreenResolution +
            // AudienceSensingService.setScreenId — no new endpoint, no parallel provisioning.
            CoroutineScope(Dispatchers.IO).launch {
                runCatching {
                    val resolved = client.fetchScreenResolution(fingerprint)
                    if (resolved != null) {
                        sensing.setScreenId(resolved.screenId)
                        resolved.venueType?.takeIf { it.isNotBlank() }?.let { sensing.setVenueType(it) }
                        // Adopt the canonical device fingerprint so audience-analyze attributes
                        // in strict-enforce mode too (soft mode already uses the resolved
                        // screenId). No-op for first-party/hardware-fp devices.
                        resolved.fingerprint?.takeIf { it.isNotBlank() }?.let { sensing.setAudienceFingerprint(it) }
                        Log.i(TAG, "Resolved screen binding: screenId=${resolved.screenId}, venueType=${resolved.venueType}")
                    } else {
                        Log.w(TAG, "No screen bound to fingerprint yet — register this device via " +
                            "POST /v1/partner/device (external_id = your config.deviceId). Sensing runs " +
                            "but emits unattributed until the binding exists.")
                    }
                }.onFailure { Log.w(TAG, "Screen resolution failed", it) }
            }

            // Opt-in partner heartbeat (default off). The embeddable SDK does NOT
            // mark the device online otherwise; when enabled, POST the partner
            // heartbeat on a timer so `lastHeartbeatAt` populates. Keyed by
            // config.deviceId (the partner external_id the endpoint accepts).
            val heartbeatDeviceId = config.deviceId?.takeIf { it.isNotBlank() }
            if (config.heartbeatEnabled && heartbeatDeviceId != null) {
                heartbeatJob = CoroutineScope(Dispatchers.IO).launch {
                    while (isActive) {
                        runCatching { client.sendPartnerHeartbeat(heartbeatDeviceId) }
                        delay(config.heartbeatIntervalMs.coerceAtLeast(10_000L))
                    }
                }
            }

            started = true

            Log.i(TAG, "TrillboardsSensingSdk started successfully")
        }
    }

    /**
     * Stop the Sensing SDK. Releases camera, audio, and Socket.io resources.
     * Safe to call from `Application.onTerminate()` or anywhere the partner
     * decides to wind down audience sensing.
     *
     * The `context` parameter is reserved for forward compatibility (e.g.,
     * unregistering broadcast receivers in a future release) — it is not
     * read in this version. Callers should still pass the same Context
     * they used in [start].
     */
    @JvmStatic
    fun stop(@Suppress("UNUSED_PARAMETER") context: Context) {
        synchronized(this) {
            if (!started) {
                Log.w(TAG, "TrillboardsSensingSdk.stop() called before start() — ignoring")
                return
            }

            Log.i(TAG, "Stopping TrillboardsSensingSdk")

            heartbeatJob?.cancel()
            heartbeatJob = null
            audienceSensing?.stop()
            socketManager?.disconnect()

            audienceSensing = null
            socketManager = null
            apiClient = null
            started = false

            Log.i(TAG, "TrillboardsSensingSdk stopped")
        }
    }

    /**
     * Returns true after [start] has run successfully.
     * Test / diagnostic helper — partners typically don't need to call this.
     */
    @JvmStatic
    fun isStarted(): Boolean = started

    /**
     * Reset internal state — used by unit tests to rerun start() with a
     * fresh fixture. Not part of the public partner API; visible to allow
     * the test class in `src/test/` to exercise the lifecycle without a
     * real `Context`.
     */
    internal fun resetForTesting() {
        synchronized(this) {
            heartbeatJob?.cancel()
            heartbeatJob = null
            audienceSensing = null
            socketManager = null
            apiClient = null
            started = false
        }
    }

    private object NoopSocketListener : AgentSocketManager.Listener {
        override fun onConnect(socketId: String?) {
            Log.d(TAG, "Socket connected: socketId=$socketId")
        }

        override fun onDisconnect() {
            Log.d(TAG, "Socket disconnected")
        }

        override fun onError(args: Array<Any?>) {
            // Socket errors are non-fatal; ApiClient HTTP path keeps emitting.
            Log.w(TAG, "Socket error: ${args.firstOrNull()}")
        }

        override fun onPrivateMessage(payload: JSONObject) {
            // Partners on this SDK don't receive private messages; ignore.
        }

        override fun onDeviceCommand(payload: JSONObject) {
            // MDM commands (restart / reboot / kiosk-mode flips) belong to the
            // first-party Trillboards agent. Partners receive them on their
            // own ad-serving control plane (Sections 1–11); ignore here.
            Log.d(TAG, "Ignoring device command in sensing-only SDK: ${payload.optString("command_type")}")
        }

        override fun onScreenBinding(payload: JSONObject) {
            // The backend pushes `screen_binding` when a device is paired to a screen
            // (e.g. the partner registers the device AFTER the SDK is already running).
            // Pick up the screenId AND venueType live so sensing attributes + venue-specific
            // sensing (VenueConfig/VAS/custom chips) initialize without an app restart —
            // same handling as BaseDeviceAgentService.handleScreenBinding.
            val action = payload.optString("action", "paired")
            if (action != "paired") return  // sensing-only SDK acts only on pairing
            val sid = payload.optString("screenId", "").ifEmpty { payload.optString("screen_id", "") }
            if (sid.isEmpty()) return
            val venueType = payload.optString("venueType", null)
            Log.i(TAG, "Screen binding pushed: screenId=$sid, venueType=$venueType")
            audienceSensing?.setScreenId(sid)
            if (!venueType.isNullOrEmpty()) audienceSensing?.setVenueType(venueType)
        }
    }
}

/**
 * Public configuration for [TrillboardsSensingSdk]. Every field has a
 * production-ready default; partners typically need to override only
 * `apiBaseUrl`/`socketUrl` for non-prod environments.
 */
data class SensingSdkConfig(
    val apiBaseUrl: String = "https://api.trillboards.com",
    val socketUrl: String = "https://chat.trillboards.com",
    val heartbeatIntervalMs: Long = 30_000L,
    val sensingEnabled: Boolean = true,
    val faceDetectionEnabled: Boolean = true,
    val audioClassificationEnabled: Boolean = true,
    /**
     * Speech intelligence requires the `sherpa-onnx-1.12.26.aar` AAR plus
     * the `moonshine-tiny/` model assets. agent-core ships sherpa-onnx as
     * `compileOnly` (the AAR is not on Maven Central), so the runtime
     * guard in [com.trillboards.ctv.core.audience.AudienceSensingService]
     * checks for the `moonshine-tiny/tokens.txt` asset before any
     * sherpa-onnx class is loaded. Default `true` so partners who DO
     * bundle the model see speech work; partners who don't are silently
     * skipped.
     */
    val speechIntelligenceEnabled: Boolean = true,
    val demographicsCaptureEnabled: Boolean = true,
    val emotionalEngagementEnabled: Boolean = true,
    /**
     * Disable `connectSocket` if you only want HTTP-side sensing emits
     * (heartbeat, audience-metrics) and prefer not to maintain a long-lived
     * Socket.io connection. The on-device sensing pipeline still runs;
     * realtime emits queue locally and are dropped if the socket isn't
     * connected.
     */
    val connectSocket: Boolean = true,
    /**
     * Stable per-device identifier that ties this install to a screen in the
     * Trillboards Venue portal. Pass the SAME value you used as `external_id`
     * when you registered the device via `POST /v1/partner/device` (mirrors
     * `MeasurementConfig.deviceId` in the CTV Measurement SDK). The backend keys
     * the device->screen binding off this, so sensing / Moments / custom chips
     * attribute to the right screen. When null, the SDK uses a hardware-derived
     * fingerprint (sensing still runs, but won't attribute until the device is
     * registered+bound).
     */
    val deviceId: String? = null,
    /**
     * Opt-in: when true, the SDK marks the device online by POSTing the partner
     * heartbeat (`/v1/partner/device/<deviceId>/heartbeat`) every
     * [heartbeatIntervalMs]. Default false — the embeddable SDK does not send a
     * heartbeat otherwise, so existing integrations are unaffected. Requires
     * [deviceId] (the partner external_id) to be set.
     */
    val heartbeatEnabled: Boolean = false,
    /**
     * Opt-in host-app callback, invoked once per aggregation window (~10s) with
     * the anonymous [SensingResult] for that window — aggregated signals only,
     * never raw frames or per-individual identity. Lets you read sensing
     * on-device without a backend round-trip. Default null (no callback, zero
     * overhead). Do not block in the callback; it runs on the SDK's sensing
     * thread.
     */
    val onSensingResult: ((SensingResult) -> Unit)? = null
)
