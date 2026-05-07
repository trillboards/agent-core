package com.trillboards.ctv.core.service

import android.Manifest
import android.app.Notification
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.trillboards.ctv.core.AgentConfig as CoreAgentConfig
import com.trillboards.ctv.core.DeviceIdentity
import com.trillboards.ctv.core.audience.AudienceSensingService
import com.trillboards.ctv.core.audience.InsightExtractor
import com.trillboards.ctv.core.audience.SensingConfig
import com.trillboards.ctv.core.audience.SensingProfileManager
import com.trillboards.ctv.core.device.AgentPowerAdapter
import com.trillboards.ctv.core.device.KioskLockManager
import com.trillboards.ctv.core.models.HeartbeatPayload
import com.trillboards.ctv.core.net.ApiClient
import com.trillboards.ctv.core.socket.AgentSocketManager
import com.trillboards.ctv.core.stability.LatestWinsCommandGate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.time.Instant

/**
 * Platform-agnostic device-agent service skeleton extracted from
 * `tablet-agent/.../service/DeviceAgentService.kt` (PR 3/5).
 *
 * Each platform agent (tablet-agent, android-tv-agent full flavor) extends this
 * class with a thin shell that supplies platform identifiers (mainActivityClass,
 * deviceAdminReceiverClass, foregroundChannelId, AgentConfig + AgentMetadata,
 * BuildConfig version), and provides the platform-specific power adapter +
 * sensing-profile-manager + command-processor wiring through abstract methods.
 *
 * Three production-bug fixes that previously lived only on tablet-agent now ride
 * the base, so android-tv-agent inherits them automatically:
 *
 *  1. `ensureMainActivityRunning(reason)` — re-launches MainActivity if killed
 *     by the OS. Critical on RockChip Gocast STBs where the OS aggressively
 *     kills the activity after the camera-permission dialog opens, causing
 *     the dialog to die without a grant.
 *  2. `scheduleCameraRebind`/`scheduleAudioRebind` — coalesces rebind
 *     broadcasts so RockChip cameras don't flap. The receiver reads
 *     `EXTRA_REBIND_REASON` + `EXTRA_REBIND_DELAY_MS`; missing extras fall
 *     through to `delayMs=0` (preserves android-tv's prior immediate-rebind
 *     behavior, gains coalescer when extras are supplied).
 *  3. Explicit `enableDemographicsCapture = hasCameraPermission` +
 *     `enableSpeechIntelligence = hasAudioPermission` flag passing to
 *     `audienceSensing.start(...)`. Eliminates the foreground-service
 *     `camera`-type race where a platform requests the camera service type
 *     but `audienceSensing.start()` did not actually open the camera (because
 *     the SensingConfig data class default skewed wrong for android-tv).
 *
 * Tablet-only items (RecoveryLadder, CrashTelemetryReporter,
 * SubsystemHealthRegistry, memory-pressure attenuation, native-sensor harvest,
 * BLE GATT enrichment, mDNS/SSDP/HTTP-probe/Auracast/UWB/ChannelSounding
 * cached discovery) stay on the platform shell — they are augmented via
 * `extraHeartbeatMetadata()` and the `onSocketConnected()` / `reloadPlayer()` /
 * `recreatePlayer()` open hooks.
 */
abstract class BaseDeviceAgentService : LifecycleService() {

    companion object {
        private const val TAG = "BaseDeviceAgentService"
        private const val MAINTENANCE_INTERVAL_MS = 6 * 60 * 60 * 1000L // 6 hours
        private const val SENSING_PROFILE_DEBOUNCE_MS = 750L
        const val ACTION_INJECT_DEBUG_SPEECH = "com.trillboards.ctv.debug.action.INJECT_SPEECH"
        const val EXTRA_DEBUG_TRANSCRIPT = "transcript"
    }

    private data class SensingProfileCommandEnvelope(
        val commandId: String,
        val payload: JSONObject?,
        val orderKeyMs: Long
    )

    // ── Per-platform abstract surface ────────────────────────────────────────
    /** Per-platform identity — emitted in heartbeat metadata + capabilities. */
    protected abstract fun agentMetadata(): AgentMetadata

    /** Per-platform Intent action namespace (rebind broadcasts, maintenance, etc.). */
    protected abstract fun intentKeys(): IntentKeys

    /** Per-platform SharedPreferences key namespace. */
    protected abstract fun prefsKeys(): PrefsKeys

    /** Per-platform agent-core config (api/socket URLs, prefs name, overlay actions). */
    protected abstract fun coreConfig(): CoreAgentConfig

    /** Platform's MainActivity class (`Intent(this, mainActivityClass())`). */
    protected abstract fun mainActivityClass(): Class<out android.app.Activity>

    /** Platform's DeviceAdminReceiver class — used by KioskLockManager. */
    protected abstract fun deviceAdminReceiverClass(): Class<*>

    /** Foreground notification channel id (declared in the platform's Application). */
    protected abstract fun foregroundChannelId(): String

    /** Foreground notification id (declared in the platform's Application). */
    protected abstract fun foregroundNotificationId(): Int

    /**
     * Build the foreground notification. The platform supplies content title,
     * text, icon — agent-core just attaches the lifecycle.
     */
    protected abstract fun buildNotification(pendingIntent: PendingIntent): Notification

    /** Platform's AgentPowerAdapter (PowerController). */
    protected abstract fun powerAdapter(): AgentPowerAdapter

    /** Platform's command processor. Lazy-extends BaseCommandProcessor. */
    protected abstract val commandProcessor: BaseCommandProcessor

    /** True when the platform's MainActivity is the foreground activity. */
    protected abstract fun isMainActivityForeground(): Boolean

    // ── Open hooks (platform-optional behavior) ──────────────────────────────
    /**
     * Subclasses may return a content-state provider so the audience-sensing
     * pipeline can correlate ad observations with currently-rendered content.
     * Tablet returns `WebViewContentStateProvider`; Android-TV returns null.
     */
    protected open fun provideContentStateProvider(): com.trillboards.ctv.core.audience.ContentStateProvider? = null

    /** Player-reload recovery hook (e.g., tablet's TabletWebViewStabilityPolicy.HARD_RELOAD). */
    protected open fun reloadPlayer(reason: String) {}

    /** Player-recreate recovery hook (e.g., tablet's RECREATE_WEBVIEW or android-tv's AgentRecreatableActivity). */
    protected open fun recreatePlayer(reason: String) {}

    /** Subclasses may attach platform-specific socket-connected hooks (e.g., HybridRenderingManager.setSocketManager). */
    protected open fun onSocketConnected(socketManager: AgentSocketManager) {}

    /** Subclasses augment the heartbeat metadata map with platform-specific entries. */
    protected open fun extraHeartbeatMetadata(): Map<String, Any?> = emptyMap()

    /** True when this build allows debug-only side channels (ACTION_INJECT_DEBUG_SPEECH, etc.). */
    protected open fun isDebugBuild(): Boolean = false

    /**
     * Subclasses with extra startup work (CSI sensing pipeline, recovery
     * ladder, crash telemetry, identity-collector caches) override this to
     * launch their additional jobs after the base lifecycle is wired.
     */
    protected open fun onAfterBaseStarted() {}

    /**
     * Subclasses with extra teardown work override this. Called from
     * onDestroy() AND onTaskRemoved() before the base disposes shared state.
     */
    protected open fun onBeforeBaseStopped() {}

    /**
     * Subclasses may produce the final HeartbeatPayload. The base provides a
     * minimal default that mirrors the legacy android-tv-full DAS behavior;
     * tablet overrides this to also collect identity signals (BLE/GATT, mDNS,
     * SSDP, HTTP probe, Auracast, UWB, Channel Sounding, native sensors,
     * skip-reason aggregator).
     */
    protected open suspend fun buildHeartbeatPayload(): HeartbeatPayload {
        if (screenId.isNullOrEmpty()) {
            resolveScreenId()
        }

        val metrics = resources.displayMetrics
        val isDeviceOwner = kioskLockManager.isDeviceOwner()
        val meta = agentMetadata()

        val metadata = mutableMapOf<String, Any?>(
            "screenWidth" to metrics.widthPixels,
            "screenHeight" to metrics.heightPixels,
            "densityDpi" to metrics.densityDpi,
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "board" to Build.BOARD,
            "hardware" to Build.HARDWARE,
            "os" to "Android ${Build.VERSION.RELEASE}",
            "sdkInt" to Build.VERSION.SDK_INT,
            "uptimeSeconds" to (android.os.SystemClock.elapsedRealtime() / 1000L),
            "timestamp" to System.currentTimeMillis(),
            "blackout" to powerAdapter().currentBlackoutState(),
            "deviceType" to meta.deviceType,
            "agentType" to meta.agentType,
            "agentVersion" to meta.agentVersion,
            "isDeviceOwner" to isDeviceOwner,
            "sensingMode" to audienceSensing.getSensingMode(),
            "sensingUptime" to audienceSensing.getSensingUptime(),
            "trackedFaceCount" to audienceSensing.getTrackedFaceCount()
        )
        metadata.putAll(extraHeartbeatMetadata())

        val powerCapabilities = powerAdapter().capabilitySnapshot()
        val managementCapabilities = com.trillboards.ctv.core.models.DeviceCapabilityPayload.ManagementCapabilities(
            restart = true,
            reboot = isDeviceOwner,
            screenshot = true,
            clearCache = true,
            kioskMode = isDeviceOwner,
            installApk = isDeviceOwner,
            wipe = isDeviceOwner
        )

        val capabilities = powerCapabilities.copy(
            audienceSensing = audienceSensing.getAudienceSensingCapabilities(),
            managementCapabilities = managementCapabilities,
            mlCapabilities = audienceSensing.getMLCapabilities(),
            isDeviceOwner = isDeviceOwner,
            agentType = meta.deviceType,
            agentVersion = meta.agentVersion
        )

        return HeartbeatPayload(
            fingerprint = fingerprint,
            screenId = screenId,
            status = "active",
            metadata = metadata,
            capabilities = capabilities
        )
    }

    // ── Base lifecycle state ──────────────────────────────────────────────────
    protected val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    protected val prefs by lazy { getSharedPreferences(coreConfig().sharedPrefsName, Context.MODE_PRIVATE) }
    protected val fingerprint by lazy { DeviceIdentity.stableFingerprint(this, coreConfig().sharedPrefsName) }
    protected val apiClient by lazy {
        ApiClient(coreConfig()) {
            prefs.getString(prefsKeys().deviceToken, null)
        }
    }
    protected val socketManager by lazy { AgentSocketManager(coreConfig()) }
    protected val sensingProfileManager by lazy {
        SensingProfileManager(context = this, fingerprint = fingerprint)
    }
    protected val audienceSensing by lazy {
        AudienceSensingService(
            context = this,
            fingerprint = fingerprint,
            socketManager = socketManager,
            apiBaseUrl = coreConfig().apiBaseUrl,
            // Same SharedPreferences-backed lookup ApiClient uses; FrameCaptureManager rides
            // this through to add X-Device-Token on the /v2/earner/audience-analyze POST.
            deviceTokenProvider = { prefs.getString(prefsKeys().deviceToken, null) }
        ).apply {
            interactiveUiProvider = { isMainActivityForeground() }
        }
    }
    protected val telemetryScheduler by lazy {
        TelemetryScheduler(
            serviceScope,
            coreConfig(),
            apiClient,
            { buildHeartbeatPayload() },
            { token -> updateDeviceToken(token) },
            { resolvedId ->
                Log.i(TAG, "[binding] Screen resolved from heartbeat: $resolvedId")
                updateScreenId(resolvedId, null)
            },
            { commands ->
                Log.i(TAG, "[commands] Received ${commands.size} command(s) via heartbeat")
                commands.forEach { cmd ->
                    try {
                        handleDeviceCommand(cmd)
                    } catch (ex: Exception) {
                        Log.w(TAG, "[commands] Failed to process heartbeat command", ex)
                    }
                }
            }
        )
    }
    protected val adminComponent by lazy { ComponentName(this, deviceAdminReceiverClass()) }
    protected val kioskLockManager by lazy { KioskLockManager(this, adminComponent) }
    protected val localBroadcastManager: LocalBroadcastManager by lazy { LocalBroadcastManager.getInstance(this) }

    @Volatile
    protected var screenId: String? = null

    private var maintenanceJob: Job? = null
    private var maintenanceCycleCount = 0
    private var ensureMainActivityJob: Job? = null
    private var cameraRebindJob: Job? = null
    private var audioRebindJob: Job? = null
    private val sensingProfileCommandGate =
        LatestWinsCommandGate<SensingProfileCommandEnvelope> { it.orderKeyMs }
    private val sensingProfileDrainMutex = Mutex()
    private var sensingProfileDebounceJob: Job? = null

    // Adam's Fix #2 — coalescing rebind receiver. Reads EXTRA_REBIND_REASON +
    // EXTRA_REBIND_DELAY_MS extras; missing extras fall through to delayMs=0
    // (preserves android-tv's prior immediate-rebind behavior).
    private val sensorRebindReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val keys = intentKeys()
            val reason = intent?.getStringExtra(keys.extraRebindReason) ?: "broadcast"
            val delayMs = intent?.getLongExtra(keys.extraRebindDelayMs, 0L)
                ?.coerceIn(0L, 5_000L)
                ?: 0L
            when (intent?.action) {
                keys.rebindCamera -> scheduleCameraRebind(reason, delayMs)
                keys.rebindAudio -> scheduleAudioRebind(reason, delayMs)
            }
        }
    }

    // Optional debug-only speech-injection receiver (gated by isDebugBuild()).
    private val debugSpeechReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (!isDebugBuild()) return
            val transcript = intent?.getStringExtra(EXTRA_DEBUG_TRANSCRIPT) ?: return
            handleDebugSpeechTranscript(transcript)
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    override fun onCreate() {
        super.onCreate()
        screenId = prefs.getString(prefsKeys().screenId, null)
        startForegroundNotification()
        // Adam's Fix #1 — re-launch MainActivity if killed before service
        ensureMainActivityRunning("service_on_create")
        connectSocket()
        telemetryScheduler.start()
        serviceScope.launch { resolveScreenId() }

        val sensorRebindFilter = IntentFilter().apply {
            addAction(intentKeys().rebindCamera)
            addAction(intentKeys().rebindAudio)
        }
        localBroadcastManager.registerReceiver(sensorRebindReceiver, sensorRebindFilter)

        if (isDebugBuild()) {
            val debugFilter = IntentFilter().apply { addAction(ACTION_INJECT_DEBUG_SPEECH) }
            localBroadcastManager.registerReceiver(debugSpeechReceiver, debugFilter)
        }

        startAudienceSensing()
        startMaintenanceCycle()
        onAfterBaseStarted()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        ensureMainActivityRunning("service_on_start_command")
        return START_STICKY
    }

    override fun onDestroy() {
        onBeforeBaseStopped()
        cameraRebindJob?.cancel()
        audioRebindJob?.cancel()
        maintenanceJob?.cancel()
        ensureMainActivityJob?.cancel()
        try { localBroadcastManager.unregisterReceiver(sensorRebindReceiver) } catch (_: Exception) {}
        if (isDebugBuild()) {
            try { localBroadcastManager.unregisterReceiver(debugSpeechReceiver) } catch (_: Exception) {}
        }
        audienceSensing.stop()
        telemetryScheduler.stop()
        socketManager.disconnect()
        serviceScope.cancel()
        super.onDestroy()
    }

    /**
     * Called when user swipes app from recents. CRITICAL: onDestroy() is NOT
     * called when the app is removed from recents, so we explicitly disconnect
     * here to avoid a phantom "online" status server-side.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        Log.i(TAG, "App removed from recents - stopping service")
        onBeforeBaseStopped()
        maintenanceJob?.cancel()
        socketManager.disconnect()
        audienceSensing.stop()
        telemetryScheduler.stop()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        super.onTaskRemoved(rootIntent)
    }

    // ── Adam's Fix #1 ─────────────────────────────────────────────────────────
    /**
     * Re-launch the platform's MainActivity if it was killed by the OS while
     * the service is still running. RockChip Gocast STBs aggressively kill
     * the activity after the camera-permission dialog opens, leaving the
     * service running in degraded mode forever (no Moments, no Sensing).
     */
    protected fun ensureMainActivityRunning(reason: String) {
        if (isMainActivityForeground()) return

        Log.w(TAG, "MainActivity missing while service is active ($reason) — relaunching kiosk UI")
        val intent = Intent(this, mainActivityClass()).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }

        try {
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to relaunch MainActivity after $reason: ${e.message}", e)
            return
        }

        ensureMainActivityJob?.cancel()
        ensureMainActivityJob = serviceScope.launch {
            repeat(10) {
                delay(1_000)
                if (isMainActivityForeground()) {
                    reapplyCurrentSensingProfile("main_activity_restored_$reason")
                    return@launch
                }
            }
            Log.w(TAG, "MainActivity did not restore within 10s after $reason")
        }
    }

    private fun reapplyCurrentSensingProfile(reason: String) {
        val profile = sensingProfileManager.getActiveProfile() ?: sensingProfileManager.loadPersistedProfile()
        if (profile == null) {
            Log.d(TAG, "No sensing profile to reapply after $reason")
            return
        }

        Log.i(TAG, "Reapplying sensing profile after $reason: ${profile.profileName}")
        audienceSensing.setActiveProfileId(profile.profileId)
        audienceSensing.setActiveProgramSpec(profile.programSpecJson, profile.programSpecVersion)
        audienceSensing.setVlmPrompt(profile.vlmPrompt)
        audienceSensing.applyProfileToInference(profile.models)
    }

    // ── Adam's Fix #2 ─────────────────────────────────────────────────────────
    /**
     * Coalesces camera-rebind requests so RockChip cameras don't flap. When
     * the receiver supplies `EXTRA_REBIND_DELAY_MS`, we delay the rebind by
     * that amount and cancel any in-flight pending rebind. Missing extras
     * fall through to `delayMs=0` (preserves android-tv's prior immediate
     * behavior).
     */
    protected fun scheduleCameraRebind(reason: String, delayMs: Long) {
        cameraRebindJob?.cancel()
        cameraRebindJob = serviceScope.launch {
            if (delayMs > 0) {
                Log.i(TAG, "Scheduling camera rebind in ${delayMs}ms (reason=$reason)")
                delay(delayMs)
            } else {
                Log.i(TAG, "Scheduling camera rebind immediately (reason=$reason)")
            }

            withContext(Dispatchers.Main.immediate) {
                if (!lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) {
                    Log.w(TAG, "Skipping camera rebind while service is not started (reason=$reason)")
                    return@withContext
                }
                Log.i(TAG, "Executing camera rebind (reason=$reason)")
                audienceSensing.rebindCamera(this@BaseDeviceAgentService)
            }
        }
    }

    /** Coalesces audio-rebind requests. Same coalescer pattern as camera. */
    protected fun scheduleAudioRebind(reason: String, delayMs: Long) {
        audioRebindJob?.cancel()
        audioRebindJob = serviceScope.launch {
            if (delayMs > 0) {
                Log.i(TAG, "Scheduling audio rebind in ${delayMs}ms (reason=$reason)")
                delay(delayMs)
            } else {
                Log.i(TAG, "Scheduling audio rebind immediately (reason=$reason)")
            }
            Log.i(TAG, "Executing audio rebind (reason=$reason)")
            audienceSensing.rebindAudio()
        }
    }

    // ── Audience sensing ─────────────────────────────────────────────────────
    private fun startAudienceSensing() {
        val restoredProfile = sensingProfileManager.loadPersistedProfile()

        screenId?.let { audienceSensing.setScreenId(it) }

        commandProcessor.onBrandListUpdate = { brands ->
            audienceSensing.updateBrandList(brands)
        }

        sensingProfileManager.addProfileListener { profile ->
            Log.i(TAG, "Sensing profile changed: ${profile.profileName} (${profile.models.size} models)")
            audienceSensing.setActiveProfileId(profile.profileId)
            audienceSensing.setActiveProgramSpec(profile.programSpecJson, profile.programSpecVersion)
            audienceSensing.setVlmPrompt(profile.vlmPrompt)
            audienceSensing.applyProfileToInference(profile.models)
        }

        restoredProfile?.let { persisted ->
            Log.i(TAG, "Restoring persisted sensing profile: ${persisted.profileName}")
            audienceSensing.setActiveProfileId(persisted.profileId)
            audienceSensing.setActiveProgramSpec(persisted.programSpecJson, persisted.programSpecVersion)
            audienceSensing.setVlmPrompt(persisted.vlmPrompt)
            audienceSensing.applyProfileToInference(persisted.models)
        }

        provideContentStateProvider()?.let { provider ->
            audienceSensing.setContentStateProvider(provider)
        }

        val hasCameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        val hasAudioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

        if (!hasCameraPermission && !hasAudioPermission) {
            Log.w(TAG, "No camera or audio permissions - audience sensing disabled")
            return
        }

        // Adam's Fix #3 — explicit demographics + speech intelligence flag passing.
        // Eliminates the foreground-service `camera`-type race where android-tv
        // requests FOREGROUND_SERVICE_TYPE_CAMERA but audienceSensing.start()
        // didn't actually open the camera (because the SensingConfig data class
        // default skewed wrong for android-tv).
        audienceSensing.start(this, SensingConfig(
            enableFaceDetection = hasCameraPermission,
            enableAudioClassification = hasAudioPermission,
            enableSpeechIntelligence = hasAudioPermission,
            enableDemographicsCapture = hasCameraPermission
        ))

        Log.i(TAG, "Audience sensing initialized (camera=$hasCameraPermission, audio=$hasAudioPermission)")
    }

    private fun startMaintenanceCycle() {
        maintenanceJob = serviceScope.launch {
            while (isActive) {
                delay(MAINTENANCE_INTERVAL_MS)
                maintenanceCycleCount++
                Log.i(TAG, "Maintenance cycle #$maintenanceCycleCount triggered - broadcasting to MainActivity")
                localBroadcastManager.sendBroadcast(Intent(intentKeys().maintenanceCycle))

                if (maintenanceCycleCount % 4 == 0) {
                    Log.i(TAG, "24h sensing restart: stopping audience sensing...")
                    audienceSensing.stop()
                    delay(2_000)
                    Log.i(TAG, "24h sensing restart: restarting audience sensing...")
                    startAudienceSensing()
                }
            }
        }
    }

    // ── Sockets / commands ──────────────────────────────────────────────────
    private fun connectSocket() {
        socketManager.connect(
            fingerprint,
            object : AgentSocketManager.Listener {
                override fun onConnect(socketId: String?) {
                    Log.i(TAG, "Socket connected: $socketId")
                    onSocketConnected(socketManager)
                }

                override fun onDisconnect() {
                    Log.i(TAG, "Socket disconnected")
                }

                override fun onError(args: Array<Any?>) {
                    Log.w(TAG, "Socket error: ${args.joinToString()}")
                }

                override fun onReconnected() {
                    Log.i(TAG, "Socket reconnected — triggering buffered signal flush")
                    audienceSensing.flushBufferedSignals()
                }

                override fun onPrivateMessage(payload: JSONObject) {
                    handlePrivateMessage(payload)
                }

                override fun onDeviceCommand(payload: JSONObject) {
                    handleDeviceCommand(payload)
                }

                override fun onScreenBinding(payload: JSONObject) {
                    handleScreenBinding(payload)
                }
            },
            prefs.getString(prefsKeys().deviceToken, null)
        )
    }

    private fun handlePrivateMessage(json: JSONObject) {
        val data = json.optJSONObject("data")
        val detectedId = json.optString("screenId", data?.optString("_id", null))
        if (!detectedId.isNullOrEmpty()) {
            updateScreenId(detectedId)
        }
    }

    private fun handleDeviceCommand(json: JSONObject) {
        val commandId = json.optString("commandId", json.optString("id", ""))
        val command = json.optString("command")
        val commandPayload = json.optJSONObject("payload")
        if (command.isNullOrEmpty()) {
            Log.w(TAG, "Invalid command payload: $json")
            return
        }

        if (command == "updateSensingProfile") {
            enqueueSensingProfileCommand(
                commandId = commandId,
                payload = commandPayload?.let { JSONObject(it.toString()) },
                queuedAt = json.optString("queuedAt", null)
            )
            return
        }

        serviceScope.launch {
            val result = commandProcessor.handle(command, commandPayload)
            emitCommandAck(commandId, result.status, result.details)
        }
    }

    private fun enqueueSensingProfileCommand(
        commandId: String,
        payload: JSONObject?,
        queuedAt: String?
    ) {
        val orderKeyMs = queuedAt
            ?.takeIf { it.isNotBlank() }
            ?.let {
                runCatching { Instant.parse(it).toEpochMilli() }.getOrNull()
            }
            ?: System.currentTimeMillis()

        val envelope = SensingProfileCommandEnvelope(
            commandId = commandId,
            payload = payload,
            orderKeyMs = orderKeyMs
        )

        val offer = sensingProfileCommandGate.offer(envelope)
        when (offer.status) {
            LatestWinsCommandGate.OfferStatus.REPLACED_PENDING -> {
                offer.replacedCommand?.let { replaced ->
                    emitCommandAck(
                        replaced.commandId,
                        "acked",
                        mapOf(
                            "superseded" to true,
                            "supersededBy" to commandId,
                            "reason" to "newer_sensing_profile_command"
                        )
                    )
                }
            }

            LatestWinsCommandGate.OfferStatus.STALE_ALREADY_APPLIED,
            LatestWinsCommandGate.OfferStatus.STALE_PENDING -> {
                emitCommandAck(
                    commandId,
                    "acked",
                    mapOf(
                        "skipped" to true,
                        "reason" to offer.status.name.lowercase()
                    )
                )
                return
            }

            LatestWinsCommandGate.OfferStatus.ENQUEUED -> Unit
        }

        scheduleSensingProfileDrain()
    }

    private fun scheduleSensingProfileDrain() {
        sensingProfileDebounceJob?.cancel()
        sensingProfileDebounceJob = serviceScope.launch {
            delay(SENSING_PROFILE_DEBOUNCE_MS)
            drainSensingProfileCommands()
        }
    }

    private suspend fun drainSensingProfileCommands() {
        if (!sensingProfileDrainMutex.tryLock()) {
            return
        }

        try {
            while (true) {
                val next = sensingProfileCommandGate.takeNext() ?: break
                val result = commandProcessor.handle("updateSensingProfile", next.payload)
                sensingProfileCommandGate.markApplied(next)
                emitCommandAck(next.commandId, result.status, result.details)
            }
        } finally {
            sensingProfileDrainMutex.unlock()
            if (sensingProfileCommandGate.hasPending()) {
                serviceScope.launch {
                    drainSensingProfileCommands()
                }
            }
        }
    }

    private fun emitCommandAck(
        commandId: String,
        status: String,
        details: Map<String, Any?>
    ) {
        val sanitized = details.filterValues { it != null }.mapValues { (_, v) -> v as Any }
        val ack = JSONObject().apply {
            if (commandId.isNotEmpty()) {
                put("commandId", commandId)
            }
            put("status", status)
            put("details", JSONObject(sanitized))
        }
        socketManager.emitCommandAck(ack)
        if (commandId.isNotEmpty()) {
            serviceScope.launch {
                apiClient.sendCommandAck(
                    commandId = commandId,
                    status = status,
                    result = JSONObject(sanitized),
                    screenId = screenId
                )
            }
        }
    }

    private fun handleScreenBinding(json: JSONObject) {
        val newScreenId = json.optString("screenId", "")
        val action = json.optString("action", "paired")
        val venueType = json.optString("venueType", null)

        when (action) {
            "paired" -> {
                if (newScreenId.isNotEmpty() && newScreenId != screenId) {
                    Log.i(TAG, "[screen_binding] Screen paired: $screenId -> $newScreenId")
                    updateScreenId(newScreenId, venueType)
                }
            }
            "deleted" -> {
                Log.i(TAG, "[screen_binding] Screen deleted, clearing screenId")
                screenId = null
                prefs.edit().remove(prefsKeys().screenId).apply()
            }
        }
    }

    // ── Screen / token resolution ───────────────────────────────────────────
    private suspend fun resolveScreenId() {
        val hasToken = !prefs.getString(prefsKeys().deviceToken, null).isNullOrEmpty()
        if (!screenId.isNullOrEmpty() && hasToken) return
        val resolved = apiClient.fetchScreenResolution(fingerprint)
        if (resolved != null) {
            updateScreenId(resolved.screenId, resolved.venueType)
            updateDeviceToken(resolved.deviceToken)
        }
    }

    protected fun updateScreenId(id: String, venueType: String? = null) {
        screenId = id
        prefs.edit().putString(prefsKeys().screenId, id).apply()
        audienceSensing.setScreenId(id)
        if (venueType != null) {
            audienceSensing.setVenueType(venueType)
        }
    }

    protected fun updateDeviceToken(token: String?) {
        if (token.isNullOrBlank()) return
        val current = prefs.getString(prefsKeys().deviceToken, null)
        if (current == token) return
        prefs.edit().putString(prefsKeys().deviceToken, token).apply()
        socketManager.disconnect()
        connectSocket()
    }

    // ── Foreground notification ─────────────────────────────────────────────
    private fun startForegroundNotification() {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, mainActivityClass()),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification = buildNotification(pendingIntent)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            var serviceType = 0

            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
                serviceType = serviceType or ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED) {
                serviceType = serviceType or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            }

            if (serviceType == 0) {
                serviceType = ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                Log.w(TAG, "No camera/microphone permissions — using SPECIAL_USE foreground service type")
            }

            try {
                startForeground(foregroundNotificationId(), notification, serviceType)
            } catch (e: Exception) {
                Log.e(TAG, "startForeground failed with type $serviceType, retrying with SPECIAL_USE", e)
                startForeground(
                    foregroundNotificationId(),
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                )
            }
        } else {
            startForeground(foregroundNotificationId(), notification)
        }
    }

    // ── Debug-only side channel ─────────────────────────────────────────────
    /**
     * Inject a synthetic transcript into the audience-sensing pipeline for
     * dev / smoke-test use. Gated by `isDebugBuild()`. Only the platform's
     * debug build registers the receiver in `onCreate()`.
     */
    private fun handleDebugSpeechTranscript(transcript: String): Boolean {
        if (!isDebugBuild()) {
            Log.w(TAG, "[DebugSpeech] Ignoring injection on non-debug build")
            return false
        }
        val normalized = transcript.trim()
        if (normalized.isBlank()) {
            Log.w(TAG, "[DebugSpeech] Ignoring empty transcript injection")
            return false
        }
        val insights = InsightExtractor.extract(normalized)
        Log.i(
            TAG,
            "[DebugSpeech] Injecting transcript (${normalized.length} chars) -> " +
                "intent=${insights.purchaseIntent.name}, stage=${insights.purchaseJourney.stage}"
        )
        audienceSensing.injectDebugSpeechInsights(insights)
        return true
    }
}
