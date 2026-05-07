package com.trillboards.ctv.core.stability

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.util.Log
import androidx.core.content.ContextCompat
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.stability.SubsystemHealthRegistry.Subsystem
import com.trillboards.ctv.core.stability.SubsystemHealthRegistry.SubsystemStatus
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Autonomous recovery ladder for self-healing at 10K unattended screens.
 *
 * Each subsystem has its own recovery path with deterministic triggers,
 * cooldowns, retry limits, and escalation rules. No magic — every action
 * is logged and observable via the socket heartbeat payload.
 *
 * Recovery Levels:
 *   1. Network Drop Recovery — infinite reconnect with exp backoff + jitter
 *   2. Player Stuck Recovery — WebView reload → recreate → service restart
 *   3. Camera/Mic Failure — unbind/rebind → restart sensing service
 *   4. Memory Pressure — handled by MemoryAttenuationManager (with restoration)
 *   5. Full Service Restart — stop + restart DeviceAgentService (3x per 24h)
 *   6. OTA Rollback — server-initiated rollback to previous APK version
 */
class RecoveryLadder(
    private val context: Context,
    private val healthRegistry: SubsystemHealthRegistry,
    private val maxRecoveryLevel: Int = SensingConfig.get().recovery.maxRecoveryLevel
) {

    companion object {
        private const val TAG = "RecoveryLadder"
        private const val PREFS_NAME = "recovery_ladder"
        private val MAX_SERVICE_RESTARTS_PER_DAY: Int get() = SensingConfig.get().recovery.maxServiceRestartsPerDay
        private const val DAY_MS = 24 * 60 * 60 * 1000L
    }

    /**
     * Callback interface for recovery actions that require interaction
     * with the hosting Activity or Service.
     */
    interface RecoveryCallback {
        /** Level 2: Reload the WebView player */
        fun onReloadPlayer()
        /** Level 2b: Destroy and recreate the WebView instance */
        fun onRecreatePlayer()
        /** Level 3: Rebind camera use cases (unbind all, wait, rebind) */
        fun onRebindCamera()
        /** Level 3: Restart the AudienceSensingService */
        fun onRestartSensingService()
        /** Level 3: Restart audio recording */
        fun onRebindMicrophone()
        /** Level 5: Restart the entire DeviceAgentService */
        fun onRestartDeviceService()
        /** Level 6: Install a rollback APK from URL */
        fun onInstallRollbackApk(apkUrl: String)
        /** Emit a recovery event to the backend via socket */
        fun onEmitRecoveryEvent(event: String, details: Map<String, Any>)
    }

    private var callback: RecoveryCallback? = null
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Per-subsystem retry state
    private data class RetryState(
        val retryCount: AtomicInteger = AtomicInteger(0),
        @Volatile var lastAttemptMs: Long = 0,
        @Volatile var currentLevel: Int = 0
    )

    private val retryStates = ConcurrentHashMap<Subsystem, RetryState>()

    // Service restart tracking (Level 5)
    private val serviceRestartTimestamps = mutableListOf<Long>()

    fun setCallback(callback: RecoveryCallback) {
        this.callback = callback
    }

    /**
     * Check all subsystems and trigger recovery for any that are failed.
     * Call this on a regular interval (e.g., every 30s from the heartbeat loop).
     */
    fun checkAndRecover() {
        if (maxRecoveryLevel < 1) return

        for (subsystem in Subsystem.values()) {
            if (healthRegistry.isSubsystemFailed(subsystem)) {
                handleFailure(subsystem)
            } else {
                // Reset retry state when subsystem is healthy
                retryStates[subsystem]?.let {
                    if (it.retryCount.get() > 0) {
                        Log.i(TAG, "${subsystem.name} recovered — resetting retry state")
                        it.retryCount.set(0)
                        it.currentLevel = 0
                    }
                }
            }
        }
    }

    private fun handleFailure(subsystem: Subsystem) {
        val state = retryStates.getOrPut(subsystem) { RetryState() }
        val health = healthRegistry.getHealth(subsystem)

        when (subsystem) {
            Subsystem.SOCKET -> handleSocketFailure(state)
            Subsystem.PLAYER -> handlePlayerFailure(state)
            Subsystem.CAMERA -> handleCameraFailure(state)
            Subsystem.MICROPHONE -> handleMicrophoneFailure(state)
            Subsystem.ML_PIPELINE -> handleMlPipelineFailure(state)
            Subsystem.MEMORY -> {} // Handled by MemoryAttenuationManager
        }
    }

    // ---- Level 1: Network Drop Recovery ----
    // Socket.io handles reconnection internally (Int.MAX_VALUE attempts with exp backoff).
    // RecoveryLadder just logs and emits alerts if disconnect persists > 5min.

    private fun handleSocketFailure(state: RetryState) {
        val health = healthRegistry.getHealth(Subsystem.SOCKET) ?: return
        if (health.ageMs > SensingConfig.get().recovery.socketDisconnectAlertMs) {
            Log.w(TAG, "Socket disconnected for ${health.ageMs / 1000}s — socket.io handling reconnection")
            // Socket.io handles reconnection with Int.MAX_VALUE attempts.
            // We just track it for fleet visibility.
        }
    }

    // ---- Level 2: Player Stuck Recovery ----

    private fun handlePlayerFailure(state: RetryState) {
        if (maxRecoveryLevel < 2) return
        val now = System.currentTimeMillis()
        val cfg = SensingConfig.get().recovery

        // Cooldown: reload vs recreate
        val cooldownMs = if (state.currentLevel <= 2) cfg.playerReloadCooldownMs else cfg.playerRecreateCooldownMs
        if (now - state.lastAttemptMs < cooldownMs) return

        state.lastAttemptMs = now
        val retries = state.retryCount.incrementAndGet()

        when {
            retries <= cfg.playerReloadMaxRetries -> {
                // Level 2a: Reload WebView
                Log.w(TAG, "Player stuck — reload attempt $retries/${cfg.playerReloadMaxRetries}")
                state.currentLevel = 2
                callback?.onReloadPlayer()
                emitRecoveryEvent("playerReload", mapOf("attempt" to retries))
            }
            retries <= cfg.playerRecreateMaxRetries -> {
                // Level 2b: Destroy and recreate WebView
                Log.w(TAG, "Player reload failed — recreate attempt ${retries - cfg.playerReloadMaxRetries}/${cfg.playerRecreateMaxRetries - cfg.playerReloadMaxRetries}")
                state.currentLevel = 3
                callback?.onRecreatePlayer()
                emitRecoveryEvent("playerRecreate", mapOf("attempt" to retries - cfg.playerReloadMaxRetries))
            }
            else -> {
                // Escalate to Level 5: Full service restart
                Log.e(TAG, "Player recovery exhausted — escalating to service restart")
                state.retryCount.set(0)
                state.currentLevel = 0
                attemptServiceRestart()
            }
        }
    }

    // ---- Level 3: Camera/Mic Failure Recovery ----

    private fun handleCameraFailure(state: RetryState) {
        if (maxRecoveryLevel < 3) return
        val now = System.currentTimeMillis()
        val cfg = SensingConfig.get().recovery
        if (now - state.lastAttemptMs < cfg.cameraMicCooldownMs) return

        state.lastAttemptMs = now
        val retries = state.retryCount.incrementAndGet()

        when {
            retries <= cfg.cameraMicMaxRetries -> {
                // Unbind all CameraX use cases, wait 2s, rebind
                Log.w(TAG, "Camera failure — rebind attempt $retries/3")
                callback?.onRebindCamera()
                emitRecoveryEvent("cameraRebind", mapOf("attempt" to retries))
            }
            else -> {
                // Restart AudienceSensingService
                Log.w(TAG, "Camera rebind failed — restarting sensing service")
                state.retryCount.set(0)
                callback?.onRestartSensingService()
                emitRecoveryEvent("cameraFailed", mapOf(
                    "action" to "restartSensingService",
                    "totalAttempts" to retries
                ))
            }
        }
    }

    private fun handleMicrophoneFailure(state: RetryState) {
        if (maxRecoveryLevel < 3) return
        val now = System.currentTimeMillis()
        val cfg = SensingConfig.get().recovery
        if (now - state.lastAttemptMs < cfg.cameraMicCooldownMs) return

        state.lastAttemptMs = now
        val retries = state.retryCount.incrementAndGet()

        when {
            retries <= cfg.cameraMicMaxRetries -> {
                Log.w(TAG, "Microphone failure — rebind attempt $retries/3")
                callback?.onRebindMicrophone()
                emitRecoveryEvent("micRebind", mapOf("attempt" to retries))
            }
            else -> {
                Log.w(TAG, "Mic rebind failed — restarting sensing service")
                state.retryCount.set(0)
                callback?.onRestartSensingService()
                emitRecoveryEvent("micFailed", mapOf(
                    "action" to "restartSensingService",
                    "totalAttempts" to retries
                ))
            }
        }
    }

    // ---- Level 4: Memory Pressure ----
    // Handled by MemoryAttenuationManager + DeviceCapabilityMatrix.
    // RecoveryLadder tracks attenuation event count for deviceUnderspecced alerting.

    private fun handleMlPipelineFailure(state: RetryState) {
        // ML pipeline failures are typically caused by memory pressure.
        // MemoryAttenuationManager handles the model shedding/restoration.
        // We just count transitions and alert if excessive.
        val health = healthRegistry.getHealth(Subsystem.ML_PIPELINE) ?: return
        if (health.status == SubsystemStatus.OFFLINE) {
            Log.w(TAG, "ML pipeline offline — memory attenuation handling")
        }
    }

    // ---- Level 5: Full Service Restart ----

    private fun attemptServiceRestart() {
        if (maxRecoveryLevel < 5) {
            Log.w(TAG, "Service restart not available at maxRecoveryLevel=$maxRecoveryLevel")
            return
        }

        val now = System.currentTimeMillis()

        // Clean up timestamps older than 24h
        serviceRestartTimestamps.removeAll { now - it > DAY_MS }

        if (serviceRestartTimestamps.size >= MAX_SERVICE_RESTARTS_PER_DAY) {
            Log.e(TAG, "Max service restarts (${MAX_SERVICE_RESTARTS_PER_DAY}) in 24h exceeded — emitting deviceNeedsReboot")
            emitRecoveryEvent("deviceNeedsReboot", mapOf(
                "restartsIn24h" to serviceRestartTimestamps.size,
                "reason" to "recovery_exhausted"
            ))
            return
        }

        serviceRestartTimestamps.add(now)
        Log.w(TAG, "Level 5: Full service restart (${serviceRestartTimestamps.size}/$MAX_SERVICE_RESTARTS_PER_DAY in 24h)")

        callback?.onRestartDeviceService()
        emitRecoveryEvent("serviceRestart", mapOf(
            "restartsIn24h" to serviceRestartTimestamps.size
        ))
    }

    // ---- Level 6: OTA Rollback ----
    // Server-initiated: backend detects health score drop after OTA update and sends
    // device.installApk command with previous APK version URL.

    /**
     * Called when the backend sends an OTA rollback command.
     */
    fun handleOtaRollback(apkUrl: String) {
        if (maxRecoveryLevel < 6) {
            Log.w(TAG, "OTA rollback not available at maxRecoveryLevel=$maxRecoveryLevel")
            return
        }

        Log.w(TAG, "Level 6: OTA rollback initiated — downloading previous APK")
        callback?.onInstallRollbackApk(apkUrl)
        emitRecoveryEvent("otaRollback", mapOf("apkUrl" to apkUrl))
    }

    // ---- Attenuation Event Tracking ----

    private val attenuationEventsToday = AtomicInteger(0)
    private var attenuationDayStart = System.currentTimeMillis()

    /**
     * Called by MemoryAttenuationManager when a tier transition occurs.
     * Tracks daily count and emits deviceUnderspecced if > 20 transitions/day.
     */
    fun onAttenuationEvent(fromTier: String, toTier: String) {
        val now = System.currentTimeMillis()
        if (now - attenuationDayStart > DAY_MS) {
            attenuationEventsToday.set(0)
            attenuationDayStart = now
        }

        val count = attenuationEventsToday.incrementAndGet()
        Log.d(TAG, "Attenuation event: $fromTier → $toTier (count today: $count)")

        val maxTransitions = SensingConfig.get().recovery.maxDailyAttenuationTransitions
        if (count > maxTransitions) {
            emitRecoveryEvent("deviceUnderspecced", mapOf(
                "attenuationEventsToday" to count,
                "threshold" to maxTransitions
            ))
        }
    }

    private fun emitRecoveryEvent(event: String, details: Map<String, Any>) {
        callback?.onEmitRecoveryEvent(event, details)
    }

    fun destroy() {
        scope.cancel()
    }
}
