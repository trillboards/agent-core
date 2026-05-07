package com.trillboards.ctv.core.service

import android.util.Log
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.models.HeartbeatPayload
import com.trillboards.ctv.core.net.ApiClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class TelemetryScheduler(
    private val scope: CoroutineScope,
    private val config: AgentConfig,
    private val apiClient: ApiClient,
    private val snapshotProvider: suspend () -> HeartbeatPayload,
    private val onSocketTokenReceived: ((String) -> Unit)? = null,
    private val onScreenIdResolved: ((String) -> Unit)? = null,
    private val onCommandsReceived: ((List<org.json.JSONObject>) -> Unit)? = null
) {
    private var job: Job? = null

    // Binding backoff: after BINDING_BACKOFF_THRESHOLD consecutive failures on
    // binding heartbeats (no screenId), slow down to BINDING_BACKOFF_INTERVAL_MS
    // to reduce 400-response churn while the device waits for pairing.
    private var bindingFailCount = 0

    fun start() {
        if (job?.isActive == true) return
        job = scope.launch {
            while (isActive) {
                try {
                    val payload = snapshotProvider()

                    // Skip heartbeat if display is off - let server TTL timeout handle offline marking.
                    // This ensures screens show "offline" when TV is powered off instead of staying
                    // in "standby" state forever (since heartbeats would keep refreshing the TTL).
                    val powerState = payload.metadata["powerState"] as? String
                    if (powerState == "off" || powerState == "standby") {
                        Log.d(TAG, "Display off (powerState=$powerState), skipping heartbeat")
                        delay(config.heartbeatIntervalMs)
                        continue
                    }

                    val isBindingHeartbeat = payload.screenId.isNullOrEmpty()

                    // Always send heartbeat, even without screenId. The server resolves
                    // screenId from the fingerprint via fp2scr Redis mapping and returns it
                    // in the heartbeat response. Blocking heartbeats on screenId creates a
                    // chicken-and-egg: no heartbeat → no server response → no screenId.
                    if (isBindingHeartbeat) {
                        Log.i(TAG, "Sending binding heartbeat (screenId not yet resolved, fingerprint=${payload.fingerprint?.take(8)}..., failCount=$bindingFailCount)")
                    }
                    val response = apiClient.sendHeartbeat(payload)
                    if (response != null) {
                        response.socketToken?.let { onSocketTokenReceived?.invoke(it) }
                        // Server returns the authoritative screenId from fp2scr lookup.
                        // If the device didn't have a screenId, this resolves the binding.
                        if (isBindingHeartbeat && !response.screenId.isNullOrEmpty()) {
                            Log.i(TAG, "Screen binding resolved from heartbeat response: ${response.screenId}")
                            bindingFailCount = 0
                            onScreenIdResolved?.invoke(response.screenId!!)
                        } else if (isBindingHeartbeat && response.screenId.isNullOrEmpty()) {
                            // Server responded but no screenId — device is not yet paired
                            bindingFailCount++
                        }

                        // Process device commands delivered via HTTP heartbeat response.
                        // Eliminates socket dependency — commands arrive reliably every 30s.
                        if (!response.commands.isNullOrEmpty()) {
                            Log.i(TAG, "Received ${response.commands.size} command(s) via heartbeat")
                            onCommandsReceived?.invoke(response.commands)
                        }
                    } else if (isBindingHeartbeat) {
                        // null response means HTTP error (likely 400 for unbound device)
                        bindingFailCount++
                    }
                } catch (ex: Exception) {
                    Log.w(TAG, "Heartbeat failed", ex)
                }

                // Use longer interval when binding heartbeats keep failing
                val interval = if (bindingFailCount >= BINDING_BACKOFF_THRESHOLD) {
                    BINDING_BACKOFF_INTERVAL_MS
                } else {
                    config.heartbeatIntervalMs
                }
                delay(interval)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }

    companion object {
        private const val TAG = "TelemetryScheduler"
        // After this many consecutive binding failures, slow down to reduce 400 churn
        private const val BINDING_BACKOFF_THRESHOLD = 10
        // Slowed-down interval for unbound devices (60s instead of default 30s)
        private const val BINDING_BACKOFF_INTERVAL_MS = 60_000L
    }
}
