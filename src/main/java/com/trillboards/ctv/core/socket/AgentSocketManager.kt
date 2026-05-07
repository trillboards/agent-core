package com.trillboards.ctv.core.socket

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.VASWeightConfig
import io.socket.client.IO
import io.socket.client.Socket
import org.json.JSONObject
import java.net.URLEncoder

class AgentSocketManager(private val config: AgentConfig) {

    companion object {
        private const val TAG = "AgentSocketManager"
        /** How often (ms) to verify socket health and force-reconnect if stuck. */
        private const val RECONNECT_HEALTH_CHECK_INTERVAL_MS = 90_000L  // 90 seconds
        /** If disconnected longer than this, force a fresh reconnect instead of waiting for auto-reconnect. */
        private const val FORCE_RECONNECT_AFTER_MS = 120_000L  // 2 minutes
    }

    private var socket: Socket? = null
    private var currentListener: Listener? = null
    private var currentFingerprint: String? = null
    private var currentDeviceToken: String? = null

    // Reconnection health check state
    private val handler = Handler(Looper.getMainLooper())
    @Volatile private var disconnectedSinceMs: Long = 0L
    @Volatile private var reconnectAttemptCount: Int = 0
    @Volatile private var isIntentionalDisconnect = false

    private val reconnectHealthCheck = object : Runnable {
        override fun run() {
            try {
                checkReconnectionHealth()
            } catch (e: Exception) {
                Log.e(TAG, "[ReconnectCheck] Error in health check: ${e.message}", e)
            }
            // Re-schedule as long as we haven't intentionally disconnected
            if (!isIntentionalDisconnect) {
                handler.postDelayed(this, RECONNECT_HEALTH_CHECK_INTERVAL_MS)
            }
        }
    }

    fun connect(fingerprint: String, listener: Listener, deviceToken: String? = null) {
        Log.i(TAG, "Connecting to ${config.socketUrl} with fingerprint=$fingerprint")

        // Store references for reconnection
        currentListener = listener
        currentFingerprint = fingerprint
        currentDeviceToken = deviceToken
        isIntentionalDisconnect = false
        disconnectedSinceMs = 0L
        reconnectAttemptCount = 0

        val options = IO.Options().apply {
            val tokenQuery = deviceToken
                ?.takeIf { it.isNotBlank() }
                ?.let { "&device_token=${URLEncoder.encode(it, "UTF-8")}" }
                ?: ""
            query = "fingerprint=${URLEncoder.encode(fingerprint, "UTF-8")}$tokenQuery"
            // Force WebSocket transport — skip HTTP long-polling handshake entirely.
            // The ALB (`chat.trillboards.com`) has sticky sessions for polling, but the
            // Java Socket.IO client doesn't persist AWSALB cookies between requests.
            // Without cookie persistence, the second polling POST routes to a different
            // ECS Fargate task which has no record of the session → "xhr post error".
            // WebSocket is a single persistent TCP connection pinned to one target after
            // the HTTP upgrade, so sticky sessions aren't needed.
            transports = arrayOf("websocket")
            reconnection = true
            reconnectionAttempts = Int.MAX_VALUE
            reconnectionDelay = 3_000L
            reconnectionDelayMax = 60_000L
        }

        socket = IO.socket(config.socketUrl, options).apply {
            on(Socket.EVENT_CONNECT) {
                val prevDisconnectDuration = if (disconnectedSinceMs > 0) {
                    (System.currentTimeMillis() - disconnectedSinceMs) / 1000
                } else 0L
                disconnectedSinceMs = 0L
                reconnectAttemptCount = 0
                Log.i(TAG, "Socket connected! ID=${id()}" +
                        if (prevDisconnectDuration > 0) " (was disconnected ${prevDisconnectDuration}s)" else "")
                listener.onConnect(id())
                // Notify listener that socket reconnected so buffers can be flushed
                if (prevDisconnectDuration > 0) {
                    listener.onReconnected()
                }
            }
            on(Socket.EVENT_DISCONNECT) {
                disconnectedSinceMs = System.currentTimeMillis()
                Log.w(TAG, "Socket disconnected")
                listener.onDisconnect()
            }
            on(Socket.EVENT_CONNECT_ERROR) { args ->
                reconnectAttemptCount++
                Log.e(TAG, "Socket connect error (attempt #$reconnectAttemptCount): ${args.joinToString()}")
                listener.onError(args)
            }
            on(config.privateMessageEvent) { args ->
                Log.d(TAG, "Received privateMessage event: ${args.firstOrNull()}")
                parsePayload(args)?.let(listener::onPrivateMessage)
            }
            on(config.deviceCommandEvent) { args ->
                Log.i(TAG, ">>> RECEIVED deviceCommand event: ${args.firstOrNull()}")
                parsePayload(args)?.let(listener::onDeviceCommand)
            }
            on("config.push") { args ->
                Log.i(TAG, ">>> RECEIVED config.push event: ${args.firstOrNull()}")
                parsePayload(args)?.let { payload ->
                    val type = payload.optString("type", "")
                    when (type) {
                        "vas_weights_update" -> {
                            val weightsJson = payload.optJSONObject("weights_by_venue")
                            if (weightsJson != null) {
                                VASWeightConfig.updateFromJson(weightsJson)
                                Log.i(TAG, "VAS weights updated from config.push")
                            }
                        }
                        "sensing_config_update" -> {
                            val configJson = payload.optJSONObject("config")
                            if (configJson != null) {
                                SensingConfig.updateFromJson(configJson)
                                Log.i(TAG, "Sensing config updated from config.push")
                            }
                        }
                    }
                    listener.onConfigPush(payload)
                }
            }
            on("screen_binding") { args ->
                Log.i(TAG, ">>> RECEIVED screen_binding event: ${args.firstOrNull()}")
                parsePayload(args)?.let(listener::onScreenBinding)
            }
            // Log incoming events for debugging (filter high-frequency noise)
            onAnyIncoming { args ->
                val eventName = args.firstOrNull()?.toString() ?: ""
                if (eventName !in setOf("heartbeat", "audience_signal", "pong")) {
                    Log.d(TAG, "ANY incoming event: ${args.joinToString()}")
                }
            }
            connect()
        }
        Log.i(TAG, "Socket connect() called")

        // Start periodic reconnection health check
        handler.removeCallbacks(reconnectHealthCheck)
        handler.postDelayed(reconnectHealthCheck, RECONNECT_HEALTH_CHECK_INTERVAL_MS)
    }

    /**
     * Periodic health check that detects stuck disconnection state and forces reconnection.
     *
     * Socket.IO's built-in auto-reconnect can get stuck when:
     * - The underlying transport is in a half-open state
     * - Network changes happen during a reconnection backoff window
     * - The reconnection timer is lost after an Android doze cycle
     *
     * This check runs every 90s and forces a fresh reconnect if the socket has been
     * disconnected for longer than FORCE_RECONNECT_AFTER_MS (2 minutes).
     */
    private fun checkReconnectionHealth() {
        val s = socket ?: return
        val listener = currentListener ?: return

        if (s.connected()) {
            // Socket is healthy — nothing to do
            if (disconnectedSinceMs > 0L) {
                // Clear stale disconnected timestamp (shouldn't happen, but defensive)
                disconnectedSinceMs = 0L
            }
            return
        }

        // Socket is disconnected
        val disconnectedMs = if (disconnectedSinceMs > 0) {
            System.currentTimeMillis() - disconnectedSinceMs
        } else {
            // First time we're seeing the disconnect via health check — start tracking
            disconnectedSinceMs = System.currentTimeMillis()
            0L
        }

        val disconnectedSec = disconnectedMs / 1000

        if (disconnectedMs < FORCE_RECONNECT_AFTER_MS) {
            Log.d(TAG, "[ReconnectCheck] Socket disconnected for ${disconnectedSec}s, " +
                    "waiting for auto-reconnect (force after ${FORCE_RECONNECT_AFTER_MS / 1000}s)")
            return
        }

        // Socket has been disconnected too long — force reconnect
        Log.w(TAG, "[ReconnectCheck] Socket stuck disconnected for ${disconnectedSec}s " +
                "(attempts=$reconnectAttemptCount). Forcing fresh reconnect.")

        try {
            // Tear down the stale socket without clearing our state
            s.off()
            s.disconnect()
        } catch (e: Exception) {
            Log.w(TAG, "[ReconnectCheck] Error tearing down stale socket: ${e.message}")
        }
        socket = null

        // Re-connect with stored parameters
        val fp = currentFingerprint
        val token = currentDeviceToken
        if (fp != null) {
            connect(fp, listener, token)
        } else {
            Log.e(TAG, "[ReconnectCheck] Cannot force reconnect — no stored fingerprint")
        }
    }

    fun emitCommandAck(payload: JSONObject) {
        socket?.emit(config.deviceCommandAckEvent, payload)
    }

    fun emitStatus(payload: JSONObject) {
        socket?.emit(config.deviceCommandStatusEvent, payload)
    }

    /**
     * Emit a generic event with a JSON payload.
     * Used for audience metrics, device capabilities, etc.
     */
    fun emit(event: String, payload: JSONObject) {
        Log.d(TAG, "Emitting $event: $payload")
        socket?.emit(event, payload)
    }

    /**
     * Check if the socket is currently connected.
     */
    fun isConnected(): Boolean = socket?.connected() == true

    fun disconnect() {
        isIntentionalDisconnect = true
        handler.removeCallbacks(reconnectHealthCheck)
        disconnectedSinceMs = 0L
        reconnectAttemptCount = 0
        currentListener = null
        currentFingerprint = null
        currentDeviceToken = null
        socket?.off()
        socket?.disconnect()
        socket = null
    }

    private fun parsePayload(args: Array<Any?>): JSONObject? {
        val payload = args.firstOrNull() ?: return null
        return when (payload) {
            is JSONObject -> payload
            is String -> runCatching { JSONObject(payload) }.getOrNull()
            else -> null
        }
    }

    interface Listener {
        fun onConnect(socketId: String?)
        fun onDisconnect()
        fun onError(args: Array<Any?>)
        fun onPrivateMessage(payload: JSONObject)
        fun onDeviceCommand(payload: JSONObject)
        fun onConfigPush(payload: JSONObject) {}
        /** Called when server pushes a screen binding change (paired/deleted). */
        fun onScreenBinding(payload: JSONObject) {}
        /** Called when socket reconnects after a disconnection. Use to flush buffered signals. */
        fun onReconnected() {}
    }
}
