package com.trillboards.ctv.core.stability

import android.app.ActivityManager
import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.util.Log
import com.trillboards.ctv.core.DeviceIdentity
import com.trillboards.ctv.core.socket.AgentSocketManager
import org.json.JSONArray
import org.json.JSONObject

/**
 * Crash and stability telemetry reporter for CTV devices.
 * All events are tagged with rendering_mode for native vs WebView A/B comparison.
 *
 * Events are buffered locally (SharedPreferences) and flushed on socket reconnect.
 * This enables offline crash collection — events are not lost if the device is
 * disconnected when a crash occurs.
 */
class CrashTelemetryReporter(
    private val context: Context,
    private val socketManager: AgentSocketManager,
    private val healthRegistry: SubsystemHealthRegistry
) {
    companion object {
        private const val TAG = "CrashTelemetryReporter"
        private const val PREFS_NAME = "crash_telemetry"
        private const val PREF_PENDING_EVENTS = "pending_events"
        private const val MAX_PENDING = 50

        /**
         * Pure-function wire-format builder for a `stabilityEvent` payload.
         * Extracted so we can unit-test the JSON shape without standing up
         * Context/SocketManager/HealthRegistry. Visible to test classpath only.
         *
         * `jvmHeapMaxMb` is the legacy `memory_total_mb` value (Runtime.maxMemory),
         * `deviceTotalRamMb` is the new physical-RAM field
         * (ActivityManager.MemoryInfo.totalMem). Both ride together; readers
         * should prefer `device_total_ram_mb` for fleet-RAM analysis. See
         * deferred-work.yaml entry rename-stability-memory-total-mb-to-jvm-heap-max-mb.
         */
        @JvmStatic
        internal fun buildStabilityEventJson(
            eventType: String,
            fingerprint: String,
            screenId: String?,
            renderingMode: String,
            agentVersion: String,
            deviceModel: String,
            chipset: String,
            attenuationTier: String,
            memoryUsedMb: Int,
            jvmHeapMaxMb: Int,
            deviceTotalRamMb: Int?,
            activeModels: JSONArray?,
            metadata: Map<String, Any>,
            stackTrace: String?,
            timestampMs: Long
        ): JSONObject {
            return JSONObject().apply {
                put("event_type", eventType)
                put("fingerprint", fingerprint)
                put("screen_id", screenId ?: "")
                put("rendering_mode", renderingMode)
                put("agent_version", agentVersion)
                put("device_model", deviceModel)
                put("chipset", chipset)
                put("attenuation_tier", attenuationTier)
                put("memory_used_mb", memoryUsedMb)
                put("memory_total_mb", jvmHeapMaxMb)
                if (deviceTotalRamMb != null) put("device_total_ram_mb", deviceTotalRamMb)
                if (stackTrace != null) put("stack_trace", stackTrace)
                if (activeModels != null) put("active_models", activeModels)
                put("metadata", JSONObject(metadata))
                put("timestamp", timestampMs)
            }
        }
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val fingerprint: String by lazy { DeviceIdentity.fingerprint(context) }
    private var screenId: String? = null
    private var renderingMode: String = "webview"
    private var agentVersion: String = ""
    private var originalUncaughtHandler: Thread.UncaughtExceptionHandler? = null

    fun initialize(screenId: String?, renderingMode: String, agentVersion: String) {
        this.screenId = screenId
        this.renderingMode = renderingMode
        this.agentVersion = agentVersion

        // Install global uncaught exception handler
        originalUncaughtHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            reportCrash("uncaughtException", throwable)
            // Forward to original handler (this will terminate the process)
            originalUncaughtHandler?.uncaughtException(thread, throwable)
        }

        // Flush any pending events from previous session
        flushPendingEvents()

        Log.i(TAG, "Initialized: renderingMode=$renderingMode, agentVersion=$agentVersion")
    }

    fun setScreenId(id: String) {
        this.screenId = id
    }

    fun setRenderingMode(mode: String) {
        this.renderingMode = mode
    }

    fun reportWebViewCrash(didCrash: Boolean, rendererPriority: Int?) {
        val metadata = mutableMapOf<String, Any>(
            "didCrash" to didCrash
        )
        if (rendererPriority != null) {
            metadata["rendererPriority"] = rendererPriority
        }
        reportEvent("webviewCrash", metadata)
    }

    fun reportPlayerStuck(currentState: String, bufferingDurationMs: Long) {
        reportEvent("playerStuck", mapOf(
            "playerState" to currentState,
            "bufferingDurationMs" to bufferingDurationMs
        ))
    }

    fun reportPlayerError(errorCode: String, errorMessage: String) {
        reportEvent("playerError", mapOf(
            "errorCode" to errorCode,
            "errorMessage" to errorMessage
        ))
    }

    fun reportVastError(source: String, vastUrl: String, error: String) {
        reportEvent("vastError", mapOf(
            "source" to source,
            "vastUrl" to vastUrl,
            "error" to error
        ))
    }

    fun reportRecovery(recoveryType: String, durationMs: Long) {
        reportEvent("recoverySuccess", mapOf(
            "recoveryType" to recoveryType,
            "durationMs" to durationMs
        ))
    }

    fun reportOomKill(memoryUsedMb: Int, memoryTotalMb: Int) {
        reportEvent("oomKill", mapOf(
            "memoryUsedMb" to memoryUsedMb,
            "memoryTotalMb" to memoryTotalMb
        ))
    }

    fun reportRenderingModeSwitch(fromMode: String, toMode: String) {
        reportEvent("renderingModeSwitch", mapOf(
            "fromMode" to fromMode,
            "toMode" to toMode
        ))
    }

    fun flushPendingEvents() {
        if (!socketManager.isConnected()) return

        val stored = prefs.getString(PREF_PENDING_EVENTS, null) ?: return
        try {
            val events = JSONArray(stored)
            if (events.length() == 0) return

            Log.i(TAG, "Flushing ${events.length()} pending stability events")
            for (i in 0 until events.length()) {
                val event = events.optJSONObject(i) ?: continue
                socketManager.emit("stabilityEvent", event)
            }
            prefs.edit().remove(PREF_PENDING_EVENTS).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to flush pending events", e)
        }
    }

    private fun reportCrash(eventType: String, throwable: Throwable) {
        val stackTrace = throwable.stackTraceToString().take(4096) // Limit size
        reportEvent(eventType, mapOf(
            "exceptionClass" to throwable.javaClass.name,
            "exceptionMessage" to (throwable.message ?: "")
        ), stackTrace)
    }

    /**
     * Read the device's total physical RAM in MB via ActivityManager.MemoryInfo.totalMem.
     * Distinct from Runtime.maxMemory() (the JVM heap cap), which is what
     * `memory_total_mb` historically carried before the 2026-05-03 fix.
     *
     * Returns null if ActivityManager is unavailable for any reason; callers
     * tolerate null because we never want a telemetry-side bug to mask a crash.
     */
    private fun readDeviceTotalRamMb(): Int? {
        return try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
                ?: return null
            val memInfo = ActivityManager.MemoryInfo()
            am.getMemoryInfo(memInfo)
            (memInfo.totalMem / (1024L * 1024L)).toInt()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read device totalMem from ActivityManager", e)
            null
        }
    }

    // TODO(2026-Q4): rename memory_total_mb -> jvm_heap_max_mb after dashboards
    // migrate to device_total_ram_mb. Tracked in deferred-work.yaml as
    // rename-stability-memory-total-mb-to-jvm-heap-max-mb.
    private fun reportEvent(
        eventType: String,
        metadata: Map<String, Any>,
        stackTrace: String? = null
    ) {
        val runtime = Runtime.getRuntime()
        val usedMemMb = ((runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)).toInt()
        // NOTE: `memory_total_mb` is the JVM heap CAP (Runtime.maxMemory),
        // not device RAM — kept for back-compat with existing readers and
        // historical rows. Use `device_total_ram_mb` for physical RAM.
        val jvmHeapMaxMb = (runtime.maxMemory() / (1024 * 1024)).toInt()
        val deviceTotalRamMb = readDeviceTotalRamMb()

        val healthSnapshot = healthRegistry.toHeartbeatPayload()
        val attenuationTier = healthSnapshot.optString("attenuation_tier", "NORMAL")
        val activeModels = healthSnapshot.optJSONArray("active_models")

        val event = buildStabilityEventJson(
            eventType = eventType,
            fingerprint = fingerprint,
            screenId = screenId,
            renderingMode = renderingMode,
            agentVersion = agentVersion,
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
            chipset = Build.HARDWARE,
            attenuationTier = attenuationTier,
            memoryUsedMb = usedMemMb,
            jvmHeapMaxMb = jvmHeapMaxMb,
            deviceTotalRamMb = deviceTotalRamMb,
            activeModels = activeModels,
            metadata = metadata,
            stackTrace = stackTrace,
            timestampMs = System.currentTimeMillis()
        )

        // Try to send immediately
        if (socketManager.isConnected()) {
            socketManager.emit("stabilityEvent", event)
        } else {
            // Buffer for later
            bufferEvent(event)
        }
    }

    private fun bufferEvent(event: JSONObject) {
        try {
            val stored = prefs.getString(PREF_PENDING_EVENTS, null)
            val events = if (stored != null) JSONArray(stored) else JSONArray()

            // Trim oldest if at capacity
            while (events.length() >= MAX_PENDING) {
                events.remove(0)
            }

            events.put(event)
            prefs.edit().putString(PREF_PENDING_EVENTS, events.toString()).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to buffer stability event", e)
        }
    }
}
