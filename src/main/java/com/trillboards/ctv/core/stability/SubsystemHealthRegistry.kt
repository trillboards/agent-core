package com.trillboards.ctv.core.stability

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap

/**
 * Registry tracking health status of all device subsystems.
 * Each subsystem writes a heartbeat at its own cadence; the registry
 * aggregates them into a fleet-visible health snapshot emitted with every
 * socket heartbeat event (every 30s).
 *
 * Subsystem health is persisted to SharedPreferences so it survives
 * process restarts and can be read by the ServiceWatchdog.
 */
class SubsystemHealthRegistry(context: Context) {

    companion object {
        private const val TAG = "SubsystemHealth"
        private const val PREFS_NAME = "subsystem_health"
    }

    enum class Subsystem {
        SOCKET,
        PLAYER,
        CAMERA,
        MICROPHONE,
        ML_PIPELINE,
        MEMORY
    }

    enum class SubsystemStatus {
        // Socket statuses
        CONNECTED, DISCONNECTED, RECONNECTING,
        // Player statuses
        PLAYING, STUCK, CRASHED, RELOADING,
        // Camera/Mic statuses
        ACTIVE, DISABLED, PERMISSION_DENIED, ERROR,
        // ML Pipeline statuses
        FULL, DEGRADED, CRITICAL, OFFLINE,
        // Memory statuses
        NORMAL, MEDIUM, HIGH,
        // Generic
        UNKNOWN
    }

    data class SubsystemHealth(
        val subsystem: Subsystem,
        val status: SubsystemStatus,
        val lastHeartbeatMs: Long,
        val errorMessage: String? = null,
        val metadata: Map<String, Any>? = null
    ) {
        val ageMs: Long get() = System.currentTimeMillis() - lastHeartbeatMs

        fun isStale(thresholdMs: Long): Boolean = ageMs > thresholdMs

        fun toJson(): JSONObject = JSONObject().apply {
            put("subsystem", subsystem.name.lowercase())
            put("status", status.name.lowercase())
            put("lastHeartbeatMs", lastHeartbeatMs)
            put("ageMs", ageMs)
            errorMessage?.let { put("errorMessage", it) }
            metadata?.forEach { (k, v) -> put(k, v) }
        }
    }

    private val healthMap = ConcurrentHashMap<Subsystem, SubsystemHealth>()
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // Stale thresholds per subsystem (ms)
    private val staleThresholds = mapOf(
        Subsystem.SOCKET to 60_000L,       // 60s — socket heartbeat every 30s
        Subsystem.PLAYER to 180_000L,      // 3min — player ping every 60s
        // Camera and mic both run behind a 30s watchdog. Give them room to miss a single
        // watchdog boundary without forcing a self-inflicted rebind while frames/audio
        // are still flowing.
        Subsystem.CAMERA to 90_000L,       // 90s — 3 missed watchdog windows before recovery
        Subsystem.MICROPHONE to 90_000L,   // 90s — avoid false-positive mic recovery loops
        Subsystem.ML_PIPELINE to 30_000L,  // 30s — inference every 10s
        Subsystem.MEMORY to 30_000L        // 30s — check every 15s
    )

    init {
        // Restore persisted health on startup
        for (sub in Subsystem.values()) {
            val statusStr = prefs.getString("${sub.name}_status", null)
            val lastBeat = prefs.getLong("${sub.name}_lastBeat", 0)
            if (statusStr != null && lastBeat > 0) {
                val status = try { SubsystemStatus.valueOf(statusStr) } catch (_: Exception) { SubsystemStatus.UNKNOWN }
                healthMap[sub] = SubsystemHealth(sub, status, lastBeat)
            }
        }
    }

    /**
     * Update a subsystem's health status. Called by each subsystem at its own cadence.
     */
    fun updateHealth(
        subsystem: Subsystem,
        status: SubsystemStatus,
        errorMessage: String? = null,
        metadata: Map<String, Any>? = null
    ) {
        val now = System.currentTimeMillis()
        val health = SubsystemHealth(subsystem, status, now, errorMessage, metadata)
        healthMap[subsystem] = health

        // Persist to SharedPreferences
        prefs.edit()
            .putString("${subsystem.name}_status", status.name)
            .putLong("${subsystem.name}_lastBeat", now)
            .apply()

        Log.d(TAG, "${subsystem.name} → ${status.name}" + (errorMessage?.let { " ($it)" } ?: ""))
    }

    /**
     * Get the current health of a specific subsystem.
     */
    fun getHealth(subsystem: Subsystem): SubsystemHealth? = healthMap[subsystem]

    /**
     * Check if a subsystem is in an error/failed state.
     */
    fun isSubsystemFailed(subsystem: Subsystem): Boolean {
        val health = healthMap[subsystem] ?: return true // No heartbeat = failed
        val threshold = staleThresholds[subsystem] ?: 60_000L
        return shouldTreatSubsystemHealthAsFailed(health.status, health.isStale(threshold))
    }

    /**
     * Build aggregate health snapshot for the socket heartbeat payload.
     */
    fun toHeartbeatPayload(): JSONObject {
        val payload = JSONObject()
        val subsystems = JSONObject()

        for (sub in Subsystem.values()) {
            val health = healthMap[sub]
            if (health != null) {
                subsystems.put(sub.name.lowercase(), health.toJson())
            } else {
                subsystems.put(sub.name.lowercase(), JSONObject().apply {
                    put("status", "unknown")
                    put("lastHeartbeatMs", 0)
                })
            }
        }

        payload.put("subsystems", subsystems)
        payload.put("overallHealthy", isOverallHealthy())
        payload.put("failedSubsystems", getFailedSubsystems().map { it.name.lowercase() }.let {
            org.json.JSONArray(it)
        })

        return payload
    }

    /**
     * Returns true if all subsystems are healthy (no failures or stale heartbeats).
     */
    fun isOverallHealthy(): Boolean = Subsystem.values().none { isSubsystemFailed(it) }

    /**
     * Returns list of subsystems currently in a failed state.
     */
    fun getFailedSubsystems(): List<Subsystem> = Subsystem.values().filter { isSubsystemFailed(it) }

    /**
     * Clear all health data (for testing or device reset).
     */
    fun clear() {
        healthMap.clear()
        prefs.edit().clear().apply()
    }
}

internal fun shouldTreatSubsystemHealthAsFailed(
    status: SubsystemHealthRegistry.SubsystemStatus,
    isStale: Boolean
): Boolean {
    if (status == SubsystemHealthRegistry.SubsystemStatus.DISABLED ||
        status == SubsystemHealthRegistry.SubsystemStatus.PERMISSION_DENIED) {
        return false
    }

    if (isStale) return true

    return status in setOf(
        SubsystemHealthRegistry.SubsystemStatus.CRASHED,
        SubsystemHealthRegistry.SubsystemStatus.ERROR,
        SubsystemHealthRegistry.SubsystemStatus.OFFLINE,
        SubsystemHealthRegistry.SubsystemStatus.DISCONNECTED
    )
}
