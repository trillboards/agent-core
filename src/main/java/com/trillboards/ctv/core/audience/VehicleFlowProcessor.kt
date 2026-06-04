package com.trillboards.ctv.core.audience

import android.util.Log
import org.json.JSONObject
import java.util.concurrent.ConcurrentLinkedDeque

/**
 * Tracks vehicle flow patterns: directional trends, speed estimation,
 * parking lot fill rate, and vehicle type composition.
 *
 * Uses consecutive ObjectDetectionProcessor results (at 10-second intervals)
 * to infer traffic flow dynamics without tracking individual vehicles.
 *
 * Privacy-Preserving: operates on aggregate counts only, never stores
 * or transmits vehicle-specific identifiers (plates, colors, etc.).
 *
 * Thread-safe: ConcurrentLinkedDeque for history, volatile for max counter.
 */
class VehicleFlowProcessor {
    companion object {
        private const val TAG = "VehicleFlow"
        private const val MAX_HISTORY = 60  // 60 snapshots (~10 min at 10s intervals)
    }

    /**
     * A single point-in-time vehicle observation.
     */
    data class VehicleSnapshot(
        val timestamp: Long,
        val vehicleCount: Int,
        val vehicleTypes: Map<String, Int>,
        val totalObjects: Int
    )

    /**
     * Aggregated flow metrics over the observation window.
     */
    data class FlowMetrics(
        val currentVehicleCount: Int,
        val avgVehicleCount: Float,
        val peakVehicleCount: Int,
        val vehicleTypes: Map<String, Int>,
        val fillRateEstimate: Float,  // 0-1, based on max observed as capacity proxy
        val trend: String             // "increasing", "decreasing", "stable"
    )

    private val history = ConcurrentLinkedDeque<VehicleSnapshot>()
    @Volatile private var maxObservedVehicles = 1  // Prevent div-by-zero

    /**
     * Record a new vehicle observation from ObjectDetectionProcessor results.
     * Call this on every detection cycle (typically every 10 seconds).
     */
    fun recordSnapshot(result: ObjectDetectionProcessor.DetectionResult) {
        val vehicleTypes = result.objectCounts.filter { (className, _) ->
            className in ObjectDetectionProcessor.VEHICLE_CLASSES
        }

        val snapshot = VehicleSnapshot(
            timestamp = System.currentTimeMillis(),
            vehicleCount = result.vehicleCount,
            vehicleTypes = vehicleTypes,
            totalObjects = result.totalObjects
        )

        history.addLast(snapshot)
        while (history.size > MAX_HISTORY) history.pollFirst()

        if (result.vehicleCount > maxObservedVehicles) {
            maxObservedVehicles = result.vehicleCount
            Log.d(TAG, "New max observed vehicles: $maxObservedVehicles")
        }
    }

    /**
     * Compute aggregated flow metrics over the observation window.
     * Returns stable defaults when insufficient data is available.
     */
    fun getFlowMetrics(): FlowMetrics {
        val snapshots = history.toList()
        if (snapshots.isEmpty()) {
            return FlowMetrics(0, 0f, 0, emptyMap(), 0f, "stable")
        }

        val current = snapshots.last()
        val avg = snapshots.map { it.vehicleCount }.average().toFloat()
        val peak = snapshots.maxOf { it.vehicleCount }

        // Aggregate vehicle types across the full observation window
        val aggregatedTypes = mutableMapOf<String, Int>()
        snapshots.forEach { snapshot ->
            snapshot.vehicleTypes.forEach { (type, count) ->
                aggregatedTypes[type] = (aggregatedTypes[type] ?: 0) + count
            }
        }

        // Trend: compare last 3 observations vs first 3 observations
        // Requires at least 6 snapshots for meaningful trend detection
        val trend = if (snapshots.size >= 6) {
            val recentAvg = snapshots.takeLast(3).map { it.vehicleCount }.average()
            val earlierAvg = snapshots.take(3).map { it.vehicleCount }.average()
            when {
                recentAvg > earlierAvg * 1.2 -> "increasing"
                recentAvg < earlierAvg * 0.8 -> "decreasing"
                else -> "stable"
            }
        } else "stable"

        return FlowMetrics(
            currentVehicleCount = current.vehicleCount,
            avgVehicleCount = avg,
            peakVehicleCount = peak,
            vehicleTypes = aggregatedTypes,
            fillRateEstimate = current.vehicleCount.toFloat() / maxObservedVehicles,
            trend = trend
        )
    }

    /**
     * Convert metrics to JSON for the audience signal payload.
     * Keys use registry-canonical snake_case so chipPersistence.sanitizeFieldKey
     * is idempotent on canonical input.
     */
    fun toJson(metrics: FlowMetrics): JSONObject {
        return JSONObject().apply {
            put("vehicle_count", metrics.currentVehicleCount)
            put("avg_vehicle_count", metrics.avgVehicleCount)
            put("peak_vehicle_count", metrics.peakVehicleCount)
            put("fill_rate_estimate", metrics.fillRateEstimate)
            put("trend", metrics.trend)

            val typesJson = JSONObject()
            metrics.vehicleTypes.forEach { (type, count) ->
                typesJson.put(type, count)
            }
            put("vehicle_types", typesJson)
        }
    }

    /**
     * Get the number of snapshots currently buffered.
     */
    fun getHistorySize(): Int = history.size

    /**
     * Reset all state. Call when the device changes location or after
     * an extended offline period.
     */
    fun reset() {
        history.clear()
        maxObservedVehicles = 1
        Log.i(TAG, "VehicleFlowProcessor reset")
    }
}
