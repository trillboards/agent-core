package com.trillboards.ctv.core.audience

import android.util.Log
import org.json.JSONObject
import java.util.concurrent.ConcurrentLinkedDeque

/**
 * Estimates queue length and wait time from face detection + spatial analysis.
 * Uses face count trends over time to infer service rate and queue dynamics.
 *
 * Algorithm:
 * - Tracks face counts and dwell time across consecutive sensing windows
 * - Estimates service rate (departures per minute) from count decreases
 * - Infers wait time from current count / service rate
 * - Detects queue presence when count exceeds configurable threshold
 *
 * Useful for: retail store queues, food court lines, transit stops,
 * or any venue where people wait in proximity to the screen.
 *
 * Thread-safe: ConcurrentLinkedDeque for history, volatile threshold.
 */
class QueueEstimationProcessor {
    companion object {
        private const val TAG = "QueueEstimation"
        private const val MAX_HISTORY = 30  // 30 snapshots (~5 min at 10s intervals)
        private const val DEFAULT_SERVICE_RATE_PER_MIN = 2.0f // people per minute fallback
        private const val MIN_SERVICE_RATE = 0.1f // floor to prevent division-by-zero-like waits
    }

    /**
     * A single queue observation at a point in time.
     */
    data class QueueSnapshot(
        val timestamp: Long,
        val faceCount: Int,
        val avgDwellTimeMs: Long
    )

    /**
     * Aggregated queue metrics over the observation window.
     */
    data class QueueMetrics(
        val estimatedQueueLength: Int,
        val estimatedWaitTimeMinutes: Float,
        val serviceRatePerMinute: Float,
        val isQueueDetected: Boolean,
        val trend: String // "growing", "shrinking", "stable"
    )

    private val history = ConcurrentLinkedDeque<QueueSnapshot>()
    @Volatile private var queueLengthThreshold: Int = 3 // min people to consider a "queue"

    /**
     * Set the minimum face count to consider a queue detected.
     * Server can tune this per venue type.
     */
    fun setThreshold(threshold: Int) {
        queueLengthThreshold = threshold.coerceAtLeast(1)
        Log.d(TAG, "Queue threshold set to $queueLengthThreshold")
    }

    /**
     * Record a new queue observation.
     *
     * @param faceCount Number of faces detected in this window
     * @param avgDwellTimeMs Average dwell time of tracked faces (0 if unknown)
     */
    fun recordSnapshot(faceCount: Int, avgDwellTimeMs: Long) {
        history.addLast(QueueSnapshot(
            timestamp = System.currentTimeMillis(),
            faceCount = faceCount,
            avgDwellTimeMs = avgDwellTimeMs
        ))
        while (history.size > MAX_HISTORY) history.pollFirst()
    }

    /**
     * Compute queue metrics from the observation window.
     * Returns stable defaults when insufficient data is available.
     */
    fun getQueueMetrics(): QueueMetrics {
        val snapshots = history.toList()
        if (snapshots.isEmpty()) {
            return QueueMetrics(
                estimatedQueueLength = 0,
                estimatedWaitTimeMinutes = 0f,
                serviceRatePerMinute = DEFAULT_SERVICE_RATE_PER_MIN,
                isQueueDetected = false,
                trend = "stable"
            )
        }

        val current = snapshots.last()

        // Estimate service rate from face count decreases over time.
        // Each consecutive decrease likely represents someone being served / departing.
        val serviceRate = if (snapshots.size >= 2) {
            val intervals = snapshots.zipWithNext()
            val departures = intervals.count { (a, b) -> b.faceCount < a.faceCount }
            val timeSpanMinutes = (snapshots.last().timestamp - snapshots.first().timestamp) / 60000.0
            if (timeSpanMinutes > 0) {
                (departures / timeSpanMinutes).toFloat()
            } else {
                DEFAULT_SERVICE_RATE_PER_MIN
            }
        } else {
            DEFAULT_SERVICE_RATE_PER_MIN
        }

        val effectiveServiceRate = maxOf(serviceRate, MIN_SERVICE_RATE)
        val waitTimeMinutes = current.faceCount / effectiveServiceRate

        // Trend: compare last 2 observations vs first 2
        // Requires at least 4 snapshots for meaningful trend detection
        val trend = if (snapshots.size >= 4) {
            val recentAvg = snapshots.takeLast(2).map { it.faceCount }.average()
            val earlierAvg = snapshots.take(2).map { it.faceCount }.average()
            when {
                recentAvg > earlierAvg + 1 -> "growing"
                recentAvg < earlierAvg - 1 -> "shrinking"
                else -> "stable"
            }
        } else "stable"

        return QueueMetrics(
            estimatedQueueLength = current.faceCount,
            estimatedWaitTimeMinutes = waitTimeMinutes,
            serviceRatePerMinute = effectiveServiceRate,
            isQueueDetected = current.faceCount >= queueLengthThreshold,
            trend = trend
        )
    }

    /**
     * Convert metrics to JSON for the audience signal payload.
     * Keys use registry-canonical snake_case so chipPersistence.sanitizeFieldKey
     * is idempotent on canonical input.
     */
    fun toJson(metrics: QueueMetrics): JSONObject {
        return JSONObject().apply {
            put("pedestrian_count", metrics.estimatedQueueLength)
            put("queue_length", metrics.estimatedQueueLength)
            put("estimated_wait_time_minutes", metrics.estimatedWaitTimeMinutes)
            put("service_rate_per_minute", metrics.serviceRatePerMinute)
            put("queue_detected", metrics.isQueueDetected)
            put("trend", metrics.trend)
        }
    }

    /**
     * Get the number of snapshots currently buffered.
     */
    fun getHistorySize(): Int = history.size

    /**
     * Reset all state. Call when the device changes venue or after
     * an extended offline period.
     */
    fun reset() {
        history.clear()
        Log.i(TAG, "QueueEstimationProcessor reset")
    }
}
