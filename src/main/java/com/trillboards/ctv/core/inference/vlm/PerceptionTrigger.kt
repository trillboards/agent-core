package com.trillboards.ctv.core.inference.vlm

import android.os.SystemClock
import android.util.Log
import com.trillboards.ctv.core.SensingConfig

/**
 * Monitors perception plane outputs and triggers VLM inference only on meaningful
 * state changes, replacing the fixed 5-second sampling interval in [VLMInferenceProcessor]
 * with event-driven triggering.
 *
 * ## Why Perception-Triggered?
 * Running VLM inference every 5 seconds is wasteful when the scene hasn't changed.
 * The perception plane (face detection, object detection, audio classification) runs
 * continuously and already knows when the scene changes. By using perception outputs
 * as a trigger signal, we:
 * - **Save compute**: Skip VLM calls when the scene is static
 * - **Reduce latency**: Trigger immediately on meaningful changes instead of waiting
 *   for the next 5-second tick
 * - **Preserve freshness**: Periodic refresh (maxTriggerIntervalMs) ensures VLM data
 *   doesn't go completely stale even in static scenes
 *
 * ## Trigger Conditions
 * A VLM inference is triggered when any of these change:
 * 1. Person count (someone entered or left the scene)
 * 2. Object inventory (new objects appeared or disappeared)
 * 3. Dominant emotion shifted (crowd mood changed)
 * 4. Noise level changed significantly (delta > 0.3)
 *
 * ## Throttling
 * - **minTriggerIntervalMs** (5s): Prevents thrashing on rapid state changes
 * - **maxTriggerIntervalMs** (30s): Forces periodic VLM refresh even in static scenes
 *
 * ## Thread Safety
 * All public methods are safe to call from the aggregation loop's coroutine scope.
 * State is read/written only from [shouldTriggerVLM] and [markTriggered], which
 * are called sequentially within the aggregation cycle.
 *
 * @see VLMInferenceProcessor The VLM processor that runs inference when triggered
 */
class PerceptionTrigger(
    private val clock: () -> Long = { SystemClock.elapsedRealtime() }
) {

    companion object {
        private const val TAG = "PerceptionTrigger"

        /** Default minimum interval between VLM triggers (prevent thrashing). */
        val DEFAULT_MIN_TRIGGER_INTERVAL_MS: Long get() = SensingConfig.get().perception.minTriggerIntervalMs

        /** Default maximum interval — force VLM even if scene hasn't changed (periodic refresh). */
        val DEFAULT_MAX_TRIGGER_INTERVAL_MS: Long get() = SensingConfig.get().perception.maxTriggerIntervalMs

        /** Noise level delta threshold for triggering VLM inference. */
        val NOISE_DELTA_THRESHOLD: Float get() = SensingConfig.get().perception.noiseDeltaThreshold
    }

    /**
     * Snapshot of the current scene as observed by the perception plane.
     *
     * Built from aggregated perception outputs (face detection, object detection,
     * audio classification) during each aggregation cycle.
     *
     * @param personCount Number of people detected in the scene (from blazeface/efficientdet)
     * @param objectCounts Map of object class name to count (from efficientdet)
     * @param dominantEmotion The most prevalent emotion in the audience (from FER+)
     * @param noiseLevel Ambient noise level (0.0 to 1.0 from YAMNet)
     * @param timestamp When this state was observed (epoch millis)
     */
    data class SceneState(
        val personCount: Int = 0,
        val objectCounts: Map<String, Int> = emptyMap(),
        val dominantEmotion: String? = null,
        val noiseLevel: Float = 0f,
        val timestamp: Long = 0L
    )

    /** The scene state at the time of the last VLM trigger. */
    @Volatile
    private var lastTriggeredState: SceneState? = null

    /** Timestamp (elapsedRealtime) of the last VLM trigger. */
    @Volatile
    private var lastTriggerTimeMs: Long = 0L

    /** Minimum interval between VLM triggers (prevent thrashing). */
    @Volatile
    private var minTriggerIntervalMs: Long = DEFAULT_MIN_TRIGGER_INTERVAL_MS

    /** Maximum interval — force VLM even if scene hasn't changed (periodic refresh). */
    @Volatile
    private var maxTriggerIntervalMs: Long = DEFAULT_MAX_TRIGGER_INTERVAL_MS

    /** Count of triggers for telemetry. */
    @Volatile
    private var triggerCount: Long = 0L

    /** Count of skipped triggers (within min interval) for telemetry. */
    @Volatile
    private var skipCount: Long = 0L

    /**
     * Evaluate whether VLM inference should be triggered based on the current
     * perception state.
     *
     * Decision logic (in priority order):
     * 1. Always trigger if max interval exceeded (periodic refresh)
     * 2. Never trigger if within min interval (prevent thrashing)
     * 3. Trigger on first observation (no previous state)
     * 4. Trigger on meaningful state changes
     *
     * @param currentState The current scene state from perception plane
     * @return true if VLM inference should be triggered
     */
    fun shouldTriggerVLM(currentState: SceneState): Boolean {
        val now = clock()

        // Always trigger if max interval exceeded (periodic refresh)
        if (now - lastTriggerTimeMs >= maxTriggerIntervalMs) {
            Log.d(TAG, "Triggering VLM: max interval exceeded " +
                "(${now - lastTriggerTimeMs}ms >= ${maxTriggerIntervalMs}ms)")
            return true
        }

        // Don't trigger if within minimum interval
        if (now - lastTriggerTimeMs < minTriggerIntervalMs) {
            skipCount++
            return false
        }

        // First observation — always trigger
        val last = lastTriggeredState ?: return true

        // Trigger on meaningful state changes
        return hasStateChanged(last, currentState)
    }

    /**
     * Record that VLM inference was triggered with the given state.
     *
     * Call this AFTER successfully triggering VLM inference (not before),
     * so that if inference fails, the next cycle will try again with
     * the same state change.
     *
     * @param state The scene state at trigger time
     */
    fun markTriggered(state: SceneState) {
        lastTriggeredState = state
        lastTriggerTimeMs = clock()
        triggerCount++

        Log.d(TAG, "VLM trigger #$triggerCount marked: persons=${state.personCount}, " +
            "objects=${state.objectCounts.size}, emotion=${state.dominantEmotion}, " +
            "noise=${state.noiseLevel}")
    }

    /**
     * Determine if the scene has changed meaningfully between two states.
     *
     * @param old Previous scene state (at last trigger)
     * @param new Current scene state
     * @return true if a meaningful change occurred
     */
    private fun hasStateChanged(old: SceneState, new: SceneState): Boolean {
        // Person count changed (someone entered/left)
        if (old.personCount != new.personCount) {
            Log.d(TAG, "State change: personCount ${old.personCount} → ${new.personCount}")
            return true
        }

        // New objects detected or objects disappeared
        if (old.objectCounts != new.objectCounts) {
            Log.d(TAG, "State change: objectCounts changed " +
                "${old.objectCounts} → ${new.objectCounts}")
            return true
        }

        // Emotion shifted significantly
        if (old.dominantEmotion != new.dominantEmotion) {
            Log.d(TAG, "State change: emotion ${old.dominantEmotion} → ${new.dominantEmotion}")
            return true
        }

        // Noise level changed significantly (>0.3 delta)
        if (Math.abs(old.noiseLevel - new.noiseLevel) > NOISE_DELTA_THRESHOLD) {
            Log.d(TAG, "State change: noise ${old.noiseLevel} → ${new.noiseLevel} " +
                "(delta=${Math.abs(old.noiseLevel - new.noiseLevel)})")
            return true
        }

        return false
    }

    /**
     * Update trigger intervals at runtime (e.g., from dynamic server config).
     *
     * @param minIntervalMs New minimum interval (null to keep current). Clamped to [1000, 60000].
     * @param maxIntervalMs New maximum interval (null to keep current). Clamped to [5000, 120000].
     */
    fun updateConfig(minIntervalMs: Long? = null, maxIntervalMs: Long? = null) {
        val cfg = SensingConfig.get().perception
        minIntervalMs?.let {
            minTriggerIntervalMs = it.coerceIn(cfg.minTriggerIntervalBoundMinMs, cfg.minTriggerIntervalBoundMaxMs)
            Log.i(TAG, "Min trigger interval updated to ${minTriggerIntervalMs}ms")
        }
        maxIntervalMs?.let {
            maxTriggerIntervalMs = it.coerceIn(cfg.maxTriggerIntervalBoundMinMs, cfg.maxTriggerIntervalBoundMaxMs)
            Log.i(TAG, "Max trigger interval updated to ${maxTriggerIntervalMs}ms")
        }
    }

    /**
     * Get telemetry counters for monitoring.
     *
     * @return Pair of (triggerCount, skipCount) since creation
     */
    fun getTelemetry(): Pair<Long, Long> = Pair(triggerCount, skipCount)

    /**
     * Reset internal state (e.g., when sensing restarts).
     */
    fun reset() {
        lastTriggeredState = null
        lastTriggerTimeMs = 0L
        triggerCount = 0L
        skipCount = 0L
        Log.i(TAG, "PerceptionTrigger reset")
    }
}
