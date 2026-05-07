package com.trillboards.ctv.core.inference.vlm

import android.os.SystemClock
import android.util.Log
import com.trillboards.ctv.core.SensingConfig

/**
 * Per-model circuit breaker for VLM inference stability.
 * Prevents repeated crashes from taking down the device.
 *
 * State machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
 * - CLOSED: Normal operation, inference allowed
 * - OPEN: Inference blocked after consecutive failures, waiting for cooldown
 * - HALF_OPEN: Cooldown expired, allowing probe inferences to test recovery
 */
class ModelCircuitBreaker(
    private val modelId: String,
    private val failureThreshold: Int = SensingConfig.get().circuitBreaker.failureThreshold,
    private val halfOpenRetryIntervalMs: Long = SensingConfig.get().circuitBreaker.halfOpenRetryIntervalMs,
    private val halfOpenSuccessThreshold: Int = SensingConfig.get().circuitBreaker.halfOpenSuccessThreshold
) {
    enum class State { CLOSED, OPEN, HALF_OPEN }

    @Volatile
    var state: State = State.CLOSED
        private set

    private var consecutiveFailures: Int = 0
    private var consecutiveSuccesses: Int = 0
    private var openedAtMs: Long = 0L
    private var totalFailures: Int = 0
    private var totalSuccesses: Int = 0

    /**
     * Check if inference is allowed by the circuit breaker.
     *
     * NOTE: This property has a side effect — when state is OPEN and cooldown
     * has expired, it transitions to HALF_OPEN. This is intentional: the
     * single caller (VLMInferenceProcessor.process()) checks this once per
     * inference cycle, making the transition atomic with the decision.
     */
    val isAllowed: Boolean
        @Synchronized get() = when (state) {
            State.CLOSED -> true
            State.OPEN -> {
                if (SystemClock.elapsedRealtime() - openedAtMs >= halfOpenRetryIntervalMs) {
                    state = State.HALF_OPEN
                    consecutiveSuccesses = 0
                    Log.i(TAG, "[$modelId] Circuit breaker: OPEN -> HALF_OPEN (cooldown expired)")
                    true
                } else {
                    false
                }
            }
            State.HALF_OPEN -> true
        }

    @Synchronized
    fun recordSuccess() {
        totalSuccesses++
        consecutiveFailures = 0
        consecutiveSuccesses++

        when (state) {
            State.HALF_OPEN -> {
                if (consecutiveSuccesses >= halfOpenSuccessThreshold) {
                    state = State.CLOSED
                    Log.i(TAG, "[$modelId] Circuit breaker: HALF_OPEN -> CLOSED ($consecutiveSuccesses consecutive successes)")
                }
            }
            State.OPEN -> {
                // Shouldn't happen, but handle gracefully
                state = State.CLOSED
                Log.w(TAG, "[$modelId] Circuit breaker: OPEN -> CLOSED (unexpected success in OPEN state)")
            }
            State.CLOSED -> { /* normal */ }
        }
    }

    @Synchronized
    fun recordFailure() {
        totalFailures++
        consecutiveFailures++
        consecutiveSuccesses = 0

        when (state) {
            State.CLOSED -> {
                if (consecutiveFailures >= failureThreshold) {
                    state = State.OPEN
                    openedAtMs = SystemClock.elapsedRealtime()
                    Log.w(TAG, "[$modelId] Circuit breaker: CLOSED -> OPEN ($consecutiveFailures consecutive failures)")
                }
            }
            State.HALF_OPEN -> {
                state = State.OPEN
                openedAtMs = SystemClock.elapsedRealtime()
                Log.w(TAG, "[$modelId] Circuit breaker: HALF_OPEN -> OPEN (probe inference failed)")
            }
            State.OPEN -> { /* already open */ }
        }
    }

    @Synchronized
    fun reset() {
        state = State.CLOSED
        consecutiveFailures = 0
        consecutiveSuccesses = 0
        totalFailures = 0
        totalSuccesses = 0
    }

    @Synchronized
    fun toTelemetry(): Map<String, Any> = mapOf(
        "circuitBreakerState" to state.name,
        "totalSuccesses" to totalSuccesses,
        "totalFailures" to totalFailures,
        "consecutiveFailures" to consecutiveFailures
    )

    companion object {
        private const val TAG = "ModelCircuitBreaker"
    }
}
