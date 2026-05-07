package com.trillboards.ctv.core.inference.vlm

/**
 * Prevents the service from copying camera bitmaps when the VLM cannot accept
 * another frame yet. This keeps large transient bitmap allocations off the hot
 * path while a slow model is still inferring.
 */
internal class VlmFrameAdmissionGate {
    @Volatile
    private var inferenceInFlight = false

    @Volatile
    private var inFlightStartedAtMs = 0L

    @Volatile
    private var nextAllowedAtMs = 0L

    @Synchronized
    fun tryAcquire(nowMs: Long, minIntervalMs: Long, maxInFlightMs: Long = 0L): Boolean {
        if (inferenceInFlight) {
            val stale = maxInFlightMs > 0L &&
                inFlightStartedAtMs > 0L &&
                nowMs - inFlightStartedAtMs > maxInFlightMs
            if (stale) {
                inferenceInFlight = false
                inFlightStartedAtMs = 0L
            } else {
                return false
            }
        }
        if (nowMs < nextAllowedAtMs) return false

        inferenceInFlight = true
        inFlightStartedAtMs = nowMs
        nextAllowedAtMs = nowMs + minIntervalMs.coerceAtLeast(0L)
        return true
    }

    @Synchronized
    fun release() {
        inferenceInFlight = false
        inFlightStartedAtMs = 0L
    }

    @Synchronized
    fun reset() {
        inferenceInFlight = false
        inFlightStartedAtMs = 0L
        nextAllowedAtMs = 0L
    }

    @Synchronized
    fun inFlightAgeMs(nowMs: Long): Long? {
        if (!inferenceInFlight || inFlightStartedAtMs <= 0L) return null
        return (nowMs - inFlightStartedAtMs).coerceAtLeast(0L)
    }
}
