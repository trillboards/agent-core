package com.trillboards.ctv.core.sensing

/**
 * Parsed CSI frame from an ESP32 sensing node.
 *
 * Wire-format and field semantics match agent-core-lite's [CsiFrame] one-to-one
 * (ADR-018 binary protocol). The shape lives independently in agent-core because
 * the two modules are never co-resident in a single APK (tablet-agent depends on
 * agent-core; tablet-agent-lite/fire-tv-agent depend on agent-core-lite).
 *
 * The actual UDP socket and parser remain in agent-core-lite. agent-core consumes
 * frames through the [CsiFrameSource] interface — production callers wire whichever
 * concrete listener is available in their flavor.
 */
data class CsiFrame(
    val nodeId: Int,
    val nAntennas: Int,
    val nSubcarriers: Int,
    val freqMhz: Int,
    val sequence: Int,
    val rssi: Int,
    val noiseFloor: Int,
    val amplitudes: FloatArray,
    val phases: FloatArray,
    val receivedAtMs: Long
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CsiFrame) return false
        return nodeId == other.nodeId && sequence == other.sequence
    }

    override fun hashCode(): Int = 31 * nodeId + sequence
}

/**
 * Minimal abstraction over a CSI frame buffer.
 *
 * Implementations include:
 *   - agent-core-lite's CsiUdpListener (production wiring via adapter)
 *   - test fakes that pre-load deterministic frames
 *
 * The collector only needs read access — it never starts or stops the underlying
 * socket. Lifecycle is owned by the caller that constructs the source.
 */
interface CsiFrameSource {
    /**
     * Return frames whose `receivedAtMs >= sinceMs`.
     *
     * @param sinceMs Epoch millis lower bound. Pass 0 to return all buffered frames.
     */
    fun getRecentFrames(sinceMs: Long = 0L): List<CsiFrame>

    /**
     * Drop all buffered frames. Optional — implementations may no-op if the buffer
     * is owned externally.
     */
    fun clearBuffer() {}
}
