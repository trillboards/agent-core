package com.trillboards.ctv.core.sensing

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.absoluteValue
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Callback interface for receiving parsed CSI frames.
 */
fun interface CsiFrameCallback {
    fun onFrame(frame: CsiFrame)
}

/**
 * UDP listener for ADR-018 CSI frames from ESP32 sensing nodes.
 *
 * Binds to a UDP port (default 5005) and continuously receives datagrams.
 * Each datagram is parsed according to the ADR-018 binary header format.
 * Valid frames are delivered to registered [CsiFrameCallback] listeners
 * and also stored in a bounded internal buffer for [CsiAggregator] to consume.
 *
 * Lifecycle: call [start] to begin listening, [stop] to shut down.
 * Uses Kotlin coroutines on [Dispatchers.IO] for non-blocking socket reads.
 *
 * Thread safety: frame buffer is a [CopyOnWriteArrayList], callbacks are
 * invoked on the IO dispatcher thread.
 */
class CsiUdpListener(
    private val port: Int = DEFAULT_PORT,
    private val maxBufferSize: Int = MAX_BUFFER_FRAMES
) {

    companion object {
        private const val TAG = "CsiUdpListener"
        const val DEFAULT_PORT = 5005
        const val MAGIC = 0xC5110001L
        const val MIN_FRAME_SIZE = 15
        const val HEADER_SIZE = MIN_FRAME_SIZE
        const val MAX_DATAGRAM_SIZE = 2048
        const val MAX_BUFFER_FRAMES = 6000  // ~60s at 100Hz
        const val MAX_NODE_ID_LEN = 32

        /**
         * Parse an ADR-018 CSI frame from a raw UDP datagram buffer.
         *
         * @param buf Raw datagram bytes
         * @param length Number of valid bytes in the buffer
         * @return Parsed [CsiFrame] or null if the datagram is invalid
         */
        fun parseFrame(buf: ByteArray, length: Int): CsiFrame? {
            if (length < MIN_FRAME_SIZE) return null

            val bb = ByteBuffer.wrap(buf, 0, length).order(ByteOrder.BIG_ENDIAN)

            // Canonical ADR-018: magic is big-endian u32.
            val magic = bb.getInt(0).toLong() and 0xFFFFFFFFL
            if (magic != MAGIC) return null

            val payloadLen = bb.getShort(4).toInt() and 0xFFFF
            if (length < 6 + payloadLen) return null

            val nodeIdLen = buf[6].toInt() and 0xFF
            if (nodeIdLen == 0 || nodeIdLen > MAX_NODE_ID_LEN) return null
            if (7 + nodeIdLen + 7 > length) return null

            val nodeIdText = try {
                String(buf, 7, nodeIdLen, Charsets.UTF_8)
            } catch (_: Exception) {
                return null
            }

            val base = 7 + nodeIdLen
            val nSubcarriers = buf[base].toInt() and 0xFF
            val freqMhz = bb.getShort(base + 1).toInt() and 0xFFFF
            val sequence = bb.getShort(base + 3).toInt() and 0xFFFF
            val rssi = buf[base + 5].toInt()
            val noiseFloor = buf[base + 6].toInt()

            val iqStart = base + 7
            val iqBytesNeeded = nSubcarriers * 2
            if (iqStart + iqBytesNeeded > length) return null

            val amplitudes = FloatArray(nSubcarriers)
            for (i in 0 until nSubcarriers) {
                val offset = iqStart + i * 2
                val iVal = buf[offset].toInt()
                val qVal = buf[offset + 1].toInt()
                amplitudes[i] = kotlin.math.sqrt((iVal * iVal + qVal * qVal).toFloat())
            }

            return CsiFrame(
                nodeId = extractNumericNodeId(nodeIdText),
                nAntennas = 1,
                nSubcarriers = nSubcarriers,
                freqMhz = freqMhz,
                sequence = sequence,
                rssi = rssi,
                noiseFloor = noiseFloor,
                amplitudes = amplitudes,
                phases = FloatArray(0),  // Phases unused — empty to avoid wasting memory
                receivedAtMs = System.currentTimeMillis()
            )
        }

        private fun extractNumericNodeId(nodeIdText: String): Int {
            nodeIdText.toIntOrNull()?.let { return it }
            Regex("(\\d+)").find(nodeIdText)?.groupValues?.getOrNull(1)?.toIntOrNull()?.let { return it }
            return (nodeIdText.hashCode().absoluteValue % 256)
        }
    }

    private var socket: DatagramSocket? = null
    private var listenJob: Job? = null
    private val callbacks = CopyOnWriteArrayList<CsiFrameCallback>()

    // Internal ring buffer of recent frames for aggregation
    private val frameBuffer = CopyOnWriteArrayList<CsiFrame>()

    @Volatile
    private var running = false

    @Volatile
    private var totalFramesReceived = 0L

    @Volatile
    private var totalFramesDropped = 0L

    /**
     * Start listening for CSI frames on the configured UDP port.
     *
     * @param scope CoroutineScope for the listener coroutine
     * @return true if started successfully, false if already running
     */
    fun start(scope: CoroutineScope): Boolean {
        if (running) {
            Log.d(TAG, "UDP listener already running on port $port")
            return false
        }

        running = true
        listenJob = scope.launch(Dispatchers.IO) {
            try {
                val dgSocket = DatagramSocket(null)
                dgSocket.reuseAddress = true
                dgSocket.bind(InetSocketAddress(port))
                dgSocket.soTimeout = 2000  // 2s timeout for graceful shutdown checks
                socket = dgSocket
                Log.d(TAG, "CSI UDP listener started on port $port")

                val buf = ByteArray(MAX_DATAGRAM_SIZE)
                val packet = DatagramPacket(buf, buf.size)

                while (isActive && running) {
                    try {
                        dgSocket.receive(packet)
                        val frame = parseFrame(buf, packet.length)
                        if (frame != null) {
                            totalFramesReceived++
                            addToBuffer(frame)
                            for (cb in callbacks) {
                                try {
                                    cb.onFrame(frame)
                                } catch (e: Exception) {
                                    Log.w(TAG, "Callback error: ${e.message}")
                                }
                            }
                        } else {
                            totalFramesDropped++
                        }
                    } catch (_: java.net.SocketTimeoutException) {
                        // Normal — allows loop to check running flag
                    } catch (e: Exception) {
                        if (running) {
                            Log.w(TAG, "UDP receive error: ${e.message}")
                            totalFramesDropped++
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to start UDP listener: ${e.message}")
            } finally {
                socket?.close()
                socket = null
                Log.d(TAG, "CSI UDP listener stopped")
            }
        }

        return true
    }

    /**
     * Stop the UDP listener and release resources.
     */
    fun stop() {
        running = false
        socket?.close()
        listenJob?.cancel()
        listenJob = null
        Log.d(TAG, "CSI UDP listener stop requested")
    }

    /**
     * Register a callback for receiving parsed CSI frames.
     */
    fun addCallback(callback: CsiFrameCallback) {
        callbacks.add(callback)
    }

    /**
     * Remove a previously registered callback.
     */
    fun removeCallback(callback: CsiFrameCallback) {
        callbacks.remove(callback)
    }

    /**
     * Get recent frames from the internal buffer.
     *
     * @param sinceMs Only return frames received after this epoch timestamp
     * @return List of frames matching the time filter
     */
    fun getRecentFrames(sinceMs: Long = 0): List<CsiFrame> {
        return if (sinceMs <= 0) {
            frameBuffer.toList()
        } else {
            frameBuffer.filter { it.receivedAtMs >= sinceMs }
        }
    }

    /**
     * Clear the internal frame buffer.
     */
    fun clearBuffer() {
        frameBuffer.clear()
    }

    /**
     * Total frames successfully parsed since listener start.
     */
    fun getTotalFramesReceived(): Long = totalFramesReceived

    /**
     * Total frames dropped (parse failure or receive error) since listener start.
     */
    fun getTotalFramesDropped(): Long = totalFramesDropped

    /**
     * Whether the listener is currently running.
     */
    fun isRunning(): Boolean = running

    /**
     * Add a frame to the ring buffer, evicting oldest entries if over capacity.
     */
    private fun addToBuffer(frame: CsiFrame) {
        frameBuffer.add(frame)
        // Trim oldest frames when buffer exceeds max size
        while (frameBuffer.size > maxBufferSize) {
            frameBuffer.removeAt(0)
        }
    }
}
