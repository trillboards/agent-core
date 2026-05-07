package com.trillboards.ctv.core.sensing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the canonical ADR-018 CSI parser used by the full tablet agent.
 *
 * The old parser expected a fixed little-endian header. The shipped CSI stack
 * uses the Rust-owned big-endian variable-length format, so these tests lock
 * the full agent back onto the canonical wire contract.
 */
class CsiUdpListenerTest {

    private fun buildTestFrame(
        nodeId: String = "1",
        nSubcarriers: Int = 4,
        freqMhz: Int = 2412,
        sequence: Int = 42,
        rssi: Int = -45,
        noiseFloor: Int = -90,
        iqPairs: List<Pair<Int, Int>>? = null
    ): ByteArray {
        val nodeBytes = nodeId.toByteArray(Charsets.UTF_8)
        val pairs = iqPairs ?: (0 until nSubcarriers).map { Pair(10, 20) }
        require(pairs.size == nSubcarriers) {
            "iqPairs.size (${pairs.size}) must match nSubcarriers ($nSubcarriers)"
        }

        val payloadLen = 1 + nodeBytes.size + 7 + (nSubcarriers * 2)
        val totalSize = 6 + payloadLen
        val buf = ByteArray(totalSize)

        // Magic (u32, big-endian)
        buf[0] = 0xC5.toByte()
        buf[1] = 0x11
        buf[2] = 0x00
        buf[3] = 0x01

        // payload_len (u16, big-endian)
        buf[4] = ((payloadLen ushr 8) and 0xFF).toByte()
        buf[5] = (payloadLen and 0xFF).toByte()

        // node_id_len + node_id bytes
        buf[6] = nodeBytes.size.toByte()
        System.arraycopy(nodeBytes, 0, buf, 7, nodeBytes.size)

        val base = 7 + nodeBytes.size
        buf[base] = nSubcarriers.toByte()
        buf[base + 1] = ((freqMhz ushr 8) and 0xFF).toByte()
        buf[base + 2] = (freqMhz and 0xFF).toByte()
        buf[base + 3] = ((sequence ushr 8) and 0xFF).toByte()
        buf[base + 4] = (sequence and 0xFF).toByte()
        buf[base + 5] = rssi.toByte()
        buf[base + 6] = noiseFloor.toByte()

        var offset = base + 7
        for ((i, q) in pairs) {
            buf[offset] = i.toByte()
            buf[offset + 1] = q.toByte()
            offset += 2
        }

        return buf
    }

    @Test
    fun `parseFrame should parse canonical ADR-018 frame`() {
        val buf = buildTestFrame(nodeId = "trillboards-csi-7")
        val frame = CsiUdpListener.parseFrame(buf, buf.size)

        assertNotNull(frame)
        assertEquals(7, frame!!.nodeId)
        assertEquals(1, frame.nAntennas)
        assertEquals(4, frame.nSubcarriers)
        assertEquals(2412, frame.freqMhz)
        assertEquals(42, frame.sequence)
        assertEquals(-45, frame.rssi)
        assertEquals(-90, frame.noiseFloor)
        assertEquals(4, frame.amplitudes.size)
        assertEquals(0, frame.phases.size)
    }

    @Test
    fun `parseFrame should reject buffer shorter than minimum frame size`() {
        val frame = CsiUdpListener.parseFrame(ByteArray(CsiUdpListener.MIN_FRAME_SIZE - 1), CsiUdpListener.MIN_FRAME_SIZE - 1)
        assertNull(frame)
    }

    @Test
    fun `parseFrame should reject invalid magic number`() {
        val buf = buildTestFrame()
        buf[0] = 0x00
        val frame = CsiUdpListener.parseFrame(buf, buf.size)
        assertNull(frame)
    }

    @Test
    fun `parseFrame should reject zero-length node id`() {
        val buf = buildTestFrame(nodeId = "1")
        buf[6] = 0
        val frame = CsiUdpListener.parseFrame(buf, buf.size)
        assertNull(frame)
    }

    @Test
    fun `parseFrame should reject oversized node id`() {
        val longId = "x".repeat(CsiUdpListener.MAX_NODE_ID_LEN + 1)
        val buf = buildTestFrame(nodeId = longId, nSubcarriers = 1, iqPairs = listOf(Pair(1, 1)))
        val frame = CsiUdpListener.parseFrame(buf, buf.size)
        assertNull(frame)
    }

    @Test
    fun `parseFrame should reject truncated payload`() {
        val buf = buildTestFrame(nSubcarriers = 4)
        val frame = CsiUdpListener.parseFrame(buf, buf.size - 2)
        assertNull(frame)
    }

    @Test
    fun `parseFrame should compute amplitude from IQ pairs`() {
        val frame = CsiUdpListener.parseFrame(
            buildTestFrame(nSubcarriers = 1, iqPairs = listOf(Pair(3, 4))),
            buildTestFrame(nSubcarriers = 1, iqPairs = listOf(Pair(3, 4))).size
        )

        assertNotNull(frame)
        assertEquals(5.0f, frame!!.amplitudes[0], 0.001f)
        assertEquals(0, frame.phases.size)
    }

    @Test
    fun `parseFrame should support zero subcarriers`() {
        val buf = buildTestFrame(nSubcarriers = 0, iqPairs = emptyList())
        val frame = CsiUdpListener.parseFrame(buf, buf.size)

        assertNotNull(frame)
        assertEquals(0, frame!!.nSubcarriers)
        assertTrue(frame.amplitudes.isEmpty())
    }

    @Test
    fun `parseFrame should extract digits from textual node ids`() {
        val frame = CsiUdpListener.parseFrame(buildTestFrame(nodeId = "node-99", nSubcarriers = 1), buildTestFrame(nodeId = "node-99", nSubcarriers = 1).size)
        assertNotNull(frame)
        assertEquals(99, frame!!.nodeId)
    }

    @Test
    fun `constants should match canonical parser contract`() {
        assertEquals(5005, CsiUdpListener.DEFAULT_PORT)
        assertEquals(0xC5110001L, CsiUdpListener.MAGIC)
        assertEquals(15, CsiUdpListener.MIN_FRAME_SIZE)
        assertEquals(15, CsiUdpListener.HEADER_SIZE)
        assertEquals(2048, CsiUdpListener.MAX_DATAGRAM_SIZE)
        assertEquals(6000, CsiUdpListener.MAX_BUFFER_FRAMES)
        assertEquals(32, CsiUdpListener.MAX_NODE_ID_LEN)
    }

    @Test
    fun `new listener should start idle`() {
        val listener = CsiUdpListener()
        assertTrue(!listener.isRunning())
        assertEquals(0L, listener.getTotalFramesReceived())
        assertEquals(0L, listener.getTotalFramesDropped())
        assertTrue(listener.getRecentFrames().isEmpty())
    }
}
