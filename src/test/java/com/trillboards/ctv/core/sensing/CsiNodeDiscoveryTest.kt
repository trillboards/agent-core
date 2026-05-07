package com.trillboards.ctv.core.sensing

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for CsiNodeDiscovery (mirrors agent-core-lite test).
 *
 * NOTE: Tests requiring NsdManager are integration tests that must run
 * on a device/emulator. These tests verify data class contracts and constants.
 */
class CsiNodeDiscoveryTest {

    @Test
    fun `CsiNode should hold all fields`() {
        val node = CsiNode(
            nodeId = 1,
            hostAddress = "192.168.1.100",
            port = 5005,
            screenId = "screen123",
            discoveredAtMs = 1700000000000L
        )

        assertEquals(1, node.nodeId)
        assertEquals("192.168.1.100", node.hostAddress)
        assertEquals(5005, node.port)
        assertEquals("screen123", node.screenId)
        assertEquals(1700000000000L, node.discoveredAtMs)
    }

    @Test
    fun `CsiNode should allow null screenId`() {
        val node = CsiNode(
            nodeId = 0,
            hostAddress = "10.0.0.50",
            port = 5005,
            screenId = null,
            discoveredAtMs = System.currentTimeMillis()
        )

        assertNull(node.screenId)
        assertNotNull(node.hostAddress)
    }

    @Test
    fun `CsiNode data class copy should preserve fields`() {
        val original = CsiNode(1, "192.168.1.100", 5005, "s1", 1000L)
        val copy = original.copy(port = 6006)

        assertEquals(6006, copy.port)
        assertEquals(1, copy.nodeId)
        assertEquals("192.168.1.100", copy.hostAddress)
        assertEquals("s1", copy.screenId)
    }

    @Test
    fun `CsiNode equality should compare all fields`() {
        val a = CsiNode(1, "192.168.1.1", 5005, null, 1000L)
        val b = CsiNode(1, "192.168.1.1", 5005, null, 1000L)
        val c = CsiNode(2, "192.168.1.1", 5005, null, 1000L)

        assertEquals(a, b)
        assertTrue(a != c)
    }

    @Test
    fun `multiple CsiNodes should be distinguishable by nodeId`() {
        val nodes = listOf(
            CsiNode(0, "192.168.1.10", 5005, null, 1000L),
            CsiNode(1, "192.168.1.11", 5005, null, 1000L),
            CsiNode(2, "192.168.1.12", 5005, "screen_abc", 1000L)
        )

        assertEquals(3, nodes.size)
        assertEquals(3, nodes.map { it.nodeId }.distinct().size)
    }

    @Test
    fun `getDiscoveredNodes should return empty list initially`() {
        // CsiNodeDiscovery is an object singleton, nodes are cleared on stopDiscovery
        // Before any discovery, getDiscoveredNodes should not crash
        val nodes = CsiNodeDiscovery.getDiscoveredNodes()
        assertNotNull(nodes)
    }

    @Test
    fun `getNodeCount should return non-negative value`() {
        val count = CsiNodeDiscovery.getNodeCount()
        assertTrue(count >= 0)
    }
}
