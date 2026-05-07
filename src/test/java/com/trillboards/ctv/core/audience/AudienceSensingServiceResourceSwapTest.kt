package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class AudienceSensingServiceResourceSwapTest {

    @Test
    fun `releases the previous processor when replacing it`() {
        val released = mutableListOf<String>()
        val current = "current"
        val next = "next"

        val result = swapReleasedReference(current, next) { released += it }

        assertSame(next, result)
        assertEquals(listOf("current"), released)
    }

    @Test
    fun `does not release when reusing the same processor`() {
        val released = mutableListOf<String>()
        val current = "same"

        val result = swapReleasedReference(current, current) { released += it }

        assertSame(current, result)
        assertEquals(emptyList<String>(), released)
    }

    @Test
    fun `does not release when there is no previous processor`() {
        val released = mutableListOf<String>()

        val result = swapReleasedReference<String>(null, "next") { released += it }

        assertEquals("next", result)
        assertEquals(emptyList<String>(), released)
    }

    @Test
    fun `wires existing capture immediately when camera is already bound`() {
        var wiredCapture: String? = null

        val wired = wireExistingCaptureIfReady(
            isCameraBound = true,
            existingCapture = "capture"
        ) { wiredCapture = it }

        assertTrue(wired)
        assertEquals("capture", wiredCapture)
    }

    @Test
    fun `does not wire existing capture when camera is not yet bound`() {
        var wiredCapture: String? = null

        val wired = wireExistingCaptureIfReady(
            isCameraBound = false,
            existingCapture = "capture"
        ) { wiredCapture = it }

        assertFalse(wired)
        assertEquals(null, wiredCapture)
    }
}
