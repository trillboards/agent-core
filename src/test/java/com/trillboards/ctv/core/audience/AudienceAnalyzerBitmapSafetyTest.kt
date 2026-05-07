package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AudienceAnalyzerBitmapSafetyTest {

    @Test
    fun `recycled bitmap short-circuits before reading dimensions`() {
        var widthRead = false
        var heightRead = false

        val needsReinitialization = shouldReinitializeSharedBitmap(
            hasBitmap = true,
            isRecycled = { true },
            widthProvider = {
                widthRead = true
                throw AssertionError("width should not be read for a recycled bitmap")
            },
            heightProvider = {
                heightRead = true
                throw AssertionError("height should not be read for a recycled bitmap")
            },
            expectedWidth = 640,
            expectedHeight = 480
        )

        assertTrue(needsReinitialization)
        assertFalse(widthRead)
        assertFalse(heightRead)
    }

    @Test
    fun `healthy bitmap with matching dimensions keeps existing buffers`() {
        val needsReinitialization = shouldReinitializeSharedBitmap(
            hasBitmap = true,
            isRecycled = { false },
            widthProvider = { 640 },
            heightProvider = { 480 },
            expectedWidth = 640,
            expectedHeight = 480
        )

        assertFalse(needsReinitialization)
    }

    @Test
    fun `dimension mismatch requests shared bitmap reinitialization`() {
        val needsReinitialization = shouldReinitializeSharedBitmap(
            hasBitmap = true,
            isRecycled = { false },
            widthProvider = { 1280 },
            heightProvider = { 720 },
            expectedWidth = 640,
            expectedHeight = 480
        )

        assertTrue(needsReinitialization)
    }

    // ---- New tests covering the reinit-cause classification (regression for
    //      "expected nullxnull" warning spam on Tab S11) ----

    @Test
    fun `null bitmap is classified as NEEDS_ALLOCATION not RESOLUTION_MISMATCH`() {
        // When sharedBitmap is null, this isn't a resolution mismatch — it's
        // simply that buffers haven't been allocated yet. The caller should
        // treat this as informational, not a HAL-vs-target dimension surprise.
        val cause = classifyReinitCause(
            hasBitmap = false,
            isRecycled = { false },
            widthProvider = { throw AssertionError("dims should not be read when bitmap is null") },
            heightProvider = { throw AssertionError("dims should not be read when bitmap is null") },
            expectedWidth = 512,
            expectedHeight = 384
        )
        assertEquals(SharedBitmapReinitCause.NEEDS_ALLOCATION, cause)
    }

    @Test
    fun `recycled bitmap is classified as NEEDS_ALLOCATION not RESOLUTION_MISMATCH`() {
        val cause = classifyReinitCause(
            hasBitmap = true,
            isRecycled = { true },
            widthProvider = { throw AssertionError("dims should not be read when bitmap recycled") },
            heightProvider = { throw AssertionError("dims should not be read when bitmap recycled") },
            expectedWidth = 512,
            expectedHeight = 384
        )
        assertEquals(SharedBitmapReinitCause.NEEDS_ALLOCATION, cause)
    }

    @Test
    fun `live bitmap with mismatched dims is classified as RESOLUTION_MISMATCH`() {
        val cause = classifyReinitCause(
            hasBitmap = true,
            isRecycled = { false },
            widthProvider = { 480 },
            heightProvider = { 360 },
            expectedWidth = 512,
            expectedHeight = 384
        )
        assertEquals(SharedBitmapReinitCause.RESOLUTION_MISMATCH, cause)
    }

    @Test
    fun `live bitmap with matching dims is classified as NONE`() {
        val cause = classifyReinitCause(
            hasBitmap = true,
            isRecycled = { false },
            widthProvider = { 512 },
            heightProvider = { 384 },
            expectedWidth = 512,
            expectedHeight = 384
        )
        assertEquals(SharedBitmapReinitCause.NONE, cause)
    }
}
