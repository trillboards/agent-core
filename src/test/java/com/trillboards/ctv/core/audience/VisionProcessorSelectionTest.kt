package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VisionProcessorSelectionTest {

    @Test
    fun `empty profile disables all optional camera processors`() {
        val selection = VisionProcessorSelection.fromModels(emptyList())

        assertFalse(selection.wantsPose)
        assertFalse(selection.wantsEmotion)
        assertFalse(selection.wantsAgeGender)
        assertFalse(selection.wantsPersonDetection)
        assertFalse(selection.wantsGaze)
        assertEquals(emptyList<String>(), selection.activeModelIds())
    }

    @Test
    fun `movenet enables pose and gaze`() {
        val selection = VisionProcessorSelection.fromModels(listOf("movenet"))

        assertTrue(selection.wantsPose)
        assertFalse(selection.wantsEmotion)
        assertFalse(selection.wantsAgeGender)
        assertFalse(selection.wantsPersonDetection)
        assertTrue(selection.wantsGaze)
        assertEquals(listOf("movenet", "gaze"), selection.activeModelIds())
    }

    @Test
    fun `fer plus enables emotion and gaze`() {
        val selection = VisionProcessorSelection.fromModels(listOf("fer_plus"))

        assertFalse(selection.wantsPose)
        assertTrue(selection.wantsEmotion)
        assertFalse(selection.wantsAgeGender)
        assertFalse(selection.wantsPersonDetection)
        assertTrue(selection.wantsGaze)
        assertEquals(listOf("fer_plus", "gaze"), selection.activeModelIds())
    }

    @Test
    fun `age gender and efficientdet stay independent`() {
        val selection = VisionProcessorSelection.fromModels(listOf("age_gender", "efficientdet"))

        assertFalse(selection.wantsPose)
        assertFalse(selection.wantsEmotion)
        assertTrue(selection.wantsAgeGender)
        assertTrue(selection.wantsPersonDetection)
        assertFalse(selection.wantsGaze)
        assertEquals(listOf("age_gender", "efficientdet"), selection.activeModelIds())
    }

    @Test
    fun `camera presence profiles enable engagement stack and person detection even without explicit legacy ids`() {
        val selection = VisionProcessorSelection.fromModels(listOf("gemma_4_e2b", "blazeface", "yamnet"))

        assertTrue(selection.wantsPose)
        assertTrue(selection.wantsEmotion)
        assertFalse(selection.wantsAgeGender)
        assertTrue(selection.wantsPersonDetection)
        assertTrue(selection.wantsGaze)
        assertEquals(listOf("fer_plus", "movenet", "efficientdet", "gaze"), selection.activeModelIds())
    }

    @Test
    fun `explicit contract can disable camera presence baseline processors`() {
        val selection = VisionProcessorSelection.fromModels(
            appliedModels = listOf("blazeface", "gemma_4_e2b"),
            enableCameraPresenceBaseline = false
        )

        assertFalse(selection.wantsPose)
        assertFalse(selection.wantsEmotion)
        assertFalse(selection.wantsAgeGender)
        assertFalse(selection.wantsPersonDetection)
        assertFalse(selection.wantsGaze)
        assertEquals(emptyList<String>(), selection.activeModelIds())
    }
}
