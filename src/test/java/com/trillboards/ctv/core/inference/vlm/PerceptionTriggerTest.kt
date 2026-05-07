package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [PerceptionTrigger].
 *
 * Uses a controllable clock to test time-dependent trigger logic without
 * relying on real SystemClock (which returns 0 in JVM unit tests).
 */
class PerceptionTriggerTest {

    /** Controllable clock for deterministic time-based testing. */
    private var currentTimeMs = 10_000L  // Start at 10s to avoid edge cases at 0
    private val clock: () -> Long = { currentTimeMs }

    private lateinit var trigger: PerceptionTrigger

    @Before
    fun setUp() {
        currentTimeMs = 10_000L
        trigger = PerceptionTrigger(clock = clock)
    }

    // --- First observation ---

    @Test
    fun `first observation always triggers`() {
        val state = PerceptionTrigger.SceneState(personCount = 2, noiseLevel = 0.5f)
        assertTrue("First observation should always trigger VLM", trigger.shouldTriggerVLM(state))
    }

    @Test
    fun `first observation with empty state triggers`() {
        val state = PerceptionTrigger.SceneState()
        assertTrue("First observation should trigger even with empty state", trigger.shouldTriggerVLM(state))
    }

    // --- Min interval throttling ---

    @Test
    fun `same state within min interval does not trigger`() {
        val state = PerceptionTrigger.SceneState(personCount = 2, noiseLevel = 0.5f)

        // First observation triggers
        assertTrue(trigger.shouldTriggerVLM(state))
        trigger.markTriggered(state)

        // Advance time but stay within min interval (5s default)
        currentTimeMs += 2_000L  // 2s later

        // Same state, within min interval — should NOT trigger
        assertFalse(
            "Should not trigger within min interval even with state change",
            trigger.shouldTriggerVLM(state)
        )
    }

    @Test
    fun `state change within min interval does not trigger`() {
        val state1 = PerceptionTrigger.SceneState(personCount = 2)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        // Only 1 second later — within min interval
        currentTimeMs += 1_000L

        val state2 = PerceptionTrigger.SceneState(personCount = 5)  // Different!
        assertFalse(
            "State change within min interval should be suppressed",
            trigger.shouldTriggerVLM(state2)
        )
    }

    // --- Max interval periodic refresh ---

    @Test
    fun `same state after max interval triggers periodic refresh`() {
        val state = PerceptionTrigger.SceneState(personCount = 2, noiseLevel = 0.5f)

        // First trigger
        assertTrue(trigger.shouldTriggerVLM(state))
        trigger.markTriggered(state)

        // Advance past max interval (30s default)
        currentTimeMs += 31_000L

        // Same state but max interval exceeded — should trigger for periodic refresh
        assertTrue(
            "Should trigger periodic refresh after max interval",
            trigger.shouldTriggerVLM(state)
        )
    }

    @Test
    fun `exactly at max interval triggers`() {
        val state = PerceptionTrigger.SceneState(personCount = 1)

        assertTrue(trigger.shouldTriggerVLM(state))
        trigger.markTriggered(state)

        // Exactly at max interval boundary
        currentTimeMs += PerceptionTrigger.DEFAULT_MAX_TRIGGER_INTERVAL_MS

        assertTrue(
            "Should trigger at exactly max interval boundary",
            trigger.shouldTriggerVLM(state)
        )
    }

    // --- Person count changes ---

    @Test
    fun `person count increase triggers`() {
        val state1 = PerceptionTrigger.SceneState(personCount = 2)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        // Advance past min interval
        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(personCount = 4)
        assertTrue("Person count increase should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `person count decrease triggers`() {
        val state1 = PerceptionTrigger.SceneState(personCount = 5)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(personCount = 1)
        assertTrue("Person count decrease should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `person count zero to nonzero triggers`() {
        val state1 = PerceptionTrigger.SceneState(personCount = 0)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(personCount = 1)
        assertTrue("First person arriving should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    // --- Object count changes ---

    @Test
    fun `new object type triggers`() {
        val state1 = PerceptionTrigger.SceneState(
            objectCounts = mapOf("person" to 1)
        )

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(
            objectCounts = mapOf("person" to 1, "backpack" to 1)
        )
        assertTrue("New object type should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `object count change triggers`() {
        val state1 = PerceptionTrigger.SceneState(
            objectCounts = mapOf("person" to 1, "chair" to 2)
        )

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(
            objectCounts = mapOf("person" to 1, "chair" to 3)
        )
        assertTrue("Object count change should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `object disappearing triggers`() {
        val state1 = PerceptionTrigger.SceneState(
            objectCounts = mapOf("person" to 2, "bag" to 1)
        )

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(
            objectCounts = mapOf("person" to 2)
        )
        assertTrue("Object disappearing should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    // --- Emotion changes ---

    @Test
    fun `emotion change triggers`() {
        val state1 = PerceptionTrigger.SceneState(dominantEmotion = "NEUTRAL")

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(dominantEmotion = "HAPPY")
        assertTrue("Emotion change should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `emotion from null to value triggers`() {
        val state1 = PerceptionTrigger.SceneState(dominantEmotion = null)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(dominantEmotion = "SURPRISED")
        assertTrue("Emotion appearing (null → value) should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `emotion from value to null triggers`() {
        val state1 = PerceptionTrigger.SceneState(dominantEmotion = "HAPPY")

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(dominantEmotion = null)
        assertTrue("Emotion disappearing (value → null) should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    // --- Noise level changes ---

    @Test
    fun `noise level large change triggers`() {
        val state1 = PerceptionTrigger.SceneState(noiseLevel = 0.2f)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(noiseLevel = 0.8f)  // delta = 0.6 > 0.3
        assertTrue("Large noise level change should trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `noise level exactly at threshold triggers`() {
        val state1 = PerceptionTrigger.SceneState(noiseLevel = 0.0f)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        // delta = 0.31 which is > 0.3 threshold
        val state2 = PerceptionTrigger.SceneState(noiseLevel = 0.31f)
        assertTrue("Noise change just over threshold should trigger", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `small noise change does not trigger`() {
        val state1 = PerceptionTrigger.SceneState(noiseLevel = 0.5f)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        val state2 = PerceptionTrigger.SceneState(noiseLevel = 0.6f)  // delta = 0.1 < 0.3
        assertFalse("Small noise change should NOT trigger VLM", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `noise at exactly threshold does not trigger`() {
        val state1 = PerceptionTrigger.SceneState(noiseLevel = 0.5f)

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        // delta = 0.3 which is NOT > 0.3 (must exceed, not equal)
        val state2 = PerceptionTrigger.SceneState(noiseLevel = 0.8f)
        assertFalse("Noise change at exactly threshold should NOT trigger", trigger.shouldTriggerVLM(state2))
    }

    // --- No change scenarios ---

    @Test
    fun `identical state after min interval does not trigger`() {
        val state = PerceptionTrigger.SceneState(
            personCount = 3,
            objectCounts = mapOf("person" to 3),
            dominantEmotion = "NEUTRAL",
            noiseLevel = 0.4f
        )

        assertTrue(trigger.shouldTriggerVLM(state))
        trigger.markTriggered(state)

        // Past min interval but before max interval
        currentTimeMs += 10_000L  // 10s

        assertFalse(
            "Identical state should not trigger between min and max interval",
            trigger.shouldTriggerVLM(state)
        )
    }

    // --- Config updates ---

    @Test
    fun `updateConfig changes min interval`() {
        trigger.updateConfig(minIntervalMs = 10_000L)

        val state1 = PerceptionTrigger.SceneState(personCount = 1)
        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        // 7 seconds later — within new 10s min interval
        currentTimeMs += 7_000L

        val state2 = PerceptionTrigger.SceneState(personCount = 5)
        assertFalse(
            "Should respect updated min interval",
            trigger.shouldTriggerVLM(state2)
        )
    }

    @Test
    fun `updateConfig changes max interval`() {
        trigger.updateConfig(maxIntervalMs = 10_000L)

        val state = PerceptionTrigger.SceneState(personCount = 1)
        assertTrue(trigger.shouldTriggerVLM(state))
        trigger.markTriggered(state)

        // 11 seconds later — past new 10s max interval
        currentTimeMs += 11_000L

        assertTrue(
            "Should respect updated max interval",
            trigger.shouldTriggerVLM(state)
        )
    }

    @Test
    fun `updateConfig clamps min interval to bounds`() {
        trigger.updateConfig(minIntervalMs = 100L)  // Below 1000 floor

        val state1 = PerceptionTrigger.SceneState(personCount = 1)
        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        // 500ms later — should still be within clamped 1000ms min
        currentTimeMs += 500L

        val state2 = PerceptionTrigger.SceneState(personCount = 5)
        assertFalse(
            "Min interval should be clamped to 1000ms floor",
            trigger.shouldTriggerVLM(state2)
        )
    }

    // --- Telemetry ---

    @Test
    fun `telemetry tracks trigger and skip counts`() {
        val state1 = PerceptionTrigger.SceneState(personCount = 1)

        // First trigger
        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        // Skip (within min interval)
        currentTimeMs += 1_000L
        assertFalse(trigger.shouldTriggerVLM(state1))

        // Another skip
        currentTimeMs += 1_000L
        assertFalse(trigger.shouldTriggerVLM(state1))

        // Second trigger (state change after min interval)
        currentTimeMs += 5_000L
        val state2 = PerceptionTrigger.SceneState(personCount = 3)
        assertTrue(trigger.shouldTriggerVLM(state2))
        trigger.markTriggered(state2)

        val (triggers, skips) = trigger.getTelemetry()
        assertEquals("Should have 2 triggers", 2L, triggers)
        assertEquals("Should have 2 skips", 2L, skips)
    }

    // --- Reset ---

    @Test
    fun `reset clears state and allows immediate trigger`() {
        val state = PerceptionTrigger.SceneState(personCount = 2)

        assertTrue(trigger.shouldTriggerVLM(state))
        trigger.markTriggered(state)

        // Within min interval — would normally be blocked
        currentTimeMs += 1_000L
        assertFalse(trigger.shouldTriggerVLM(state))

        // Reset
        trigger.reset()

        // Now should trigger as first observation
        assertTrue("After reset, should trigger as first observation", trigger.shouldTriggerVLM(state))

        // Telemetry should be cleared
        val (triggers, skips) = trigger.getTelemetry()
        assertEquals("Triggers should be 0 after reset", 0L, triggers)
        assertEquals("Skips should be 0 after reset", 0L, skips)
    }

    // --- Multiple state change types ---

    @Test
    fun `multiple simultaneous changes trigger`() {
        val state1 = PerceptionTrigger.SceneState(
            personCount = 1,
            dominantEmotion = "NEUTRAL",
            noiseLevel = 0.2f
        )

        assertTrue(trigger.shouldTriggerVLM(state1))
        trigger.markTriggered(state1)

        currentTimeMs += 6_000L

        // Everything changed at once
        val state2 = PerceptionTrigger.SceneState(
            personCount = 5,
            dominantEmotion = "HAPPY",
            noiseLevel = 0.9f
        )
        assertTrue("Multiple simultaneous changes should trigger", trigger.shouldTriggerVLM(state2))
    }

    @Test
    fun `consecutive triggers with state changes`() {
        // Simulate a dynamic scene: person enters, emotion shifts, person leaves

        // Step 1: Empty scene
        val empty = PerceptionTrigger.SceneState(personCount = 0, dominantEmotion = null)
        assertTrue(trigger.shouldTriggerVLM(empty))
        trigger.markTriggered(empty)

        // Step 2: Person enters (6s later)
        currentTimeMs += 6_000L
        val oneArrived = PerceptionTrigger.SceneState(personCount = 1, dominantEmotion = "NEUTRAL")
        assertTrue("Person entering should trigger", trigger.shouldTriggerVLM(oneArrived))
        trigger.markTriggered(oneArrived)

        // Step 3: Emotion changes (6s later)
        currentTimeMs += 6_000L
        val happy = PerceptionTrigger.SceneState(personCount = 1, dominantEmotion = "HAPPY")
        assertTrue("Emotion change should trigger", trigger.shouldTriggerVLM(happy))
        trigger.markTriggered(happy)

        // Step 4: Person leaves (6s later)
        currentTimeMs += 6_000L
        val left = PerceptionTrigger.SceneState(personCount = 0, dominantEmotion = null)
        assertTrue("Person leaving should trigger", trigger.shouldTriggerVLM(left))
        trigger.markTriggered(left)

        val (triggers, _) = trigger.getTelemetry()
        assertEquals("Should have 4 triggers for dynamic scene", 4L, triggers)
    }
}
