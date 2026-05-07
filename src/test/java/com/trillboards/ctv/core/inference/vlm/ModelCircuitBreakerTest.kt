package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Tests for ModelCircuitBreaker state machine.
 *
 * State machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
 *
 * Note: These tests use a custom SystemClock provider via the
 * halfOpenRetryIntervalMs parameter set to 0ms for instant transitions,
 * or they test the failure/success counting which doesn't depend on time.
 */
class ModelCircuitBreakerTest {

    private lateinit var breaker: ModelCircuitBreaker

    @Before
    fun setup() {
        // Use threshold of 3 for faster tests, 0ms cooldown for instant HALF_OPEN transition
        breaker = ModelCircuitBreaker(
            modelId = "test_model",
            failureThreshold = 3,
            halfOpenRetryIntervalMs = 0L, // instant cooldown for testing
            halfOpenSuccessThreshold = 2
        )
    }

    // --- Initial state ---

    @Test
    fun `initial state is CLOSED`() {
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
    }

    @Test
    fun `initial state allows inference`() {
        assertTrue(breaker.isAllowed)
    }

    // --- CLOSED -> OPEN transition ---

    @Test
    fun `opens after failureThreshold consecutive failures`() {
        breaker.recordFailure()
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
        breaker.recordFailure()
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
        breaker.recordFailure() // 3rd failure = threshold
        assertEquals(ModelCircuitBreaker.State.OPEN, breaker.state)
    }

    @Test
    fun `does not open before failureThreshold`() {
        breaker.recordFailure()
        breaker.recordFailure()
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
        assertTrue(breaker.isAllowed)
    }

    @Test
    fun `success resets consecutive failure count`() {
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordSuccess() // resets consecutive failures
        breaker.recordFailure()
        breaker.recordFailure()
        // Should still be CLOSED — only 2 consecutive failures
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
    }

    // --- OPEN state blocks inference ---

    @Test
    fun `blocks inference when OPEN with non-zero cooldown`() {
        // Use a breaker with long cooldown so it stays OPEN
        val longCooldown = ModelCircuitBreaker(
            modelId = "test",
            failureThreshold = 3,
            halfOpenRetryIntervalMs = 300_000L, // 5 minutes
            halfOpenSuccessThreshold = 2
        )
        longCooldown.recordFailure()
        longCooldown.recordFailure()
        longCooldown.recordFailure()
        assertEquals(ModelCircuitBreaker.State.OPEN, longCooldown.state)
        assertFalse(longCooldown.isAllowed)
    }

    // --- OPEN -> HALF_OPEN transition ---

    @Test
    fun `transitions to HALF_OPEN after cooldown`() {
        // With 0ms cooldown, checking isAllowed immediately transitions
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()
        assertEquals(ModelCircuitBreaker.State.OPEN, breaker.state)

        // With 0ms cooldown, next isAllowed check transitions to HALF_OPEN
        assertTrue(breaker.isAllowed)
        assertEquals(ModelCircuitBreaker.State.HALF_OPEN, breaker.state)
    }

    // --- HALF_OPEN -> CLOSED transition ---

    @Test
    fun `closes after halfOpenSuccessThreshold consecutive successes in HALF_OPEN`() {
        // Open the breaker
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()

        // Transition to HALF_OPEN
        breaker.isAllowed // triggers transition with 0ms cooldown

        // First success — still HALF_OPEN
        breaker.recordSuccess()
        assertEquals(ModelCircuitBreaker.State.HALF_OPEN, breaker.state)

        // Second success — closes
        breaker.recordSuccess()
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
    }

    @Test
    fun `allows inference in HALF_OPEN`() {
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.isAllowed // transition to HALF_OPEN
        assertTrue(breaker.isAllowed)
    }

    // --- HALF_OPEN -> OPEN re-open ---

    @Test
    fun `re-opens on failure in HALF_OPEN`() {
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.isAllowed // transition to HALF_OPEN
        assertEquals(ModelCircuitBreaker.State.HALF_OPEN, breaker.state)

        breaker.recordFailure() // probe failure
        assertEquals(ModelCircuitBreaker.State.OPEN, breaker.state)
    }

    // --- Reset ---

    @Test
    fun `reset returns to CLOSED with zeroed counters`() {
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()
        assertEquals(ModelCircuitBreaker.State.OPEN, breaker.state)

        breaker.reset()
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)
        assertTrue(breaker.isAllowed)

        val telemetry = breaker.toTelemetry()
        assertEquals(0, telemetry["totalSuccesses"])
        assertEquals(0, telemetry["totalFailures"])
        assertEquals(0, telemetry["consecutiveFailures"])
    }

    // --- Telemetry ---

    @Test
    fun `toTelemetry returns correct map`() {
        breaker.recordSuccess()
        breaker.recordSuccess()
        breaker.recordFailure()

        val telemetry = breaker.toTelemetry()
        assertEquals("CLOSED", telemetry["circuitBreakerState"])
        assertEquals(2, telemetry["totalSuccesses"])
        assertEquals(1, telemetry["totalFailures"])
        assertEquals(1, telemetry["consecutiveFailures"])
    }

    @Test
    fun `toTelemetry reflects OPEN state`() {
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()

        val telemetry = breaker.toTelemetry()
        assertEquals("OPEN", telemetry["circuitBreakerState"])
        assertEquals(0, telemetry["totalSuccesses"])
        assertEquals(3, telemetry["totalFailures"])
        assertEquals(3, telemetry["consecutiveFailures"])
    }

    @Test
    fun `toTelemetry reflects HALF_OPEN state`() {
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.recordFailure()
        breaker.isAllowed // transition to HALF_OPEN

        val telemetry = breaker.toTelemetry()
        assertEquals("HALF_OPEN", telemetry["circuitBreakerState"])
    }

    // --- Edge cases ---

    @Test
    fun `multiple failures beyond threshold stay OPEN`() {
        for (i in 1..10) {
            breaker.recordFailure()
        }
        assertEquals(ModelCircuitBreaker.State.OPEN, breaker.state)

        val telemetry = breaker.toTelemetry()
        assertEquals(10, telemetry["totalFailures"])
    }

    @Test
    fun `success in CLOSED state is normal operation`() {
        breaker.recordSuccess()
        breaker.recordSuccess()
        breaker.recordSuccess()
        assertEquals(ModelCircuitBreaker.State.CLOSED, breaker.state)

        val telemetry = breaker.toTelemetry()
        assertEquals(3, telemetry["totalSuccesses"])
        assertEquals(0, telemetry["consecutiveFailures"])
    }
}
