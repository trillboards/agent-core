package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for CameraHealthMonitor.
 *
 * Pure-function tests run on the JVM without Robolectric. The class itself
 * uses `android.util.Log` which is unavailable on the JVM, but the
 * pure-function helpers (`computeNextRetryDelayMs`, `shouldDevolveCap`,
 * `classifyFailure`, `shouldGiveUp`) are exposed via @JvmStatic on the
 * companion so we can validate the policy without instantiating the
 * class. The instance-level tests use a fake clock.
 */
class CameraHealthMonitorTest {

    // ---------- Pure-function policy tests ----------

    @Test
    fun `computeNextRetryDelayMs returns first delay for negative attempts`() {
        assertEquals(1_000L, CameraHealthMonitor.computeNextRetryDelayMs(-1))
        assertEquals(1_000L, CameraHealthMonitor.computeNextRetryDelayMs(-100))
    }

    @Test
    fun `computeNextRetryDelayMs follows expected backoff schedule`() {
        // 1s → 5s → 30s → 5min → 30min → 1h
        assertEquals(1_000L, CameraHealthMonitor.computeNextRetryDelayMs(0))
        assertEquals(5_000L, CameraHealthMonitor.computeNextRetryDelayMs(1))
        assertEquals(30_000L, CameraHealthMonitor.computeNextRetryDelayMs(2))
        assertEquals(5 * 60_000L, CameraHealthMonitor.computeNextRetryDelayMs(3))
        assertEquals(30 * 60_000L, CameraHealthMonitor.computeNextRetryDelayMs(4))
        assertEquals(60 * 60_000L, CameraHealthMonitor.computeNextRetryDelayMs(5))
    }

    @Test
    fun `computeNextRetryDelayMs holds at 1 hour past schedule end`() {
        assertEquals(60 * 60_000L, CameraHealthMonitor.computeNextRetryDelayMs(6))
        assertEquals(60 * 60_000L, CameraHealthMonitor.computeNextRetryDelayMs(50))
        assertEquals(60 * 60_000L, CameraHealthMonitor.computeNextRetryDelayMs(Int.MAX_VALUE - 1))
    }

    @Test
    fun `shouldDevolveCap is false when state is OPEN regardless of failedSince`() {
        assertFalse(CameraHealthMonitor.shouldDevolveCap(
            state = CameraHealthMonitor.CameraState.OPEN,
            failedSinceMs = 1L,
            nowMs = 100L * 24 * 60 * 60 * 1000
        ))
    }

    @Test
    fun `shouldDevolveCap is false when state is NEVER_STARTED`() {
        // Fresh boots that haven't tried yet should keep the declared capability.
        assertFalse(CameraHealthMonitor.shouldDevolveCap(
            state = CameraHealthMonitor.CameraState.NEVER_STARTED,
            failedSinceMs = 0L,
            nowMs = 100L * 24 * 60 * 60 * 1000
        ))
    }

    @Test
    fun `shouldDevolveCap is false when failedSinceMs is zero`() {
        assertFalse(CameraHealthMonitor.shouldDevolveCap(
            state = CameraHealthMonitor.CameraState.FAILED,
            failedSinceMs = 0L,
            nowMs = Long.MAX_VALUE
        ))
    }

    @Test
    fun `shouldDevolveCap is false when in FAILED for less than 24h`() {
        val nowMs = 1_000_000L
        val failedSince = nowMs - (12 * 60 * 60 * 1000)  // 12h ago
        assertFalse(CameraHealthMonitor.shouldDevolveCap(
            state = CameraHealthMonitor.CameraState.FAILED,
            failedSinceMs = failedSince,
            nowMs = nowMs
        ))
    }

    @Test
    fun `shouldDevolveCap is true when in FAILED for at least 24h - the Adam case`() {
        // Adam @ Focus Media: 30+ days of FAILED. Once the 24h threshold is
        // crossed the heartbeat starts reporting cap_face_detection=false
        // so the server's view of the device matches reality.
        val nowMs = 30L * 24 * 60 * 60 * 1000
        val failedSince = 0L + (1 * 60 * 60 * 1000)  // failed at hour 1; 30 days ago
        assertTrue(CameraHealthMonitor.shouldDevolveCap(
            state = CameraHealthMonitor.CameraState.FAILED,
            failedSinceMs = failedSince,
            nowMs = nowMs
        ))
    }

    @Test
    fun `shouldDevolveCap is exactly true at the 24h boundary`() {
        // failedSinceMs=0 is treated as "never failed" — use a small positive
        // value to represent the moment the camera entered FAILED state.
        val failedAt = 1L
        val nowMs = failedAt + (24L * 60 * 60 * 1000)
        assertTrue(CameraHealthMonitor.shouldDevolveCap(
            state = CameraHealthMonitor.CameraState.FAILED,
            failedSinceMs = failedAt,
            nowMs = nowMs
        ))
    }

    // ---------- classifyFailure tests ----------

    @Test
    fun `classifyFailure returns UNKNOWN for null`() {
        assertEquals(CameraHealthMonitor.FailureClass.UNKNOWN, CameraHealthMonitor.classifyFailure(null))
    }

    @Test
    fun `classifyFailure detects permission denial`() {
        val ex = SecurityException("Permission denied to access CAMERA")
        assertEquals(CameraHealthMonitor.FailureClass.PERMISSION_DENIED, CameraHealthMonitor.classifyFailure(ex))
    }

    @Test
    fun `classifyFailure detects camera unavailable`() {
        val ex = RuntimeException("CameraUnavailableException: camera is in use")
        assertEquals(CameraHealthMonitor.FailureClass.CAMERA_UNAVAILABLE, CameraHealthMonitor.classifyFailure(ex))
    }

    @Test
    fun `classifyFailure detects bind failure`() {
        val ex = IllegalStateException("Failed in bindToLifecycle: UseCase already attached")
        assertEquals(CameraHealthMonitor.FailureClass.BIND_FAILED, CameraHealthMonitor.classifyFailure(ex))
    }

    @Test
    fun `classifyFailure detects provider init failure`() {
        val ex = RuntimeException("ProcessCameraProvider.getInstance failed: InitializationException")
        assertEquals(CameraHealthMonitor.FailureClass.PROVIDER_INIT_FAILED, CameraHealthMonitor.classifyFailure(ex))
    }

    @Test
    fun `classifyFailure detects lens-facing query throw`() {
        val ex = RuntimeException("LENS_FACING characteristic threw")
        assertEquals(CameraHealthMonitor.FailureClass.LENS_FACING_QUERY_FAILED, CameraHealthMonitor.classifyFailure(ex))
    }

    // ---------- shouldGiveUp tests ----------

    @Test
    fun `shouldGiveUp never gives up on permission denial`() {
        assertFalse(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.PERMISSION_DENIED, 100, 5))
        assertFalse(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.PERMISSION_DENIED, 1000, 5))
    }

    @Test
    fun `shouldGiveUp gives up immediately on emulator`() {
        assertTrue(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.EMULATOR_OR_HEADLESS, 0, 5))
    }

    @Test
    fun `shouldGiveUp gives up after 1 attempt for no hardware`() {
        assertFalse(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.NO_CAMERA_HARDWARE, 0, 5))
        assertTrue(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.NO_CAMERA_HARDWARE, 1, 5))
    }

    @Test
    fun `shouldGiveUp gives up after maxRetries for general failures`() {
        assertFalse(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.BIND_FAILED, 4, 5))
        assertTrue(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.BIND_FAILED, 5, 5))
        assertTrue(CameraHealthMonitor.shouldGiveUp(CameraHealthMonitor.FailureClass.BIND_FAILED, 100, 5))
    }

    // ---------- Instance / state machine tests with fake clock ----------

    private class FakeClock {
        var nowMs: Long = 1_000_000L
        fun get(): Long = nowMs
    }

    @Test
    fun `initial snapshot is NEVER_STARTED with empty fields`() {
        val clock = FakeClock()
        val monitor = CameraHealthMonitor(now = { clock.get() })
        val s = monitor.snapshot()
        assertEquals(CameraHealthMonitor.CameraState.NEVER_STARTED, s.state)
        assertEquals(0L, s.lastFrameAtMs)
        assertEquals(0, s.retryCount)
        assertNull(s.lastFailureClass)
        assertNull(s.lastFailureMessage)
    }

    @Test
    fun `onFrameProcessed transitions to OPEN and resets retry state`() {
        val clock = FakeClock()
        val monitor = CameraHealthMonitor(now = { clock.get() })
        monitor.setMaxRetries(3)

        // Simulate a failure first
        monitor.onStartAttempt()
        val firstDecision = monitor.onStartFailure(IllegalStateException("bindToLifecycle"))
        assertFalse(firstDecision.giveUp)

        val mid = monitor.snapshot()
        assertEquals(CameraHealthMonitor.CameraState.FAILED, mid.state)
        assertEquals(1, mid.retryCount)
        assertNotNull(mid.lastFailureClass)

        // Now camera recovers
        clock.nowMs += 10_000
        monitor.onFrameProcessed()
        val after = monitor.snapshot()
        assertEquals(CameraHealthMonitor.CameraState.OPEN, after.state)
        assertEquals(0, after.retryCount)  // reset on recovery
        assertNull(after.lastFailureClass)
        assertNull(after.lastFailureMessage)
        assertTrue(after.lastFrameAtMs > 0)
    }

    @Test
    fun `repeated failures escalate after maxRetries`() {
        val monitor = CameraHealthMonitor()
        monitor.setMaxRetries(3)

        monitor.onStartAttempt()
        var d = monitor.onStartFailure(IllegalStateException("bind 1"))
        assertFalse(d.giveUp)
        d = monitor.onStartFailure(IllegalStateException("bind 2"))
        assertFalse(d.giveUp)
        d = monitor.onStartFailure(IllegalStateException("bind 3"))
        assertTrue(d.giveUp)
        assertEquals(0L, d.retryAfterMs)
    }

    @Test
    fun `permission denied retries forever`() {
        val monitor = CameraHealthMonitor()
        monitor.setMaxRetries(3)

        repeat(20) {
            val d = monitor.onStartFailure(SecurityException("permission denied"), CameraHealthMonitor.FailureClass.PERMISSION_DENIED)
            assertFalse("permission_denied should never give up at attempt $it", d.giveUp)
            assertTrue("retry delay must be positive at attempt $it", d.retryAfterMs > 0)
        }
    }

    @Test
    fun `Adam @ Focus Media scenario - cap devolution after 24h FAILED`() {
        // Use a non-zero clock origin — failedSinceMs=0 is the "never failed"
        // sentinel and would short-circuit the policy.
        val clock = FakeClock()
        val origin = 1_700_000_000_000L  // Nov 2023 ish
        clock.nowMs = origin
        val monitor = CameraHealthMonitor(now = { clock.get() })
        monitor.setMaxRetries(5)

        // Camera fails on first start
        monitor.onStartAttempt()
        monitor.onStartFailure(
            t = RuntimeException("LENS_FACING threw on every camera"),
            hint = CameraHealthMonitor.FailureClass.LENS_FACING_QUERY_FAILED
        )
        // After 1h: still in retry window, no devolve
        clock.nowMs = origin + (1L * 60 * 60 * 1000)
        assertFalse(monitor.shouldDevolveFaceCap())

        // After 23h59min: still no devolve
        clock.nowMs = origin + (24L * 60 * 60 * 1000) - 1
        assertFalse(monitor.shouldDevolveFaceCap())

        // After exactly 24h: devolve
        clock.nowMs = origin + (24L * 60 * 60 * 1000)
        assertTrue(monitor.shouldDevolveFaceCap())

        // After 30 days (Adam's actual case): still devolved
        clock.nowMs = origin + (30L * 24 * 60 * 60 * 1000)
        assertTrue(monitor.shouldDevolveFaceCap())
    }

    @Test
    fun `cap devolution reverts when camera recovers`() {
        val clock = FakeClock()
        val origin = 1_700_000_000_000L
        clock.nowMs = origin
        val monitor = CameraHealthMonitor(now = { clock.get() })

        // Fail and stay failed for >24h
        monitor.onStartAttempt()
        monitor.onStartFailure(IllegalStateException("bind"))
        clock.nowMs = origin + (25L * 60 * 60 * 1000)
        assertTrue(monitor.shouldDevolveFaceCap())

        // Camera recovers
        monitor.onFrameProcessed()
        assertFalse(monitor.shouldDevolveFaceCap())
    }

    @Test
    fun `onTransientFailure flips OPEN to FAILED and starts the failedSince clock`() {
        val clock = FakeClock()
        clock.nowMs = 0L
        val monitor = CameraHealthMonitor(now = { clock.get() })

        monitor.onFrameProcessed()
        assertEquals(CameraHealthMonitor.CameraState.OPEN, monitor.snapshot().state)

        clock.nowMs = 1_000
        monitor.onTransientFailure("watchdog")
        val after = monitor.snapshot()
        assertEquals(CameraHealthMonitor.CameraState.FAILED, after.state)
        assertEquals(CameraHealthMonitor.FailureClass.FRAME_TIMEOUT, after.lastFailureClass)
        assertEquals("watchdog", after.lastFailureMessage)
    }

    @Test
    fun `toHeartbeatJson surfaces all fields`() {
        val monitor = CameraHealthMonitor()
        monitor.onStartAttempt()
        monitor.onStartFailure(RuntimeException("bind threw"))

        val json = monitor.toHeartbeatJson()
        assertTrue(json.has("state"))
        assertTrue(json.has("retry_count"))
        assertTrue(json.has("max_retries"))
        assertTrue(json.has("last_failure_class"))
        assertTrue(json.has("last_failure_message"))
        // Values must reflect the failure we just applied
        assertEquals("failed", json.optString("state"))
        assertEquals(1, json.optInt("retry_count"))
    }

    @Test
    fun `buildEscalationContext carries Build fields and duration`() {
        val clock = FakeClock()
        clock.nowMs = 1_000_000L
        val monitor = CameraHealthMonitor(now = { clock.get() })

        monitor.onStartAttempt()
        // Pass an explicit hint so the failure-class classifier doesn't have
        // to infer from the message — ensures the test is unambiguous.
        monitor.onStartFailure(
            t = IllegalStateException("bindToLifecycle threw"),
            hint = CameraHealthMonitor.FailureClass.BIND_FAILED
        )
        clock.nowMs = 1_010_000L

        val ctx = monitor.buildEscalationContext(
            deviceManufacturer = "RockChip",
            deviceModel = "Generic",
            osVersion = "13",
            osApiLevel = 33,
            cpuAbi = "arm64-v8a",
            totalCameraCount = 1
        )
        assertEquals("RockChip", ctx.deviceManufacturer)
        assertEquals("Generic", ctx.deviceModel)
        assertEquals("13", ctx.osVersion)
        assertEquals(33, ctx.osApiLevel)
        assertEquals("arm64-v8a", ctx.cpuAbi)
        assertEquals(1, ctx.totalCameraCount)
        assertEquals(10_000L, ctx.durationMsSinceFirstAttempt)
        assertEquals(CameraHealthMonitor.FailureClass.BIND_FAILED, ctx.failureClass)
    }
}
