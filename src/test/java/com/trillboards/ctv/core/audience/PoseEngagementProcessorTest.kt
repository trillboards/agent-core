package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.SensingConfig
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for the pure movement-classification logic in [PoseEngagementProcessor].
 *
 * The processor itself requires Android Context + MediaPipe model load, so we
 * test the pure helper [PoseEngagementProcessor.classifyMovement] which is
 * a static, side-effect-free function that takes raw hip displacement +
 * engagement context and returns a [MovementState].
 *
 * Bug context (Samsung S11 ADB log, 2026-05-02):
 *   "Pose: facing=80deg, lean=LEANING_BACK, movement=WALKING_FAST, engagement=0.10"
 * User was sitting at a laptop typing — not walking. Hip-center sway during
 * typing produced enough frame-to-frame displacement that the speed-only
 * classifier flagged WALKING_FAST. Fix: gate WALKING_FAST/WALKING_SLOW on
 * face count + screen engagement + raw hip-magnitude threshold so a user
 * stably looking at the screen with small upper-body motion is classified
 * as STOPPED.
 */
class PoseEngagementProcessorTest {

    private val cfg = SensingConfig.BodyConfig()  // defaults

    // Time interval between frames (33ms ≈ 30fps).
    private val frameDeltaMs = 33L

    // ── Engagement gate: typing-at-desk ──────────────────────────────────────

    @Test
    fun `typing at desk - single face engaged, hip sway above speed threshold - returns STOPPED`() {
        // Hip sway during typing: ~0.06 in normalized coords over 33ms
        // raw distance = 0.06; speed = 0.06 / 0.033 = 1.8 → coerced to 1.0 → WALKING_FAST without gate
        // hipMagnitude = 0.06 < engagedStationaryHipThreshold (0.08) → STOPPED
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 1,
            screenEngaged = true,
            cfg = cfg
        ).first
        assertEquals(MovementState.STOPPED, state)
    }

    @Test
    fun `typing at desk - speed alone says WALKING_FAST baseline regression`() {
        // Verifies we have a real WALKING_FAST trigger: faceCount=0 (no engagement gate
        // applies) and the same displacement → original behavior returns WALKING_FAST.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 0,
            screenEngaged = false,
            cfg = cfg
        ).first
        assertEquals(MovementState.WALKING_FAST, state)
    }

    @Test
    fun `walking by - face transient and screen disengaged - still WALKING_FAST`() {
        // Person walking past camera: large hip displacement (0.20 in normalized coords)
        // No engagement gate (screenEngaged=false), so the original speed classifier wins.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.30f, prevHipY = 0.50f,
            currHipX = 0.50f, currHipY = 0.50f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 0,
            screenEngaged = false,
            cfg = cfg
        ).first
        assertEquals(MovementState.WALKING_FAST, state)
    }

    @Test
    fun `walking towards screen - single engaged face but large hip displacement - still classified as walking`() {
        // True walking-while-engaged: hip displacement 0.20 >> engagedStationaryHipThreshold (0.08).
        // The gate explicitly DOES NOT override large displacements; this preserves real-walking
        // detection even when the user happens to be facing the screen.
        // Speed = 0.20 / 0.033 = 6.0 → coerced to 1.0; classifier returns WALKING_FAST.
        val (state, _) = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.30f, prevHipY = 0.50f,
            currHipX = 0.50f, currHipY = 0.50f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 1,
            screenEngaged = true,
            cfg = cfg
        )
        // Either WALKING_FAST or WALKING_SLOW is acceptable — the key invariant
        // is "not STOPPED". We assert specifically against STOPPED so future
        // threshold tweaks don't trip this test.
        assert(state == MovementState.WALKING_FAST || state == MovementState.WALKING_SLOW) {
            "Expected WALKING_FAST or WALKING_SLOW, got $state"
        }
        assert(state != MovementState.STOPPED) {
            "Engagement gate must NOT override truly-walking hip displacement"
        }
    }

    // ── Multi-face engagement (P2.1 fix, audit 2026-05-03) ──────────────────

    @Test
    fun `two engaged faces with stationary hip - gate fires (STOPPED)`() {
        // Two coworkers / a couple browsing a kiosk — both stably looking at the
        // screen. screenEngaged=true (caller upstream computes max-face engagement,
        // so any engaged face turns this on). Hip displacement is below the
        // engaged-stationary threshold → upper-body micro-motion, not walking.
        // Pre-2026-05-03 the gate was hardcoded to `faceCount==1`, so this case
        // fell through to WALKING_FAST. New gate: `faceCount >= 1 AND
        // screenEngaged` → STOPPED override fires.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 2,
            screenEngaged = true,
            cfg = cfg
        ).first
        assertEquals(MovementState.STOPPED, state)
    }

    @Test
    fun `one engaged plus one disengaged - gate fires (highest engagement dominates)`() {
        // Two faces in frame: one engaged, one not. Caller upstream computes
        // screenEngaged via `faces.any { isLooking }`, which is equivalent to
        // taking max-face engagement — so screenEngaged=true even with a
        // disengaged second face. Two-face mixed-engagement scenarios are
        // exactly what the original `faceCount==1` hardcode missed.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 2,
            screenEngaged = true,
            cfg = cfg
        ).first
        assertEquals(MovementState.STOPPED, state)
    }

    @Test
    fun `screen disengaged single face - engagement gate does NOT trigger`() {
        // Single face but person is looking away (e.g. checking phone) → not engaged.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 1,
            screenEngaged = false,
            cfg = cfg
        ).first
        assertEquals(MovementState.WALKING_FAST, state)
    }

    @Test
    fun `screen disengaged multi-face - engagement gate does NOT trigger`() {
        // Two people in frame, neither engaged → fall back to speed-only.
        // Preserves the WALKING_FAST classification for true crowd-walking-by.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 2,
            screenEngaged = false,
            cfg = cfg
        ).first
        assertEquals(MovementState.WALKING_FAST, state)
    }

    @Test
    fun `zero faces - engagement gate does NOT trigger`() {
        // No face detected at all (e.g. person turned away or out of frame) →
        // we have no engagement signal, so fall back to speed-only. Confirms
        // the gate's `faceCount >= 1` guard rejects zero-face frames even when
        // screenEngaged is somehow set true (defensive — caller should never
        // do that, but the gate handles it gracefully).
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 0,
            screenEngaged = true,
            cfg = cfg
        ).first
        assertEquals(MovementState.WALKING_FAST, state)
    }

    // ── No-prior-frame baseline ──────────────────────────────────────────────

    @Test
    fun `zero time delta returns UNKNOWN regardless of context`() {
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.50f, currHipY = 0.40f,
            timeDeltaSec = 0f,
            faceCount = 1,
            screenEngaged = true,
            cfg = cfg
        ).first
        assertEquals(MovementState.UNKNOWN, state)
    }

    // ── Stationary baseline (no engagement context) ─────────────────────────

    @Test
    fun `truly stationary hips - small displacement returns STOPPED without engagement gate`() {
        // Tiny hip displacement (0.001 normalized) → speed below movementSpeedThreshold.
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.500f, prevHipY = 0.400f,
            currHipX = 0.501f, currHipY = 0.400f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 0,
            screenEngaged = false,
            cfg = cfg
        ).first
        assertEquals(MovementState.STOPPED, state)
    }

    // ── Threshold knob is server-tunable ────────────────────────────────────

    @Test
    fun `engagedStationaryHipThreshold is tunable via SensingConfig BodyConfig`() {
        // Default 0.08 — user with hip displacement 0.06 (typing) gets gated to STOPPED.
        // If we tighten to 0.04, the same displacement should fall through to WALKING_FAST.
        val tighter = cfg.copy(engagedStationaryHipThreshold = 0.04f)
        val state = PoseEngagementProcessor.classifyMovement(
            prevHipX = 0.50f, prevHipY = 0.40f,
            currHipX = 0.56f, currHipY = 0.40f,
            timeDeltaSec = frameDeltaMs / 1000f,
            faceCount = 1,
            screenEngaged = true,
            cfg = tighter
        ).first
        assertEquals(MovementState.WALKING_FAST, state)
    }
}
