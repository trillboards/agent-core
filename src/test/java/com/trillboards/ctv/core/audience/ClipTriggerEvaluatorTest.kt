package com.trillboards.ctv.core.audience

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Phase 4 PR 2 — ClipTriggerEvaluator unit tests.
 *
 * Pure-function evaluator. Uses a controllable clock for deterministic
 * debounce assertions; never touches Android Looper or SystemClock.
 *
 * Coverage:
 *   - face_count threshold trigger: gt / gte / lt / lte / eq / neq
 *   - boolean equality (safety_concern)
 *   - string equality (dominant_emotion)
 *   - debounce suppression vs allowed re-fire
 *   - independent debounce per trigger_id
 *   - missing signal → no fire
 *   - reset clears debounce map
 *   - parseClipTriggersFromMetricsSchema: skip non-trigger entries, skip
 *     deprecated, skip out-of-range, skip injection-substring values
 */
class ClipTriggerEvaluatorTest {

    private var nowMs = 100_000L
    private val clock: () -> Long = { nowMs }

    private lateinit var evaluator: ClipTriggerEvaluator

    @Before
    fun setUp() {
        nowMs = 100_000L
        evaluator = ClipTriggerEvaluator(clock = clock)
    }

    private fun rule(
        id: String = "face_count_spike",
        field: String = "face_count",
        op: String = "gt",
        value: Any = 5,
        debounce: Long = 30L,
        capture: Int = 30,
    ) = ClipTriggerEvaluator.TriggerRule(
        triggerId = id,
        condition = ClipTriggerEvaluator.Condition(field = field, op = op, value = value),
        debounceSeconds = debounce,
        captureDurationSeconds = capture,
    )

    // ── Face-count threshold (gt, gte, lt, lte) ─────────────────────────────

    @Test
    fun `face_count gt 5 fires when observed is 6`() {
        val r = rule(op = "gt", value = 5)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 6))
        assertEquals(1, fired.size)
        assertEquals("face_count_spike", fired[0].rule.triggerId)
        assertEquals(6, fired[0].observedValue)
    }

    @Test
    fun `face_count gt 5 does NOT fire when observed is 5`() {
        val r = rule(op = "gt", value = 5)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 5))
        assertTrue(fired.isEmpty())
    }

    @Test
    fun `face_count gte 5 fires when observed is 5`() {
        val r = rule(op = "gte", value = 5)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 5))
        assertEquals(1, fired.size)
    }

    @Test
    fun `face_count lt 3 fires when observed is 2`() {
        val r = rule(op = "lt", value = 3)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 2))
        assertEquals(1, fired.size)
    }

    @Test
    fun `face_count lte 3 fires when observed equals 3`() {
        val r = rule(op = "lte", value = 3)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 3))
        assertEquals(1, fired.size)
    }

    @Test
    fun `face_count eq 0 fires when observed is 0 (empty venue)`() {
        val r = rule(op = "eq", value = 0)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 0))
        assertEquals(1, fired.size)
    }

    @Test
    fun `face_count neq 0 fires when observed is 5`() {
        val r = rule(op = "neq", value = 0)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 5))
        assertEquals(1, fired.size)
    }

    @Test
    fun `numeric eq tolerates float jitter via epsilon`() {
        val r = rule(op = "eq", value = 0.5)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 0.5000000001))
        assertEquals(1, fired.size)
    }

    @Test
    fun `string-encoded numbers compare numerically`() {
        // The signal map can carry "5" instead of 5 (e.g., from JSON.optString).
        val r = rule(op = "gt", value = 5)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to "10"))
        assertEquals(1, fired.size)
    }

    @Test
    fun `non-numeric string with numeric op does not fire`() {
        val r = rule(op = "gt", value = 5)
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to "abc"))
        assertTrue(fired.isEmpty())
    }

    // ── Boolean equality ────────────────────────────────────────────────────

    @Test
    fun `safety_concern eq true fires when observed is true`() {
        val r = rule(field = "safety_concern", op = "eq", value = true)
        val fired = evaluator.evaluate(listOf(r), mapOf("safety_concern" to true))
        assertEquals(1, fired.size)
    }

    @Test
    fun `safety_concern eq true does NOT fire when observed is false`() {
        val r = rule(field = "safety_concern", op = "eq", value = true)
        val fired = evaluator.evaluate(listOf(r), mapOf("safety_concern" to false))
        assertTrue(fired.isEmpty())
    }

    @Test
    fun `safety_concern neq false fires when observed is true`() {
        val r = rule(field = "safety_concern", op = "neq", value = false)
        val fired = evaluator.evaluate(listOf(r), mapOf("safety_concern" to true))
        assertEquals(1, fired.size)
    }

    @Test
    fun `string-encoded boolean compares correctly`() {
        val r = rule(field = "safety_concern", op = "eq", value = true)
        val fired = evaluator.evaluate(listOf(r), mapOf("safety_concern" to "true"))
        assertEquals(1, fired.size)
    }

    @Test
    fun `boolean op gt does not fire (semantically meaningless)`() {
        val r = rule(field = "safety_concern", op = "gt", value = true)
        val fired = evaluator.evaluate(listOf(r), mapOf("safety_concern" to true))
        assertTrue(fired.isEmpty())
    }

    // ── String equality ─────────────────────────────────────────────────────

    @Test
    fun `dominant_emotion eq HAPPY fires`() {
        val r = rule(field = "dominant_emotion", op = "eq", value = "HAPPY")
        val fired = evaluator.evaluate(listOf(r), mapOf("dominant_emotion" to "HAPPY"))
        assertEquals(1, fired.size)
    }

    @Test
    fun `dominant_emotion eq HAPPY does NOT fire on NEUTRAL`() {
        val r = rule(field = "dominant_emotion", op = "eq", value = "HAPPY")
        val fired = evaluator.evaluate(listOf(r), mapOf("dominant_emotion" to "NEUTRAL"))
        assertTrue(fired.isEmpty())
    }

    @Test
    fun `dominant_emotion neq UNKNOWN fires on HAPPY`() {
        val r = rule(field = "dominant_emotion", op = "neq", value = "UNKNOWN")
        val fired = evaluator.evaluate(listOf(r), mapOf("dominant_emotion" to "HAPPY"))
        assertEquals(1, fired.size)
    }

    @Test
    fun `string op gt does not fire (semantically meaningless on strings)`() {
        val r = rule(field = "dominant_emotion", op = "gt", value = "A")
        val fired = evaluator.evaluate(listOf(r), mapOf("dominant_emotion" to "B"))
        assertTrue(fired.isEmpty())
    }

    // ── Missing signal ──────────────────────────────────────────────────────

    @Test
    fun `missing signal field does NOT fire`() {
        val r = rule()
        val fired = evaluator.evaluate(listOf(r), mapOf("noise_level" to 0.5))
        assertTrue(fired.isEmpty())
    }

    @Test
    fun `null signal value does NOT fire`() {
        val r = rule()
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to null))
        assertTrue(fired.isEmpty())
    }

    @Test
    fun `empty rule list returns empty fired list`() {
        val fired = evaluator.evaluate(emptyList(), mapOf("face_count" to 100))
        assertTrue(fired.isEmpty())
    }

    @Test
    fun `unknown operator does NOT fire`() {
        val r = rule(op = "regex", value = ".*")
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 5))
        assertTrue(fired.isEmpty())
    }

    // ── Debounce ────────────────────────────────────────────────────────────

    @Test
    fun `debounce suppresses re-fire within window`() {
        val r = rule(op = "gt", value = 5, debounce = 30L)

        val fired1 = evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        assertEquals(1, fired1.size)

        // Advance 10s — still inside 30s debounce window
        nowMs += 10_000L
        val fired2 = evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        assertTrue("Should be suppressed by debounce", fired2.isEmpty())
    }

    @Test
    fun `debounce allows re-fire after window`() {
        val r = rule(op = "gt", value = 5, debounce = 30L)

        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))

        nowMs += 31_000L
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        assertEquals(1, fired.size)
    }

    @Test
    fun `debounce exactly at boundary allows re-fire`() {
        val r = rule(op = "gt", value = 5, debounce = 30L)

        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))

        // elapsed >= debounceMs allows re-fire
        nowMs += 30_000L
        val fired = evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        assertEquals(1, fired.size)
    }

    @Test
    fun `debounce is independent per trigger_id`() {
        val r1 = rule(id = "alpha", op = "gt", value = 5, debounce = 30L)
        val r2 = rule(id = "beta", field = "noise_level", op = "gt", value = 0.5, debounce = 30L)

        val fired1 = evaluator.evaluate(
            listOf(r1, r2),
            mapOf("face_count" to 10, "noise_level" to 0.8)
        )
        assertEquals(2, fired1.size)

        // alpha is debounced, beta is also debounced — both should suppress
        nowMs += 5_000L
        val fired2 = evaluator.evaluate(
            listOf(r1, r2),
            mapOf("face_count" to 10, "noise_level" to 0.8)
        )
        assertTrue(fired2.isEmpty())

        // Now advance past alpha's debounce (set to a SHORT debounce on alpha)
        // Re-create r1 with shorter debounce to verify per-id isolation
        val r1Short = rule(id = "alpha", op = "gt", value = 5, debounce = 6L)
        nowMs += 7_000L  // total since first fire = 12s
        val fired3 = evaluator.evaluate(
            listOf(r1Short, r2),
            mapOf("face_count" to 10, "noise_level" to 0.8)
        )
        // alpha should fire (6s debounce + advance to 12s = past debounce)
        // beta should NOT fire (still inside 30s)
        assertEquals(1, fired3.size)
        assertEquals("alpha", fired3[0].rule.triggerId)
    }

    @Test
    fun `multiple rules can fire in same evaluation pass`() {
        val r1 = rule(id = "alpha", op = "gt", value = 5)
        val r2 = rule(id = "beta", field = "noise_level", op = "gte", value = 0.5)
        val fired = evaluator.evaluate(
            listOf(r1, r2),
            mapOf("face_count" to 10, "noise_level" to 0.7)
        )
        assertEquals(2, fired.size)
        val ids = fired.map { it.rule.triggerId }.toSet()
        assertEquals(setOf("alpha", "beta"), ids)
    }

    // ── Reset ───────────────────────────────────────────────────────────────

    @Test
    fun `reset clears debounce map`() {
        val r = rule(op = "gt", value = 5, debounce = 30L)
        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))

        nowMs += 5_000L
        val firedBefore = evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        assertTrue(firedBefore.isEmpty())

        evaluator.reset()

        // After reset — within original debounce window, but map was cleared
        nowMs += 1_000L
        val firedAfter = evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        assertEquals(1, firedAfter.size)
    }

    // ── Telemetry ───────────────────────────────────────────────────────────

    @Test
    fun `telemetry tracks fired and suppressed counts`() {
        val r = rule(op = "gt", value = 5, debounce = 30L)

        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))

        nowMs += 5_000L
        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))

        nowMs += 5_000L
        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))

        val t = evaluator.getTelemetry()
        assertEquals(1L, t.firedCount)
        assertEquals(2L, t.suppressedCount)
        assertEquals(1, t.activeIdCount)
    }

    @Test
    fun `telemetry reset to zero on reset()`() {
        val r = rule(op = "gt", value = 5)
        evaluator.evaluate(listOf(r), mapOf("face_count" to 10))
        evaluator.reset()
        val t = evaluator.getTelemetry()
        assertEquals(0L, t.firedCount)
        assertEquals(0L, t.suppressedCount)
        assertEquals(0, t.activeIdCount)
    }

    // ── parseClipTriggersFromMetricsSchema ──────────────────────────────────

    @Test
    fun `parser extracts well-formed trigger entries`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "face_count_spike")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "gt")
                put("value", 5)
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })

        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertEquals(1, rules.size)
        assertEquals("face_count_spike", rules[0].triggerId)
        assertEquals("face_count", rules[0].condition.field)
        assertEquals("gt", rules[0].condition.op)
        assertEquals(30L, rules[0].debounceSeconds)
        assertEquals(30, rules[0].captureDurationSeconds)
    }

    @Test
    fun `parser skips non-trigger entries`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "counter")
            put("field", "face_count")
        })
        arr.put(JSONObject().apply {
            put("provenance", "operator_bonus")
            put("field", "wears_uniform")
            put("type", "boolean")
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser skips deprecated trigger entries`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "old_trigger")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "gt")
                put("value", 5)
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
            put("deprecated_at", "2026-01-01T00:00:00Z")
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser skips entries with bad trigger_id`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "BAD-NAME") // not snake_case
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "gt")
                put("value", 5)
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser skips entries with bad operator`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "regex")
                put("value", ".*")
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser skips entries with out-of-range debounce`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "gt")
                put("value", 5)
            })
            put("debounce_seconds", 0) // below floor
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser skips entries with out-of-range capture_duration`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "gt")
                put("value", 5)
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 999) // above ceiling
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser rejects condition value with require( substring`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "foo")
                put("op", "eq")
                put("value", "require(\"fs\")")
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser rejects condition value containing template marker`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "foo")
                put("op", "eq")
                put("value", "\${process.env.SECRET}")
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser rejects object-valued condition value`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "foo")
                put("op", "eq")
                put("value", JSONObject().apply { put("nested", 1) })
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser rejects array-valued condition value`() {
        val arr = JSONArray()
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "valid_id")
            put("condition", JSONObject().apply {
                put("field", "foo")
                put("op", "eq")
                put("value", JSONArray().apply { put(1); put(2) })
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertTrue(rules.isEmpty())
    }

    @Test
    fun `parser handles null and empty input`() {
        assertTrue(parseClipTriggersFromMetricsSchema(null).isEmpty())
        assertTrue(parseClipTriggersFromMetricsSchema(JSONArray()).isEmpty())
    }

    @Test
    fun `parser handles mixed-shape array (triggers + bonus + counters)`() {
        val arr = JSONArray()
        // counter
        arr.put(JSONObject().apply {
            put("type", "counter")
            put("field", "face_count")
        })
        // bonus
        arr.put(JSONObject().apply {
            put("type", "boolean")
            put("field", "wears_uniform")
            put("provenance", "operator_bonus")
        })
        // VALID trigger
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "spike")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "gt")
                put("value", 5)
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })
        // INVALID trigger (bad op)
        arr.put(JSONObject().apply {
            put("type", "trigger")
            put("trigger_id", "bad")
            put("condition", JSONObject().apply {
                put("field", "face_count")
                put("op", "regex")
                put("value", ".*")
            })
            put("debounce_seconds", 30)
            put("capture_duration_seconds", 30)
        })

        val rules = parseClipTriggersFromMetricsSchema(arr)
        assertEquals(1, rules.size)
        assertEquals("spike", rules[0].triggerId)
    }
}
