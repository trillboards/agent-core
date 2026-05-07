package com.trillboards.ctv.core.audience

import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Phase 4 PR 2 — Edge clip-trigger evaluator.
 *
 * Pure-function evaluator: given the latest audience-signal snapshot and the
 * `triggers[]` array shipped down inside the active profile's `metrics_schema`,
 * returns the list of [TriggerRule]s that fired in this evaluation pass.
 *
 * ## What this does (PR 2 scope)
 * - Parse `metrics_schema[]` entries with `type == 'trigger'` into typed rules.
 * - Evaluate each rule against the current snapshot using the allow-listed
 *   comparison operators ([gt, gte, lt, lte, eq, neq]).
 * - Debounce per `trigger_id` — track the last fire time and suppress fires
 *   within `debounce_seconds`.
 * - Return a list of fired rules. The caller (PR 3) is responsible for
 *   actually capturing clips; this PR only LOGs telemetry.
 *
 * ## Why pure function (no IO)
 * The evaluator is invoked from `AudienceSensingService.aggregateAndEmit` on
 * the same coroutine that emits `audienceSignals`. Every aggregation cycle
 * (default 10s) hits this code, so adding even a single PG/HTTP call would
 * blow the cycle budget. The debounce map is in-memory; it resets when the
 * service restarts (which is fine — the server-side downstream PR 3 will
 * dedupe by `(screen_id, trigger_id, fired_at)` over a wider window).
 *
 * ## Thread safety
 * The debounce map is guarded by [debounceLock]. Calls into [evaluate] are
 * sequential within `aggregateAndEmit`, but any future caller must respect
 * that contract — multiple concurrent callers would race on the debounce map
 * and could double-fire.
 *
 * @see AudienceSensingService.aggregateAndEmit For the call site
 */
class ClipTriggerEvaluator(
    private val clock: () -> Long = { SystemClock.elapsedRealtime() }
) {

    companion object {
        private const val TAG = "ClipTriggerEvaluator"

        /** Allow-listed comparison operators. Must match server clipTriggerSchema.js. */
        private val VALID_OPERATORS = setOf("gt", "gte", "lt", "lte", "eq", "neq")

        /** Field-name regex — must match server clipTriggerSchema.js. */
        private val FIELD_NAME_RE = Regex("^[a-z_][a-z0-9_]{0,63}$")

        /** Trigger-id regex — must match server clipTriggerSchema.js. */
        private val TRIGGER_ID_RE = Regex("^[a-z_][a-z0-9_]{0,63}$")

        /** Numeric tolerance for float equality (operator `eq`/`neq` on doubles). */
        private const val DOUBLE_EPSILON = 1e-9
    }

    /**
     * One declarative trigger rule parsed from the profile's `metrics_schema`.
     *
     * Mirrors the shape of `services/fein/clipTriggerSchema.js`'s
     * `TriggerEntrySchema`. The Zod validator on the server is the canonical
     * input gate; this Kotlin parser is defensive (fails closed) but should
     * never see invalid input on a well-behaved profile push.
     */
    data class TriggerRule(
        val triggerId: String,
        val condition: Condition,
        val debounceSeconds: Long,
        val captureDurationSeconds: Int,
        val description: String? = null,
    )

    data class Condition(
        val field: String,
        val op: String,
        val value: Any, // Number | Boolean | String — never null per schema
    )

    /**
     * Result of a single evaluate() pass — the rules that fired AND the
     * matching snapshot value for telemetry.
     */
    data class FiredTrigger(
        val rule: TriggerRule,
        val observedValue: Any?,
        val firedAtMs: Long,
    )

    private val debounceLock = Any()
    private val lastFiredAtByTriggerId: MutableMap<String, Long> = mutableMapOf()

    private var totalFiredCount: Long = 0L
    private var totalSuppressedByDebounceCount: Long = 0L

    /**
     * Evaluate the active rules against the current signal snapshot.
     *
     * @param rules The active trigger rules (from the profile's metrics_schema)
     * @param signals A flat map of `field_name → observed value` derived from
     *                the current aggregation cycle's outputs (face_count,
     *                noise_level, dominant_emotion, etc.)
     * @return The rules that fired in this pass, with their observed values.
     */
    fun evaluate(rules: List<TriggerRule>, signals: Map<String, Any?>): List<FiredTrigger> {
        if (rules.isEmpty()) return emptyList()
        val now = clock()
        val fired = mutableListOf<FiredTrigger>()
        synchronized(debounceLock) {
            for (rule in rules) {
                val observed = signals[rule.condition.field]
                if (observed == null) continue
                val matches = compareScalar(observed, rule.condition.op, rule.condition.value)
                if (!matches) continue

                val lastFiredMs = lastFiredAtByTriggerId[rule.triggerId]
                if (lastFiredMs != null) {
                    val elapsedMs = now - lastFiredMs
                    val debounceMs = rule.debounceSeconds * 1000L
                    if (elapsedMs < debounceMs) {
                        totalSuppressedByDebounceCount++
                        Log.d(TAG, "[debounce] trigger=${rule.triggerId} suppressed " +
                            "(${elapsedMs}ms < ${debounceMs}ms)")
                        continue
                    }
                }

                lastFiredAtByTriggerId[rule.triggerId] = now
                totalFiredCount++
                fired += FiredTrigger(rule = rule, observedValue = observed, firedAtMs = now)
                Log.i(TAG, "[fire] trigger=${rule.triggerId} field=${rule.condition.field} " +
                    "${rule.condition.op} ${rule.condition.value} (observed=$observed) " +
                    "capture_duration=${rule.captureDurationSeconds}s")
            }
        }
        return fired
    }

    /**
     * Compare two scalars under the allow-listed operator set. Returns false
     * for any invalid op or unsupported value combination — never throws.
     *
     * Numeric comparisons coerce the LHS via [toDoubleOrNull]; boolean
     * comparisons coerce the LHS via [toBooleanOrNull]; string comparisons
     * use exact equality on stringified forms (op `gt`/`lt` over strings is
     * NOT supported and returns false).
     */
    private fun compareScalar(observed: Any?, op: String, expected: Any): Boolean {
        if (op !in VALID_OPERATORS) return false
        if (observed == null) return false

        // Numeric path
        val obsNum = observed.toDoubleOrNullSafe()
        val expNum = expected.toDoubleOrNullSafe()
        if (obsNum != null && expNum != null) {
            return when (op) {
                "gt" -> obsNum > expNum
                "gte" -> obsNum >= expNum
                "lt" -> obsNum < expNum
                "lte" -> obsNum <= expNum
                "eq" -> Math.abs(obsNum - expNum) < DOUBLE_EPSILON
                "neq" -> Math.abs(obsNum - expNum) >= DOUBLE_EPSILON
                else -> false
            }
        }

        // Boolean path (eq/neq only; gt/lt on booleans is meaningless)
        if (expected is Boolean) {
            val obsBool = when (observed) {
                is Boolean -> observed
                is String -> when (observed.lowercase()) {
                    "true" -> true
                    "false" -> false
                    else -> return false
                }
                else -> return false
            }
            return when (op) {
                "eq" -> obsBool == expected
                "neq" -> obsBool != expected
                else -> false
            }
        }

        // String path (eq/neq only)
        val obsStr = observed.toString()
        val expStr = expected.toString()
        return when (op) {
            "eq" -> obsStr == expStr
            "neq" -> obsStr != expStr
            else -> false
        }
    }

    /** Reset all in-memory state — used when the active profile changes. */
    fun reset() {
        synchronized(debounceLock) {
            lastFiredAtByTriggerId.clear()
            totalFiredCount = 0L
            totalSuppressedByDebounceCount = 0L
        }
        Log.i(TAG, "[reset] cleared debounce map")
    }

    /** Telemetry counters for fleet observability. */
    data class Telemetry(val firedCount: Long, val suppressedCount: Long, val activeIdCount: Int)

    fun getTelemetry(): Telemetry = synchronized(debounceLock) {
        Telemetry(
            firedCount = totalFiredCount,
            suppressedCount = totalSuppressedByDebounceCount,
            activeIdCount = lastFiredAtByTriggerId.size,
        )
    }
}

// ─── Parsers ────────────────────────────────────────────────────────────────

/**
 * Parse the `metrics_schema` array (already JSON-deserialized) into a list of
 * [ClipTriggerEvaluator.TriggerRule]s. Skips entries whose `type` is not
 * `'trigger'` or which fail the snake_case / range / op-allowlist checks.
 *
 * The server-side Zod validator (`services/fein/clipTriggerSchema.js`) is the
 * canonical input gate; this parser is defensive so that a misconfigured
 * profile push doesn't crash the edge.
 */
fun parseClipTriggersFromMetricsSchema(metricsSchema: JSONArray?): List<ClipTriggerEvaluator.TriggerRule> {
    if (metricsSchema == null || metricsSchema.length() == 0) return emptyList()
    val out = mutableListOf<ClipTriggerEvaluator.TriggerRule>()
    for (i in 0 until metricsSchema.length()) {
        val entry = metricsSchema.optJSONObject(i) ?: continue
        if (entry.optString("type") != "trigger") continue
        if (entry.has("deprecated_at") && !entry.isNull("deprecated_at")) continue

        val triggerId = entry.optString("trigger_id").takeIf { it.isNotBlank() } ?: continue
        if (!parseTriggerIdSafe(triggerId)) continue

        val condJson = entry.optJSONObject("condition") ?: continue
        val field = condJson.optString("field").takeIf { it.isNotBlank() } ?: continue
        if (!parseFieldNameSafe(field)) continue

        val op = condJson.optString("op").takeIf { it.isNotBlank() } ?: continue
        if (!parseOperatorSafe(op)) continue

        val value = parseConditionValueSafe(condJson) ?: continue

        val debounceSeconds = entry.optLong("debounce_seconds", -1L)
        if (debounceSeconds < 1L || debounceSeconds > 86400L) continue

        val captureDurationSeconds = entry.optInt("capture_duration_seconds", -1)
        if (captureDurationSeconds < 5 || captureDurationSeconds > 60) continue

        val description = entry.optString("description").takeIf { it.isNotBlank() }

        out += ClipTriggerEvaluator.TriggerRule(
            triggerId = triggerId,
            condition = ClipTriggerEvaluator.Condition(
                field = field,
                op = op,
                value = value,
            ),
            debounceSeconds = debounceSeconds,
            captureDurationSeconds = captureDurationSeconds,
            description = description,
        )
    }
    return out
}

/** Safe parse for trigger_id snake_case. */
private fun parseTriggerIdSafe(triggerId: String): Boolean =
    Regex("^[a-z_][a-z0-9_]{0,63}$").matches(triggerId)

/** Safe parse for field name snake_case. */
private fun parseFieldNameSafe(field: String): Boolean =
    Regex("^[a-z_][a-z0-9_]{0,63}$").matches(field)

/** Safe parse for operator allow-list. */
private fun parseOperatorSafe(op: String): Boolean =
    op in setOf("gt", "gte", "lt", "lte", "eq", "neq")

/**
 * Pull `condition.value` out of JSON; reject objects/arrays and dangerous
 * substrings in strings (mirrors `clipTriggerSchema.js#FORBIDDEN_VALUE_SUBSTRINGS`).
 *
 * Returns null if the value is missing, non-scalar, or fails the substring guard.
 */
private fun parseConditionValueSafe(condition: JSONObject): Any? {
    if (!condition.has("value") || condition.isNull("value")) return null
    val raw = condition.get("value")
    if (raw is JSONObject || raw is JSONArray) return null

    return when (raw) {
        is Number -> {
            val d = raw.toDouble()
            if (d.isFinite()) raw else null
        }
        is Boolean -> raw
        is String -> {
            if (raw.length > 200) return null
            val forbidden = listOf(
                "require(", "eval(", "import(", "process.", "Function(",
                "__proto__", "constructor", "prototype", "<script", "\${", "`"
            )
            if (forbidden.any { raw.contains(it) }) null else raw
        }
        else -> null
    }
}

/**
 * Coerce arbitrary values to Double for numeric comparison. Handles primitive
 * Number subclasses + numeric strings ("5", "5.0"). Returns null for booleans
 * (so the boolean code path takes over) and non-numeric strings.
 */
internal fun Any?.toDoubleOrNullSafe(): Double? = when (this) {
    null -> null
    is Boolean -> null
    is Number -> {
        val d = this.toDouble()
        if (d.isFinite()) d else null
    }
    is String -> this.toDoubleOrNull()?.takeIf { it.isFinite() }
    else -> null
}

/** Coerce arbitrary values to Boolean. Used only for boolean-equality path. */
internal fun Any?.toBooleanOrNull(): Boolean? = when (this) {
    is Boolean -> this
    is String -> when (this.lowercase()) {
        "true" -> true
        "false" -> false
        else -> null
    }
    else -> null
}
