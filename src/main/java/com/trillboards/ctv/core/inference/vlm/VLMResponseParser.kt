package com.trillboards.ctv.core.inference.vlm

import android.util.Log
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.math.BigDecimal
import java.math.BigInteger

/**
 * Robust JSON parser for VLM text output.
 *
 * On-device VLMs produce outputs that range from clean JSON to partially
 * malformed text with embedded JSON fragments. This parser handles the
 * full spectrum:
 *
 * 1. **Clean JSON**: `{"face_count": 3, "dominant_emotion": "happy"}`
 * 2. **Markdown-fenced JSON**: `` ```json\n{...}\n``` ``
 * 3. **JSON with preamble/postamble**: `Here is the result: {...} Hope this helps!`
 * 4. **Partial/truncated JSON**: `{"face_count": 3, "dominant_em` (max_tokens hit)
 * 5. **Regex fallback**: extract `"key": value` pairs individually when JSON.parse fails
 *
 * ## Type Coercion
 * The parser normalizes types to match downstream expectations:
 * - String numbers (`"3"`) are coerced to Int/Long/Double as appropriate
 * - String booleans (`"true"`, `"false"`) are coerced to Boolean
 * - Nested JSONObject/JSONArray are converted to Map/List recursively
 * - JSONObject.NULL is converted to the string "null" (not Kotlin null)
 *
 * ## Thread Safety
 * All methods are stateless and thread-safe.
 */
object VLMResponseParser {

    private const val TAG = "VLMResponseParser"

    /**
     * Parse VLM output text into a flat key-value map.
     *
     * Tries strategies in order of reliability:
     * 1. Direct JSON parse of the full text
     * 2. Extract JSON from markdown code fences
     * 3. Find the first `{...}` block in the text
     * 4. Regex fallback to extract individual `"key": value` pairs
     *
     * @param text Raw VLM output text.
     * @return Parsed key-value fields, or empty map if all strategies fail.
     */
    fun parse(text: String): ParseResult {
        if (text.isBlank()) {
            return ParseResult(fields = emptyMap(), strategy = ParseStrategy.NONE, success = false)
        }

        val trimmed = text.trim()

        // Strategy 1: direct JSON parse
        tryParseJson(trimmed)?.let { fields ->
            return ParseResult(fields = fields, strategy = ParseStrategy.DIRECT_JSON, success = true)
        }

        // Strategy 2: extract from markdown code fences (```json ... ``` or ``` ... ```)
        extractFromCodeFence(trimmed)?.let { jsonStr ->
            tryParseJson(jsonStr)?.let { fields ->
                return ParseResult(fields = fields, strategy = ParseStrategy.CODE_FENCE, success = true)
            }
        }

        // Strategy 3: find the first { ... } block
        extractFirstJsonObject(trimmed)?.let { jsonStr ->
            tryParseJson(jsonStr)?.let { fields ->
                return ParseResult(fields = fields, strategy = ParseStrategy.BRACE_EXTRACTION, success = true)
            }
        }

        // Strategy 4: regex fallback for "key": value pairs
        val regexFields = extractKeyValuePairs(trimmed)
        if (regexFields.isNotEmpty()) {
            return ParseResult(fields = regexFields, strategy = ParseStrategy.REGEX_FALLBACK, success = true)
        }

        Log.w(TAG, "All parse strategies failed for text (${trimmed.length} chars): " +
            "${trimmed.take(100)}...")
        return ParseResult(fields = emptyMap(), strategy = ParseStrategy.NONE, success = false)
    }

    /**
     * Try to parse a string as a JSON object and flatten it into a Map.
     * Returns null if the string is not valid JSON.
     */
    private fun tryParseJson(text: String): Map<String, Any>? {
        return try {
            val json = JSONObject(text)
            flattenJsonObject(json)
        } catch (e: JSONException) {
            null
        }
    }

    /**
     * Extract JSON content from markdown code fences.
     * Handles:
     * - ```json\n{...}\n```
     * - ```\n{...}\n```
     * - ```json {... }```  (no newlines)
     */
    private fun extractFromCodeFence(text: String): String? {
        // Pattern: ```json ... ``` or ``` ... ```
        val fencePattern = Regex(
            """```(?:json)?\s*\n?\s*(\{[\s\S]*?\})\s*\n?\s*```""",
            RegexOption.DOT_MATCHES_ALL
        )
        val match = fencePattern.find(text)
        return match?.groupValues?.get(1)?.trim()
    }

    /**
     * Find the first balanced { ... } block in the text.
     * Handles nested objects by tracking brace depth.
     */
    private fun extractFirstJsonObject(text: String): String? {
        val startIdx = text.indexOf('{')
        if (startIdx == -1) return null

        var depth = 0
        var inString = false
        var escape = false

        for (i in startIdx until text.length) {
            val c = text[i]

            if (escape) {
                escape = false
                continue
            }

            if (c == '\\' && inString) {
                escape = true
                continue
            }

            if (c == '"') {
                inString = !inString
                continue
            }

            if (!inString) {
                when (c) {
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) {
                            return text.substring(startIdx, i + 1)
                        }
                    }
                }
            }
        }

        // If we reached the end with unclosed braces, try to close the JSON
        // This handles truncated output from max_tokens
        if (depth > 0 && startIdx < text.length - 1) {
            val partial = text.substring(startIdx)
            return tryRepairTruncatedJson(partial)
        }

        return null
    }

    /**
     * Attempt to repair truncated JSON by closing unclosed braces/brackets
     * and removing trailing incomplete key-value pairs.
     */
    private fun tryRepairTruncatedJson(partial: String): String? {
        // Remove trailing incomplete value (after last comma or colon)
        val lastCompleteEntry = findLastCompleteEntry(partial)
        if (lastCompleteEntry <= 0) return null

        val trimmed = partial.substring(0, lastCompleteEntry + 1)

        // Count unclosed braces and brackets
        var braceCount = 0
        var bracketCount = 0
        var inString = false
        var escape = false

        for (c in trimmed) {
            if (escape) { escape = false; continue }
            if (c == '\\' && inString) { escape = true; continue }
            if (c == '"') { inString = !inString; continue }
            if (!inString) {
                when (c) {
                    '{' -> braceCount++
                    '}' -> braceCount--
                    '[' -> bracketCount++
                    ']' -> bracketCount--
                }
            }
        }

        // Close unclosed structures
        val suffix = "]".repeat(maxOf(0, bracketCount)) + "}".repeat(maxOf(0, braceCount))
        val repaired = trimmed + suffix

        // Validate the repair produced valid JSON
        return try {
            JSONObject(repaired)
            repaired
        } catch (e: JSONException) {
            null
        }
    }

    /**
     * Find the index of the last character of the last complete JSON entry.
     * A complete entry ends with a value terminator: number, string quote,
     * true/false/null literal, closing brace/bracket.
     */
    private fun findLastCompleteEntry(text: String): Int {
        // Walk backwards to find the last complete value
        var i = text.length - 1
        // Skip trailing whitespace
        while (i >= 0 && text[i].isWhitespace()) i--
        if (i < 0) return -1

        // If last char is a value terminator, we're at a good boundary
        val terminators = setOf('"', '}', ']', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', 'e', 'l', 's')  // true/false/null end chars

        if (text[i] in terminators) return i

        // Otherwise, scan back to find the last comma and use everything before it
        val lastComma = text.lastIndexOf(',')
        if (lastComma > 0) return lastComma - 1

        return -1
    }

    /**
     * Regex-based fallback: extract individual "key": value pairs.
     * Handles string, number, boolean, and null values.
     */
    private fun extractKeyValuePairs(text: String): Map<String, Any> {
        val fields = mutableMapOf<String, Any>()

        // Match "key": "string_value"
        val stringPattern = Regex(""""(\w+)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"""")
        for (match in stringPattern.findAll(text)) {
            val key = match.groupValues[1]
            val value = match.groupValues[2]
            fields[key] = coerceType(value)
        }

        // Match "key": number (int or float)
        val numberPattern = Regex(""""(\w+)"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)""")
        for (match in numberPattern.findAll(text)) {
            val key = match.groupValues[1]
            if (key !in fields) {  // Don't overwrite string matches
                val numStr = match.groupValues[2]
                fields[key] = parseNumber(numStr)
            }
        }

        // Match "key": true/false/null
        val literalPattern = Regex(""""(\w+)"\s*:\s*(true|false|null)\b""")
        for (match in literalPattern.findAll(text)) {
            val key = match.groupValues[1]
            if (key !in fields) {
                fields[key] = when (match.groupValues[2]) {
                    "true" -> true
                    "false" -> false
                    else -> "null"
                }
            }
        }

        return fields
    }

    /**
     * Flatten a JSONObject into a Map<String, Any>, recursively converting
     * nested JSONObject to Map and JSONArray to List.
     */
    private fun flattenJsonObject(json: JSONObject): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        for (key in json.keys()) {
            val value = json.get(key)
            result[key] = convertJsonValue(value)
        }
        return result
    }

    /**
     * Convert a JSON value to its Kotlin equivalent with type coercion.
     *
     * Note: The `org.json:json` library (used in unit tests) returns
     * `BigDecimal` for fractional numbers and `BigInteger` for large integers,
     * while Android's built-in `org.json` returns `Double`/`Long`. We handle
     * both to ensure consistent behavior across test and runtime environments.
     */
    private fun convertJsonValue(value: Any): Any {
        return when (value) {
            is JSONObject -> flattenJsonObject(value)
            is JSONArray -> convertJsonArray(value)
            JSONObject.NULL -> "null"
            is String -> coerceType(value)
            is Int -> value
            is Long -> value
            is Double -> {
                // Convert doubles that are actually integers (e.g. 3.0 -> 3)
                if (value == value.toLong().toDouble()) value.toLong() else value
            }
            is Float -> {
                val d = value.toDouble()
                if (d == d.toLong().toDouble()) d.toLong() else d
            }
            is BigDecimal -> {
                // BigDecimal from org.json:json library (non-Android)
                // Convert to the most appropriate Kotlin numeric type
                try {
                    val longVal = value.longValueExact()
                    if (longVal in Int.MIN_VALUE..Int.MAX_VALUE) longVal.toInt()
                    else longVal
                } catch (e: ArithmeticException) {
                    // Has fractional part — return as Double
                    value.toDouble()
                }
            }
            is BigInteger -> {
                val longVal = value.toLong()
                if (longVal in Int.MIN_VALUE..Int.MAX_VALUE) longVal.toInt()
                else longVal
            }
            is Boolean -> value
            is Number -> {
                // Catch-all for any other numeric types
                val d = value.toDouble()
                if (d == d.toLong().toDouble()) d.toLong() else d
            }
            else -> value.toString()
        }
    }

    /**
     * Convert a JSONArray to a Kotlin List.
     */
    private fun convertJsonArray(array: JSONArray): List<Any> {
        val result = mutableListOf<Any>()
        for (i in 0 until array.length()) {
            result.add(convertJsonValue(array.get(i)))
        }
        return result
    }

    /**
     * Coerce a string value to the most appropriate type.
     * - "3" -> 3 (Int)
     * - "3.14" -> 3.14 (Double)
     * - "true"/"false" -> Boolean
     * - everything else stays String
     */
    private fun coerceType(value: String): Any {
        // Boolean coercion
        if (value.equals("true", ignoreCase = true)) return true
        if (value.equals("false", ignoreCase = true)) return false

        // Integer coercion
        value.toIntOrNull()?.let { return it }

        // Long coercion (for large numbers)
        value.toLongOrNull()?.let { return it }

        // Double coercion
        value.toDoubleOrNull()?.let { return it }

        return value
    }

    /**
     * Parse a number string to the most appropriate numeric type.
     */
    private fun parseNumber(numStr: String): Any {
        // Try integer first
        numStr.toIntOrNull()?.let { return it }
        numStr.toLongOrNull()?.let { return it }
        numStr.toDoubleOrNull()?.let { return it }
        return numStr
    }

    /**
     * Result of parsing VLM output.
     *
     * @param fields Extracted key-value fields (empty if parsing failed).
     * @param strategy Which parsing strategy succeeded.
     * @param success Whether any strategy produced non-empty fields.
     */
    data class ParseResult(
        val fields: Map<String, Any>,
        val strategy: ParseStrategy,
        val success: Boolean
    )

    /**
     * Parsing strategy used to extract fields from VLM output.
     * Ordered from most reliable to least reliable.
     */
    enum class ParseStrategy {
        DIRECT_JSON,        // Text was valid JSON as-is
        CODE_FENCE,         // JSON extracted from markdown code fence
        BRACE_EXTRACTION,   // Found balanced { ... } in surrounding text
        REGEX_FALLBACK,     // Individual key-value pairs extracted via regex
        NONE                // All strategies failed
    }
}
