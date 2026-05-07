package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VLMResponseParserTest {

    // --- Strategy 1: Direct JSON ---

    @Test
    fun `parses clean JSON object`() {
        val text = """{"face_count": 3, "dominant_emotion": "happy", "age_range": "25-34"}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.DIRECT_JSON, result.strategy)
        assertEquals(3, result.fields["face_count"])
        assertEquals("happy", result.fields["dominant_emotion"])
        assertEquals("25-34", result.fields["age_range"])
    }

    @Test
    fun `parses JSON with nested objects`() {
        val text = """{"scene": {"brightness": 0.8, "color": "warm"}, "count": 5}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.DIRECT_JSON, result.strategy)
        assertEquals(5, result.fields["count"])
        @Suppress("UNCHECKED_CAST")
        val scene = result.fields["scene"] as Map<String, Any>
        assertEquals(0.8, scene["brightness"])
        assertEquals("warm", scene["color"])
    }

    @Test
    fun `parses JSON with arrays`() {
        val text = """{"emotions": ["happy", "neutral"], "scores": [0.9, 0.7]}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        @Suppress("UNCHECKED_CAST")
        val emotions = result.fields["emotions"] as List<Any>
        assertEquals(listOf("happy", "neutral"), emotions)
    }

    @Test
    fun `parses JSON with boolean values`() {
        val text = """{"is_crowded": true, "is_dark": false}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(true, result.fields["is_crowded"])
        assertEquals(false, result.fields["is_dark"])
    }

    @Test
    fun `parses JSON with null values`() {
        val text = """{"face_count": 0, "dominant_emotion": null}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(0, result.fields["face_count"])
        assertEquals("null", result.fields["dominant_emotion"])
    }

    @Test
    fun `coerces integer-valued doubles to int`() {
        // JSON spec allows 3.0 which org.json parses as BigDecimal/Double
        val text = """{"count": 3.0}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        // 3.0 should be coerced to Int 3 (fits in Int range)
        assertEquals(3, result.fields["count"])
    }

    // --- Strategy 2: Code Fence ---

    @Test
    fun `extracts JSON from markdown code fence with json tag`() {
        val text = """Here is the analysis:
```json
{"face_count": 2, "engagement": "high"}
```
Hope this helps!"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.CODE_FENCE, result.strategy)
        assertEquals(2, result.fields["face_count"])
        assertEquals("high", result.fields["engagement"])
    }

    @Test
    fun `extracts JSON from markdown code fence without tag`() {
        val text = """```
{"brightness": 0.7}
```"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.CODE_FENCE, result.strategy)
        assertEquals(0.7, result.fields["brightness"])
    }

    // --- Strategy 3: Brace Extraction ---

    @Test
    fun `extracts JSON from surrounding text`() {
        val text = """Based on my analysis of the image, here is the result:
{"face_count": 1, "mood": "neutral"}
This concludes my observation."""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.BRACE_EXTRACTION, result.strategy)
        assertEquals(1, result.fields["face_count"])
        assertEquals("neutral", result.fields["mood"])
    }

    @Test
    fun `handles nested braces in extracted JSON`() {
        val text = """Result: {"scene": {"type": "retail"}, "count": 4}."""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(4, result.fields["count"])
        @Suppress("UNCHECKED_CAST")
        val scene = result.fields["scene"] as Map<String, Any>
        assertEquals("retail", scene["type"])
    }

    // --- Strategy 4: Regex Fallback ---

    @Test
    fun `extracts key-value pairs via regex from malformed text`() {
        // Completely broken JSON that no parser can handle, but has recognizable pairs
        val text = """"face_count": 3, "mood": "happy", "brightness": 0.8, something broken here"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.REGEX_FALLBACK, result.strategy)
        assertEquals(3, result.fields["face_count"])
        assertEquals("happy", result.fields["mood"])
        assertEquals(0.8, result.fields["brightness"])
    }

    @Test
    fun `regex extracts boolean literals`() {
        val text = """"is_crowded": true, "is_dark": false"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(true, result.fields["is_crowded"])
        assertEquals(false, result.fields["is_dark"])
    }

    @Test
    fun `regex extracts null literal`() {
        val text = """"value": null"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals("null", result.fields["value"])
    }

    // --- Edge cases ---

    @Test
    fun `empty string returns failure`() {
        val result = VLMResponseParser.parse("")
        assertFalse(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.NONE, result.strategy)
        assertTrue(result.fields.isEmpty())
    }

    @Test
    fun `blank string returns failure`() {
        val result = VLMResponseParser.parse("   \n\t  ")
        assertFalse(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.NONE, result.strategy)
    }

    @Test
    fun `completely unparseable text returns failure`() {
        val result = VLMResponseParser.parse("I cannot analyze this image because it is too dark.")
        assertFalse(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.NONE, result.strategy)
        assertTrue(result.fields.isEmpty())
    }

    @Test
    fun `JSON with leading whitespace and trailing newlines`() {
        val text = "  \n  {\"count\": 5}  \n  "
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(5, result.fields["count"])
    }

    @Test
    fun `type coercion for string numbers`() {
        val text = """{"int_val": "42", "float_val": "3.14", "bool_val": "true", "str_val": "hello"}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        // String "42" should be coerced to Int 42
        assertEquals(42, result.fields["int_val"])
        // String "3.14" should be coerced to Double 3.14
        assertEquals(3.14, result.fields["float_val"])
        // String "true" should be coerced to Boolean true
        assertEquals(true, result.fields["bool_val"])
        // "hello" stays String
        assertEquals("hello", result.fields["str_val"])
    }

    @Test
    fun `negative numbers are parsed correctly`() {
        val text = """{"temperature": -5, "change": -0.3}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(-5, result.fields["temperature"])
        assertEquals(-0.3, result.fields["change"])
    }

    @Test
    fun `large integer stays as int when within int range`() {
        val text = """{"small": 100, "medium": 100000}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(100, result.fields["small"])
        assertEquals(100000, result.fields["medium"])
    }

    @Test
    fun `JSON with escaped quotes in strings`() {
        val text = """{"label": "person says \"hello\"", "count": 1}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(1, result.fields["count"])
    }

    @Test
    fun `empty JSON object returns success with empty fields`() {
        val text = """{}"""
        val result = VLMResponseParser.parse(text)

        // Empty JSON is valid but has no fields
        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.DIRECT_JSON, result.strategy)
        assertTrue(result.fields.isEmpty())
    }

    // --- ParseResult structure ---

    @Test
    fun `ParseResult contains all expected fields`() {
        val text = """{"a": 1}"""
        val result = VLMResponseParser.parse(text)

        assertTrue(result.success)
        assertEquals(VLMResponseParser.ParseStrategy.DIRECT_JSON, result.strategy)
        assertEquals(1, result.fields.size)
        assertEquals(1, result.fields["a"])
    }

    // --- Strategy ordering verification ---

    @Test
    fun `prefers direct JSON over code fence extraction`() {
        // This text IS valid JSON, so direct parse should win
        val text = """{"key": "value"}"""
        val result = VLMResponseParser.parse(text)
        assertEquals(VLMResponseParser.ParseStrategy.DIRECT_JSON, result.strategy)
    }

    @Test
    fun `code fence extraction wins over brace extraction`() {
        // JSON inside code fence with preamble text
        val text = """Here it is:
```json
{"key": "from_fence"}
```
Also {"key": "from_braces"} here."""
        val result = VLMResponseParser.parse(text)
        assertEquals(VLMResponseParser.ParseStrategy.CODE_FENCE, result.strategy)
        assertEquals("from_fence", result.fields["key"])
    }
}
