package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for VLMConfig defaults and VLMResponse data class behavior.
 * LiteRTLMEngine itself requires Android context; tested via VLMInferenceProcessorTest
 * with FakeVLMEngine, and integration tests on device.
 */
class VLMEngineTest {

    // --- VLMConfig defaults ---

    @Test
    fun `VLMConfig has sensible defaults`() {
        val config = VLMConfig()
        assertEquals(512, config.maxTokens)
        assertEquals(0.1f, config.temperature, 0.001f)
        assertTrue(config.useGpu)
        assertEquals(4, config.numThreads)
        assertNull(config.systemPrompt)
    }

    @Test
    fun `VLMConfig custom values are preserved`() {
        val config = VLMConfig(
            maxTokens = 1024,
            temperature = 0.7f,
            useGpu = false,
            numThreads = 8
        )
        assertEquals(1024, config.maxTokens)
        assertEquals(0.7f, config.temperature, 0.001f)
        assertFalse(config.useGpu)
        assertEquals(8, config.numThreads)
    }

    @Test
    fun `VLMConfig copy works for partial overrides`() {
        val base = VLMConfig()
        val modified = base.copy(maxTokens = 256, useGpu = false)

        assertEquals(256, modified.maxTokens)
        assertEquals(0.1f, modified.temperature, 0.001f)  // Unchanged
        assertFalse(modified.useGpu)
        assertEquals(4, modified.numThreads)  // Unchanged
        assertNull(modified.systemPrompt)  // Unchanged
    }

    @Test
    fun `VLMConfig systemPrompt can be set`() {
        val config = VLMConfig(systemPrompt = "Custom system prompt for testing")
        assertEquals("Custom system prompt for testing", config.systemPrompt)
        // Other defaults unchanged
        assertEquals(512, config.maxTokens)
        assertTrue(config.useGpu)
    }

    @Test
    fun `VLMConfig copy can override systemPrompt`() {
        val base = VLMConfig()
        assertNull(base.systemPrompt)

        val withPrompt = base.copy(systemPrompt = "Analyze retail traffic")
        assertEquals("Analyze retail traffic", withPrompt.systemPrompt)
        assertEquals(base.maxTokens, withPrompt.maxTokens)
    }

    // --- VLMResponse ---

    @Test
    fun `VLMResponse defaults`() {
        val response = VLMResponse(text = "hello", latencyMs = 100)
        assertEquals("hello", response.text)
        assertEquals(100L, response.latencyMs)
        assertEquals(0, response.tokensGenerated)
        assertFalse(response.parseSuccess)
        assertTrue(response.parsedFields.isEmpty())
    }

    @Test
    fun `VLMResponse with all fields set`() {
        val fields = mapOf<String, Any>("count" to 3, "mood" to "happy")
        val response = VLMResponse(
            text = "{\"count\": 3}",
            latencyMs = 5000,
            tokensGenerated = 25,
            parseSuccess = true,
            parsedFields = fields
        )
        assertEquals("{\"count\": 3}", response.text)
        assertEquals(5000L, response.latencyMs)
        assertEquals(25, response.tokensGenerated)
        assertTrue(response.parseSuccess)
        assertEquals(3, response.parsedFields["count"])
        assertEquals("happy", response.parsedFields["mood"])
    }

    @Test
    fun `VLMResponse equality and copy`() {
        val a = VLMResponse(text = "test", latencyMs = 50)
        val b = VLMResponse(text = "test", latencyMs = 50)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())

        val c = a.copy(latencyMs = 100)
        assertEquals("test", c.text)
        assertEquals(100L, c.latencyMs)
    }

    // --- FakeVLMEngine contract verification ---

    @Test
    fun `FakeVLMEngine implements VLMEngine interface correctly`() {
        val engine: VLMEngine = FakeVLMEngine()
        assertEquals("FakeVLMEngine", engine.engineName)
        assertFalse(engine.isLoaded)
        assertEquals(0f, engine.getMemoryUsageMb(), 0.001f)
    }

    @Test
    fun `FakeVLMEngine load and unload lifecycle`() {
        val engine = FakeVLMEngine()
        engine.loadShouldSucceed = true

        assertTrue(engine.loadModel("/test/model.bin"))
        assertTrue(engine.isLoaded)
        assertTrue(engine.getMemoryUsageMb() > 0f)

        engine.unload()
        assertFalse(engine.isLoaded)
        assertEquals(0f, engine.getMemoryUsageMb(), 0.001f)
    }

    @Test
    fun `FakeVLMEngine load failure`() {
        val engine = FakeVLMEngine()
        engine.loadShouldSucceed = false

        assertFalse(engine.loadModel("/test/model.bin"))
        assertFalse(engine.isLoaded)
    }
}
