package com.trillboards.ctv.core.inference.vlm

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.io.File

/**
 * Tests for [OnnxVLMEngine] — verifies graceful degradation when ONNX Runtime
 * is not available (which is always the case in JVM unit tests).
 *
 * OnnxVLMEngine accepts nullable Context, so we pass null for JVM unit testing.
 */
class OnnxVLMEngineTest {

    private fun createEngine(): OnnxVLMEngine = OnnxVLMEngine(null)

    @Test
    fun `engineName is ONNX-GenAI`() {
        val engine = createEngine()
        assertEquals("ONNX-GenAI", engine.engineName)
    }

    @Test
    fun `isLoaded returns false initially`() {
        val engine = createEngine()
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `getMemoryUsageMb returns 0 when not loaded`() {
        val engine = createEngine()
        assertEquals(0f, engine.getMemoryUsageMb(), 0.001f)
    }

    @Test
    fun `loadModel returns false when model file does not exist`() {
        val engine = createEngine()
        assertFalse(engine.loadModel("/nonexistent/model.onnx"))
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel returns false when model file is empty`() {
        val emptyFile = File.createTempFile("empty_model", ".onnx")
        emptyFile.writeText("")
        emptyFile.deleteOnExit()

        val engine = createEngine()
        assertFalse(engine.loadModel(emptyFile.absolutePath))
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel returns false when SDK unavailable`() {
        val modelFile = File.createTempFile("test_model", ".onnx")
        modelFile.writeText("fake ONNX model data")
        modelFile.deleteOnExit()

        val engine = createEngine()
        assertFalse(engine.loadModel(modelFile.absolutePath))
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel handles model directory`() {
        val modelDir = File.createTempFile("test_model_dir", "")
        modelDir.delete()
        modelDir.mkdirs()
        modelDir.deleteOnExit()

        val modelFile = File(modelDir, "model.onnx")
        modelFile.writeText("fake model weights")
        modelFile.deleteOnExit()

        val configFile = File(modelDir, "genai_config.json")
        configFile.writeText("{}")
        configFile.deleteOnExit()

        val engine = createEngine()
        assertFalse(engine.loadModel(modelDir.absolutePath))
    }

    @Test
    fun `isSdkAvailable returns false in JVM test environment`() {
        assertFalse(OnnxVLMEngine.isSdkAvailable())
    }

    @Test
    fun `generate returns empty response when not loaded`() = runBlocking {
        val engine = createEngine()
        val response = engine.generate("test prompt")
        assertEquals("", response.text)
        assertEquals(0L, response.latencyMs)
        assertEquals(0, response.tokensGenerated)
        assertFalse(response.parseSuccess)
    }

    @Test
    fun `unload is safe when not loaded`() {
        val engine = createEngine()
        engine.unload()
        assertFalse(engine.isLoaded)
        assertEquals(0f, engine.getMemoryUsageMb(), 0.001f)
    }

    @Test
    fun `unload is safe to call multiple times`() {
        val engine = createEngine()
        engine.unload()
        engine.unload()
        engine.unload()
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel with custom config does not throw`() {
        val modelFile = File.createTempFile("test_model", ".onnx")
        modelFile.writeText("fake ONNX model data")
        modelFile.deleteOnExit()

        val config = VLMConfig(
            maxTokens = 256,
            temperature = 0.3f,
            useGpu = true,
            numThreads = 2
        )

        val engine = createEngine()
        assertFalse(engine.loadModel(modelFile.absolutePath, config))
    }

    @Test
    fun `engine implements VLMEngine interface`() {
        val engine: VLMEngine = createEngine()
        assertNotNull(engine.engineName)
        assertFalse(engine.isLoaded)
    }
}
