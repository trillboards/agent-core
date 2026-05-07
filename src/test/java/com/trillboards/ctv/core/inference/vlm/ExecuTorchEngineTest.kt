package com.trillboards.ctv.core.inference.vlm

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.io.File

/**
 * Tests for [ExecuTorchEngine] — verifies graceful degradation when ExecuTorch
 * SDK is not available (which is always the case in JVM unit tests).
 *
 * ExecuTorchEngine accepts nullable Context, so we pass null for JVM unit testing.
 */
class ExecuTorchEngineTest {

    private fun createEngine(): ExecuTorchEngine = ExecuTorchEngine(null)

    @Test
    fun `engineName is ExecuTorch`() {
        val engine = createEngine()
        assertEquals("ExecuTorch", engine.engineName)
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
        assertFalse(engine.loadModel("/nonexistent/model.pte"))
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel returns false when model file is empty`() {
        val emptyFile = File.createTempFile("empty_model", ".pte")
        emptyFile.writeText("")
        emptyFile.deleteOnExit()

        val engine = createEngine()
        assertFalse(engine.loadModel(emptyFile.absolutePath))
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel returns false when SDK unavailable`() {
        val modelFile = File.createTempFile("test_model", ".pte")
        modelFile.writeText("fake ExecuTorch model data")
        modelFile.deleteOnExit()

        val engine = createEngine()
        assertFalse(engine.loadModel(modelFile.absolutePath))
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `isSdkAvailable returns false in JVM test environment`() {
        assertFalse(ExecuTorchEngine.isSdkAvailable())
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
        val modelFile = File.createTempFile("test_model", ".pte")
        modelFile.writeText("fake ExecuTorch model data")
        modelFile.deleteOnExit()

        val config = VLMConfig(
            maxTokens = 2048,
            temperature = 0.5f,
            useGpu = true,
            numThreads = 4
        )

        val engine = createEngine()
        assertFalse(engine.loadModel(modelFile.absolutePath, config))
    }

    @Test
    fun `loadModel resolves tokenizer path with co-located tokenizer`() {
        val modelDir = File.createTempFile("et_model_dir", "")
        modelDir.delete()
        modelDir.mkdirs()
        modelDir.deleteOnExit()

        val modelFile = File(modelDir, "model.pte")
        modelFile.writeText("fake ExecuTorch model")
        modelFile.deleteOnExit()

        val tokenizerFile = File(modelDir, "tokenizer.bin")
        tokenizerFile.writeText("fake tokenizer data")
        tokenizerFile.deleteOnExit()

        val engine = createEngine()
        assertFalse(engine.loadModel(modelFile.absolutePath))
    }

    @Test
    fun `engine implements VLMEngine interface`() {
        val engine: VLMEngine = createEngine()
        assertNotNull(engine.engineName)
        assertFalse(engine.isLoaded)
    }
}
