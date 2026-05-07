package com.trillboards.ctv.core.inference.vlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [VLMEngineFactory].
 *
 * The factory accepts nullable Context, so we pass null for JVM unit testing.
 * LiteRTLMEngine requires non-null Context, so createEngine("litert-lm") returns
 * null in JVM tests — this is correct behavior (litert-lm needs Android Context).
 * The remaining engines (onnx-vlm, executorch) accept nullable Context.
 *
 * Note: GGUF/llama.cpp support was removed. LiteRT-LM is the sole VLM runtime.
 */
class VLMEngineFactoryTest {

    // --- Format validation ---

    @Test
    fun `isFormatSupported returns true for litert-lm`() {
        assertTrue(VLMEngineFactory.isFormatSupported("litert-lm"))
    }

    @Test
    fun `isFormatSupported returns false for gguf`() {
        assertFalse(VLMEngineFactory.isFormatSupported("gguf"))
    }

    @Test
    fun `isFormatSupported returns true for onnx-vlm`() {
        assertTrue(VLMEngineFactory.isFormatSupported("onnx-vlm"))
    }

    @Test
    fun `isFormatSupported returns true for executorch`() {
        assertTrue(VLMEngineFactory.isFormatSupported("executorch"))
    }

    @Test
    fun `isFormatSupported returns false for unknown`() {
        assertFalse(VLMEngineFactory.isFormatSupported("unknown"))
    }

    @Test
    fun `isFormatSupported returns false for empty string`() {
        assertFalse(VLMEngineFactory.isFormatSupported(""))
    }

    @Test
    fun `isFormatSupported returns false for misspelled format`() {
        assertFalse(VLMEngineFactory.isFormatSupported("litert_lm"))
    }

    @Test
    fun `isFormatSupported returns false for uppercase format`() {
        assertFalse(VLMEngineFactory.isFormatSupported("GGUF"))
    }

    @Test
    fun `isFormatSupported returns false for tflite`() {
        assertFalse(VLMEngineFactory.isFormatSupported("tflite"))
    }

    @Test
    fun `isFormatSupported returns false for pytorch`() {
        assertFalse(VLMEngineFactory.isFormatSupported("pytorch"))
    }

    // --- supportedFormats ---

    @Test
    fun `supportedFormats contains exactly three formats`() {
        assertEquals(3, VLMEngineFactory.supportedFormats.size)
    }

    @Test
    fun `supportedFormats contains litert-lm`() {
        assertTrue(VLMEngineFactory.supportedFormats.contains("litert-lm"))
    }

    @Test
    fun `supportedFormats does not contain gguf`() {
        assertFalse(VLMEngineFactory.supportedFormats.contains("gguf"))
    }

    @Test
    fun `supportedFormats contains onnx-vlm`() {
        assertTrue(VLMEngineFactory.supportedFormats.contains("onnx-vlm"))
    }

    @Test
    fun `supportedFormats contains executorch`() {
        assertTrue(VLMEngineFactory.supportedFormats.contains("executorch"))
    }

    // --- Factory creates correct engine types ---

    @Test
    fun `createEngine returns null for litert-lm with null context`() {
        // LiteRTLMEngine requires non-null Context — factory returns null when context is null
        val engine = VLMEngineFactory.createEngine(null, "litert-lm")
        assertNull("litert-lm needs non-null Context", engine)
    }

    @Test
    fun `createEngine returns null for gguf format`() {
        val engine = VLMEngineFactory.createEngine(null, "gguf")
        assertNull("gguf format is no longer supported", engine)
    }

    @Test
    fun `createEngine returns OnnxVLMEngine for onnx-vlm format`() {
        val engine = VLMEngineFactory.createEngine(null, "onnx-vlm")
        assertNotNull("Should create engine for onnx-vlm", engine)
        assertTrue("Should be OnnxVLMEngine", engine is OnnxVLMEngine)
        assertEquals("ONNX-GenAI", engine!!.engineName)
    }

    @Test
    fun `createEngine returns ExecuTorchEngine for executorch format`() {
        val engine = VLMEngineFactory.createEngine(null, "executorch")
        assertNotNull("Should create engine for executorch", engine)
        assertTrue("Should be ExecuTorchEngine", engine is ExecuTorchEngine)
        assertEquals("ExecuTorch", engine!!.engineName)
    }

    @Test
    fun `createEngine returns null for unknown format`() {
        val engine = VLMEngineFactory.createEngine(null, "unknown-format")
        assertNull("Should return null for unknown format", engine)
    }

    @Test
    fun `createEngine returns null for empty format`() {
        val engine = VLMEngineFactory.createEngine(null, "")
        assertNull("Should return null for empty format", engine)
    }

    // --- Engine state after creation ---

    @Test
    fun `new engines start unloaded`() {
        // Test the engines that accept null context (gguf removed)
        val formats = listOf("onnx-vlm", "executorch")
        for (format in formats) {
            val engine = VLMEngineFactory.createEngine(null, format)
            assertNotNull("Factory should create engine for $format", engine)
            assertFalse("$format engine should start unloaded", engine!!.isLoaded)
            assertEquals("$format engine memory should be 0",
                0f, engine.getMemoryUsageMb(), 0.001f)
        }
    }

    @Test
    fun `new engines have distinct engine names`() {
        val formats = listOf("onnx-vlm", "executorch")
        val names = formats.mapNotNull { format ->
            VLMEngineFactory.createEngine(null, format)?.engineName
        }.toSet()
        assertEquals("All testable engines should have unique names", 2, names.size)
    }

    // --- SDK unavailability ---

    @Test
    fun `all new engines return false for loadModel with missing file`() {
        val formats = listOf("onnx-vlm", "executorch")
        for (format in formats) {
            val engine = VLMEngineFactory.createEngine(null, format) ?: continue
            assertFalse("$format engine should fail loadModel for missing file",
                engine.loadModel("/nonexistent/model.bin"))
            assertFalse("$format engine should not be loaded after failure",
                engine.isLoaded)
        }
    }
}
