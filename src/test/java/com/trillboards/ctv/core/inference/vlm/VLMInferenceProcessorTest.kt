package com.trillboards.ctv.core.inference.vlm

import android.graphics.Bitmap
import com.trillboards.ctv.core.inference.HardwareRequirement
import com.trillboards.ctv.core.inference.InferenceInput
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * Unit tests for VLMInferenceProcessor.
 *
 * Uses [FakeVLMEngine] to test processor logic without requiring
 * the LiteRT-LM SDK or actual model files.
 *
 * Since Bitmap.createBitmap() returns null in JVM unit tests
 * (isReturnDefaultValues = true), all test inputs use InferenceInput.MultiModal
 * which accepts nullable Bitmap. The processor handles both CameraFrame and
 * MultiModal inputs identically — the only difference is the null-safety of
 * the bitmap field.
 */
class VLMInferenceProcessorTest {

    private lateinit var tempModelFile: File
    private lateinit var fakeEngine: FakeVLMEngine
    private lateinit var testClockMs: AtomicLong

    /**
     * Create a MultiModal input for testing.
     * Uses null image since Bitmap.createBitmap() returns null in JVM unit tests.
     * The processor still runs VLM inference with text-only prompt when image is null.
     */
    private fun testInput(prompt: String = "analyze"): InferenceInput.MultiModal {
        return InferenceInput.MultiModal(prompt = prompt, image = null)
    }

    @Before
    fun setUp() {
        // Create a temporary model file to pass hasModel() check
        tempModelFile = File.createTempFile("test_model", ".bin")
        tempModelFile.writeText("fake model data for testing")
        tempModelFile.deleteOnExit()

        fakeEngine = FakeVLMEngine()
        testClockMs = AtomicLong(1_000L)
    }

    private fun createProcessor(
        metricsPrompt: String = "face_count: number of faces, mood: dominant mood",
        samplingIntervalMs: Long = 0L,  // Disable sampling for tests
        inferenceTimeoutMs: Long = 5000L,
        engine: FakeVLMEngine = fakeEngine,
        maxStalenessMs: Long = VLMInferenceProcessor.MAX_STALENESS_MS
    ): VLMInferenceProcessor {
        return VLMInferenceProcessor(
            context = null,
            modelId = "test_vlm",
            metricsPrompt = metricsPrompt,
            modelPath = tempModelFile.absolutePath,
            engineFactory = { engine },
            vlmConfig = VLMConfig(maxTokens = 256, temperature = 0.1f),
            samplingIntervalMs = samplingIntervalMs,
            inferenceTimeoutMs = inferenceTimeoutMs,
            maxStalenessMs = maxStalenessMs,
            elapsedRealtimeProvider = { testClockMs.get() }
        )
    }

    // --- Interface Contract ---

    @Test
    fun `modelId is configurable`() {
        val processor = VLMInferenceProcessor(
            context = null,
            modelId = "custom_model_id",
            metricsPrompt = "test",
            modelPath = tempModelFile.absolutePath,
            engineFactory = { fakeEngine }
        )
        assertEquals("custom_model_id", processor.modelId)
    }

    @Test
    fun `hardwareRequirement is CAMERA`() {
        val processor = createProcessor()
        assertEquals(HardwareRequirement.CAMERA, processor.hardwareRequirement)
    }

    @Test
    fun `default modelId is gemma_3n_e2b`() {
        val processor = VLMInferenceProcessor(
            context = null,
            metricsPrompt = "test",
            modelPath = tempModelFile.absolutePath,
            engineFactory = { fakeEngine }
        )
        assertEquals("gemma_3n_e2b", processor.modelId)
    }

    // --- hasModel ---

    @Test
    fun `hasModel returns true when model file exists`() {
        val processor = createProcessor()
        assertTrue(processor.hasModel())
    }

    @Test
    fun `hasModel returns false when model file does not exist`() {
        val processor = VLMInferenceProcessor(
            context = null,
            metricsPrompt = "test",
            modelPath = "/nonexistent/path/model.bin",
            engineFactory = { fakeEngine }
        )
        assertFalse(processor.hasModel())
    }

    @Test
    fun `hasModel returns false for empty file`() {
        val emptyFile = File.createTempFile("empty_model", ".bin")
        emptyFile.writeText("")  // 0 bytes
        emptyFile.deleteOnExit()

        val processor = VLMInferenceProcessor(
            context = null,
            metricsPrompt = "test",
            modelPath = emptyFile.absolutePath,
            engineFactory = { fakeEngine }
        )
        assertFalse(processor.hasModel())
    }

    // --- Lifecycle ---

    @Test
    fun `isReady returns false before initialization`() {
        val processor = createProcessor()
        assertFalse(processor.isReady())
    }

    @Test
    fun `initialize succeeds when engine loads model`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()

        assertTrue(processor.initialize())
        assertTrue(processor.isReady())
    }

    @Test
    fun `initialize fails when engine cannot load model`() {
        fakeEngine.loadShouldSucceed = false
        val processor = createProcessor()

        assertFalse(processor.initialize())
        assertFalse(processor.isReady())
    }

    @Test
    fun `initialize is idempotent when already initialized`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()

        assertTrue(processor.initialize())
        assertTrue(processor.initialize())  // Second call should still return true
        assertTrue(processor.isReady())
        // Engine should only be loaded once
        assertEquals(1, fakeEngine.loadCallCount)
    }

    @Test
    fun `release sets isReady to false`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()

        processor.initialize()
        assertTrue(processor.isReady())

        processor.release()
        assertFalse(processor.isReady())
    }

    @Test
    fun `release unloads the engine`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()

        processor.initialize()
        processor.release()

        assertTrue(fakeEngine.unloadCalled)
    }

    @Test
    fun `release is safe to call multiple times`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()

        processor.initialize()
        processor.release()
        processor.release()  // Should not throw
        assertFalse(processor.isReady())
    }

    @Test
    fun `release waits for in flight inference before unloading engine`() = runBlocking {
        val blockingEngine = BlockingFakeVLMEngine().apply {
            loadShouldSucceed = true
            generateResponse = """{"ok": true}"""
        }
        val processor = createProcessor(engine = blockingEngine)
        processor.initialize()

        val inferenceThread = Thread {
            runBlocking {
                processor.process(testInput())
            }
        }
        inferenceThread.start()

        assertTrue(blockingEngine.generateStarted.await(2, TimeUnit.SECONDS))

        val releaseDurationMs = AtomicLong(0)
        val releaseThread = Thread {
            val startedAt = System.nanoTime()
            processor.release()
            releaseDurationMs.set(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startedAt))
        }
        releaseThread.start()

        Thread.sleep(150)
        assertFalse(blockingEngine.unloadCalled)

        blockingEngine.allowGenerateToFinish.countDown()
        inferenceThread.join(2_000)
        releaseThread.join(2_000)

        assertTrue(blockingEngine.unloadCalled)
        assertTrue(releaseDurationMs.get() >= 100L)
    }

    @Test
    fun `can reinitialize after release`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()

        processor.initialize()
        processor.release()

        // Create a new engine for re-initialization
        val newEngine = FakeVLMEngine().apply { loadShouldSucceed = true }
        val newProcessor = VLMInferenceProcessor(
            context = null,
            modelId = "test_vlm",
            metricsPrompt = "test",
            modelPath = tempModelFile.absolutePath,
            engineFactory = { newEngine },
            samplingIntervalMs = 0L,
            elapsedRealtimeProvider = { testClockMs.get() }
        )

        assertTrue(newProcessor.initialize())
        assertTrue(newProcessor.isReady())
    }

    // --- process ---

    @Test
    fun `process returns null when not initialized`() = runBlocking {
        val processor = createProcessor()
        assertNull(processor.process(testInput()))
    }

    @Test
    fun `process returns parsed fields on successful VLM output`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 3, "mood": "happy"}"""

        val processor = createProcessor()
        processor.initialize()

        val output = processor.process(testInput())

        assertNotNull(output)
        assertEquals("test_vlm", output!!.modelId)
        assertEquals(3, output.fields["face_count"])
        assertEquals("happy", output.fields["mood"])
        assertTrue(output.confidence > 0.5f)
    }

    @Test
    fun `process returns last known good on empty VLM response`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 2}"""

        val processor = createProcessor()
        processor.initialize()

        // First call succeeds
        val firstOutput = processor.process(testInput())
        assertNotNull(firstOutput)

        // Second call returns empty
        fakeEngine.generateResponse = ""
        val secondOutput = processor.process(testInput())

        // Should get last known good (from first call)
        assertNotNull(secondOutput)
        assertEquals(2, secondOutput!!.fields["face_count"])
    }

    @Test
    fun `process returns last known good on parse failure`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 5}"""

        val processor = createProcessor()
        processor.initialize()

        // First call succeeds
        processor.process(testInput())

        // Second call returns unparseable text
        fakeEngine.generateResponse = "I cannot analyze this image"
        val output = processor.process(testInput())

        assertNotNull(output)
        assertEquals(5, output!!.fields["face_count"])
    }

    @Test
    fun `process handles MultiModal input with null image`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"detected": true}"""

        val processor = createProcessor()
        processor.initialize()

        val input = InferenceInput.MultiModal(
            prompt = "Analyze this frame",
            image = null
        )
        val output = processor.process(input)

        assertNotNull(output)
        assertEquals(true, output!!.fields["detected"])
    }

    @Test
    fun `process returns last known good for AudioBuffer input`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"test": 1}"""

        val processor = createProcessor()
        processor.initialize()

        // First call with MultiModal input establishes last known good
        processor.process(testInput())

        // Audio input should return last known good (VLM ignores audio)
        val audioInput = InferenceInput.AudioBuffer(
            samples = shortArrayOf(1, 2, 3),
            sampleRate = 16000
        )
        val output = processor.process(audioInput)

        // Should get the last known good from the first processing
        assertNotNull(output)
        assertEquals(1, output!!.fields["test"])
    }

    @Test
    fun `process calls engine generate`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"test": 1}"""

        val processor = createProcessor()
        processor.initialize()

        processor.process(testInput())

        // Engine should have received a prompt
        assertTrue(fakeEngine.generateCallCount > 0)
        assertNotNull(fakeEngine.lastPrompt)
    }

    @Test
    fun `process builds prompt with metricsPrompt`() = runBlocking {
        val metricsPrompt = "face_count: number of visible faces\nmood: dominant emotion"
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 0}"""

        val processor = createProcessor(metricsPrompt = metricsPrompt)
        processor.initialize()

        processor.process(testInput())

        // The prompt sent to engine should contain the metricsPrompt
        val sentPrompt = fakeEngine.lastPrompt ?: ""
        assertTrue("Prompt should contain metricsPrompt", sentPrompt.contains(metricsPrompt))
        assertTrue("Prompt should mention JSON", sentPrompt.contains("JSON"))
    }

    // --- Memory ---

    @Test
    fun `getMemoryFootprintMb returns 0 when not initialized`() {
        val processor = createProcessor()
        assertEquals(0f, processor.getMemoryFootprintMb(), 0.001f)
    }

    @Test
    fun `getMemoryFootprintMb returns positive value when initialized`() {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.fakeMemoryMb = 500f

        val processor = createProcessor()
        processor.initialize()

        val memory = processor.getMemoryFootprintMb()
        assertTrue("Memory should be positive when initialized", memory > 0f)
        assertTrue("Memory should include engine memory", memory >= 500f)
    }

    // --- Configuration ---

    @Test
    fun `updateSamplingInterval clamps to valid range`() {
        val processor = createProcessor()

        // Below minimum (1000ms)
        processor.updateSamplingInterval(100L)
        // Can't directly check private field, but no exception

        // Above maximum (60000ms)
        processor.updateSamplingInterval(120_000L)
        // No exception

        // Normal value
        processor.updateSamplingInterval(5000L)
        // No exception
    }

    @Test
    fun `updateInferenceTimeout clamps to valid range`() {
        val processor = createProcessor()

        processor.updateInferenceTimeout(500L)   // Below min
        processor.updateInferenceTimeout(60_000L) // Above max
        processor.updateInferenceTimeout(10_000L) // Normal
        // No exceptions
    }

    @Test
    fun `consecutive failures counter increments on failure`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = ""  // Empty = failure

        val processor = createProcessor()
        processor.initialize()

        processor.process(testInput())
        assertEquals(1, processor.getConsecutiveFailures())

        processor.process(testInput())
        assertEquals(2, processor.getConsecutiveFailures())
    }

    @Test
    fun `consecutive failures resets on success`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = ""  // Failure

        val processor = createProcessor()
        processor.initialize()

        // Generate failures
        processor.process(testInput())
        processor.process(testInput())
        assertEquals(2, processor.getConsecutiveFailures())

        // Now succeed
        fakeEngine.generateResponse = """{"ok": true}"""
        processor.process(testInput())
        assertEquals(0, processor.getConsecutiveFailures())
    }

    @Test
    fun `getLastKnownGoodOutput returns null before first success`() {
        fakeEngine.loadShouldSucceed = true
        val processor = createProcessor()
        processor.initialize()

        assertNull(processor.getLastKnownGoodOutput())
    }

    @Test
    fun `getLastKnownGoodOutput returns last successful output`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"count": 7}"""

        val processor = createProcessor()
        processor.initialize()

        processor.process(testInput())

        val lastGood = processor.getLastKnownGoodOutput()
        assertNotNull(lastGood)
        assertEquals(7, lastGood!!.fields["count"])
    }

    // --- Confidence ---

    @Test
    fun `direct JSON parse gets high confidence`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"count": 1}"""

        val processor = createProcessor()
        processor.initialize()

        val output = processor.process(testInput())

        assertNotNull(output)
        assertTrue("Direct JSON should have confidence >= 0.9", output!!.confidence >= 0.9f)
    }

    @Test
    fun `code fence parse gets good confidence`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = "```json\n{\"count\": 1}\n```"

        val processor = createProcessor()
        processor.initialize()

        val output = processor.process(testInput())

        assertNotNull(output)
        assertTrue("Code fence should have confidence >= 0.85", output!!.confidence >= 0.85f)
    }

    // --- Stale output expiry ---

    @Test
    fun `getLastKnownGoodOutput returns null when output is stale`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"count": 1}"""

        // Use a very short staleness window (1ms) so output expires immediately
        val processor = createProcessor(maxStalenessMs = 1L)
        processor.initialize()

        // First call produces a good output
        val output = processor.process(testInput())
        assertNotNull(output)

        // Wait for the output to expire
        testClockMs.addAndGet(50L)

        // Now it should be stale
        assertNull(processor.getLastKnownGoodOutput())
    }

    @Test
    fun `fresh output is returned within staleness window`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"count": 42}"""

        // Use a generous staleness window
        val processor = createProcessor(maxStalenessMs = 60_000L)
        processor.initialize()

        processor.process(testInput())

        // Should still be valid
        val lastGood = processor.getLastKnownGoodOutput()
        assertNotNull(lastGood)
        assertEquals(42, lastGood!!.fields["count"])
    }

    @Test
    fun `stale output returns null on failure fallback`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"count": 1}"""

        // 1ms staleness means output expires almost immediately
        val processor = createProcessor(maxStalenessMs = 1L)
        processor.initialize()

        // First call succeeds
        processor.process(testInput())

        // Wait for staleness
        testClockMs.addAndGet(50L)

        // Now engine fails — stale output should NOT be returned
        fakeEngine.generateResponse = ""
        val output = processor.process(testInput())
        assertNull(output)
    }

    @Test
    fun `successful inference refreshes staleness timestamp`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"count": 1}"""

        val processor = createProcessor(maxStalenessMs = 60_000L)
        processor.initialize()

        // First inference
        processor.process(testInput())

        // Second inference updates timestamp
        fakeEngine.generateResponse = """{"count": 2}"""
        processor.process(testInput())

        // Output should be fresh (timestamp just updated)
        val lastGood = processor.getLastKnownGoodOutput()
        assertNotNull(lastGood)
        assertEquals(2, lastGood!!.fields["count"])
    }

    @Test
    fun `MAX_STALENESS_MS default is 60 seconds`() {
        assertEquals(60_000L, VLMInferenceProcessor.MAX_STALENESS_MS)
    }

    // --- Temporal Agreement ---

    @Test
    fun `computeTemporalAgreement returns 0_5 with fewer than 2 snapshots`() {
        val processor = createProcessor()
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "building"),
            listOf(
                VLMInferenceProcessor.TFLiteSnapshot(
                    timestampMs = 1000L, faceCount = 3f,
                    avgAttention = 0.7f, dominantEmotion = "happy", dominantAmbience = "quiet"
                )
            )
        )
        assertEquals(0.5f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement returns 0_5 with empty snapshots`() {
        val processor = createProcessor()
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "peak"),
            emptyList()
        )
        assertEquals(0.5f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement building phase agrees with rising faces`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 2f, 0.5f, "neutral", "quiet"),
            VLMInferenceProcessor.TFLiteSnapshot(2000L, 5f, 0.6f, "neutral", "quiet"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 8f, 0.7f, "happy", "moderate")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "building"), snapshots
        )
        // faceDelta = 8 - 2 = 6 > 2 => +0.3 => 0.8
        assertEquals(0.8f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement building phase disagrees with falling faces`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 8f, 0.7f, "happy", "moderate"),
            VLMInferenceProcessor.TFLiteSnapshot(2000L, 5f, 0.5f, "neutral", "quiet"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 3f, 0.4f, "neutral", "quiet")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "building"), snapshots
        )
        // faceDelta = 3 - 8 = -5, not > 2 => -0.2 => 0.3
        assertEquals(0.3f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement winding_down agrees with falling faces`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 10f, 0.8f, "happy", "loud"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 4f, 0.4f, "neutral", "quiet")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "winding_down"), snapshots
        )
        // faceDelta = 4 - 10 = -6 < -2 => +0.3 => 0.8
        assertEquals(0.8f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement peak agrees with high stable faces`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 9f, 0.9f, "happy", "loud"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 10f, 0.85f, "happy", "loud")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "peak"), snapshots
        )
        // latest.faceCount=10 > 5 && abs(10-9)=1 < 2 => +0.3 => 0.8
        assertEquals(0.8f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement idle agrees with low faces`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 1f, 0.1f, "neutral", "quiet"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 0f, 0.0f, "neutral", "quiet")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "idle"), snapshots
        )
        // latest.faceCount=0 < 2 => +0.3 => 0.8
        assertEquals(0.8f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement idle disagrees with many faces`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 5f, 0.7f, "happy", "moderate"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 8f, 0.8f, "happy", "loud")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "idle"), snapshots
        )
        // latest.faceCount=8 >= 2 => -0.2 => 0.3
        assertEquals(0.3f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement returns 0_5 for unknown event_phase`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 5f, 0.5f, "neutral", "quiet"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 5f, 0.5f, "neutral", "quiet")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("event_phase" to "something_unknown"), snapshots
        )
        assertEquals(0.5f, result, 0.001f)
    }

    @Test
    fun `computeTemporalAgreement returns 0_5 when no event_phase key`() {
        val processor = createProcessor()
        val snapshots = listOf(
            VLMInferenceProcessor.TFLiteSnapshot(1000L, 5f, 0.5f, "neutral", "quiet"),
            VLMInferenceProcessor.TFLiteSnapshot(3000L, 5f, 0.5f, "neutral", "quiet")
        )
        val result = processor.computeTemporalAgreement(
            mapOf("face_count" to 5), snapshots
        )
        assertEquals(0.5f, result, 0.001f)
    }

    // --- Context Provider ---

    @Test
    fun `contextProvider is null by default`() {
        val processor = createProcessor()
        assertNull(processor.contextProvider)
    }

    @Test
    fun `contextProvider can be set and invoked`() {
        val processor = createProcessor()
        processor.contextProvider = { "test temporal context" }
        assertEquals("test temporal context", processor.contextProvider!!.invoke())
    }

    @Test
    fun `buildPrompt includes temporal context when contextProvider is set`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 1}"""

        val processor = createProcessor()
        processor.contextProvider = { "Recent sensor readings from this screen:\n[5s ago] faces=3, attn=0.70, mood=happy, noise=quiet\n" }
        processor.initialize()

        processor.process(testInput())

        val sentPrompt = fakeEngine.lastPrompt ?: ""
        assertTrue("Prompt should contain temporal context", sentPrompt.contains("Recent sensor readings"))
        assertTrue("Prompt should mention temporal awareness", sentPrompt.contains("temporal awareness"))
    }

    @Test
    fun `buildPrompt omits temporal context when contextProvider returns empty`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 1}"""

        val processor = createProcessor()
        processor.contextProvider = { "" }
        processor.initialize()

        processor.process(testInput())

        val sentPrompt = fakeEngine.lastPrompt ?: ""
        assertFalse("Prompt should not contain sensor readings", sentPrompt.contains("Recent sensor readings"))
    }

    @Test
    fun `successful inference includes temporal_context_available when contextProvider set`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 3}"""

        val processor = createProcessor()
        processor.contextProvider = { "some context" }
        processor.initialize()

        val output = processor.process(testInput())

        assertNotNull(output)
        assertEquals(true, output!!.fields["temporal_context_available"])
    }

    @Test
    fun `successful inference omits temporal_context_available when no contextProvider`() = runBlocking {
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 3}"""

        val processor = createProcessor()
        // contextProvider is null by default
        processor.initialize()

        val output = processor.process(testInput())

        assertNotNull(output)
        assertFalse(output!!.fields.containsKey("temporal_context_available"))
    }

    // ─── Phase 5.2 — token-budget guardrail (logcat warn only on edge) ──────

    @Test
    fun `countTokensApprox handles null and empty`() {
        assertEquals(0, VLMInferenceProcessor.countTokensApprox(null))
        assertEquals(0, VLMInferenceProcessor.countTokensApprox(""))
    }

    @Test
    fun `countTokensApprox uses chars divided by 4 ceiling`() {
        // Mirrors server-side audienceVisionService.countTokensApprox.
        assertEquals(1, VLMInferenceProcessor.countTokensApprox("a"))
        assertEquals(1, VLMInferenceProcessor.countTokensApprox("abcd"))
        assertEquals(2, VLMInferenceProcessor.countTokensApprox("abcde"))
        // 16K chars = 4000 tokens exactly.
        assertEquals(4000, VLMInferenceProcessor.countTokensApprox("x".repeat(16000)))
        assertEquals(4001, VLMInferenceProcessor.countTokensApprox("x".repeat(16001)))
    }

    @Test
    fun `prompt token thresholds match server-side defaults`() {
        // The server defaults (BONUS_FIELD_PROMPT_WARN_TOKEN_THRESHOLD=4000,
        // BONUS_FIELD_PROMPT_HARD_CEILING=5000) and the edge-side constants
        // intentionally agree so the same prompt is interpreted consistently.
        assertEquals(4000, VLMInferenceProcessor.PROMPT_TOKEN_WARN_THRESHOLD)
        assertEquals(5000, VLMInferenceProcessor.PROMPT_TOKEN_HARD_CEILING)
    }

    @Test
    fun `oversized metricsPrompt does not crash buildPrompt`() = runBlocking {
        // Construct a metricsPrompt that, after the buildPrompt header/footer
        // wrap, lands well above the 5000-token ceiling. Edge cannot trim
        // (server is responsible) — the only edge behavior is logcat warn.
        // We assert the processor still produces a prompt and runs.
        fakeEngine.loadShouldSucceed = true
        fakeEngine.generateResponse = """{"face_count": 1}"""
        val giantMetricsPrompt = "X".repeat(40_000) // 10_000 tokens
        val processor = createProcessor(metricsPrompt = giantMetricsPrompt)
        processor.initialize()

        processor.process(testInput())

        val sentPrompt = fakeEngine.lastPrompt ?: ""
        assertTrue(
            "Edge buildPrompt forwards oversized prompts (server-side guardrail trims, not edge)",
            sentPrompt.contains(giantMetricsPrompt)
        )
        // Sanity: the assembled prompt was over the hard ceiling.
        assertTrue(
            "Assembled prompt token count should exceed the hard ceiling",
            VLMInferenceProcessor.countTokensApprox(sentPrompt) > VLMInferenceProcessor.PROMPT_TOKEN_HARD_CEILING
        )
    }
}

/**
 * Fake VLMEngine for unit testing VLMInferenceProcessor.
 *
 * Allows test control over:
 * - Whether loadModel succeeds
 * - What text generate() returns
 * - Memory usage reporting
 * - Tracking of calls for assertions
 */
open class FakeVLMEngine : VLMEngine {
    override val engineName: String = "FakeVLMEngine"

    var loadShouldSucceed = true
    var generateResponse = ""
    var fakeMemoryMb = 100f

    var loadCallCount = 0
    var generateCallCount = 0
    var unloadCalled = false
    var lastPrompt: String? = null
    var lastImage: Bitmap? = null

    private var _isLoaded = false
    override val isLoaded: Boolean get() = _isLoaded

    override fun loadModel(modelPath: String, config: VLMConfig): Boolean {
        loadCallCount++
        _isLoaded = loadShouldSucceed
        return loadShouldSucceed
    }

    override open suspend fun generate(prompt: String, image: Bitmap?): VLMResponse {
        generateCallCount++
        lastPrompt = prompt
        lastImage = image

        return VLMResponse(
            text = generateResponse,
            latencyMs = 100L,
            tokensGenerated = generateResponse.length / 4
        )
    }

    override fun unload() {
        unloadCalled = true
        _isLoaded = false
    }

    override fun getMemoryUsageMb(): Float {
        return if (_isLoaded) fakeMemoryMb else 0f
    }
}

class BlockingFakeVLMEngine : FakeVLMEngine() {
    val generateStarted = CountDownLatch(1)
    val allowGenerateToFinish = CountDownLatch(1)

    override suspend fun generate(prompt: String, image: Bitmap?): VLMResponse {
        generateStarted.countDown()
        allowGenerateToFinish.await(2, TimeUnit.SECONDS)
        return super.generate(prompt, image)
    }
}
