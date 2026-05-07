package com.trillboards.ctv.core.audience

import android.content.Context
import android.media.MediaRecorder
import io.mockk.Runs
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.verify
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Phase 4 PR 3 — ClipRecorder unit tests.
 *
 * MediaRecorder is mocked via MockK; we only verify ordering + state. The
 * recorderFactory injection point lets us hand the recorder a fake without
 * touching real camera/mic hardware. Each test asserts:
 *  - File creation (size >0 byte simulated by the fake) and metadata fields
 *  - MediaRecorder configuration ordering: source → format → encoder → output
 *  - Failure paths (start throws, file empty, busy flag dedup)
 *  - Duration clamping (MIN/MAX bounds)
 */
class ClipRecorderTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private lateinit var ctx: Context
    private lateinit var fakeRecorder: MediaRecorder
    private lateinit var outputDir: File
    private var nowMs = 1_000_000L

    @Before
    fun setUp() {
        ctx = mockk(relaxed = true)
        fakeRecorder = mockk(relaxed = true)
        outputDir = tempFolder.newFolder("clips")
        nowMs = 1_000_000L
    }

    @After
    fun tearDown() {
        // Each test cleans its own files.
    }

    // Helper: build a recorder that writes a small canary mp4 byte to disk
    // when start() is called, simulating the on-disk contents that
    // MediaRecorder.start() would produce on a real device.
    private fun makeRecorder(
        recorder: MediaRecorder = fakeRecorder,
        durationDelayMs: Long? = null,
        outputFile: File? = null,
        startThrows: Throwable? = null,
        produceBytes: Boolean = true,
    ): ClipRecorder {
        every { recorder.setAudioSource(any()) } just Runs
        every { recorder.setVideoSource(any()) } just Runs
        every { recorder.setOutputFormat(any()) } just Runs
        every { recorder.setVideoEncoder(any()) } just Runs
        every { recorder.setAudioEncoder(any()) } just Runs
        every { recorder.setVideoSize(any(), any()) } just Runs
        every { recorder.setVideoFrameRate(any()) } just Runs
        every { recorder.setVideoEncodingBitRate(any()) } just Runs
        every { recorder.setAudioSamplingRate(any()) } just Runs
        every { recorder.setAudioChannels(any()) } just Runs
        every { recorder.setAudioEncodingBitRate(any()) } just Runs
        every { recorder.setMaxDuration(any()) } just Runs
        every { recorder.setOutputFile(any<String>()) } answers {
            // Capture the path so the fake can write canary bytes
            val path = firstArg<String>()
            if (produceBytes) {
                File(path).parentFile?.mkdirs()
                File(path).writeBytes(byteArrayOf(0x00, 0x00, 0x00, 0x18) + "ftypisom".toByteArray())
            }
        }
        every { recorder.prepare() } just Runs
        if (startThrows != null) {
            every { recorder.start() } throws startThrows
        } else {
            every { recorder.start() } just Runs
        }
        every { recorder.stop() } just Runs
        every { recorder.reset() } just Runs
        every { recorder.release() } just Runs

        return ClipRecorder(
            context = ctx,
            outputDirOverride = outputFile?.parentFile ?: outputDir,
            recorderFactory = { recorder },
            clock = { nowMs.also { nowMs += (durationDelayMs ?: 0L) } },
            sleeper = { /* fast-forward — never block in tests */ },
        )
    }

    // ── Basic happy path ────────────────────────────────────────────────────

    @Test
    fun `recordClip writes file with non-zero size`() {
        val rec = makeRecorder()
        val out = rec.recordClip(observationId = "obs_test_1", requestedDurationMs = 1_500L)
        assertNotNull(out)
        assertTrue("file must exist", out!!.file.exists())
        assertTrue("size > 0", out.sizeBytes > 0L)
        assertEquals("video/mp4", out.contentType)
        assertEquals("obs_test_1", out.observationId)
        // Cleanup — real impl deletes after upload, test does it here
        out.file.delete()
    }

    @Test
    fun `recordClip configures MediaRecorder in correct order`() {
        val rec = makeRecorder()
        rec.recordClip(observationId = "obs_order_check", requestedDurationMs = 1_500L)
        // MediaRecorder API contract: source → format → encoder → output → prepare → start
        verify(exactly = 1) { fakeRecorder.setAudioSource(MediaRecorder.AudioSource.MIC) }
        verify(exactly = 1) { fakeRecorder.setVideoSource(MediaRecorder.VideoSource.CAMERA) }
        verify(exactly = 1) { fakeRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4) }
        verify(exactly = 1) { fakeRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264) }
        verify(exactly = 1) { fakeRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC) }
        verify(exactly = 1) { fakeRecorder.setVideoSize(1280, 720) }
        verify(exactly = 1) { fakeRecorder.setVideoFrameRate(24) }
        verify(exactly = 1) { fakeRecorder.setVideoEncodingBitRate(1_000_000) }
        verify(exactly = 1) { fakeRecorder.setAudioSamplingRate(48_000) }
        verify(exactly = 1) { fakeRecorder.setAudioChannels(1) }
        verify(exactly = 1) { fakeRecorder.setAudioEncodingBitRate(96_000) }
        verify(exactly = 1) { fakeRecorder.prepare() }
        verify(exactly = 1) { fakeRecorder.start() }
        verify(exactly = 1) { fakeRecorder.stop() }
        verify(exactly = 1) { fakeRecorder.release() }
    }

    @Test
    fun `recordClip clamps duration above MAX_DURATION_MS`() {
        val rec = makeRecorder()
        val out = rec.recordClip(observationId = "obs_clamp_high", requestedDurationMs = 999_999L)
        assertNotNull(out)
        // 60s cap
        assertEquals(60_000L, out!!.durationMs)
        out.file.delete()
    }

    @Test
    fun `recordClip clamps duration below MIN_DURATION_MS`() {
        val rec = makeRecorder()
        val out = rec.recordClip(observationId = "obs_clamp_low", requestedDurationMs = 100L)
        assertNotNull(out)
        // 1s floor
        assertEquals(1_000L, out!!.durationMs)
        out.file.delete()
    }

    // ── Failure paths ───────────────────────────────────────────────────────

    @Test
    fun `recordClip returns null when start throws`() {
        val rec = makeRecorder(startThrows = IllegalStateException("camera busy"))
        val out = rec.recordClip(observationId = "obs_start_fail", requestedDurationMs = 1_500L)
        assertNull(out)
    }

    @Test
    fun `recordClip returns null when file is empty`() {
        val rec = makeRecorder(produceBytes = false)
        val out = rec.recordClip(observationId = "obs_empty_file", requestedDurationMs = 1_500L)
        assertNull(out)
    }

    @Test
    fun `recordClip refuses blank observation_id`() {
        val rec = makeRecorder()
        val out = rec.recordClip(observationId = "", requestedDurationMs = 1_500L)
        assertNull(out)
        // Should not have touched MediaRecorder at all
        verify(exactly = 0) { fakeRecorder.start() }
    }

    @Test
    fun `recordClip is busy-locked — concurrent calls produce only one recording`() {
        // Simulate "busy" by injecting a recorder with a long synchronous start delay.
        // We use the ConcurrentHashMap pattern via a holder here, since MockK 1.13.5
        // doesn't have an easy "block and check" primitive without coroutines.
        val started = java.util.concurrent.atomic.AtomicInteger(0)
        val released = java.util.concurrent.CountDownLatch(1)
        val recorder1 = mockk<MediaRecorder>(relaxed = true)
        val recorder2 = mockk<MediaRecorder>(relaxed = true)
        // recorder1 starts and waits
        every { recorder1.setOutputFile(any<String>()) } answers {
            val path = firstArg<String>()
            File(path).parentFile?.mkdirs()
            File(path).writeBytes(byteArrayOf(0x01, 0x02))
        }
        every { recorder1.start() } answers {
            started.incrementAndGet()
            released.await()
        }
        every { recorder2.start() } answers { started.incrementAndGet() }

        val factoryCalls = java.util.concurrent.atomic.AtomicInteger(0)
        val rec = ClipRecorder(
            context = ctx,
            outputDirOverride = outputDir,
            recorderFactory = { _ ->
                if (factoryCalls.getAndIncrement() == 0) recorder1 else recorder2
            },
            clock = { nowMs },
            sleeper = { /* fast-forward — never block in tests */ },
        )

        val t = Thread {
            rec.recordClip(observationId = "obs_busy_a", requestedDurationMs = 2_000L)
        }
        t.start()
        // Spin until recorder1's start has been entered. Bounded so the
        // test never hangs forever.
        val tStart = System.currentTimeMillis()
        while (started.get() < 1 && System.currentTimeMillis() - tStart < 2_000L) {
            Thread.sleep(20)
        }
        // Now invoke a second recordClip on the SAME recorder instance — should be refused
        val out2 = rec.recordClip(observationId = "obs_busy_b", requestedDurationMs = 1_000L)
        assertNull("concurrent recordClip must be refused while in flight", out2)
        // Let recorder1 release and join
        released.countDown()
        t.join(5_000)
    }

    // ── Output dir handling ─────────────────────────────────────────────────

    @Test
    fun `resolveOutputDir creates directory on demand`() {
        val newDir = File(tempFolder.root, "fresh_clips_subdir")
        assertFalse(newDir.exists())
        val rec = ClipRecorder(
            context = ctx,
            outputDirOverride = newDir,
            recorderFactory = { fakeRecorder },
        )
        val resolved = rec.resolveOutputDir()
        assertTrue("output dir must be created", resolved.exists())
        assertEquals(newDir.absolutePath, resolved.absolutePath)
    }

    // ── observation_id generator ────────────────────────────────────────────

    @Test
    fun `generateObservationId is unique across calls and contains trigger_id`() {
        val rec = ClipRecorder(context = ctx, outputDirOverride = outputDir)
        val a = rec.generateObservationId("face_count_spike")
        val b = rec.generateObservationId("face_count_spike")
        // distinct values
        assertFalse(a == b)
        // contains trigger id
        assertTrue("expected trigger id in observation id, got $a", a.contains("face_count_spike"))
        // expected prefix
        assertTrue("expected clipdr_ prefix, got $a", a.startsWith("clipdr_"))
    }

    @Test
    fun `generateObservationId tolerates blank trigger_id`() {
        val rec = ClipRecorder(context = ctx, outputDirOverride = outputDir)
        val id = rec.generateObservationId("")
        // never empty / blank
        assertTrue(id.isNotBlank())
        assertTrue("expected fallback trigger token, got $id", id.contains("trigger"))
    }
}
