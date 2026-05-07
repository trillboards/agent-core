package com.trillboards.ctv.core.audience

import com.trillboards.ctv.core.socket.AgentSocketManager
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Phase 4 PR 3 — ClipUploadManager unit tests.
 *
 * The HTTP client and socket manager are mocked. We test the pure-function
 * payload-builder fully and validate the orchestration paths via the
 * mocked socket emitter.
 */
class ClipUploadManagerTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private lateinit var socketManager: AgentSocketManager
    private lateinit var httpClient: OkHttpClient

    @Before
    fun setUp() {
        socketManager = mockk(relaxed = true)
        httpClient = OkHttpClient()
    }

    private fun manager(socket: AgentSocketManager = socketManager): ClipUploadManager =
        ClipUploadManager(
            client = httpClient,
            socketManager = socket,
            apiBaseUrl = "https://api.test.example",
        )

    private fun fakeRecording(file: File): ClipRecording = ClipRecording(
        observationId = "obs_test_xyz",
        file = file,
        durationMs = 5_000L,
        sizeBytes = file.length(),
        recordedAtMs = 1_700_000_000_000L,
        contentType = "video/mp4",
    )

    // ── buildObservationPayload ─────────────────────────────────────────────

    @Test
    fun `buildObservationPayload includes all required fields`() {
        val mgr = manager()
        val payload = mgr.buildObservationPayload(
            observationId = "obs_payload_test",
            s3Key = "clips/obs_payload_test.mp4",
            cdnUrl = "https://cdn.example/clips/obs_payload_test.mp4",
            durationMs = 5_000L,
            sizeBytes = 12345L,
            contentType = "video/mp4",
            recordedAtMs = 1_700_000_000_000L,
            triggerId = "face_count_spike",
            observedValue = 7,
        )
        assertEquals("obs_payload_test", payload.getString("observation_id"))
        assertEquals("clips/obs_payload_test.mp4", payload.getString("s3_key"))
        assertEquals("video/mp4", payload.getString("content_type"))
        assertEquals(5_000L, payload.getLong("duration_ms"))
        assertEquals(12345L, payload.getLong("size_bytes"))
        assertEquals(1_700_000_000_000L, payload.getLong("recorded_at"))
        assertEquals("face_count_spike", payload.getString("trigger_id"))
        assertEquals(7, payload.getInt("observed_value"))
    }

    @Test
    fun `buildObservationPayload encodes observed_value types correctly`() {
        val mgr = manager()
        val intP = mgr.buildObservationPayload(
            "o1", "k1", "u1", 1L, 1L, "video/mp4", 0L, "t", 42,
        )
        assertEquals(42, intP.getInt("observed_value"))

        val boolP = mgr.buildObservationPayload(
            "o2", "k2", "u2", 1L, 1L, "video/mp4", 0L, "t", true,
        )
        assertEquals(true, boolP.getBoolean("observed_value"))

        val strP = mgr.buildObservationPayload(
            "o3", "k3", "u3", 1L, 1L, "video/mp4", 0L, "t", "happy",
        )
        assertEquals("happy", strP.getString("observed_value"))

        val nullP = mgr.buildObservationPayload(
            "o4", "k4", "u4", 1L, 1L, "video/mp4", 0L, "t", null,
        )
        assertEquals(JSONObject.NULL, nullP.get("observed_value"))

        // Non-primitive coerces to string
        data class Custom(val name: String) {
            override fun toString(): String = "Custom($name)"
        }
        val customP = mgr.buildObservationPayload(
            "o5", "k5", "u5", 1L, 1L, "video/mp4", 0L, "t", Custom("zz"),
        )
        assertEquals("Custom(zz)", customP.getString("observed_value"))
    }

    // ── emitObservation ─────────────────────────────────────────────────────

    @Test
    fun `emitObservation succeeds when socket connected`() {
        every { socketManager.isConnected() } returns true
        val eventSlot = slot<String>()
        val payloadSlot = slot<JSONObject>()
        every { socketManager.emit(capture(eventSlot), capture(payloadSlot)) } returns Unit

        val mgr = manager()
        val ok = mgr.emitObservation(
            observationId = "obs_emit_ok",
            s3Key = "clips/obs_emit_ok.mp4",
            cdnUrl = "https://cdn.example/clips/obs_emit_ok.mp4",
            durationMs = 5_000L,
            sizeBytes = 999L,
            contentType = "video/mp4",
            recordedAtMs = 1_700_000_000_000L,
            triggerId = "high_attention",
            observedValue = 0.92,
        )
        assertTrue(ok)
        assertEquals(ClipUploadManager.OBSERVATION_EVENT, eventSlot.captured)
        assertEquals("obs_emit_ok", payloadSlot.captured.getString("observation_id"))
        assertEquals("high_attention", payloadSlot.captured.getString("trigger_id"))
    }

    @Test
    fun `emitObservation skips when socket disconnected`() {
        every { socketManager.isConnected() } returns false

        val mgr = manager()
        val ok = mgr.emitObservation(
            observationId = "obs_emit_skip",
            s3Key = "clips/obs_emit_skip.mp4",
            cdnUrl = "https://cdn.example/clips/obs_emit_skip.mp4",
            durationMs = 5_000L,
            sizeBytes = 999L,
            contentType = "video/mp4",
            recordedAtMs = 1_700_000_000_000L,
            triggerId = "noise_spike",
            observedValue = "loud",
        )
        assertFalse(ok)
        verify(exactly = 0) { socketManager.emit(any(), any()) }
    }

    // ── uploadAndEmit refusal paths ────────────────────────────────────────

    @Test
    fun `uploadAndEmit returns null when staging file is missing`() = runBlocking {
        val missing = File(tempFolder.root, "doesnotexist.mp4")
        val mgr = manager()
        val result = mgr.uploadAndEmit(
            recording = fakeRecording(missing),
            fingerprint = "fp_123",
            triggerId = "face_count_spike",
            observedValue = 7,
        )
        assertNull(result)
    }

    @Test
    fun `uploadAndEmit returns null when staging file is empty`() = runBlocking {
        val empty = tempFolder.newFile("empty.mp4")
        // empty file: length == 0
        assertEquals(0L, empty.length())
        val mgr = manager()
        val result = mgr.uploadAndEmit(
            recording = fakeRecording(empty),
            fingerprint = "fp_123",
            triggerId = "face_count_spike",
            observedValue = 7,
        )
        assertNull(result)
    }

    // ── Companion contract ──────────────────────────────────────────────────

    @Test
    fun `OBSERVATION_EVENT name is stable contract`() {
        // Server-side socket.on must match this exact string. If you rename
        // it, you MUST also rename the server handler. CLAUDE.md "Before
        // renaming: grep all producers AND consumers".
        assertEquals("clipDryrunObservation", ClipUploadManager.OBSERVATION_EVENT)
    }
}
