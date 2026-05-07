package com.trillboards.ctv.core.audience

import android.util.Log
import com.trillboards.ctv.core.socket.AgentSocketManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Phase 4 PR 3 — staged-clip uploader for the DRY-RUN path.
 *
 * Given a [ClipRecording] produced by [ClipRecorder]:
 *   1. POST `/v2/earner/clip-dryrun-upload-url` with the device fingerprint
 *      and observation_id; receive `{uploadUrl, cdnUrl, key, expiresIn}`.
 *   2. PUT the local `mp4` to the presigned URL with `Content-Type: video/mp4`.
 *   3. Emit a `clipDryrunObservation` socket event with the
 *      `{observation_id, s3_key, duration_ms, content_type, recorded_at,
 *      trigger_id, observed_value}` payload so the API can write a row
 *      to `observation_stream` with `observation_family='clip_dryrun'`.
 *   4. Delete the local file. Failures are logged loudly and the observation
 *      is NOT emitted (caller can re-trigger; idempotency at the API end
 *      catches dupes when the same observation_id eventually lands).
 *
 * ## Why presigned URL (not direct PUT to AWS SDK from device)
 * Mirrors the existing `ScreenshotManager` flow. The API mints a short-lived
 * presigned URL bound to the device's authenticated identity, so:
 * - The device never holds AWS credentials.
 * - The S3 key namespace (`clips/<observation_id>.mp4`) is server-controlled.
 * - The expiry window caps the blast radius of a leaked URL.
 *
 * ## Why DRY-RUN (no Gemini call)
 * PR 4 will gate the actual `multimodalDryrun → multimodalLive` cutover on
 * a successful flight of this PR's emit path. PR 3 only establishes the
 * pipeline; the row in `observation_stream` is the verification signal.
 *
 * @see ClipRecorder For the recorder half
 * @see com.trillboards.ctv.core.net.ApiClient For the auth-bearing HTTP client
 */
class ClipUploadManager(
    private val client: OkHttpClient,
    private val socketManager: AgentSocketManager,
    private val apiBaseUrl: String,
    private val authHeaderProvider: () -> Map<String, String> = { emptyMap() },
) {

    companion object {
        private const val TAG = "ClipUploadManager"
        private const val ENDPOINT_PATH = "/v2/earner/clip-dryrun-upload-url"
        private val APPLICATION_JSON = "application/json; charset=utf-8".toMediaType()

        /** Network connect/read timeouts for the presigned-URL request. */
        private const val PRESIGN_TIMEOUT_S = 10L

        /** Network timeouts for the actual S3 PUT. */
        private const val UPLOAD_TIMEOUT_S = 60L

        /** Socket event name — must match server `socket.on` handler. */
        const val OBSERVATION_EVENT = "clipDryrunObservation"
    }

    /** API response shape for `POST /v2/earner/clip-dryrun-upload-url`. */
    data class PresignedClipUpload(
        val uploadUrl: String,
        val cdnUrl: String,
        val key: String,
        val expiresIn: Int,
    )

    /** Final result of the upload+emit cycle. */
    data class UploadResult(
        val observationId: String,
        val s3Key: String,
        val cdnUrl: String,
        val sizeBytes: Long,
        val durationMs: Long,
        val emitted: Boolean,
    )

    /**
     * Run the full presign → PUT → emit → cleanup flow for a recorded clip.
     *
     * Caller must invoke from an IO context (the work itself does
     * `withContext(Dispatchers.IO)` for HTTP I/O).
     *
     * @param recording The completed clip recording.
     * @param fingerprint Device fingerprint (carried in the presign request).
     * @param triggerId Originating trigger_id from [ClipTriggerEvaluator].
     * @param observedValue The signal value that fired the rule.
     * @return [UploadResult] on success (including failed-emit, where the
     *         upload landed but the socket event didn't), null on hard
     *         presign / upload failure.
     */
    suspend fun uploadAndEmit(
        recording: ClipRecording,
        fingerprint: String,
        triggerId: String,
        observedValue: Any?,
    ): UploadResult? = withContext(Dispatchers.IO) {
        if (!recording.file.exists() || recording.file.length() == 0L) {
            Log.w(TAG, "[upload] refused — recording file missing or empty: ${recording.file}")
            return@withContext null
        }

        val presigned = requestPresignedUrl(recording.observationId, fingerprint)
        if (presigned == null) {
            Log.w(TAG, "[upload] presign failed for observation_id=${recording.observationId}")
            // Keep file so a retry path could pick it up later. Caller does not
            // retry in PR 3 — this is the dry-run; PR 4 will widen with retry.
            return@withContext null
        }

        val uploaded = putToS3(presigned.uploadUrl, recording.file, recording.contentType)
        if (!uploaded) {
            Log.w(TAG, "[upload] S3 PUT failed for observation_id=${recording.observationId} key=${presigned.key}")
            return@withContext null
        }

        val emitted = emitObservation(
            observationId = recording.observationId,
            s3Key = presigned.key,
            cdnUrl = presigned.cdnUrl,
            durationMs = recording.durationMs,
            sizeBytes = recording.sizeBytes,
            contentType = recording.contentType,
            recordedAtMs = recording.recordedAtMs,
            triggerId = triggerId,
            observedValue = observedValue,
        )

        // Always delete the local file once the bytes are durable in S3 —
        // the device cache is finite (Tab S11 has 12 GB RAM but only ~2 GB
        // free on the partition shared with the WebView profile).
        try {
            if (!recording.file.delete()) {
                Log.w(TAG, "[upload] could not delete staging file ${recording.file.absolutePath}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "[upload] delete failed: ${e.message}")
        }

        UploadResult(
            observationId = recording.observationId,
            s3Key = presigned.key,
            cdnUrl = presigned.cdnUrl,
            sizeBytes = recording.sizeBytes,
            durationMs = recording.durationMs,
            emitted = emitted,
        )
    }

    /**
     * Hit the API for a presigned PUT URL. Returns null on any non-2xx /
     * malformed response.
     *
     * The API endpoint mirrors `getScreenshotUploadUrl` semantically — it's
     * the same auth gate (device-token via `requireDeviceRequestAuth`).
     */
    internal suspend fun requestPresignedUrl(
        observationId: String,
        fingerprint: String,
    ): PresignedClipUpload? = withContext(Dispatchers.IO) {
        val payload = JSONObject().apply {
            put("fingerprint", fingerprint)
            put("observation_id", observationId)
            put("content_type", ClipRecorder.CONTENT_TYPE)
        }

        val builder = Request.Builder()
            .url("$apiBaseUrl$ENDPOINT_PATH")
            .post(payload.toString().toRequestBody(APPLICATION_JSON))
        // Add any auth headers the surrounding service has wired up
        for ((k, v) in authHeaderProvider()) {
            builder.addHeader(k, v)
        }
        val request = builder.build()

        val timedClient = client.newBuilder()
            .connectTimeout(PRESIGN_TIMEOUT_S, TimeUnit.SECONDS)
            .readTimeout(PRESIGN_TIMEOUT_S, TimeUnit.SECONDS)
            .build()

        try {
            timedClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w(TAG, "[presign] failed: ${response.code} ${response.message}")
                    return@withContext null
                }
                val body = response.body?.string() ?: return@withContext null
                val json = JSONObject(body)
                val uploadUrl = json.optString("uploadUrl").takeIf { it.isNotBlank() } ?: return@withContext null
                val cdnUrl = json.optString("cdnUrl")
                val key = json.optString("key").takeIf { it.isNotBlank() } ?: return@withContext null
                val expiresIn = json.optInt("expiresIn", 1800)
                PresignedClipUpload(uploadUrl, cdnUrl, key, expiresIn)
            }
        } catch (e: Exception) {
            Log.w(TAG, "[presign] exception", e)
            null
        }
    }

    /**
     * PUT the recorded file to S3 via the presigned URL. Returns true on 2xx.
     */
    internal suspend fun putToS3(uploadUrl: String, file: File, contentType: String): Boolean =
        withContext(Dispatchers.IO) {
            val mediaType = contentType.toMediaType()
            val body: RequestBody = file.asRequestBody(mediaType)
            val request = Request.Builder()
                .url(uploadUrl)
                .put(body)
                .header("Content-Type", contentType)
                .build()

            val timedClient = client.newBuilder()
                .connectTimeout(UPLOAD_TIMEOUT_S, TimeUnit.SECONDS)
                .writeTimeout(UPLOAD_TIMEOUT_S, TimeUnit.SECONDS)
                .readTimeout(UPLOAD_TIMEOUT_S, TimeUnit.SECONDS)
                .build()

            try {
                timedClient.newCall(request).execute().use { response ->
                    val ok = response.isSuccessful
                    if (!ok) Log.w(TAG, "[s3] PUT failed: ${response.code} ${response.message}")
                    ok
                }
            } catch (e: Exception) {
                Log.w(TAG, "[s3] PUT exception", e)
                false
            }
        }

    /**
     * Emit the `clipDryrunObservation` socket event so the API writes a row
     * to `observation_stream` with `observation_family='clip_dryrun'`.
     *
     * Returns true if the event made it to the socket bus; false if the
     * socket isn't connected or the build failed.
     */
    internal fun emitObservation(
        observationId: String,
        s3Key: String,
        cdnUrl: String,
        durationMs: Long,
        sizeBytes: Long,
        contentType: String,
        recordedAtMs: Long,
        triggerId: String,
        observedValue: Any?,
    ): Boolean {
        if (!socketManager.isConnected()) {
            Log.w(TAG, "[emit] socket not connected — observation NOT emitted (s3_key=$s3Key)")
            return false
        }
        val payload = buildObservationPayload(
            observationId = observationId,
            s3Key = s3Key,
            cdnUrl = cdnUrl,
            durationMs = durationMs,
            sizeBytes = sizeBytes,
            contentType = contentType,
            recordedAtMs = recordedAtMs,
            triggerId = triggerId,
            observedValue = observedValue,
        )
        return try {
            socketManager.emit(OBSERVATION_EVENT, payload)
            Log.i(TAG, "[emit] observation_id=$observationId trigger_id=$triggerId s3_key=$s3Key")
            true
        } catch (e: Exception) {
            Log.w(TAG, "[emit] socket emit failed", e)
            false
        }
    }

    /**
     * Build the JSON payload sent over the socket. Pure function — exposed
     * `internal` for unit tests so we can verify field set and types
     * without standing up a real socket.
     */
    internal fun buildObservationPayload(
        observationId: String,
        s3Key: String,
        cdnUrl: String,
        durationMs: Long,
        sizeBytes: Long,
        contentType: String,
        recordedAtMs: Long,
        triggerId: String,
        observedValue: Any?,
    ): JSONObject = JSONObject().apply {
        put("observation_id", observationId)
        put("s3_key", s3Key)
        put("cdn_url", cdnUrl)
        put("duration_ms", durationMs)
        put("size_bytes", sizeBytes)
        put("content_type", contentType)
        put("recorded_at", recordedAtMs)
        put("trigger_id", triggerId)
        // observed_value is Any?; coerce to JSON-safe primitive or string
        when (observedValue) {
            null -> put("observed_value", JSONObject.NULL)
            is Number -> put("observed_value", observedValue)
            is Boolean -> put("observed_value", observedValue)
            is String -> put("observed_value", observedValue)
            else -> put("observed_value", observedValue.toString())
        }
    }
}
