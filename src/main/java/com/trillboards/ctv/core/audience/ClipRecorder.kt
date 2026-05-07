package com.trillboards.ctv.core.audience

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Phase 4 PR 3 — Edge clip recorder (DRY-RUN path).
 *
 * Records a short audio+video clip when a [ClipTriggerEvaluator] rule fires
 * and stages it on the local cache directory. The companion [ClipUploadManager]
 * then uploads to S3 staging via a presigned URL and emits a
 * `clipDryrunObservation` socket event so the API can write a
 * `observation_family='clip_dryrun'` row to `observation_stream`.
 *
 * ## Why a separate recorder (not CameraX VideoCapture)
 * `AudienceAnalyzer` already owns the camera lifecycle for face/pose analysis
 * via CameraX `ImageAnalysis` + `ImageCapture`. Adding `VideoCapture` to that
 * binding requires a `bindToLifecycle` re-bind, which interrupts the live
 * inference path during the rebind window. For a DRY-RUN path the priority
 * is reliability + isolation: capture happens off the hot inference path,
 * uses the platform [MediaRecorder] (independent camera handle inside its
 * own scope) and never touches the bound CameraX provider. PR 4 (real
 * Gemini call) can revisit and consolidate to a shared CameraX VideoCapture
 * once we know the failure rates of the dry-run path.
 *
 * ## Recording knobs (defaults match Gemini multimodal sweet-spot)
 * - 5-second duration (the default `capture_duration_seconds` from
 *   `clipTriggerSchema.js` is 30s, but PR 3 caps at 5s so a triggered clip
 *   never blocks a follow-up trigger for too long. PR 4 will widen.)
 * - 1280×720 H.264 video at 24fps, 1 Mbps bitrate
 * - 48kHz mono AAC audio at 96 kbps
 * - Output container: MP4
 *
 * ## Test seam
 * Construction takes a [recorderFactory] callable so unit tests can swap in a
 * fake `MediaRecorder` that records calls without touching real hardware.
 * The default factory uses the API-version-correct constructor.
 *
 * ## Concurrency
 * `recordClip` is guarded by [isRecording]; concurrent fires drop subsequent
 * requests until the in-flight one completes. The `aggregateAndEmit` path
 * is sequential per cycle so this is normally a non-issue, but safety
 * matters if PR 4 widens the trigger surface.
 *
 * @see AudienceSensingService.aggregateAndEmit For the call site
 * @see ClipUploadManager For the upload + observation-emit half
 * @see ClipTriggerEvaluator For the trigger evaluator that drives this
 */
class ClipRecorder(
    private val context: Context,
    private val outputDirOverride: File? = null,
    private val recorderFactory: (Context) -> MediaRecorder = ::defaultRecorderFactory,
    private val clock: () -> Long = { System.currentTimeMillis() },
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
) {

    companion object {
        private const val TAG = "ClipRecorder"

        /** Default cap on per-trigger record duration. PR 4 may widen via profile. */
        const val DEFAULT_DURATION_MS = 5_000L

        /** Hard cap so a misconfigured profile never holds the recorder open. */
        private const val MAX_DURATION_MS = 60_000L

        /** Min cap so we never produce <1s clips that Gemini will reject. */
        private const val MIN_DURATION_MS = 1_000L

        /** Output media MIME — matches s3StorageService allowed list. */
        const val CONTENT_TYPE = "video/mp4"

        /** Subdirectory under cache where staged clips live. */
        const val CLIPS_SUBDIR = "clips"

        @JvmStatic
        fun defaultRecorderFactory(ctx: Context): MediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(ctx)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }
    }

    private val isRecording = AtomicBoolean(false)

    /**
     * Record a clip in response to a trigger fire.
     *
     * The function is synchronous from the caller's perspective — it blocks
     * until the recording is finished and the file is closed. Run on
     * `Dispatchers.IO` (which the call site does via `sensingScope.launch`).
     *
     * @param observationId Deterministic ID from the caller (used as filename base).
     *                      Caller is responsible for uniqueness; this method does
     *                      NOT regenerate it on retry.
     * @param requestedDurationMs Desired clip duration; clamped to
     *                            [MIN_DURATION_MS]..[MAX_DURATION_MS].
     * @return [ClipRecording] on success, null if hardware busy / failed.
     */
    fun recordClip(
        observationId: String,
        requestedDurationMs: Long = DEFAULT_DURATION_MS,
    ): ClipRecording? {
        if (observationId.isBlank()) {
            Log.w(TAG, "[record] refused — observation_id required")
            return null
        }
        if (!isRecording.compareAndSet(false, true)) {
            Log.w(TAG, "[record] refused — recorder busy (already recording)")
            return null
        }

        val clampedDuration = requestedDurationMs.coerceIn(MIN_DURATION_MS, MAX_DURATION_MS)
        val outputDir = resolveOutputDir()
        val outputFile = File(outputDir, "$observationId.mp4")
        val startedAtMs = clock()

        val recorder = recorderFactory(context)
        val ok = try {
            // MediaRecorder ordering matters: setAudioSource → setVideoSource → setOutputFormat
            // → setAudioEncoder → setVideoEncoder → setOutputFile → prepare → start.
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
            recorder.setVideoSource(MediaRecorder.VideoSource.CAMERA)
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            recorder.setVideoSize(1280, 720)
            recorder.setVideoFrameRate(24)
            recorder.setVideoEncodingBitRate(1_000_000)
            recorder.setAudioSamplingRate(48_000)
            recorder.setAudioChannels(1)
            recorder.setAudioEncodingBitRate(96_000)
            recorder.setMaxDuration(clampedDuration.toInt())
            recorder.setOutputFile(outputFile.absolutePath)
            recorder.prepare()
            recorder.start()

            // Block on the duration window; MediaRecorder fires onInfo with
            // MAX_DURATION_REACHED but we don't use a Looper here so we sleep
            // to the duration and stop manually. A small grace ensures the
            // last frame and audio sample land in the moov box.
            sleeper(clampedDuration + 200L)
            true
        } catch (e: Exception) {
            Log.e(TAG, "[record] recording failed (observation_id=$observationId)", e)
            false
        } finally {
            try { recorder.stop() } catch (e: Exception) { Log.w(TAG, "[record] stop failed: ${e.message}") }
            try { recorder.reset() } catch (_: Exception) { }
            try { recorder.release() } catch (_: Exception) { }
            isRecording.set(false)
        }

        if (!ok || !outputFile.exists() || outputFile.length() == 0L) {
            // Empty file on failure — clean up so we don't leak partial bytes.
            try { outputFile.delete() } catch (_: Exception) { }
            return null
        }

        val recordedAtMs = startedAtMs
        val actualDurationMs = clock() - startedAtMs
        Log.i(TAG, "[record] ok observation_id=$observationId file=${outputFile.absolutePath} " +
            "size=${outputFile.length()}B duration=${actualDurationMs}ms")
        return ClipRecording(
            observationId = observationId,
            file = outputFile,
            durationMs = clampedDuration,
            sizeBytes = outputFile.length(),
            recordedAtMs = recordedAtMs,
            contentType = CONTENT_TYPE,
        )
    }

    /**
     * Resolve where staged clips live. Call site can override (used by tests).
     * Defaults to `<cacheDir>/clips/`, created on demand.
     */
    fun resolveOutputDir(): File {
        val dir = outputDirOverride ?: File(context.cacheDir, CLIPS_SUBDIR)
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /** Generate a fresh observation_id when the caller hasn't already minted one. */
    fun generateObservationId(triggerId: String): String {
        val tid = triggerId.takeIf { it.isNotBlank() } ?: "trigger"
        // Match the server's `obs_<16-hex>` namespace approximately —
        // we use a UUID v4 (8 hex chars) prefixed with `clipdr_` to make
        // collisions across PRs / paths obvious.
        val ts = clock().toString(36)
        val rand = UUID.randomUUID().toString().replace("-", "").substring(0, 8)
        return "clipdr_${tid}_${ts}_${rand}"
    }
}

/**
 * Return value of [ClipRecorder.recordClip].
 *
 * @property observationId Deterministic ID — same value the API receives
 *           via the `clipDryrunObservation` socket event payload.
 * @property file On-device staging file (must be deleted after upload).
 * @property durationMs Actual cap applied (after MIN/MAX clamping).
 * @property sizeBytes File size in bytes (>0 on success).
 * @property recordedAtMs Wall-clock millis when recording started.
 * @property contentType MIME of the staged file (always `video/mp4`).
 */
data class ClipRecording(
    val observationId: String,
    val file: File,
    val durationMs: Long,
    val sizeBytes: Long,
    val recordedAtMs: Long,
    val contentType: String,
)
