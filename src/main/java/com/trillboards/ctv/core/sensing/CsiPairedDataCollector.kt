package com.trillboards.ctv.core.sensing

import android.util.Log
import com.trillboards.ctv.core.net.ApiClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Phase 5 of the WiFi CSI integration: passively collect paired
 * (camera ground-truth pose, CSI 200 ms window) samples for training the
 * RuView WiFlow CSI-only pose estimator.
 *
 * ## How pairing works
 *
 * On every camera pose observation:
 *   1. Reject if `overallConfidence < minConfidence` (skip ambiguous frames)
 *   2. Pull all CSI frames received between `[poseTime - csiWindowMs, poseTime]`
 *   3. Reject if fewer than [framesPerWindow] frames are present (incomplete window)
 *   4. Reject if subcarrier counts disagree across the window
 *   5. Reject if the most recent CSI frame is more than [maxTimestampDriftMs]
 *      after the pose timestamp (clock skew between Android & ESP32)
 *   6. Otherwise pack the frame amplitudes into bytes and write a sample
 *
 * The collector exits cleanly once [samplesTarget] is reached. ~9,000 samples
 * (5 minutes of passive collection at one decisionable pose per second) is
 * the WiFlow paper's reported sweet-spot for 92.9% PCK@20.
 *
 * ## Thread model
 *
 * - [pushPose] (via subscription) is called on the producer thread (the
 *   AudienceAnalyzer pose worker). Pairing logic is synchronous and lock-free
 *   (atomic counters + thread-safe store).
 * - [uploadToBackend] is a suspend function that runs on whatever scope the
 *   caller provides — usually a periodic background flush.
 * - [startCollection] is a suspend function that owns the target-watch loop
 *   so the caller can `await` completion.
 *
 * ## Lifecycle
 *
 * ```kotlin
 * val collector = CsiPairedDataCollector(
 *     csiFrameSource = csiUdpListener,        // wraps agent-core-lite's listener
 *     poseProvider = mediaPipePoseAdapter,    // pushed by AudienceAnalyzer
 *     sampleStore = RoomPairedSampleStore(context),
 *     screenMongoId = currentScreenId,
 *     venueType = currentVenueType,
 *     cameraModelVersion = "mediapipe-blazepose-lite-v0.10.33"
 * )
 *
 * scope.launch { collector.startCollection(targetSampleCount = 9000) }
 * ```
 */
class CsiPairedDataCollector(
    private val csiFrameSource: CsiFrameSource,
    private val poseProvider: PoseKeypointProvider,
    private val sampleStore: PairedSampleStore,
    private val screenMongoId: String?,
    private val venueType: String?,
    private val cameraModelVersion: String?,
    private val minConfidence: Float = DEFAULT_MIN_CONFIDENCE,
    private val csiWindowMs: Long = DEFAULT_CSI_WINDOW_MS,
    private val maxTimestampDriftMs: Long = DEFAULT_MAX_DRIFT_MS,
    private val framesPerWindow: Int = DEFAULT_FRAMES_PER_WINDOW
) {

    companion object {
        private const val TAG = "CsiPairedCollector"

        /** WiFlow uses 200 ms windows of 20 frames at 100 Hz. */
        const val DEFAULT_CSI_WINDOW_MS = 200L
        const val DEFAULT_FRAMES_PER_WINDOW = 20

        /** Default 1 second drift budget — generous enough for unsynced clocks. */
        const val DEFAULT_MAX_DRIFT_MS = 1_000L

        /** Default minimum overall pose confidence to accept a sample. */
        const val DEFAULT_MIN_CONFIDENCE = 0.5f

        /** WiFlow paper's sweet-spot for 92.9% PCK@20. */
        const val DEFAULT_TARGET_SAMPLE_COUNT = 9_000

        /** Backend upload batch size — keeps each request under ~10 MB. */
        const val DEFAULT_UPLOAD_BATCH_SIZE = 500

        /** Default poll interval used by [startCollection] to check sample count. */
        private const val POLL_INTERVAL_MS = 250L
    }

    private val collectorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val samplesCollected = AtomicInteger(0)
    private val skippedLowConfidence = AtomicInteger(0)
    private val skippedTimestampMismatch = AtomicInteger(0)
    private val uploadedCount = AtomicInteger(0)
    private val lastSampleMs = AtomicLong(0L)

    private val collectionActive = AtomicBoolean(false)
    @Volatile
    private var samplesTarget: Int = DEFAULT_TARGET_SAMPLE_COUNT

    private var poseSubscription: AutoCloseable? = null

    /**
     * Begin a passive collection run. Subscribes to the pose provider, then
     * suspends until either:
     *   - [samplesCollected] reaches [targetSampleCount], or
     *   - the surrounding scope is cancelled
     *
     * Idempotent — calling startCollection while already running is a no-op.
     */
    suspend fun startCollection(targetSampleCount: Int = DEFAULT_TARGET_SAMPLE_COUNT) {
        if (!collectionActive.compareAndSet(false, true)) {
            Log.d(TAG, "startCollection already active")
            return
        }
        samplesTarget = targetSampleCount.coerceAtLeast(1)
        Log.i(TAG, "starting paired collection target=$samplesTarget")

        poseSubscription = poseProvider.subscribePose { obs ->
            // Synchronous handling — keeps the producer thread predictable.
            // The pairing work is bounded (a few list filters + one DB insert).
            handlePoseSync(obs)
        }

        try {
            while (collectionActive.get() && samplesCollected.get() < samplesTarget) {
                delay(POLL_INTERVAL_MS)
            }
            Log.i(
                TAG,
                "collection finished collected=${samplesCollected.get()} " +
                    "skippedLowConfidence=${skippedLowConfidence.get()} " +
                    "skippedTimestampMismatch=${skippedTimestampMismatch.get()}"
            )
        } finally {
            stopCollection()
        }
    }

    /**
     * Subscribe the collector to the pose provider WITHOUT awaiting completion.
     *
     * Used by callers that want to drive the lifecycle externally (tests,
     * services that need to start/stop on demand). The returned [AutoCloseable]
     * removes the subscription.
     */
    fun startSubscription(): AutoCloseable {
        if (!collectionActive.compareAndSet(false, true)) {
            // Already subscribed — return a no-op closer that doesn't affect state.
            return AutoCloseable { }
        }
        val subscription = poseProvider.subscribePose { obs -> handlePoseSync(obs) }
        poseSubscription = subscription
        return AutoCloseable {
            subscription.close()
            collectionActive.set(false)
        }
    }

    /**
     * Stop the collection loop and unsubscribe from the pose provider.
     */
    fun stopCollection() {
        collectionActive.set(false)
        poseSubscription?.close()
        poseSubscription = null
    }

    /**
     * Update the in-flight target sample count. Used by callers (and tests)
     * that want to short-circuit a long collection run.
     */
    fun setTargetSampleCount(target: Int) {
        samplesTarget = target.coerceAtLeast(1)
    }

    /**
     * Snapshot of progress counters for heartbeat telemetry.
     */
    fun getStats(): CollectionStats {
        return CollectionStats(
            samplesCollected = samplesCollected.get(),
            samplesTarget = samplesTarget,
            skippedLowConfidence = skippedLowConfidence.get(),
            skippedTimestampMismatch = skippedTimestampMismatch.get(),
            lastSampleMs = lastSampleMs.get(),
            uploadedCount = uploadedCount.get()
        )
    }

    /**
     * Snapshot in the heartbeat-friendly [CsiTrainingMetrics] shape.
     */
    fun getTrainingMetrics(): CsiTrainingMetrics {
        return CsiTrainingMetrics(
            collectionActive = collectionActive.get(),
            samplesCollected = samplesCollected.get(),
            samplesTarget = samplesTarget,
            lastSampleMs = lastSampleMs.get(),
            uploadedCount = uploadedCount.get()
        )
    }

    /**
     * Upload pending samples to the training backend in batches.
     *
     * Each batch is sent as a single POST. Successful batches are marked
     * uploaded in the local store. Failed batches are left for the next call.
     *
     * @param apiClient The agent-core API client (for base URL + auth headers)
     * @param batchSize Max samples per request (defaults to [DEFAULT_UPLOAD_BATCH_SIZE])
     * @return The number of samples successfully uploaded across all batches
     */
    suspend fun uploadToBackend(
        apiClient: ApiClient,
        batchSize: Int = DEFAULT_UPLOAD_BATCH_SIZE
    ): Int {
        var totalUploaded = 0
        while (true) {
            val batch = sampleStore.getUnuploadedBatch(batchSize)
            if (batch.isEmpty()) break

            val ok = postBatch(apiClient, batch)
            if (!ok) {
                Log.w(TAG, "uploadToBackend batch failed, leaving ${batch.size} unuploaded")
                break
            }

            sampleStore.markUploaded(batch.map { it.sampleId })
            uploadedCount.addAndGet(batch.size)
            totalUploaded += batch.size

            if (batch.size < batchSize) break  // No more pending after this batch
        }
        return totalUploaded
    }

    /**
     * Drop samples that have been uploaded. Pending uploads are preserved.
     */
    fun clearUploaded() {
        sampleStore.clearUploaded()
    }

    // ============================================================
    // Pairing logic — kept package-private for tests
    // ============================================================

    /**
     * Test hook: simulate a pose observation arriving without going through
     * the subscription path. Identical to the subscriber callback.
     */
    internal fun testOnPoseObservation(observation: PoseObservation) {
        handlePoseSync(observation)
    }

    /**
     * Test hook: mark uploaded by ID through the same path that
     * [uploadToBackend] uses, so collector counters stay consistent.
     */
    internal fun testMarkUploaded(ids: List<String>) {
        if (ids.isEmpty()) return
        sampleStore.markUploaded(ids)
        uploadedCount.addAndGet(ids.size)
    }

    private fun handlePoseSync(observation: PoseObservation) {
        // Honour the target cap — even if more poses arrive after the cap was hit
        if (samplesCollected.get() >= samplesTarget) return

        val confidence = computeOverallConfidence(observation)
        if (confidence < minConfidence) {
            skippedLowConfidence.incrementAndGet()
            return
        }

        val poseTime = observation.timestampMs
        val windowStart = poseTime - csiWindowMs
        val recent = csiFrameSource.getRecentFrames(windowStart)

        // Filter to frames whose receivedAtMs is in [windowStart, poseTime + drift]
        val windowFrames = recent
            .filter { it.receivedAtMs in windowStart..(poseTime + maxTimestampDriftMs) }
            .sortedBy { it.receivedAtMs }

        if (windowFrames.size < framesPerWindow) {
            skippedTimestampMismatch.incrementAndGet()
            return
        }

        // Truncate to exactly framesPerWindow most-recent frames in the window
        val pickedFrames = windowFrames.takeLast(framesPerWindow)

        // Subcarrier count must be consistent across the window
        val firstSubcarriers = pickedFrames[0].nSubcarriers
        if (firstSubcarriers <= 0 || pickedFrames.any { it.nSubcarriers != firstSubcarriers }) {
            skippedTimestampMismatch.incrementAndGet()
            return
        }

        // Drift check — newest frame should not be ahead of pose by more than drift budget
        val newest = pickedFrames.last().receivedAtMs
        if (newest > poseTime + maxTimestampDriftMs) {
            skippedTimestampMismatch.incrementAndGet()
            return
        }
        if (poseTime - pickedFrames.first().receivedAtMs > csiWindowMs + maxTimestampDriftMs) {
            // Window is too old — frame buffer hasn't been updated since long ago
            skippedTimestampMismatch.incrementAndGet()
            return
        }

        // Pack amplitudes into a single FloatArray for byte serialization.
        // Layout: [frame0_sc0, frame0_sc1, ..., frame19_scN-1] (frame-major)
        val totalFloats = framesPerWindow * firstSubcarriers
        val packed = FloatArray(totalFloats)
        var dst = 0
        for (frame in pickedFrames) {
            // Defensive: amplitudes may include multiple antennas — take the first
            // antenna's worth of subcarriers, matching what RuView WiFlow expects.
            val src = frame.amplitudes
            val copyLen = minOf(firstSubcarriers, src.size)
            System.arraycopy(src, 0, packed, dst, copyLen)
            // Zero-fill the rest of the row if antennas mismatch
            if (copyLen < firstSubcarriers) {
                for (i in copyLen until firstSubcarriers) packed[dst + i] = 0f
            }
            dst += firstSubcarriers
        }

        val sample = PairedTrainingSample(
            sampleId = UUID.randomUUID().toString(),
            csiWindowBytes = PairedTrainingSample.serializeFloats(packed),
            subcarrierCount = firstSubcarriers,
            keypoints = observation.keypoints,
            keypointConfidences = observation.confidences,
            overallConfidence = confidence,
            numCameraFrames = pickedFrames.size,
            windowStartMs = pickedFrames.first().receivedAtMs,
            windowEndMs = pickedFrames.last().receivedAtMs,
            screenMongoId = screenMongoId,
            venueType = venueType,
            cameraModelVersion = cameraModelVersion ?: observation.modelVersion
        )

        if (sampleStore.insert(sample)) {
            samplesCollected.incrementAndGet()
            lastSampleMs.set(observation.timestampMs)
        }
    }

    private fun computeOverallConfidence(observation: PoseObservation): Float {
        if (observation.confidences.isEmpty()) return 0f
        var sum = 0f
        for (c in observation.confidences) sum += c
        return sum / observation.confidences.size
    }

    private suspend fun postBatch(
        apiClient: ApiClient,
        batch: List<PairedTrainingSample>
    ): Boolean {
        return try {
            apiClient.sendPairedTrainingSamples(buildBatchJson(batch))
        } catch (e: Exception) {
            Log.w(TAG, "postBatch failed: ${e.message}")
            false
        }
    }

    private fun buildBatchJson(batch: List<PairedTrainingSample>): JSONObject {
        val samplesArr = JSONArray()
        for (s in batch) {
            val sampleJson = JSONObject().apply {
                put("sampleId", s.sampleId)
                put(
                    "csiWindowBase64",
                    android.util.Base64.encodeToString(s.csiWindowBytes, android.util.Base64.NO_WRAP)
                )
                put("subcarrierCount", s.subcarrierCount)
                put("keypoints", JSONArray().apply { s.keypoints.forEach { put(it.toDouble()) } })
                put("keypointConfidences", JSONArray().apply { s.keypointConfidences.forEach { put(it.toDouble()) } })
                put("overallConfidence", s.overallConfidence.toDouble())
                put("numCameraFrames", s.numCameraFrames)
                put("windowStartMs", s.windowStartMs)
                put("windowEndMs", s.windowEndMs)
                s.venueType?.let { put("venueType", it) }
                s.cameraModelVersion?.let { put("cameraModelVersion", it) }
            }
            samplesArr.put(sampleJson)
        }
        return JSONObject().apply {
            screenMongoId?.let { put("screenId", it) }
            put("samples", samplesArr)
        }
    }

    /**
     * Cancel the collector's coroutine scope. Call from the host service's
     * `onDestroy` to release any in-flight work.
     */
    fun release() {
        stopCollection()
        try {
            collectorScope.cancel()
        } catch (_: Exception) { /* ignore */ }
    }

    /**
     * Per-call snapshot of collection counters.
     */
    data class CollectionStats(
        val samplesCollected: Int,
        val samplesTarget: Int,
        val skippedLowConfidence: Int,
        val skippedTimestampMismatch: Int,
        val lastSampleMs: Long,
        val uploadedCount: Int
    )
}
