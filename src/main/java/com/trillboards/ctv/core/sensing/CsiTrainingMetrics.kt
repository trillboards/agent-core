package com.trillboards.ctv.core.sensing

/**
 * Snapshot of WiFlow paired-data collection progress for heartbeat telemetry.
 *
 * Embedded in [com.trillboards.ctv.core.models.HeartbeatPayload] so the server
 * can answer fleet-wide questions like:
 *   - "Which screens are actively collecting training data right now?"
 *   - "How close are we to the 9,000-sample target across the venue category?"
 *   - "Are uploads keeping up with collection?"
 *
 * All counters are monotonic since process start. They reset on restart;
 * persistent state lives in [PairedSampleStore].
 *
 * @property collectionActive Whether the collector is currently subscribed and accepting samples
 * @property samplesCollected Total samples written to the store this session
 * @property samplesTarget Target sample count for the current collection run
 * @property lastSampleMs Epoch millis of the most recent successful sample (0 if none)
 * @property uploadedCount Total samples uploaded to the backend this session
 */
data class CsiTrainingMetrics(
    val collectionActive: Boolean,
    val samplesCollected: Int,
    val samplesTarget: Int,
    val lastSampleMs: Long,
    val uploadedCount: Int
)
