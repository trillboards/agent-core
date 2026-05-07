package com.trillboards.ctv.core.sensing

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One paired training sample for the WiFlow CSI-pose model.
 *
 * Captures a 20-frame CSI window (200 ms @ 100 Hz) alongside the camera's
 * ground-truth pose estimate from MediaPipe Pose Landmarker (mapped to the
 * 17 COCO keypoint convention used by MoveNet/RuView WiFlow).
 *
 * Once 9,000 of these samples are collected from a tablet-agent fleet, the
 * server can train a CSI-only pose estimator that achieves ~92.9% PCK@20.
 * After training, the camera can be removed from the deployment — the CSI
 * model produces camera-level keypoint accuracy from passive WiFi alone.
 *
 * @property sampleId UUID identifier (server-side dedup key)
 * @property csiWindowBytes 20 frames × subcarrierCount × 4 bytes (little-endian floats)
 * @property subcarrierCount Number of OFDM subcarriers per frame (56 or 128)
 * @property keypoints 34 floats — 17 COCO joints × (x, y), normalized [0, 1]
 * @property keypointConfidences 17 per-joint confidences in [0, 1]
 * @property overallConfidence Weighted-average confidence used for filtering
 * @property numCameraFrames How many camera frames produced the pose (always 1 for snapshot)
 * @property windowStartMs Epoch millis of the first CSI frame in the window
 * @property windowEndMs Epoch millis of the last CSI frame in the window
 * @property screenMongoId Trillboards screen ID (Mongo ObjectId string), or null if unbound
 * @property venueType Venue category for downstream training stratification
 * @property cameraModelVersion The pose model version that produced the keypoints
 */
data class PairedTrainingSample(
    val sampleId: String,
    val csiWindowBytes: ByteArray,
    val subcarrierCount: Int,
    val keypoints: FloatArray,
    val keypointConfidences: FloatArray,
    val overallConfidence: Float,
    val numCameraFrames: Int,
    val windowStartMs: Long,
    val windowEndMs: Long,
    val screenMongoId: String?,
    val venueType: String?,
    val cameraModelVersion: String?
) {
    // sampleId is a UUID — equality / hashing on identity only
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PairedTrainingSample) return false
        return sampleId == other.sampleId
    }

    override fun hashCode(): Int = sampleId.hashCode()

    companion object {
        /**
         * Pack a FloatArray into little-endian bytes (4 bytes per float).
         *
         * Little-endian matches the existing ADR-018 CSI wire format and the
         * Java ByteBuffer convention used elsewhere in the sensing pipeline.
         */
        fun serializeFloats(values: FloatArray): ByteArray {
            val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (v in values) buf.putFloat(v)
            return buf.array()
        }

        /**
         * Unpack little-endian bytes into a FloatArray.
         *
         * @throws IllegalArgumentException if the byte length is not a multiple of 4
         */
        fun deserializeFloats(bytes: ByteArray): FloatArray {
            require(bytes.size % 4 == 0) {
                "byte buffer length ${bytes.size} is not a multiple of 4 (cannot decode floats)"
            }
            val out = FloatArray(bytes.size / 4)
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (i in out.indices) out[i] = buf.float
            return out
        }
    }
}
