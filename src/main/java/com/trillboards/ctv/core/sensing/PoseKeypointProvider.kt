package com.trillboards.ctv.core.sensing

/**
 * One pose observation produced by an on-device camera pose model.
 *
 * Keypoints follow the 17-joint COCO convention (the format MoveNet,
 * RuView WiFlow, and most CSI pose papers use):
 *
 * ```
 *  0 nose            5 left_shoulder   11 left_hip      14 right_knee
 *  1 left_eye        6 right_shoulder  12 right_hip     15 left_ankle
 *  2 right_eye       7 left_elbow      13 left_knee     16 right_ankle
 *  3 left_ear        8 right_elbow
 *  4 right_ear       9 left_wrist
 *                   10 right_wrist
 * ```
 *
 * @property keypoints 34 floats — 17 joints × (x, y), normalized [0, 1]
 * @property confidences 17 per-joint confidences in [0, 1]
 * @property timestampMs Epoch millis when the source frame was captured
 * @property modelVersion The pose model identifier (e.g. "mediapipe-blazepose-lite-v0.10.33")
 */
data class PoseObservation(
    val keypoints: FloatArray,
    val confidences: FloatArray,
    val timestampMs: Long,
    val modelVersion: String
) {
    init {
        require(keypoints.size == 34) {
            "PoseObservation requires 34 keypoint floats (17 joints × x,y), got ${keypoints.size}"
        }
        require(confidences.size == 17) {
            "PoseObservation requires 17 confidences (one per joint), got ${confidences.size}"
        }
    }
}

/**
 * Pull/push abstraction over an on-device pose source.
 *
 * Implementations decide how poses get pushed in (e.g. [MediaPipePoseAdapter]
 * exposes `pushPose` for AudienceAnalyzer to call from its inference loop).
 * Consumers (like [CsiPairedDataCollector]) only see the read-side contract.
 */
interface PoseKeypointProvider {
    /**
     * The most recent pose observation, or null if none has been produced yet.
     */
    fun getLatestPose(): PoseObservation?

    /**
     * Subscribe to subsequent pose observations.
     *
     * @return [AutoCloseable] — call `close()` to stop receiving notifications.
     *         The handler runs on the same thread that called `pushPose` —
     *         keep work short or hand off to a worker.
     */
    fun subscribePose(handler: (PoseObservation) -> Unit): AutoCloseable
}
