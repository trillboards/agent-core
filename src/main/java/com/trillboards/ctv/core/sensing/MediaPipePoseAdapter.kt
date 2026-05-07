package com.trillboards.ctv.core.sensing

import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicReference

/**
 * Bridge between the existing MediaPipe Pose Landmarker pipeline (33 BlazePose
 * landmarks) and the [PoseKeypointProvider] interface used by the WiFlow paired
 * training collector (17 COCO keypoints).
 *
 * ## Why an adapter?
 *
 * 1. The on-device pose model in agent-core is BlazePose Lite (33 landmarks).
 *    The CSI training pipeline (and most pose-from-CSI papers) uses the COCO
 *    17-keypoint convention. The adapter does that mapping in pure Kotlin.
 *
 * 2. The collector should not own pose inference — that belongs to
 *    [com.trillboards.ctv.core.audience.AudienceAnalyzer], which already runs
 *    MediaPipe Pose at the right cadence. The adapter is a passive holder that
 *    AudienceAnalyzer pushes into; the collector subscribes to the holder.
 *
 * ## Wiring (production)
 *
 * AudienceAnalyzer (or any pose producer) calls [pushPose] after each MediaPipe
 * inference. The collector subscribes once at startup. If the pose pipeline is
 * not active on a given device, [getLatestPose] returns null and the collector
 * naturally collects nothing.
 *
 * ## Thread safety
 *
 * - [pushPose] and [getLatestPose] use [AtomicReference] for lock-free access
 * - subscribers list is a [CopyOnWriteArrayList] — safe for concurrent
 *   add/remove during iteration
 * - handlers run synchronously on the producer thread; keep them short
 */
class MediaPipePoseAdapter : PoseKeypointProvider {

    private val latest = AtomicReference<PoseObservation?>(null)
    private val subscribers = CopyOnWriteArrayList<(PoseObservation) -> Unit>()

    /**
     * Publish a pose observation to the latest slot AND notify subscribers.
     *
     * Producers (e.g. AudienceAnalyzer) call this after MediaPipe inference.
     */
    fun pushPose(observation: PoseObservation) {
        latest.set(observation)
        for (handler in subscribers) {
            try {
                handler(observation)
            } catch (e: Exception) {
                // Subscriber errors must not break the producer pipeline.
                // The collector handles its own errors; this is a defensive guard.
            }
        }
    }

    override fun getLatestPose(): PoseObservation? = latest.get()

    override fun subscribePose(handler: (PoseObservation) -> Unit): AutoCloseable {
        subscribers.add(handler)
        return AutoCloseable { subscribers.remove(handler) }
    }

    companion object {
        // ===== BlazePose 33-landmark indices =====
        private const val BP_NOSE = 0
        // BlazePose has 3 landmarks per eye (inner / center / outer); use the center
        private const val BP_LEFT_EYE = 2
        private const val BP_RIGHT_EYE = 5
        private const val BP_LEFT_EAR = 7
        private const val BP_RIGHT_EAR = 8
        private const val BP_LEFT_SHOULDER = 11
        private const val BP_RIGHT_SHOULDER = 12
        private const val BP_LEFT_ELBOW = 13
        private const val BP_RIGHT_ELBOW = 14
        private const val BP_LEFT_WRIST = 15
        private const val BP_RIGHT_WRIST = 16
        private const val BP_LEFT_HIP = 23
        private const val BP_RIGHT_HIP = 24
        private const val BP_LEFT_KNEE = 25
        private const val BP_RIGHT_KNEE = 26
        private const val BP_LEFT_ANKLE = 27
        private const val BP_RIGHT_ANKLE = 28

        /**
         * Mapping from COCO joint index → BlazePose landmark index.
         *
         * COCO joint order:
         *   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
         *   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
         *   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
         *   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
         */
        private val COCO_TO_BLAZEPOSE = intArrayOf(
            BP_NOSE,
            BP_LEFT_EYE,
            BP_RIGHT_EYE,
            BP_LEFT_EAR,
            BP_RIGHT_EAR,
            BP_LEFT_SHOULDER,
            BP_RIGHT_SHOULDER,
            BP_LEFT_ELBOW,
            BP_RIGHT_ELBOW,
            BP_LEFT_WRIST,
            BP_RIGHT_WRIST,
            BP_LEFT_HIP,
            BP_RIGHT_HIP,
            BP_LEFT_KNEE,
            BP_RIGHT_KNEE,
            BP_LEFT_ANKLE,
            BP_RIGHT_ANKLE
        )

        /**
         * Convert MediaPipe Pose Landmarker output (33 landmarks × 3 floats:
         * x, y, z) into the COCO 17-keypoint format used by WiFlow training.
         *
         * @param blazePoseLandmarks Flattened 33×3 = 99 floats.
         *        Layout per landmark: [x, y, z]. The z (depth) channel is dropped
         *        because the COCO convention is 2D (x, y) and the WiFlow ground-truth
         *        format uses 2D keypoints.
         * @param visibilities 33 visibility values from the MediaPipe LandmarkResult,
         *        used as per-joint confidence in the COCO output.
         * @return A pair of (34-float COCO keypoints [x, y per joint], 17-float COCO confidences).
         * @throws IllegalArgumentException if input array sizes are wrong.
         */
        fun mapBlazePoseToCoco(
            blazePoseLandmarks: FloatArray,
            visibilities: FloatArray
        ): Pair<FloatArray, FloatArray> {
            require(blazePoseLandmarks.size == 33 * 3) {
                "BlazePose landmarks must be 33 × 3 = 99 floats, got ${blazePoseLandmarks.size}"
            }
            require(visibilities.size == 33) {
                "BlazePose visibilities must be 33 floats, got ${visibilities.size}"
            }

            val keypoints = FloatArray(17 * 2)
            val confidences = FloatArray(17)
            for (cocoIdx in 0 until 17) {
                val bpIdx = COCO_TO_BLAZEPOSE[cocoIdx]
                val xOffset = bpIdx * 3
                keypoints[cocoIdx * 2] = blazePoseLandmarks[xOffset]
                keypoints[cocoIdx * 2 + 1] = blazePoseLandmarks[xOffset + 1]
                confidences[cocoIdx] = visibilities[bpIdx]
            }
            return Pair(keypoints, confidences)
        }
    }
}
