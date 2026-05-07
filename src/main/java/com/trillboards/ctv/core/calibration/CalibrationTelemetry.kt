package com.trillboards.ctv.core.calibration

import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import org.json.JSONObject

/**
 * Collects calibration state telemetry for heartbeat reporting.
 *
 * Packages the current effective configuration — thresholds, weights, and their source
 * (default / server-pushed / auto-calibrated) — into a JSON object that the server can
 * use for fleet-wide observability and diagnosis.
 *
 * The fleet copilot can then answer queries like:
 *   - "show me devices where gaze_yaw_threshold is still at default"
 *   - "which devices have low signal agreement rates"
 *   - "are lux calibrations converging across the bar venue type"
 *
 * ## Config Source Tracking
 *
 * Each key threshold is annotated with its source:
 *   - **"default"** — hardcoded default value, never overridden
 *   - **"server"** — pushed by the server via config.push ("sensing_config_update")
 *   - **"calibrated"** — auto-calibrated on-device (DeploymentCalibration, LuxCalibrator)
 *
 * ## Signal Agreement (Phase 3 Integration)
 *
 * When a ConfidenceCalibrator instance is registered, the telemetry includes recent
 * signal agreement statistics computed from the last N calibration windows:
 *   - Overall agreement rate
 *   - Per-signal-group agreement (face count, emotion, engagement, presence)
 *   - VLM vs TFLite face count agreement
 *
 * These are rolling averages — not single-window snapshots — to reduce noise.
 */
object CalibrationTelemetry {

    private const val TAG = "CalibrationTelemetry"

    // Rolling window for signal agreement stats (last N calibration results)
    private const val AGREEMENT_WINDOW_SIZE = 30

    // Thread-safe ring buffer for recent CalibratedConfidence results
    private val recentCalibrations = ArrayDeque<CalibratedConfidence>(AGREEMENT_WINDOW_SIZE + 1)
    private val lock = Any()

    // Optional: track recent VLM vs TFLite face count agreement
    // Each entry is a pair (tfliteFaceCount, vlmFaceCount) from the same window
    private val recentFaceAgreementPairs = ArrayDeque<Pair<Int, Int>>(AGREEMENT_WINDOW_SIZE + 1)

    // Optional: LuxCalibrator reference for lux calibration metadata
    @Volatile
    private var luxCalibrator: LuxCalibrator? = null

    // Optional: DeploymentCalibration derived thresholds
    @Volatile
    private var derivedThresholds: DerivedThresholds? = null

    /**
     * Record a calibration result from ConfidenceCalibrator for rolling averages.
     * Called at the end of each aggregation window when calibration is computed.
     */
    fun recordCalibration(result: CalibratedConfidence) {
        synchronized(lock) {
            if (recentCalibrations.size >= AGREEMENT_WINDOW_SIZE) {
                recentCalibrations.removeFirst()
            }
            recentCalibrations.addLast(result)
        }
    }

    /**
     * Record a VLM vs TFLite face count comparison for agreement tracking.
     * Called when both TFLite and VLM face counts are available in the same window.
     *
     * @param tfliteFaceCount Face count from BlazeFace/EfficientDet TFLite models
     * @param vlmFaceCount Face count extracted from VLM structured output
     */
    fun recordFaceAgreement(tfliteFaceCount: Int, vlmFaceCount: Int) {
        synchronized(lock) {
            if (recentFaceAgreementPairs.size >= AGREEMENT_WINDOW_SIZE) {
                recentFaceAgreementPairs.removeFirst()
            }
            recentFaceAgreementPairs.addLast(Pair(tfliteFaceCount, vlmFaceCount))
        }
    }

    /**
     * Register the LuxCalibrator instance for lux calibration metadata in telemetry.
     */
    fun setLuxCalibrator(calibrator: LuxCalibrator) {
        luxCalibrator = calibrator
    }

    /**
     * Register deployment-derived thresholds for telemetry reporting.
     */
    fun setDerivedThresholds(thresholds: DerivedThresholds) {
        derivedThresholds = thresholds
    }

    /**
     * Build the complete calibration telemetry payload for heartbeat metadata.
     *
     * @return JSONObject containing effective thresholds, config sources, signal agreement,
     *         and calibration subsystem status
     */
    fun buildTelemetry(): JSONObject {
        val config = SensingConfig.get()

        return JSONObject().apply {
            // 1. Effective thresholds — the values actually in use right now
            put("effective_thresholds", buildEffectiveThresholds(config))

            // 2. Config source tracking — which values are default vs server vs calibrated
            put("config_source", buildConfigSources())

            // 3. Signal agreement statistics (from ConfidenceCalibrator rolling window)
            put("signal_agreement", buildSignalAgreement())

            // 4. Calibration subsystem status
            put("calibration_status", buildCalibrationStatus())

            // 5. Has server config been received at all
            put("has_server_config", SensingConfig.hasServerConfig())
        }
    }

    /**
     * Build effective threshold values currently in use.
     * These are the actual numbers processors are using for gaze/detection/engagement/etc.
     */
    private fun buildEffectiveThresholds(config: SensingConfig.Config): JSONObject {
        return JSONObject().apply {
            // Gaze thresholds — critical for attention scoring
            put("gaze_yaw_threshold", config.gaze.headYawThreshold)
            put("gaze_pitch_threshold", config.gaze.headPitchThreshold)
            put("gaze_min_face_confidence", config.gaze.minFaceConfidence)

            // Detection thresholds — controls person/face filtering
            put("person_confidence_threshold", config.detection.personConfidenceThreshold)
            put("min_face_size", config.detection.minFaceSize)
            put("iou_threshold", config.detection.iouThreshold)

            // Engagement weights — how pose/emotion/gaze combine
            put("engagement_weights", JSONObject().apply {
                put("pose", config.engagement.poseWeight)
                put("emotion", config.engagement.emotionWeight)
                put("gaze", config.engagement.gazeWeight)
            })

            // Body thresholds — facing/lean/movement
            put("body_facing_angle_threshold", config.body.facingAngleThreshold)

            // Viewability thresholds — lux breakpoints
            put("viewability", JSONObject().apply {
                put("dark_lux", config.viewability.darkLuxThreshold)
                put("dim_lux", config.viewability.dimLuxThreshold)
                put("good_lux", config.viewability.goodLuxThreshold)
                put("bright_lux", config.viewability.brightLuxThreshold)
            })

            // Emotion thresholds
            put("emotion_min_confidence", config.emotion.minConfidence)

            // Attention divisors
            put("attention", JSONObject().apply {
                put("yaw_divisor", config.attention.yawDivisor)
                put("pitch_divisor", config.attention.pitchDivisor)
            })
        }
    }

    /**
     * Build config source tracking — for each key threshold, report whether it's
     * at its default value, was server-pushed, or was auto-calibrated.
     */
    private fun buildConfigSources(): JSONObject {
        return JSONObject().apply {
            // Gaze thresholds (may be deployment-calibrated or server-pushed)
            put("gaze_yaw", SensingConfig.getFieldSource("gaze", "headYawThreshold"))
            put("gaze_pitch", SensingConfig.getFieldSource("gaze", "headPitchThreshold"))
            put("gaze_min_face_confidence", SensingConfig.getFieldSource("gaze", "minFaceConfidence"))

            // Detection thresholds
            put("person_confidence", SensingConfig.getFieldSource("detection", "personConfidenceThreshold"))

            // Engagement weights
            put("engagement_pose_weight", SensingConfig.getFieldSource("engagement", "poseWeight"))
            put("engagement_emotion_weight", SensingConfig.getFieldSource("engagement", "emotionWeight"))
            put("engagement_gaze_weight", SensingConfig.getFieldSource("engagement", "gazeWeight"))

            // Body thresholds (may be deployment-calibrated)
            put("body_facing_angle", SensingConfig.getFieldSource("body", "facingAngleThreshold"))

            // Viewability thresholds (may be lux-calibrated)
            put("viewability_dark_lux", SensingConfig.getFieldSource("viewability", "darkLuxThreshold"))
            put("viewability_dim_lux", SensingConfig.getFieldSource("viewability", "dimLuxThreshold"))
            put("viewability_good_lux", SensingConfig.getFieldSource("viewability", "goodLuxThreshold"))
            put("viewability_bright_lux", SensingConfig.getFieldSource("viewability", "brightLuxThreshold"))

            // Attention divisors (may be deployment-calibrated)
            put("attention_yaw_divisor", SensingConfig.getFieldSource("attention", "yawDivisor"))
            put("attention_pitch_divisor", SensingConfig.getFieldSource("attention", "pitchDivisor"))
        }
    }

    /**
     * Compute rolling signal agreement statistics from recent calibration results.
     * Returns averages over the last AGREEMENT_WINDOW_SIZE calibration windows.
     */
    private fun buildSignalAgreement(): JSONObject {
        return JSONObject().apply {
            synchronized(lock) {
                val calibrations = recentCalibrations.toList()
                if (calibrations.isEmpty()) {
                    put("sample_count", 0)
                    put("overall_agreement_rate", JSONObject.NULL)
                    put("face_count_agreement", JSONObject.NULL)
                    put("emotion_agreement", JSONObject.NULL)
                    put("engagement_agreement", JSONObject.NULL)
                    put("presence_agreement", JSONObject.NULL)
                    put("avg_signal_count", JSONObject.NULL)
                    put("vlm_tflite_face_agreement", JSONObject.NULL)
                    return@apply
                }

                put("sample_count", calibrations.size)
                put("overall_agreement_rate", roundTo3(calibrations.map { it.overall }.average()))
                put("face_count_agreement", roundTo3(calibrations.map { it.faceCountConfidence }.average()))
                put("emotion_agreement", roundTo3(calibrations.map { it.emotionConfidence }.average()))
                put("engagement_agreement", roundTo3(calibrations.map { it.engagementConfidence }.average()))
                put("presence_agreement", roundTo3(calibrations.map { it.presenceConfidence }.average()))
                put("avg_signal_count", roundTo3(calibrations.map { it.signalCount.toDouble() }.average()))

                // VLM vs TFLite face agreement — fraction of windows where they match within 1
                val facePairs = recentFaceAgreementPairs.toList()
                if (facePairs.isEmpty()) {
                    put("vlm_tflite_face_agreement", JSONObject.NULL)
                } else {
                    val matchCount = facePairs.count {
                        kotlin.math.abs(it.first - it.second) <= 1
                    }
                    put("vlm_tflite_face_agreement", roundTo3(matchCount.toDouble() / facePairs.size))
                    put("vlm_tflite_face_pairs_count", facePairs.size)
                }
            }
        }
    }

    /**
     * Build calibration subsystem status — state of each calibrator.
     */
    private fun buildCalibrationStatus(): JSONObject {
        return JSONObject().apply {
            // Lux calibration status
            val lux = luxCalibrator
            if (lux != null) {
                put("lux", JSONObject().apply {
                    put("state", lux.getState().name)
                    put("sample_count", lux.getSampleCount())
                    put("elapsed_ms", lux.getElapsedMs())
                    lux.getCalibrationResult()?.let { result ->
                        put("median_lux", result.medianLux)
                        put("calibrated_at_ms", result.calibratedAtMs)
                    }
                })
            } else {
                put("lux", JSONObject.NULL)
            }

            // Deployment calibration status
            val deployment = derivedThresholds
            if (deployment != null) {
                put("deployment", JSONObject().apply {
                    put("active", true)
                    put("yaw_threshold_deg", deployment.yawThresholdDeg)
                    put("pitch_threshold_deg", deployment.pitchThresholdDeg)
                    put("facing_angle_threshold_deg", deployment.facingAngleThresholdDeg)
                    put("screen_width_cm", deployment.sourceConfig.screenWidthCm)
                    put("viewing_distance_cm", deployment.sourceConfig.viewingDistanceCm)
                })
            } else {
                put("deployment", JSONObject().apply {
                    put("active", false)
                })
            }

            // Confidence calibrator — just report whether we have data
            synchronized(lock) {
                put("confidence_calibrator", JSONObject().apply {
                    put("active", recentCalibrations.isNotEmpty())
                    put("windows_buffered", recentCalibrations.size)
                })
            }

            // Server-pushed field counts per section
            val pushed = SensingConfig.getServerPushedFields()
            val calibrated = SensingConfig.getAutoCalibratedFields()
            put("server_pushed_sections", pushed.size)
            put("server_pushed_field_count", pushed.values.sumOf { it.size })
            put("auto_calibrated_sections", calibrated.size)
            put("auto_calibrated_field_count", calibrated.values.sumOf { it.size })
        }
    }

    /**
     * Round a double to 3 decimal places for compact JSON output.
     */
    private fun roundTo3(value: Double): Double {
        return Math.round(value * 1000.0) / 1000.0
    }
}
