package com.trillboards.ctv.core.sensing

/**
 * Aggregated CSI measurement snapshot from an ESP32-S3 WiFi CSI sensing node.
 *
 * Produced by [CsiAggregator] from raw CSI frames received via UDP.
 * Included in heartbeat payloads for server-side audience analytics.
 *
 * @property nodeId ESP32 node identifier (0-255)
 * @property occupantCount Estimated number of occupants derived from subcarrier amplitude variance
 * @property motionScore Normalized motion intensity (0.0 = still, 1.0 = maximum motion)
 * @property signalQuality Frame delivery ratio (0.0 = all dropped, 1.0 = all received)
 * @property subcarrierCount Number of OFDM subcarriers reported by the ESP32
 * @property captureRateHz Effective frame capture rate over the aggregation window
 * @property avgRssiDbm Average RSSI in dBm across frames in the window
 * @property framesProcessed Total frames successfully parsed in the window
 * @property framesDropped Frames lost or failed to parse in the window
 * @property windowStartMs Epoch millis of the aggregation window start
 * @property windowEndMs Epoch millis of the aggregation window end
 * @property hardwareType Hardware identifier string (default "esp32-s3")
 */
data class CsiSnapshot(
    val nodeId: Int,
    val occupantCount: Int,
    val motionScore: Float,
    val signalQuality: Float,
    val subcarrierCount: Int,
    val captureRateHz: Float,
    val avgRssiDbm: Int,
    val framesProcessed: Int,
    val framesDropped: Int,
    val windowStartMs: Long,
    val windowEndMs: Long,
    val hardwareType: String = "esp32-s3"
)
