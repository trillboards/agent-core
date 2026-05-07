package com.trillboards.ctv.core.inference

/**
 * Standardized output from any [InferenceProcessor].
 *
 * Existing processors return wildly different types — Int (person count),
 * PoseMetrics, EmotionResult, AudioMetrics, SpeechInsights, etc.
 * InferenceOutput provides a uniform envelope that carries:
 *
 * 1. **modelId** — identifies which processor produced this result
 * 2. **fields** — processor-specific key-value results (e.g., "person_count" → 3,
 *    "dominant_emotion" → "happy", "age_range" → "25-34")
 * 3. **latencyMs** — wall-clock inference time for performance monitoring
 * 4. **confidence** — overall confidence score (0.0–1.0) for the result
 *
 * The [fields] map is intentionally untyped (Map<String, Any>) to accommodate
 * the diverse output shapes across processors without forcing a lowest-common-
 * denominator schema. Downstream consumers (SensingProfileManager, heartbeat
 * telemetry) can extract typed values using extension functions or inline casts.
 *
 * @param modelId Unique identifier matching [InferenceProcessor.modelId],
 *   e.g., "efficientdet", "yamnet", "fer_plus", "movenet", "age_gender".
 * @param fields Processor-specific result fields as key-value pairs.
 * @param latencyMs Wall-clock inference time in milliseconds.
 * @param confidence Overall confidence score for this result (0.0 = no confidence,
 *   1.0 = maximum confidence). Processors that don't produce a single confidence
 *   score should use the highest individual detection confidence or 1.0f.
 *   For VLM processors, this is the parse confidence (how cleanly the JSON was
 *   extracted). See [semanticConfidence] for multi-signal agreement-based confidence.
 * @param semanticConfidence Multi-signal calibrated confidence (0.0 to 1.0), or null
 *   if calibration has not been performed. When non-null, this reflects
 *   P(observation correct | sensor data) measured by agreement across independent
 *   signal sources (BlazeFace, EfficientDet, VLM, audio, shadow). Consumers should
 *   prefer this over [confidence] when available. The effective confidence is:
 *   `semanticConfidence ?: confidence` (backwards-compatible).
 * @param timestampMs Timestamp when inference completed (epoch milliseconds).
 *   Defaults to current time.
 */
data class InferenceOutput(
    val modelId: String,
    val fields: Map<String, Any>,
    val latencyMs: Long,
    val confidence: Float = 1.0f,
    val semanticConfidence: Float? = null,
    val timestampMs: Long = System.currentTimeMillis()
) {
    /**
     * Effective confidence: prefers calibrated semantic confidence over parse confidence.
     * Backwards-compatible — returns [confidence] when [semanticConfidence] is null.
     */
    val effectiveConfidence: Float get() = semanticConfidence ?: confidence

    /**
     * Get a typed field value, or null if the key doesn't exist or the type doesn't match.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T> getField(key: String): T? = fields[key] as? T

    /**
     * Get a typed field value with a default fallback.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T> getField(key: String, default: T): T = (fields[key] as? T) ?: default

    /**
     * Check if the result has a specific field.
     */
    fun hasField(key: String): Boolean = fields.containsKey(key)

    /**
     * True if confidence meets or exceeds the given threshold.
     */
    fun isConfident(threshold: Float = 0.5f): Boolean = confidence >= threshold
}
