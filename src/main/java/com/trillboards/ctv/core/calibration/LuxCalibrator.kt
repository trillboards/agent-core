package com.trillboards.ctv.core.calibration

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import org.json.JSONObject
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Calibrates ambient lux thresholds during the first 30 minutes after deployment.
 *
 * Problem: The default ViewabilityConfig lux breakpoints (dark=5, dim=50, good=500,
 * bright=10000) are generic. A screen deployed in a dim bar has very different ambient
 * light than one in a brightly-lit mall food court. Using the same breakpoints means
 * the dim bar screen always reports "dark" viewability (false negative) while the mall
 * screen never triggers "bright" glare correction (false positive).
 *
 * Solution: During the first 30 minutes after deployment (or after a manual recalibration
 * trigger), this class collects ambient lux readings from SensorCollector and computes
 * percentile breakpoints:
 *   - dark  = 10th percentile
 *   - dim   = 25th percentile
 *   - good  = 75th percentile
 *   - bright = 95th percentile
 *
 * These are persisted to SharedPreferences and applied to SensingConfig.viewability,
 * so the viewability scoring adapts to each screen's actual environment.
 *
 * Lifecycle:
 *   1. AudienceSensingService creates LuxCalibrator on start
 *   2. Each sensor tick calls recordLuxReading()
 *   3. After calibrationWindowMs, finalize() is called automatically
 *   4. On subsequent boots, the persisted calibration is loaded and applied
 *   5. Server can trigger recalibration via a profile update
 */
class LuxCalibrator(
    private val context: Context,
    private val calibrationWindowMs: Long = DEFAULT_CALIBRATION_WINDOW_MS
) {
    companion object {
        private const val TAG = "LuxCalibrator"
        private const val PREFS_NAME = "lux_calibration"
        private const val KEY_CALIBRATION_JSON = "calibration_data"
        private const val KEY_CALIBRATED_AT = "calibrated_at_ms"
        private const val DEFAULT_CALIBRATION_WINDOW_MS = 30L * 60L * 1000L // 30 minutes
        private const val MIN_SAMPLES_FOR_CALIBRATION = 30 // At least 30 readings
        private const val MAX_SAMPLE_SIZE = 5000 // Cap memory usage
    }

    /** Current calibration state. */
    enum class State {
        /** No calibration data, collecting samples */
        COLLECTING,
        /** Calibration complete, thresholds applied */
        CALIBRATED,
        /** Loaded from persisted storage (previous calibration) */
        LOADED_FROM_STORAGE
    }

    private val state = AtomicReference(State.COLLECTING)
    private val luxSamples = CopyOnWriteArrayList<Float>()
    private val startTimeMs = System.currentTimeMillis()
    private val finalized = AtomicBoolean(false)
    private val calibrationResult = AtomicReference<LuxCalibrationResult?>(null)

    init {
        // Try to load a previous calibration on construction
        val persisted = loadPersistedCalibration()
        if (persisted != null) {
            calibrationResult.set(persisted)
            state.set(State.LOADED_FROM_STORAGE)
            applyToSensingConfig(persisted)
            Log.i(TAG, "Loaded persisted lux calibration: $persisted")
        }
    }

    /**
     * Record a single lux reading from the light sensor.
     * Called by the sensor listener on each TYPE_LIGHT event.
     *
     * @param lux Raw lux value from the sensor
     */
    fun recordLuxReading(lux: Float) {
        // Ignore invalid readings
        if (lux < 0f) return

        // If already finalized (from this session), don't collect more
        if (finalized.get()) return

        // Cap sample count to avoid unbounded memory
        if (luxSamples.size >= MAX_SAMPLE_SIZE) return

        luxSamples.add(lux)

        // Check if calibration window has elapsed
        val elapsedMs = System.currentTimeMillis() - startTimeMs
        if (elapsedMs >= calibrationWindowMs && !finalized.get()) {
            finalize()
        }
    }

    /**
     * Finalize the calibration: sort samples, compute percentiles, persist, and apply.
     * Idempotent -- calling multiple times has no additional effect.
     *
     * @return LuxCalibrationResult if calibration succeeded, null if insufficient data
     */
    fun finalize(): LuxCalibrationResult? {
        if (!finalized.compareAndSet(false, true)) {
            // Already finalized
            return calibrationResult.get()
        }

        if (luxSamples.size < MIN_SAMPLES_FOR_CALIBRATION) {
            Log.w(TAG, "Insufficient lux samples for calibration: " +
                "${luxSamples.size} < $MIN_SAMPLES_FOR_CALIBRATION")
            finalized.set(false) // Allow retry
            return null
        }

        val sorted = luxSamples.toList().sorted()
        val result = LuxCalibrationResult(
            darkLuxThreshold = percentile(sorted, 10),
            dimLuxThreshold = percentile(sorted, 25),
            goodLuxThreshold = percentile(sorted, 75),
            brightLuxThreshold = percentile(sorted, 95),
            sampleCount = sorted.size,
            minLux = sorted.first(),
            maxLux = sorted.last(),
            medianLux = percentile(sorted, 50),
            calibratedAtMs = System.currentTimeMillis(),
            calibrationWindowMs = calibrationWindowMs
        )

        calibrationResult.set(result)
        state.set(State.CALIBRATED)
        persistCalibration(result)
        applyToSensingConfig(result)

        Log.i(TAG, "Lux calibration complete: dark=${result.darkLuxThreshold}, " +
            "dim=${result.dimLuxThreshold}, good=${result.goodLuxThreshold}, " +
            "bright=${result.brightLuxThreshold} (${result.sampleCount} samples, " +
            "range=${result.minLux}-${result.maxLux})")

        return result
    }

    /**
     * Force a recalibration by resetting state and collecting new samples.
     * Use when the server sends a recalibrate command or the deployment changes.
     */
    fun resetAndRecalibrate() {
        finalized.set(false)
        luxSamples.clear()
        state.set(State.COLLECTING)
        calibrationResult.set(null)
        Log.i(TAG, "Lux calibration reset -- collecting new samples")
    }

    /** Get the current calibration state. */
    fun getState(): State = state.get()

    /** Get the current calibration result (null if not yet calibrated). */
    fun getCalibrationResult(): LuxCalibrationResult? = calibrationResult.get()

    /** Get the number of samples collected so far. */
    fun getSampleCount(): Int = luxSamples.size

    /** Get elapsed time since calibration started. */
    fun getElapsedMs(): Long = System.currentTimeMillis() - startTimeMs

    /**
     * Get calibration metadata for heartbeat reporting.
     * Returns a JSON object with state, sample count, and thresholds if calibrated.
     */
    fun toHeartbeatMetadata(): JSONObject = JSONObject().apply {
        put("state", state.get().name)
        put("sampleCount", luxSamples.size)
        put("elapsedMs", getElapsedMs())
        calibrationResult.get()?.let { result ->
            put("darkLux", result.darkLuxThreshold)
            put("dimLux", result.dimLuxThreshold)
            put("goodLux", result.goodLuxThreshold)
            put("brightLux", result.brightLuxThreshold)
            put("medianLux", result.medianLux)
            put("calibratedAtMs", result.calibratedAtMs)
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /**
     * Compute a percentile value from a pre-sorted list.
     * Uses linear interpolation between adjacent values.
     */
    internal fun percentile(sorted: List<Float>, p: Int): Float {
        require(sorted.isNotEmpty()) { "Cannot compute percentile of empty list" }
        require(p in 0..100) { "Percentile must be 0..100, got $p" }

        if (sorted.size == 1) return sorted[0]

        val index = (p / 100.0) * (sorted.size - 1)
        val lower = index.toInt()
        val upper = (lower + 1).coerceAtMost(sorted.size - 1)
        val fraction = index - lower

        return sorted[lower] + fraction.toFloat() * (sorted[upper] - sorted[lower])
    }

    /**
     * Apply calibration results to SensingConfig.viewability via updateFromJson.
     */
    private fun applyToSensingConfig(result: LuxCalibrationResult) {
        val json = JSONObject().apply {
            put("viewability", JSONObject().apply {
                put("darkLuxThreshold", result.darkLuxThreshold)
                put("dimLuxThreshold", result.dimLuxThreshold)
                put("goodLuxThreshold", result.goodLuxThreshold)
                put("brightLuxThreshold", result.brightLuxThreshold)
            })
        }
        SensingConfig.updateFromJson(json)
        // Mark these fields as auto-calibrated so CalibrationTelemetry can
        // distinguish them from server-pushed overrides
        SensingConfig.markAutoCalibrated("viewability", setOf(
            "darkLuxThreshold", "dimLuxThreshold", "goodLuxThreshold", "brightLuxThreshold"
        ))
        Log.d(TAG, "Applied lux calibration to SensingConfig.viewability")
    }

    /**
     * Persist calibration to SharedPreferences for cross-boot survival.
     */
    private fun persistCalibration(result: LuxCalibrationResult) {
        try {
            val prefs = getPrefs()
            prefs.edit()
                .putString(KEY_CALIBRATION_JSON, result.toJson().toString())
                .putLong(KEY_CALIBRATED_AT, result.calibratedAtMs)
                .apply()
            Log.d(TAG, "Persisted lux calibration to SharedPreferences")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to persist lux calibration", e)
        }
    }

    /**
     * Load a previous calibration from SharedPreferences.
     */
    private fun loadPersistedCalibration(): LuxCalibrationResult? {
        return try {
            val prefs = getPrefs()
            val json = prefs.getString(KEY_CALIBRATION_JSON, null) ?: return null
            LuxCalibrationResult.fromJson(JSONObject(json))
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load persisted lux calibration", e)
            null
        }
    }

    private fun getPrefs(): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }
}

/**
 * Results of the lux calibration process.
 * Contains the computed percentile breakpoints and metadata about the calibration.
 */
data class LuxCalibrationResult(
    /** 10th percentile lux -- below this is considered "dark" */
    val darkLuxThreshold: Float,
    /** 25th percentile lux -- between dark and dim is "dim" */
    val dimLuxThreshold: Float,
    /** 75th percentile lux -- between dim and good is "good" */
    val goodLuxThreshold: Float,
    /** 95th percentile lux -- above this is "bright/glare" */
    val brightLuxThreshold: Float,
    /** Number of samples used for calibration */
    val sampleCount: Int,
    /** Minimum lux observed */
    val minLux: Float,
    /** Maximum lux observed */
    val maxLux: Float,
    /** Median lux (50th percentile) */
    val medianLux: Float,
    /** Timestamp when calibration was finalized */
    val calibratedAtMs: Long,
    /** Duration of the calibration window */
    val calibrationWindowMs: Long
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("darkLuxThreshold", darkLuxThreshold)
        put("dimLuxThreshold", dimLuxThreshold)
        put("goodLuxThreshold", goodLuxThreshold)
        put("brightLuxThreshold", brightLuxThreshold)
        put("sampleCount", sampleCount)
        put("minLux", minLux)
        put("maxLux", maxLux)
        put("medianLux", medianLux)
        put("calibratedAtMs", calibratedAtMs)
        put("calibrationWindowMs", calibrationWindowMs)
    }

    companion object {
        fun fromJson(json: JSONObject): LuxCalibrationResult = LuxCalibrationResult(
            darkLuxThreshold = json.optDouble("darkLuxThreshold", 5.0).toFloat(),
            dimLuxThreshold = json.optDouble("dimLuxThreshold", 50.0).toFloat(),
            goodLuxThreshold = json.optDouble("goodLuxThreshold", 500.0).toFloat(),
            brightLuxThreshold = json.optDouble("brightLuxThreshold", 10000.0).toFloat(),
            sampleCount = json.optInt("sampleCount", 0),
            minLux = json.optDouble("minLux", 0.0).toFloat(),
            maxLux = json.optDouble("maxLux", 0.0).toFloat(),
            medianLux = json.optDouble("medianLux", 0.0).toFloat(),
            calibratedAtMs = json.optLong("calibratedAtMs", 0),
            calibrationWindowMs = json.optLong("calibrationWindowMs", 30L * 60L * 1000L)
        )
    }
}
