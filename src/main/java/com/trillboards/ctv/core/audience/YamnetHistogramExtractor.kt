package com.trillboards.ctv.core.audience

import org.tensorflow.lite.support.label.Category

/**
 * YamnetHistogramExtractor — accumulates YAMNet classification results into a
 * per-class histogram binned by 0.5-second windows (Phase 4 PR 8 item #3).
 *
 * Why a histogram (not "dominant class"):
 *   - The existing AudioClassificationProcessor already emits a *dominant*
 *     class per ~1s window (AudioMetrics.dominantClass). That collapses
 *     multi-class signal — a venue with simultaneous "music_playing" and
 *     "speech" loses one of those signals.
 *   - The histogram preserves all classes that fired above the score
 *     threshold in each 0.5s bin, sized for downstream typed fan-out into
 *     observation_field_values via embeddingWorker.fanOutProfileFields:
 *     each (bin_index, class_label) pair lands as one row.
 *
 * Output shape (audienceSignals.audio_class_histogram):
 *
 *   {
 *     "<bin_0_offset_ms>": { "<class_label>": count, ... },
 *     "<bin_1_offset_ms>": { ... },
 *     ...
 *   }
 *
 * The keys are bin offsets in milliseconds relative to the window start
 * (e.g. "0", "500", "1000", ... up to BIN_SIZE_MS * BIN_COUNT).
 *
 * Privacy:
 *   - Accepts only category labels + scores from the YAMNet classifier; no
 *     raw audio bytes ever pass through this extractor.
 *   - Class labels are sanitized to lowercase snake_case to keep the
 *     downstream typed map key shape predictable for embeddingWorker.
 *
 * Concurrency: not thread-safe by itself. The caller (AudienceSensingService)
 * already serializes audio classification results through audioMetricsBuffer,
 * so a single extractor instance is sufficient.
 */
class YamnetHistogramExtractor(
    private val binSizeMs: Long = DEFAULT_BIN_SIZE_MS,
    private val scoreThreshold: Float = DEFAULT_SCORE_THRESHOLD,
    private val maxClassesPerBin: Int = DEFAULT_MAX_CLASSES_PER_BIN,
) {
    companion object {
        const val DEFAULT_BIN_SIZE_MS: Long = 500L
        const val DEFAULT_SCORE_THRESHOLD: Float = 0.10f
        // Cap classes-per-bin at a defensive ceiling so a runaway model can't
        // explode the typed-fanout row count.
        const val DEFAULT_MAX_CLASSES_PER_BIN: Int = 16

        /**
         * Sanitize a YAMNet category label to a stable snake_case key.
         * Mirrors the server-side embeddingWorker.sanitizeFieldKey contract
         * so a class label produced here can land directly as a field_key
         * without further normalization.
         */
        fun sanitizeClassLabel(rawLabel: String?): String? {
            if (rawLabel == null) return null
            val trimmed = rawLabel.trim()
            if (trimmed.isEmpty()) return null
            // Lowercase, then collapse any non-[a-z0-9] run to a single underscore.
            val lowered = trimmed.lowercase()
            val collapsed = StringBuilder()
            var lastWasUnderscore = false
            for (ch in lowered) {
                if (ch in 'a'..'z' || ch in '0'..'9') {
                    collapsed.append(ch)
                    lastWasUnderscore = false
                } else if (!lastWasUnderscore) {
                    collapsed.append('_')
                    lastWasUnderscore = true
                }
            }
            // Trim leading/trailing underscores.
            var start = 0
            var endExclusive = collapsed.length
            while (start < endExclusive && collapsed[start] == '_') start++
            while (endExclusive > start && collapsed[endExclusive - 1] == '_') endExclusive--
            if (start >= endExclusive) return null
            val cleaned = collapsed.substring(start, endExclusive)
            return if (cleaned.length > 60) cleaned.substring(0, 60) else cleaned
        }
    }

    /**
     * Per-bin map: classLabel -> count.
     * Keyed by the bin's offset (ms) within the current window.
     */
    private val bins: MutableMap<Long, MutableMap<String, Int>> = LinkedHashMap()

    /**
     * Window start timestamp (ms since epoch). Set on first observation in the
     * window. Used to compute bin offsets relative to the window.
     */
    private var windowStartMs: Long = -1L

    /**
     * Total count of observed (label,score>=threshold) entries.
     * Exposed for telemetry / asserts.
     */
    var totalObservations: Long = 0L
        private set

    /**
     * Add YAMNet classification results from a single classify() call.
     *
     * @param categories  YAMNet classifier output (top-N categories).
     * @param timestampMs Wall-clock time (ms) the classification was produced.
     */
    fun add(categories: List<Category>, timestampMs: Long) {
        if (categories.isEmpty()) return
        if (windowStartMs < 0L) {
            windowStartMs = timestampMs
        }
        // Map the timestamp into a bin offset relative to window start.
        val deltaMs = timestampMs - windowStartMs
        if (deltaMs < 0L) return // out-of-order observation — skip
        val binOffset = (deltaMs / binSizeMs) * binSizeMs

        val perClass = bins.getOrPut(binOffset) { LinkedHashMap() }

        for (category in categories) {
            val score = category.score
            if (score.isNaN() || score < scoreThreshold) continue
            val label = sanitizeClassLabel(category.label) ?: continue
            // Defensive cap: if the bin is already full and we'd add a new
            // class, drop the new one. Existing classes still increment.
            if (perClass.containsKey(label)) {
                perClass[label] = (perClass[label] ?: 0) + 1
                totalObservations++
            } else if (perClass.size < maxClassesPerBin) {
                perClass[label] = 1
                totalObservations++
            }
        }
    }

    /**
     * Snapshot the current histogram as an immutable (Long->Map<String,Int>)
     * structure suitable for JSON serialization in audienceSignals.
     *
     * The returned map is bin-offset (ms) -> (className -> count).
     */
    fun snapshot(): Map<Long, Map<String, Int>> {
        if (bins.isEmpty()) return emptyMap()
        val out = LinkedHashMap<Long, Map<String, Int>>(bins.size)
        for ((binOffset, perClass) in bins) {
            out[binOffset] = perClass.toMap()
        }
        return out
    }

    /**
     * Snapshot as a JSON-friendly structure where bin keys are stringified
     * offsets ("0", "500", ...). This is the exact shape that lands in
     * audienceSignals.audio_class_histogram and gets typed-fanned-out by
     * embeddingWorker.fanOutProfileFields into observation_field_values rows
     * keyed `audio_class_histogram_<bin_offset>_<class_label>`.
     */
    fun snapshotForPayload(): Map<String, Map<String, Int>> {
        if (bins.isEmpty()) return emptyMap()
        val out = LinkedHashMap<String, Map<String, Int>>(bins.size)
        for ((binOffset, perClass) in bins) {
            out[binOffset.toString()] = perClass.toMap()
        }
        return out
    }

    /**
     * Reset the extractor for a new window.
     */
    fun reset() {
        bins.clear()
        windowStartMs = -1L
        totalObservations = 0L
    }

    /**
     * @return number of populated bins (useful for assertions/logging).
     */
    fun binCount(): Int = bins.size

    /**
     * @return whether any classes have been recorded yet.
     */
    fun isEmpty(): Boolean = bins.isEmpty()
}
