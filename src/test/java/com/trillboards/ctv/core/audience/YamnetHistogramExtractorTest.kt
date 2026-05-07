package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.tensorflow.lite.support.label.Category

/**
 * Unit tests for YamnetHistogramExtractor (Phase 4 PR 8).
 *
 * Covers:
 *   - per-class binned counts in 0.5s windows
 *   - score-threshold filtering
 *   - class-label sanitization (whitespace, mixed case, punctuation)
 *   - empty / out-of-order input
 *   - reset semantics
 *   - JSON-friendly snapshot shape (matches audienceSignals contract)
 */
class YamnetHistogramExtractorTest {

    private fun cat(label: String, score: Float): Category =
        Category.create(label, "", score)

    @Test
    fun `single classification with one class lands in one bin`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_000L)

        val snap = extractor.snapshotForPayload()
        assertEquals(1, snap.size)
        assertEquals(mapOf("speech" to 1), snap["0"])
    }

    @Test
    fun `multiple classes in same bin all land`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(
            listOf(
                cat("Speech", 0.6f),
                cat("Music", 0.7f),
                cat("Crowd", 0.5f),
            ),
            timestampMs = 1_000L,
        )

        val snap = extractor.snapshotForPayload()
        assertEquals(1, snap.size)
        val bin0 = snap["0"]!!
        assertEquals(1, bin0["speech"])
        assertEquals(1, bin0["music"])
        assertEquals(1, bin0["crowd"])
    }

    @Test
    fun `repeated class in same bin increments count`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_000L)
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_100L)
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_300L)

        val snap = extractor.snapshotForPayload()
        // All three observations fall in the 0-499ms bin (offset "0").
        assertEquals(1, snap.size)
        assertEquals(3, snap["0"]!!["speech"])
    }

    @Test
    fun `observations cross 500ms bin boundary into separate bins`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_000L)  // bin 0
        extractor.add(listOf(cat("Music", 0.9f)), timestampMs = 1_500L)   // bin 500
        extractor.add(listOf(cat("Music", 0.9f)), timestampMs = 1_900L)   // bin 500
        extractor.add(listOf(cat("Crowd", 0.9f)), timestampMs = 2_000L)   // bin 1000

        val snap = extractor.snapshotForPayload()
        assertEquals(3, snap.size)
        assertEquals(mapOf("speech" to 1), snap["0"])
        assertEquals(mapOf("music" to 2), snap["500"])
        assertEquals(mapOf("crowd" to 1), snap["1000"])
    }

    @Test
    fun `score below threshold is filtered out`() {
        val extractor = YamnetHistogramExtractor(scoreThreshold = 0.5f)
        extractor.add(
            listOf(
                cat("Speech", 0.9f),    // above threshold
                cat("Music", 0.10f),    // below threshold (filtered)
            ),
            timestampMs = 1_000L,
        )

        val snap = extractor.snapshotForPayload()
        val bin0 = snap["0"]!!
        assertEquals(1, bin0["speech"])
        assertNull("music below threshold should be filtered", bin0["music"])
    }

    @Test
    fun `class label is sanitized to snake_case lowercase`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(
            listOf(
                cat("  Pop Music  ", 0.9f),     // whitespace + space-separated
                cat("Hip Hop Music", 0.9f),     // multi-word
                cat("Heavy/Metal", 0.9f),       // punctuation
                cat("UPPERCASE", 0.9f),         // case
            ),
            timestampMs = 1_000L,
        )

        val snap = extractor.snapshotForPayload()
        val bin0 = snap["0"]!!
        assertEquals(1, bin0["pop_music"])
        assertEquals(1, bin0["hip_hop_music"])
        assertEquals(1, bin0["heavy_metal"])
        assertEquals(1, bin0["uppercase"])
    }

    @Test
    fun `empty or blank class label is dropped`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(
            listOf(
                cat("", 0.9f),
                cat("   ", 0.9f),
                cat("???", 0.9f), // sanitizes to empty after stripping non-alnum
                cat("Speech", 0.9f),
            ),
            timestampMs = 1_000L,
        )

        val snap = extractor.snapshotForPayload()
        val bin0 = snap["0"]!!
        assertEquals(1, bin0.size)
        assertEquals(1, bin0["speech"])
    }

    @Test
    fun `class-label sanitizer truncates over 60 chars`() {
        val longLabel = "a".repeat(120)
        val sanitized = YamnetHistogramExtractor.sanitizeClassLabel(longLabel)
        assertNotNull(sanitized)
        assertEquals(60, sanitized!!.length)
    }

    @Test
    fun `out-of-order timestamp before window start is skipped`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 2_000L)
        // This timestamp is BEFORE the first window-start (2000) — should skip.
        extractor.add(listOf(cat("Music", 0.9f)), timestampMs = 1_500L)

        val snap = extractor.snapshotForPayload()
        assertEquals(1, snap.size)
        assertEquals(mapOf("speech" to 1), snap["0"])
    }

    @Test
    fun `empty categories list is a no-op`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(emptyList(), timestampMs = 1_000L)

        assertTrue(extractor.isEmpty())
        assertEquals(0, extractor.binCount())
        assertEquals(emptyMap<String, Map<String, Int>>(), extractor.snapshotForPayload())
    }

    @Test
    fun `reset clears state and re-anchors window start`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_000L)
        extractor.add(listOf(cat("Speech", 0.9f)), timestampMs = 1_100L)
        assertFalse(extractor.isEmpty())

        extractor.reset()

        assertTrue(extractor.isEmpty())
        assertEquals(0, extractor.binCount())
        assertEquals(0L, extractor.totalObservations)

        // After reset, a new window starts at the next add().
        extractor.add(listOf(cat("Crowd", 0.9f)), timestampMs = 5_000L)
        val snap = extractor.snapshotForPayload()
        assertEquals(1, snap.size)
        assertEquals(mapOf("crowd" to 1), snap["0"])
    }

    @Test
    fun `maxClassesPerBin caps NEW classes but still increments existing ones`() {
        val extractor = YamnetHistogramExtractor(maxClassesPerBin = 2)
        extractor.add(
            listOf(
                cat("Class1", 0.9f),
                cat("Class2", 0.9f),
                cat("Class3", 0.9f),  // dropped (at cap)
                cat("Class4", 0.9f),  // dropped (at cap)
            ),
            timestampMs = 1_000L,
        )
        // Existing classes should still increment.
        extractor.add(
            listOf(cat("Class1", 0.9f), cat("Class3", 0.9f)),
            timestampMs = 1_100L,
        )

        val snap = extractor.snapshotForPayload()
        val bin0 = snap["0"]!!
        assertEquals(2, bin0.size)
        assertEquals(2, bin0["class1"])  // got two adds
        assertEquals(1, bin0["class2"])  // single add
        // Class3/Class4 never registered because the bin was at cap.
    }

    @Test
    fun `NaN score is filtered out`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(
            listOf(
                cat("Speech", Float.NaN),
                cat("Music", 0.9f),
            ),
            timestampMs = 1_000L,
        )
        val bin0 = extractor.snapshotForPayload()["0"]!!
        assertNull(bin0["speech"])
        assertEquals(1, bin0["music"])
    }

    @Test
    fun `totalObservations counts only above-threshold entries`() {
        val extractor = YamnetHistogramExtractor(scoreThreshold = 0.5f)
        extractor.add(
            listOf(
                cat("Above", 0.9f),
                cat("Below", 0.1f),  // filtered
                cat("AlsoAbove", 0.6f),
            ),
            timestampMs = 1_000L,
        )
        assertEquals(2L, extractor.totalObservations)
    }

    @Test
    fun `snapshotForPayload uses string bin keys for JSON serialization`() {
        val extractor = YamnetHistogramExtractor()
        extractor.add(listOf(cat("A", 0.9f)), timestampMs = 1_000L)
        extractor.add(listOf(cat("A", 0.9f)), timestampMs = 1_500L)
        extractor.add(listOf(cat("A", 0.9f)), timestampMs = 2_000L)

        val snap = extractor.snapshotForPayload()
        // Bin keys MUST be strings — JSONObject.put(key,...) requires String keys.
        for (key in snap.keys) {
            // Each bin key is a stringified Long, parseable cleanly.
            key.toLong()
        }
        assertEquals(setOf("0", "500", "1000"), snap.keys)
    }
}
