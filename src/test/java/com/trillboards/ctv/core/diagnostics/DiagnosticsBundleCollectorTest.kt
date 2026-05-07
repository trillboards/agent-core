package com.trillboards.ctv.core.diagnostics

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-Kotlin tests for the redaction security fence. Asserts every
 * pattern in `DiagnosticsBundleCollector.PARTNER_CODE_PATTERNS` strips
 * its target form, and that an innocuous log line passes through
 * unchanged. Live `collect()` requires an Android Context, so it's
 * exercised on-device / via the existing CTV instrumentation tests
 * — not here.
 */
class DiagnosticsBundleCollectorTest {

    @Test
    fun `redacts variant-code aliases like v1234`() {
        val out = DiagnosticsBundleCollector.redactPartnerCodes("partner=v1234 in waterfall")
        assertEquals("partner=[REDACTED] in waterfall", out)
    }

    @Test
    fun `redacts v9999 and other 4-digit variant codes`() {
        val out = DiagnosticsBundleCollector.redactPartnerCodes("v9999 v1000 v0042")
        assertEquals("[REDACTED] [REDACTED] [REDACTED]", out)
    }

    @Test
    fun `redacts variant_xyz alternate alias form`() {
        val out = DiagnosticsBundleCollector.redactPartnerCodes("ssp_alias=variant_xyz; route=variant_alpha9")
        assertEquals("ssp_alias=[REDACTED]; route=[REDACTED]", out)
    }

    @Test
    fun `redacts bare partner names case-insensitively`() {
        val cases = listOf(
            "Adipolo wins" to "[REDACTED] wins",
            "vidverto returns 200" to "[REDACTED] returns 200",
            "VidVerto returns 200" to "[REDACTED] returns 200",
            "JustBaat error" to "[REDACTED] error",
            "adtelligent timeout" to "[REDACTED] timeout",
            "Take10 ad served" to "[REDACTED] ad served",
            "BCTV partner ack" to "[REDACTED] partner ack",
            "NRS demand chain" to "[REDACTED] demand chain"
        )
        for ((input, expected) in cases) {
            assertEquals(
                "Failed to redact: $input",
                expected,
                DiagnosticsBundleCollector.redactPartnerCodes(input)
            )
        }
    }

    @Test
    fun `redacts partner-namespaced subdomains`() {
        val cases = listOf(
            "GET https://ssp.adipolo.com/bid -> 200" to "GET https://[REDACTED]/bid -> 200",
            "host=dsp.vidverto.io path=/" to "host=[REDACTED] path=/",
            "ssp.justbaat.net responded" to "[REDACTED] responded"
        )
        for ((input, expected) in cases) {
            assertEquals(
                "Failed to redact: $input",
                expected,
                DiagnosticsBundleCollector.redactPartnerCodes(input)
            )
        }
    }

    @Test
    fun `passes through innocuous log lines unchanged`() {
        val benignLines = listOf(
            "AudienceController: emitted 3 face frames",
            "VAST request: outcome=requested",
            "INFO: heartbeat tick at 1714915200000",
            "Camera permission: granted",
            "[Logcat] kiosk lock active=true",
            "version 0.6.23-tablet built at 2026-05-06"
        )
        for (line in benignLines) {
            assertEquals(
                "Innocuous line was modified: $line",
                line,
                DiagnosticsBundleCollector.redactPartnerCodes(line)
            )
        }
    }

    @Test
    fun `redaction is idempotent`() {
        val input = "Adipolo via variant_xyz at ssp.vidverto.com"
        val once = DiagnosticsBundleCollector.redactPartnerCodes(input)
        val twice = DiagnosticsBundleCollector.redactPartnerCodes(once)
        assertEquals(once, twice)
        assertFalse(twice.contains("Adipolo", ignoreCase = true))
        assertFalse(twice.contains("vidverto", ignoreCase = true))
        assertFalse(twice.contains("variant_xyz", ignoreCase = true))
    }

    @Test
    fun `redaction handles multiple matches on same line`() {
        val out = DiagnosticsBundleCollector.redactPartnerCodes(
            "v1234 -> Adipolo via ssp.adipolo.com"
        )
        assertEquals("[REDACTED] -> [REDACTED] via [REDACTED]", out)
    }

    @Test
    fun `redaction does not mangle unrelated 4-digit numbers`() {
        // 4-digit numbers without the leading 'v' must NOT be redacted.
        // The pattern uses `\bv\d{4}\b` so a bare year like "2026" stays.
        val out = DiagnosticsBundleCollector.redactPartnerCodes("year=2026 errno=1234")
        assertEquals("year=2026 errno=1234", out)
    }

    @Test
    fun `redaction does not affect empty strings`() {
        assertEquals("", DiagnosticsBundleCollector.redactPartnerCodes(""))
    }

    @Test
    fun `redaction strips a real-looking error log line`() {
        // Crafted to look like a real prod log line: a partner integration
        // failure that includes both a variant code and a domain. The fence
        // must catch BOTH halves without any per-call helper coordination.
        val raw = """
            ERROR VastWaterfall: source v0042 (ssp.adipolo.com) returned 502 — falling
              back to JustBaat backup. partner_alias=variant_alpha7
        """.trimIndent()
        val redacted = DiagnosticsBundleCollector.redactPartnerCodes(raw)
        assertFalse("Domain leaked", redacted.contains("adipolo", ignoreCase = true))
        assertFalse("Bare partner name leaked", redacted.contains("JustBaat", ignoreCase = true))
        assertFalse("Variant code leaked", redacted.contains("v0042"))
        assertFalse("Variant alias leaked", redacted.contains("variant_alpha7"))
        // Sanity: the surrounding error context survives.
        assertTrue(redacted.contains("returned 502"))
        assertTrue(redacted.contains("falling"))
    }
}
