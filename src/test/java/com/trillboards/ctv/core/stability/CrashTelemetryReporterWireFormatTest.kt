package com.trillboards.ctv.core.stability

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the [CrashTelemetryReporter.Companion.buildStabilityEventJson]
 * wire-format builder. The helper is extracted from `reportEvent` so we can
 * verify the JSON shape without standing up Context / SocketManager / health
 * registry dependencies.
 *
 * Background — bug fix dated 2026-05-03:
 *   The stability reporter previously sent only `memory_total_mb`, populated
 *   from `Runtime.getRuntime().maxMemory()` (the JVM heap cap, ~512 MB on
 *   every Android device by default). 100% of `device_stability_events` rows
 *   carried `memory_total_mb=512` regardless of physical device RAM, which
 *   made the column useless for fleet-RAM analysis.
 *
 *   The fix adds a NEW `device_total_ram_mb` field populated from
 *   `ActivityManager.MemoryInfo.totalMem` — the real physical RAM total. The
 *   legacy `memory_total_mb` field is preserved (it still carries the JVM
 *   heap cap) for back-compat with existing readers and historical rows.
 *   Rename to `jvm_heap_max_mb` is deferred (see deferred-work.yaml).
 *
 * These tests pin the contract:
 *   1. Both fields appear in the wire payload when both are supplied.
 *   2. `device_total_ram_mb` is omitted when null (ActivityManager unavailable).
 *   3. The field separation is real — RAM > heap cap on a device with > 512 MB.
 */
class CrashTelemetryReporterWireFormatTest {

    private fun build(
        deviceTotalRamMb: Int? = 12_288,
        jvmHeapMaxMb: Int = 512,
        memoryUsedMb: Int = 384,
        stackTrace: String? = null,
        activeModels: JSONArray? = null,
        metadata: Map<String, Any> = emptyMap()
    ) = CrashTelemetryReporter.buildStabilityEventJson(
        eventType = "crash",
        fingerprint = "fp-tab-s11",
        screenId = "screen-123",
        renderingMode = "native",
        agentVersion = "8.4.0",
        deviceModel = "samsung SM-X910",
        chipset = "qcom",
        attenuationTier = "NORMAL",
        memoryUsedMb = memoryUsedMb,
        jvmHeapMaxMb = jvmHeapMaxMb,
        deviceTotalRamMb = deviceTotalRamMb,
        activeModels = activeModels,
        metadata = metadata,
        stackTrace = stackTrace,
        timestampMs = 1_730_000_000_000L
    )

    @Test
    fun `wire format includes both memory_total_mb and device_total_ram_mb when supplied`() {
        val json = build(deviceTotalRamMb = 12_288, jvmHeapMaxMb = 512)

        // Legacy field — still carries JVM heap cap value (Runtime.maxMemory).
        assertTrue(
            "memory_total_mb must be present (back-compat)",
            json.has("memory_total_mb")
        )
        assertEquals(512, json.getInt("memory_total_mb"))

        // New field — physical device RAM (ActivityManager.MemoryInfo.totalMem).
        assertTrue(
            "device_total_ram_mb must be present when supplied",
            json.has("device_total_ram_mb")
        )
        assertEquals(12_288, json.getInt("device_total_ram_mb"))
    }

    @Test
    fun `device_total_ram_mb is omitted when null (ActivityManager unavailable)`() {
        val json = build(deviceTotalRamMb = null)

        assertFalse(
            "device_total_ram_mb must be omitted when null — server tolerates absence",
            json.has("device_total_ram_mb")
        )
        // memory_total_mb still rides — never lose the legacy signal.
        assertTrue(json.has("memory_total_mb"))
    }

    @Test
    fun `Tab S11 12 GB device shows distinct heap cap and physical RAM`() {
        // The bug-of-record: every row reported memory_total_mb=512 on a
        // Tab S11 with 12 GB of physical RAM. After the fix, the two values
        // diverge by ~24x, which is the signal we lost for two months.
        val json = build(deviceTotalRamMb = 12_288, jvmHeapMaxMb = 512)

        val heapCap = json.getInt("memory_total_mb")
        val physicalRam = json.getInt("device_total_ram_mb")

        assertNotEquals(
            "Pre-fix bug: heap cap and physical RAM were the same field",
            heapCap, physicalRam
        )
        assertTrue(
            "Physical RAM ($physicalRam MB) must exceed heap cap ($heapCap MB) on a 12 GB device",
            physicalRam > heapCap
        )
        // Sanity: not the bug value.
        assertTrue(
            "device_total_ram_mb=$physicalRam — must not collapse to the 512 MB heap cap",
            physicalRam > 512
        )
    }

    @Test
    fun `wire format keeps existing core fields stable`() {
        // Regression guard: the helper is the new home of the JSON shape; if
        // any of these field names change, server `stabilityService.recordStabilityEvent`
        // and the device_stability_events INSERT will silently lose data.
        val json = build()

        listOf(
            "event_type", "fingerprint", "screen_id", "rendering_mode",
            "agent_version", "device_model", "chipset", "attenuation_tier",
            "memory_used_mb", "memory_total_mb", "metadata", "timestamp"
        ).forEach { field ->
            assertTrue("required field $field missing from wire payload", json.has(field))
        }
    }

    @Test
    fun `optional fields are conditional on presence`() {
        val withoutOptionals = build(
            deviceTotalRamMb = null,
            stackTrace = null,
            activeModels = null
        )
        assertFalse(withoutOptionals.has("device_total_ram_mb"))
        assertFalse(withoutOptionals.has("stack_trace"))
        assertFalse(withoutOptionals.has("active_models"))

        val withOptionals = build(
            deviceTotalRamMb = 4096,
            stackTrace = "java.lang.NullPointerException at Foo.kt:42",
            activeModels = JSONArray(listOf("blazeface", "movenet"))
        )
        assertTrue(withOptionals.has("device_total_ram_mb"))
        assertTrue(withOptionals.has("stack_trace"))
        assertTrue(withOptionals.has("active_models"))
        assertEquals(4096, withOptionals.getInt("device_total_ram_mb"))
    }
}
