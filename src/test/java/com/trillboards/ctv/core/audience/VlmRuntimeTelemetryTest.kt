package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VlmRuntimeTelemetryTest {

    @Test
    fun `serializes blocked runtime state with capacity reason`() {
        val telemetry = VlmRuntimeTelemetry(
            requestedModelId = "gemma_4_e2b",
            activeModelId = null,
            state = VlmRuntimeTelemetry.STATE_BLOCKED,
            reason = "manifest_capacity_exceeded",
            requiredAvailableRamMb = 5406,
            estimatedRuntimeFootprintMb = 3870,
            availableRamMb = 6200,
            nativeHeapMb = 320,
            promptChars = 941
        )

        val json = telemetry.withOutput(hasLastOutput = false, lastOutputModelId = null).toJson()

        assertEquals("gemma_4_e2b", json.getString("requested_model_id"))
        assertEquals("blocked", json.getString("state"))
        assertEquals("manifest_capacity_exceeded", json.getString("reason"))
        assertEquals(5406, json.getInt("required_available_ram_mb"))
        assertFalse(json.getBoolean("has_last_output"))
    }

    @Test
    fun `serializes active runtime state with last parsed output`() {
        val telemetry = VlmRuntimeTelemetry(
            requestedModelId = "gemma_4_e2b",
            activeModelId = "gemma_4_e2b",
            state = VlmRuntimeTelemetry.STATE_ACTIVE,
            reason = "ok"
        )

        val json = telemetry.withOutput(hasLastOutput = true, lastOutputModelId = "gemma_4_e2b").toJson()

        assertEquals("active", json.getString("state"))
        assertEquals("gemma_4_e2b", json.getString("active_model_id"))
        assertTrue(json.getBoolean("has_last_output"))
        assertEquals("gemma_4_e2b", json.getString("last_output_model_id"))
    }

    @Test
    fun `formats compact runtime summary for adb logcat verification`() {
        val telemetry = VlmRuntimeTelemetry(
            requestedModelId = "gemma_4_e2b",
            activeModelId = "gemma_4_e2b",
            state = VlmRuntimeTelemetry.STATE_ACTIVE,
            reason = "ok",
            availableRamMb = 6200,
            nativeHeapMb = 320,
            promptChars = 941
        ).withOutput(hasLastOutput = true, lastOutputModelId = "gemma_4_e2b")

        assertEquals(
            "state=active, requested=gemma_4_e2b, active=gemma_4_e2b, reason=ok, lastOutput=true/gemma_4_e2b, ramMb=6200, heapMb=320, promptChars=941",
            telemetry.toLogSummary()
        )
    }
}
