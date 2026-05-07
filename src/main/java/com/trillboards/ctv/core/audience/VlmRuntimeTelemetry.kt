package com.trillboards.ctv.core.audience

import org.json.JSONObject

internal data class VlmRuntimeTelemetry(
    val requestedModelId: String? = null,
    val activeModelId: String? = null,
    val state: String = STATE_DISABLED,
    val reason: String? = null,
    val requiredAvailableRamMb: Int? = null,
    val estimatedRuntimeFootprintMb: Int? = null,
    val availableRamMb: Int? = null,
    val nativeHeapMb: Int? = null,
    val promptChars: Int? = null,
    val hasLastOutput: Boolean = false,
    val lastOutputModelId: String? = null
) {
    fun withOutput(hasLastOutput: Boolean, lastOutputModelId: String?): VlmRuntimeTelemetry {
        return copy(hasLastOutput = hasLastOutput, lastOutputModelId = lastOutputModelId)
    }

    fun toJson(): JSONObject {
        return JSONObject().apply {
            putNullable("requested_model_id", requestedModelId)
            putNullable("active_model_id", activeModelId)
            put("state", state)
            putNullable("reason", reason)
            putNullable("required_available_ram_mb", requiredAvailableRamMb)
            putNullable("estimated_runtime_footprint_mb", estimatedRuntimeFootprintMb)
            putNullable("available_ram_mb", availableRamMb)
            putNullable("native_heap_mb", nativeHeapMb)
            putNullable("prompt_chars", promptChars)
            put("has_last_output", hasLastOutput)
            putNullable("last_output_model_id", lastOutputModelId)
        }
    }

    fun toLogSummary(): String {
        return listOf(
            "state=$state",
            "requested=${requestedModelId ?: "none"}",
            "active=${activeModelId ?: "none"}",
            "reason=${reason ?: "none"}",
            "lastOutput=$hasLastOutput/${lastOutputModelId ?: "none"}",
            "ramMb=${availableRamMb ?: -1}",
            "heapMb=${nativeHeapMb ?: -1}",
            "promptChars=${promptChars ?: -1}"
        ).joinToString(", ")
    }

    private fun JSONObject.putNullable(key: String, value: Any?) {
        put(key, value ?: JSONObject.NULL)
    }

    companion object {
        const val STATE_DISABLED = "disabled"
        const val STATE_BLOCKED = "blocked"
        const val STATE_DOWNLOADING = "downloading"
        const val STATE_DOWNLOAD_IN_FLIGHT = "download_in_flight"
        const val STATE_ACTIVE = "active"
        const val STATE_INITIALIZING = "initializing"
        const val STATE_INITIALIZE_FAILED = "initialize_failed"
        const val STATE_MANIFEST_FETCH_FAILED = "manifest_fetch_failed"
        const val STATE_MANIFEST_MISSING_MODEL = "manifest_missing_model"
        const val STATE_DOWNLOAD_FAILED = "download_failed"
    }
}
