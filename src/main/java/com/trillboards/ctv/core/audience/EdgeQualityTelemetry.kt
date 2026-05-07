package com.trillboards.ctv.core.audience

import org.json.JSONArray
import org.json.JSONObject

data class CaptureDiagnostics(
    val reason: String? = null,
    val shouldCaptureDiagnostic: Boolean = false,
    val captureReady: Boolean = false,
    val captureInProgress: Boolean = false,
    val nextAllowedCaptureInMs: Long = 0L
) {
    fun toJson(): JSONObject = JSONObject().apply {
        reason?.let { put("reason", it) }
        put("shouldCaptureDiagnostic", shouldCaptureDiagnostic)
        put("captureReady", captureReady)
        put("captureInProgress", captureInProgress)
        put("nextAllowedCaptureInMs", nextAllowedCaptureInMs)
    }
}

data class EdgePerformanceSnapshot(
    val actualFps: Float = 0f,
    val targetFps: Int = 0,
    val skippedFrames: Long = 0,
    val skipRatio: Float = 0f,
    val avgInferenceMs: Double = 0.0,
    val lastInferenceMs: Long = 0L,
    val webviewFps: Double = -1.0
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("actualFps", actualFps)
        put("targetFps", targetFps)
        put("skippedFrames", skippedFrames)
        put("skipRatio", skipRatio)
        put("avgInferenceMs", avgInferenceMs)
        put("lastInferenceMs", lastInferenceMs)
        if (webviewFps >= 0) put("webviewFps", webviewFps)
    }
}

data class EdgeMemorySnapshot(
    val availableRamMb: Int,
    val totalRamMb: Int,
    val thresholdMb: Int,
    val javaHeapMb: Int,
    val javaHeapMaxMb: Int,
    val nativeHeapMb: Int,
    val pressure: String,
    val lowMemory: Boolean,
    val thermalStatus: Int? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("availableRamMb", availableRamMb)
        put("totalRamMb", totalRamMb)
        put("thresholdMb", thresholdMb)
        put("javaHeapMb", javaHeapMb)
        put("javaHeapMaxMb", javaHeapMaxMb)
        put("nativeHeapMb", nativeHeapMb)
        put("pressure", pressure)
        put("lowMemory", lowMemory)
        thermalStatus?.let { put("thermalStatus", it) }
    }
}

data class EdgeSensingSnapshot(
    val currentFaceCount: Int = 0,
    val currentPersonCount: Int = 0,
    val avgFaceCountWindow: Double = 0.0,
    val avgPersonCountWindow: Double = 0.0,
    val maxFaceCountWindow: Int = 0,
    val estimatedOccupancy: String? = null,
    val speechSignalsPresent: Boolean = false
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("currentFaceCount", currentFaceCount)
        put("currentPersonCount", currentPersonCount)
        put("avgFaceCountWindow", avgFaceCountWindow)
        put("avgPersonCountWindow", avgPersonCountWindow)
        put("maxFaceCountWindow", maxFaceCountWindow)
        estimatedOccupancy?.let { put("estimatedOccupancy", it) }
        put("speechSignalsPresent", speechSignalsPresent)
    }
}

data class EdgeQualityTelemetry(
    val measurementQuality: String,
    val observationFamily: String,
    val evidenceGrade: String,
    val decisionability: String,
    val decisionBlockReasons: List<String> = emptyList(),
    val reliability: String,
    val proxySignalsPresent: Boolean,
    val proxySignalMismatch: Boolean,
    val attenuationTier: String,
    val performance: EdgePerformanceSnapshot,
    val memory: EdgeMemorySnapshot,
    val sensing: EdgeSensingSnapshot,
    val captureDiagnostics: CaptureDiagnostics = CaptureDiagnostics(),
    val activeModels: Map<String, Boolean> = emptyMap(),
    val zeroAllocPipelineActive: Boolean? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("measurementQuality", measurementQuality)
        put("observationFamily", observationFamily)
        put("evidenceGrade", evidenceGrade)
        put("decisionability", decisionability)
        put("decisionBlockReasons", JSONArray(decisionBlockReasons))
        put("reliability", reliability)
        put("proxySignalsPresent", proxySignalsPresent)
        put("proxySignalMismatch", proxySignalMismatch)
        put("attenuationTier", attenuationTier)
        put("performance", performance.toJson())
        put("memory", memory.toJson())
        put("sensing", sensing.toJson())
        put("captureDiagnostics", captureDiagnostics.toJson())
        if (activeModels.isNotEmpty()) {
            put("activeModels", JSONObject(activeModels))
        }
        zeroAllocPipelineActive?.let { put("zeroAllocPipelineActive", it) }
    }
}
