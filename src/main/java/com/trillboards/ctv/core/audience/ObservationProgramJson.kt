package com.trillboards.ctv.core.audience

import org.json.JSONObject

data class ObservationProgramMetadata(
    val programSpecJson: String? = null,
    val programSpecVersion: String? = null
)

data class ObservationWorkerContract(
    val key: String,
    val worker: String,
    val enabled: Boolean,
    val modelIds: List<String> = emptyList(),
    val outputs: List<String> = emptyList(),
    val primaryFocus: String? = null
)

data class ObservationProgramRuntimeContract(
    val objective: String? = null,
    val staticSemantics: ObservationWorkerContract? = null,
    val temporalSemantics: ObservationWorkerContract? = null,
    val speechSemantics: ObservationWorkerContract? = null,
    val physicalCorroboration: ObservationWorkerContract? = null,
    val fusionMode: String? = null,
    val joinWindowMs: Long? = null
) {
    fun enabledWorkerModelIds(): List<String> =
        listOfNotNull(staticSemantics, temporalSemantics, speechSemantics, physicalCorroboration)
            .filter { it.enabled }
            .flatMap { it.modelIds }
            .distinct()

    fun hasExplicitWorkerModels(): Boolean =
        listOfNotNull(staticSemantics, temporalSemantics, speechSemantics, physicalCorroboration)
            .any { it.modelIds.isNotEmpty() }

    fun temporalPromptFallback(): String? {
        val focus = temporalSemantics?.primaryFocus
        val outputs = temporalSemantics?.outputs.orEmpty()
        if (objective.isNullOrBlank() && focus.isNullOrBlank() && outputs.isEmpty()) {
            return null
        }

        val parts = mutableListOf<String>()
        objective?.takeIf { it.isNotBlank() }?.let { parts += "Observation objective: $it." }
        focus?.takeIf { it.isNotBlank() }?.let { parts += "Temporal focus: $it." }
        if (outputs.isNotEmpty()) {
            parts += "Return JSON with: ${outputs.joinToString(", ")}."
        }
        return parts.joinToString(" ")
    }
}

internal fun extractObservationProgramMetadata(payload: JSONObject?): ObservationProgramMetadata {
    if (payload == null) {
        return ObservationProgramMetadata()
    }

    val programSpecJson = ObservationProgramJson.extractCanonicalString(payload)
    return ObservationProgramMetadata(
        programSpecJson = programSpecJson,
        programSpecVersion = ObservationProgramJson.extractVersion(payload, programSpecJson)
    )
}

internal fun parseObservationProgramJson(programSpecJson: String?): JSONObject? =
    ObservationProgramJson.parse(programSpecJson)

internal fun parseObservationProgramRuntimeContract(programSpecJson: String?): ObservationProgramRuntimeContract? =
    ObservationProgramJson.parseRuntimeContract(programSpecJson)

internal fun appendObservationProgramMetadata(
    target: JSONObject,
    programSpecJson: String?,
    programSpecVersion: String?
) {
    val parsedProgramSpec = ObservationProgramJson.parse(programSpecJson)
    if (parsedProgramSpec != null) {
        target.put("program_spec", parsedProgramSpec)
    }

    firstNonBlank(
        programSpecVersion,
        parsedProgramSpec?.opt("spec_version")?.toString()?.takeIf { it.isNotBlank() },
        parsedProgramSpec?.opt("specVersion")?.toString()?.takeIf { it.isNotBlank() }
    )?.let { version ->
        target.put("program_spec_version", version)
    }
}

internal object ObservationProgramJson {
    private const val PROGRAM_SPEC = "program_spec"
    private const val PROGRAM_SPEC_CAMEL = "programSpec"
    private const val PROGRAM_SPEC_VERSION = "program_spec_version"
    private const val PROGRAM_SPEC_VERSION_CAMEL = "programSpecVersion"
    private const val SPEC_VERSION = "spec_version"
    private const val SPEC_VERSION_CAMEL = "specVersion"

    fun extractJson(payload: JSONObject): JSONObject? {
        return payload.optJSONObject(PROGRAM_SPEC)
            ?: payload.optJSONObject(PROGRAM_SPEC_CAMEL)
    }

    fun extractCanonicalString(payload: JSONObject): String? {
        return extractJson(payload)?.toString()
    }

    fun extractVersion(payload: JSONObject, rawProgramSpecJson: String? = null): String? {
        coerceVersion(payload.opt(PROGRAM_SPEC_VERSION))?.let { return it }
        coerceVersion(payload.opt(PROGRAM_SPEC_VERSION_CAMEL))?.let { return it }

        val programSpec = parse(rawProgramSpecJson) ?: extractJson(payload)
        coerceVersion(programSpec?.opt(SPEC_VERSION))?.let { return it }
        coerceVersion(programSpec?.opt(SPEC_VERSION_CAMEL))?.let { return it }

        return null
    }

    fun parse(rawProgramSpecJson: String?): JSONObject? {
        if (rawProgramSpecJson.isNullOrBlank()) {
            return null
        }
        return try {
            JSONObject(rawProgramSpecJson)
        } catch (_: Exception) {
            null
        }
    }

    fun parseRuntimeContract(rawProgramSpecJson: String?): ObservationProgramRuntimeContract? {
        val programSpec = parse(rawProgramSpecJson) ?: return null
        val inferenceContract = programSpec.optJSONObject("inference_contract") ?: return null

        return ObservationProgramRuntimeContract(
            objective = programSpec.optJSONObject("observation_program")?.optString("objective")?.takeIf { it.isNotBlank() },
            staticSemantics = parseWorkerContract("static_semantics", inferenceContract.optJSONObject("static_semantics"), parseOutputs = false),
            temporalSemantics = parseWorkerContract("temporal_semantics", inferenceContract.optJSONObject("temporal_semantics"), parseOutputs = true),
            speechSemantics = parseWorkerContract("speech_semantics", inferenceContract.optJSONObject("speech_semantics"), parseOutputs = false),
            physicalCorroboration = parseWorkerContract("physical_corroboration", inferenceContract.optJSONObject("physical_corroboration"), parseOutputs = false),
            fusionMode = programSpec.optJSONObject("fusion_contract")?.optString("mode")?.takeIf { it.isNotBlank() },
            joinWindowMs = programSpec.optJSONObject("fusion_contract")
                ?.opt("join_window_ms")
                ?.toString()
                ?.toLongOrNull()
        )
    }

    private fun coerceVersion(value: Any?): String? {
        return when (value) {
            null, JSONObject.NULL -> null
            is String -> value.takeIf { it.isNotBlank() }
            else -> value.toString()
        }
    }

    private fun parseWorkerContract(key: String, workerJson: JSONObject?, parseOutputs: Boolean = false): ObservationWorkerContract? {
        if (workerJson == null) {
            return null
        }

        return ObservationWorkerContract(
            key = key,
            worker = workerJson.optString("worker").takeIf { it.isNotBlank() } ?: key,
            enabled = workerJson.optBoolean("enabled", false),
            modelIds = jsonStringList(workerJson.optJSONArray("model_ids")),
            outputs = if (parseOutputs) jsonStringList(workerJson.optJSONArray("outputs")) else emptyList(),
            primaryFocus = workerJson.optString("primary_focus").takeIf { it.isNotBlank() }
        )
    }

    private fun jsonStringList(array: org.json.JSONArray?): List<String> {
        if (array == null) return emptyList()
        return buildList {
            for (index in 0 until array.length()) {
                array.optString(index)
                    ?.takeIf { it.isNotBlank() }
                    ?.let(::add)
            }
        }
    }
}

private fun firstNonBlank(vararg values: String?): String? =
    values.firstOrNull { !it.isNullOrBlank() }
