package com.trillboards.ctv.core.audience

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ObservationProgramJsonTest {

    @Test
    fun `extracts program spec metadata from snake case payload`() {
        val payload = JSONObject(
            """
            {
              "program_spec": {
                "spec_version": "1.2.3",
                "observation_program": {
                  "objective": "measure speech"
                }
              }
            }
            """.trimIndent()
        )

        val metadata = extractObservationProgramMetadata(payload)
        val parsedProgramSpec = parseObservationProgramJson(metadata.programSpecJson)

        assertEquals("1.2.3", metadata.programSpecVersion)
        assertNotNull(parsedProgramSpec)
        assertEquals(
            "measure speech",
            parsedProgramSpec
                ?.getJSONObject("observation_program")
                ?.getString("objective")
        )
    }

    @Test
    fun `prefers explicit program spec version and handles camel case payloads`() {
        val payload = JSONObject(
            """
            {
              "programSpecVersion": "9.9.9",
              "programSpec": {
                "spec_version": "1.0.0",
                "deployment_context": {
                  "source": "fein"
                }
              }
            }
            """.trimIndent()
        )

        val metadata = extractObservationProgramMetadata(payload)
        val parsedProgramSpec = parseObservationProgramJson(metadata.programSpecJson)

        assertEquals("9.9.9", metadata.programSpecVersion)
        assertEquals("fein", parsedProgramSpec?.getJSONObject("deployment_context")?.getString("source"))
    }

    @Test
    fun `returns null metadata when program spec is absent`() {
        val metadata = extractObservationProgramMetadata(JSONObject("""{"profile_name":"custom"}"""))

        assertNull(metadata.programSpecJson)
        assertNull(metadata.programSpecVersion)
        assertNull(parseObservationProgramJson(metadata.programSpecJson))
    }

    @Test
    fun `parses runtime worker contracts from program spec`() {
        val payload = JSONObject(
            """
            {
              "program_spec": {
                "spec_version": "2.0.0",
                "observation_program": {
                  "objective": "measure comparative shopper intent"
                },
                "inference_contract": {
                  "temporal_semantics": {
                    "worker": "edge_temporal_semantics",
                    "enabled": true,
                    "model_ids": ["gemma_4_e2b"],
                    "outputs": ["scene_type_transition", "event_phase"],
                    "primary_focus": "change_over_time"
                  },
                  "speech_semantics": {
                    "worker": "speech_semantics",
                    "enabled": false,
                    "model_ids": ["whisper_tiny"]
                  },
                  "physical_corroboration": {
                    "worker": "physical_corroboration",
                    "enabled": false,
                    "model_ids": ["yamnet", "movenet"]
                  }
                },
                "fusion_contract": {
                  "mode": "scene_primary",
                  "join_window_ms": 30000
                }
              }
            }
            """.trimIndent()
        )

        val metadata = extractObservationProgramMetadata(payload)
        val contract = parseObservationProgramRuntimeContract(metadata.programSpecJson)

        assertNotNull(contract)
        assertEquals("measure comparative shopper intent", contract?.objective)
        assertEquals("scene_primary", contract?.fusionMode)
        assertEquals(30000L, contract?.joinWindowMs)
        assertTrue(contract?.temporalSemantics?.enabled == true)
        assertEquals(listOf("gemma_4_e2b"), contract?.temporalSemantics?.modelIds)
        assertFalse(contract?.speechSemantics?.enabled == true)
        assertFalse(contract?.physicalCorroboration?.enabled == true)
        assertEquals(listOf("gemma_4_e2b"), contract?.enabledWorkerModelIds())
        assertTrue(contract?.hasExplicitWorkerModels() == true)
    }
}
