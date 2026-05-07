package com.trillboards.ctv.core

import android.util.Log
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicReference

/**
 * Thread-safe singleton holding dynamic VAS weights received from the server.
 *
 * Weights are pushed via config.push socket event and cached in-memory.
 * Falls back to hardcoded defaults if no server weights are available.
 *
 * Lookup chain: exact(venueType+daypart) -> venue-only -> defaults
 */
object VASWeightConfig {
    private const val TAG = "VASWeightConfig"

    data class Weights(
        val attention: Float = 0.4f,
        val emotion: Float = 0.3f,
        val body: Float = 0.2f,
        val focus: Float = 0.1f
    )

    private val weightsByVenue = AtomicReference<Map<String, Map<String, Weights>>>(emptyMap())
    private val defaults = Weights()

    /**
     * Update weights from server payload.
     * Called when a config.push with type "vas_weights_update" is received.
     *
     * @param weightsByVenueJson Map of venueType -> Map of daypart -> weight values
     */
    fun updateFromServer(weightsByVenueJson: Map<String, Map<String, Weights>>) {
        weightsByVenue.set(weightsByVenueJson)
        Log.i(TAG, "Updated VAS weights: ${weightsByVenueJson.size} venue types configured")
    }

    /**
     * Parse and update weights from a JSONObject payload.
     *
     * Expected format:
     * { "transit": { "morning": { "attention_weight": 0.45, ... } } }
     */
    fun updateFromJson(json: JSONObject) {
        val parsed = mutableMapOf<String, Map<String, Weights>>()
        val venueTypes = json.keys()
        while (venueTypes.hasNext()) {
            val venueType = venueTypes.next()
            val daypartObj = json.optJSONObject(venueType) ?: continue
            val daypartMap = mutableMapOf<String, Weights>()
            val dayparts = daypartObj.keys()
            while (dayparts.hasNext()) {
                val daypart = dayparts.next()
                val wObj = daypartObj.optJSONObject(daypart) ?: continue
                daypartMap[daypart] = Weights(
                    attention = wObj.optDouble("attention_weight", 0.4).toFloat(),
                    emotion = wObj.optDouble("emotion_weight", 0.3).toFloat(),
                    body = wObj.optDouble("body_weight", 0.2).toFloat(),
                    focus = wObj.optDouble("focus_weight", 0.1).toFloat()
                )
            }
            if (daypartMap.isNotEmpty()) {
                parsed[venueType] = daypartMap
            }
        }
        updateFromServer(parsed)
    }

    /**
     * Get VAS weights for a given venue type and daypart.
     *
     * Fallback chain:
     * 1. Exact match: venueType + daypart
     * 2. Venue-only: venueType + "default" daypart
     * 3. Global defaults: hardcoded 0.4/0.3/0.2/0.1
     */
    fun getWeights(venueType: String?, daypart: String?): Weights {
        val map = weightsByVenue.get()
        if (map.isEmpty() || venueType == null) return defaults

        val venueDayparts = map[venueType] ?: return defaults

        // Try exact match first
        if (daypart != null) {
            venueDayparts[daypart]?.let { return it }
        }

        // Try "default" daypart for this venue
        venueDayparts["default"]?.let { return it }

        // Fall back to global defaults
        return defaults
    }

    /**
     * Check if server-pushed weights are loaded.
     */
    fun hasServerWeights(): Boolean = weightsByVenue.get().isNotEmpty()
}
