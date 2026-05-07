package com.trillboards.ctv.core.stability

import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.audience.DeviceProfile.DeviceCapabilityMatrix
import com.trillboards.ctv.core.audience.DeviceProfile.ModelAvailability

/**
 * Centralized memory attenuation manager with hysteresis and restoration.
 *
 * Implements a tiered approach to reducing memory pressure by selectively unloading
 * ML models and reducing sensing frequency. Uses time-based hysteresis to prevent
 * oscillation between tiers.
 *
 * Enhancement over original:
 * - Reads shed order from DeviceCapabilityMatrix (single source of truth)
 * - Restoration path: when RAM recovers above de-escalation threshold for > 120s,
 *   reloads models in reverse shed order (Whisper last loaded, AgeGender first loaded)
 * - Tier 1 devices (RK3588, 8GB) have attenuation disabled — always NORMAL
 * - Tracks transition events for RecoveryLadder alerting
 *
 * Tier Design:
 * - NORMAL:   All models active, full FPS
 * - MEDIUM:   Shed models per capability matrix (typically AgeGender + FootTraffic)
 * - HIGH:     Also shed Pose + Emotion + Gaze
 * - CRITICAL: Also shed Whisper, keep only Face + YAMNet at 1fps
 */
class MemoryAttenuationManager(
    private val capabilityMatrix: DeviceCapabilityMatrix? = null,
    private val onTierChanged: ((AttenuationTier, AttenuationTier, Map<String, Boolean>) -> Unit)? = null,
    private val onAttenuationEvent: ((String, String) -> Unit)? = null,
    private val zeroAllocPipelineActive: Boolean = false
) {
    enum class AttenuationTier {
        NORMAL,   // All models loaded
        MEDIUM,   // Age/gender + foot traffic unloaded
        HIGH,     // Also pose/emotion/gaze unloaded
        CRITICAL  // Also whisper unloaded
    }

    @Volatile private var currentTier = AttenuationTier.NORMAL
    private var lastEscalationTimeMs = 0L
    private var lastDeescalationTimeMs = 0L

    // Restoration tracking: how long RAM has been below de-escalation threshold
    private var restorationCandidateSinceMs = 0L
    private val restorationHoldTimeMs: Long get() = SensingConfig.get().recovery.restorationHoldTimeMs

    // Model status tracking
    private val modelStatus = mutableMapOf(
        MODEL_FACE to true,
        MODEL_YAMNET to true,
        MODEL_WHISPER to true,
        MODEL_AGE_GENDER to true,
        MODEL_POSE to true,
        MODEL_EMOTION to true,
        MODEL_GAZE to true,
        MODEL_FOOT_TRAFFIC to true
    )

    // Shed order from capability matrix, or default if not provided
    private val shedOrder: List<String> by lazy {
        capabilityMatrix?.getShedOrder() ?: listOf(
            MODEL_FOOT_TRAFFIC, MODEL_AGE_GENDER, MODEL_GAZE, MODEL_POSE, MODEL_EMOTION, MODEL_WHISPER
        )
    }

    // Restoration order is reverse of shed order (restore least-expensive first)
    private val restoreOrder: List<String> by lazy {
        shedOrder.reversed()
    }

    init {
        // If capability matrix says attenuation is disabled (e.g., Tier 1 RK3588 with 8GB),
        // ensure we never leave NORMAL
        if (capabilityMatrix?.attenuationDisabled == true) {
            Log.i(TAG, "[Attenuation] DISABLED by capability matrix (device has sufficient RAM)")
        }
        if (zeroAllocPipelineActive) {
            Log.i(TAG, "[Attenuation] DISABLED — zero-allocation pipeline active, no per-frame memory growth")
        }
    }

    /**
     * Get current attenuation tier.
     * When zero-allocation pipeline is active, always returns NORMAL
     * since there is no per-frame memory growth to attenuate.
     */
    fun getCurrentTier(): AttenuationTier {
        if (zeroAllocPipelineActive) return AttenuationTier.NORMAL
        return currentTier
    }

    /**
     * Get current model status.
     */
    fun getModelStatus(): Map<String, Boolean> = modelStatus.toMap()

    /**
     * Check if a specific model is currently active.
     */
    fun isModelActive(modelName: String): Boolean = modelStatus[modelName] ?: false

    /**
     * Check if we should escalate to a higher tier (more aggressive attenuation).
     * Requires 60s cooldown since last escalation.
     * Returns false if attenuation is disabled by capability matrix.
     */
    fun shouldEscalate(requestedTier: AttenuationTier): Boolean {
        if (capabilityMatrix?.attenuationDisabled == true) return false
        if (requestedTier.ordinal <= currentTier.ordinal) return false
        val now = System.currentTimeMillis()
        val timeSinceLastEscalation = now - lastEscalationTimeMs
        return timeSinceLastEscalation >= SensingConfig.get().recovery.escalationCooldownMs
    }

    /**
     * Check if we should de-escalate to a lower tier (less aggressive attenuation).
     * Requires 120s cooldown since last de-escalation.
     */
    fun shouldDeescalate(requestedTier: AttenuationTier): Boolean {
        if (requestedTier.ordinal >= currentTier.ordinal) return false
        val now = System.currentTimeMillis()
        val timeSinceLastDeescalation = now - lastDeescalationTimeMs
        return timeSinceLastDeescalation >= SensingConfig.get().recovery.deescalationCooldownMs
    }

    /**
     * Check RAM level and trigger restoration if conditions are met.
     * Call this periodically (every 15s from the memory check loop).
     *
     * @param availableRamPercent Current available RAM as percentage (0-100)
     */
    fun checkForRestoration(availableRamPercent: Float) {
        if (currentTier == AttenuationTier.NORMAL) {
            restorationCandidateSinceMs = 0
            return
        }

        // Determine target tier based on current RAM
        val targetTier = tierForRamPercent(availableRamPercent)
        if (targetTier.ordinal >= currentTier.ordinal) {
            // RAM is still pressured, reset restoration timer
            restorationCandidateSinceMs = 0
            return
        }

        val now = System.currentTimeMillis()
        if (restorationCandidateSinceMs == 0L) {
            // Start the restoration hold timer
            restorationCandidateSinceMs = now
            Log.d(TAG, "[Restoration] RAM recovered to ${availableRamPercent.toInt()}% — " +
                    "waiting ${restorationHoldTimeMs / 1000}s for stable recovery before restoring to $targetTier")
            return
        }

        if (now - restorationCandidateSinceMs >= restorationHoldTimeMs) {
            // RAM has been stable for the hold period — restore one tier at a time
            val nextTier = AttenuationTier.values().getOrNull(currentTier.ordinal - 1) ?: return
            Log.i(TAG, "[Restoration] RAM stable at ${availableRamPercent.toInt()}% for " +
                    "${(now - restorationCandidateSinceMs) / 1000}s — restoring to $nextTier")
            transitionTo(nextTier)
            restorationCandidateSinceMs = 0 // Reset for next restoration step
        }
    }

    /**
     * Determine the appropriate tier for a given RAM percentage.
     * Uses thresholds from capability matrix if available.
     */
    private fun tierForRamPercent(availableRamPercent: Float): AttenuationTier {
        // On high-RAM devices (8GB+), thresholds are lower since absolute RAM is higher
        val thresholds = if (capabilityMatrix != null && capabilityMatrix.attenuationDisabled) {
            // Should never reach here, but return NORMAL thresholds
            return AttenuationTier.NORMAL
        } else if (capabilityMatrix != null) {
            capabilityMatrix.attenuationThresholds
        } else {
            // Default thresholds for 4GB devices
            AttenuationThresholds(mediumPercent = 40f, highPercent = 25f, criticalPercent = 15f)
        }

        return when {
            availableRamPercent <= thresholds.criticalPercent -> AttenuationTier.CRITICAL
            availableRamPercent <= thresholds.highPercent -> AttenuationTier.HIGH
            availableRamPercent <= thresholds.mediumPercent -> AttenuationTier.MEDIUM
            else -> AttenuationTier.NORMAL
        }
    }

    /**
     * Transition to a new tier (with hysteresis checks).
     * Returns the actual tier after hysteresis.
     */
    @Synchronized
    fun transitionTo(tier: AttenuationTier): AttenuationTier {
        // Zero-alloc pipeline active — always stay NORMAL (no per-frame memory growth)
        if (zeroAllocPipelineActive && tier != AttenuationTier.NORMAL) {
            Log.d(TAG, "[Attenuation] Transition to $tier blocked — zero-allocation pipeline active")
            return AttenuationTier.NORMAL
        }

        // Attenuation disabled — always stay NORMAL
        if (capabilityMatrix?.attenuationDisabled == true && tier != AttenuationTier.NORMAL) {
            Log.d(TAG, "[Attenuation] Transition to $tier blocked — attenuation disabled by capability matrix")
            return AttenuationTier.NORMAL
        }

        val oldTier = currentTier

        // Check hysteresis
        if (tier.ordinal > currentTier.ordinal) {
            // Escalation
            if (!shouldEscalate(tier)) {
                Log.d(TAG, "[Hysteresis] Escalation to $tier blocked (cooldown: ${SensingConfig.get().recovery.escalationCooldownMs}ms)")
                return currentTier
            }
            lastEscalationTimeMs = System.currentTimeMillis()
        } else if (tier.ordinal < currentTier.ordinal) {
            // De-escalation (restoration)
            if (!shouldDeescalate(tier)) {
                Log.d(TAG, "[Hysteresis] De-escalation to $tier blocked (cooldown: ${SensingConfig.get().recovery.deescalationCooldownMs}ms)")
                return currentTier
            }
            lastDeescalationTimeMs = System.currentTimeMillis()
        } else {
            // No change
            return currentTier
        }

        // Update model status based on tier, using capability matrix shed order
        updateModelStatusFromMatrix(tier)

        // Calculate estimated memory savings
        val estimatedSavingsMB = estimateMemorySavings(oldTier, tier)

        currentTier = tier

        Log.i(TAG, "[Attenuation] Tier transition: $oldTier → $tier (est. ${estimatedSavingsMB}MB freed)")
        Log.d(TAG, "[Attenuation] Model status: $modelStatus")

        // Notify listeners
        onTierChanged?.invoke(oldTier, tier, modelStatus.toMap())

        // Notify RecoveryLadder for transition counting
        onAttenuationEvent?.invoke(oldTier.name, tier.name)

        return tier
    }

    /**
     * Update model status using the capability matrix's model availability rules.
     * Falls back to hardcoded tiers if no matrix is provided.
     */
    private fun updateModelStatusFromMatrix(tier: AttenuationTier) {
        if (capabilityMatrix != null) {
            // Use capability matrix: each model has a shed tier (ALWAYS_ON, SHED_AT_MEDIUM, etc.)
            for ((modelName, _) in modelStatus) {
                val availability = capabilityMatrix.getModelAvailability(modelName)
                modelStatus[modelName] = when (availability) {
                    ModelAvailability.ALWAYS_ON -> true
                    ModelAvailability.SHED_AT_MEDIUM -> tier.ordinal < AttenuationTier.MEDIUM.ordinal
                    ModelAvailability.SHED_AT_HIGH -> tier.ordinal < AttenuationTier.HIGH.ordinal
                    ModelAvailability.OFF -> false
                }
            }
            // Face and YAMNet are always on (core sensors)
            modelStatus[MODEL_FACE] = true
            modelStatus[MODEL_YAMNET] = true
        } else {
            // Fallback: hardcoded tiers (original behavior)
            updateModelStatusLegacy(tier)
        }
    }

    /**
     * Legacy model status update (used when no capability matrix is provided).
     */
    private fun updateModelStatusLegacy(tier: AttenuationTier) {
        when (tier) {
            AttenuationTier.NORMAL -> {
                modelStatus[MODEL_FACE] = true
                modelStatus[MODEL_YAMNET] = true
                modelStatus[MODEL_WHISPER] = true
                modelStatus[MODEL_AGE_GENDER] = true
                modelStatus[MODEL_POSE] = true
                modelStatus[MODEL_EMOTION] = true
                modelStatus[MODEL_GAZE] = true
                modelStatus[MODEL_FOOT_TRAFFIC] = true
            }
            AttenuationTier.MEDIUM -> {
                modelStatus[MODEL_FACE] = true
                modelStatus[MODEL_YAMNET] = true
                modelStatus[MODEL_WHISPER] = true
                modelStatus[MODEL_AGE_GENDER] = false
                modelStatus[MODEL_POSE] = true
                modelStatus[MODEL_EMOTION] = true
                modelStatus[MODEL_GAZE] = true
                modelStatus[MODEL_FOOT_TRAFFIC] = false
            }
            AttenuationTier.HIGH -> {
                modelStatus[MODEL_FACE] = true
                modelStatus[MODEL_YAMNET] = true
                modelStatus[MODEL_WHISPER] = true
                modelStatus[MODEL_AGE_GENDER] = false
                modelStatus[MODEL_POSE] = false
                modelStatus[MODEL_EMOTION] = false
                modelStatus[MODEL_GAZE] = false
                modelStatus[MODEL_FOOT_TRAFFIC] = false
            }
            AttenuationTier.CRITICAL -> {
                modelStatus[MODEL_FACE] = true
                modelStatus[MODEL_YAMNET] = true
                modelStatus[MODEL_WHISPER] = false
                modelStatus[MODEL_AGE_GENDER] = false
                modelStatus[MODEL_POSE] = false
                modelStatus[MODEL_EMOTION] = false
                modelStatus[MODEL_GAZE] = false
                modelStatus[MODEL_FOOT_TRAFFIC] = false
            }
        }
    }

    /**
     * Estimate memory savings from tier transition.
     */
    private fun estimateMemorySavings(oldTier: AttenuationTier, newTier: AttenuationTier): Int {
        var savings = 0
        val oldModels = getModelsForTier(oldTier)
        val newModels = getModelsForTier(newTier)

        // Calculate freed models using companion object MODEL_SIZES_MB
        for ((model, size) in MODEL_SIZES_MB) {
            if (oldModels.contains(model) && !newModels.contains(model)) {
                savings += size
            } else if (!oldModels.contains(model) && newModels.contains(model)) {
                savings -= size  // Negative if re-loading
            }
        }

        return savings
    }

    /**
     * Get list of active models for a tier.
     */
    private fun getModelsForTier(tier: AttenuationTier): Set<String> {
        return when (tier) {
            AttenuationTier.NORMAL -> setOf(MODEL_WHISPER, MODEL_AGE_GENDER, MODEL_POSE, MODEL_EMOTION, MODEL_GAZE, MODEL_FOOT_TRAFFIC)
            AttenuationTier.MEDIUM -> setOf(MODEL_WHISPER, MODEL_POSE, MODEL_EMOTION, MODEL_GAZE)
            AttenuationTier.HIGH -> setOf(MODEL_WHISPER)
            AttenuationTier.CRITICAL -> emptySet()
        }
    }

    /**
     * Get a snapshot of the current attenuation state for telemetry.
     */
    fun getAttenuationSnapshot(): Map<String, Any> {
        return mapOf(
            "tier" to getCurrentTier().name,
            "models" to modelStatus.toMap(),
            "shedOrder" to shedOrder,
            "attenuationDisabled" to (capabilityMatrix?.attenuationDisabled ?: false),
            "zeroAllocPipelineActive" to zeroAllocPipelineActive,
            "restorationPending" to (restorationCandidateSinceMs > 0)
        )
    }

    companion object {
        private const val TAG = "MemoryAttenuation"

        // Model name constants
        const val MODEL_FACE = "face"
        const val MODEL_YAMNET = "yamnet"
        const val MODEL_WHISPER = "whisper"
        const val MODEL_AGE_GENDER = "ageGender"
        const val MODEL_POSE = "pose"
        const val MODEL_EMOTION = "emotion"
        const val MODEL_GAZE = "gaze"
        const val MODEL_FOOT_TRAFFIC = "footTraffic"

        // Model size estimates (MB) — single source of truth
        val MODEL_SIZES_MB = mapOf(
            MODEL_WHISPER to 100,      // Whisper Tiny TFLite model + runtime
            MODEL_AGE_GENDER to 20,    // Age/gender TFLite model
            MODEL_POSE to 15,          // MediaPipe Pose Lite
            MODEL_EMOTION to 10,       // FER TFLite model
            MODEL_GAZE to 5,           // Gaze tracking (minimal)
            MODEL_FOOT_TRAFFIC to 5    // Foot traffic estimator (minimal)
        )
    }
}

/**
 * RAM threshold percentages for attenuation tier transitions.
 * Different device tiers have different thresholds (e.g., 8GB devices use lower
 * percentages since absolute RAM is much higher).
 */
data class AttenuationThresholds(
    val mediumPercent: Float = 40f,
    val highPercent: Float = 25f,
    val criticalPercent: Float = 15f
)
