package com.trillboards.ctv.core.inference

import android.util.Log

/**
 * Registry that manages [InferenceProcessor] instances by their [InferenceProcessor.modelId].
 *
 * Provides discovery, lifecycle management, and memory budgeting for all
 * registered processors. Used by AudienceSensingService to:
 * 1. Register available processors at startup
 * 2. Enable/disable processors based on server-side SensingConfig
 * 3. Query total memory footprint before enabling new processors
 * 4. Release all processors on shutdown
 *
 * ## Thread Safety
 * All mutations are synchronized. Read operations ([get], [getAll]) return
 * snapshots — callers may hold references to processors that are later
 * removed via [unregister] or [releaseAll].
 *
 * ## Usage
 * ```kotlin
 * val registry = ProcessorRegistry()
 *
 * // Register processors
 * registry.register(MyVlmProcessor())
 * registry.register(MyAudioProcessor())
 *
 * // Get processors enabled by server config
 * val enabled = registry.enabledProcessors(config.enabledModelIds)
 *
 * // Initialize only enabled processors
 * enabled.forEach { it.initialize() }
 *
 * // Check memory budget
 * if (registry.totalMemoryMb() > MAX_MEMORY_MB) { ... }
 *
 * // Shutdown
 * registry.releaseAll()
 * ```
 */
class ProcessorRegistry {
    companion object {
        private const val TAG = "ProcessorRegistry"
    }

    private val processors = mutableMapOf<String, InferenceProcessor>()
    private val lock = Any()

    /**
     * Register a processor. If a processor with the same [InferenceProcessor.modelId]
     * is already registered, the old one is released and replaced.
     *
     * @param processor The processor to register.
     */
    fun register(processor: InferenceProcessor) {
        synchronized(lock) {
            val existing = processors[processor.modelId]
            if (existing != null) {
                Log.w(TAG, "Replacing existing processor: ${processor.modelId}")
                try {
                    existing.release()
                } catch (e: Exception) {
                    Log.e(TAG, "Error releasing replaced processor: ${processor.modelId}", e)
                }
            }
            processors[processor.modelId] = processor
            Log.i(TAG, "Registered processor: ${processor.modelId} " +
                "(hardware=${processor.hardwareRequirement})")
        }
    }

    /**
     * Unregister and release a processor by modelId.
     *
     * @param modelId The model ID to unregister.
     * @return true if a processor was found and removed.
     */
    fun unregister(modelId: String): Boolean {
        synchronized(lock) {
            val processor = processors.remove(modelId)
            if (processor != null) {
                try {
                    processor.release()
                } catch (e: Exception) {
                    Log.e(TAG, "Error releasing unregistered processor: $modelId", e)
                }
                Log.i(TAG, "Unregistered processor: $modelId")
                return true
            }
            return false
        }
    }

    /**
     * Get a processor by its modelId.
     *
     * @param modelId The unique model identifier.
     * @return The processor, or null if not registered.
     */
    fun get(modelId: String): InferenceProcessor? {
        synchronized(lock) {
            return processors[modelId]
        }
    }

    /**
     * Get all registered processors.
     *
     * @return Snapshot list of all registered processors.
     */
    fun getAll(): List<InferenceProcessor> {
        synchronized(lock) {
            return processors.values.toList()
        }
    }

    /**
     * Get processors whose modelId is in the given list.
     *
     * Used to filter processors based on server-side SensingConfig's enabled
     * model list. Processors not in [modelIds] are excluded but not released.
     *
     * @param modelIds List of model IDs to include.
     * @return Processors matching the given model IDs, in the order they appear
     *   in [modelIds]. Unknown model IDs are silently skipped.
     */
    fun enabledProcessors(modelIds: List<String>): List<InferenceProcessor> {
        synchronized(lock) {
            return modelIds.mapNotNull { id -> processors[id] }
        }
    }

    /**
     * Get processors filtered by hardware requirement.
     *
     * Useful for filtering processors based on available device hardware
     * (e.g., skip all CAMERA processors on devices without cameras).
     *
     * @param requirement The hardware requirement to filter by.
     * @return Processors matching the given hardware requirement.
     */
    fun getByHardware(requirement: HardwareRequirement): List<InferenceProcessor> {
        synchronized(lock) {
            return processors.values.filter { it.hardwareRequirement == requirement }
        }
    }

    /**
     * Get all processors that are currently ready (initialized and not released).
     *
     * @return List of processors where [InferenceProcessor.isReady] returns true.
     */
    fun readyProcessors(): List<InferenceProcessor> {
        synchronized(lock) {
            return processors.values.filter { it.isReady() }
        }
    }

    /**
     * Release all registered processors and clear the registry.
     *
     * Should be called during AudienceSensingService shutdown. After this call,
     * [getAll] returns an empty list and [totalMemoryMb] returns 0.
     */
    fun releaseAll() {
        synchronized(lock) {
            var released = 0
            var errors = 0
            for ((modelId, processor) in processors) {
                try {
                    processor.release()
                    released++
                } catch (e: Exception) {
                    Log.e(TAG, "Error releasing processor: $modelId", e)
                    errors++
                }
            }
            processors.clear()
            Log.i(TAG, "Released all processors: $released succeeded, $errors errors")
        }
    }

    /**
     * Total estimated memory footprint of all initialized processors in MB.
     *
     * Only counts processors where [InferenceProcessor.isReady] is true,
     * since uninitialized processors haven't allocated their model memory.
     *
     * @return Sum of [InferenceProcessor.getMemoryFootprintMb] for ready processors.
     */
    fun totalMemoryMb(): Float {
        synchronized(lock) {
            return processors.values
                .filter { it.isReady() }
                .map { it.getMemoryFootprintMb() }
                .sum()
        }
    }

    /**
     * Number of registered processors.
     */
    fun size(): Int {
        synchronized(lock) {
            return processors.size
        }
    }

    /**
     * Check if a processor with the given modelId is registered.
     */
    fun contains(modelId: String): Boolean {
        synchronized(lock) {
            return processors.containsKey(modelId)
        }
    }

    /**
     * Initialize all registered processors that have their model available.
     *
     * Skips processors where [InferenceProcessor.hasModel] returns false.
     * Returns the number of processors successfully initialized.
     *
     * @return Number of processors that were successfully initialized.
     */
    fun initializeAll(): Int {
        synchronized(lock) {
            var initialized = 0
            for ((modelId, processor) in processors) {
                if (processor.isReady()) {
                    initialized++
                    continue
                }
                if (!processor.hasModel()) {
                    Log.w(TAG, "Skipping $modelId: model not available")
                    continue
                }
                try {
                    if (processor.initialize()) {
                        initialized++
                        Log.i(TAG, "Initialized $modelId " +
                            "(~${processor.getMemoryFootprintMb()} MB)")
                    } else {
                        Log.e(TAG, "Failed to initialize $modelId")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Exception initializing $modelId", e)
                }
            }
            Log.i(TAG, "Initialization complete: $initialized/${processors.size} ready, " +
                "total memory: ${totalMemoryMb()} MB")
            return initialized
        }
    }
}
