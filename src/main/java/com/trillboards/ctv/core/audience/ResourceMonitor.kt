package com.trillboards.ctv.core.audience

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Resource monitor for adaptive sensing under memory pressure.
 *
 * Monitors memory usage and triggers callbacks when pressure is detected.
 * This enables graceful degradation - reducing FPS or resolution when
 * the device is under memory stress.
 */
class ResourceMonitor(
    private val context: Context,
    private val checkIntervalMs: Long = SensingConfig.get().memory.checkIntervalMs
) {
    companion object {
        private const val TAG = "ResourceMonitor"
    }

    /**
     * Memory pressure levels.
     */
    enum class MemoryPressure {
        LOW,       // < 25% used - full features
        MEDIUM,    // 25-50% used - reduce non-essential
        HIGH,      // 50-75% used - reduce aggressively
        CRITICAL   // > 75% used or lowMemory flag - minimal operation
    }

    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    private var monitorJob: Job? = null
    private var lastPressure = MemoryPressure.LOW

    // Callbacks
    var onMemoryPressureChanged: ((MemoryPressure) -> Unit)? = null
    var onCriticalMemory: (() -> Unit)? = null

    /**
     * Get current memory pressure level with hysteresis.
     * De-escalation requires 5% more headroom than escalation to prevent oscillation.
     */
    fun getMemoryPressure(): MemoryPressure {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memInfo)

        // Check system low memory flag first
        if (memInfo.lowMemory) {
            lastPressure = MemoryPressure.CRITICAL
            return MemoryPressure.CRITICAL
        }

        // Calculate available memory ratio
        val availableRatio = memInfo.availMem.toFloat() / memInfo.totalMem

        // Hysteresis: de-escalation thresholds are 5% higher than escalation
        val mem = SensingConfig.get().memory
        val newPressure = when (lastPressure) {
            MemoryPressure.CRITICAL -> when {
                availableRatio >= mem.criticalDeescalationThreshold -> MemoryPressure.HIGH
                else -> MemoryPressure.CRITICAL
            }
            MemoryPressure.HIGH -> when {
                availableRatio < mem.criticalEscalationThreshold -> MemoryPressure.CRITICAL
                availableRatio >= mem.highDeescalationThreshold -> MemoryPressure.MEDIUM
                else -> MemoryPressure.HIGH
            }
            MemoryPressure.MEDIUM -> when {
                availableRatio < mem.criticalEscalationThreshold -> MemoryPressure.CRITICAL
                availableRatio < mem.highEscalationThreshold -> MemoryPressure.HIGH
                availableRatio >= mem.mediumDeescalationThreshold -> MemoryPressure.LOW
                else -> MemoryPressure.MEDIUM
            }
            MemoryPressure.LOW -> when {
                availableRatio < mem.criticalEscalationThreshold -> MemoryPressure.CRITICAL
                availableRatio < mem.highEscalationThreshold -> MemoryPressure.HIGH
                availableRatio < mem.mediumEscalationThreshold -> MemoryPressure.MEDIUM
                else -> MemoryPressure.LOW
            }
        }

        lastPressure = newPressure
        return newPressure
    }

    /**
     * Get detailed memory info for diagnostics.
     */
    fun getMemoryInfo(): MemoryInfo {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memInfo)

        // Get app-specific memory
        val runtime = Runtime.getRuntime()
        val nativeHeap = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)

        return MemoryInfo(
            totalMb = (memInfo.totalMem / (1024 * 1024)).toInt(),
            availableMb = (memInfo.availMem / (1024 * 1024)).toInt(),
            thresholdMb = (memInfo.threshold / (1024 * 1024)).toInt(),
            lowMemory = memInfo.lowMemory,
            javaHeapMb = ((runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)).toInt(),
            javaHeapMaxMb = (runtime.maxMemory() / (1024 * 1024)).toInt(),
            nativeHeapMb = nativeHeap.toInt(),
            pressure = getMemoryPressure()
        )
    }

    /**
     * Suggest FPS based on memory pressure.
     * Returns recommended FPS that won't cause OOM.
     */
    fun suggestFps(baseFps: Int): Int {
        val mem = SensingConfig.get().memory
        return when (getMemoryPressure()) {
            MemoryPressure.CRITICAL -> mem.criticalFps
            MemoryPressure.HIGH -> mem.highFps
            MemoryPressure.MEDIUM -> (baseFps * mem.mediumFpsFraction).toInt().coerceAtLeast(mem.mediumFpsMin)
            MemoryPressure.LOW -> baseFps
        }
    }

    /**
     * Start continuous monitoring.
     */
    fun startMonitoring(scope: CoroutineScope) {
        if (monitorJob?.isActive == true) {
            Log.d(TAG, "Already monitoring")
            return
        }

        Log.i(TAG, "Starting resource monitoring (interval: ${checkIntervalMs}ms)")

        monitorJob = scope.launch(Dispatchers.Default) {
            var previousMonitorPressure = lastPressure
            while (isActive) {
                val currentPressure = getMemoryPressure()

                if (currentPressure != previousMonitorPressure) {
                    // Log memory stats only on pressure transitions
                    val info = getMemoryInfo()
                    Log.w(TAG, "Memory pressure changed: $previousMonitorPressure -> $currentPressure " +
                            "(available=${info.availableMb}MB/${info.totalMb}MB, " +
                            "javaHeap=${info.javaHeapMb}MB, native=${info.nativeHeapMb}MB)")
                    onMemoryPressureChanged?.invoke(currentPressure)

                    if (currentPressure == MemoryPressure.CRITICAL) {
                        Log.e(TAG, ">>> CRITICAL MEMORY PRESSURE - triggering emergency callback")
                        onCriticalMemory?.invoke()
                    }

                    previousMonitorPressure = currentPressure
                }

                delay(checkIntervalMs)
            }
        }
    }

    /**
     * Stop monitoring.
     */
    fun stopMonitoring() {
        monitorJob?.cancel()
        monitorJob = null
        Log.d(TAG, "Resource monitoring stopped")
    }

    /**
     * Trigger garbage collection and log results.
     * Use sparingly - GC pauses can cause UI jank.
     */
    fun requestGC() {
        val beforeInfo = getMemoryInfo()
        Log.d(TAG, "Requesting GC - before: javaHeap=${beforeInfo.javaHeapMb}MB, native=${beforeInfo.nativeHeapMb}MB")

        System.gc()
        System.runFinalization()

        val afterInfo = getMemoryInfo()
        val freedMb = beforeInfo.javaHeapMb - afterInfo.javaHeapMb
        Log.i(TAG, "GC complete - after: javaHeap=${afterInfo.javaHeapMb}MB, freed=${freedMb}MB")
    }

    /**
     * Detailed memory information.
     */
    data class MemoryInfo(
        val totalMb: Int,
        val availableMb: Int,
        val thresholdMb: Int,
        val lowMemory: Boolean,
        val javaHeapMb: Int,
        val javaHeapMaxMb: Int,
        val nativeHeapMb: Int,
        val pressure: MemoryPressure
    )
}
