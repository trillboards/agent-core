package com.trillboards.ctv.core.identity

import android.content.Context
import android.os.Build
import android.util.Log

/**
 * Phase 6 bonus signal — BLE Channel Sounding (Android 16 / API 36+).
 *
 * Channel Sounding is a Bluetooth 6.0 ranging technology that produces
 * sub-meter distance + (optional) angle-of-arrival measurements between
 * paired BLE peers. On Android 16+ it is exposed through
 * [android.ranging.RangingManager] (system service `ranging`) with the
 * `RANGING_TECHNOLOGY_BLE_CS` technology constant — *not* through
 * `BluetoothAdapter`. An earlier draft of this file reflected
 * `BluetoothAdapter.getChannelSoundingMeasurements`, which does not exist
 * on any Android version; that path always threw `NoSuchMethodException`
 * and silently returned empty (Codex P1 review on PR #4558).
 *
 * Design constraints that shape this scaffold:
 *
 *  - **Session-based, not snapshot.** The real `RangingManager` API is
 *    callback-driven: the caller starts a `RangingSession` with a
 *    `RangingPreference` declaring `BLE_CS`, registers a
 *    `RangingSession.Callback`, receives `onResults(...)` events
 *    asynchronously, then closes the session. There is no "give me the
 *    current measurements" snapshot call. Heartbeat-time discovery wants
 *    a snapshot, not a session, so the production-grade implementation
 *    is non-trivial (see follow-up plan in `deferred-work.yaml`).
 *
 *  - **Paired-peer-only.** Channel Sounding requires a paired BLE 6.0
 *    peer on both sides. With no paired peers the session has nothing
 *    to range against and produces zero results. Tab S11 (the dev
 *    device, API 36) has no paired BLE_CS peers; the production fleet
 *    likewise has none today. The signal will be empty in production
 *    until paired-peer scenarios exist.
 *
 *  - **API not on compile classpath.** AGP 8.5+ + compileSdk 35 means
 *    `android.ranging.*` types are not on the build classpath. Until
 *    we bump compileSdk to 36, the API surface must be accessed via
 *    reflection. Reflection probes for the system service + technology
 *    constant + `getSupportedRangingTechnologies()` introspection.
 *
 * Behavior on:
 *   - API < 36 → returns `emptyList()` immediately.
 *   - API 36+ without `RangingManager` system service → returns
 *     `emptyList()` (logged once at INFO).
 *   - API 36+ with `RangingManager` but BLE_CS not in supported set
 *     → returns `emptyList()` (logged once at INFO).
 *   - API 36+ with BLE_CS supported but no paired peers → returns
 *     `emptyList()` (no session attempted; documented above).
 *
 * Caller contract: never throws. Always returns a (possibly-empty) list
 * of measurements sorted by address for deterministic per-cycle output.
 */
data class ChannelSoundingMeasurement(
    val address: String,
    val distance_mm: Float,
    val aoa_deg: Float?
)

object ChannelSoundingDiscovery {
    private const val TAG = "ChannelSoundingDiscovery"

    // API 36 corresponds to Android 16 / VANILLA_ICE_CREAM_PLUS / BAKLAVA
    // (the constant moved between betas). We use the integer literal 36
    // rather than `Build.VERSION_CODES.BAKLAVA` so this compiles on
    // build environments with older platform-tools jars.
    private const val MIN_SDK_BLE_CS = 36

    // android.ranging.RangingManager.RANGING_TECHNOLOGY_BLE_CS — read
    // reflectively because the constant isn't on the compile classpath
    // (compileSdk = 35 today). Reflection probe is documented at point
    // of use to keep the rationale local.
    private const val RANGING_MANAGER_CLASS = "android.ranging.RangingManager"
    private const val SYSTEM_SERVICE_RANGING = "ranging"
    private const val TECH_CONSTANT_BLE_CS = "RANGING_TECHNOLOGY_BLE_CS"

    /**
     * Collect Channel Sounding measurements via the
     * [android.ranging.RangingManager] system service.
     *
     * @param context Android context used to acquire the system service.
     *   Pass `null` (the JVM-test path) to force the legacy code path
     *   that returns `emptyList()` without touching Android internals.
     * @return measurements sorted by address; empty when API < 36, no
     *   `RangingManager`, BLE_CS not supported, or no paired peers.
     */
    @JvmStatic
    fun collectChannelSoundingMeasurements(
        context: Context? = null
    ): List<ChannelSoundingMeasurement> {
        if (Build.VERSION.SDK_INT < MIN_SDK_BLE_CS) {
            return emptyList()
        }
        if (context == null) {
            // Test environment / no Android context. Don't try reflection.
            return emptyList()
        }
        return try {
            val rangingManager = context.getSystemService(SYSTEM_SERVICE_RANGING)
                ?: run {
                    Log.i(TAG, "RangingManager system service unavailable on this device")
                    return emptyList()
                }
            // Probe RangingManager.getSupportedRangingTechnologies() — if
            // it exists AND BLE_CS is in the set, we know the device
            // hardware is capable. Without this gate we'd attempt to
            // start a session on devices that don't ship the radio,
            // which silently times out the session.
            val supported = readSupportedTechnologies(rangingManager)
            val bleCs = readBleCsConstant()
            if (bleCs == null || !supported.contains(bleCs)) {
                Log.i(TAG, "BLE_CS technology not supported by RangingManager (supported=$supported)")
                return emptyList()
            }
            // The actual ranging-session implementation is deferred
            // (`deferred-work.yaml#ctv-channel-sounding-session-impl`).
            // Until then, return empty cleanly: the column will be
            // populated by the SAME wire format once the session impl
            // ships, with no schema or server-side change required.
            Log.d(TAG, "BLE_CS supported; session-based ranging not yet implemented (deferred)")
            emptyList()
        } catch (e: Exception) {
            Log.w(TAG, "Channel Sounding collection failed", e)
            emptyList()
        }
    }

    /**
     * Read the integer set returned by
     * `RangingManager.getSupportedRangingTechnologies()` via reflection.
     * Returns the empty set if the method isn't found OR the call
     * throws — never propagates an exception.
     */
    private fun readSupportedTechnologies(rangingManager: Any): Set<Int> {
        return try {
            val cls = rangingManager.javaClass
            val method = cls.methods.firstOrNull {
                it.name == "getSupportedRangingTechnologies" && it.parameterCount == 0
            } ?: return emptySet()
            val raw = method.invoke(rangingManager)
            when (raw) {
                is IntArray -> raw.toSet()
                is Collection<*> -> raw.filterIsInstance<Int>().toSet()
                else -> emptySet()
            }
        } catch (e: Exception) {
            emptySet()
        }
    }

    /**
     * Read the integer value of
     * `android.ranging.RangingManager.RANGING_TECHNOLOGY_BLE_CS` via
     * reflection. Returns null when the constant isn't present (the
     * device's Android 16 ships an older preview without BLE_CS).
     */
    private fun readBleCsConstant(): Int? {
        return try {
            val cls = Class.forName(RANGING_MANAGER_CLASS)
            val field = cls.getField(TECH_CONSTANT_BLE_CS)
            (field.get(null) as? Int)
        } catch (e: ClassNotFoundException) {
            null
        } catch (e: NoSuchFieldException) {
            null
        } catch (e: Exception) {
            null
        }
    }
}
