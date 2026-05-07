package com.trillboards.ctv.core.service

/**
 * Per-platform `SharedPreferences` key namespace.
 *
 * Each platform stores in its own prefs file (`trillboard_tablet_agent_prefs` vs
 * `trillboard_android_tv_prefs`) and has its own subset of optional pref keys.
 * `renderingMode` is nullable (tablet-only); `kioskPin` and `kioskModeEnabled` exist
 * on both. `setupComplete` is tablet-only (TV has no setup-wizard activity).
 */
data class PrefsKeys(
    val sharedPrefsName: String,
    val screenId: String,
    val deviceToken: String,
    val kioskPin: String,
    val kioskModeEnabled: String,
    val autoWakeEnabled: String,
    val fingerprint: String? = null,
    val lastTelemetry: String? = null,
    val setupComplete: String? = null,
    val deviceDataDisclosureAccepted: String? = null,
    val batteryExemptionAsked: String? = null,
    val renderingMode: String? = null
)
