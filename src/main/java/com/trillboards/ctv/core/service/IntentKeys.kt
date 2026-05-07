package com.trillboards.ctv.core.service

/**
 * Per-platform broadcast-Intent action strings.
 *
 * Each platform's `AgentConfig` namespaces actions to its applicationId
 * (e.g., `com.trillboards.ctv.tablet.REBIND_CAMERA` vs `com.trillboards.ctv.android.REBIND_CAMERA`).
 * `BaseDeviceAgentService` and `BaseAgentActivity` register receivers and send broadcasts
 * against this struct so the agent-core code stays platform-neutral.
 *
 * `renderingModeChanged` is nullable: only the tablet ships `HybridRenderingManager`.
 */
data class IntentKeys(
    val rebindCamera: String,
    val rebindAudio: String,
    val maintenanceCycle: String,
    val watchdogCheck: String,
    val clearCache: String,
    val kioskControl: String,
    val overlayRefresh: String,
    val overlayBlackout: String,
    val renderingModeChanged: String? = null,
    val extraRebindReason: String = "rebind_reason",
    val extraRebindDelayMs: String = "rebind_delay_ms"
)
