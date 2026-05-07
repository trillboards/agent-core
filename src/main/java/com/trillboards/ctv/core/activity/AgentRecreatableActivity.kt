package com.trillboards.ctv.core.activity

/**
 * Marker interface for an Activity that can recreate its WebView.
 *
 * Used by command-processor / recovery-ladder code paths in agent-core that need to
 * trigger a "soft reboot" of the player without coupling to a specific platform's
 * `MainActivity` class. Each platform's `MainActivity` implements this; agent-core
 * code path uses `(activity as? AgentRecreatableActivity)?.recreateWebView(reason)`.
 *
 * Reason strings are free-form, captured in logs and recovery-event telemetry —
 * "manual", "render_process_gone", "memory_pressure", "config_push", etc.
 */
interface AgentRecreatableActivity {
    fun recreateWebView(reason: String = "manual")
}
