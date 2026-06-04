package com.trillboards.ctv.core

/**
 * Cross-package action for nudging [com.trillboards.ctv.core.activity.BaseAgentActivity]
 * to re-prompt the edge-AI runtime permissions (CAMERA / RECORD_AUDIO). Originally
 * lived only on `fire-tv-agent/.../AgentConfig.kt` so [com.trillboards.ctv.core.service.BaseDeviceAgentService]
 * could tell Fire TV's MainActivity to re-fire the system dialog whenever a heartbeat
 * found edge-AI hardware present but the perms missing. Hoisted here so the
 * tablet, android-tv (full flavor), and fire-tv MainActivities all listen on
 * the same action — every agent inherits the same recoverable re-prompt loop
 * without each platform redefining the constant.
 *
 * Dispatched via `LocalBroadcastManager` (process-local), so the action string
 * stays package-neutral — no namespace collision across the three apps.
 */
const val INTENT_EDGE_AI_PERMISSION_PROMPT = "com.trillboards.ctv.EDGE_AI_PERMISSION_PROMPT"

data class AgentConfig(
    val apiBaseUrl: String = "https://api.trillboards.com",
    val socketUrl: String = "https://chat.trillboards.com",
    val heartbeatPath: String = "/openrtb/v1/heartbeat",
    val privateMessageEvent: String = "privateMessage",
    val deviceCommandEvent: String = "deviceCommand",
    val deviceCommandAckEvent: String = "deviceCommandAck",
    val deviceCommandStatusEvent: String = "deviceCommandStatus",
    val sharedPrefsName: String,
    val overlayRefreshAction: String,
    val overlayBlackoutAction: String,
    val heartbeatIntervalMs: Long = 30_000L
)
