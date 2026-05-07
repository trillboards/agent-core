package com.trillboards.ctv.core

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
