package com.trillboards.ctv.core.service

/**
 * Per-platform identity emitted in heartbeat metadata + capabilities payload.
 *
 * `deviceType` — server-side platform routing key. Values: "tablet", "android-tv", "fire-tv", "vidaa", "tizen", "webos".
 * `agentType`  — APK identifier. Values: "tablet-agent", "android-tv-agent", "android-tv-agent-lite", "fire-tv-agent".
 * `agentVersion` — `BuildConfig.VERSION_NAME`. Used by the OTA manifest service to determine if a self-update is needed.
 */
data class AgentMetadata(
    val deviceType: String,
    val agentType: String,
    val agentVersion: String
)
