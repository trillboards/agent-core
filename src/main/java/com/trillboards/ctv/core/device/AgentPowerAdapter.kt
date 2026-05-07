package com.trillboards.ctv.core.device

import com.trillboards.ctv.core.models.DeviceCapabilityPayload

/**
 * Platform-agnostic power-control surface. Each platform's `PowerController` implements
 * this; methods that don't apply to the platform return `false` (or sensible defaults).
 *
 * Tablets: blackout overlay + brightness + wake-lock based screen-on.
 * Android TV: HDMI-CEC + soft blackout + AudioManager volume/mute + display state.
 *
 * The union surface is wide so `BaseCommandProcessor` can dispatch any command to any
 * platform without per-platform `if` branches. Capability negotiation happens via the
 * `supports*()` methods + `capabilitySnapshot()` payload — server reads those and sends
 * only commands the device declared it supports.
 *
 * All methods return `Boolean` for consistency: `true` = "command was acted on", `false`
 * = "platform doesn't support this command" or "command failed". Callers map this to
 * the `commandResult.success` / `commandResult.error` ack envelope.
 */
interface AgentPowerAdapter {
    // ── Capability declaration ────────────────────────────────────────────────
    fun supportsCec(): Boolean = false
    fun supportsInputSwitching(): Boolean = false
    fun supportsVolumeControl(): Boolean = false
    fun supportsMute(): Boolean = false
    fun supportsSoftBlackout(): Boolean = false
    fun supportsBrightness(): Boolean = false
    fun supportsDisplayState(): Boolean = false

    // ── Power / blackout ──────────────────────────────────────────────────────
    fun powerOn(): Boolean = false
    fun powerOff(): Boolean = false
    fun togglePower(): Boolean = false
    fun setBlackout(enabled: Boolean): Boolean = false
    fun currentBlackoutState(): Boolean = false

    // ── Brightness / wake ─────────────────────────────────────────────────────
    fun screenOn(): Boolean = false
    fun setBrightness(level: Int): Boolean = false

    // ── HDMI input switching (TV-specific) ────────────────────────────────────
    fun selectInput(source: String?): Boolean = false

    // ── Audio (TV-specific; tablets defer to system volume keys) ──────────────
    fun setVolume(level: Int): Boolean = false
    fun toggleMute(): Boolean = false

    // ── Audio extensions (PR 2a) — BaseCommandProcessor dispatches volume.up,
    //     volume.down, mute.on, mute.off, mute.toggle through these. Each
    //     platform's PowerController overrides what it can do; defaults
    //     fail-soft (return false / -1) when the platform doesn't support
    //     them. `showUi=true` enables the system volume OSD on tablet; TV
    //     adapters typically ignore the flag (TV shows OSD natively).
    fun volumeUp(showUi: Boolean = true): Boolean = false
    fun volumeDown(showUi: Boolean = true): Boolean = false
    fun setVolumePercent(level: Int, showUi: Boolean = false): Boolean = setVolume(level)
    fun muteOn(showUi: Boolean = false): Boolean = false
    fun muteOff(showUi: Boolean = false): Boolean = false
    fun isMuted(): Boolean = false
    fun currentVolumePercent(): Int = -1
    fun maxVolumePercent(): Int = 100

    // ── Display state for heartbeat telemetry ─────────────────────────────────
    fun isDisplayOn(): Boolean = true
    fun getPowerState(): String = "unknown"

    // ── Capability snapshot — emitted in heartbeat ────────────────────────────
    fun capabilitySnapshot(): DeviceCapabilityPayload
}
