package com.trillboards.ctv.core.models

data class DeviceCapabilityPayload(
    val powerControl: PowerControl = PowerControl(),
    val inputControl: InputControl = InputControl(),
    val audioControl: AudioControl = AudioControl(),
    val environmentSensors: EnvironmentSensors = EnvironmentSensors(),
    val audienceSensing: AudienceSensing = AudienceSensing(),
    // MDM capabilities
    val managementCapabilities: ManagementCapabilities = ManagementCapabilities(),
    // ML hardware capabilities from HardwareManifest
    val mlCapabilities: MLCapabilities = MLCapabilities(),
    // WiFi CSI sensing hardware capabilities (mirrors agent-core-lite)
    val wifiCsiSensing: WifiCsiSensing = WifiCsiSensing(),
    val isDeviceOwner: Boolean = false,
    val agentType: String = "unknown",  // "fire-tv", "android-tv", "tablet"
    val agentVersion: String = "",
    val capabilitiesVersion: Int = 1
) {
    data class PowerControl(
        val cec: Boolean = false,
        val wol: Boolean = false,
        val softBlackout: Boolean = true
    )

    data class InputControl(
        val hdmi: Boolean = false,
        val appSwitch: Boolean = false
    )

    data class AudioControl(
        val volume: Boolean = true,
        val mute: Boolean = true
    )

    data class EnvironmentSensors(
        val ambientLight: Boolean = false,
        val temperature: Boolean = false
    )

    /**
     * Audience sensing capabilities including camera status.
     * Useful for fleet dashboard to show USB vs internal camera adoption.
     *
     * `cameraHealth` (added 2026-05-03 in fix-ctv-camera-never-started-product-fix):
     * surfaces the [com.trillboards.ctv.core.audience.CameraHealthMonitor]
     * snapshot to the heartbeat so the server can detect "camera was never
     * started" / "stuck failing for 30+ days" without having to infer it
     * from missing `vision_call_log` rows. Server reads it via
     * `services/deviceCommandSupport.js#buildCapabilityUpdateDoc` and
     * persists to `earner_screen_devices.camera_health_*` typed columns.
     */
    data class AudienceSensing(
        val cameraAvailable: Boolean = false,
        val cameraType: String = "none",  // "usb", "internal", "none"
        val cameraCount: Int = 0,
        val microphoneAvailable: Boolean = false,
        val sensingMode: String = "NONE",  // "FULL", "FACE_ONLY", "AUDIO_ONLY", "NONE"
        val cameraHealth: CameraHealth = CameraHealth()
    )

    /**
     * Per-screen camera health gauge. State machine values mirror
     * [com.trillboards.ctv.core.audience.CameraHealthMonitor.CameraState]
     * (lowercased) so dashboards can group on a stable enum.
     *
     * - `state`: never_started | starting | open | failed | closed
     * - `lastFrameAtMs`: System.currentTimeMillis() of last successful frame
     *   (0 if camera has never delivered a frame — the Adam @ Focus Media case)
     * - `retryCount`: consecutive failures since last OPEN
     * - `lastFailureClass`: permission_denied | provider_init_failed |
     *   no_camera_selected | lens_facing_query_failed | bind_failed |
     *   camera_unavailable | frame_timeout | etc.
     * - `lastFailureMessage`: short Throwable.message clipped to 512
     */
    data class CameraHealth(
        val state: String = "never_started",
        val lastFrameAtMs: Long = 0L,
        val retryCount: Int = 0,
        val lastFailureClass: String? = null,
        val lastFailureMessage: String? = null
    )

    /**
     * MDM/Device Management capabilities.
     * Reports what management commands the device supports.
     */
    data class ManagementCapabilities(
        val restart: Boolean = true,        // All agents support app restart
        val reboot: Boolean = false,        // Device Owner only
        val screenshot: Boolean = true,     // All agents
        val clearCache: Boolean = true,     // All agents
        val kioskMode: Boolean = false,     // Device Owner or soft kiosk (Fire TV)
        val installApk: Boolean = false,    // Device Owner only for silent install
        val wipe: Boolean = false           // Device Owner only
    )

    /**
     * ML hardware capabilities from HardwareManifest.
     * Reports GPU/NPU/chipset info for fleet-wide model deployment planning.
     *
     * Phase 2 prereq PR 2 (audit `audit-2026-05-03-device-telemetry-deep-inventory.md`):
     * The cpuAbi / osApiLevel / screenWidthPx / screenHeightPx / densityDpi
     * fields are the new wire additions sourced from
     * `HardwareManifest.detect()`. Server-side typed columns landed in
     * migration 20260503082254 — server is already wired to read them
     * defensively (snake_case OR camelCase, plus a positive-int filter on
     * the JS side via `safePositiveInt`).
     *
     * Codex P1 (PR #4540): geometry / API / RAM fields are NULLABLE so a
     * default-ctor `MLCapabilities()` produced when `HardwareManifest.detect()`
     * fails on the edge emits JSON null instead of zero. The server then
     * writes NULL (no-data state) instead of overwriting previously-valid
     * data. Real probes always return positive ints; null = "couldn't
     * measure", 0 was the old fail-zero sentinel that overwrote good data.
     */
    data class MLCapabilities(
        val chipsetVendor: String = "unknown",
        val chipsetName: String = "",
        val totalRamMb: Int? = null,
        val availableRamMb: Int? = null,
        val gpuName: String? = null,
        val hasNpu: Boolean = false,
        val npuName: String? = null,
        val gpuDelegateSupported: Boolean = false,
        val nnapiSupported: Boolean = false,
        val recommendedModelTier: String = "STANDARD",
        val maxVlmSizeMb: Int? = null,
        // Phase 2 prereq PR 2 — additive only, default-safe (NULL on probe failure)
        val cpuAbi: String? = null,
        val osApiLevel: Int? = null,
        val screenWidthPx: Int? = null,
        val screenHeightPx: Int? = null,
        val densityDpi: Int? = null
    )

    /**
     * WiFi CSI sensing hardware capabilities.
     * Reports whether ESP32-S3 CSI nodes are available on the local network.
     */
    data class WifiCsiSensing(
        val csiAvailable: Boolean = false,
        val csiNodeCount: Int = 0,
        val csiHardwareType: String = "none"  // "esp32-s3", "none"
    )
}
