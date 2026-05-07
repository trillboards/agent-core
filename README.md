# Trillboards Sensing SDK for Android

`agent-core` is the Android sensing primitive that powers Trillboards on-device audience intelligence: face detection, audio classification, speech recognition with diarization, BLE / WiFi / mDNS / SSDP / ARP / HTTP discovery, native sensor reads, and Vertex multimodal embeddings. Targets Android 8.0+ (`minSdk=26`), built against `compileSdk=35`, ARM64 only.

Apache 2.0. Public. Anonymous-readable via JitPack.

## Install

`settings.gradle.kts` (or `settings.gradle`):

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```

`app/build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.github.trillboards:agent-core:v1.0.0")
}
```

(Or pin to a specific commit SHA: `com.github.trillboards:agent-core:<sha>`.)

## Quick start

```kotlin
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.TrillboardsSensingSdk

val sdk = TrillboardsSensingSdk(
    context = applicationContext,
    config = AgentConfig.Builder()
        .partnerApiKey("YOUR_PARTNER_API_KEY")
        .deviceCode("YOUR_DEVICE_CODE")  // your fleet's device identifier
        .enableFaceSensing(true)
        .enableAudioSensing(true)
        .build()
)
sdk.start()
```

## What ships in the AAR

- **Face**: FaceXFormer demographics (age / gender bucketed when on-device model loaded; null otherwise), gaze estimation, per-track dwell + emotion classifier
- **Audio**: MediaPipe AudioClassifier (yamnet labels), Moonshine ASR with speaker diarization, voice-activity detection
- **Discovery**: BLE scan + Phase 6 (UWB / Auracast / ChannelSounding API 36+), WiFi BSSID/SSID hashing + nearby networks, mDNS / SSDP / ARP / HTTP probes
- **Native sensors**: ambient light, barometer, IMU
- **Cloud inference**: Vertex multimodal embeddings via Trillboards API (no GCP credentials needed on device)

## Configuration

The full surface (~30+ config knobs: enable/disable each sensing primitive, sampling rates, diarization on/off, FaceXFormer on/off, debug logging, etc.) is documented inline in `AgentConfig.kt`. Common partner integrations:

- **Camera + mic + face/audio sensing** (default — full surface): leave defaults; SDK adapts to device capabilities
- **Audio-only** (no camera): set `enableFaceSensing(false)`, leave `enableAudioSensing(true)`
- **Discovery-only** (no camera, no mic): set `enableFaceSensing(false)`, `enableAudioSensing(false)`, `enableDiscovery(true)` — useful for kiosks where audio/video privacy is restricted

## Privacy

- All face / audio / image inference happens **on-device**. No raw frames or audio buffers leave the device.
- Cloud calls send only structured features (face counts, attention level, speech transcripts when diarization is enabled, etc.) — never images.
- GDPR / CCPA opt-out is per-screen via the Trillboards portal; SDK respects opt-out flags from the heartbeat response.

## License

Apache License, Version 2.0. See `LICENSE`. The bundled ML models (Moonshine ASR, MediaPipe AudioClassifier, MediaPipe pose landmarker, yamnet) ship under their respective licenses (Apache 2.0 for all).

## Reporting issues

File an issue on this repo, or email `engineering@trillboards.com`.
