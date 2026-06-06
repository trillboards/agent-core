# Trillboards Sensing SDK for Android

`agent-core` is the Android sensing primitive that powers Trillboards on-device
audience intelligence: face detection, audio classification, speech recognition
with diarization, BLE / WiFi / mDNS / SSDP / ARP / HTTP discovery, native sensor
reads, and Vertex multimodal embeddings. Targets Android 8.0+ (`minSdk=26`), built
against `compileSdk=35`.

Apache 2.0. **Anonymous-readable via https://maven.trillboards.com/** — no tokens,
no GitHub account. This repo is the Apache-2.0 source mirror.

## Install

`settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://maven.trillboards.com/") }
    }
}
```

`app/build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.trillboards:agent-core:1.1.+")            // floats on the latest 1.1.x

    // Optional — on-device speech (Moonshine ASR). Omit to ship without it.
    implementation("com.trillboards.bundled:sherpa-onnx:1.12.26")
}
```

Pin an exact version (e.g. `1.1.4`) for reproducible builds.

## Quick start

**Foreground** — SDK embedded in your on-screen player app. The camera follows app
visibility via `ProcessLifecycleOwner` automatically; no service needed:

```kotlin
import com.trillboards.sdk.TrillboardsSensingSdk

TrillboardsSensingSdk.start(
    context = applicationContext,
    partnerApiKey = "YOUR_PARTNER_API_KEY",
)
```

**Background / headless** — sensing runs in its own app with no visible Activity
(e.g. beneath a separate ad-player). Pass your `FOREGROUND_SERVICE_TYPE_CAMERA`
`LifecycleService` as `lifecycleOwner` so the camera stays bound while backgrounded
(start the service from a visible Activity per the Android 14 while-in-use rule):

```kotlin
import androidx.lifecycle.LifecycleService
import com.trillboards.sdk.SensingSdkConfig
import com.trillboards.sdk.TrillboardsSensingSdk

class CaptureService : LifecycleService() {
    override fun onCreate() {
        super.onCreate()
        startForeground(
            NOTIF_ID, notification,
            FOREGROUND_SERVICE_TYPE_CAMERA or FOREGROUND_SERVICE_TYPE_MICROPHONE,
        )
        TrillboardsSensingSdk.start(
            context = applicationContext,
            partnerApiKey = "YOUR_PARTNER_API_KEY",
            config = SensingSdkConfig(deviceId = "YOUR_DEVICE_ID"),
            lifecycleOwner = this,
        )
    }

    override fun onDestroy() {
        TrillboardsSensingSdk.stop(this)
        super.onDestroy()
    }
}
```

`deviceId` ties the install to a screen you registered via `POST /v1/partner/device`
(the same value you used as `external_id`). When null, the SDK uses a hardware
fingerprint and sensing runs unattributed until the device is bound.

## What ships in the AAR

- **Face**: FaceXFormer demographics (age / gender), gaze estimation, per-track dwell + emotion
- **Audio**: MediaPipe AudioClassifier (yamnet), Moonshine ASR with speaker diarization, VAD
- **Discovery**: BLE scan (+ UWB / Auracast / ChannelSounding on API 36+), WiFi BSSID/SSID hashing, mDNS / SSDP / ARP / HTTP probes
- **Native sensors**: ambient light, barometer, IMU
- **Cloud inference**: Vertex multimodal embeddings via the Trillboards API (no GCP credentials on device)

## Configuration

`SensingSdkConfig` exposes ~30 knobs (toggle each sensing primitive, sampling
rates, diarization, FaceXFormer, debug logging). Common integrations:

- **Full (camera + mic + face/audio)** — defaults; the SDK adapts to device capabilities.
- **Audio-only** — `SensingSdkConfig(faceDetectionEnabled = false)`.
- **Discovery-only** — `SensingSdkConfig(faceDetectionEnabled = false, audioClassificationEnabled = false)`.

## Privacy

- All face / audio / image inference happens **on-device**. No raw frames or audio buffers leave the device.
- Cloud calls send only structured features (counts, attention level, transcripts when diarization is enabled) — never images.
- GDPR / CCPA opt-out is per-screen via the Trillboards portal; the SDK respects opt-out flags from the heartbeat response.

## License

Apache License, Version 2.0. See `LICENSE`. The bundled ML models (Moonshine ASR,
MediaPipe AudioClassifier, pose landmarker, yamnet) ship under their respective
licenses (Apache 2.0).

## Reporting issues

File an issue on this repo, or email `developers@trillboards.com`.
