# Paired Training Samples API Contract

This document is the source of truth for the request/response shape used by
[CsiPairedDataCollector](./CsiPairedDataCollector.kt) when uploading paired
camera-pose / CSI-window samples to the WiFlow training backend.

The collector targets ~9,000 paired samples per active screen. Once enough
data is collected fleet-wide, the server can train a CSI-only pose estimator
that achieves the WiFlow paper's reported 92.9% PCK@20.

## Endpoint

```
POST /v2/sensing/paired-training-samples
Content-Type: application/json
X-Device-Token: <token>
X-Device-Timestamp: <epoch_millis>
X-Device-Nonce: <uuid>
```

Auth headers match the shape of every other agent-core endpoint
(see [`ApiClient.authRequestBuilder`](../net/ApiClient.kt)).

## Request body

```json
{
  "screenId": "<mongo_object_id_or_omit>",
  "samples": [
    {
      "sampleId": "uuid",
      "csiWindowBase64": "<base64-encoded little-endian floats>",
      "subcarrierCount": 56,
      "keypoints": [x1, y1, x2, y2, ..., x17, y17],
      "keypointConfidences": [c1, c2, ..., c17],
      "overallConfidence": 0.87,
      "numCameraFrames": 5,
      "windowStartMs": 1712345678000,
      "windowEndMs": 1712345678200,
      "venueType": "retail",
      "cameraModelVersion": "mediapipe-blazepose-lite-v0.10.33"
    }
  ]
}
```

### Field semantics

| Field | Type | Notes |
|---|---|---|
| `screenId` | string \| omit | Mongo ObjectId of the screen the agent is bound to. Omitted if the agent is not yet bound. |
| `samples[]` | array | Up to 500 samples per request (collector batches client-side; see `DEFAULT_UPLOAD_BATCH_SIZE`). |
| `samples[].sampleId` | string (UUID v4) | Server-side dedupe key. Idempotent — duplicates must return success and not insert. |
| `samples[].csiWindowBase64` | string | Base64-no-wrap encoding of `subcarrierCount × 20` little-endian floats (4 bytes each). Frame-major: `[frame0_sc0, frame0_sc1, ..., frame19_scN-1]`. Total decoded byte length is `subcarrierCount × 20 × 4`. |
| `samples[].subcarrierCount` | int | Number of OFDM subcarriers per CSI frame. Currently 56 (ESP32-S3 HT20) or 128 (HT40). |
| `samples[].keypoints` | float[34] | 17 COCO joints × (x, y), each in `[0, 1]`. Joint order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle. |
| `samples[].keypointConfidences` | float[17] | Per-joint visibility from MediaPipe Pose Landmarker, mapped through `MediaPipePoseAdapter.mapBlazePoseToCoco`. |
| `samples[].overallConfidence` | float | Mean of `keypointConfidences`. Filtered client-side at `>= 0.5`. |
| `samples[].numCameraFrames` | int | How many camera frames were used to derive the pose. Always equals `framesPerWindow` (default 20) for the current implementation. |
| `samples[].windowStartMs` | long | Epoch millis of the first CSI frame in the 200ms window. |
| `samples[].windowEndMs` | long | Epoch millis of the last CSI frame. `windowEndMs - windowStartMs` is approximately 200 ms (≤ 1000 ms upper bound to tolerate clock skew). |
| `samples[].venueType` | string \| omit | Trillboards venue category (retail, bar, restaurant, etc.) for downstream training stratification. |
| `samples[].cameraModelVersion` | string \| omit | The pose model identifier that produced the keypoints (e.g. `mediapipe-blazepose-lite-v0.10.33`). |

### Validation rules the server should enforce

- `samples` length in `[1, 500]`
- `keypoints.length == 34`
- `keypointConfidences.length == 17`
- `subcarrierCount in {56, 128}`
- `decoded(csiWindowBase64).length == subcarrierCount * 20 * 4`
- `windowEndMs >= windowStartMs`
- `overallConfidence in [0, 1]`

## Response

Success:

```json
{
  "ok": true,
  "accepted": 487,
  "deduped": 13
}
```

| Field | Type | Notes |
|---|---|---|
| `ok` | boolean | `true` for any 2xx response. |
| `accepted` | int | Number of new rows the server inserted. |
| `deduped` | int | Number of `sampleId`s that already existed (idempotent). |

The collector treats any 2xx as success and marks all `sampleId`s in the
batch as uploaded locally. Per-sample failure responses are NOT supported —
servers should reject the whole batch or accept the whole batch.

### Failure response

Any non-2xx is treated as a transient failure. The collector leaves the
samples unuploaded for the next flush. The collector does NOT retry within
a single `uploadToBackend` call to keep flushes bounded.

```json
{
  "ok": false,
  "error": "human-readable message"
}
```

## Sizing

A single sample is approximately:

| Component | Size |
|---|---|
| 20 frames × 56 subcarriers × 4 bytes = 4480 bytes (raw) | ~6 KB base64 |
| keypoints (34 floats × ~6 char each as JSON) | ~250 B |
| confidences (17 × ~6 char each) | ~120 B |
| metadata + JSON overhead | ~200 B |
| **Per-sample total** | **~6.5 KB** |

A 500-sample batch is approximately **3.3 MB** uncompressed JSON. With
gzip compression on the wire, ~1 MB. Comfortably under the 10 MB API gateway
limit.

A full 9,000-sample collection is approximately **18 batches** = **18 HTTPS
requests** per screen. Network impact is negligible compared to ad-decision
traffic.
