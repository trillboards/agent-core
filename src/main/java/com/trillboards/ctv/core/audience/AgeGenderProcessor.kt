package com.trillboards.ctv.core.audience

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.util.Log
import com.trillboards.ctv.core.ml.ModelDownloadManager
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * On-device age/gender estimation using TFLite.
 *
 * Phase 2 (2026-05-04): backs onto FaceXFormer (Kartik-3004/facexformer),
 * a Swin-B transformer trained on UTKFace + FairFace. The 9-task model is
 * converted to TFLite via tools/edge-models/convert_facexformer.py with
 * only the age + gender heads exposed (FaceXFormerAgeGenderBypass wrapper
 * skips landmark/parsing/expression/etc. heads since we don't consume them).
 *
 * Input:    1×3×224×224 NCHW float32, ImageNet normalization
 *           ((R/255-0.485)/0.229, (G/255-0.456)/0.224, (B/255-0.406)/0.225)
 * Output 0: [1, 8] float32 age logits
 * Output 1: [1, 2] float32 gender logits ([male, female])
 * Size:     178 MB FP16 (delivered via OTA — never bundled in APK)
 * Latency:  ~100 ms per face on Mac M-series CPU @ 4 threads;
 *           comparable on Tab S11 (MediaTek 9300+, 4× big cores)
 *
 * Age bin labels (FaceXFormer paper §3 + §6 — UTKFace+FairFace decade bins):
 *   "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
 *
 * Privacy: No face images stored. Only aggregate distributions emitted.
 */
class AgeGenderProcessor(private val context: Context) {

    companion object {
        private const val TAG = "AgeGender"

        /**
         * Legacy APK-bundled asset filename, preserved for older builds that
         * shipped a custom MobileFaceNet-style classifier. The Phase 2
         * production path is OTA-only (FaceXFormer at 178 MB cannot live in
         * the APK).
         */
        private const val LEGACY_MODEL_FILE = "age_gender_model.tflite"

        /**
         * Foundation-model registry IDs that ModelDownloadManager may have
         * delivered via OTA. Phase 2 ships FaceXFormer under the legacy
         * `age_gender` registry slot (migration 215 row, updated by
         * `add_face_foundation_model.sql`). We also accept an explicit
         * `face_foundation` registry ID for forward-compatibility with a
         * future re-keying.
         */
        private val OTA_MODEL_IDS = listOf("age_gender", "face_foundation")

        /**
         * FaceXFormer expects 224×224 ImageNet-normalized RGB. The
         * LEGACY model used 96×96 [0,1]-normalized RGB; we DO NOT support
         * the legacy path for Phase 2 — if the legacy asset is somehow
         * still on disk (older OTA-installed APK), the I/O shapes will
         * mismatch and inference will fail loudly rather than silently
         * producing garbage labels.
         *
         * Exposed publicly so AudienceAnalyzer can size its pre-allocated
         * crop bitmap to match (`Bitmap.createBitmap(MODEL_INPUT_SIZE, ...)`).
         */
        const val MODEL_INPUT_SIZE = 224
        private const val INPUT_SIZE = MODEL_INPUT_SIZE

        /**
         * ImageNet preprocessing constants (FaceXFormer training inherits
         * these from the Swin backbone via torchvision).
         */
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        private const val NUM_AGE_BINS = 8
        private const val NUM_GENDER_CLASSES = 2

        /**
         * FaceXFormer's UTKFace+FairFace decade-bin scheme. Documented in
         * the paper §3 ("Age annotations are categorized into decade bins:
         * 0–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60–69, and over 70").
         * The output index ↔ label mapping is positional: argmax index 0 →
         * "0-9", index 7 → "70+".
         */
        val AGE_RANGES = arrayOf(
            "0-9", "10-19", "20-29", "30-39",
            "40-49", "50-59", "60-69", "70+"
        )
        val GENDER_LABELS = arrayOf("male", "female")
    }

    private var interpreter: Interpreter? = null
    private var isInitialized = false

    // Accumulator for current 10s window
    private val ageAccumulator = IntArray(NUM_AGE_BINS)
    private val genderAccumulator = IntArray(NUM_GENDER_CLASSES)
    private var totalFacesProcessed = 0
    private var totalConfidence = 0f

    // Phase 1c collapse-point #4 fix — per-face age/gender overlay.
    // The legacy `accumulate()` API immediately collapsed each result into
    // histogram bin counters, throwing away which face it came from. We
    // keep that path (so existing readers continue to see age_distribution
    // / gender_split JSON) AND maintain a parallel per-face map keyed by
    // ML Kit face track ID. AudienceAnalyzer reads this map and threads
    // age/gender into the PerFaceAttention bucket for the matching faceId
    // at drain time. ConcurrentHashMap: producer (analyzer thread) and
    // consumer (window-flush thread) overlap.
    private val perFaceLatest =
        java.util.concurrent.ConcurrentHashMap<Int, AgeGenderResult>()

    // Pre-allocated inference buffers (reused per face to prevent GC pressure)
    private var inputBuffer: ByteBuffer? = null
    private var pixelBuffer: IntArray? = null

    // Pre-allocated output arrays (reused per inference to prevent GC pressure)
    private var ageOutput: Array<FloatArray>? = null
    private var genderOutput: Array<FloatArray>? = null

    /**
     * Check if a model file is available — either bundled in APK assets or
     * OTA-downloaded by ModelDownloadManager into cacheDir/models/<id>/.
     *
     * Phase 2 (2026-05-04): the FaceXFormer foundation model is 178 MB FP16,
     * far too large to bundle in the APK, so the production path is OTA-only.
     * The `LEGACY_MODEL_FILE` asset path remains as a safety check for older
     * builds, but the I/O shapes differ between the two so a pre-existing
     * legacy asset will fail at initialize() and the heartbeat will emit no
     * onDeviceDemographics block (honest null over fabricated demographics).
     */
    fun hasModel(): Boolean {
        // 1. APK-bundled asset (legacy / dev-time path)
        try {
            context.assets.open(LEGACY_MODEL_FILE).close()
            Log.w(TAG, "Legacy age_gender_model.tflite found in APK assets — " +
                "this build expects 224×224 FaceXFormer, not 96×96 legacy. " +
                "Inference may fail at initialize() if the legacy asset is loaded first.")
            return true
        } catch (_: Exception) { /* fall through to OTA check */ }

        // 2. OTA-downloaded artifact via ModelDownloadManager
        val ota = findOtaModelFile()
        if (ota != null) {
            Log.i(TAG, "Age/gender model available via OTA: ${ota.absolutePath}")
            return true
        }

        Log.w(TAG, "[AgeGender] Model NOT loaded: asset+OTA both missing — " +
            "on_device_demographics will remain null until OTA download completes.")
        return false
    }

    /**
     * Locate an OTA-downloaded age/gender model on disk. Returns null if no
     * registered OTA_MODEL_IDS have a complete file. ModelDownloadManager
     * places verified binaries at `cacheDir/models/<modelId>/model.bin`
     * (or `.litertlm` / `.onnx` depending on format) — getModelPath()
     * resolves the right one.
     */
    private fun findOtaModelFile(): File? {
        val downloadManager = ModelDownloadManager(context)
        for (id in OTA_MODEL_IDS) {
            val path = downloadManager.getModelPath(id)
            if (path != null && path.length() > 0) return path
        }
        return null
    }

    /**
     * Initialize the TFLite interpreter and validate model I/O shape.
     *
     * Validates that the loaded model matches the expected FaceXFormer
     * contract: 1×3×224×224 float32 input, two output tensors of [1,8] (age)
     * and [1,2] (gender). If the shapes don't match (e.g. legacy 96×96
     * asset is loaded), this returns false rather than producing garbage
     * predictions — a downstream null is honest; mislabelled data is not.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        return try {
            val model = loadModelFile()
            // FaceXFormer 178MB Swin-B transformer: prefer NNAPI delegate so
            // we actually use the MediaTek APU on Dimensity 9000+ (e.g. Tab
            // S11's Dimensity 9400 reports MT6991 → APU per HardwareManifest)
            // or the Hexagon DSP on Snapdragon 8-series, where CPU-only
            // inference sat at ~100ms/face — the single biggest item in the
            // per-frame budget (live ADB telemetry on the S11 showed
            // avgInferenceMs=103 with this path on pure CPU). NNAPI does
            // per-op partitioning and falls back to CPU for unsupported ops
            // automatically. Codex P2 review on PR #6229: a try around the
            // Interpreter ctor is not enough — some NNAPI drivers initialize
            // successfully but fail at the FIRST runForMultipleInputsOutputs()
            // call for this specific model/driver combo, after which every
            // classifyWithPreallocatedCrop() silently returns null while
            // isInitialized stays true (demographics regress to permanent
            // null on those devices). So we run a synthetic test inference
            // BEFORE publishing the interpreter; on either construction OR
            // invoke failure we close the bad candidate and rebuild plain
            // 4-thread CPU so devices without functional NNAPI keep working.
            val interp = try {
                val candidate = Interpreter(model, Interpreter.Options().apply {
                    setUseNNAPI(true)
                    setNumThreads(4) // CPU fallback within NNAPI partitioning
                })
                try {
                    // Synthetic invoke: zero-filled 1×3×224×224 float32 input +
                    // output tensors matching the expected [1,8] age and
                    // [1,2] gender shapes. Zeros are fine — we only care that
                    // the delegate completes the invoke without throwing.
                    val testInput = ByteBuffer
                        .allocateDirect(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4)
                        .order(ByteOrder.nativeOrder())
                    val testAgeOut = Array(1) { FloatArray(NUM_AGE_BINS) }
                    val testGenderOut = Array(1) { FloatArray(NUM_GENDER_CLASSES) }
                    candidate.runForMultipleInputsOutputs(
                        arrayOf(testInput),
                        mapOf(0 to testAgeOut, 1 to testGenderOut)
                    )
                    Log.i(TAG, "AgeGender NNAPI delegate validated via test inference")
                    candidate
                } catch (invokeErr: Exception) {
                    Log.w(TAG, "AgeGender NNAPI invoke failed (${invokeErr.message}); " +
                        "falling back to 4-thread CPU")
                    candidate.close()
                    Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
                }
            } catch (e: Exception) {
                Log.w(TAG, "AgeGender NNAPI interpreter construction failed: ${e.message}; " +
                    "falling back to 4-thread CPU")
                Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
            }

            // Validate I/O contract before publishing the interpreter
            val inDetails = interp.getInputTensor(0)
            val inShape = inDetails.shape()
            val expected = intArrayOf(1, 3, INPUT_SIZE, INPUT_SIZE)
            if (!inShape.contentEquals(expected)) {
                Log.e(TAG, "Input shape mismatch: got=${inShape.toList()}, " +
                    "expected=${expected.toList()}. Refusing to initialize " +
                    "(legacy asset or wrong artifact?). Heartbeat will emit no demographics.")
                interp.close()
                return false
            }
            val outCount = interp.outputTensorCount
            if (outCount != 2) {
                Log.e(TAG, "Output tensor count mismatch: got=$outCount, expected=2 " +
                    "(age + gender). Refusing to initialize.")
                interp.close()
                return false
            }
            val ageShape = interp.getOutputTensor(0).shape()
            val genderShape = interp.getOutputTensor(1).shape()
            if (!ageShape.contentEquals(intArrayOf(1, NUM_AGE_BINS)) ||
                !genderShape.contentEquals(intArrayOf(1, NUM_GENDER_CLASSES))) {
                Log.e(TAG, "Output shape mismatch: age=${ageShape.toList()}, " +
                    "gender=${genderShape.toList()}, expected=[1,$NUM_AGE_BINS]+[1,$NUM_GENDER_CLASSES]. " +
                    "Refusing to initialize.")
                interp.close()
                return false
            }

            interpreter = interp
            isInitialized = true

            // Pre-allocate inference buffers once
            // 1×3×224×224 float32 = 602,112 bytes
            inputBuffer = ByteBuffer.allocateDirect(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4)
            inputBuffer?.order(ByteOrder.nativeOrder())
            pixelBuffer = IntArray(INPUT_SIZE * INPUT_SIZE)

            // Pre-allocate output arrays once (reused per inference)
            ageOutput = Array(1) { FloatArray(NUM_AGE_BINS) }
            genderOutput = Array(1) { FloatArray(NUM_GENDER_CLASSES) }

            // PR 10 diagnostic: one-shot success breadcrumb so operators can
            // confirm via logcat that the FaceXFormer load gate cleared on a
            // given device — paired with the gate-state diag in
            // AudienceSensingService.aggregateAndEmit (~line 2518).
            val loadedFrom = findOtaModelFile()?.absolutePath ?: "apk_assets:$LEGACY_MODEL_FILE"
            Log.i(TAG, "[AgeGender] Model loaded successfully: tflite=$loadedFrom, inputSize=$MODEL_INPUT_SIZE")
            true
        } catch (e: Exception) {
            Log.w(TAG, "[AgeGender] Model NOT loaded: initialize() threw ${e.javaClass.simpleName}: ${e.message} — " +
                "on_device_demographics will remain null")
            false
        }
    }

    /**
     * Memory-map the model file from its actual on-device location.
     *
     * Phase 2: prefer the OTA-downloaded FaceXFormer artifact over any
     * legacy APK-bundled asset. Older builds that shipped a 96×96 custom
     * classifier under `assets/age_gender_model.tflite` would crash at
     * runtime if we loaded the legacy asset alongside the new 224×224
     * preprocessing pipeline — so we PREFER OTA. If neither exists,
     * throws FileNotFoundException.
     */
    private fun loadModelFile(): MappedByteBuffer {
        // Try OTA-downloaded model first (production path for FaceXFormer)
        val ota = findOtaModelFile()
        if (ota != null && ota.length() > 0) {
            Log.i(TAG, "Loading age/gender model from OTA path: ${ota.absolutePath} " +
                "(${ota.length() / 1024 / 1024} MB)")
            val raf = RandomAccessFile(ota, "r")
            return raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, ota.length())
        }

        // Fall back to APK assets (legacy / dev-time path only)
        try {
            val fd = context.assets.openFd(LEGACY_MODEL_FILE)
            Log.w(TAG, "Loading legacy 96×96 age/gender asset from APK — " +
                "this is incompatible with the 224×224 FaceXFormer preprocessing. " +
                "Inference will fail; heartbeat emits no onDeviceDemographics.")
            val input = FileInputStream(fd.fileDescriptor)
            val channel = input.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        } catch (_: Exception) { /* neither path has a model */ }

        throw java.io.FileNotFoundException(
            "Age/gender model not found in OTA cache or APK assets — " +
            "this should be unreachable since hasModel() returned true"
        )
    }

    /**
     * Fill the pre-allocated input buffer from the 224×224 ARGB crop using
     * FaceXFormer's NCHW + ImageNet preprocessing.
     *
     * Layout: [N=1, C=3, H=224, W=224]. The TFLite interpreter sees the
     * buffer as a flat float32 sequence: all R values first (224×224
     * elements), then all G, then all B. Per-channel normalization is
     * `(pixel/255 - mean) / std` with mean=[0.485,0.456,0.406] and
     * std=[0.229,0.224,0.225].
     */
    private fun fillNchwImageNetBuffer(pixels: IntArray, buffer: ByteBuffer) {
        buffer.rewind()
        // Channel R
        val rMean = IMAGENET_MEAN[0]; val rStd = IMAGENET_STD[0]
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            buffer.putFloat((r - rMean) / rStd)
        }
        // Channel G
        val gMean = IMAGENET_MEAN[1]; val gStd = IMAGENET_STD[1]
        for (pixel in pixels) {
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            buffer.putFloat((g - gMean) / gStd)
        }
        // Channel B
        val bMean = IMAGENET_MEAN[2]; val bStd = IMAGENET_STD[2]
        for (pixel in pixels) {
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat((b - bMean) / bStd)
        }
        buffer.rewind()
    }

    /**
     * Zero-allocation age/gender classification using a pre-allocated crop bitmap.
     *
     * Uses Canvas.drawBitmap to crop+scale the face region directly into the
     * pre-allocated 224x224 bitmap — no Bitmap.createBitmap or createScaledBitmap.
     *
     * The bitmap MUST be 224x224 ARGB_8888 (FaceXFormer input size). The
     * caller (AudienceAnalyzer) sizes the pre-allocated bitmap accordingly.
     *
     * @param sourceBitmap Full camera frame (the shared bitmap)
     * @param faceRect Face bounding box from ML Kit
     * @param preallocatedCrop Pre-allocated 224x224 ARGB_8888 bitmap (owned by caller)
     * @return AgeGenderResult, or null on failure
     */
    fun classifyWithPreallocatedCrop(
        sourceBitmap: Bitmap,
        faceRect: Rect,
        preallocatedCrop: Bitmap
    ): AgeGenderResult? {
        if (!isInitialized || interpreter == null) return null

        return try {
            // Validate caller's pre-allocated bitmap matches model input size
            if (preallocatedCrop.width != INPUT_SIZE || preallocatedCrop.height != INPUT_SIZE) {
                Log.w(TAG, "Pre-allocated crop is ${preallocatedCrop.width}×" +
                    "${preallocatedCrop.height}, expected ${INPUT_SIZE}×${INPUT_SIZE}. " +
                    "Skipping inference — caller must rotate to FaceXFormer size.")
                return null
            }

            // Ensure bounds are within source bitmap
            val left = faceRect.left.coerceIn(0, sourceBitmap.width - 1)
            val top = faceRect.top.coerceIn(0, sourceBitmap.height - 1)
            val right = faceRect.right.coerceIn(left + 1, sourceBitmap.width)
            val bottom = faceRect.bottom.coerceIn(top + 1, sourceBitmap.height)

            val w = right - left
            val h = bottom - top
            if (w <= 10 || h <= 10) return null

            // Crop + scale into pre-allocated bitmap in one Canvas operation — zero allocation
            val srcRect = Rect(left, top, right, bottom)
            val dstRect = Rect(0, 0, INPUT_SIZE, INPUT_SIZE)
            val canvas = Canvas(preallocatedCrop)
            canvas.drawBitmap(sourceBitmap, srcRect, dstRect, null)

            // Fill input buffer from the pre-allocated crop using NCHW + ImageNet
            val buffer = inputBuffer ?: return null
            val pixels = pixelBuffer ?: return null
            preallocatedCrop.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            fillNchwImageNetBuffer(pixels, buffer)

            val localAgeOutput = ageOutput ?: return null
            val localGenderOutput = genderOutput ?: return null

            val outputs = mapOf(
                0 to localAgeOutput,
                1 to localGenderOutput
            )

            interpreter?.runForMultipleInputsOutputs(arrayOf(buffer), outputs)

            parseOutputs(localAgeOutput, localGenderOutput)
        } catch (e: Exception) {
            Log.w(TAG, "Age/gender inference (preallocated) failed: ${e.message}")
            null
        }
    }

    /**
     * Parse pre-allocated output arrays into an AgeGenderResult.
     */
    private fun parseOutputs(
        ageOut: Array<FloatArray>,
        genderOut: Array<FloatArray>
    ): AgeGenderResult {
        val ageSoftmax = softmax(ageOut[0])
        val ageIdx = ageSoftmax.indices.maxByOrNull { ageSoftmax[it] } ?: 0
        val ageRange = AGE_RANGES[ageIdx]
        val ageConfidence = ageSoftmax[ageIdx]

        val genderSoftmax = softmax(genderOut[0])
        val genderIdx = genderSoftmax.indices.maxByOrNull { genderSoftmax[it] } ?: 0
        val gender = GENDER_LABELS[genderIdx]
        val genderConfidence = genderSoftmax[genderIdx]

        val avgConfidence = (ageConfidence + genderConfidence) / 2f

        return AgeGenderResult(
            ageRange = ageRange,
            ageConfidence = ageConfidence,
            gender = gender,
            genderConfidence = genderConfidence,
            confidence = avgConfidence
        )
    }

    /**
     * Accumulate a classification result for the current 10s window.
     */
    fun accumulate(result: AgeGenderResult) {
        val ageIdx = AGE_RANGES.indexOf(result.ageRange)
        if (ageIdx >= 0) ageAccumulator[ageIdx]++

        val genderIdx = GENDER_LABELS.indexOf(result.gender)
        if (genderIdx >= 0) genderAccumulator[genderIdx]++

        totalFacesProcessed++
        totalConfidence += result.confidence
    }

    /**
     * Phase 1c — accumulate alongside the per-face overlay so the
     * AudienceAnalyzer can thread age/gender into PerFaceAttention
     * buckets by ML Kit face track ID. The histogram counters still
     * update (preserves the existing aggregate emit). When the same
     * faceId is sampled multiple times in a window, the latest result
     * wins — older samples are statistically aggregated via the
     * histogram anyway, and keeping a list per face explodes memory on
     * long dwell faces.
     */
    fun accumulateWithFaceId(faceId: Int, result: AgeGenderResult) {
        accumulate(result)
        perFaceLatest[faceId] = result
    }

    /**
     * Phase 1c — read the per-face overlay (latest age/gender result per
     * ML Kit face track ID). Returns a snapshot copy so the caller can
     * iterate without worrying about concurrent mutation.
     */
    fun getPerFaceLatestSnapshot(): Map<Int, AgeGenderResult> {
        val snap = mutableMapOf<Int, AgeGenderResult>()
        for ((k, v) in perFaceLatest) snap[k] = v
        return snap
    }

    /**
     * Get the accumulated age distribution as normalized percentages.
     */
    fun getAgeDistribution(): JSONObject {
        val total = totalFacesProcessed.coerceAtLeast(1)
        return JSONObject().apply {
            AGE_RANGES.forEachIndexed { idx, range ->
                put(range, ageAccumulator[idx].toFloat() / total)
            }
        }
    }

    /**
     * Get the accumulated gender split as normalized percentages.
     */
    fun getGenderSplit(): JSONObject {
        val total = totalFacesProcessed.coerceAtLeast(1)
        return JSONObject().apply {
            GENDER_LABELS.forEachIndexed { idx, label ->
                put(label, genderAccumulator[idx].toFloat() / total)
            }
        }
    }

    /**
     * Get average confidence across all accumulated faces.
     */
    fun getAvgConfidence(): Float {
        return if (totalFacesProcessed > 0) totalConfidence / totalFacesProcessed else 0f
    }

    /**
     * Reset accumulators for a new 10s window.
     */
    fun resetAccumulators() {
        ageAccumulator.fill(0)
        genderAccumulator.fill(0)
        totalFacesProcessed = 0
        totalConfidence = 0f
        // Phase 1c — per-face overlay also resets at window boundaries.
        perFaceLatest.clear()
    }

    /**
     * Check if any faces have been processed in this window.
     */
    fun hasFaces(): Boolean = totalFacesProcessed > 0

    fun close() {
        interpreter?.close()
        interpreter = null
        inputBuffer = null
        pixelBuffer = null
        ageOutput = null
        genderOutput = null
        isInitialized = false
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }
        val sum = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }
}

/**
 * Result of age/gender classification for a single face.
 */
data class AgeGenderResult(
    val ageRange: String,     // "18-24", "25-34", etc.
    val ageConfidence: Float,
    val gender: String,       // "male" or "female"
    val genderConfidence: Float,
    val confidence: Float     // Average of age + gender confidence
)
