package com.trillboards.ctv.core.audience

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.inference.DelegateSelector
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import java.nio.ByteBuffer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Local TFLite-based person detection processor using MediaPipe Object Detector.
 *
 * Privacy-Preserving Features:
 * - Processes camera frames in-memory only.
 * - Never stores or transmits raw images/frames.
 * - Only extracts anonymous count and bounding box metadata.
 * - Discards frame data immediately after inference.
 *
 * Lightweight for Android TV:
 * - Uses EfficientDet-Lite0 TFLite model (optimized for mobile CPU/GPU).
 * - Configurable detection threshold and category filtering.
 * - Supports hardware acceleration via GPU delegate when available.
 */
class PersonDetectionProcessor(
    private val context: Context,
    private val threshold: Float = SensingConfig.get().detection.personConfidenceThreshold
) {
    companion object {
        private const val TAG = "PersonDetection"
        private const val MODEL_FILE = "efficientdet_lite0.tflite"
        private const val CATEGORY_PERSON = "person"
    }

    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var objectDetector: ObjectDetector? = null
    private var isInitialized = false

    private val _currentPersonCount = MutableStateFlow(0)
    val currentPersonCount: StateFlow<Int> = _currentPersonCount

    /**
     * Check if the model file exists in assets.
     */
    fun hasModel(): Boolean {
        return try {
            context.assets.open(MODEL_FILE).close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Person detection model not found: $MODEL_FILE")
            false
        }
    }

    /**
     * Initialize the MediaPipe Object Detector.
     * Uses DelegateSelector to pick the optimal delegate for the current chipset,
     * with automatic fallback to CPU if the primary delegate fails.
     *
     * Includes runtime inference validation: some GPU drivers (notably Samsung
     * Android 16) pass initialization but fail at inference time in the
     * ToTensorConverter. A test inference with a synthetic bitmap catches this
     * during init, so the fallback to CPU happens before any real frames arrive.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        if (!hasModel()) {
            Log.e(TAG, "Model file $MODEL_FILE not found in assets. Cannot initialize PersonDetectionProcessor.")
            return false
        }

        return try {
            val profile = DeviceProfile.detect(context)
            val recommendation = DelegateSelector.recommendAndLog(profile.chipsetVendor, "vision")

            val delegates = if (recommendation.primary != recommendation.fallback) {
                listOf(recommendation.primary, recommendation.fallback)
            } else {
                listOf(recommendation.primary)
            }

            for (delegate in delegates) {
                try {
                    val baseOptions = BaseOptions.builder()
                        .setModelAssetPath(MODEL_FILE)
                        .setDelegate(delegate)
                        .build()

                    val options = ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(baseOptions)
                        .setScoreThreshold(threshold)
                        .setRunningMode(RunningMode.IMAGE)
                        .setCategoryAllowlist(listOf(CATEGORY_PERSON))
                        .build()

                    val detector = ObjectDetector.createFromOptions(context, options)

                    // Validate with a test inference — catches GPU drivers that initialize
                    // successfully but fail at runtime (Samsung Android 16 ToTensorConverter)
                    val testBitmap = Bitmap.createBitmap(320, 320, Bitmap.Config.ARGB_8888)
                    try {
                        val testImage = BitmapImageBuilder(testBitmap).build()
                        try {
                            detector.detect(testImage)
                        } finally {
                            testImage.close()
                        }
                        Log.i(TAG, "Delegate $delegate passed test inference")
                    } finally {
                        testBitmap.recycle()
                    }

                    objectDetector = detector
                    isInitialized = true
                    Log.i(TAG, "PersonDetectionProcessor initialized with delegate=$delegate, model=$MODEL_FILE")
                    return true
                } catch (e: Exception) {
                    Log.w(TAG, "Delegate $delegate failed (init or test inference): ${e.message}")
                    // Continue to next delegate
                }
            }

            Log.e(TAG, "All delegates failed for PersonDetectionProcessor")
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ObjectDetector", e)
            false
        }
    }

    /**
     * Process a single frame bitmap and count persons.
     *
     * IMPORTANT: MediaPipe's `BitmapImageContainer.close()` (called via
     * `mpImage.close()`) calls `recycle()` on the source bitmap. If we passed
     * the caller's bitmap directly, it would be recycled out from under the
     * caller — which is the AudienceAnalyzer's pre-allocated zero-alloc
     * sharedBitmap. That broke the entire zero-allocation frame pipeline:
     * every frame's `imageProxyToSharedBitmap` saw a recycled bitmap, called
     * `releaseBuffers` + `initializeBuffers` (allocating a fresh 512×384
     * ARGB_8888 Bitmap = ~786 KB per frame at 5–10 Hz, ~4–8 MB/sec of GC
     * churn), and AgeGenderProcessor.hasFaces() never accumulated because
     * the face crop bitmap also got reallocated each frame.
     *
     * Fix: pass a defensive copy to MediaPipe. The copy gets recycled by
     * mpImage.close(); the caller's bitmap stays alive for the next frame.
     * The per-call Bitmap.copy() is ~786 KB at typical 512×384 ARGB_8888,
     * which is well under the per-call inference cost and far less than
     * the GC churn the old (broken) zero-alloc path was paying.
     *
     * @param bitmap The camera frame in-memory bitmap (NOT recycled by this call).
     * @return Number of persons detected.
     */
    fun process(bitmap: Bitmap): Int {
        if (!isInitialized || objectDetector == null) return 0

        return try {
            val frameCopy = try {
                bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to copy bitmap for person detection: ${e.message}")
                return 0
            }

            val result = try {
                val mpImage = BitmapImageBuilder(frameCopy).build()
                try {
                    objectDetector?.detect(mpImage)
                } finally {
                    // mpImage.close() calls BitmapImageContainer.close() which
                    // calls frameCopy.recycle(). The shared bitmap is unaffected
                    // because we passed a copy.
                    mpImage.close()
                }
            } finally {
                // Defensive: if MediaPipe's BitmapImageContainer ever stops
                // recycling on close (API change), recycle the throwaway here
                // so we don't leak the 786 KB per call.
                if (!frameCopy.isRecycled) {
                    frameCopy.recycle()
                }
            }

            val count = result?.detections()?.size ?: 0
            _currentPersonCount.value = count

            if (count > 0) {
                Log.d(TAG, "Detected $count person(s) in frame")
            }

            count
        } catch (e: Exception) {
            Log.e(TAG, "Error during person detection inference", e)
            0
        }
    }

    /**
     * Zero-allocation variant of [process] using a caller-owned RGB ByteBuffer.
     *
     * [ByteBufferImageBuilder] wraps the buffer without copying and `close()` frees
     * only the wrapper, so unlike [process] — which pays a per-frame `Bitmap.copy()`
     * (~786 KB) because [BitmapImageBuilder] recycles its source — this allocates
     * nothing per frame. Output is identical to [process] (same detector, count).
     *
     * @param rgbByteBuffer Pre-filled RGB bytes (3 per pixel, 0-255); width*height*3.
     * @return Number of persons detected.
     */
    fun processWithBuffer(rgbByteBuffer: ByteBuffer, width: Int, height: Int): Int {
        if (!isInitialized || objectDetector == null) return 0

        return try {
            rgbByteBuffer.rewind()
            val mpImage = ByteBufferImageBuilder(
                rgbByteBuffer, width, height, MPImage.IMAGE_FORMAT_RGB
            ).build()
            val result = try {
                objectDetector?.detect(mpImage)
            } finally {
                // Frees only the MPImage wrapper — the caller's buffer survives.
                mpImage.close()
            }

            val count = result?.detections()?.size ?: 0
            _currentPersonCount.value = count
            count
        } catch (e: Exception) {
            Log.e(TAG, "processWithBuffer error: ${e.message}", e)
            0
        }
    }

    /**
     * Process frame asynchronously.
     */
    fun processAsync(bitmap: Bitmap, onResult: (Int) -> Unit) {
        processorScope.launch {
            val count = process(bitmap)
            onResult(count)
        }
    }

    /**
     * Release resources.
     */
    fun release() {
        objectDetector?.close()
        objectDetector = null
        isInitialized = false
        Log.i(TAG, "PersonDetectionProcessor released")
    }

    fun isReady(): Boolean = isInitialized
}
