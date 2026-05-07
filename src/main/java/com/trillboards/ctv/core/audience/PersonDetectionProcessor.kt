package com.trillboards.ctv.core.audience

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.inference.DelegateSelector
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
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
     * @param bitmap The camera frame in-memory bitmap.
     * @return Number of persons detected.
     */
    fun process(bitmap: Bitmap): Int {
        if (!isInitialized || objectDetector == null) return 0

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = try {
                objectDetector?.detect(mpImage)
            } finally {
                mpImage.close()
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
