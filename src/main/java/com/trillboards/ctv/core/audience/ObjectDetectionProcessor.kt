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
 * Multi-class object detection processor using MediaPipe Object Detector
 * with EfficientDet-Lite0 (same model as PersonDetectionProcessor, but
 * configured for all 80 COCO classes instead of just "person").
 *
 * Privacy-Preserving Features:
 * - Processes camera frames in-memory only.
 * - Never stores or transmits raw images/frames.
 * - Only extracts anonymous object counts and class labels.
 * - Discards frame data immediately after inference.
 *
 * Used by: VehicleFlowProcessor (vehicle counts), QueueEstimationProcessor
 * (person counts in queue context), scene composition analysis.
 *
 * Thread-safe: synchronized inference, StateFlow result updates.
 */
class ObjectDetectionProcessor(
    private val context: Context,
    private val targetClasses: List<String> = emptyList(),
    private val minConfidence: Float = SensingConfig.get().detection.personConfidenceThreshold
) {
    companion object {
        private const val TAG = "ObjectDetection"
        private const val MODEL_FILE = "efficientdet_lite0.tflite"

        // Vehicle-related COCO classes
        val VEHICLE_CLASSES = setOf("car", "truck", "bus", "motorcycle", "bicycle")

        // Commerce/retail COCO classes (useful for store context)
        val COMMERCE_CLASSES = setOf(
            "bottle", "cup", "bowl", "handbag", "suitcase",
            "backpack", "umbrella", "cell_phone", "laptop", "book"
        )

        // All 80 COCO class names for label mapping
        val COCO_CLASSES = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
            "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
            "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
            "hair_drier", "toothbrush"
        )
    }

    /**
     * A single detected object with class, confidence, and bounding box.
     */
    data class Detection(
        val className: String,
        val classId: Int,
        val confidence: Float,
        val bbox: FloatArray // [top, left, bottom, right] normalized 0-1
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Detection) return false
            return className == other.className && classId == other.classId &&
                confidence == other.confidence && bbox.contentEquals(other.bbox)
        }

        override fun hashCode(): Int {
            var result = className.hashCode()
            result = 31 * result + classId
            result = 31 * result + confidence.hashCode()
            result = 31 * result + bbox.contentHashCode()
            return result
        }
    }

    /**
     * Aggregated detection result for a single frame.
     */
    data class DetectionResult(
        val detections: List<Detection>,
        val objectCounts: Map<String, Int>,
        val totalObjects: Int,
        val vehicleCount: Int,
        val inferenceTimeMs: Long
    )

    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var objectDetector: ObjectDetector? = null
    @Volatile private var isInitialized = false

    private val _lastResult = MutableStateFlow(DetectionResult(emptyList(), emptyMap(), 0, 0, 0))
    val lastResult: StateFlow<DetectionResult> = _lastResult

    /**
     * Check if the model file exists in assets.
     */
    fun hasModel(): Boolean {
        return try {
            context.assets.open(MODEL_FILE).close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Object detection model not found: $MODEL_FILE")
            false
        }
    }

    /**
     * Initialize the MediaPipe Object Detector for multi-class detection.
     * Uses DelegateSelector to pick the optimal delegate for the current chipset,
     * with automatic fallback to CPU if the primary delegate fails.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        if (!hasModel()) {
            Log.e(TAG, "Model file $MODEL_FILE not found in assets. Cannot initialize ObjectDetectionProcessor.")
            return false
        }

        return try {
            val profile = DeviceProfile.detect(context)
            val recommendation = DelegateSelector.recommendAndLog(profile.chipsetVendor, "vision")

            DelegateSelector.withFallback(
                primary = recommendation.primary,
                fallback = recommendation.fallback
            ) { delegate ->
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath(MODEL_FILE)
                    .setDelegate(delegate)
                    .build()

                val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setScoreThreshold(minConfidence)
                    .setRunningMode(RunningMode.IMAGE)
                    .setMaxResults(25)

                // If target classes specified, filter to only those
                if (targetClasses.isNotEmpty()) {
                    optionsBuilder.setCategoryAllowlist(targetClasses)
                }

                objectDetector = ObjectDetector.createFromOptions(context, optionsBuilder.build())
            }

            isInitialized = true
            Log.i(TAG, "ObjectDetectionProcessor initialized with $MODEL_FILE " +
                "(targetClasses=${if (targetClasses.isEmpty()) "all" else targetClasses.joinToString()})")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ObjectDetector", e)
            false
        }
    }

    /**
     * Run object detection on a camera frame.
     * Returns detected objects filtered by target classes and confidence threshold.
     *
     * @param bitmap The camera frame in-memory bitmap.
     * @return DetectionResult with all detected objects and aggregated counts.
     */
    @Synchronized
    fun detect(bitmap: Bitmap): DetectionResult {
        val startMs = System.currentTimeMillis()

        if (!isInitialized || objectDetector == null) {
            return DetectionResult(emptyList(), emptyMap(), 0, 0, 0)
        }

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = try {
                objectDetector?.detect(mpImage)
            } finally {
                mpImage.close()
            }

            val detections = parseDetections(result)

            // Aggregate counts by class name
            val objectCounts = detections.groupBy { it.className }
                .mapValues { it.value.size }

            val vehicleCount = detections.count { it.className in VEHICLE_CLASSES }

            val inferenceTimeMs = System.currentTimeMillis() - startMs

            val detectionResult = DetectionResult(
                detections = detections,
                objectCounts = objectCounts,
                totalObjects = detections.size,
                vehicleCount = vehicleCount,
                inferenceTimeMs = inferenceTimeMs
            )

            _lastResult.value = detectionResult

            if (detections.isNotEmpty()) {
                Log.d(TAG, "Detected ${detections.size} objects " +
                    "(vehicles=$vehicleCount) in ${inferenceTimeMs}ms: " +
                    objectCounts.entries.joinToString { "${it.key}=${it.value}" })
            }

            detectionResult
        } catch (e: Exception) {
            Log.e(TAG, "Error during object detection inference", e)
            DetectionResult(emptyList(), emptyMap(), 0, 0, System.currentTimeMillis() - startMs)
        }
    }

    /**
     * Parse MediaPipe ObjectDetectorResult into our Detection data class.
     */
    private fun parseDetections(result: ObjectDetectorResult?): List<Detection> {
        if (result == null) return emptyList()

        return result.detections().mapNotNull { detection ->
            val category = detection.categories().firstOrNull() ?: return@mapNotNull null
            val className = category.categoryName() ?: return@mapNotNull null
            val confidence = category.score()

            if (confidence < minConfidence) return@mapNotNull null

            // Map class name to COCO class index
            val classId = COCO_CLASSES.indexOf(className)

            // Extract bounding box (normalized coordinates)
            val box = detection.boundingBox()
            val bbox = floatArrayOf(
                box.top.toFloat() / 1f,    // Already normalized by MediaPipe
                box.left.toFloat() / 1f,
                box.bottom.toFloat() / 1f,
                box.right.toFloat() / 1f
            )

            Detection(
                className = className,
                classId = if (classId >= 0) classId else -1,
                confidence = confidence,
                bbox = bbox
            )
        }
    }

    /**
     * Process frame asynchronously with callback.
     */
    fun detectAsync(bitmap: Bitmap, onResult: (DetectionResult) -> Unit) {
        processorScope.launch {
            val result = detect(bitmap)
            onResult(result)
        }
    }

    /**
     * Release resources.
     */
    fun release() {
        objectDetector?.close()
        objectDetector = null
        isInitialized = false
        Log.i(TAG, "ObjectDetectionProcessor released")
    }

    fun isReady(): Boolean = isInitialized
}
