package com.trillboards.ctv.core.audience

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Base64
import android.util.Log
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import android.os.Handler
import android.os.Looper
import com.trillboards.ctv.core.SensingConfig
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Manages frame capture for Gemini Vision demographics analysis.
 *
 * Features:
 * - Captures frames only when faces are detected
 * - Rate-limited to max 1 capture per 5 minutes
 * - Compresses to 640x480 JPEG for cost optimization
 * - Sends base64-encoded frames to API for Gemini analysis
 * - Privacy-preserving: frames are processed and discarded immediately
 *
 * IMPORTANT: This class does NOT manage camera binding. The ImageCapture use case
 * must be provided by AudienceAnalyzer which manages the shared camera lifecycle.
 *
 * Usage:
 * ```kotlin
 * val captureManager = FrameCaptureManager(context, apiBaseUrl, fingerprint)
 * captureManager.setImageCapture(sharedImageCapture)  // From AudienceAnalyzer
 *
 * // Call when faces are detected
 * captureManager.onFacesDetected(faceCount = 3)
 * ```
 */
class FrameCaptureManager(
    private val context: Context,
    private val apiBaseUrl: String,
    private val fingerprint: String,
    private val deviceTokenProvider: () -> String? = { null }
) {
    companion object {
        private const val TAG = "FrameCaptureManager"

        // Delegated to SensingConfig.capture — public for AudienceAnalyzer camera setup
        private val captureCfg get() = SensingConfig.get().capture
        val MAX_IMAGE_WIDTH get() = captureCfg.maxImageWidth
        val MAX_IMAGE_HEIGHT get() = captureCfg.maxImageHeight
        val JPEG_QUALITY get() = captureCfg.jpegQuality

        // Internal rate-limiting / timeout constants
        private val MIN_CAPTURE_INTERVAL_MS get() = captureCfg.minCaptureIntervalMs
        private val CAMERA_WARMUP_MS get() = captureCfg.cameraWarmupMs
        private val DIAGNOSTIC_CAPTURE_INTERVAL_MS get() = captureCfg.diagnosticCaptureIntervalMs
        private val CONNECT_TIMEOUT_MS get() = captureCfg.connectTimeoutMs
        private val READ_TIMEOUT_MS get() = captureCfg.readTimeoutMs
    }

    private val captureExecutor = Executors.newSingleThreadExecutor()
    private val networkScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // ImageCapture is provided externally by AudienceAnalyzer (shared camera lifecycle)
    private var imageCapture: ImageCapture? = null

    private val lastCaptureTime = AtomicLong(0)
    private val lastDiagnosticCaptureTime = AtomicLong(0)
    private val isCapturing = AtomicBoolean(false)
    private val isInitialized = AtomicBoolean(false)

    // Periodic scene capture (no face required) — every 5 minutes
    private var periodicCaptureHandler: Handler? = null
    private val periodicCaptureRunnable = object : Runnable {
        override fun run() {
            if (isCapturing.get()) {
                // Retry in 30s if capture in progress
                periodicCaptureHandler?.postDelayed(this, MIN_CAPTURE_INTERVAL_MS)
                return
            }
            if (!isInitialized.get() || imageCapture == null) {
                periodicCaptureHandler?.postDelayed(this, SensingConfig.get().capture.periodicIntervalMs)
                return
            }

            val now = System.currentTimeMillis()
            val lastCapture = lastCaptureTime.get()
            val timeSince = now - lastCapture
            // Only fire if enough time passed since last capture (face-triggered or periodic)
            if (lastCapture > 0 && timeSince < MIN_CAPTURE_INTERVAL_MS) {
                periodicCaptureHandler?.postDelayed(this, MIN_CAPTURE_INTERVAL_MS - timeSince + 1000)
                return
            }

            if (!isCapturing.compareAndSet(false, true)) {
                periodicCaptureHandler?.postDelayed(this, MIN_CAPTURE_INTERVAL_MS)
                return
            }

            Log.i(TAG, "[FrameCapture] Periodic scene capture (no face required)")
            currentCaptureMode = "periodic_scene"
            currentCaptureReason = "periodic_scene_baseline"
            captureFrame()
            periodicCaptureHandler?.postDelayed(this, SensingConfig.get().capture.periodicIntervalMs)
        }
    }

    // Track current capture mode for payload tagging
    @Volatile private var currentCaptureMode: String = "face_triggered"
    @Volatile private var currentCaptureReason: String? = null

    // Screen ID for API calls
    @Volatile private var screenId: String? = null

    // Callback for demographics results
    var onDemographicsReceived: ((DemographicsResult) -> Unit)? = null

    // Callback for errors
    var onError: ((String) -> Unit)? = null

    // Runtime telemetry provider injected by AudienceSensingService
    var edgeQualityProvider: (() -> EdgeQualityTelemetry?)? = null

    // Edge intelligence context — VLM + TFLite signals for Gemini cloud enrichment
    var edgeIntelligenceProvider: (() -> JSONObject?)? = null

    /**
     * Set the ImageCapture use case from AudienceAnalyzer.
     * This is required because CameraX only allows one camera binding,
     * and AudienceAnalyzer manages the shared camera lifecycle.
     */
    fun setImageCapture(capture: ImageCapture) {
        imageCapture = capture
        isInitialized.set(true)
        Log.i(TAG, "ImageCapture set from AudienceAnalyzer - ready for frame capture")
    }

    /**
     * Check if frame capture is ready.
     */
    fun isReady(): Boolean = isInitialized.get() && imageCapture != null

    /**
     * Set the screen ID for API calls.
     */
    fun setScreenId(id: String) {
        screenId = id
        // Seed lastCaptureTime so the rate limiter enforces a warmup delay before first capture.
        // This makes the rate limit check (lastCapture > 0 && timeSince < MIN_INTERVAL) true
        // for the first CAMERA_WARMUP_MS, giving the camera time to stabilize.
        val warmupSeed = System.currentTimeMillis() - MIN_CAPTURE_INTERVAL_MS + CAMERA_WARMUP_MS
        lastCaptureTime.set(warmupSeed)
        Log.i(TAG, "[FrameCapture] Screen ID set: $id (first capture in ~${CAMERA_WARMUP_MS / 1000}s after warmup)")

        // Start periodic scene capture — captures every 5 min regardless of face detection
        startPeriodicCapture()
    }

    private fun startPeriodicCapture() {
        periodicCaptureHandler?.removeCallbacks(periodicCaptureRunnable)
        periodicCaptureHandler = Handler(Looper.getMainLooper())
        val interval = SensingConfig.get().capture.periodicIntervalMs
        periodicCaptureHandler?.postDelayed(periodicCaptureRunnable, interval)
        Log.i(TAG, "[FrameCapture] Periodic scene capture started (every ${interval / 1000}s)")
    }

    /**
     * Called when faces are detected.
     * Will trigger a frame capture if rate limit allows.
     *
     * @param faceCount Number of faces currently detected
     */
    fun onFacesDetected(faceCount: Int) {
        if (faceCount <= 0) {
            Log.v(TAG, "[FrameCapture] No faces detected, skipping")
            return
        }

        if (!isInitialized.get() || imageCapture == null) {
            Log.w(TAG, "[FrameCapture] Not ready for capture (faces=$faceCount) - " +
                    "initialized=${isInitialized.get()}, imageCapture=${imageCapture != null}. " +
                    "Waiting for onCameraReady callback from AudienceAnalyzer.")
            return
        }

        val now = System.currentTimeMillis()
        val lastCapture = lastCaptureTime.get()
        val timeSinceLastCapture = now - lastCapture

        // Check rate limit
        if (lastCapture > 0 && timeSinceLastCapture < MIN_CAPTURE_INTERVAL_MS) {
            val remainingSeconds = (MIN_CAPTURE_INTERVAL_MS - timeSinceLastCapture) / 1000
            Log.d(TAG, "[FrameCapture] Rate limited - ${remainingSeconds}s until next capture (faces=$faceCount)")
            return
        }

        // Prevent concurrent captures
        if (!isCapturing.compareAndSet(false, true)) {
            Log.d(TAG, "[FrameCapture] Capture already in progress, skipping")
            return
        }

        Log.i(TAG, "[FrameCapture] >>> TRIGGERING CAPTURE (faces=$faceCount, timeSinceLast=${timeSinceLastCapture}ms, screenId=$screenId)")
        currentCaptureMode = "face_triggered"
        currentCaptureReason = "face_detected"
        captureFrame()
    }

    /**
     * Request a diagnostic probe capture when edge proxy signals and face sensing diverge.
     * This remains rate-limited to prevent runaway Gemini spend on weak hardware.
     */
    fun requestDiagnosticCapture(reason: String): Boolean {
        if (!isInitialized.get() || imageCapture == null) {
            Log.w(TAG, "[FrameCapture] Diagnostic capture skipped - not ready ($reason)")
            return false
        }

        val now = System.currentTimeMillis()
        val lastCapture = lastCaptureTime.get()
        val lastDiagnosticCapture = lastDiagnosticCaptureTime.get()

        if (lastCapture > 0 && now - lastCapture < MIN_CAPTURE_INTERVAL_MS) {
            Log.d(TAG, "[FrameCapture] Diagnostic capture rate-limited by global interval ($reason)")
            return false
        }

        if (lastDiagnosticCapture > 0 && now - lastDiagnosticCapture < DIAGNOSTIC_CAPTURE_INTERVAL_MS) {
            Log.d(TAG, "[FrameCapture] Diagnostic capture cooldown active ($reason)")
            return false
        }

        if (!isCapturing.compareAndSet(false, true)) {
            Log.d(TAG, "[FrameCapture] Diagnostic capture already in progress")
            return false
        }

        lastDiagnosticCaptureTime.set(now)
        currentCaptureMode = "signal_mismatch_probe"
        currentCaptureReason = reason
        Log.i(TAG, "[FrameCapture] >>> DIAGNOSTIC CAPTURE requested ($reason)")
        captureFrame()
        return true
    }

    /**
     * Force a frame capture regardless of rate limiting.
     * Use sparingly - mainly for testing/debugging.
     */
    fun forceCaptureNow() {
        if (!isInitialized.get() || imageCapture == null) {
            Log.w(TAG, "[FrameCapture] Not ready for forced capture")
            return
        }

        if (!isCapturing.compareAndSet(false, true)) {
            Log.d(TAG, "[FrameCapture] Capture already in progress")
            return
        }

        Log.i(TAG, "[FrameCapture] >>> FORCING CAPTURE (bypassing rate limit)")
        currentCaptureMode = "manual_force"
        currentCaptureReason = "manual_force"
        captureFrame()
    }

    private fun captureFrame() {
        val capture = imageCapture
        if (capture == null) {
            Log.e(TAG, "[FrameCapture] ERROR: ImageCapture not available - was setImageCapture() called?")
            isCapturing.set(false)
            return
        }

        Log.d(TAG, "[FrameCapture] Calling takePicture()...")
        capture.takePicture(
            captureExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val captureStartTime = System.currentTimeMillis()
                    try {
                        lastCaptureTime.set(captureStartTime)

                        Log.d(TAG, "[FrameCapture] Image captured - format=${image.format}, " +
                                "size=${image.width}x${image.height}, rotation=${image.imageInfo.rotationDegrees}°, " +
                                "planes=${image.planes.size}")

                        // Convert ImageProxy to compressed JPEG bytes
                        val jpegBytes = imageProxyToJpegBytes(image)
                        image.close()

                        if (jpegBytes != null) {
                            val conversionTime = System.currentTimeMillis() - captureStartTime
                            Log.i(TAG, "[FrameCapture] Frame captured successfully - size=${jpegBytes.size} bytes, conversion=${conversionTime}ms")
                            sendToApi(jpegBytes)
                        } else {
                            Log.e(TAG, "[FrameCapture] ERROR: Failed to convert image to JPEG (returned null)")
                            onError?.invoke("Failed to convert image to JPEG")
                            isCapturing.set(false)
                        }

                    } catch (e: Exception) {
                        Log.e(TAG, "[FrameCapture] ERROR processing captured frame: ${e.message}", e)
                        image.close()
                        onError?.invoke("Processing error: ${e.message}")
                        isCapturing.set(false)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "[FrameCapture] ERROR: takePicture failed - code=${exception.imageCaptureError}, message=${exception.message}", exception)
                    onError?.invoke("Capture failed: ${exception.message}")
                    isCapturing.set(false)
                }
            }
        )
    }

    /**
     * Convert ImageProxy (YUV_420_888 or JPEG) to compressed JPEG bytes.
     *
     * CameraX ImageCapture can return different formats depending on device:
     * - JPEG (ImageFormat.JPEG = 256): Direct use
     * - YUV_420_888 (ImageFormat.YUV_420_888 = 35): Needs conversion
     */
    private fun imageProxyToJpegBytes(image: ImageProxy): ByteArray? {
        var bitmap: Bitmap? = null
        return try {
            val format = image.format
            Log.d(TAG, "[FrameCapture] Converting image format=$format (JPEG=256, YUV=35)")

            bitmap = when (format) {
                ImageFormat.JPEG -> {
                    // Direct JPEG - just decode the buffer
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    Log.d(TAG, "[FrameCapture] Decoding JPEG buffer (${bytes.size} bytes)")
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                }
                ImageFormat.YUV_420_888 -> {
                    // YUV format - need to convert to NV21 then to JPEG
                    Log.d(TAG, "[FrameCapture] Converting YUV_420_888 to JPEG")
                    yuvToJpegBitmap(image)
                }
                else -> {
                    Log.w(TAG, "[FrameCapture] Unexpected image format: $format, attempting NV21 conversion")
                    yuvToJpegBitmap(image)
                }
            }

            if (bitmap == null) {
                Log.e(TAG, "[FrameCapture] ERROR: Bitmap conversion returned null")
                return null
            }

            Log.d(TAG, "[FrameCapture] Bitmap created: ${bitmap.width}x${bitmap.height}")

            // Rotate if needed — recycle intermediate bitmaps to prevent leaks
            val rotationDegrees = image.imageInfo.rotationDegrees
            if (rotationDegrees != 0) {
                Log.d(TAG, "[FrameCapture] Rotating bitmap by $rotationDegrees degrees")
                val matrix = Matrix().apply {
                    postRotate(rotationDegrees.toFloat())
                }
                val rotated = Bitmap.createBitmap(bitmap!!, 0, 0, bitmap!!.width, bitmap!!.height, matrix, true)
                if (rotated !== bitmap) bitmap!!.recycle()
                bitmap = rotated
            }

            // Scale down if larger than target — recycle pre-scaled bitmap
            if (bitmap!!.width > MAX_IMAGE_WIDTH || bitmap!!.height > MAX_IMAGE_HEIGHT) {
                val scale = minOf(
                    MAX_IMAGE_WIDTH.toFloat() / bitmap!!.width,
                    MAX_IMAGE_HEIGHT.toFloat() / bitmap!!.height
                )
                val newWidth = (bitmap!!.width * scale).toInt()
                val newHeight = (bitmap!!.height * scale).toInt()
                Log.d(TAG, "[FrameCapture] Scaling bitmap from ${bitmap!!.width}x${bitmap!!.height} to ${newWidth}x${newHeight}")
                val scaled = Bitmap.createScaledBitmap(bitmap!!, newWidth, newHeight, true)
                if (scaled !== bitmap) bitmap!!.recycle()
                bitmap = scaled
            }

            // Compress to JPEG
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, outputStream)
            val result = outputStream.toByteArray()
            bitmap.recycle()
            Log.d(TAG, "[FrameCapture] Final JPEG size: ${result.size} bytes")
            result

        } catch (e: Exception) {
            bitmap?.recycle()
            Log.e(TAG, "[FrameCapture] ERROR converting image: ${e.message}", e)
            null
        }
    }

    /**
     * Convert YUV_420_888 ImageProxy to Bitmap via NV21 format.
     */
    private fun yuvToJpegBitmap(image: ImageProxy): Bitmap? {
        try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            // NV21 format: Y plane followed by interleaved VU
            val nv21 = ByteArray(ySize + uSize + vSize)

            // Copy Y plane
            yBuffer.get(nv21, 0, ySize)

            // Copy VU planes (interleaved for NV21)
            val vBytes = ByteArray(vSize)
            val uBytes = ByteArray(uSize)
            vBuffer.get(vBytes)
            uBuffer.get(uBytes)

            // Interleave V and U for NV21
            for (i in 0 until minOf(vSize, uSize)) {
                nv21[ySize + i * 2] = vBytes[i]
                nv21[ySize + i * 2 + 1] = uBytes[i]
            }

            // Create YuvImage and compress to JPEG
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val outputStream = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), JPEG_QUALITY, outputStream)
            val jpegBytes = outputStream.toByteArray()

            Log.d(TAG, "[FrameCapture] YUV→NV21→JPEG conversion complete (${jpegBytes.size} bytes)")

            // Decode JPEG to Bitmap
            return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

        } catch (e: Exception) {
            Log.e(TAG, "[FrameCapture] ERROR in YUV conversion: ${e.message}", e)
            return null
        }
    }

    /**
     * Send the captured frame to the API for Gemini analysis.
     */
    private fun sendToApi(jpegBytes: ByteArray) {
        networkScope.launch {
            val apiStartTime = System.currentTimeMillis()
            try {
                val base64Image = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
                val base64Size = base64Image.length
                val effectiveEdgeQuality = edgeQualityProvider
                    ?.invoke()
                    ?.let { ObservationSignalClassifier.applyCaptureModeOverride(it, currentCaptureMode) }

                val payload = JSONObject().apply {
                    put("fingerprint", fingerprint)
                    put("screenId", screenId)
                    put("imageBase64", base64Image)
                    put("capturedAt", System.currentTimeMillis())
                    put("imageWidth", MAX_IMAGE_WIDTH)
                    put("imageHeight", MAX_IMAGE_HEIGHT)
                    put("captureMode", currentCaptureMode)
                    effectiveEdgeQuality?.let { edgeQuality ->
                        put("observationFamily", edgeQuality.observationFamily)
                        put("evidenceGrade", edgeQuality.evidenceGrade)
                        put("decisionability", edgeQuality.decisionability)
                        put("decisionBlockReasons", org.json.JSONArray(edgeQuality.decisionBlockReasons))
                        put("edgeQuality", edgeQuality.copy(
                            captureDiagnostics = edgeQuality.captureDiagnostics.copy(
                                reason = currentCaptureReason ?: edgeQuality.captureDiagnostics.reason,
                                captureReady = isReady(),
                                captureInProgress = isCapturing.get(),
                                nextAllowedCaptureInMs = getTimeUntilNextCapture()
                            )
                        ).toJson())
                    }
                    // Edge intelligence context — VLM + TFLite signals for Gemini cloud enrichment
                    edgeIntelligenceProvider?.invoke()?.let { edgeCtx ->
                        put("edgeContext", edgeCtx)
                    }
                }

                val endpoint = "$apiBaseUrl/v2/earner/audience-analyze"
                Log.i(TAG, "[FrameCapture] >>> SENDING TO API - endpoint=$endpoint, " +
                        "imageSize=${jpegBytes.size}b, base64Size=${base64Size}b, screenId=$screenId")

                val url = URL(endpoint)
                val connection = url.openConnection() as HttpURLConnection

                connection.apply {
                    requestMethod = "POST"
                    setRequestProperty("Content-Type", "application/json")
                    setRequestProperty("Accept", "application/json")
                    connectTimeout = CONNECT_TIMEOUT_MS
                    readTimeout = READ_TIMEOUT_MS
                    doOutput = true
                }

                // Device-request auth headers (matches ApiClient.kt pattern). The /v2/earner/audience-analyze
                // route now requires X-Device-Token via requireDeviceRequestAuth middleware. Replay-guard
                // headers (timestamp + nonce) are validated server-side; sending both forms of the token
                // (X-Device-Token + X-Trillboard-Device-Token) for backward compat with the alias header.
                deviceTokenProvider().orEmpty().trim().takeIf { it.isNotEmpty() }?.let { token ->
                    connection.setRequestProperty("X-Device-Token", token)
                    connection.setRequestProperty("X-Trillboard-Device-Token", token)
                    connection.setRequestProperty("X-Device-Timestamp", System.currentTimeMillis().toString())
                    connection.setRequestProperty("X-Device-Nonce", UUID.randomUUID().toString())
                }

                // Send request
                connection.outputStream.bufferedWriter().use { writer ->
                    writer.write(payload.toString())
                }

                // Read response
                val responseCode = connection.responseCode
                val requestTime = System.currentTimeMillis() - apiStartTime

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val responseBody = connection.inputStream.bufferedReader().readText()
                    val response = JSONObject(responseBody)

                    // Log enhanced demographics fields
                    val incomeLevel = response.optJSONObject("incomeSignals")?.optString("estimatedLevel") ?: "n/a"
                    val segments = response.optJSONArray("lifestyleSegments")
                    val segmentNames = (0 until (segments?.length() ?: 0)).mapNotNull {
                        segments?.optJSONObject(it)?.optString("segment")
                    }.joinToString(", ").ifEmpty { "n/a" }
                    val groupComp = response.optJSONObject("groupComposition")
                    val purchaseIntent = response.optJSONObject("purchaseIntent")?.optString("category") ?: "n/a"
                    val screenEngagement = response.optJSONObject("behavioralContext")?.optString("screenEngagement") ?: "n/a"

                    Log.i(TAG, "[FrameCapture] <<< API RESPONSE OK (${requestTime}ms)")
                    Log.i(TAG, "[FrameCapture] Core: viewers=${response.optInt("estimatedViewerCount")}, " +
                            "attention=${response.optString("attentionLevel")}, mood=${response.optString("ambientMood")}")
                    Log.i(TAG, "[FrameCapture] Enhanced: income=$incomeLevel, segments=[$segmentNames], " +
                            "purchaseIntent=$purchaseIntent, engagement=$screenEngagement")
                    if (groupComp != null) {
                        Log.i(TAG, "[FrameCapture] Groups: solo=${groupComp.optInt("solo")}, " +
                                "couples=${groupComp.optInt("couples")}, families=${groupComp.optInt("familiesWithChildren")}, " +
                                "friends=${groupComp.optInt("friendGroups")}, work=${groupComp.optInt("workGroups")}")
                    }

                    // Log full response at debug level
                    Log.d(TAG, "[FrameCapture] Full API response: $response")

                    // Parse demographics result
                    val result = parseDemographicsResponse(response)

                    // Notify callback on main thread
                    withContext(Dispatchers.Main) {
                        onDemographicsReceived?.invoke(result)
                    }

                } else {
                    val errorBody = connection.errorStream?.bufferedReader()?.readText() ?: "Unknown error"
                    Log.e(TAG, "[FrameCapture] <<< API ERROR ($responseCode, ${requestTime}ms): $errorBody")

                    withContext(Dispatchers.Main) {
                        onError?.invoke("API error: $responseCode - $errorBody")
                    }
                }

                connection.disconnect()

            } catch (e: Exception) {
                val errorTime = System.currentTimeMillis() - apiStartTime
                Log.e(TAG, "[FrameCapture] <<< NETWORK ERROR (${errorTime}ms): ${e.javaClass.simpleName} - ${e.message}", e)

                withContext(Dispatchers.Main) {
                    onError?.invoke("Network error: ${e.message}")
                }

            } finally {
                isCapturing.set(false)
            }
        }
    }

    /**
     * Parse the demographics response from the API.
     * Handles both basic demographics and enhanced targeting fields.
     */
    private fun parseDemographicsResponse(response: JSONObject): DemographicsResult {
        val demographics = response.optJSONObject("demographics")
        val ageRanges = mutableMapOf<String, Int>()
        val genderDistribution = mutableMapOf<String, Int>()

        demographics?.optJSONObject("ageRanges")?.let { ages ->
            ages.keys().forEach { key ->
                ageRanges[key] = ages.optInt(key, 0)
            }
        }

        demographics?.optJSONObject("genderDistribution")?.let { genders ->
            genders.keys().forEach { key ->
                genderDistribution[key] = genders.optInt(key, 0)
            }
        }

        // Parse enhanced targeting fields
        val incomeSignalsObj = response.optJSONObject("incomeSignals")
        val lifestyleSegmentsArr = response.optJSONArray("lifestyleSegments")
        val groupComposition = response.optJSONObject("groupComposition")
        val purchaseIntent = response.optJSONObject("purchaseIntent")
        val behavioralContext = response.optJSONObject("behavioralContext")

        // Extract income signals list (e.g., "business_attire", "luxury_accessories")
        val incomeSignalsList = mutableListOf<String>()
        incomeSignalsObj?.optJSONArray("signals")?.let { signals ->
            for (i in 0 until signals.length()) {
                signals.optString(i)?.let { incomeSignalsList.add(it) }
            }
        }

        // Extract all lifestyle segments
        val allLifestyleSegments = mutableListOf<String>()
        lifestyleSegmentsArr?.let { segments ->
            for (i in 0 until segments.length()) {
                segments.optJSONObject(i)?.optString("segment")?.let { allLifestyleSegments.add(it) }
            }
        }

        // Log enhanced fields at debug level
        Log.d(TAG, "[FrameCapture] Enhanced targeting: " +
                "income=${incomeSignalsObj?.optString("estimatedLevel") ?: "n/a"}, " +
                "incomeSignals=$incomeSignalsList, " +
                "segments=$allLifestyleSegments, " +
                "groups=${groupComposition != null}, " +
                "purchaseIntent=${purchaseIntent?.optString("category") ?: "n/a"}, " +
                "mood=${behavioralContext?.optString("primaryMood") ?: "n/a"}")

        return DemographicsResult(
            estimatedViewerCount = response.optInt("estimatedViewerCount", 0),
            ageRanges = ageRanges,
            genderDistribution = genderDistribution,
            attentionLevel = response.optString("attentionLevel", "unknown"),
            crowdDensity = response.optString("crowdDensity", "unknown"),
            ambientMood = response.optString("ambientMood", "unknown"),
            confidence = response.optDouble("confidence", 0.0),
            model = response.optString("model", "unknown"),
            processingTimeMs = response.optLong("processingTimeMs", 0),
            // Enhanced targeting fields
            incomeLevel = incomeSignalsObj?.optString("estimatedLevel"),
            incomeConfidence = incomeSignalsObj?.optDouble("confidence", 0.0) ?: 0.0,
            incomeSignals = incomeSignalsList,
            lifestyleSegments = allLifestyleSegments,
            topLifestyleSegment = allLifestyleSegments.firstOrNull(),
            groupCompositionJson = groupComposition,
            purchaseIntentCategory = purchaseIntent?.optString("category"),
            purchaseIntentVisible = purchaseIntent?.optBoolean("shoppingBagsVisible", false) ?: false,
            screenEngagement = behavioralContext?.optString("screenEngagement"),
            movementPace = behavioralContext?.optString("movementPace"),
            primaryMood = behavioralContext?.optString("primaryMood"),
            sceneDescription = if (response.has("sceneDescription") && !response.isNull("sceneDescription")) response.getString("sceneDescription") else null,
            contextualRelevance = if (response.has("contextualRelevance") && !response.isNull("contextualRelevance")) response.getString("contextualRelevance") else null,
            measurementQuality = if (response.has("measurementQuality") && !response.isNull("measurementQuality")) response.getString("measurementQuality") else null,
            observationFamily = when {
                response.has("observationFamily") && !response.isNull("observationFamily") -> response.getString("observationFamily")
                response.optJSONObject("edgeQuality")?.has("observationFamily") == true -> response.optJSONObject("edgeQuality")?.optString("observationFamily")
                else -> null
            },
            evidenceGrade = when {
                response.has("evidenceGrade") && !response.isNull("evidenceGrade") -> response.getString("evidenceGrade")
                response.optJSONObject("edgeQuality")?.has("evidenceGrade") == true -> response.optJSONObject("edgeQuality")?.optString("evidenceGrade")
                else -> null
            },
            decisionability = when {
                response.has("decisionability") && !response.isNull("decisionability") -> response.getString("decisionability")
                response.optJSONObject("edgeQuality")?.has("decisionability") == true -> response.optJSONObject("edgeQuality")?.optString("decisionability")
                else -> null
            },
            decisionBlockReasons = buildList {
                val explicitReasons = response.optJSONArray("decisionBlockReasons")
                val edgeReasons = response.optJSONObject("edgeQuality")?.optJSONArray("decisionBlockReasons")
                val reasonsJson = explicitReasons ?: edgeReasons
                if (reasonsJson != null) {
                    for (index in 0 until reasonsJson.length()) {
                        val value = reasonsJson.optString(index)
                        if (!value.isNullOrBlank()) add(value)
                    }
                }
            },
            edgeQualityJson = response.optJSONObject("edgeQuality"),
            captureMode = if (response.has("captureMode") && !response.isNull("captureMode")) response.getString("captureMode") else currentCaptureMode,
            analysisTimestamp = System.currentTimeMillis()
        )
    }

    /**
     * Release resources.
     * Note: Does NOT release camera - that's managed by AudienceAnalyzer.
     */
    fun release() {
        isInitialized.set(false)
        periodicCaptureHandler?.removeCallbacks(periodicCaptureRunnable)
        periodicCaptureHandler = null
        captureExecutor.shutdown()
        networkScope.cancel()
        imageCapture = null  // Don't close - AudienceAnalyzer owns it
        Log.i(TAG, "[FrameCapture] Released (camera managed by AudienceAnalyzer)")
    }

    /**
     * Check if a capture is currently in progress.
     */
    fun isCapturing(): Boolean = isCapturing.get()

    /**
     * Get time until next capture is allowed (in milliseconds).
     * Returns 0 if capture is allowed now.
     */
    fun getTimeUntilNextCapture(): Long {
        val lastCapture = lastCaptureTime.get()
        if (lastCapture == 0L) return 0
        val timeSince = System.currentTimeMillis() - lastCapture
        return maxOf(0, MIN_CAPTURE_INTERVAL_MS - timeSince)
    }

    /**
     * Get statistics for debugging.
     */
    fun getStats(): Map<String, Any> = mapOf(
        "isReady" to isReady(),
        "isCapturing" to isCapturing(),
        "lastCaptureTime" to lastCaptureTime.get(),
        "timeUntilNextCapture" to getTimeUntilNextCapture(),
        "screenId" to (screenId ?: "not set")
    )
}

/**
 * Demographics analysis result from Gemini Vision.
 * Includes both basic demographics and enhanced targeting fields.
 */
data class DemographicsResult(
    // Core demographics
    val estimatedViewerCount: Int,
    val ageRanges: Map<String, Int>,  // e.g., {"18-24": 2, "25-34": 3}
    val genderDistribution: Map<String, Int>,  // e.g., {"male": 3, "female": 2}
    val attentionLevel: String,  // "high", "medium", "low", "none"
    val crowdDensity: String,  // "sparse", "moderate", "crowded"
    val ambientMood: String,  // "engaged", "casual", "rushing"

    // Analysis metadata
    val confidence: Double = 0.0,
    val model: String = "unknown",
    val processingTimeMs: Long = 0,

    // Enhanced targeting fields (from Gemini Vision)
    val incomeLevel: String? = null,  // "low", "medium", "high", "premium"
    val incomeConfidence: Double = 0.0,
    val incomeSignals: List<String> = emptyList(),  // e.g., ["business_attire", "luxury_accessories"]
    val lifestyleSegments: List<String> = emptyList(),  // All segments: ["professional", "fitness_enthusiast"]
    val topLifestyleSegment: String? = null,  // Primary segment (first in list)
    val groupCompositionJson: JSONObject? = null,  // { solo, couples, familiesWithChildren, etc. }
    val purchaseIntentCategory: String? = null,  // "electronics", "fashion", etc.
    val purchaseIntentVisible: Boolean = false,  // Shopping bags visible
    val screenEngagement: String? = null,  // "high", "medium", "low", "none"
    val movementPace: String? = null,  // "stationary", "slow", "moderate", "fast"
    val primaryMood: String? = null,  // "relaxed", "focused", "social", etc.

    // Scene context (from Gemini Vision)
    val sceneDescription: String? = null,
    val contextualRelevance: String? = null,
    val measurementQuality: String? = null,
    val observationFamily: String? = null,
    val evidenceGrade: String? = null,
    val decisionability: String? = null,
    val decisionBlockReasons: List<String> = emptyList(),
    val edgeQualityJson: JSONObject? = null,
    val captureMode: String? = null,  // "face_triggered" or "periodic_scene"

    val analysisTimestamp: Long
) {
    /**
     * Convert to JSON for Socket.io emission.
     * Includes enhanced targeting fields when available.
     */
    fun toJson(): JSONObject = JSONObject().apply {
        // Core demographics
        put("estimatedViewerCount", estimatedViewerCount)
        put("ageRanges", JSONObject(ageRanges.mapValues { it.value }))
        put("genderDistribution", JSONObject(genderDistribution.mapValues { it.value }))
        put("attentionLevel", attentionLevel)
        put("crowdDensity", crowdDensity)
        put("ambientMood", ambientMood)
        put("confidence", confidence)
        put("model", model)
        put("processingTimeMs", processingTimeMs)
        put("analysisTimestamp", analysisTimestamp)

        // Enhanced targeting - Income signals
        incomeLevel?.let {
            put("incomeSignals", JSONObject().apply {
                put("estimatedLevel", incomeLevel)
                put("confidence", incomeConfidence)
                if (incomeSignals.isNotEmpty()) {
                    put("signals", org.json.JSONArray(incomeSignals))
                }
            })
        }

        // Enhanced targeting - Lifestyle segments (full list)
        if (lifestyleSegments.isNotEmpty()) {
            put("lifestyleSegments", org.json.JSONArray(lifestyleSegments.map { segment ->
                JSONObject().apply { put("segment", segment) }
            }))
        }
        topLifestyleSegment?.let {
            put("topLifestyleSegment", it)
        }

        // Group composition
        groupCompositionJson?.let {
            put("groupComposition", it)
        }

        // Purchase intent
        if (purchaseIntentCategory != null || purchaseIntentVisible) {
            put("purchaseIntent", JSONObject().apply {
                purchaseIntentCategory?.let { put("category", it) }
                put("shoppingBagsVisible", purchaseIntentVisible)
            })
        }

        // Behavioral context
        if (screenEngagement != null || movementPace != null || primaryMood != null) {
            put("behavioralContext", JSONObject().apply {
                screenEngagement?.let { put("screenEngagement", it) }
                movementPace?.let { put("movementPace", it) }
                primaryMood?.let { put("primaryMood", it) }
            })
        }

        // Scene context (from Gemini Vision)
        sceneDescription?.let { put("sceneDescription", it) }
        contextualRelevance?.let { put("contextualRelevance", it) }
        measurementQuality?.let { put("measurementQuality", it) }
        observationFamily?.let { put("observationFamily", it) }
        evidenceGrade?.let { put("evidenceGrade", it) }
        decisionability?.let { put("decisionability", it) }
        put("decisionBlockReasons", org.json.JSONArray(decisionBlockReasons))
        edgeQualityJson?.let { put("edgeQuality", it) }
        captureMode?.let { put("captureMode", it) }
    }

    override fun toString(): String {
        val segmentsStr = if (lifestyleSegments.isNotEmpty()) lifestyleSegments.joinToString(",") else "n/a"
        return "[Demographics: viewers=$estimatedViewerCount, attention=$attentionLevel, " +
                "income=$incomeLevel, segments=[$segmentsStr], engagement=$screenEngagement, mood=$primaryMood]"
    }
}
