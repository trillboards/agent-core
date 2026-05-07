package com.trillboards.ctv.core.device

import android.app.Activity
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.PixelCopy
import android.view.View
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import kotlin.coroutines.resume

/**
 * Manages screenshot capture and upload for MDM device.screenshot command.
 * Shared across all CTV agents (Fire TV, Android TV, Tablet).
 *
 * Flow:
 * 1. Capture the current screen using PixelCopy API
 * 2. Compress to JPEG
 * 3. Upload to S3 via presigned URL
 * 4. Return CDN URL for the screenshot
 *
 * @param client OkHttpClient instance for S3 upload
 */
class ScreenshotManager(private val client: OkHttpClient) {

    companion object {
        private const val TAG = "ScreenshotManager"
        private const val DEFAULT_JPEG_QUALITY = 80
    }

    /**
     * Result of a screenshot capture and upload operation
     */
    data class ScreenshotResult(
        val success: Boolean,
        val cdnUrl: String? = null,
        val error: String? = null
    )

    /**
     * Capture a screenshot from the given activity's window and upload to S3.
     *
     * @param activity The activity whose window to capture
     * @param uploadUrl Presigned S3 upload URL (PUT)
     * @param cdnUrl The CDN URL that will serve the screenshot after upload
     * @param quality JPEG compression quality (0-100, default 80)
     * @return ScreenshotResult indicating success/failure with CDN URL or error
     */
    suspend fun captureAndUpload(
        activity: Activity,
        uploadUrl: String,
        cdnUrl: String,
        quality: Int = DEFAULT_JPEG_QUALITY
    ): ScreenshotResult = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Starting screenshot capture")

            // 1. Capture bitmap from activity's window
            val bitmap = captureFromActivity(activity)
                ?: return@withContext ScreenshotResult(
                    success = false,
                    error = "Failed to capture screen - PixelCopy not available or failed"
                )

            Log.d(TAG, "Captured bitmap: ${bitmap.width}x${bitmap.height}")

            // 2. Compress to JPEG
            val jpegBytes = compressToJpeg(bitmap, quality)
            bitmap.recycle()  // Free memory immediately

            Log.d(TAG, "Compressed to ${jpegBytes.size} bytes (quality=$quality)")

            // 3. Upload to S3 via presigned URL
            val uploadSuccess = uploadToS3(uploadUrl, jpegBytes)

            if (uploadSuccess) {
                Log.i(TAG, "Screenshot uploaded successfully: $cdnUrl")
                ScreenshotResult(success = true, cdnUrl = cdnUrl)
            } else {
                Log.e(TAG, "S3 upload failed")
                ScreenshotResult(success = false, error = "S3 upload failed")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Screenshot capture/upload failed", e)
            ScreenshotResult(success = false, error = e.message ?: "Unknown error")
        }
    }

    /**
     * Capture a bitmap from the activity's window using PixelCopy (API 24+)
     * Falls back to View.drawingCache for older APIs.
     */
    private suspend fun captureFromActivity(activity: Activity): Bitmap? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            captureWithPixelCopy(activity)
        } else {
            captureWithDrawingCache(activity)
        }
    }

    /**
     * Capture using PixelCopy API (recommended for API 26+)
     * Provides accurate hardware-accelerated capture.
     */
    private suspend fun captureWithPixelCopy(activity: Activity): Bitmap? =
        suspendCancellableCoroutine { cont ->
            try {
                val window = activity.window
                val view = window.decorView.rootView

                // Create bitmap with same dimensions as the window
                val bitmap = Bitmap.createBitmap(
                    view.width,
                    view.height,
                    Bitmap.Config.ARGB_8888
                )

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    PixelCopy.request(
                        window,
                        bitmap,
                        { copyResult ->
                            if (copyResult == PixelCopy.SUCCESS) {
                                cont.resume(bitmap)
                            } else {
                                Log.e(TAG, "PixelCopy failed with result: $copyResult")
                                bitmap.recycle()
                                cont.resume(null)
                            }
                        },
                        Handler(Looper.getMainLooper())
                    )
                } else {
                    bitmap.recycle()
                    cont.resume(null)
                }
            } catch (e: Exception) {
                Log.e(TAG, "PixelCopy exception", e)
                cont.resume(null)
            }
        }

    /**
     * Fallback capture using deprecated drawingCache (for API < 26)
     */
    @Suppress("DEPRECATION")
    private suspend fun captureWithDrawingCache(activity: Activity): Bitmap? =
        withContext(Dispatchers.Main) {
            try {
                val view = activity.window.decorView.rootView
                view.isDrawingCacheEnabled = true
                view.buildDrawingCache()
                val cache = view.drawingCache
                val bitmap = if (cache != null) {
                    Bitmap.createBitmap(cache)
                } else {
                    null
                }
                view.isDrawingCacheEnabled = false
                view.destroyDrawingCache()
                bitmap
            } catch (e: Exception) {
                Log.e(TAG, "Drawing cache capture failed", e)
                null
            }
        }

    /**
     * Compress bitmap to JPEG byte array
     */
    private fun compressToJpeg(bitmap: Bitmap, quality: Int): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality.coerceIn(1, 100), stream)
        return stream.toByteArray()
    }

    /**
     * Upload JPEG bytes to S3 using presigned URL
     */
    private suspend fun uploadToS3(uploadUrl: String, data: ByteArray): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url(uploadUrl)
                    .put(data.toRequestBody("image/jpeg".toMediaType()))
                    .build()

                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.e(TAG, "S3 upload failed: ${response.code} ${response.message}")
                    }
                    response.isSuccessful
                }
            } catch (e: Exception) {
                Log.e(TAG, "S3 upload exception", e)
                false
            }
        }
}
