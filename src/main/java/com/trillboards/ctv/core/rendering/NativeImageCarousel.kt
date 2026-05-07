package com.trillboards.ctv.core.rendering

import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.ImageView
import coil.ImageLoader
import coil.request.ImageRequest
import coil.transition.CrossfadeTransition
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.rendering.models.ImageItem

class NativeImageCarousel(
    private val listener: Listener
) {
    companion object {
        private const val TAG = "NativeImageCarousel"
        private val renderCfg get() = SensingConfig.get().rendering
        private val DEFAULT_DURATION_MS get() = renderCfg.defaultImageDurationMs
        private val CROSSFADE_DURATION_MS get() = renderCfg.crossfadeDurationMs
    }

    interface Listener {
        fun onImageDisplayed(index: Int, url: String)
        fun onCarouselComplete()
    }

    private var imageView: ImageView? = null
    private var imageLoader: ImageLoader? = null
    private var images: List<ImageItem> = emptyList()
    private var currentIndex = 0
    @Volatile private var isRunning = false
    private val handler = Handler(Looper.getMainLooper())
    private var rotationRunnable: Runnable? = null

    fun initialize(imageView: ImageView) {
        this.imageView = imageView
        this.imageLoader = ImageLoader.Builder(imageView.context)
            .crossfade(CROSSFADE_DURATION_MS)
            .build()
        Log.i(TAG, "Initialized")
    }

    fun showImages(images: List<ImageItem>, durationMs: Long = DEFAULT_DURATION_MS) {
        if (images.isEmpty()) {
            Log.w(TAG, "No images to show")
            return
        }

        stop()
        this.images = images
        this.currentIndex = 0
        this.isRunning = true

        Log.i(TAG, "Starting carousel: ${images.size} images, ${durationMs}ms each")
        displayCurrentImage()
        scheduleNext(durationMs)
    }

    fun showSingleImage(url: String) {
        stop()
        this.images = listOf(ImageItem(url))
        this.currentIndex = 0
        this.isRunning = true
        displayCurrentImage()
    }

    fun stop() {
        isRunning = false
        rotationRunnable?.let { handler.removeCallbacks(it) }
        rotationRunnable = null
    }

    fun getCurrentIndex(): Int = currentIndex

    fun isActive(): Boolean = isRunning

    fun release() {
        stop()
        imageLoader?.shutdown()
        imageLoader = null
        imageView = null
        images = emptyList()
    }

    private fun displayCurrentImage() {
        val view = imageView ?: return
        if (currentIndex >= images.size) return

        val imageItem = images[currentIndex]
        val request = ImageRequest.Builder(view.context)
            .data(imageItem.url)
            .target(view)
            .crossfade(CROSSFADE_DURATION_MS)
            .listener(
                onSuccess = { _, _ ->
                    listener.onImageDisplayed(currentIndex, imageItem.url)
                },
                onError = { _, result ->
                    Log.e(TAG, "Failed to load image: ${imageItem.url}", result.throwable)
                }
            )
            .build()

        imageLoader?.enqueue(request)

        // Pre-fetch next image
        if (images.size > 1) {
            val nextIndex = (currentIndex + 1) % images.size
            val prefetchRequest = ImageRequest.Builder(view.context)
                .data(images[nextIndex].url)
                .size(view.width.coerceAtLeast(1), view.height.coerceAtLeast(1))
                .build()
            imageLoader?.enqueue(prefetchRequest)
        }
    }

    private fun scheduleNext(durationMs: Long) {
        val effectiveDuration = images.getOrNull(currentIndex)?.durationMs ?: durationMs
        rotationRunnable = Runnable {
            if (!isRunning) return@Runnable

            currentIndex++
            if (currentIndex >= images.size) {
                // Completed one full cycle
                listener.onCarouselComplete()
                currentIndex = 0
            }

            displayCurrentImage()
            scheduleNext(durationMs)
        }
        handler.postDelayed(rotationRunnable!!, effectiveDuration)
    }
}
