package com.trillboards.ctv.core.bridge

import android.content.Context
import android.hardware.display.DisplayManager
import android.os.Build
import android.util.Log
import android.view.Display
import android.webkit.JavascriptInterface
import android.webkit.WebView
import com.trillboards.ctv.core.identity.AdvertisingIdCollector
import com.trillboards.ctv.core.identity.AdvertisingIdResult
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject

/**
 * JavaScript bridge exposed to WebView as `window.TrillboardsNativeDevice`.
 *
 * Provides ground-truth device identity from the native Android layer,
 * replacing unreliable UA-parsing in the JavaScript SDK.
 *
 * Thread safety: @JavascriptInterface methods run on a WebView background
 * thread. All fields are read from immutable Build.* constants or the
 * cached AdvertisingIdCollector (volatile + TTL).
 */
class NativeDeviceBridge(private val context: Context) {

    @Volatile
    private var cachedAdId: AdvertisingIdResult? = null

    companion object {
        const val JS_INTERFACE_NAME = "TrillboardsNativeDevice"
        private const val TAG = "NativeDeviceBridge"
        private const val AD_ID_TIMEOUT_MS = 3000L

        /** Latest WebView FPS reported by the injected rAF counter.
         *  Read by AudienceSensingService for edgeQuality telemetry. */
        @Volatile
        @JvmStatic
        var lastWebViewFps: Double = -1.0
            private set

        @Volatile
        @JvmStatic
        var lastOverlayHealthSnapshot: OverlayHealthSnapshot? = null
            private set

        /**
         * Attach this bridge to a WebView. Must be called BEFORE loadUrl().
         */
        fun attach(webView: WebView, context: Context): NativeDeviceBridge {
            val bridge = NativeDeviceBridge(context.applicationContext)
            webView.addJavascriptInterface(bridge, JS_INTERFACE_NAME)
            Log.i(TAG, "Native device bridge attached to WebView")
            return bridge
        }
    }

    /**
     * Called from JavaScript: window.TrillboardsNativeDevice.getDeviceInfo()
     * Returns a JSON string with ground-truth device identity.
     */
    @JavascriptInterface
    fun getDeviceInfo(): String {
        val json = JSONObject()
        json.put("make", Build.BRAND)
        json.put("model", Build.MODEL)
        json.put("os", "Android")
        json.put("osVersion", Build.VERSION.RELEASE)
        json.put("apiLevel", Build.VERSION.SDK_INT)
        json.put("deviceType", "dooh")
        json.put("bridgeVersion", 1)

        // Advertising ID (may be null if LAT=true or GMS unavailable)
        val adResult = getAdvertisingId()
        if (adResult != null) {
            json.put("advertisingId", adResult.id)
            json.put("advertisingIdType", adResult.type)
            json.put("limitAdTracking", adResult.isLat)
        } else {
            json.put("advertisingId", JSONObject.NULL)
            json.put("advertisingIdType", JSONObject.NULL)
            json.put("limitAdTracking", JSONObject.NULL)
        }

        return json.toString()
    }

    /**
     * Called from JavaScript: window.TrillboardsNativeDevice.getBridgeVersion()
     * Returns the bridge protocol version (for forward compatibility).
     */
    @JavascriptInterface
    fun getBridgeVersion(): Int = 1

    /**
     * Called from JavaScript: window.TrillboardsNativeDevice.getDisplayState()
     *
     * Returns a JSON string {powerState, displayOn, source, timestamp}. Consumed
     * by the content-play viewability gate in trillboard-screen to suppress
     * phantom contentPlayStart / reportPlaybackPosition emits while the TV
     * display is physically off (HDMI/CEC disconnect). Queries the Android
     * DisplayManager directly so Fire TV, tablet and Android TV hosts all
     * resolve the same signal.
     *
     * Values:
     *   powerState: "on" | "off" | "standby" | "unknown"
     *   displayOn:  true when STATE_ON
     */
    @JavascriptInterface
    fun getDisplayState(): String {
        val json = JSONObject()
        try {
            val displayManager = context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
            val primary = displayManager.getDisplay(Display.DEFAULT_DISPLAY)
            val state = primary?.state
            val powerState = when (state) {
                Display.STATE_ON -> "on"
                Display.STATE_OFF -> "off"
                Display.STATE_DOZE, Display.STATE_DOZE_SUSPEND -> "standby"
                Display.STATE_VR -> "on"
                Display.STATE_ON_SUSPEND -> "standby"
                else -> "unknown"
            }
            json.put("powerState", powerState)
            json.put("displayOn", state == Display.STATE_ON)
            json.put("source", "agent")
            json.put("timestamp", System.currentTimeMillis())
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read display state: ${e.message}")
            json.put("powerState", "unknown")
            json.put("displayOn", false)
            json.put("source", "agent")
            json.put("timestamp", System.currentTimeMillis())
        }
        return json.toString()
    }

    /**
     * Called from JavaScript: window.TrillboardsNativeDevice.reportFps(fps)
     * Receives the WebView's measured requestAnimationFrame FPS every 5 seconds.
     * Stored in a companion @Volatile field for telemetry emission.
     */
    @JavascriptInterface
    fun reportFps(fps: Double) {
        lastWebViewFps = fps
        Log.d(TAG, "[WebViewFPS] ${"%.1f".format(fps)} fps")
    }

    /**
     * Called from JavaScript: window.TrillboardsNativeDevice.reportOverlayHealth(json)
     * Stores the latest structured overlay health snapshot for inclusion in heartbeats.
     */
    @JavascriptInterface
    fun reportOverlayHealth(payload: String?) {
        val parsed = OverlayHealthSnapshot.fromJson(payload)
        if (parsed != null) {
            lastOverlayHealthSnapshot = parsed
            Log.d(
                TAG,
                "[OverlayHealth] stream=${parsed.currentStreamType ?: "unknown"} " +
                    "heap=${parsed.jsHeapUsedMb ?: -1}MB socket=${parsed.socketConnected}"
            )
        } else {
            Log.w(TAG, "[OverlayHealth] Failed to parse payload")
        }
    }

    /**
     * Pre-warm the advertising ID cache. Call from a coroutine scope
     * during Activity onCreate to avoid blocking the JS bridge thread.
     */
    suspend fun prewarmAdvertisingId() {
        try {
            cachedAdId = AdvertisingIdCollector.collect(context)
            Log.d(TAG, "Advertising ID pre-warmed: ${if (cachedAdId != null) "available" else "unavailable"}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to pre-warm advertising ID: ${e.message}")
        }
    }

    private fun getAdvertisingId(): AdvertisingIdResult? {
        // Return pre-warmed cache if available
        cachedAdId?.let { return it }

        // Fallback: blocking collect with short timeout
        // (only hit if prewarm didn't run)
        return try {
            runBlocking {
                withTimeoutOrNull(AD_ID_TIMEOUT_MS) {
                    AdvertisingIdCollector.collect(context)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to collect advertising ID: ${e.message}")
            null
        }
    }
}
