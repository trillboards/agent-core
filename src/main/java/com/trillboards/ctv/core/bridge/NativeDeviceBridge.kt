package com.trillboards.ctv.core.bridge

import android.app.Activity
import android.content.Context
import android.hardware.display.DisplayManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Display
import android.webkit.JavascriptInterface
import android.webkit.WebView
import com.trillboards.ctv.core.activity.BaseAgentActivity
import com.trillboards.ctv.core.identity.AdvertisingIdCollector
import com.trillboards.ctv.core.identity.AdvertisingIdResult
import java.lang.ref.WeakReference
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
 *
 * @param context Application context — used for everything not requiring
 *   an Activity (DisplayManager, AdvertisingIdCollector, etc.).
 * @param activityRef Weak ref to the host Activity, optional, used solely
 *   by [requestEdgeAiPermissions] to route into
 *   [BaseAgentActivity.promptEdgeAiPermissionsViaSettings]. Held weakly so
 *   the bridge does not leak the host across WebView recreation.
 */
class NativeDeviceBridge(
    private val context: Context,
    private val activityRef: WeakReference<Activity>? = null
) {

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
         *
         * Optional `activity` parameter wires the
         * `window.TrillboardsNativeDevice.requestEdgeAiPermissions()` JS path
         * into [BaseAgentActivity.promptEdgeAiPermissionsViaSettings]; pass
         * the host Activity when the WebView is rendered inside a
         * [BaseAgentActivity] subclass. Older callers that pass only the
         * Context retain the original behavior — the JS bridge's permission
         * RPC silently no-ops (and logs once) instead of crashing.
         */
        @JvmOverloads
        fun attach(webView: WebView, context: Context, activity: Activity? = null): NativeDeviceBridge {
            val bridge = NativeDeviceBridge(
                context.applicationContext,
                activity?.let { WeakReference(it) }
            )
            webView.addJavascriptInterface(bridge, JS_INTERFACE_NAME)
            Log.i(TAG, "Native device bridge attached to WebView (activity=${activity?.javaClass?.simpleName ?: "none"})")
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
     * Called from JavaScript: window.TrillboardsNativeDevice.requestEdgeAiPermissions()
     *
     * Opt-in path for the React PWA inside the kiosk WebView to surface the
     * system Settings → App info page so the user can grant
     * CAMERA / RECORD_AUDIO. Mirrors the on-device wizard CTA (which fires
     * the same [BaseAgentActivity.promptEdgeAiPermissionsViaSettings]
     * helper) — the JS-driven path is intended as a secondary trigger; the
     * primary remains the in-app SetupWizardActivity.
     *
     * No-op (with a Log.w) when the bridge wasn't attached with an Activity
     * reference, or when the weak ref has already been GC'd. Returning a
     * boolean lets the JS side branch on "intent dispatched" vs "fallback to
     * notice copy".
     */
    @JavascriptInterface
    fun requestEdgeAiPermissions(): Boolean {
        val activity = activityRef?.get()
        if (activity !is BaseAgentActivity) {
            Log.w(
                TAG,
                "requestEdgeAiPermissions() invoked but no BaseAgentActivity available " +
                    "(activity=${activity?.javaClass?.simpleName ?: "null"}); silently no-op"
            )
            return false
        }
        // Settings deep-link must dispatch on the main thread — JS bridge
        // callbacks run on a WebView background thread.
        Handler(Looper.getMainLooper()).post {
            try {
                activity.promptEdgeAiPermissionsViaSettings()
            } catch (e: Exception) {
                Log.w(TAG, "promptEdgeAiPermissionsViaSettings() threw from JS bridge", e)
            }
        }
        return true
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
