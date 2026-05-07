package com.trillboards.ctv.core.activity

import android.content.BroadcastReceiver
import android.content.ComponentCallbacks2
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.Log
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.ConsoleMessage
import android.webkit.GeolocationPermissions
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import kotlinx.coroutines.launch
import com.trillboards.ctv.core.AgentConfig as CoreAgentConfig
import com.trillboards.ctv.core.DeviceIdentity
import com.trillboards.ctv.core.bridge.NativeDeviceBridge
import com.trillboards.ctv.core.bridge.WebViewAdsRegistrar
import com.trillboards.ctv.core.device.KioskLockManager
import com.trillboards.ctv.core.rendering.OverlayUrlBuilder
import com.trillboards.ctv.core.service.IntentKeys
import com.trillboards.ctv.core.service.PrefsKeys
import com.trillboards.ctv.core.stability.ServiceWatchdog
import org.json.JSONObject

/**
 * Platform-agnostic agent activity skeleton extracted from
 * `tablet-agent/.../MainActivity.kt` and `android-tv-agent/.../MainActivity.kt` (PR 4/5).
 *
 * Each platform's `MainActivity` extends this class and supplies platform identifiers
 * (agentServiceClass, deviceAdminClass, AgentConfig, IntentKeys, PrefsKeys, overlay
 * URL resource id) plus a [bindingAdapter] that exposes the platform's view binding's
 * `webView` / `blackoutView` / `loadingIndicator` / optional `touchInterceptor`.
 *
 * This base owns:
 *   - permission request flow
 *   - LocalBroadcastReceiver registration (overlay/maintenance/clear-cache/kiosk-control)
 *   - WebView setup (FPS counter, IMA gesture injection, transport-type injection,
 *     geolocation, render-process-gone handler) + page-load timeout / retry ladder
 *   - WebView recreation (single signature, programmatic re-add to layout)
 *   - kiosk PIN dialog flow + Settings dialog
 *   - service watchdog scheduling
 *   - onTrimMemory cooldown-guarded cache clear (TV-style — safer than tablet's old
 *     aggressive RUNNING_LOW path)
 *   - onResume sensor-rebind broadcast (gated on `wasStoppedSinceLastResume` so a
 *     transient permission/dialog bounce doesn't trigger a rebind storm)
 *   - transport-type registration (wifi/ethernet/cellular/other) for telemetry
 *
 * Open hooks per platform:
 *   - [useImmersiveFullscreen] — tablet=true, TV=false
 *   - [useBatteryOptimizationPrompt] — tablet=true, TV=false
 *   - [wireKioskTrigger] — tablet hooks long-press on touchInterceptor; TV intercepts
 *     KEYCODE_MENU in dispatchKeyEvent
 *   - [isRefreshKey] — base default = D-pad center / Enter / Numpad enter (TV's
 *     superset; tablet uses the same)
 *   - [onPlatformResume] / [onPlatformPause] / [onPlatformDestroy] /
 *     [onPlatformContentRefresh] / [onPlatformTrimMemory] — platform-specific
 *     extras (tablet routes through HybridRenderingManager, TV cleans up
 *     AntiStandbyManager).
 */
abstract class BaseAgentActivity : ComponentActivity(), AgentRecreatableActivity {

    companion object {
        private const val TAG = "BaseAgentActivity"
        private const val SENSOR_REBIND_AFTER_RESUME_DELAY_MS = 1_500L
        private const val PAGE_LOAD_TIMEOUT_MS = 30_000L
        private const val BASE_RETRY_DELAY_MS = 2_000L
        private const val MAX_RETRY_DELAY_MS = 60_000L
        private const val MAX_RETRY_ATTEMPTS = 5
        private const val LONG_PRESS_DURATION_MS = 5_000L
        // 5-min cooldown on cache clear under critical memory pressure.
        // Tablet's old RUNNING_LOW path was aggressive and caused WebView re-fetch
        // storms; TV's cooldown-guarded RUNNING_CRITICAL is safer for both
        // platforms. RUNNING_LOW is now ignored.
        private const val CACHE_CLEAR_COOLDOWN_MS = 5 * 60 * 1000L
    }

    // ── Per-platform abstract surface ─────────────────────────────────────────

    /** Platform's foreground service class — `Intent(this, agentServiceClass())`. */
    protected abstract fun agentServiceClass(): Class<out android.app.Service>

    /** Adapter exposing `webView`/`blackoutView`/`loadingIndicator`/`touchInterceptorOrNull` from the platform's generated `ActivityMainBinding`. */
    protected abstract fun bindingAdapter(): AgentBindingAdapter

    /** Platform's `DeviceAdminReceiver` class (used by [KioskLockManager]). */
    protected abstract fun deviceAdminClass(): Class<*>

    /** Per-platform agent-core config (api/socket URLs, prefs name, overlay actions). */
    protected abstract fun coreAgentConfig(): CoreAgentConfig

    /** Per-platform Intent action namespace. */
    protected abstract fun intentKeys(): IntentKeys

    /** Per-platform SharedPreferences key namespace. */
    protected abstract fun prefsKeys(): PrefsKeys

    /** Resource id for the overlay base URL string (e.g. `R.string.overlay_url`). */
    protected abstract fun overlayUrlResId(): Int

    /** Resource id for the application name (e.g. `R.string.app_name`). */
    protected abstract fun appNameResId(): Int

    /** Permissions to request on first launch (camera/audio/location/bluetooth/etc.). */
    protected abstract fun requiredPermissions(): Array<String>

    // ── Open hooks (default implementations — override per platform) ──────────

    /** Tablet: true (immersive fullscreen). TV: false. */
    protected open fun useImmersiveFullscreen(): Boolean = true

    /** Tablet: true (prompts ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS once). TV: false. */
    protected open fun useBatteryOptimizationPrompt(): Boolean = true

    /**
     * Wire the platform-specific kiosk-settings entry trigger.
     *
     * Tablet implements via `bindingAdapter().touchInterceptorOrNull?.setOnTouchListener`
     * and the long-press helper in this base ([armLongPressOnInterceptor]).
     *
     * TV implements via `dispatchKeyEvent(KEYCODE_MENU)` — see TV's MainActivity.
     *
     * Both ultimately call [triggerKioskSettingsDialog] to start the PIN flow.
     */
    protected abstract fun wireKioskTrigger()

    /**
     * Default refresh key set: D-pad center, Enter, Numpad enter (TV's superset).
     * Tablet inherits this; KEYCODE_DPAD_CENTER triggers a full reload.
     */
    protected open fun isRefreshKey(keyCode: Int): Boolean =
        keyCode == KeyEvent.KEYCODE_DPAD_CENTER ||
            keyCode == KeyEvent.KEYCODE_ENTER ||
            keyCode == KeyEvent.KEYCODE_NUMPAD_ENTER

    /** Tablet routes to `hybridRenderingManager.onContentRefresh()`. */
    protected open fun onPlatformContentRefresh() {}

    /** Tablet routes to `hybridRenderingManager.onResume()`. */
    protected open fun onPlatformResume() {}

    /** Tablet routes to `hybridRenderingManager.onPause()`. */
    protected open fun onPlatformPause() {}

    /** Tablet routes to `hybridRenderingManager.onDestroy()`. TV routes to `antiStandbyManager.cleanup()`. */
    protected open fun onPlatformDestroy() {}

    /** Tablet routes to `hybridRenderingManager.onTrimMemory(level)`. */
    protected open fun onPlatformTrimMemory(level: Int) {}

    /**
     * Whether the activity should fan out `INTENT_REBIND_CAMERA` /
     * `INTENT_REBIND_AUDIO` broadcasts on `onResume` (after a stop).
     *
     * Tablet always returns true; TV's lite flavor returns false (no camera /
     * audio sensing — `BuildConfig.HAS_AUDIENCE_SENSING=false`). The full flavor
     * of TV returns true.
     */
    protected open fun shouldRebindSensorsOnResume(): Boolean = true

    /**
     * Hook fired immediately after a new WebView replaces the previous one (e.g. after
     * a render-process-gone crash). Tablet refreshes HybridRenderingManager surface
     * ownership and re-arms the long-press interceptor.
     */
    protected open fun onWebViewRecreated(newWebView: WebView) {}

    /**
     * Hook fired at the start of [reloadWebView]. Tablet uses this to fail-open the
     * HybridRenderingManager to WebView mode (so a manual reload guarantees we leave
     * any wedged native-rendering surface). TV / lite have no hybrid surface — no-op.
     */
    protected open fun onPlatformReload(reason: String) {}

    // ── Internal state ────────────────────────────────────────────────────────

    protected lateinit var kioskLockManager: KioskLockManager
        private set
    protected val broadcastManager: LocalBroadcastManager by lazy { LocalBroadcastManager.getInstance(this) }
    protected val prefs by lazy { getSharedPreferences(prefsKeys().sharedPrefsName, Context.MODE_PRIVATE) }
    protected var nativeDeviceBridge: NativeDeviceBridge? = null
        private set

    private var activeWebView: WebView? = null
    private var networkCallback: ConnectivityManager.NetworkCallback? = null
    private var pendingSensorRebindOnResume = false
    private var wasStoppedSinceLastResume = false
    private val webViewRecoveryHandler = Handler(Looper.getMainLooper())
    private var retryRunnable: Runnable? = null
    private var pageLoadTimeoutRunnable: Runnable? = null
    private var retryCount = 0
    private val longPressHandler = Handler(Looper.getMainLooper())
    private var longPressRunnable: Runnable? = null

    // RUNNING_CRITICAL cache-clear cooldown shared across both platforms.
    private var lastCacheClearMs = 0L

    /** Accessor: prefer the post-recreation WebView, fall back to the original from XML binding. */
    protected fun currentWebView(): WebView = activeWebView ?: bindingAdapter().webView

    /** Public so platform-specific code (content state polling) can read the live WebView. */
    fun getWebView(): WebView? = activeWebView ?: runCatching { bindingAdapter().webView }.getOrNull()

    // ── Permission request launcher ───────────────────────────────────────────

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        Log.i(TAG, "Permissions result: $permissions")

        // Reload WebView if location permission was just granted (re-init geolocation).
        val fineLocation = permissions[android.Manifest.permission.ACCESS_FINE_LOCATION] == true
        val coarseLocation = permissions[android.Manifest.permission.ACCESS_COARSE_LOCATION] == true
        if (fineLocation || coarseLocation) {
            Log.i(TAG, "Location permission granted, reloading WebView for geolocation re-init")
            reloadWebView("location_permission_granted")
        }

        // Start service after permission result (granted or denied — service handles fallback).
        startAgentService()
        if (useBatteryOptimizationPrompt()) {
            promptBatteryOptimizationExemption()
        }
    }

    // ── Broadcast receivers ───────────────────────────────────────────────────

    private val overlayReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val keys = intentKeys()
            when (intent?.action) {
                keys.overlayRefresh -> triggerContentRefresh()
                keys.overlayBlackout -> {
                    val enabled = intent.getBooleanExtra("enabled", false)
                    bindingAdapter().blackoutView.visibility = if (enabled) View.VISIBLE else View.GONE
                }
            }
        }
    }

    private val maintenanceReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == intentKeys().maintenanceCycle) {
                Log.i(TAG, "Maintenance cycle: clearing non-essential WebView cache")
                currentWebView().clearCache(false)
                requestOverlayMaintenance("maintenance_cycle", aggressive = false)
            }
        }
    }

    private val clearCacheReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == intentKeys().clearCache) {
                Log.i(TAG, "Received CLEAR_CACHE intent - clearing WebView cache")
                currentWebView().clearCache(true)
            }
        }
    }

    private val kioskControlReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == intentKeys().kioskControl) {
                val enable = intent.getBooleanExtra("enable", true)
                Log.i(TAG, "Received KIOSK_CONTROL intent - enable=$enable")
                if (enable) {
                    if (kioskLockManager.isDeviceOwner()) {
                        kioskLockManager.enableKioskMode()
                        startLockTask()
                    }
                } else {
                    stopLockTask()
                    kioskLockManager.disableKioskMode()
                }
            }
        }
    }

    // ── onCreate ──────────────────────────────────────────────────────────────

    /**
     * Subclasses must call [initializeAfterContentView] from `onCreate` AFTER
     * inflating their `ActivityMainBinding` and calling `setContentView(binding.root)`.
     * This base does NOT override `onCreate` itself, because the subclass-specific
     * binding inflation has to happen first (the base reads `bindingAdapter().webView`
     * etc. only after the subclass has wired up the binding).
     *
     * Typical subclass pattern:
     * ```kotlin
     * override fun onCreate(savedInstanceState: Bundle?) {
     *     super.onCreate(savedInstanceState)
     *     binding = ActivityMainBinding.inflate(layoutInflater)
     *     setContentView(binding.root)
     *     // Optional platform-specific pre-setup (e.g. HybridRenderingManager init)
     *     hybridRenderingManager = HybridRenderingManager(...)
     *     hybridRenderingManager.initialize(binding, fingerprint, screenId)
     *     initializeAfterContentView()
     * }
     * ```
     */
    protected fun initializeAfterContentView() {
        val adminComponent = ComponentName(this, deviceAdminClass())
        kioskLockManager = KioskLockManager(this, adminComponent)

        if (BuildConfigDebug.isDebug(this)) {
            WebView.setWebContentsDebuggingEnabled(true)
        }

        if (useImmersiveFullscreen()) {
            configureFullscreen()
        } else {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }

        setupWebView()
        wireKioskTrigger()
        registerReceivers()
        startKioskModeIfEnabled()

        ServiceWatchdog.schedule(this, intentKeys().watchdogCheck)
        registerTransportTypeCallback()

        // Pre-warm Google Advertising ID via the native device bridge (async).
        nativeDeviceBridge?.let { bridge ->
            lifecycleScope.launch {
                try {
                    bridge.prewarmAdvertisingId()
                } catch (e: Exception) {
                    Log.w(TAG, "Advertising-ID prewarm failed", e)
                }
            }
        }

        if (hasAllRequiredPermissions()) {
            startAgentService()
            if (useBatteryOptimizationPrompt()) {
                promptBatteryOptimizationExemption()
            }
        } else {
            requestRequiredPermissions()
        }
    }

    private fun hasAllRequiredPermissions(): Boolean {
        return requiredPermissions().all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestRequiredPermissions() {
        val perms = requiredPermissions()
        val missing = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            Log.i(TAG, "Requesting permissions: ${missing.joinToString()}")
            permissionLauncher.launch(missing.toTypedArray())
        } else {
            Log.i(TAG, "All required permissions already granted")
            startAgentService()
        }
    }

    private fun promptBatteryOptimizationExemption() {
        val keys = prefsKeys()
        val askedKey = keys.batteryExemptionAsked ?: return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return
        if (prefs.getBoolean(askedKey, false)) return

        val powerManager = getSystemService(Context.POWER_SERVICE) as? android.os.PowerManager ?: return
        if (powerManager.isIgnoringBatteryOptimizations(packageName)) {
            prefs.edit().putBoolean(askedKey, true).apply()
            return
        }

        prefs.edit().putBoolean(askedKey, true).apply()
        try {
            val intent = Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                data = android.net.Uri.parse("package:$packageName")
            }
            startActivity(intent)
            Log.i(TAG, "Prompted for battery optimization exemption")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to prompt for battery optimization exemption", e)
        }
    }

    private fun configureFullscreen() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                )
        }
    }

    protected fun startAgentService() {
        val intent = Intent(this, agentServiceClass())
        ContextCompat.startForegroundService(this, intent)
    }

    // ── Content refresh (JS injection) ────────────────────────────────────────

    private fun triggerContentRefresh() {
        onPlatformContentRefresh()

        currentWebView().evaluateJavascript(
            """
            (function() {
                console.log('[Agent] Content refresh triggered via JS injection');
                if (window.minimalLBar && typeof window.minimalLBar.refresh === 'function') {
                    window.minimalLBar.refresh(true);
                }
                if (window.minimalStreamManager && typeof window.minimalStreamManager.fetchAndUpdateStream === 'function') {
                    window.minimalStreamManager.fetchAndUpdateStream(true);
                }
                if (window.minimalAdvertisementOverlay && typeof window.minimalAdvertisementOverlay.checkAndShowAds === 'function') {
                    window.minimalAdvertisementOverlay.checkAndShowAds(true, true);
                }
                return 'refresh_triggered';
            })();
            """.trimIndent(),
            null
        )
    }

    private fun requestOverlayMaintenance(reason: String, aggressive: Boolean) {
        val webView = activeWebView ?: runCatching { bindingAdapter().webView }.getOrNull() ?: return
        val escapedReason = JSONObject.quote(reason)
        val aggressiveLiteral = if (aggressive) "true" else "false"
        webView.post {
            try {
                webView.evaluateJavascript(
                    """
                    (function() {
                        try {
                            if (typeof window.__trillboardsRunMaintenance === 'function') {
                                return window.__trillboardsRunMaintenance($escapedReason, $aggressiveLiteral);
                            }
                            if (typeof performance !== 'undefined' && typeof performance.clearResourceTimings === 'function') {
                                performance.clearResourceTimings();
                            }
                            return JSON.stringify({ ok: true, fallback: true, reason: $escapedReason });
                        } catch (e) {
                            return JSON.stringify({
                                ok: false,
                                reason: $escapedReason,
                                error: String((e && e.message) || e || 'unknown')
                            });
                        }
                    })();
                    """.trimIndent()
                ) { result ->
                    Log.d(TAG, "Overlay maintenance ($reason): $result")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Overlay maintenance injection failed for $reason", e)
            }
        }
    }

    // ── WebView reload + retry ladder ─────────────────────────────────────────

    fun reloadWebView(reason: String = "manual") {
        Log.i(TAG, "Reloading WebView: $reason")
        onPlatformReload(reason)
        bindingAdapter().loadingIndicator.visibility = View.VISIBLE
        startPageLoadTimeout()
        currentWebView().reload()
    }

    private fun scheduleRetry(reason: String) {
        cancelRetry()
        bindingAdapter().loadingIndicator.visibility = View.VISIBLE
        val delayMs = if (retryCount >= MAX_RETRY_ATTEMPTS) {
            MAX_RETRY_DELAY_MS
        } else {
            (BASE_RETRY_DELAY_MS * (1L shl retryCount.coerceAtMost(4))).coerceAtMost(MAX_RETRY_DELAY_MS)
        }
        Log.i(TAG, "Scheduling WebView retry #${retryCount + 1} in ${delayMs}ms (reason=$reason)")
        val runnable = Runnable {
            retryCount++
            if (retryCount > MAX_RETRY_ATTEMPTS) {
                retryCount = 0
            }
            reloadWebView("retry:$reason")
        }
        retryRunnable = runnable
        webViewRecoveryHandler.postDelayed(runnable, delayMs)
    }

    private fun cancelRetry() {
        retryRunnable?.let { webViewRecoveryHandler.removeCallbacks(it) }
        retryRunnable = null
    }

    private fun startPageLoadTimeout() {
        cancelPageLoadTimeout()
        val runnable = Runnable {
            Log.w(TAG, "Page load timeout (${PAGE_LOAD_TIMEOUT_MS}ms) — forcing WebView retry")
            scheduleRetry("page_load_timeout")
        }
        pageLoadTimeoutRunnable = runnable
        webViewRecoveryHandler.postDelayed(runnable, PAGE_LOAD_TIMEOUT_MS)
    }

    private fun cancelPageLoadTimeout() {
        pageLoadTimeoutRunnable?.let { webViewRecoveryHandler.removeCallbacks(it) }
        pageLoadTimeoutRunnable = null
    }

    // ── WebView setup ─────────────────────────────────────────────────────────

    private fun setupWebView() {
        val webView = currentWebView()
        with(webView.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            cacheMode = WebSettings.LOAD_DEFAULT
            allowFileAccess = true
            allowContentAccess = true
            mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
            mediaPlaybackRequiresUserGesture = false
            setSupportZoom(false)
            builtInZoomControls = false
            displayZoomControls = false
            // 1:1 pixel mapping — prevents ads from being zoomed/cropped.
            loadWithOverviewMode = false
            useWideViewPort = false
            // Geolocation for mobile screen tracking (rideshare/taxi).
            setGeolocationEnabled(true)
            setGeolocationDatabasePath(filesDir.path)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            webView.setRendererPriorityPolicy(WebView.RENDERER_PRIORITY_IMPORTANT, false)
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                cancelRetry()
                cancelPageLoadTimeout()
                retryCount = 0
                bindingAdapter().loadingIndicator.visibility = View.GONE

                // Synthetic touch + IMA user-activation injection (audio unlock + ad readiness).
                view?.postDelayed({
                    val x = view.width / 2f
                    val y = view.height / 2f
                    val downTime = android.os.SystemClock.uptimeMillis()
                    val downEvent = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, x, y, 0)
                    val upEvent = MotionEvent.obtain(downTime, downTime + 50, MotionEvent.ACTION_UP, x, y, 0)
                    view.dispatchTouchEvent(downEvent)
                    view.dispatchTouchEvent(upEvent)
                    downEvent.recycle()
                    upEvent.recycle()
                }, 1500)

                // IMA user-activation event — fires BEFORE IMA tries to init at ~600ms.
                view?.postDelayed({
                    view.evaluateJavascript(
                        """
                        (function() {
                            console.log('[Agent] Triggering user activation for IMA');
                            window.__TRILL_USER_ACTIVATION_TS__ = Date.now();
                            window.__TRILL_USER_ACTIVATION_SOURCE__ = 'native_injection';
                            try {
                                document.dispatchEvent(new CustomEvent('trillboards:userActivation', {
                                    detail: { source: 'native_injection', ts: Date.now() }
                                }));
                            } catch (e) {
                                console.log('[Agent] User activation dispatch error:', e);
                            }
                        })();
                        """.trimIndent(),
                        null
                    )
                }, 300)

                // Inject physical connection type from Android ConnectivityManager.
                view?.postDelayed({
                    view.evaluateJavascript(
                        """window.__TRILL_CONNECTION_TYPE__ = '${getTransportType()}';""",
                        null
                    )
                }, 200)

                // JS-fallback audio unlock if synthetic touch did not propagate.
                view?.postDelayed({
                    view.evaluateJavascript(
                        """
                        (function() {
                            if (window.audioStateManager && !window.audioStateManager.audioUnlocked) {
                                console.log('[Agent] JS fallback: unlocking audio');
                                window.audioStateManager.audioUnlocked = true;
                                window.audioStateManager.currentAudioState = 'unmuted';
                                window.audioStateManager.savePersistedState();
                                window.audioStateManager.unlockAllAudio();
                            }
                        })();
                        """.trimIndent(),
                        null
                    )
                }, 3000)

                // FPS counter — measured via requestAnimationFrame, reported through
                // TrillboardsNativeDevice bridge every 5 s. Used to correlate WebView
                // smoothness with ML inference load.
                view?.postDelayed({
                    view.evaluateJavascript(
                        """
                        (function() {
                            if (window.__trillFpsRunning) return;
                            window.__trillFpsRunning = true;
                            var frames = 0, lastTime = performance.now();
                            function tick(now) {
                                frames++;
                                var elapsed = now - lastTime;
                                if (elapsed >= 5000) {
                                    var fps = frames * 1000 / elapsed;
                                    frames = 0;
                                    lastTime = now;
                                    try {
                                        window.TrillboardsNativeDevice.reportFps(fps);
                                    } catch(e) {}
                                }
                                requestAnimationFrame(tick);
                            }
                            requestAnimationFrame(tick);
                        })();
                        """.trimIndent(),
                        null
                    )
                }, 2000)
            }

            override fun onRenderProcessGone(view: WebView?, detail: RenderProcessGoneDetail?): Boolean {
                val didCrash = detail?.didCrash() ?: true
                Log.e(TAG, "WebView render process gone! didCrash=$didCrash, rendererPriority=${detail?.rendererPriorityAtExit()}")
                recreateWebView("render_process_gone")
                return true
            }

            override fun onReceivedError(
                view: WebView?,
                request: android.webkit.WebResourceRequest?,
                error: android.webkit.WebResourceError?
            ) {
                if (request?.isForMainFrame == true) {
                    Log.e(TAG, "WebView error: ${error?.errorCode} - ${error?.description}")
                    scheduleRetry("main_frame_error")
                }
            }

            override fun onReceivedHttpError(
                view: WebView?,
                request: android.webkit.WebResourceRequest?,
                errorResponse: android.webkit.WebResourceResponse?
            ) {
                if (request?.isForMainFrame == true) {
                    Log.e(TAG, "WebView HTTP error: ${errorResponse?.statusCode} for ${request.url}")
                    scheduleRetry("main_frame_http_error")
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                bindingAdapter().loadingIndicator.visibility = if (newProgress == 100) View.GONE else View.VISIBLE
            }

            override fun onConsoleMessage(consoleMessage: ConsoleMessage?): Boolean {
                consoleMessage?.let {
                    val tag = "WebViewConsole"
                    val msg = "[${it.sourceId()}:${it.lineNumber()}] ${it.message()}"
                    when (it.messageLevel()) {
                        ConsoleMessage.MessageLevel.ERROR -> Log.e(tag, msg)
                        ConsoleMessage.MessageLevel.WARNING -> Log.w(tag, msg)
                        else -> Log.i(tag, msg)
                    }
                }
                return true
            }

            override fun onGeolocationPermissionsShowPrompt(
                origin: String?,
                callback: GeolocationPermissions.Callback?
            ) {
                Log.d(TAG, "Geolocation permission requested for origin: $origin")
                if (hasLocationPermission()) {
                    callback?.invoke(origin, true, false)
                } else {
                    Log.w(TAG, "Location permission not granted, denying webview geolocation")
                    callback?.invoke(origin, false, false)
                }
            }
        }

        activeWebView = webView
        // Register WebView with Google Mobile Ads SDK (must be before loadUrl).
        WebViewAdsRegistrar.register(webView)
        // Attach native device identity bridge (must be before loadUrl).
        nativeDeviceBridge = NativeDeviceBridge.attach(webView, this)
        loadScreenUrl()
    }

    private fun loadScreenUrl() {
        val keys = prefsKeys()
        val fingerprint = DeviceIdentity.stableFingerprint(this, keys.sharedPrefsName)
        val sessionId = DeviceIdentity.ephemeralSessionId()
        val deviceToken = prefs.getString(keys.deviceToken, null)
        val targetUrl = OverlayUrlBuilder.build(
            baseUrl = getString(overlayUrlResId()),
            fingerprint = fingerprint,
            sessionId = sessionId,
            deviceToken = deviceToken
        )
        bindingAdapter().loadingIndicator.visibility = View.VISIBLE
        startPageLoadTimeout()
        currentWebView().loadUrl(targetUrl)
    }

    /**
     * Recreate the WebView — programmatically destroy the current instance, build
     * a fresh WebView, insert it into the layout (before the touch interceptor on
     * tablet, as the first child on TV), and re-run [setupWebView] which re-loads
     * the screen URL.
     *
     * Implements [AgentRecreatableActivity] — agent-core's recovery ladder + command
     * processor invoke this to soft-reboot the player without coupling to a specific
     * platform's MainActivity.
     */
    override fun recreateWebView(reason: String) {
        Log.i(TAG, "Recreating WebView: $reason")
        cancelRetry()
        cancelPageLoadTimeout()
        bindingAdapter().loadingIndicator.visibility = View.VISIBLE

        webViewRecoveryHandler.postDelayed({
            val adapter = bindingAdapter()
            val rootGroup = (adapter.webView.parent as? ViewGroup) ?: return@postDelayed
            val toReplace = activeWebView ?: adapter.webView
            (toReplace.parent as? ViewGroup)?.removeView(toReplace)
            try {
                toReplace.stopLoading()
                toReplace.webChromeClient = null
                toReplace.destroy()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to fully destroy replaced WebView", e)
            }
            activeWebView = null
            val newWebView = WebView(this)
            newWebView.layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            // On tablet the touch interceptor sits above the WebView; insert before it
            // so kiosk long-press still receives events. On TV touchInterceptorOrNull is
            // null, so insert at index 0.
            val touchInterceptor = adapter.touchInterceptorOrNull
            if (touchInterceptor != null) {
                val interceptorIdx = rootGroup.indexOfChild(touchInterceptor)
                if (interceptorIdx > 0) {
                    rootGroup.addView(newWebView, interceptorIdx)
                } else {
                    rootGroup.addView(newWebView, 0)
                }
            } else {
                rootGroup.addView(newWebView, 0)
            }
            activeWebView = newWebView
            setupWebView()
            onWebViewRecreated(newWebView)
        }, 500)
    }

    // ── Receivers ─────────────────────────────────────────────────────────────

    private fun registerReceivers() {
        val keys = intentKeys()
        val overlayFilter = IntentFilter().apply {
            addAction(keys.overlayRefresh)
            addAction(keys.overlayBlackout)
        }
        broadcastManager.registerReceiver(overlayReceiver, overlayFilter)
        broadcastManager.registerReceiver(maintenanceReceiver, IntentFilter(keys.maintenanceCycle))
        broadcastManager.registerReceiver(clearCacheReceiver, IntentFilter(keys.clearCache))
        broadcastManager.registerReceiver(kioskControlReceiver, IntentFilter(keys.kioskControl))
    }

    private fun unregisterReceivers() {
        runCatching { broadcastManager.unregisterReceiver(overlayReceiver) }
        runCatching { broadcastManager.unregisterReceiver(maintenanceReceiver) }
        runCatching { broadcastManager.unregisterReceiver(clearCacheReceiver) }
        runCatching { broadcastManager.unregisterReceiver(kioskControlReceiver) }
    }

    private fun startKioskModeIfEnabled() {
        if (prefs.getBoolean(prefsKeys().kioskModeEnabled, false)) {
            if (kioskLockManager.isDeviceOwner()) {
                kioskLockManager.enableKioskMode()
                kioskLockManager.configureDeviceOwnerPolicies()
                startLockTask()
            }
        }
    }

    // ── Long-press helper (tablet uses this; TV does not) ─────────────────────

    /**
     * Tablet uses this to wire its touch-interceptor view to the kiosk PIN flow.
     * Call from [wireKioskTrigger] in tablet's `MainActivity`. Returns immediately
     * (no-op) if `bindingAdapter().touchInterceptorOrNull` is null.
     */
    protected fun armLongPressOnInterceptor() {
        val interceptor = bindingAdapter().touchInterceptorOrNull ?: return
        interceptor.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    longPressRunnable = Runnable {
                        triggerKioskSettingsDialog()
                    }
                    longPressHandler.postDelayed(longPressRunnable!!, LONG_PRESS_DURATION_MS)
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    longPressRunnable?.let { longPressHandler.removeCallbacks(it) }
                }
            }
            currentWebView().dispatchTouchEvent(event)
            true
        }
    }

    /** Both platforms call this to start the PIN-protected Settings dialog. */
    protected fun triggerKioskSettingsDialog() {
        showPinDialog { pinCorrect ->
            if (pinCorrect) {
                showSettingsDialog()
            } else {
                Toast.makeText(this, "Incorrect PIN", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showPinDialog(onResult: (Boolean) -> Unit) {
        val storedPin = prefs.getString(prefsKeys().kioskPin, "0000")
        val pinInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            hint = "Enter 4-digit PIN"
        }
        AlertDialog.Builder(this)
            .setTitle("Enter PIN")
            .setMessage("Enter the kiosk PIN to access settings")
            .setView(pinInput)
            .setPositiveButton("OK") { _, _ ->
                val enteredPin = pinInput.text.toString()
                onResult(enteredPin == storedPin)
            }
            .setNegativeButton("Cancel") { _, _ ->
                onResult(false)
            }
            .show()
    }

    private fun showChangePinDialog() {
        val pinInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            hint = "Enter new 4-digit PIN"
        }
        AlertDialog.Builder(this)
            .setTitle("Change PIN")
            .setMessage("Enter a new 4-digit PIN for kiosk settings")
            .setView(pinInput)
            .setPositiveButton("Save") { _, _ ->
                val newPin = pinInput.text.toString()
                if (newPin.length == 4 && newPin.all { it.isDigit() }) {
                    prefs.edit().putString(prefsKeys().kioskPin, newPin).apply()
                    Toast.makeText(this, "PIN changed successfully", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "PIN must be exactly 4 digits", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showSettingsDialog() {
        val isKioskEnabled = kioskLockManager.isInLockTaskMode()
        val isDeviceOwner = kioskLockManager.isDeviceOwner()
        val keys = prefsKeys()

        val options = mutableListOf<String>()
        if (isDeviceOwner) {
            if (isKioskEnabled) options.add("Disable Kiosk Mode") else options.add("Enable Kiosk Mode")
        } else {
            options.add("Kiosk Mode (Device Owner Required)")
        }
        options.add("Change PIN")
        options.add("Refresh Screen")
        options.add("Exit App")
        options.add("Cancel")

        AlertDialog.Builder(this)
            .setTitle("Trillboard Settings")
            .setItems(options.toTypedArray()) { _, which ->
                when (options[which]) {
                    "Enable Kiosk Mode" -> {
                        if (kioskLockManager.enableKioskMode()) {
                            kioskLockManager.configureDeviceOwnerPolicies()
                            prefs.edit().putBoolean(keys.kioskModeEnabled, true).apply()
                            startLockTask()
                            Toast.makeText(this, "Kiosk mode enabled", Toast.LENGTH_SHORT).show()
                        }
                    }
                    "Disable Kiosk Mode" -> {
                        stopLockTask()
                        kioskLockManager.disableKioskMode()
                        prefs.edit().putBoolean(keys.kioskModeEnabled, false).apply()
                        Toast.makeText(this, "Kiosk mode disabled", Toast.LENGTH_SHORT).show()
                    }
                    "Change PIN" -> showChangePinDialog()
                    "Refresh Screen" -> {
                        reloadWebView("settings_refresh")
                        Toast.makeText(this, "Refreshing...", Toast.LENGTH_SHORT).show()
                    }
                    "Exit App" -> {
                        if (isKioskEnabled) {
                            stopLockTask()
                            kioskLockManager.disableKioskMode()
                        }
                        finishAffinity()
                    }
                }
            }
            .show()
    }

    // ── Memory pressure (cooldown-guarded; TV-style behavior shared) ──────────

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        when (level) {
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> {
                Log.w(TAG, "onTrimMemory: RUNNING_LOW - ignoring (cooldown-guarded at CRITICAL only)")
            }
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> {
                val now = System.currentTimeMillis()
                if (now - lastCacheClearMs >= CACHE_CLEAR_COOLDOWN_MS) {
                    Log.e(TAG, "onTrimMemory: RUNNING_CRITICAL - clearing WebView cache (cooldown elapsed)")
                    currentWebView().clearCache(true)
                    requestOverlayMaintenance("trim_memory_running_critical", aggressive = true)
                    onPlatformTrimMemory(level)
                    lastCacheClearMs = now
                    System.gc()
                } else {
                    Log.w(
                        TAG,
                        "onTrimMemory: RUNNING_CRITICAL - skipping (cooldown active, " +
                            "${(CACHE_CLEAR_COOLDOWN_MS - (now - lastCacheClearMs)) / 1000}s remaining)"
                    )
                }
            }
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onResume() {
        super.onResume()
        currentWebView().onResume()
        onPlatformResume()
        if (useImmersiveFullscreen()) {
            configureFullscreen()
        }

        pendingSensorRebindOnResume = wasStoppedSinceLastResume
        wasStoppedSinceLastResume = false

        if (!pendingSensorRebindOnResume) {
            Log.d(TAG, "onResume: skipping sensor rebind without a full background transition")
            return
        }
        pendingSensorRebindOnResume = false

        if (!shouldRebindSensorsOnResume()) {
            Log.d(TAG, "onResume: sensor rebind disabled by platform (HAS_AUDIENCE_SENSING=false)")
            return
        }

        val keys = intentKeys()
        Log.d(TAG, "onResume: Scheduling INTENT_REBIND_CAMERA broadcast")
        broadcastManager.sendBroadcast(
            Intent(keys.rebindCamera)
                .putExtra(keys.extraRebindReason, "activity_resume")
                .putExtra(keys.extraRebindDelayMs, SENSOR_REBIND_AFTER_RESUME_DELAY_MS)
        )
        Log.d(TAG, "onResume: Scheduling INTENT_REBIND_AUDIO broadcast")
        broadcastManager.sendBroadcast(
            Intent(keys.rebindAudio)
                .putExtra(keys.extraRebindReason, "activity_resume")
                .putExtra(keys.extraRebindDelayMs, SENSOR_REBIND_AFTER_RESUME_DELAY_MS)
        )
    }

    override fun onPause() {
        currentWebView().onPause()
        onPlatformPause()
        super.onPause()
    }

    override fun onStop() {
        wasStoppedSinceLastResume = true
        super.onStop()
    }

    override fun onDestroy() {
        unregisterTransportTypeCallback()
        longPressRunnable?.let { longPressHandler.removeCallbacks(it) }
        cancelRetry()
        cancelPageLoadTimeout()
        unregisterReceivers()
        onPlatformDestroy()
        val webViewToDestroy = activeWebView ?: runCatching { bindingAdapter().webView }.getOrNull()
        try {
            webViewToDestroy?.stopLoading()
            webViewToDestroy?.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to destroy active WebView", e)
        }
        activeWebView = null
        super.onDestroy()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (kioskLockManager.isInLockTaskMode()) {
            return
        }
        val webView = currentWebView()
        if (webView.canGoBack()) {
            webView.goBack()
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus && useImmersiveFullscreen()) {
            configureFullscreen()
        }
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (event.action == KeyEvent.ACTION_DOWN && isRefreshKey(event.keyCode)) {
            reloadWebView("dpad_or_enter")
            return true
        }
        return super.dispatchKeyEvent(event)
    }

    // ── Transport type for telemetry ──────────────────────────────────────────

    protected fun getTransportType(): String {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.activeNetwork?.let { cm.getNetworkCapabilities(it) }
        return when {
            caps == null -> "none"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "wifi"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
            caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
            else -> "other"
        }
    }

    private fun registerTransportTypeCallback() {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val request = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) {
                val type = when {
                    caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "wifi"
                    caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
                    caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
                    else -> "other"
                }
                val webView = activeWebView ?: runCatching { bindingAdapter().webView }.getOrNull()
                webView?.post {
                    webView.evaluateJavascript("window.__TRILL_CONNECTION_TYPE__ = '$type';", null)
                }
            }
        }
        networkCallback = callback
        try {
            cm.registerNetworkCallback(request, callback)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to register transport-type network callback", e)
        }
    }

    private fun unregisterTransportTypeCallback() {
        networkCallback?.let {
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            try { cm.unregisterNetworkCallback(it) } catch (_: Exception) {}
        }
        networkCallback = null
    }
}

/**
 * Internal helper to read `BuildConfig.DEBUG` reflectively, since each platform
 * generates its own `BuildConfig` and `BaseAgentActivity` doesn't know which one.
 *
 * Avoids leaking each platform's `BuildConfig.DEBUG` into the base via an
 * abstract `isDebug()` (every platform would have to wire it identically).
 */
private object BuildConfigDebug {
    fun isDebug(context: Context): Boolean {
        return try {
            val info = context.packageManager.getApplicationInfo(context.packageName, 0)
            (info.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
        } catch (_: Exception) {
            false
        }
    }
}
