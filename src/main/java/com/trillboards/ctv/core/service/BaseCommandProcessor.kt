package com.trillboards.ctv.core.service

import android.app.Activity
import android.app.PendingIntent
import android.app.admin.DevicePolicyManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.net.Uri
import android.util.Log
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.trillboards.ctv.core.activity.AgentRecreatableActivity
import com.trillboards.ctv.core.device.AgentPowerAdapter
import com.trillboards.ctv.core.device.KioskLockManager
import com.trillboards.ctv.core.device.ScreenshotManager
import com.trillboards.ctv.core.diagnostics.DiagnosticsBundleCollector
import com.trillboards.ctv.core.net.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Platform-agnostic command processor extracted from `tablet-agent/.../service/CommandProcessor.kt`
 * (PR 2b/5 of the agent-core unification series).
 *
 * Each platform agent (tablet-agent, android-tv-agent, fire-tv-agent, tablet-agent-lite)
 * extends this class with a thin shell that supplies platform-specific identifiers
 * (apkManifestKey, installCompleteAction, deviceAdminComponent, mainActivityClass,
 * BuildConfig version code/name, supported rendering modes).
 *
 * Capability gating happens via the `AgentPowerAdapter` interface — power, blackout,
 * volume, mute, brightness all delegate to the platform's `PowerController`. Platforms
 * that don't support a capability return `false` from the adapter; the base maps that
 * to `failed` ack envelope. This avoids per-platform `if` branches at the top of
 * `handle()`.
 *
 * Sensing profile application (`updateSensingProfile`) is a protected open hook: the
 * full agent-core `audience.SensingProfileManager` lives outside agent-core-lite, so
 * the base does NOT take a manager reference directly. Tablet's thin shell overrides
 * `applySensingProfile()` to delegate to its `SensingProfileManager`; lite agents
 * inherit the default `not supported` failure.
 *
 * Brand-list updates (`audience.updateBrandList`) ride a public callback
 * `onBrandListUpdate` so DeviceAgentService can wire it to its
 * `AudienceSensingService` lifecycle without coupling the base to that class.
 */
abstract class BaseCommandProcessor(
    protected val context: Context,
    protected val powerAdapter: AgentPowerAdapter,
    protected val apiClient: ApiClient,
    protected val fingerprint: String,
    protected val activityProvider: () -> Activity?,
    protected val intentKeys: IntentKeys,
    protected val prefsKeys: PrefsKeys
) {
    companion object {
        private const val TAG = "CommandProcessor"
        private const val ACK_ACKED = "acked"
        private const val ACK_FAILED = "failed"
    }

    /**
     * Wired by `BaseDeviceAgentService` (PR 3) so `audience.updateBrandList`
     * commands route into the running `AudienceSensingService` without coupling
     * BaseCommandProcessor to that class.
     */
    var onBrandListUpdate: ((List<String>) -> Unit)? = null

    protected val broadcastManager: LocalBroadcastManager = LocalBroadcastManager.getInstance(context)
    protected val devicePolicyManager: DevicePolicyManager =
        context.getSystemService(Context.DEVICE_POLICY_SERVICE) as DevicePolicyManager
    protected val kioskLockManager: KioskLockManager by lazy {
        KioskLockManager(context, deviceAdminComponent())
    }
    protected val screenshotManager: ScreenshotManager by lazy {
        ScreenshotManager(
            OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .readTimeout(15, TimeUnit.SECONDS)
                .build()
        )
    }

    data class CommandResult(
        val status: String,
        val details: Map<String, Any?>
    )

    // ── Per-platform abstract surface ────────────────────────────────────────
    /** APK manifest key for OTA update lookups (e.g., "tablet-agent", "android-tv-agent"). */
    protected abstract fun apkManifestKey(): String

    /** Broadcast action fired by `PackageInstaller.commit` after silent APK install. */
    protected abstract fun installCompleteAction(): String

    /** ComponentName of the platform's `DeviceAdminReceiver` (used by `kioskLockManager` + reboot). */
    protected abstract fun deviceAdminComponent(): ComponentName

    /** Platform's MainActivity class — used by `app.resume`. */
    protected abstract fun mainActivityClass(): Class<out Activity>

    /** `BuildConfig.VERSION_CODE` from the platform's app module. */
    protected abstract fun currentVersionCode(): Int

    /** `BuildConfig.VERSION_NAME` from the platform's app module. */
    protected abstract fun currentVersionName(): String

    /**
     * Set of valid rendering modes for `config.push.rendering_mode`. Empty set =
     * platform doesn't support rendering-mode switching (all TV agents). The
     * tablet returns the 3-mode set (webview / native / hybrid).
     */
    protected open fun supportedRenderingModes(): Set<String> = emptySet()

    /**
     * Subclasses with a `SensingProfileManager` (full agent-core flavor on
     * tablet-agent, full android-tv-agent, etc.) override this to apply the
     * profile. Lite flavors inherit the `not supported` failure. Called from
     * `updateSensingProfile` command handler.
     */
    protected open suspend fun applySensingProfile(payload: JSONObject): CommandResult =
        failure("SensingProfileManager not available")

    suspend fun handle(command: String, payload: JSONObject?): CommandResult {
        return when {
            command.startsWith("power.") -> handlePowerCommand(command)
            command.startsWith("blackout.") -> handleBlackoutCommand(command)
            command.startsWith("volume.") -> handleVolumeCommand(command, payload)
            command.startsWith("mute.") -> handleMuteCommand(command)
            command.startsWith("refresh.") -> handleRefreshCommand(command)
            command.startsWith("brightness.") -> handleBrightnessCommand(command, payload)
            command == "app.reload" -> handleAppReload()
            command == "app.launch" -> handleAppLaunch(payload)
            command == "app.home" -> handleAppHome()
            command == "app.resume" -> handleAppResume()
            command == "device.screenshot" -> handleScreenshotCommand()
            command == "device.checkUpdate" -> handleCheckUpdateCommand()
            command == "device.installApk" -> handleInstallApkCommand(payload)
            command == "device.diagBundle" -> handleDiagBundleCommand()
            command.startsWith("device.") -> handleDeviceCommand(command)
            command.startsWith("kiosk.") -> handleKioskCommand(command)
            command == "config.push" -> handleConfigPushCommand(payload)
            command == "audience.updateBrandList" -> handleUpdateBrandListCommand(payload)
            command == "updateSensingProfile" -> handleUpdateSensingProfile(payload)
            else -> failure("Unknown command: $command")
        }
    }

    // ── Power / blackout ──────────────────────────────────────────────────────

    private fun handlePowerCommand(command: String): CommandResult {
        val ok = when (command) {
            "power.on" -> powerAdapter.powerOn()
            "power.off" -> powerAdapter.powerOff()
            "power.toggle" -> powerAdapter.togglePower()
            else -> false
        }
        return CommandResult(
            status = if (ok) ACK_ACKED else ACK_FAILED,
            details = mapOf(
                "command" to command,
                "success" to ok,
                "mode" to "soft_blackout"
            )
        )
    }

    private fun handleBlackoutCommand(command: String): CommandResult {
        val enabled = command == "blackout.start"
        val ok = powerAdapter.setBlackout(enabled)
        return if (ok) {
            CommandResult(
                status = ACK_ACKED,
                details = mapOf("blackout" to enabled)
            )
        } else {
            failure("Blackout not supported on this device")
        }
    }

    // ── Volume / mute ─────────────────────────────────────────────────────────

    private fun handleVolumeCommand(command: String, payload: JSONObject?): CommandResult {
        return try {
            val ok = when (command) {
                "volume.set" -> {
                    val level = payload?.optInt("level", -1) ?: -1
                    if (level !in 0..100) {
                        return failure("Invalid volume level: $level")
                    }
                    powerAdapter.setVolumePercent(level, showUi = true)
                }
                "volume.up" -> powerAdapter.volumeUp(showUi = true)
                "volume.down" -> powerAdapter.volumeDown(showUi = true)
                else -> false
            }
            if (!ok) {
                return failure("Volume control not supported on this device")
            }
            val current = powerAdapter.currentVolumePercent()
            CommandResult(
                status = ACK_ACKED,
                details = mapOf("volume" to current)
            )
        } catch (e: Exception) {
            failure(e.message ?: "Volume control failed")
        }
    }

    private fun handleMuteCommand(command: String): CommandResult {
        return try {
            val ok = when (command) {
                "mute.toggle" -> powerAdapter.toggleMute()
                "mute.on" -> powerAdapter.muteOn(showUi = true)
                "mute.off" -> powerAdapter.muteOff(showUi = true)
                else -> false
            }
            if (!ok) {
                return failure("Mute control not supported on this device")
            }
            CommandResult(
                status = ACK_ACKED,
                details = mapOf("muted" to powerAdapter.isMuted())
            )
        } catch (e: Exception) {
            failure(e.message ?: "Mute control failed")
        }
    }

    // ── Refresh / brightness ──────────────────────────────────────────────────

    private fun handleRefreshCommand(command: String): CommandResult {
        broadcastManager.sendBroadcast(Intent(intentKeys.overlayRefresh))
        return CommandResult(
            status = ACK_ACKED,
            details = mapOf("action" to "refresh")
        )
    }

    private fun handleBrightnessCommand(command: String, payload: JSONObject?): CommandResult {
        val level = payload?.optInt("level", -1) ?: -1
        return if (level in 0..100) {
            val brightness = (level * 255 / 100)
            val success = powerAdapter.setBrightness(brightness)
            CommandResult(
                status = if (success) ACK_ACKED else ACK_FAILED,
                details = mapOf("brightness" to level, "success" to success)
            )
        } else {
            failure("Invalid brightness level: $level")
        }
    }

    // ── App lifecycle ─────────────────────────────────────────────────────────

    private fun handleAppReload(): CommandResult {
        val activity = activityProvider()
        val recreatable = activity as? AgentRecreatableActivity
        if (recreatable != null && activity != null) {
            activity.runOnUiThread {
                recreatable.recreateWebView("app_reload")
            }
            return CommandResult(
                status = ACK_ACKED,
                details = mapOf("action" to "recreate_webview")
            )
        }

        broadcastManager.sendBroadcast(Intent(intentKeys.overlayRefresh))
        return CommandResult(
            status = ACK_ACKED,
            details = mapOf("action" to "reload")
        )
    }

    private suspend fun handleAppLaunch(payload: JSONObject?): CommandResult {
        val packageName =
            payload?.optString("package")?.takeIf { it.isNotBlank() }
                ?: payload?.optString("packageName")?.takeIf { it.isNotBlank() }
        val uriString =
            payload?.optString("deepLink")?.takeIf { it.isNotBlank() }
                ?: payload?.optString("url")?.takeIf { it.isNotBlank() }

        val intent = when {
            !uriString.isNullOrBlank() -> Intent(Intent.ACTION_VIEW, Uri.parse(uriString)).apply {
                if (!packageName.isNullOrBlank()) setPackage(packageName)
            }
            !packageName.isNullOrBlank() -> context.packageManager.getLaunchIntentForPackage(packageName)
            else -> null
        }?.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            ?: return failure("Invalid app.launch payload")

        val launchError = startActivitySafely(intent)
        return if (launchError == null) {
            CommandResult(
                status = ACK_ACKED,
                details = mapOf(
                    "launched" to true,
                    "package" to (packageName ?: ""),
                    "uri" to (uriString ?: "")
                )
            )
        } else {
            failure(launchError)
        }
    }

    private suspend fun handleAppHome(): CommandResult {
        val intent = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_HOME)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        val launchError = startActivitySafely(intent)
        return if (launchError == null) {
            CommandResult(status = ACK_ACKED, details = mapOf("home" to true))
        } else {
            failure(launchError)
        }
    }

    private suspend fun handleAppResume(): CommandResult {
        val intent = Intent(context, mainActivityClass()).apply {
            addFlags(
                Intent.FLAG_ACTIVITY_NEW_TASK or
                    Intent.FLAG_ACTIVITY_CLEAR_TOP or
                    Intent.FLAG_ACTIVITY_SINGLE_TOP
            )
        }
        val launchError = startActivitySafely(intent)
        return if (launchError == null) {
            CommandResult(status = ACK_ACKED, details = mapOf("resumed" to true))
        } else {
            failure(launchError)
        }
    }

    private suspend fun startActivitySafely(intent: Intent): String? = withContext(Dispatchers.Main) {
        try {
            context.startActivity(intent)
            null
        } catch (ex: Exception) {
            ex.message ?: "Failed to start activity"
        }
    }

    // ── Device-level commands ─────────────────────────────────────────────────

    private fun handleDeviceCommand(command: String): CommandResult {
        return when (command) {
            "device.restart" -> {
                val intent = context.packageManager.getLaunchIntentForPackage(context.packageName)
                intent?.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK)
                if (intent != null) {
                    context.startActivity(intent)
                    android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                        android.os.Process.killProcess(android.os.Process.myPid())
                    }, 500)
                    CommandResult(
                        status = ACK_ACKED,
                        details = mapOf("action" to "restart")
                    )
                } else {
                    failure("Could not create restart intent")
                }
            }
            "device.clearCache" -> {
                broadcastManager.sendBroadcast(Intent(intentKeys.clearCache))
                CommandResult(
                    status = ACK_ACKED,
                    details = mapOf("cache" to "cleared")
                )
            }
            "device.reboot" -> {
                if (!kioskLockManager.isDeviceOwner()) {
                    return failure("Device Owner required for reboot")
                }
                try {
                    devicePolicyManager.reboot(deviceAdminComponent())
                    CommandResult(
                        status = ACK_ACKED,
                        details = mapOf("action" to "reboot")
                    )
                } catch (e: Exception) {
                    failure("Reboot failed: ${e.message}")
                }
            }
            else -> failure("Unknown device command: $command")
        }
    }

    private suspend fun handleScreenshotCommand(): CommandResult {
        val activity = activityProvider()
            ?: return failure("No activity available for screenshot")

        val uploadInfo = apiClient.getScreenshotUploadUrl(fingerprint)
            ?: return failure("Failed to get screenshot upload URL")

        val result = screenshotManager.captureAndUpload(
            activity = activity,
            uploadUrl = uploadInfo.uploadUrl,
            cdnUrl = uploadInfo.cdnUrl
        )

        return if (result.success) {
            CommandResult(
                status = ACK_ACKED,
                details = mapOf("cdnUrl" to (result.cdnUrl ?: ""))
            )
        } else {
            failure(result.error ?: "Screenshot capture failed")
        }
    }

    // ── Kiosk ─────────────────────────────────────────────────────────────────

    private fun handleKioskCommand(command: String): CommandResult {
        return when (command) {
            "kiosk.enable" -> {
                if (!kioskLockManager.isDeviceOwner()) {
                    return failure("Device Owner required for kiosk mode")
                }
                val enabled = kioskLockManager.enableKioskMode()
                if (enabled) {
                    val prefs = context.getSharedPreferences(prefsKeys.sharedPrefsName, Context.MODE_PRIVATE)
                    prefs.edit().putBoolean(prefsKeys.kioskModeEnabled, true).apply()
                    broadcastManager.sendBroadcast(
                        Intent(intentKeys.kioskControl).putExtra("enabled", true)
                    )
                    CommandResult(
                        status = ACK_ACKED,
                        details = mapOf("kiosk" to true, "mode" to "full")
                    )
                } else {
                    failure("Failed to enable kiosk mode")
                }
            }
            "kiosk.disable" -> {
                if (!kioskLockManager.isDeviceOwner()) {
                    return failure("Device Owner required for kiosk mode")
                }
                kioskLockManager.disableKioskMode()
                val prefs = context.getSharedPreferences(prefsKeys.sharedPrefsName, Context.MODE_PRIVATE)
                prefs.edit().putBoolean(prefsKeys.kioskModeEnabled, false).apply()
                broadcastManager.sendBroadcast(
                    Intent(intentKeys.kioskControl).putExtra("enabled", false)
                )
                CommandResult(
                    status = ACK_ACKED,
                    details = mapOf("kiosk" to false, "mode" to "full")
                )
            }
            else -> failure("Unknown kiosk command: $command")
        }
    }

    // ── Config push ───────────────────────────────────────────────────────────

    private fun handleConfigPushCommand(payload: JSONObject?): CommandResult {
        val config = payload ?: return failure("No config payload")

        val prefs = context.getSharedPreferences(prefsKeys.sharedPrefsName, Context.MODE_PRIVATE)
        val appliedSettings = mutableMapOf<String, Any?>()

        // Volume (0-100). Drives the platform's PowerController; tablet maps to
        // AudioManager.STREAM_MUSIC, TV maps to its native volume control.
        config.optInt("volume", -1).takeIf { it in 0..100 }?.let { level ->
            if (powerAdapter.setVolumePercent(level, showUi = false)) {
                appliedSettings["volume"] = level
            }
        }

        // Kiosk PIN (4 digits).
        config.optString("kioskPin", "").takeIf {
            it.length == 4 && it.all { c -> c.isDigit() }
        }?.let { pin ->
            prefs.edit().putString(prefsKeys.kioskPin, pin).apply()
            appliedSettings["kioskPin"] = "updated"
        }

        // Auto-wake setting — only apply if value is explicitly a Boolean.
        if (config.has("autoWake")) {
            val rawValue = config.opt("autoWake")
            if (rawValue is Boolean) {
                prefs.edit().putBoolean(prefsKeys.autoWakeEnabled, rawValue).apply()
                appliedSettings["autoWake"] = rawValue
            }
        }

        // Rendering mode — gated on the platform exposing both a non-empty
        // supportedRenderingModes() set AND the corresponding intent. Tablet
        // ships the 3-mode set + INTENT_RENDERING_MODE_CHANGED; TV doesn't.
        val supportedModes = supportedRenderingModes()
        val renderingChangedAction = intentKeys.renderingModeChanged
        val renderingModePref = prefsKeys.renderingMode
        if (supportedModes.isNotEmpty() && renderingChangedAction != null && renderingModePref != null) {
            config.optString("rendering_mode", "").takeIf { it.isNotBlank() }?.let { mode ->
                if (mode in supportedModes) {
                    prefs.edit().putString(renderingModePref, mode).apply()
                    broadcastManager.sendBroadcast(Intent(renderingChangedAction))
                    appliedSettings["rendering_mode"] = mode
                    Log.i(TAG, "Rendering mode switched to: $mode")
                }
            }
        }

        return CommandResult(
            status = ACK_ACKED,
            details = mapOf("config" to "applied", "settings" to appliedSettings)
        )
    }

    // ── OTA Update / APK install ──────────────────────────────────────────────

    private suspend fun handleCheckUpdateCommand(): CommandResult {
        val manifest = apiClient.fetchApkManifest(apkManifestKey())
            ?: return failure("Failed to fetch APK manifest")

        val current = currentVersionCode()
        val available = manifest.versionCode
        val updateAvailable = available > current

        return CommandResult(
            status = ACK_ACKED,
            details = mapOf(
                "currentVersion" to currentVersionName(),
                "currentVersionCode" to current,
                "availableVersion" to manifest.version,
                "availableVersionCode" to available,
                "updateAvailable" to updateAvailable,
                "downloadUrl" to manifest.url,
                "changelog" to manifest.changelog
            )
        )
    }

    /**
     * Earner-pull diagnostics bundle. Collects logcat tail (partner-codes
     * scrubbed), sensing pipeline state, network reachability, and last
     * crash trace via ApplicationExitInfo. The bundle returns through the
     * existing command-ack channel (`CommandResult.details`).
     *
     * See `DiagnosticsBundleCollector` kdoc for the security fence
     * (compiled-in redaction list) and bundle shape.
     */
    private fun handleDiagBundleCommand(): CommandResult {
        val bundle = DiagnosticsBundleCollector(context, prefsKeys).collect()
        return CommandResult(
            status = ACK_ACKED,
            details = bundle
        )
    }

    private suspend fun handleInstallApkCommand(payload: JSONObject?): CommandResult {
        if (!kioskLockManager.isDeviceOwner()) {
            return failure("Silent APK install requires Device Owner")
        }

        val url = payload?.optString("url")?.takeIf { it.isNotBlank() }
            ?: return failure("No APK URL provided")

        val apkFile = downloadApkToCache(url)
            ?: return failure("Failed to download APK")

        // Note: don't delete apkFile here — install is async and file is needed
        // until the session commit completes. Cache will be overwritten on
        // next install attempt.
        val installed = installApkSilently(apkFile)

        return if (installed) {
            CommandResult(
                status = ACK_ACKED,
                details = mapOf("installed" to true, "url" to url)
            )
        } else {
            failure("APK installation failed")
        }
    }

    private fun downloadApkToCache(url: String): File? {
        return try {
            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .build()
            val request = Request.Builder().url(url).build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return null
                val apkFile = File(context.cacheDir, "update.apk")
                response.body?.byteStream()?.use { input ->
                    apkFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                apkFile
            }
        } catch (e: Exception) {
            Log.e(TAG, "APK download failed", e)
            null
        }
    }

    private fun installApkSilently(apkFile: File): Boolean {
        return try {
            val packageInstaller = context.packageManager.packageInstaller
            val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL)
            // Don't set app package name — let PackageInstaller detect from APK manifest.

            val sessionId = packageInstaller.createSession(params)
            val session = packageInstaller.openSession(sessionId)

            session.openWrite("package", 0, apkFile.length()).use { output ->
                apkFile.inputStream().use { input ->
                    input.copyTo(output)
                }
                session.fsync(output)
            }

            val intent = Intent().apply {
                component = deviceAdminComponent()
                action = installCompleteAction()
            }
            val pendingIntent = PendingIntent.getBroadcast(
                context,
                sessionId,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            session.commit(pendingIntent.intentSender)
            session.close()
            true
        } catch (e: Exception) {
            Log.e(TAG, "APK install failed", e)
            false
        }
    }

    // ── Audience brand list ───────────────────────────────────────────────────

    private fun handleUpdateBrandListCommand(payload: JSONObject?): CommandResult {
        val brandsArray = payload?.optJSONArray("brands")
            ?: return failure("No brands array in payload")

        val brands = mutableListOf<String>()
        for (i in 0 until brandsArray.length()) {
            brandsArray.optString(i)?.takeIf { it.isNotBlank() }?.let { brands.add(it) }
        }

        Log.i(TAG, "Received dynamic brand list: ${brands.size} brands")
        onBrandListUpdate?.invoke(brands)

        return CommandResult(
            status = ACK_ACKED,
            details = mapOf("brandsReceived" to brands.size)
        )
    }

    // ── Sensing profile ───────────────────────────────────────────────────────

    private suspend fun handleUpdateSensingProfile(payload: JSONObject?): CommandResult {
        val profilePayload = payload
            ?: return failure("No profile payload provided")
        return applySensingProfile(profilePayload)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    protected fun success(details: Map<String, Any?> = emptyMap()): CommandResult =
        CommandResult(status = ACK_ACKED, details = details)

    protected fun failure(error: String): CommandResult =
        CommandResult(status = ACK_FAILED, details = mapOf("error" to error))
}
