package com.trillboards.ctv.core.diagnostics

import android.Manifest
import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import androidx.core.content.ContextCompat
import com.trillboards.ctv.core.service.PrefsKeys
import java.net.NetworkInterface

/**
 * Earner-pull diagnostics bundle assembler.
 *
 * Collects a redacted, per-screen health bundle that operators can use
 * to triage black-screen / kiosk-policy issues (e.g. RockChip Gocast STBs)
 * without shipping a debug APK. The bundle is best-effort — every field
 * carries a fallback ("0", "[REDACTED]", "unknown", etc.) so a missing
 * sensor or revoked permission never blocks the pull.
 *
 * Security fence: partner-code redaction is compiled into this class
 * (NOT JSON config, NOT env-driven) so a runtime-flag flip cannot
 * disable it. The list of partner names lives only here (and the
 * agent-core-lite mirror) to avoid a parallel taxonomy. Every string
 * extracted from logcat / crash trace passes through `redactPartnerCodes`
 * before leaving the device.
 *
 * Bundle shape (schemaVersion=1):
 * - buildInfo: {versionCode, versionName, packageName}
 * - logcatTail: last ~500 lines, partner-codes scrubbed
 * - sensingPipelineStatus: {moonshineLoaded, lastFaceDetectAtMs,
 *     lastAudienceEmitAtMs, cameraPermissionGranted, audioPermissionGranted}
 * - networkReachability: {lastSuccessfulHeartbeatAtMs, wifiConnected, ipAddress}
 * - lastCrash: {type, timestampMs, summary} via ApplicationExitInfo (API 30+)
 *
 * Mirrored to `agent-core-lite/src/main/java/.../diagnostics/DiagnosticsBundleCollector.kt`
 * per `.claude/lessons.md` row 69 (every new agent-core type referenced from
 * shared `app/src/main/` MUST mirror to agent-core-lite).
 */
class DiagnosticsBundleCollector(
    private val context: Context,
    private val prefsKeys: PrefsKeys
) {
    companion object {
        private const val SCHEMA_VERSION = 1

        // Logcat tail size — 500 lines is enough to catch a recent
        // crash window without exceeding the command-ack envelope.
        private const val LOGCAT_LINE_COUNT = 500

        // Partner-code redaction patterns. Order matters: longest-match-first
        // so a single token doesn't get partial-redacted by a narrower pattern
        // before the wider one runs. Subdomains contain bare partner names,
        // so we strip the full subdomain first; only then do we run the
        // narrower forms (variant aliases + bare names) to catch leftovers.
        // List is compiled-in by design — see class kdoc.
        private val PARTNER_CODE_PATTERNS = listOf(
            // Partner-namespaced subdomains — must run BEFORE the bare-name
            // regex below or `ssp.adipolo.com` would become `ssp.[REDACTED].com`.
            Regex(
                """\b(?:ssp|dsp)\.[a-z0-9-]+\.(?:com|net|io)\b""",
                RegexOption.IGNORE_CASE
            ),
            // v1234, v9999, etc. — variant code aliases
            Regex("""\bv\d{4}\b"""),
            // variant_<id> — alternate alias form
            Regex("""\bvariant_[a-z0-9]+\b""", RegexOption.IGNORE_CASE),
            // Bare partner names, case-insensitive
            Regex(
                """\b(adipolo|vidverto|justbaat|adtelligent|bctv|take10|nrs)\b""",
                RegexOption.IGNORE_CASE
            )
        )

        private const val REDACT_TOKEN = "[REDACTED]"

        // Visible-for-testing prefs keys. These are read-only — the
        // collector does not write them. Sensing services populate
        // them at runtime; missing keys default to 0L.
        const val PREF_LAST_FACE_DETECT_AT_MS = "diag_last_face_detect_at_ms"
        const val PREF_LAST_AUDIENCE_EMIT_AT_MS = "diag_last_audience_emit_at_ms"
        const val PREF_LAST_HEARTBEAT_AT_MS = "diag_last_heartbeat_at_ms"
        const val PREF_MOONSHINE_LOADED = "diag_moonshine_loaded"

        /**
         * Apply the compiled-in redaction patterns to an arbitrary string.
         * Pure function — visible for unit tests. Every code path in this
         * class that emits a string into the bundle MUST pass it through here.
         */
        fun redactPartnerCodes(input: String): String {
            var out = input
            for (pattern in PARTNER_CODE_PATTERNS) {
                out = pattern.replace(out, REDACT_TOKEN)
            }
            return out
        }
    }

    private val sharedPrefs: SharedPreferences by lazy {
        context.getSharedPreferences(prefsKeys.sharedPrefsName, Context.MODE_PRIVATE)
    }

    /**
     * Assemble the bundle. Returns a Map suitable for direct inclusion in
     * `CommandResult.details` — the JSON marshaller in the command-ack
     * pipeline will serialize it. Each section is built independently so
     * a single sub-collector failure (e.g. logcat denied) does not abort
     * the whole bundle.
     */
    fun collect(): Map<String, Any?> {
        return mapOf(
            "schemaVersion" to SCHEMA_VERSION,
            "collectedAtMs" to System.currentTimeMillis(),
            "buildInfo" to buildInfo(),
            "logcatTail" to safeCall("logcat", { captureLogcatTail() }, "[logcat unavailable]"),
            "sensingPipelineStatus" to safeCall(
                "sensingPipelineStatus",
                { sensingPipelineStatus() },
                emptyMap<String, Any?>()
            ),
            "networkReachability" to safeCall(
                "networkReachability",
                { networkReachability() },
                emptyMap<String, Any?>()
            ),
            "lastCrash" to safeCall(
                "lastCrash",
                { getLastCrash() },
                mapOf("type" to "unavailable")
            )
        )
    }

    private fun <T> safeCall(label: String, fn: () -> T, fallback: T): T {
        return try {
            fn()
        } catch (t: Throwable) {
            android.util.Log.w(
                "DiagBundle",
                "Sub-collector failed: $label (${t.javaClass.simpleName})"
            )
            fallback
        }
    }

    private fun buildInfo(): Map<String, Any?> {
        val pkg = context.packageName
        var versionCode = -1L
        var versionName = "unknown"
        try {
            val info = context.packageManager.getPackageInfo(pkg, 0)
            versionName = info.versionName ?: "unknown"
            versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                info.longVersionCode
            } else {
                @Suppress("DEPRECATION")
                info.versionCode.toLong()
            }
        } catch (t: Throwable) {
            android.util.Log.w("DiagBundle", "buildInfo failed: ${t.javaClass.simpleName}")
        }
        return mapOf(
            "versionCode" to versionCode,
            "versionName" to versionName,
            "packageName" to pkg
        )
    }

    private fun captureLogcatTail(lineCount: Int = LOGCAT_LINE_COUNT): String {
        return try {
            val process = Runtime.getRuntime().exec(
                arrayOf("logcat", "-d", "-t", lineCount.toString())
            )
            val raw = process.inputStream.bufferedReader().use { it.readText() }
            process.waitFor()
            redactPartnerCodes(raw)
        } catch (e: Exception) {
            "[logcat unavailable: ${e.javaClass.simpleName}]"
        }
    }

    private fun sensingPipelineStatus(): Map<String, Any?> {
        val cameraGranted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        val audioGranted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        return mapOf(
            "moonshineLoaded" to sharedPrefs.getBoolean(PREF_MOONSHINE_LOADED, false),
            "lastFaceDetectAtMs" to sharedPrefs.getLong(PREF_LAST_FACE_DETECT_AT_MS, 0L),
            "lastAudienceEmitAtMs" to sharedPrefs.getLong(PREF_LAST_AUDIENCE_EMIT_AT_MS, 0L),
            "cameraPermissionGranted" to cameraGranted,
            "audioPermissionGranted" to audioGranted
        )
    }

    private fun networkReachability(): Map<String, Any?> {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
        var wifiConnected = false
        if (cm != null) {
            wifiConnected = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                val active = cm.activeNetwork
                val caps = active?.let { cm.getNetworkCapabilities(it) }
                caps?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
            } else {
                @Suppress("DEPRECATION")
                cm.activeNetworkInfo?.let { it.isConnected && it.type == ConnectivityManager.TYPE_WIFI } == true
            }
        }

        return mapOf(
            "lastSuccessfulHeartbeatAtMs" to sharedPrefs.getLong(PREF_LAST_HEARTBEAT_AT_MS, 0L),
            "wifiConnected" to wifiConnected,
            "ipAddress" to redactedLocalIp()
        )
    }

    /**
     * Local-IP last octet redacted ("192.168.1.[REDACTED]"). Avoids leaking
     * the device's full LAN address through the bundle while preserving
     * subnet info for debugging "is this device even on the right network".
     */
    private fun redactedLocalIp(): String {
        return try {
            val interfaces = NetworkInterface.getNetworkInterfaces() ?: return "unknown"
            val addrs = interfaces.toList()
                .filter { it.isUp && !it.isLoopback }
                .flatMap { it.inetAddresses.toList() }
                .filter { !it.isLoopbackAddress && it.hostAddress != null }
            val ipv4 = addrs.firstOrNull { it.hostAddress?.contains(":") == false }
            val raw = ipv4?.hostAddress ?: addrs.firstOrNull()?.hostAddress ?: "unknown"
            // For IPv4 dotted-quad, redact last octet. For everything else,
            // pass through the raw form (IPv6 is already opaque enough).
            val dotted = Regex("""^(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}$""").find(raw)
            if (dotted != null) "${dotted.groupValues[1]}.[REDACTED]" else raw
        } catch (t: Throwable) {
            "unknown"
        }
    }

    private fun getLastCrash(): Map<String, Any?> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            return mapOf("type" to "unsupported_api_level")
        }
        return getLastCrashApi30()
    }

    @androidx.annotation.RequiresApi(Build.VERSION_CODES.R)
    private fun getLastCrashApi30(): Map<String, Any?> {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            ?: return mapOf("type" to "unavailable")
        val exitInfos = try {
            am.getHistoricalProcessExitReasons(context.packageName, 0, 5)
        } catch (t: Throwable) {
            return mapOf("type" to "unavailable", "error" to t.javaClass.simpleName)
        }
        val crash = exitInfos.firstOrNull {
            it.reason == ApplicationExitInfo.REASON_CRASH ||
                it.reason == ApplicationExitInfo.REASON_CRASH_NATIVE ||
                it.reason == ApplicationExitInfo.REASON_ANR
        } ?: return mapOf("type" to "none")
        val typeStr = when (crash.reason) {
            ApplicationExitInfo.REASON_CRASH -> "java"
            ApplicationExitInfo.REASON_CRASH_NATIVE -> "native"
            ApplicationExitInfo.REASON_ANR -> "anr"
            else -> "other"
        }
        return mapOf(
            "type" to typeStr,
            "timestampMs" to crash.timestamp,
            "summary" to redactPartnerCodes(crash.description ?: "")
        )
    }
}
