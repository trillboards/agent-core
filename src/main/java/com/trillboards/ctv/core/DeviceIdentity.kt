package com.trillboards.ctv.core

import android.content.Context
import android.os.Build
import android.provider.Settings
import android.util.Log
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withTimeoutOrNull
import java.security.MessageDigest
import java.util.UUID

object DeviceIdentity {
    private const val DEFAULT_PREFS = "trillboard_device_identity"
    private const val FINGERPRINT_VERSION = 2
    private const val FINGERPRINT_VERSION_KEY = "fingerprint_version"

    /**
     * Generates or retrieves a persistent device fingerprint.
     *
     * v1: SHA-256(androidId|brand|model|SDK_INT) — SDK_INT caused fingerprint
     *     changes on OS updates, creating orphan devices in the backend.
     * v2: SHA-256(androidId|brand|model) — stable across OS updates.
     *
     * When the stored version is older than FINGERPRINT_VERSION, the fingerprint
     * is recomputed and the new value is persisted.
     */
    fun fingerprint(context: Context, prefsName: String = DEFAULT_PREFS): String {
        val prefs = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        val version = prefs.getInt(FINGERPRINT_VERSION_KEY, 0)
        val cached = prefs.getString("fingerprint", null)

        // Return cached if version is current
        if (!cached.isNullOrEmpty() && version >= FINGERPRINT_VERSION) return cached

        val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
        val payload = listOf(androidId, Build.BRAND, Build.MODEL)
            .joinToString(separator = "|")
        val digest = MessageDigest.getInstance("SHA-256").digest(payload.toByteArray())
        val fingerprint = digest.joinToString(separator = "") { b -> "%02x".format(b) }
        prefs.edit()
            .putString("fingerprint", fingerprint)
            .putInt(FINGERPRINT_VERSION_KEY, FINGERPRINT_VERSION)
            .apply()
        return fingerprint
    }

    /**
     * Returns a stable per-install identifier that survives app restarts but
     * not reinstalls. Format: "DEV-" + 8 hex chars from a UUID.
     *
     * Used by the backend to distinguish multiple installs on the same hardware
     * (same fingerprint) — e.g., after a factory reset where ANDROID_ID resets too.
     */
    fun installId(context: Context, prefsName: String = DEFAULT_PREFS): String {
        val prefs = context.getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        val cached = prefs.getString("install_id", null)
        if (!cached.isNullOrEmpty()) return cached

        val id = "DEV-" + UUID.randomUUID().toString().replace("-", "").take(8)
        prefs.edit().putString("install_id", id).apply()
        return id
    }

    /**
     * Returns the install-aware fingerprint used by the backend identity
     * resolver: `${baseFingerprint}_${installId}`.
     */
    fun stableFingerprint(context: Context, prefsName: String = DEFAULT_PREFS): String {
        return "${fingerprint(context, prefsName)}_${installId(context, prefsName)}"
    }

    /**
     * Retrieves the Firebase Installation ID (FID) for cross-install identity.
     * FID survives app reinstalls on the same device (tied to Firebase project +
     * device hardware), making it useful for detecting reinstalls.
     *
     * Returns null if Firebase is not configured or the call times out (5s).
     */
    suspend fun getFirebaseInstallationId(): String? {
        return try {
            withTimeoutOrNull(5000L) {
                com.google.firebase.installations.FirebaseInstallations.getInstance()
                    .id.await()
            }
        } catch (e: Exception) {
            Log.w("DeviceIdentity", "FID unavailable: ${e.message}")
            null
        }
    }

    /**
     * Generates a new ephemeral session ID (UUID v4).
     * This is not persisted and changes each app launch.
     */
    fun ephemeralSessionId(): String = UUID.randomUUID().toString()

    /**
     * Alias for ephemeralSessionId for backward compatibility.
     */
    fun sessionId(): String = ephemeralSessionId()
}
