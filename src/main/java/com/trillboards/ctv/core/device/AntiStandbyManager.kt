package com.trillboards.ctv.core.device

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.util.Log

/**
 * Manages anti-standby features to keep the device awake and prevent sleep.
 * Used for enterprise/kiosk deployments where the screen should always be active.
 *
 * Features:
 * - Wake locks to prevent device sleep
 * - Screen-off interception with auto-wake
 * - Periodic wake monitoring
 */
class AntiStandbyManager(private val context: Context) {

    companion object {
        private const val TAG = "AntiStandbyManager"
        private const val WAKE_TAG = "Trillboards:AntiStandby"
        private const val AUTO_WAKE_DELAY_MS = 1000L
    }

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private val handler = Handler(Looper.getMainLooper())

    private var partialWakeLock: PowerManager.WakeLock? = null
    private var screenOffReceiver: BroadcastReceiver? = null
    private var autoWakeEnabled = false

    /**
     * Callback invoked when screen turns off and auto-wake is triggered
     */
    var onScreenOffDetected: (() -> Unit)? = null

    /**
     * Acquire a partial wake lock to keep the CPU running.
     * This prevents the device from entering deep sleep.
     */
    fun acquireWakeLock() {
        if (partialWakeLock?.isHeld == true) {
            Log.d(TAG, "Wake lock already held")
            return
        }

        partialWakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$WAKE_TAG:Service"
        ).apply {
            acquire()
        }
        Log.i(TAG, "Partial wake lock acquired")
    }

    /**
     * Release the partial wake lock
     */
    fun releaseWakeLock() {
        partialWakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.i(TAG, "Partial wake lock released")
            }
        }
        partialWakeLock = null
    }

    /**
     * Enable auto-wake functionality.
     * When the screen turns off, we'll automatically wake it back up.
     */
    fun enableAutoWake() {
        if (autoWakeEnabled) {
            Log.d(TAG, "Auto-wake already enabled")
            return
        }

        screenOffReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                when (intent.action) {
                    Intent.ACTION_SCREEN_OFF -> {
                        Log.i(TAG, "Screen off detected, scheduling wake")
                        onScreenOffDetected?.invoke()
                        handler.postDelayed({
                            wakeScreen()
                        }, AUTO_WAKE_DELAY_MS)
                    }
                }
            }
        }

        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(screenOffReceiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            context.registerReceiver(screenOffReceiver, filter)
        }

        autoWakeEnabled = true
        Log.i(TAG, "Auto-wake enabled")
    }

    /**
     * Disable auto-wake functionality
     */
    fun disableAutoWake() {
        screenOffReceiver?.let {
            try {
                context.unregisterReceiver(it)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to unregister screen off receiver", e)
            }
        }
        screenOffReceiver = null
        autoWakeEnabled = false
        Log.i(TAG, "Auto-wake disabled")
    }

    /**
     * Wake up the screen immediately
     */
    fun wakeScreen() {
        try {
            @Suppress("DEPRECATION")
            val wakeLock = powerManager.newWakeLock(
                PowerManager.SCREEN_BRIGHT_WAKE_LOCK or
                    PowerManager.ACQUIRE_CAUSES_WAKEUP or
                    PowerManager.ON_AFTER_RELEASE,
                "$WAKE_TAG:WakeUp"
            )
            wakeLock.acquire(10_000L) // Hold for 10 seconds
            wakeLock.release()
            Log.i(TAG, "Screen wake triggered")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to wake screen", e)
        }
    }

    /**
     * Check if the screen is currently on
     */
    fun isScreenOn(): Boolean {
        return powerManager.isInteractive
    }

    /**
     * Check if we should request battery optimization exemption.
     * Returns true if the app is NOT already exempt.
     */
    fun shouldRequestBatteryOptimizationExemption(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            !powerManager.isIgnoringBatteryOptimizations(context.packageName)
        } else {
            false // Not applicable on older Android versions
        }
    }

    /**
     * Get the intent to request battery optimization exemption.
     * The calling activity should start this intent.
     */
    fun getBatteryOptimizationExemptionIntent(): Intent? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            return null
        }

        return Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
            data = android.net.Uri.parse("package:${context.packageName}")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
    }

    /**
     * Clean up all resources
     */
    fun cleanup() {
        disableAutoWake()
        releaseWakeLock()
        handler.removeCallbacksAndMessages(null)
        Log.i(TAG, "AntiStandbyManager cleaned up")
    }
}
