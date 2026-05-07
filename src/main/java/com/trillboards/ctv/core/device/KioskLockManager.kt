package com.trillboards.ctv.core.device

import android.app.ActivityManager
import android.app.admin.DevicePolicyManager
import android.content.ComponentName
import android.content.Context
import android.content.IntentFilter
import android.content.Intent
import android.os.Build
import android.os.UserManager
import android.util.Log

/**
 * Manages kiosk lock mode using Android's Device Owner / Lock Task features.
 * Shared across Tablet and Android TV agents.
 *
 * There are three levels of kiosk lock:
 * 1. FULL (Device Owner Mode) - Complete lockdown, requires provisioning during factory reset
 * 2. BASIC (Lock Task Mode) - Pins the app, user can exit with certain gestures
 * 3. NONE - No kiosk lock
 *
 * Device Owner Mode requires one of:
 * - QR code provisioning during initial device setup
 * - NFC provisioning
 * - ADB command: adb shell dpm set-device-owner <package>/.receiver.DeviceAdminReceiver
 *
 * @param context Application context
 * @param adminComponent The ComponentName of the DeviceAdminReceiver for this agent
 */
class KioskLockManager(
    private val context: Context,
    private val adminComponent: ComponentName
) {

    private val devicePolicyManager = context.getSystemService(Context.DEVICE_POLICY_SERVICE) as DevicePolicyManager
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    companion object {
        private const val TAG = "KioskLockManager"
    }

    enum class KioskMode {
        FULL,   // Device Owner - complete lock
        BASIC,  // Lock Task - pinned app
        NONE    // No kiosk mode
    }

    /**
     * Check if this app is set as the Device Owner
     */
    fun isDeviceOwner(): Boolean {
        return devicePolicyManager.isDeviceOwnerApp(context.packageName)
    }

    /**
     * Check if the app is currently in Lock Task mode
     */
    fun isInLockTaskMode(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            activityManager.lockTaskModeState != ActivityManager.LOCK_TASK_MODE_NONE
        } else {
            @Suppress("DEPRECATION")
            activityManager.isInLockTaskMode
        }
    }

    /**
     * Get the current kiosk mode level
     */
    fun getCurrentMode(): KioskMode {
        return when {
            isDeviceOwner() && isInLockTaskMode() -> KioskMode.FULL
            isInLockTaskMode() -> KioskMode.BASIC
            else -> KioskMode.NONE
        }
    }

    /**
     * Enable kiosk mode for the specified activity.
     * Must be called from the activity that should be locked.
     *
     * @return true if kiosk mode was enabled successfully
     */
    fun enableKioskMode(): Boolean {
        if (!isDeviceOwner()) {
            Log.w(TAG, "Cannot enable full kiosk mode - not device owner")
            return false
        }

        return try {
            // Set this package as allowed for lock task
            devicePolicyManager.setLockTaskPackages(adminComponent, arrayOf(context.packageName))

            // Configure lock task features (hide status bar, navigation, etc.)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                devicePolicyManager.setLockTaskFeatures(
                    adminComponent,
                    DevicePolicyManager.LOCK_TASK_FEATURE_NONE  // Most restrictive
                )
            }

            Log.i(TAG, "Kiosk mode enabled successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enable kiosk mode", e)
            false
        }
    }

    /**
     * Disable kiosk mode.
     * Only works if we're the device owner.
     */
    fun disableKioskMode(): Boolean {
        if (!isDeviceOwner()) {
            Log.w(TAG, "Cannot disable kiosk mode - not device owner")
            return false
        }

        return try {
            // Clear lock task packages
            devicePolicyManager.setLockTaskPackages(adminComponent, emptyArray())
            Log.i(TAG, "Kiosk mode disabled")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to disable kiosk mode", e)
            false
        }
    }

    /**
     * Configure kiosk mode features (what's visible/hidden in lock task mode).
     * Only available on Android 9 (API 28) and above.
     *
     * @param showHomeButton Show home button in navigation
     * @param showNotifications Allow notifications
     * @param showSystemInfo Show clock, battery, etc.
     */
    fun configureKioskFeatures(
        showHomeButton: Boolean = false,
        showNotifications: Boolean = false,
        showSystemInfo: Boolean = false
    ): Boolean {
        if (!isDeviceOwner()) return false
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return false

        return try {
            var features = DevicePolicyManager.LOCK_TASK_FEATURE_NONE

            if (showHomeButton) {
                features = features or DevicePolicyManager.LOCK_TASK_FEATURE_HOME
            }
            if (showNotifications) {
                features = features or DevicePolicyManager.LOCK_TASK_FEATURE_NOTIFICATIONS
            }
            if (showSystemInfo) {
                features = features or DevicePolicyManager.LOCK_TASK_FEATURE_SYSTEM_INFO
            }

            devicePolicyManager.setLockTaskFeatures(adminComponent, features)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure kiosk features", e)
            false
        }
    }

    /**
     * Get instructions for setting up Device Owner mode
     *
     * @param packageName The package name of the agent
     */
    fun getSetupInstructions(packageName: String = context.packageName): String {
        return """
            To enable full kiosk mode, you need to set this app as Device Owner.

            Option 1: ADB Command (for development)
            1. Enable Developer Options and USB Debugging on the device
            2. Connect via USB and run:
               adb shell dpm set-device-owner $packageName/.receiver.DeviceAdminReceiver

            Option 2: QR Code Provisioning (for production)
            1. Factory reset the device
            2. On the initial "Hi there" screen, tap the screen 6 times quickly
            3. Scan the provisioning QR code

            Option 3: NFC Provisioning
            1. Factory reset the device
            2. Tap an NFC tag with provisioning data during setup

            Note: Device Owner can only be set on a device with no existing accounts.
            You may need to remove all Google accounts first.
        """.trimIndent()
    }

    /**
     * Generate provisioning QR code data for device setup.
     * This QR code should be scanned during initial device setup.
     *
     * @param packageName The package name of the agent
     * @param apkDownloadUrl URL to download the APK
     */
    fun getProvisioningQrData(
        packageName: String = context.packageName,
        apkDownloadUrl: String
    ): String {
        return """
            {
                "android.app.extra.PROVISIONING_DEVICE_ADMIN_COMPONENT_NAME":
                    "$packageName/.receiver.DeviceAdminReceiver",
                "android.app.extra.PROVISIONING_DEVICE_ADMIN_PACKAGE_DOWNLOAD_LOCATION":
                    "$apkDownloadUrl",
                "android.app.extra.PROVISIONING_SKIP_ENCRYPTION": true,
                "android.app.extra.PROVISIONING_LEAVE_ALL_SYSTEM_APPS_ENABLED": true
            }
        """.trimIndent()
    }

    /**
     * Apply additional Device Owner policies for 24/7 reliability.
     * Only effective when the app is Device Owner.
     *
     * Policies applied:
     * - STAY_ON_WHILE_PLUGGED_IN (AC + USB + wireless)
     * - DISALLOW_SAFE_BOOT (prevent accidental safe mode)
     * - DISALLOW_FACTORY_RESET (prevent accidental reset)
     */
    fun configureDeviceOwnerPolicies(): Boolean {
        if (!isDeviceOwner()) {
            Log.w(TAG, "Cannot configure Device Owner policies - not device owner")
            return false
        }

        return try {
            // Keep screen on while plugged in (AC, USB, or wireless)
            @Suppress("DEPRECATION")
            devicePolicyManager.setGlobalSetting(
                adminComponent,
                android.provider.Settings.Global.STAY_ON_WHILE_PLUGGED_IN,
                "${android.os.BatteryManager.BATTERY_PLUGGED_AC or android.os.BatteryManager.BATTERY_PLUGGED_USB or android.os.BatteryManager.BATTERY_PLUGGED_WIRELESS}"
            )
            Log.i(TAG, "Set STAY_ON_WHILE_PLUGGED_IN")

            // Prevent safe boot (API 23+)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                devicePolicyManager.addUserRestriction(adminComponent, UserManager.DISALLOW_SAFE_BOOT)
                Log.i(TAG, "Set DISALLOW_SAFE_BOOT")
            }

            // Prevent factory reset
            devicePolicyManager.addUserRestriction(adminComponent, UserManager.DISALLOW_FACTORY_RESET)
            Log.i(TAG, "Set DISALLOW_FACTORY_RESET")

            // Set this app as the persistent default HOME launcher
            // Without this, Fire TV home screen takes focus after every reboot
            try {
                val homeFilter = IntentFilter(Intent.ACTION_MAIN).apply {
                    addCategory(Intent.CATEGORY_HOME)
                    addCategory(Intent.CATEGORY_DEFAULT)
                }
                devicePolicyManager.addPersistentPreferredActivity(
                    adminComponent,
                    homeFilter,
                    ComponentName(context.packageName, "${context.packageName}.MainActivity")
                )
                Log.i(TAG, "Set as persistent default HOME launcher")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to set persistent preferred activity", e)
            }

            // Disable keyguard so lock screen doesn't appear between boot and app
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                try {
                    devicePolicyManager.setKeyguardDisabled(adminComponent, true)
                    Log.i(TAG, "Keyguard disabled")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to disable keyguard", e)
                }
            }

            // Disable screensaver to prevent it activating over the app
            try {
                devicePolicyManager.setSecureSetting(
                    adminComponent, "screensaver_enabled", "0"
                )
                Log.i(TAG, "Screensaver disabled")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to disable screensaver", e)
            }

            Log.i(TAG, "Device Owner policies configured successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure Device Owner policies", e)
            false
        }
    }
}
