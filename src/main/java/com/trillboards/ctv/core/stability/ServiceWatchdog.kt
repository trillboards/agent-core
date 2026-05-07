package com.trillboards.ctv.core.stability

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.File

/**
 * AlarmManager-based watchdog that checks if a foreground service is alive
 * every 15 minutes using file-based heartbeat detection.
 *
 * The previous implementation used getRunningServices() which returns empty on
 * API 26+ (our minSdk). This version uses a heartbeat file: the foreground service
 * writes a timestamp every 30s, and the watchdog checks if that timestamp is stale.
 *
 * Usage:
 * 1. Register WatchdogReceiver in AndroidManifest.xml
 * 2. Call ServiceWatchdog.schedule() from BootCompletedReceiver and MainActivity.onCreate()
 * 3. Call ServiceWatchdog.writeHeartbeat() from your foreground service every 30s
 */
object ServiceWatchdog {

    private const val TAG = "ServiceWatchdog"
    private const val INTERVAL_MS = 15 * 60 * 1000L // 15 minutes
    private const val REQUEST_CODE = 9901

    /** How often the foreground service should call writeHeartbeat() */
    const val HEARTBEAT_INTERVAL_MS = 30_000L // 30 seconds

    /** If heartbeat file is older than this, service is considered dead */
    private const val HEARTBEAT_STALE_THRESHOLD_MS = 2 * 60 * 1000L // 2 minutes

    private const val HEARTBEAT_DIR = "watchdog"

    /**
     * Write a heartbeat file for the given service. Call this every 30s from
     * your foreground service's main loop or a scheduled handler.
     */
    fun writeHeartbeat(context: Context, serviceClassName: String) {
        try {
            val dir = File(context.filesDir, HEARTBEAT_DIR)
            if (!dir.exists()) dir.mkdirs()
            val file = heartbeatFile(context, serviceClassName)
            file.writeText(System.currentTimeMillis().toString())
        } catch (e: Exception) {
            Log.w(TAG, "Failed to write heartbeat for $serviceClassName: ${e.message}")
        }
    }

    /**
     * Check if a service is alive by reading its heartbeat file.
     * Returns true if the heartbeat file exists and was written within the stale threshold.
     */
    fun isServiceAlive(context: Context, serviceClassName: String): Boolean {
        return try {
            val file = heartbeatFile(context, serviceClassName)
            if (!file.exists()) {
                Log.d(TAG, "No heartbeat file for $serviceClassName — service never started or file cleared")
                return false
            }
            val lastBeat = file.readText().trim().toLongOrNull() ?: return false
            val age = System.currentTimeMillis() - lastBeat
            val alive = age < HEARTBEAT_STALE_THRESHOLD_MS
            if (!alive) {
                Log.w(TAG, "Heartbeat stale for $serviceClassName: ${age / 1000}s old (threshold: ${HEARTBEAT_STALE_THRESHOLD_MS / 1000}s)")
            }
            alive
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read heartbeat for $serviceClassName: ${e.message}")
            false
        }
    }

    /**
     * Clear the heartbeat file when a service is intentionally stopped.
     */
    fun clearHeartbeat(context: Context, serviceClassName: String) {
        try {
            val file = heartbeatFile(context, serviceClassName)
            if (file.exists()) file.delete()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to clear heartbeat for $serviceClassName: ${e.message}")
        }
    }

    private fun heartbeatFile(context: Context, serviceClassName: String): File {
        val dir = File(context.filesDir, HEARTBEAT_DIR)
        val safeName = serviceClassName.replace('.', '_')
        return File(dir, "heartbeat_$safeName")
    }

    /**
     * Schedule repeating watchdog alarm.
     *
     * @param context Application context
     * @param watchdogAction The intent action that WatchdogReceiver listens for
     */
    fun schedule(context: Context, watchdogAction: String) {
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return
        val intent = Intent(watchdogAction).setPackage(context.packageName)
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            REQUEST_CODE,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val triggerAt = SystemClock.elapsedRealtime() + INTERVAL_MS

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                // API 31+: check if exact alarms are permitted
                if (alarmManager.canScheduleExactAlarms()) {
                    alarmManager.setExactAndAllowWhileIdle(
                        AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pendingIntent
                    )
                } else {
                    // Fall back to inexact alarm — still fires within the window
                    alarmManager.setAndAllowWhileIdle(
                        AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pendingIntent
                    )
                    Log.d(TAG, "Exact alarm not permitted, using inexact alarm")
                }
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                alarmManager.setExactAndAllowWhileIdle(
                    AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pendingIntent
                )
            } else {
                alarmManager.setExact(
                    AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pendingIntent
                )
            }
            Log.d(TAG, "Watchdog alarm scheduled (${INTERVAL_MS / 60000}min)")
        } catch (e: SecurityException) {
            // Fallback: use inexact alarm if exact alarm permission denied
            Log.w(TAG, "Exact alarm denied, falling back to inexact: ${e.message}")
            alarmManager.set(
                AlarmManager.ELAPSED_REALTIME_WAKEUP, triggerAt, pendingIntent
            )
        }
    }

    /**
     * Cancel the watchdog alarm.
     */
    fun cancel(context: Context, watchdogAction: String) {
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return
        val intent = Intent(watchdogAction).setPackage(context.packageName)
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            REQUEST_CODE,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_NO_CREATE
        )
        if (pendingIntent != null) {
            alarmManager.cancel(pendingIntent)
            Log.d(TAG, "Watchdog alarm cancelled")
        }
    }

    /**
     * Abstract receiver to be extended by each agent.
     * Subclasses provide the service class and watchdog action.
     */
    abstract class BaseWatchdogReceiver : BroadcastReceiver() {

        abstract val serviceClass: Class<*>
        abstract val watchdogAction: String

        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != watchdogAction) return

            if (!isServiceAlive(context, serviceClass.name)) {
                Log.w(TAG, "Service ${serviceClass.simpleName} heartbeat stale or missing! Restarting...")
                val serviceIntent = Intent(context, serviceClass)
                ContextCompat.startForegroundService(context, serviceIntent)
            } else {
                Log.d(TAG, "Service ${serviceClass.simpleName} heartbeat fresh - OK")
            }

            // Reschedule since setExactAndAllowWhileIdle is one-shot
            schedule(context, watchdogAction)
        }
    }
}
