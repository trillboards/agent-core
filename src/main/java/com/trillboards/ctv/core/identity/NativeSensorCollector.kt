package com.trillboards.ctv.core.identity

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicReference

/**
 * Lightweight one-shot harvest of two passive Android hardware sensors:
 *  - TYPE_LIGHT (ambient illuminance, lux)
 *  - TYPE_PRESSURE (barometric pressure, hPa)
 *
 * These two values are sampled briefly and shipped on each heartbeat. The
 * server-side `SignalIngestService.normalize()` hoists them onto every
 * emitted BLE / WiFi / mDNS row in CH `signal_observations` (no dedicated
 * `source='native_sensor'` row anymore — see redesign 2026-05-01 + CH
 * migration 059).
 *
 * Capture window defaults to 5 seconds. Both sensors are passive and have
 * trivial CPU/battery cost. Previous incarnations also sampled
 * accelerometer / gyroscope / magnetometer / proximity but those were
 * retired in this PR because:
 *   - footstep peak-finding from a wall-mounted tablet picks up ambient
 *     vibration, not pedestrian gait
 *   - magnetometer flux deltas correlate with vehicles in some venues but
 *     not others (parking lot vs storefront), and the
 *     vehicle-pass-rate proxy was unvalidated sensor fusion
 *   - proximity events on tablets are usually near/far binary and never
 *     reliably classified real audience presence
 *   - gait fingerprints introduced privacy / RTBF complexity for ~zero
 *     downstream value (the storyboard never surfaced them)
 *
 * No additional permissions required — both sensors are freely readable.
 * Sensors that aren't present on the device contribute null fields.
 */
data class NativeSensorSnapshot(
    val ambientLightLux: Float?,
    val barometerPressureHpa: Float?,
    val sensorTimestampMs: Long
)

object NativeSensorCollector {

    private const val TAG = "NativeSensorCollector"
    const val DEFAULT_WINDOW_MS = 5_000L

    /**
     * Collect a single native-sensor snapshot. Suspends for [windowMs]
     * milliseconds while the listeners run, then unregisters.
     *
     * @return a snapshot whose individual fields are nullable when the
     *   underlying hardware is missing or no events fired during the window.
     */
    suspend fun collect(context: Context, windowMs: Long = DEFAULT_WINDOW_MS): NativeSensorSnapshot? = withContext(Dispatchers.IO) {
        try {
            val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
                ?: return@withContext null

            val light = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
            val pressure = sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE)

            if (light == null && pressure == null) {
                Log.d(TAG, "No light/pressure sensors available")
                return@withContext null
            }

            val lightReading = AtomicReference<Float?>(null)
            val pressureReading = AtomicReference<Float?>(null)

            val handler = Handler(Looper.getMainLooper())

            val lightListener = light?.let {
                SensorListener { ev -> if (ev.values.isNotEmpty()) lightReading.set(ev.values[0]) }
                    .also { sensorManager.registerListener(it, light, SensorManager.SENSOR_DELAY_NORMAL, handler) }
            }
            val pressureListener = pressure?.let {
                SensorListener { ev -> if (ev.values.isNotEmpty()) pressureReading.set(ev.values[0]) }
                    .also { sensorManager.registerListener(it, pressure, SensorManager.SENSOR_DELAY_NORMAL, handler) }
            }

            delay(windowMs)

            lightListener?.let { sensorManager.unregisterListener(it) }
            pressureListener?.let { sensorManager.unregisterListener(it) }

            NativeSensorSnapshot(
                ambientLightLux = lightReading.get(),
                barometerPressureHpa = pressureReading.get(),
                sensorTimestampMs = System.currentTimeMillis()
            )
        } catch (e: Exception) {
            Log.w(TAG, "Native sensor collection failed: ${e.message}")
            null
        }
    }

    /**
     * Adapter so we can reuse the same lambda interface for every sensor.
     */
    private class SensorListener(private val onEvent: (SensorEvent) -> Unit) : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent?) { event?.let(onEvent) }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }
}
