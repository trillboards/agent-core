package com.trillboards.ctv.core.audience

import android.content.Context
import android.hardware.Sensor
import com.trillboards.ctv.core.SensingConfig
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.display.DisplayManager
import android.os.Build
import android.util.Log
import android.view.Display
import android.view.Surface
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.ln

/**
 * Collects environmental sensor data for audience measurement.
 * - Ambient light (viewability verification + shadow event detection)
 * - Accelerometer (orientation, tampering detection)
 * - Temperature (if available)
 *
 * Shadow events: Rapid lux drops indicate a person walking past the screen.
 * These are counted per aggregation window and reset by the caller.
 */
class SensorCollector(private val context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val displayManager = context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager

    private var lightSensor: Sensor? = null
    private var accelerometer: Sensor? = null
    private var temperatureSensor: Sensor? = null

    @Volatile private var currentLightLux: Float = -1f
    @Volatile private var currentTemperature: Float = -1f
    @Volatile private var accelerometerValues: FloatArray = floatArrayOf(0f, 0f, 0f)
    @Volatile private var lastAccelerometerUpdate: Long = 0
    private val tamperingDetected = AtomicBoolean(false)

    // Shadow event detection state
    private var previousLux: Float = -1f
    private var shadowEventCount = AtomicInteger(0)
    private var lastShadowEventMs: Long = 0
    private var luxBaseline: Float = -1f
    private var luxBaselineSamples: Int = 0

    // Shadow/tampering config from SensingConfig
    private val shadowCfg = SensingConfig.get().shadow

    // Threshold for detecting movement/tampering (m/s²)
    private val tamperingThreshold = shadowCfg.tamperingThreshold
    private var lastAccelMagnitude = 0f

    companion object {
        private const val TAG = "SensorCollector"
        private const val SENSOR_DELAY = SensorManager.SENSOR_DELAY_NORMAL
    }

    /**
     * Check if light sensor is available.
     */
    fun hasLightSensor(): Boolean {
        return sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT) != null
    }

    /**
     * Check if accelerometer is available.
     */
    fun hasAccelerometer(): Boolean {
        return sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
    }

    /**
     * Start collecting sensor data.
     */
    fun start() {
        Log.d(TAG, "[Sensors] Starting sensor collection...")

        lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)?.also {
            sensorManager.registerListener(this, it, SENSOR_DELAY)
            Log.d(TAG, "[Sensors] Light sensor registered: ${it.name}")
        }
        if (lightSensor == null) Log.d(TAG, "[Sensors] Light sensor not available")

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.also {
            sensorManager.registerListener(this, it, SENSOR_DELAY)
            Log.d(TAG, "[Sensors] Accelerometer registered: ${it.name}")
        }
        if (accelerometer == null) Log.d(TAG, "[Sensors] Accelerometer not available")

        temperatureSensor = sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE)?.also {
            sensorManager.registerListener(this, it, SENSOR_DELAY)
            Log.d(TAG, "[Sensors] Temperature sensor registered: ${it.name}")
        }
        if (temperatureSensor == null) Log.d(TAG, "[Sensors] Temperature sensor not available")

        Log.i(TAG, "[Sensors] Sensor collection started - light=${lightSensor != null}, " +
                "accel=${accelerometer != null}, temp=${temperatureSensor != null}")
    }

    /**
     * Stop collecting sensor data.
     */
    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LIGHT -> {
                val newLux = event.values[0]
                detectShadowEvent(newLux)
                currentLightLux = newLux
            }
            Sensor.TYPE_ACCELEROMETER -> {
                accelerometerValues = event.values.copyOf()
                lastAccelerometerUpdate = System.currentTimeMillis()
                checkForTampering(event.values)
            }
            Sensor.TYPE_AMBIENT_TEMPERATURE -> {
                currentTemperature = event.values[0]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // Not used
    }

    /**
     * Detect shadow events from rapid lux drops.
     * A shadow event occurs when ambient light drops significantly from the
     * rolling baseline, indicating a person walked between the light source
     * and the sensor (i.e., in front of the screen).
     */
    private fun detectShadowEvent(newLux: Float) {
        val now = System.currentTimeMillis()

        // Update rolling baseline (slow-moving exponential average)
        if (luxBaseline < 0) {
            luxBaseline = newLux
            luxBaselineSamples = 1
        } else {
            luxBaselineSamples++
            // Only update baseline with non-shadow values (lux near or above baseline)
            if (newLux >= luxBaseline * shadowCfg.baselineNonShadowRatio) {
                luxBaseline = luxBaseline * (1 - shadowCfg.baselineUpdateAlpha) + newLux * shadowCfg.baselineUpdateAlpha
            }
        }

        // Only detect shadows once baseline is established
        if (luxBaselineSamples < shadowCfg.baselineMinSamples) return
        if (luxBaseline <= 0) return

        // Check for rapid lux drop from baseline
        val dropFromBaseline = luxBaseline - newLux
        val dropRatio = dropFromBaseline / luxBaseline

        if (dropRatio >= shadowCfg.luxDropRatio &&
            dropFromBaseline >= shadowCfg.minLuxDrop &&
            now - lastShadowEventMs > shadowCfg.cooldownMs) {

            shadowEventCount.incrementAndGet()
            lastShadowEventMs = now
            Log.d(TAG, "[Shadow] Shadow event detected: lux ${luxBaseline.toInt()} → ${newLux.toInt()} " +
                    "(drop=${dropRatio * 100}%, total=${shadowEventCount.get()})")
        }
    }

    /**
     * Get and reset shadow event count for the current aggregation window.
     * Called by AudienceSensingService at the end of each window.
     */
    fun getAndResetShadowEvents(): Int {
        return shadowEventCount.getAndSet(0)
    }

    /**
     * Get current shadow event count without resetting.
     */
    fun getShadowEventCount(): Int = shadowEventCount.get()

    /**
     * Get normalized ambient light value (0-1, log scale).
     * Useful for the audience vector dimension.
     * Maps 0 lux → 0.0, ~100 lux → 0.5, ~10000 lux → 1.0
     */
    fun getNormalizedAmbientLight(): Float {
        if (currentLightLux <= 0) return 0f
        // Log scale: ln(lux + 1) / ln(10001) maps [0, 10000] to [0, 1]
        return (ln(currentLightLux.toDouble() + 1) / ln(10001.0)).toFloat().coerceIn(0f, 1f)
    }

    /**
     * Check for sudden movements that might indicate tampering.
     */
    private fun checkForTampering(values: FloatArray) {
        val magnitude = kotlin.math.sqrt(
            values[0] * values[0] +
            values[1] * values[1] +
            values[2] * values[2]
        )

        // Subtract gravity (~9.8 m/s²) and check for sudden changes
        val delta = abs(magnitude - lastAccelMagnitude)
        if (delta > tamperingThreshold) {
            tamperingDetected.set(true)
        }
        lastAccelMagnitude = magnitude
    }

    /**
     * Get current ambient light level in lux.
     * Returns -1 if sensor not available.
     */
    fun getAmbientLight(): Float = currentLightLux

    /**
     * Get current device temperature in Celsius.
     * Returns -1 if sensor not available.
     */
    fun getTemperature(): Float = currentTemperature

    /**
     * Get current device orientation based on screen rotation.
     * Uses DisplayManager which works for both Activity and Service contexts.
     */
    fun getOrientation(): String {
        val rotation = try {
            // Use DisplayManager which works for all context types (including Services)
            displayManager.getDisplay(Display.DEFAULT_DISPLAY)?.rotation ?: Surface.ROTATION_0
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get display rotation: ${e.message}")
            Surface.ROTATION_0
        }

        return when (rotation) {
            Surface.ROTATION_0, Surface.ROTATION_180 -> "portrait"
            Surface.ROTATION_90, Surface.ROTATION_270 -> "landscape"
            else -> "unknown"
        }
    }

    /**
     * Check if the device has been moved/tampered with since last check.
     * Resets the flag after checking.
     */
    fun checkAndResetTampering(): Boolean {
        return tamperingDetected.getAndSet(false)
    }

    /**
     * Get current environment metrics snapshot.
     */
    fun getEnvironmentMetrics(): EnvironmentMetrics {
        return EnvironmentMetrics(
            ambientLightLux = currentLightLux,
            deviceTemperatureC = currentTemperature,
            orientation = getOrientation()
        )
    }

    /**
     * Determine if screen is likely visible based on ambient light.
     * Screens in very dark environments might not be visible.
     */
    fun isScreenLikelyVisible(): Boolean {
        val vCfg = SensingConfig.get().viewability
        return currentLightLux < 0 || currentLightLux > vCfg.minLux
    }

    /**
     * Get a viewability score based on ambient conditions.
     * Higher score = better viewing conditions.
     */
    fun getViewabilityScore(): Float {
        val vCfg = SensingConfig.get().viewability
        if (currentLightLux < 0) return 0.5f  // Unknown, assume average

        return when {
            currentLightLux < vCfg.darkLuxThreshold -> 0.3f      // Very dark, hard to see
            currentLightLux < vCfg.dimLuxThreshold -> 0.7f     // Dim, but viewable
            currentLightLux < vCfg.goodLuxThreshold -> 1.0f    // Good indoor lighting
            currentLightLux < vCfg.brightLuxThreshold -> 0.8f  // Bright, some glare possible
            else -> 0.5f                      // Very bright, likely outdoor glare
        }
    }
}
