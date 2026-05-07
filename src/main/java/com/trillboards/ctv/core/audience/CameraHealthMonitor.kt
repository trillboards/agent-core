package com.trillboards.ctv.core.audience

import android.os.Build
import android.util.Log
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/**
 * Camera lifecycle health gauge with retry/backoff and escalation.
 *
 * Background — Adam @ Focus Media incident (RockChip Generic Android 13 tablet,
 * 698f7b52b6b0431f0f994e6c): 30+ days of `avg_fps=null`, zero `vision_call_log`
 * rows, heartbeat fine. Camera was never started, but every failure mode
 * silently no-op'd. Examples:
 *
 *   1. `selectBestCamera()` did `try { lensFacing == EXTERNAL } catch (e) { false }`
 *      — on RockChip, lensFacing throws on certain HAL builds, all three
 *      priority filters fall through, function returns null → "no camera
 *      found" log + audio-only fallback, no telemetry, no retry.
 *   2. `bindToLifecycle()` exception was caught with one Log.e and never
 *      re-tried.
 *   3. `ProcessCameraProvider.getInstance()` failed → catch with one Log.e,
 *      no retry, no escalation.
 *
 * This class exists so every failure mode in the camera-start path is:
 *   - Loud:        `Log.e` + telemetry to `device_stability_events`.
 *   - Recoverable: exponential backoff retry (per-class strategy).
 *   - Escalatable: after N retries, `CameraInitializationFailed` stability
 *                  event posted to backend with full diagnostics.
 *   - Self-correcting: when state stays `FAILED` for >24h, the heartbeat
 *                  reports `cap_face_detection=false` so the model_tier /
 *                  scheduling loop devolves to a cap that matches reality.
 *
 * Pure-function helpers (`computeNextRetryDelayMs`, `shouldDevolveCap`,
 * `classifyFailure`) are exposed for unit tests so we don't need a Robolectric
 * harness to validate the policy.
 */
class CameraHealthMonitor(
    private val now: () -> Long = { System.currentTimeMillis() }
) {
    companion object {
        private const val TAG = "CameraHealthMonitor"

        /**
         * After this many consecutive failures the monitor stops retrying for
         * `CameraXBindException` / `NoCameraSelected` classes and emits the
         * `CameraInitializationFailed` escalation event. `PermissionDenied`
         * keeps retrying indefinitely (every 1h) — permissions can be granted
         * at any time without restarting the app.
         */
        const val DEFAULT_MAX_RETRIES = 5

        /**
         * Cap-devolution threshold. If the camera has been in FAILED state for
         * longer than this, `cap_face_detection` flips to false on the next
         * heartbeat so the server stops claiming this device can do face
         * detection — matches reality.
         */
        const val DEVOLVE_CAP_AFTER_MS = 24 * 60 * 60 * 1000L  // 24 hours

        /**
         * Backoff schedule by retry attempt index (0-based). After the last
         * entry we hold at the last delay (every 1 hour forever).
         *
         * 1s → 5s → 30s → 5min → 30min → 1h → 1h → 1h ...
         */
        @JvmStatic
        internal val BACKOFF_SCHEDULE_MS = longArrayOf(
            1_000L,        // attempt 0 → wait 1s before retry 1
            5_000L,        // attempt 1 → wait 5s
            30_000L,       // attempt 2 → wait 30s
            5 * 60_000L,   // attempt 3 → wait 5 min
            30 * 60_000L,  // attempt 4 → wait 30 min
            60 * 60_000L   // attempt 5+ → wait 1 hour
        )

        /**
         * Compute the next retry delay given the current attempt count. Pure
         * function so unit tests can validate the policy without mocking time.
         */
        @JvmStatic
        fun computeNextRetryDelayMs(attemptCount: Int): Long {
            if (attemptCount < 0) return BACKOFF_SCHEDULE_MS[0]
            val idx = attemptCount.coerceAtMost(BACKOFF_SCHEDULE_MS.size - 1)
            return BACKOFF_SCHEDULE_MS[idx]
        }

        /**
         * Decide whether the heartbeat should devolve `cap_face_detection`
         * to false. Devolves only when:
         *   - state is FAILED (camera repeatedly couldn't start), AND
         *   - stuck in FAILED for ≥ DEVOLVE_CAP_AFTER_MS.
         *
         * NEVER_STARTED alone does not devolve — fresh boots that haven't
         * tried yet should keep their declared capability.
         *
         * Pure function — extracted for tests.
         */
        @JvmStatic
        fun shouldDevolveCap(
            state: CameraState,
            failedSinceMs: Long,
            nowMs: Long
        ): Boolean {
            if (state != CameraState.FAILED) return false
            if (failedSinceMs <= 0L) return false
            return (nowMs - failedSinceMs) >= DEVOLVE_CAP_AFTER_MS
        }

        /**
         * Map a Throwable thrown anywhere in the camera-start path to a
         * stable failure-class string. Pure — exported for tests.
         */
        @JvmStatic
        fun classifyFailure(t: Throwable?): FailureClass {
            if (t == null) return FailureClass.UNKNOWN
            // Drill through CameraX's wrapping exceptions; the root cause is
            // what tells us whether to retry forever (permission) or give up
            // (HAL bind failure). InitializationException / IllegalStateException
            // wrap most CameraX errors.
            val root = generateSequence<Throwable>(t) { it.cause }.last()
            val msg = (t.message ?: "") + " | " + (root.message ?: "")
            val cls = root::class.java.simpleName
            return when {
                msg.contains("permission", ignoreCase = true) ||
                    cls.contains("SecurityException", ignoreCase = true) -> FailureClass.PERMISSION_DENIED
                msg.contains("CameraUnavailableException", ignoreCase = true) ||
                    msg.contains("camera disabled", ignoreCase = true) ||
                    msg.contains("camera in use", ignoreCase = true) -> FailureClass.CAMERA_UNAVAILABLE
                msg.contains("InitializationException", ignoreCase = true) ||
                    msg.contains("getInstance", ignoreCase = true) -> FailureClass.PROVIDER_INIT_FAILED
                msg.contains("bindToLifecycle", ignoreCase = true) ||
                    msg.contains("UseCase", ignoreCase = true) -> FailureClass.BIND_FAILED
                msg.contains("LENS_FACING", ignoreCase = true) ||
                    msg.contains("lensFacing", ignoreCase = true) -> FailureClass.LENS_FACING_QUERY_FAILED
                else -> FailureClass.UNKNOWN
            }
        }

        /**
         * Decide whether to keep retrying or escalate. Permission denial keeps
         * retrying forever (granting permission is a runtime config flip).
         * Other classes give up after maxRetries and escalate to operator.
         */
        @JvmStatic
        fun shouldGiveUp(failureClass: FailureClass, attemptCount: Int, maxRetries: Int): Boolean {
            return when (failureClass) {
                FailureClass.PERMISSION_DENIED -> false  // retry forever; user may grant
                FailureClass.NO_CAMERA_HARDWARE -> attemptCount >= 1  // no point retrying
                FailureClass.EMULATOR_OR_HEADLESS -> true  // never retry
                else -> attemptCount >= maxRetries
            }
        }
    }

    enum class CameraState {
        NEVER_STARTED,  // boot state — no start attempt yet
        STARTING,       // start attempt in flight
        OPEN,           // bound + first frame received
        FAILED,         // start path raised; retries scheduled or exhausted
        CLOSED          // intentionally stopped (e.g. service shutdown)
    }

    enum class FailureClass {
        PERMISSION_DENIED,           // CAMERA permission not granted
        NO_CAMERA_HARDWARE,          // CameraManager.cameraIdList empty
        PROVIDER_INIT_FAILED,        // ProcessCameraProvider.getInstance() failed
        NO_CAMERA_SELECTED,          // selectBestCamera returned null (RockChip class)
        LENS_FACING_QUERY_FAILED,    // CameraInfo.lensFacing threw on every camera
        BIND_FAILED,                 // bindToLifecycle threw
        CAMERA_UNAVAILABLE,          // CameraX reported camera in use / disabled
        EMULATOR_OR_HEADLESS,        // detected emulator / headless build
        FRAME_TIMEOUT,               // camera bound but no frames within grace
        UNKNOWN
    }

    data class HealthSnapshot(
        val state: CameraState,
        val lastFrameAtMs: Long,
        val retryCount: Int,
        val lastFailureClass: FailureClass?,
        val lastFailureMessage: String?,
        val failedSinceMs: Long,
        val openSinceMs: Long,
        val nextRetryAtMs: Long,
        val maxRetries: Int
    )

    /**
     * Diagnostics included in the escalation `stabilityEvent` when retries
     * are exhausted. Captured fresh at escalation time so the operator sees
     * the device's actual ABI / API level / Build state at the moment of
     * failure.
     */
    data class EscalationContext(
        val failureClass: FailureClass,
        val failureMessage: String?,
        val retryCount: Int,
        val maxRetries: Int,
        val deviceManufacturer: String,
        val deviceModel: String,
        val osVersion: String,
        val osApiLevel: Int,
        val cpuAbi: String,
        val totalCameraCount: Int,
        val durationMsSinceFirstAttempt: Long
    )

    private val state = AtomicReference(CameraState.NEVER_STARTED)
    private val retryCount = AtomicLong(0L)
    private val lastFrameAtMs = AtomicLong(0L)
    private val lastFailureClass = AtomicReference<FailureClass?>(null)
    private val lastFailureMessage = AtomicReference<String?>(null)
    private val failedSinceMs = AtomicLong(0L)
    private val openSinceMs = AtomicLong(0L)
    private val firstAttemptAtMs = AtomicLong(0L)
    private val nextRetryAtMs = AtomicLong(0L)
    private val maxRetries = AtomicLong(DEFAULT_MAX_RETRIES.toLong())

    /**
     * Set the maximum retry count. Used for unit tests that want a tighter
     * loop, and for prod overrides via SensingConfig push.
     */
    fun setMaxRetries(value: Int) {
        maxRetries.set(value.coerceAtLeast(1).toLong())
    }

    /**
     * Mark a start attempt as in flight. Called immediately before any of:
     *   - ProcessCameraProvider.getInstance(...)
     *   - bindToLifecycle(...)
     *   - selectBestCamera(...)
     */
    fun onStartAttempt() {
        if (firstAttemptAtMs.get() == 0L) {
            firstAttemptAtMs.set(now())
        }
        state.set(CameraState.STARTING)
        Log.i(TAG, "[Health] state=STARTING, attempt=${retryCount.get() + 1}, maxRetries=${maxRetries.get()}")
    }

    /**
     * Mark camera as bound — bindToLifecycle returned without throwing AND
     * `onCameraReady` callback fired. We don't yet have a frame; the OPEN
     * transition only happens once the first frame arrives. This keeps OPEN
     * meaningful: "the camera is actually delivering frames", not "the bind
     * call returned".
     */
    fun onCameraBound() {
        Log.i(TAG, "[Health] camera bound, awaiting first frame")
    }

    /**
     * Called from `AudienceAnalyzer.processImage` on every successful frame.
     * First frame after a STARTING / FAILED state transitions to OPEN.
     */
    fun onFrameProcessed() {
        val nowMs = now()
        lastFrameAtMs.set(nowMs)
        val prev = state.getAndSet(CameraState.OPEN)
        if (prev != CameraState.OPEN) {
            openSinceMs.set(nowMs)
            // Reset retry/failure state — we're healthy again.
            retryCount.set(0L)
            failedSinceMs.set(0L)
            lastFailureClass.set(null)
            lastFailureMessage.set(null)
            nextRetryAtMs.set(0L)
            Log.i(TAG, "[Health] state=$prev → OPEN (first frame received)")
        }
    }

    /**
     * Camera-start path raised. Returns the next retry delay in ms (0 if
     * `shouldGiveUp` is true), and the caller must schedule its own retry
     * — this class is policy-only, not scheduler.
     *
     * The caller MUST also fire telemetry + stability event for escalation;
     * `buildEscalationContext` returns the payload.
     */
    fun onStartFailure(t: Throwable?, hint: FailureClass? = null): RetryDecision {
        val klass = hint ?: classifyFailure(t)
        val msg = (t?.message ?: klass.name).take(512)
        val attempts = retryCount.incrementAndGet().toInt()
        val nowMs = now()

        if (failedSinceMs.get() == 0L) {
            failedSinceMs.set(nowMs)
        }
        lastFailureClass.set(klass)
        lastFailureMessage.set(msg)
        state.set(CameraState.FAILED)

        val giveUp = shouldGiveUp(klass, attempts, maxRetries.get().toInt())
        if (giveUp) {
            nextRetryAtMs.set(0L)
            Log.e(
                TAG,
                "[Health] CAMERA START FAILED (terminal) - class=$klass, attempts=$attempts, " +
                    "maxRetries=${maxRetries.get()}, msg=$msg",
                t
            )
            return RetryDecision(retryAfterMs = 0L, giveUp = true, failureClass = klass, failureMessage = msg)
        }

        val delayMs = computeNextRetryDelayMs(attempts - 1)
        nextRetryAtMs.set(nowMs + delayMs)
        Log.e(
            TAG,
            "[Health] CAMERA START FAILED (will retry) - class=$klass, attempt=$attempts, " +
                "nextRetryInMs=$delayMs, msg=$msg",
            t
        )
        return RetryDecision(retryAfterMs = delayMs, giveUp = false, failureClass = klass, failureMessage = msg)
    }

    /**
     * Called when AudienceAnalyzer.stop() runs. Differentiated from FAILED
     * so the heartbeat can show "intentionally closed" vs "broken".
     */
    fun onStopRequested() {
        state.set(CameraState.CLOSED)
        nextRetryAtMs.set(0L)
        Log.i(TAG, "[Health] state=CLOSED (stop requested)")
    }

    /**
     * Camera was open and delivering frames but we noticed a long stall —
     * push back to FAILED so the retry/escalate path engages.
     */
    fun onTransientFailure(reason: String) {
        val nowMs = now()
        if (state.get() == CameraState.OPEN) {
            Log.w(TAG, "[Health] OPEN → FAILED (transient): $reason")
        }
        state.set(CameraState.FAILED)
        if (failedSinceMs.get() == 0L) {
            failedSinceMs.set(nowMs)
        }
        lastFailureClass.set(FailureClass.FRAME_TIMEOUT)
        lastFailureMessage.set(reason)
    }

    fun snapshot(): HealthSnapshot = HealthSnapshot(
        state = state.get(),
        lastFrameAtMs = lastFrameAtMs.get(),
        retryCount = retryCount.get().toInt(),
        lastFailureClass = lastFailureClass.get(),
        lastFailureMessage = lastFailureMessage.get(),
        failedSinceMs = failedSinceMs.get(),
        openSinceMs = openSinceMs.get(),
        nextRetryAtMs = nextRetryAtMs.get(),
        maxRetries = maxRetries.get().toInt()
    )

    /**
     * Heartbeat-shaped projection of the current camera health state.
     *
     * Wire format mirrors `device_stability_events.metadata` so the existing
     * cold-storage exporter and dashboards pick this up without schema
     * changes. Reads the AtomicReferences atomically (single snapshot()
     * call) so all fields share a consistent view.
     */
    fun toHeartbeatJson(): org.json.JSONObject {
        val s = snapshot()
        return org.json.JSONObject().apply {
            put("state", s.state.name.lowercase())
            put("last_frame_at", s.lastFrameAtMs)
            put("retry_count", s.retryCount)
            put("max_retries", s.maxRetries)
            put("last_failure_class", s.lastFailureClass?.name?.lowercase())
            put("last_failure_message", s.lastFailureMessage)
            put("failed_since", s.failedSinceMs)
            put("open_since", s.openSinceMs)
            put("next_retry_at", s.nextRetryAtMs)
        }
    }

    /**
     * Whether the heartbeat should devolve cap_face_detection on this tick.
     * Pure proxy to `shouldDevolveCap` so the agent doesn't have to call the
     * companion explicitly.
     */
    fun shouldDevolveFaceCap(): Boolean = shouldDevolveCap(state.get(), failedSinceMs.get(), now())

    /**
     * Build the escalation context that ships in the
     * `CameraInitializationFailed` stabilityEvent. Caller adds Build.* fields
     * here so we don't drag a Context through the pure-function tests.
     */
    fun buildEscalationContext(
        deviceManufacturer: String,
        deviceModel: String,
        osVersion: String,
        osApiLevel: Int,
        cpuAbi: String,
        totalCameraCount: Int
    ): EscalationContext {
        val s = snapshot()
        val nowMs = now()
        val firstAttempt = firstAttemptAtMs.get()
        val durationMs = if (firstAttempt > 0) nowMs - firstAttempt else 0L
        return EscalationContext(
            failureClass = s.lastFailureClass ?: FailureClass.UNKNOWN,
            failureMessage = s.lastFailureMessage,
            retryCount = s.retryCount,
            maxRetries = s.maxRetries,
            deviceManufacturer = deviceManufacturer,
            deviceModel = deviceModel,
            osVersion = osVersion,
            osApiLevel = osApiLevel,
            cpuAbi = cpuAbi,
            totalCameraCount = totalCameraCount,
            durationMsSinceFirstAttempt = durationMs
        )
    }

    /**
     * Convenience helper that fills in the deviceManufacturer/model/api fields
     * from android.os.Build at the call site. Kept distinct from
     * `buildEscalationContext` so the unit tests can construct the context
     * without touching `Build.*`.
     */
    fun buildEscalationContextFromBuild(totalCameraCount: Int): EscalationContext =
        buildEscalationContext(
            deviceManufacturer = Build.MANUFACTURER ?: "unknown",
            deviceModel = Build.MODEL ?: "unknown",
            osVersion = Build.VERSION.RELEASE ?: "unknown",
            osApiLevel = Build.VERSION.SDK_INT,
            cpuAbi = (Build.SUPPORTED_ABIS?.firstOrNull() ?: Build.CPU_ABI ?: "unknown"),
            totalCameraCount = totalCameraCount
        )

    data class RetryDecision(
        val retryAfterMs: Long,
        val giveUp: Boolean,
        val failureClass: FailureClass,
        val failureMessage: String?
    )
}
