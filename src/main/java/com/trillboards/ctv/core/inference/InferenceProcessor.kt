package com.trillboards.ctv.core.inference

/**
 * Hardware requirement for an inference processor.
 *
 * Maps to the physical sensors the processor needs:
 * - [CAMERA] — PersonDetection, ObjectDetection, Emotion, Pose, AgeGender, GazeTracking
 * - [MICROPHONE] — AudioClassification (YAMNet), SpeechIntelligence (Moonshine)
 * - [CAMERA_AND_MICROPHONE] — future multi-modal processors (VLM with audio context)
 * - [NONE] — processors that work on pre-computed data (e.g., post-aggregation enrichment)
 */
enum class HardwareRequirement {
    CAMERA,
    MICROPHONE,
    CAMERA_AND_MICROPHONE,
    NONE
}

/**
 * Formal interface for inference processors in the Trillboards CTV sensing pipeline.
 *
 * This interface codifies the convention pattern that all 8 existing processors
 * follow organically. Each processor in `com.trillboards.ctv.core.audience` has:
 * - `hasModel(): Boolean` — checks if model asset exists
 * - `initialize(): Boolean` — loads model, allocates buffers
 * - `process(...)` — runs inference (various signatures)
 * - `release()` — frees resources
 * - `isReady(): Boolean` — checks initialization state
 *
 * By formalizing this into an interface, we enable:
 * 1. **ProcessorRegistry** — discover and manage processors by modelId
 * 2. **Uniform lifecycle** — initialize/release all processors through one API
 * 3. **Memory budgeting** — sum memory footprints before enabling processors
 * 4. **VLM integration** — new Gemini Nano/on-device VLM processors implement
 *    the same interface with [InferenceInput.MultiModal] input
 *
 * ## Lifecycle
 * ```
 * hasModel() → initialize() → isReady() → process() → ... → release()
 * ```
 *
 * ## Thread Safety
 * Implementations MUST be safe to call from coroutines on Dispatchers.Default.
 * The [process] function is suspend to allow async inference (e.g., server-side
 * fallback in SpeechIntelligenceProcessor's hybrid race architecture).
 *
 * ## Existing Processors (NOT retrofitted in this PR)
 * - PersonDetectionProcessor (efficientdet) — CAMERA
 * - ObjectDetectionProcessor (efficientdet) — CAMERA
 * - FaceLandmarkerProcessor (fer_plus emotion + iris gaze + head pose) — CAMERA
 * - PoseEngagementProcessor (movenet) — CAMERA
 * - AgeGenderProcessor (age_gender) — CAMERA
 * - AudioClassificationProcessor (yamnet) — MICROPHONE
 * - SpeechIntelligenceProcessor (whisper_tiny) — MICROPHONE
 */
interface InferenceProcessor {

    /**
     * Unique identifier for this processor's model.
     *
     * Convention: use the model name without extension, e.g., "efficientdet",
     * "yamnet", "fer_plus", "movenet", "age_gender", "blazeface", "whisper_tiny".
     * Must be stable across versions — used as a key in [ProcessorRegistry]
     * and in heartbeat telemetry payloads.
     */
    val modelId: String

    /**
     * Hardware sensors required by this processor.
     *
     * Used by the registry to filter processors based on available hardware
     * (e.g., skip CAMERA processors on devices without a camera, skip
     * MICROPHONE processors when audio permission is denied).
     */
    val hardwareRequirement: HardwareRequirement

    /**
     * Check if the model asset (TFLite, ONNX, MediaPipe task file, etc.) is
     * available on this device.
     *
     * Existing processors check `context.assets.open(MODEL_FILE).close()`.
     * This should be a fast, non-blocking check — no model loading.
     *
     * @return true if the model file exists and is accessible.
     */
    fun hasModel(): Boolean

    /**
     * Load the model and allocate inference buffers.
     *
     * This is the expensive operation — it loads the model into memory,
     * creates the interpreter/detector/landmarker, and pre-allocates I/O buffers.
     * Should be called once, typically during AudienceSensingService startup.
     *
     * Implementations should be idempotent — calling initialize() when already
     * initialized should return true without re-loading.
     *
     * @return true if initialization succeeded and [isReady] will return true.
     */
    fun initialize(): Boolean

    /**
     * Check if the processor is initialized and ready to accept input.
     *
     * @return true if [initialize] has been called successfully and [release]
     *   has not been called since.
     */
    fun isReady(): Boolean

    /**
     * Run inference on the given input.
     *
     * This is the core inference method. Implementations should:
     * 1. Validate that [isReady] is true
     * 2. Convert [InferenceInput] to model-specific format
     * 3. Run inference
     * 4. Wrap results in [InferenceOutput]
     * 5. Return null if inference fails (don't throw)
     *
     * The suspend modifier allows processors with async components (e.g.,
     * SpeechIntelligenceProcessor's server Gemini fallback race) to use
     * coroutines naturally.
     *
     * @param input The inference input. Processors should check the subtype
     *   matches their [hardwareRequirement] and return null for mismatches.
     * @return Inference result wrapped in [InferenceOutput], or null if
     *   inference failed or the input type is incompatible.
     */
    suspend fun process(input: InferenceInput): InferenceOutput?

    /**
     * Release all resources — model interpreter, buffers, native handles.
     *
     * After calling release(), [isReady] must return false. The processor
     * can be re-initialized by calling [initialize] again.
     *
     * Implementations should be safe to call multiple times (idempotent).
     */
    fun release()

    /**
     * Estimated memory footprint of this processor when initialized, in megabytes.
     *
     * Used by [ProcessorRegistry.totalMemoryMb] to budget memory across
     * processors. This is an estimate — actual memory may vary by device
     * and model variant.
     *
     * Common values from existing processors:
     * - EfficientDet-Lite0: ~5 MB
     * - FER emotion: ~0.3 MB
     * - Pose Lite: ~3.5 MB
     * - YAMNet: ~3 MB
     * - Age/Gender: ~1 MB
     * - Moonshine ASR: ~60 MB
     *
     * @return Estimated memory in MB. Return 0f if not initialized.
     */
    fun getMemoryFootprintMb(): Float
}
