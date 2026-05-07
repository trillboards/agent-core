package com.trillboards.ctv.core.inference

import android.graphics.Bitmap
import java.nio.ByteBuffer

/**
 * Sealed class representing the possible input types for inference processors.
 *
 * Existing processors use three distinct input patterns:
 * - Camera frame (Bitmap or pre-allocated ByteBuffer) for vision models
 * - Audio samples (ShortArray) for audio classification / ASR
 * - MultiModal (prompt + image + optional audio) for VLM processors
 *
 * Each processor declares its [HardwareRequirement] and accepts the
 * corresponding [InferenceInput] subtype in [InferenceProcessor.process].
 */
sealed class InferenceInput {

    /**
     * A single camera frame for vision-based processors.
     *
     * Matches the pattern used by PersonDetectionProcessor, EmotionClassificationProcessor,
     * PoseEngagementProcessor, AgeGenderProcessor, ObjectDetectionProcessor, and
     * GazeTrackingProcessor — all of which accept a Bitmap per frame.
     *
     * @param bitmap In-memory camera frame. Never stored or transmitted.
     * @param timestamp Capture timestamp in milliseconds (SystemClock or epoch).
     * @param rgbByteBuffer Optional pre-allocated RGB ByteBuffer for zero-allocation
     *   pipelines (see PoseEngagementProcessor.processWithBuffer). When provided,
     *   processors that support it should prefer this over [bitmap] to avoid
     *   Bitmap→ByteBuffer conversion and GC pressure.
     * @param width Image width in pixels (required when using [rgbByteBuffer]).
     * @param height Image height in pixels (required when using [rgbByteBuffer]).
     */
    data class CameraFrame(
        val bitmap: Bitmap,
        val timestamp: Long,
        val rgbByteBuffer: ByteBuffer? = null,
        val width: Int = bitmap.width,
        val height: Int = bitmap.height
    ) : InferenceInput()

    /**
     * A chunk of audio samples for audio-based processors.
     *
     * Matches the pattern used by AudioClassificationProcessor (YAMNet)
     * and SpeechIntelligenceProcessor (Moonshine ASR). Both expect PCM 16-bit
     * mono audio at a specific sample rate.
     *
     * @param samples Raw PCM 16-bit audio samples.
     * @param sampleRate Sample rate in Hz (typically 16000 for speech, 16000 for YAMNet).
     * @param durationMs Duration of this audio chunk in milliseconds.
     */
    data class AudioBuffer(
        val samples: ShortArray,
        val sampleRate: Int,
        val durationMs: Long = (samples.size * 1000L) / sampleRate
    ) : InferenceInput() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is AudioBuffer) return false
            return samples.contentEquals(other.samples) &&
                sampleRate == other.sampleRate &&
                durationMs == other.durationMs
        }

        override fun hashCode(): Int {
            var result = samples.contentHashCode()
            result = 31 * result + sampleRate
            result = 31 * result + durationMs.hashCode()
            return result
        }
    }

    /**
     * Multi-modal input combining text prompt with optional image and audio.
     *
     * Designed for VLM (Vision-Language Model) processors that accept
     * a natural language prompt alongside visual and/or audio context.
     * No existing processors use this yet — it's the target for Phase 1B
     * Gemini Nano / on-device VLM integration.
     *
     * @param prompt Natural language prompt or instruction for the model.
     * @param image Optional image context (e.g., current camera frame).
     * @param audio Optional audio context (e.g., recent ambient audio).
     * @param metadata Optional key-value metadata for processor-specific context
     *   (e.g., scene description, detected objects, venue type).
     */
    data class MultiModal(
        val prompt: String,
        val image: Bitmap? = null,
        val audio: ShortArray? = null,
        val metadata: Map<String, String> = emptyMap()
    ) : InferenceInput() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is MultiModal) return false
            return prompt == other.prompt &&
                image == other.image &&
                (audio?.contentEquals(other.audio) ?: (other.audio == null)) &&
                metadata == other.metadata
        }

        override fun hashCode(): Int {
            var result = prompt.hashCode()
            result = 31 * result + (image?.hashCode() ?: 0)
            result = 31 * result + (audio?.contentHashCode() ?: 0)
            result = 31 * result + metadata.hashCode()
            return result
        }
    }
}
