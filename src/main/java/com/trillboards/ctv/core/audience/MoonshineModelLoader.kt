package com.trillboards.ctv.core.audience

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Extracts Moonshine ASR ONNX model files from APK assets to the filesystem.
 * sherpa-onnx requires filesystem paths (not asset streams), so we copy once on first run.
 * Subsequent runs skip extraction if files already exist with correct sizes.
 */
object MoonshineModelLoader {

    private const val TAG = "MoonshineModelLoader"
    private const val ASSET_DIR = "moonshine-tiny"
    private const val OUTPUT_DIR = "moonshine-tiny"

    private val MODEL_FILES = listOf(
        "preprocess.onnx",
        "encode.int8.onnx",
        "uncached_decode.int8.onnx",
        "cached_decode.int8.onnx",
        "tokens.txt"
    )

    data class ModelPaths(
        val preprocessor: String,
        val encoder: String,
        val uncachedDecoder: String,
        val cachedDecoder: String,
        val tokens: String
    )

    /**
     * Extract model files from assets to filesDir if not already present.
     * Returns ModelPaths with absolute filesystem paths, or null on failure.
     */
    fun extractIfNeeded(context: Context): ModelPaths? {
        val outputDir = File(context.filesDir, OUTPUT_DIR)
        if (!outputDir.exists()) outputDir.mkdirs()

        var allPresent = true
        for (fileName in MODEL_FILES) {
            val outputFile = File(outputDir, fileName)
            if (!outputFile.exists() || outputFile.length() == 0L) {
                allPresent = false
                break
            }
        }

        if (allPresent) {
            Log.i(TAG, "Moonshine model files already extracted")
        } else {
            Log.i(TAG, "Extracting Moonshine model files from assets...")
            for (fileName in MODEL_FILES) {
                val outputFile = File(outputDir, fileName)
                try {
                    context.assets.open("$ASSET_DIR/$fileName").use { input ->
                        FileOutputStream(outputFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.d(TAG, "Extracted: $fileName (${outputFile.length()} bytes)")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to extract $fileName", e)
                    // Clean up partial extraction
                    outputFile.delete()
                    return null
                }
            }
            Log.i(TAG, "Moonshine model extracted successfully")
        }

        return ModelPaths(
            preprocessor = File(outputDir, "preprocess.onnx").absolutePath,
            encoder = File(outputDir, "encode.int8.onnx").absolutePath,
            uncachedDecoder = File(outputDir, "uncached_decode.int8.onnx").absolutePath,
            cachedDecoder = File(outputDir, "cached_decode.int8.onnx").absolutePath,
            tokens = File(outputDir, "tokens.txt").absolutePath
        )
    }

    /**
     * Check if model files are available in assets.
     */
    fun hasModelAssets(context: Context): Boolean {
        return try {
            for (fileName in MODEL_FILES) {
                context.assets.open("$ASSET_DIR/$fileName").close()
            }
            true
        } catch (e: Exception) {
            Log.w(TAG, "Moonshine model assets not found: ${e.message}")
            false
        }
    }

    /**
     * Delete extracted model files (for cleanup or re-extraction).
     */
    fun deleteExtractedFiles(context: Context) {
        val outputDir = File(context.filesDir, OUTPUT_DIR)
        if (outputDir.exists()) {
            outputDir.deleteRecursively()
            Log.i(TAG, "Deleted extracted Moonshine model files")
        }
    }
}
