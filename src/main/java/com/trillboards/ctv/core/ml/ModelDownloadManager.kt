package com.trillboards.ctv.core.ml

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * OTA Model Binary Download Manager for large model files (1-4 GB).
 *
 * Provides resumable downloads with HTTP Range headers, SHA-256 checksum
 * validation, progress reporting via Kotlin Flow, and disk space management.
 *
 * Storage layout:
 *   context.cacheDir/models/{modelId}/
 *     model.bin          — completed model binary
 *     model.bin.part     — partial download (in progress)
 *     download.meta      — JSON metadata (bytesDownloaded, totalBytes, checksumSha256, downloadUrl)
 *
 * The cache directory is used so the OS can reclaim space if needed,
 * but completed models are validated via checksum before use.
 */
class ModelDownloadManager(
    private val context: Context,
    private val maxBytesPerSecond: Long = DEFAULT_MAX_BYTES_PER_SECOND
) {

    companion object {
        private const val TAG = "ModelDownloadManager"
        private const val MODEL_FILENAME = "model.bin"
        private const val PARTIAL_FILENAME = "model.bin.part"
        const val DEFAULT_MAX_BYTES_PER_SECOND = 2L * 1024L * 1024L // 2 MB/s default

        /**
         * Get the appropriate model filename based on format.
         * LiteRT-LM SDK requires .litertlm extension, others use .bin.
         */
        fun getModelFilename(modelFormat: String?): String = when (modelFormat) {
            "litert-lm" -> "model.litertlm"
            "onnx-vlm" -> "model.onnx"
            else -> MODEL_FILENAME
        }
        private const val META_FILENAME = "download.meta"
        private const val PROGRESS_EMIT_INTERVAL_BYTES = 256 * 1024L // emit progress every 256 KB
        private const val DOWNLOAD_BUFFER_SIZE = 8192
        private const val CONNECT_TIMEOUT_SECONDS = 30L
        private const val READ_TIMEOUT_SECONDS = 60L
    }

    /**
     * Progress state emitted during model downloads.
     */
    data class DownloadProgress(
        val modelId: String,
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val percentComplete: Float,
        val state: DownloadState
    )

    enum class DownloadState {
        QUEUED, DOWNLOADING, VERIFYING, COMPLETE, FAILED, CANCELLED
    }

    /**
     * Manifest describing a model to download.
     * Provided by the server's model registry endpoint.
     */
    /**
     * Companion file to download alongside the primary model.
     * Used for auxiliary files required for multimodal vision inference.
     */
    data class CompanionFile(
        val filename: String,
        val downloadUrl: String,
        val checksumSha256: String,
        val sizeBytes: Long
    )

    data class ModelManifest(
        val modelId: String,
        val downloadUrl: String,
        val checksumSha256: String,
        val sizeBytes: Long,
        val modelFormat: String,  // litert-lm, executorch, onnx, tflite
        val companionFiles: List<CompanionFile> = emptyList()
    )

    /** Root directory for all downloaded models. */
    private val modelsDir: File get() = File(context.cacheDir, "models")

    /** Tracks active download coroutine jobs for cancellation. */
    private val activeDownloads = ConcurrentHashMap<String, Job>()

    /** Shared OkHttpClient with appropriate timeouts for large file downloads. */
    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()
    }

    /**
     * Download a model binary with resumable progress, checksum validation,
     * and cancellation support.
     *
     * The returned Flow emits [DownloadProgress] updates as data is received.
     * On success the final emission has state [DownloadState.COMPLETE].
     * On failure the final emission has state [DownloadState.FAILED].
     *
     * If a partial download already exists (from a previous interrupted attempt),
     * the download resumes from where it left off using HTTP Range headers.
     *
     * @param manifest Model manifest with download URL, checksum, and size
     * @return Flow of DownloadProgress updates
     */
    fun downloadModel(manifest: ModelManifest): Flow<DownloadProgress> = flow {
        val modelId = manifest.modelId
        val modelDir = File(modelsDir, modelId)
        // Use format-specific extension so LiteRT-LM gets .litertlm, ONNX gets .onnx, etc.
        val modelFilename = getModelFilename(manifest.modelFormat)
        val completeFile = File(modelDir, modelFilename)
        val partialFile = File(modelDir, "$modelFilename.part")
        val metaFile = File(modelDir, META_FILENAME)

        // If already fully downloaded and valid, short-circuit
        if (completeFile.exists() && verifyChecksum(completeFile, manifest.checksumSha256)) {
            emit(DownloadProgress(modelId, manifest.sizeBytes, manifest.sizeBytes, 100f, DownloadState.COMPLETE))
            return@flow
        }

        // Check available storage before downloading
        // Use parent dir (cacheDir) for space check — modelsDir may not exist yet on fresh install
        // (File.usableSpace returns 0 for non-existent paths on some Android versions)
        val requiredBytes = manifest.sizeBytes + (manifest.companionFiles.sumOf { it.sizeBytes })
        val storageCheckDir = if (modelsDir.exists()) modelsDir else (modelsDir.parentFile ?: modelsDir)
        val availableBytes = storageCheckDir.usableSpace
        val safetyMarginBytes = 500L * 1024L * 1024L // 500MB headroom for OS + other apps
        if (availableBytes < requiredBytes + safetyMarginBytes) {
            Log.e(TAG, "Insufficient storage for model $modelId: need ${(requiredBytes + safetyMarginBytes) / (1024 * 1024)}MB, available ${availableBytes / (1024 * 1024)}MB")
            emit(DownloadProgress(modelId, 0, manifest.sizeBytes, 0f, DownloadState.FAILED))
            return@flow
        }

        // Ensure directory exists
        modelDir.mkdirs()

        // Retain previous version for instant rollback before new download replaces it
        retainPreviousVersion(modelDir, modelFilename)

        emit(DownloadProgress(modelId, 0, manifest.sizeBytes, 0f, DownloadState.QUEUED))

        // Determine resume point from partial file
        var bytesDownloaded = 0L
        if (partialFile.exists() && metaFile.exists()) {
            try {
                val meta = JSONObject(metaFile.readText())
                val savedBytes = meta.optLong("bytesDownloaded", 0L)
                val savedChecksum = meta.optString("checksumSha256", "")
                // Only resume if the checksum matches (same file being downloaded)
                if (savedChecksum == manifest.checksumSha256 && savedBytes > 0 && savedBytes == partialFile.length()) {
                    bytesDownloaded = savedBytes
                    Log.i(TAG, "Resuming download of $modelId from byte $bytesDownloaded")
                } else {
                    // Metadata mismatch — start fresh
                    partialFile.delete()
                    metaFile.delete()
                    Log.d(TAG, "Metadata mismatch for $modelId, restarting download")
                }
            } catch (e: Exception) {
                partialFile.delete()
                metaFile.delete()
                Log.d(TAG, "Corrupt metadata for $modelId, restarting download")
            }
        }

        // Build HTTP request with Range header for resume
        val requestBuilder = Request.Builder().url(manifest.downloadUrl)
        if (bytesDownloaded > 0) {
            requestBuilder.header("Range", "bytes=$bytesDownloaded-")
        }

        emit(DownloadProgress(modelId, bytesDownloaded, manifest.sizeBytes,
            percentOf(bytesDownloaded, manifest.sizeBytes), DownloadState.DOWNLOADING))

        try {
            val response = httpClient.newCall(requestBuilder.build()).execute()

            if (!response.isSuccessful && response.code != 206) {
                Log.e(TAG, "Download failed for $modelId: HTTP ${response.code}")
                response.close()
                autoRestorePreviousVersion(modelDir, modelFilename)
                emit(DownloadProgress(modelId, bytesDownloaded, manifest.sizeBytes,
                    percentOf(bytesDownloaded, manifest.sizeBytes), DownloadState.FAILED))
                return@flow
            }

            // If server doesn't support Range and sent full content, reset
            if (bytesDownloaded > 0 && response.code == 200) {
                bytesDownloaded = 0
                partialFile.delete()
                Log.d(TAG, "Server doesn't support Range for $modelId, restarting from 0")
            }

            val totalBytes = if (response.code == 206) {
                // Content-Range: bytes start-end/total
                val contentRange = response.header("Content-Range")
                contentRange?.substringAfterLast("/")?.toLongOrNull() ?: manifest.sizeBytes
            } else {
                (response.body?.contentLength() ?: manifest.sizeBytes) + bytesDownloaded
            }

            val body = response.body
            if (body == null) {
                Log.e(TAG, "Empty response body for $modelId")
                response.close()
                autoRestorePreviousVersion(modelDir, modelFilename)
                emit(DownloadProgress(modelId, bytesDownloaded, totalBytes,
                    percentOf(bytesDownloaded, totalBytes), DownloadState.FAILED))
                return@flow
            }

            // Write to partial file (append if resuming)
            val raf = RandomAccessFile(partialFile, "rw")
            raf.seek(bytesDownloaded)

            val buffer = ByteArray(DOWNLOAD_BUFFER_SIZE)
            var lastProgressEmitBytes = bytesDownloaded
            val inputStream = body.byteStream()
            val downloadStartTimeMs = SystemClock.elapsedRealtime()

            try {
                var bytesRead: Int
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    // Check for cancellation
                    if (!currentCoroutineContext().isActive) {
                        saveDownloadMeta(metaFile, bytesDownloaded, manifest)
                        autoRestorePreviousVersion(modelDir, modelFilename)
                        emit(DownloadProgress(modelId, bytesDownloaded, totalBytes,
                            percentOf(bytesDownloaded, totalBytes), DownloadState.CANCELLED))
                        return@flow
                    }

                    raf.write(buffer, 0, bytesRead)
                    bytesDownloaded += bytesRead

                    // Bandwidth throttling — prevent CDN overload when 3000+ devices download simultaneously
                    if (maxBytesPerSecond > 0) {
                        val expectedElapsedMs = (bytesDownloaded * 1000L) / maxBytesPerSecond
                        val actualElapsedMs = SystemClock.elapsedRealtime() - downloadStartTimeMs
                        val delayMs = expectedElapsedMs - actualElapsedMs
                        if (delayMs > 0) {
                            // Cap delay to prevent HTTP connection timeout (OkHttp default read timeout = 10s)
                            delay(delayMs.coerceAtMost(5000L))
                        }
                    }

                    // Emit progress at intervals to avoid flooding
                    if (bytesDownloaded - lastProgressEmitBytes >= PROGRESS_EMIT_INTERVAL_BYTES) {
                        saveDownloadMeta(metaFile, bytesDownloaded, manifest)
                        emit(DownloadProgress(modelId, bytesDownloaded, totalBytes,
                            percentOf(bytesDownloaded, totalBytes), DownloadState.DOWNLOADING))
                        lastProgressEmitBytes = bytesDownloaded
                    }
                }
            } finally {
                raf.close()
                inputStream.close()
                response.close()
            }

            // Save final progress
            saveDownloadMeta(metaFile, bytesDownloaded, manifest)

            // Verify checksum
            emit(DownloadProgress(modelId, bytesDownloaded, totalBytes,
                percentOf(bytesDownloaded, totalBytes), DownloadState.VERIFYING))

            if (verifyChecksum(partialFile, manifest.checksumSha256)) {
                // Rename partial to final
                partialFile.renameTo(completeFile)
                metaFile.delete()
                Log.i(TAG, "Download complete and verified: $modelId (${bytesDownloaded / (1024 * 1024)} MB)")

                // Download companion files — failures degrade to text-only (no vision), don't block primary model
                for (companion in manifest.companionFiles) {
                    val companionFile = File(modelDir, companion.filename)
                    if (companionFile.exists() && verifyChecksum(companionFile, companion.checksumSha256)) {
                        Log.d(TAG, "Companion file already exists and verified: ${companion.filename}")
                        continue
                    }
                    try {
                        Log.i(TAG, "Downloading companion file: ${companion.filename} (${companion.sizeBytes / (1024 * 1024)} MB)")
                        downloadCompanionFileAsync(companion, companionFile)
                        Log.i(TAG, "Companion file downloaded: ${companion.filename}")
                    } catch (e: Exception) {
                        Log.w(TAG, "Companion file download failed (model will work in text-only mode): ${companion.filename} - ${e.message}")
                        // Don't fail the whole download — primary model is still usable without companion
                    }
                }

                emit(DownloadProgress(modelId, bytesDownloaded, totalBytes, 100f, DownloadState.COMPLETE))
            } else {
                // Checksum mismatch — delete and report failure
                Log.e(TAG, "Checksum mismatch for $modelId, deleting corrupt download")
                partialFile.delete()
                metaFile.delete()
                autoRestorePreviousVersion(modelDir, modelFilename)
                emit(DownloadProgress(modelId, bytesDownloaded, totalBytes,
                    percentOf(bytesDownloaded, totalBytes), DownloadState.FAILED))
            }
        } catch (e: CancellationException) {
            saveDownloadMeta(metaFile, bytesDownloaded, manifest)
            Log.i(TAG, "Download cancelled for $modelId at $bytesDownloaded bytes")
            autoRestorePreviousVersion(modelDir, modelFilename)
            emit(DownloadProgress(modelId, bytesDownloaded, manifest.sizeBytes,
                percentOf(bytesDownloaded, manifest.sizeBytes), DownloadState.CANCELLED))
        } catch (e: Exception) {
            saveDownloadMeta(metaFile, bytesDownloaded, manifest)
            Log.e(TAG, "Download error for $modelId: ${e.message}", e)
            autoRestorePreviousVersion(modelDir, modelFilename)
            emit(DownloadProgress(modelId, bytesDownloaded, manifest.sizeBytes,
                percentOf(bytesDownloaded, manifest.sizeBytes), DownloadState.FAILED))
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Check if a model is already downloaded and its checksum-validated binary
     * exists on disk.
     *
     * @param modelId Model identifier
     * @return true if the complete model binary exists
     */
    fun isModelAvailable(modelId: String): Boolean {
        val modelDir = File(modelsDir, modelId)
        if (!modelDir.exists()) return false
        // Check all supported extensions
        return modelDir.listFiles()?.any {
            it.isFile && it.length() > 0 && (
                it.name == MODEL_FILENAME ||
                it.name.endsWith(".litertlm") ||
                it.name.endsWith(".onnx")
            )
        } == true
    }

    /**
     * Get the filesystem path to a downloaded model binary.
     *
     * @param modelId Model identifier
     * @return File pointing to the model binary, or null if not downloaded
     */
    fun getModelPath(modelId: String): File? {
        val modelDir = File(modelsDir, modelId)
        if (!modelDir.exists()) return null
        // Find the model file with any supported extension
        return modelDir.listFiles()?.firstOrNull {
            it.isFile && it.length() > 0 &&
                // Exclude companion files (mmproj) — they're not the main model
                !it.name.startsWith("mmproj") && (
                it.name == MODEL_FILENAME ||
                it.name.endsWith(".litertlm") ||
                it.name.endsWith(".onnx")
            )
        }
    }

    /**
     * Delete a downloaded model to free disk space.
     *
     * @param modelId Model identifier
     * @return true if the model directory was successfully deleted
     */
    fun deleteModel(modelId: String): Boolean {
        // Cancel any in-progress download first
        cancelDownload(modelId)
        val modelDir = File(modelsDir, modelId)
        if (!modelDir.exists()) return true
        val deleted = modelDir.deleteRecursively()
        if (deleted) {
            Log.i(TAG, "Deleted model: $modelId")
        } else {
            Log.w(TAG, "Failed to fully delete model directory: $modelId")
        }
        return deleted
    }

    /**
     * Get total disk space used by all downloaded models in megabytes.
     *
     * @return Disk usage in MB
     */
    fun getTotalDiskUsageMb(): Float {
        if (!modelsDir.exists()) return 0f
        var totalBytes = 0L
        modelsDir.walkTopDown().filter { it.isFile }.forEach { totalBytes += it.length() }
        return totalBytes / (1024f * 1024f)
    }

    /**
     * Cancel an in-progress download for a model.
     * The partial file and metadata are preserved so the download
     * can be resumed later.
     *
     * @param modelId Model identifier
     */
    fun cancelDownload(modelId: String) {
        val job = activeDownloads.remove(modelId)
        if (job != null) {
            job.cancel()
            Log.i(TAG, "Cancelled download for $modelId")
        }
    }

    /**
     * Register a download job so it can be cancelled later.
     * Called by consumers that launch the download flow in a coroutine scope.
     *
     * @param modelId Model identifier
     * @param job The coroutine Job running the download flow collection
     */
    fun registerDownloadJob(modelId: String, job: Job) {
        activeDownloads[modelId] = job
    }

    /**
     * Unregister a download job after it completes (success, failure, or cancellation).
     *
     * @param modelId Model identifier
     */
    fun unregisterDownloadJob(modelId: String) {
        activeDownloads.remove(modelId)
    }

    /**
     * List all model IDs that have been fully downloaded.
     *
     * @return List of model ID strings
     */
    fun listDownloadedModels(): List<String> {
        if (!modelsDir.exists()) return emptyList()
        return modelsDir.listFiles()
            ?.filter { it.isDirectory && File(it, MODEL_FILENAME).exists() }
            ?.map { it.name }
            ?: emptyList()
    }

    /**
     * Check if a partial download exists for a model (interrupted previous attempt).
     *
     * @param modelId Model identifier
     * @return true if a partial download exists that can be resumed
     */
    fun hasPartialDownload(modelId: String): Boolean {
        val partialFile = File(modelsDir, "$modelId/$PARTIAL_FILENAME")
        val metaFile = File(modelsDir, "$modelId/$META_FILENAME")
        return partialFile.exists() && metaFile.exists() && partialFile.length() > 0
    }

    // ---- Private helpers ----

    /**
     * Persist download progress metadata so downloads can be resumed after
     * app restarts or network interruptions.
     */
    private fun saveDownloadMeta(metaFile: File, bytesDownloaded: Long, manifest: ModelManifest) {
        try {
            val meta = JSONObject().apply {
                put("modelId", manifest.modelId)
                put("bytesDownloaded", bytesDownloaded)
                put("totalBytes", manifest.sizeBytes)
                put("checksumSha256", manifest.checksumSha256)
                put("downloadUrl", manifest.downloadUrl)
                put("modelFormat", manifest.modelFormat)
                put("lastUpdated", System.currentTimeMillis())
            }
            metaFile.writeText(meta.toString())
        } catch (e: Exception) {
            Log.w(TAG, "Failed to save download metadata: ${e.message}")
        }
    }

    /**
     * Verify a file's SHA-256 checksum against an expected hex string.
     *
     * @param file The file to verify
     * @param expectedSha256 Expected SHA-256 hex string (lowercase)
     * @return true if the checksum matches
     */
    private fun verifyChecksum(file: File, expectedSha256: String): Boolean {
        if (expectedSha256.isBlank()) {
            Log.w(TAG, "No checksum provided, skipping verification for ${file.name}")
            return true
        }
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(DOWNLOAD_BUFFER_SIZE)
            FileInputStream(file).use { fis ->
                var bytesRead: Int
                while (fis.read(buffer).also { bytesRead = it } != -1) {
                    digest.update(buffer, 0, bytesRead)
                }
            }
            val actualHash = digest.digest().joinToString("") { "%02x".format(it) }
            val matches = actualHash.equals(expectedSha256, ignoreCase = true)
            if (!matches) {
                Log.e(TAG, "Checksum mismatch: expected=$expectedSha256, actual=$actualHash")
            }
            matches
        } catch (e: Exception) {
            Log.e(TAG, "Checksum verification failed: ${e.message}", e)
            false
        }
    }

    /**
     * Calculate percentage completed safely (avoids division by zero).
     */
    private fun percentOf(downloaded: Long, total: Long): Float {
        if (total <= 0) return 0f
        return (downloaded.toFloat() / total.toFloat() * 100f).coerceIn(0f, 100f)
    }

    /**
     * Download a companion file asynchronously with a 5-minute timeout.
     * Uses the same resumable download + checksum validation as the primary model.
     * Wraps blocking I/O in Dispatchers.IO so it doesn't block the calling coroutine.
     *
     * @param companion The companion file descriptor
     * @param outputFile Target file on disk
     * @throws TimeoutCancellationException if download exceeds 5 minutes
     */
    private suspend fun downloadCompanionFileAsync(companion: CompanionFile, outputFile: File) {
        withTimeout(5 * 60 * 1000L) { // 5 minute timeout
            withContext(Dispatchers.IO) {
                val partialFile = File(outputFile.parent, "${outputFile.name}.part")
                var bytesDownloaded = if (partialFile.exists()) partialFile.length() else 0L

                val request = Request.Builder().url(companion.downloadUrl).apply {
                    if (bytesDownloaded > 0) header("Range", "bytes=$bytesDownloaded-")
                }.build()

                val response = httpClient.newCall(request).execute()
                if (!response.isSuccessful) {
                    response.close()
                    throw IllegalStateException("Companion download failed: HTTP ${response.code} for ${companion.filename}")
                }

                val body = response.body
                    ?: throw IllegalStateException("Empty response body for companion ${companion.filename}")

                try {
                    RandomAccessFile(partialFile, "rw").use { raf ->
                        raf.seek(bytesDownloaded)
                        val buffer = ByteArray(DOWNLOAD_BUFFER_SIZE)
                        body.byteStream().use { stream ->
                            var bytesRead: Int
                            while (stream.read(buffer).also { bytesRead = it } != -1) {
                                raf.write(buffer, 0, bytesRead)
                                bytesDownloaded += bytesRead
                            }
                        }
                    }
                } finally {
                    response.close()
                }

                if (companion.checksumSha256.isNotBlank() && !verifyChecksum(partialFile, companion.checksumSha256)) {
                    partialFile.delete()
                    throw IllegalStateException("Companion file checksum mismatch: ${companion.filename}")
                }

                partialFile.renameTo(outputFile)
            }
        }
    }

    /**
     * Retain the previous version of a model for instant rollback.
     * Called automatically after a new version is successfully downloaded.
     */
    private fun retainPreviousVersion(modelDir: File, modelFilename: String) {
        val currentModel = File(modelDir, modelFilename)
        val prevModel = File(modelDir, "$modelFilename.prev")

        // Delete any existing .prev (we only keep 1 previous version)
        if (prevModel.exists()) {
            prevModel.delete()
        }

        // Rename current to .prev before new download replaces it
        if (currentModel.exists()) {
            currentModel.renameTo(prevModel)
            Log.i(TAG, "Retained previous version: ${prevModel.name}")
        }
    }

    /**
     * Auto-restore the previous model version if the current model is missing.
     * Called after download failure or cancellation to ensure the device is never
     * left model-less (the current model was renamed to .prev before download started).
     */
    private fun autoRestorePreviousVersion(modelDir: File, modelFilename: String) {
        val currentFile = File(modelDir, modelFilename)
        val prevFile = File(modelDir, "$modelFilename.prev")
        if (!currentFile.exists() && prevFile.exists()) {
            if (prevFile.renameTo(currentFile)) {
                Log.i(TAG, "Auto-restored previous model version after download failure: ${currentFile.name}")
            } else {
                Log.w(TAG, "Failed to auto-restore previous version: ${prevFile.name}")
            }
        }
    }

    /**
     * Roll back to the previous version of a model.
     * Returns true if rollback succeeded, false if no previous version exists.
     */
    fun rollbackModel(modelId: String): Boolean {
        val modelDir = File(modelsDir, modelId)
        if (!modelDir.exists()) return false

        val modelFiles = modelDir.listFiles { f -> f.name.endsWith(".prev") } ?: return false
        if (modelFiles.isEmpty()) return false

        for (prevFile in modelFiles) {
            val currentName = prevFile.name.removeSuffix(".prev")
            val currentFile = File(modelDir, currentName)

            // Delete the current (bad) version
            if (currentFile.exists()) currentFile.delete()

            // Rename .prev back to current
            prevFile.renameTo(currentFile)
            Log.i(TAG, "Rolled back model $modelId: ${currentFile.name}")
        }
        return true
    }

    /**
     * Clean up old versions to free storage space.
     * Removes all .prev files across all model directories.
     *
     * @return Total bytes freed
     */
    fun cleanupOldVersions(): Long {
        var freedBytes = 0L
        modelsDir.listFiles()?.forEach { modelDir ->
            if (modelDir.isDirectory) {
                modelDir.listFiles { f -> f.name.endsWith(".prev") }?.forEach { prevFile ->
                    freedBytes += prevFile.length()
                    prevFile.delete()
                    Log.i(TAG, "Cleaned up old version: ${prevFile.absolutePath}")
                }
            }
        }
        return freedBytes
    }
}
