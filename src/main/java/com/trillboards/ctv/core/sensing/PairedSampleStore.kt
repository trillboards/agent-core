package com.trillboards.ctv.core.sensing

import android.content.Context
import android.util.Log
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * Local store for paired CSI/pose training samples.
 *
 * This is a write-buffer, not long-term storage. Samples sit here only until
 * they are uploaded to the training backend (`POST /v2/sensing/paired-training-samples`),
 * then they are dropped via [clearUploaded].
 *
 * Two implementations exist:
 *   - [RoomPairedSampleStore]: production, backed by Room/SQLite
 *   - [InMemoryPairedSampleStore]: tests, plain map (no Android dependency)
 *
 * The schema and contract are intentionally minimal — the upload pipeline owns
 * the persistence guarantees, this store just survives a process restart.
 */
interface PairedSampleStore {
    /**
     * Insert a sample. Idempotent on `sampleId` (UUID); duplicate IDs are
     * silently ignored. Returns true if a new row was added.
     */
    fun insert(sample: PairedTrainingSample): Boolean

    /**
     * Return up to `limit` unuploaded samples in oldest-first order
     * (sorted by `windowStartMs ASC, sampleId ASC` for stable batching).
     */
    fun getUnuploadedBatch(limit: Int): List<PairedTrainingSample>

    /**
     * Mark the given sample IDs as uploaded. Unknown IDs are silently ignored.
     */
    fun markUploaded(sampleIds: List<String>)

    /**
     * Total number of samples in the store, regardless of upload state.
     */
    fun count(): Int

    /**
     * Number of samples that have not been marked uploaded.
     */
    fun countUnuploaded(): Int

    /**
     * Drop all samples that have been marked uploaded. Unuploaded samples are
     * preserved.
     */
    fun clearUploaded()
}

// ========================================================================
// In-memory implementation (used by unit tests; safe in any JVM context)
// ========================================================================

/**
 * Thread-safe in-memory implementation of [PairedSampleStore].
 *
 * Used by unit tests because Room requires an Android Context that the
 * `unitTests.isReturnDefaultValues = true` test environment cannot provide.
 *
 * The data structures here mirror the Room schema one-to-one, so any test
 * passing against this implementation also passes against the Room one
 * (modulo Android-specific bugs like SQLite parameter limits, which the
 * collector keeps within bounds).
 */
class InMemoryPairedSampleStore : PairedSampleStore {
    private data class Row(val sample: PairedTrainingSample, var uploaded: Boolean)

    private val rows = mutableMapOf<String, Row>()

    @Synchronized
    override fun insert(sample: PairedTrainingSample): Boolean {
        if (rows.containsKey(sample.sampleId)) return false
        rows[sample.sampleId] = Row(sample = sample, uploaded = false)
        return true
    }

    @Synchronized
    override fun getUnuploadedBatch(limit: Int): List<PairedTrainingSample> {
        return rows.values
            .asSequence()
            .filter { !it.uploaded }
            .sortedWith(compareBy({ it.sample.windowStartMs }, { it.sample.sampleId }))
            .take(limit)
            .map { it.sample }
            .toList()
    }

    @Synchronized
    override fun markUploaded(sampleIds: List<String>) {
        for (id in sampleIds) {
            rows[id]?.uploaded = true
        }
    }

    @Synchronized
    override fun count(): Int = rows.size

    @Synchronized
    override fun countUnuploaded(): Int = rows.values.count { !it.uploaded }

    @Synchronized
    override fun clearUploaded() {
        val toRemove = rows.entries.filter { it.value.uploaded }.map { it.key }
        for (id in toRemove) rows.remove(id)
    }
}

// ========================================================================
// Room implementation (production)
// ========================================================================

@Entity(tableName = "paired_training_samples")
data class PairedTrainingSampleEntity(
    @PrimaryKey
    @ColumnInfo(name = "sample_id")
    val sampleId: String,

    @ColumnInfo(name = "csi_window_bytes", typeAffinity = ColumnInfo.BLOB)
    val csiWindowBytes: ByteArray,

    @ColumnInfo(name = "subcarrier_count")
    val subcarrierCount: Int,

    @ColumnInfo(name = "keypoints_bytes", typeAffinity = ColumnInfo.BLOB)
    val keypointsBytes: ByteArray,

    @ColumnInfo(name = "keypoint_confidences_bytes", typeAffinity = ColumnInfo.BLOB)
    val keypointConfidencesBytes: ByteArray,

    @ColumnInfo(name = "overall_confidence")
    val overallConfidence: Float,

    @ColumnInfo(name = "num_camera_frames")
    val numCameraFrames: Int,

    @ColumnInfo(name = "window_start_ms")
    val windowStartMs: Long,

    @ColumnInfo(name = "window_end_ms")
    val windowEndMs: Long,

    @ColumnInfo(name = "screen_mongo_id")
    val screenMongoId: String?,

    @ColumnInfo(name = "venue_type")
    val venueType: String?,

    @ColumnInfo(name = "camera_model_version")
    val cameraModelVersion: String?,

    @ColumnInfo(name = "uploaded")
    val uploaded: Int = 0,

    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis()
)

@Dao
interface PairedSampleDao {
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    fun insert(entity: PairedTrainingSampleEntity): Long

    @Query(
        """
        SELECT * FROM paired_training_samples
        WHERE uploaded = 0
        ORDER BY window_start_ms ASC, sample_id ASC
        LIMIT :limit
        """
    )
    fun getUnuploadedBatch(limit: Int): List<PairedTrainingSampleEntity>

    @Query(
        """
        UPDATE paired_training_samples
        SET uploaded = 1
        WHERE sample_id IN (:ids)
        """
    )
    fun markUploaded(ids: List<String>): Int

    @Query("SELECT COUNT(*) FROM paired_training_samples")
    fun count(): Int

    @Query("SELECT COUNT(*) FROM paired_training_samples WHERE uploaded = 0")
    fun countUnuploaded(): Int

    @Query("DELETE FROM paired_training_samples WHERE uploaded = 1")
    fun clearUploaded(): Int
}

@Database(
    entities = [PairedTrainingSampleEntity::class],
    version = 1,
    exportSchema = false
)
abstract class PairedSampleDatabase : RoomDatabase() {
    abstract fun pairedSampleDao(): PairedSampleDao

    companion object {
        @Volatile
        private var INSTANCE: PairedSampleDatabase? = null

        fun getInstance(context: Context): PairedSampleDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    PairedSampleDatabase::class.java,
                    "paired_training_samples.db"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}

/**
 * Production [PairedSampleStore] backed by Room/SQLite.
 *
 * All DAO calls happen on the caller's thread (the collector already runs
 * on a coroutine dispatcher); Room's per-call lock keeps writes safe.
 */
class RoomPairedSampleStore(context: Context) : PairedSampleStore {

    companion object {
        private const val TAG = "PairedSampleStore"
    }

    private val dao: PairedSampleDao = PairedSampleDatabase.getInstance(context).pairedSampleDao()

    override fun insert(sample: PairedTrainingSample): Boolean {
        return try {
            val rowId = dao.insert(sample.toEntity())
            rowId >= 0L
        } catch (e: Exception) {
            Log.w(TAG, "insert failed for ${sample.sampleId}: ${e.message}")
            false
        }
    }

    override fun getUnuploadedBatch(limit: Int): List<PairedTrainingSample> {
        return try {
            dao.getUnuploadedBatch(limit).map { it.toSample() }
        } catch (e: Exception) {
            Log.w(TAG, "getUnuploadedBatch failed: ${e.message}")
            emptyList()
        }
    }

    override fun markUploaded(sampleIds: List<String>) {
        if (sampleIds.isEmpty()) return
        try {
            // SQLite has a parameter limit (~999 by default). Chunk to be safe.
            sampleIds.chunked(500).forEach { dao.markUploaded(it) }
        } catch (e: Exception) {
            Log.w(TAG, "markUploaded failed: ${e.message}")
        }
    }

    override fun count(): Int = try {
        dao.count()
    } catch (e: Exception) {
        Log.w(TAG, "count failed: ${e.message}")
        0
    }

    override fun countUnuploaded(): Int = try {
        dao.countUnuploaded()
    } catch (e: Exception) {
        Log.w(TAG, "countUnuploaded failed: ${e.message}")
        0
    }

    override fun clearUploaded() {
        try {
            dao.clearUploaded()
        } catch (e: Exception) {
            Log.w(TAG, "clearUploaded failed: ${e.message}")
        }
    }
}

internal fun PairedTrainingSample.toEntity(): PairedTrainingSampleEntity {
    return PairedTrainingSampleEntity(
        sampleId = sampleId,
        csiWindowBytes = csiWindowBytes,
        subcarrierCount = subcarrierCount,
        keypointsBytes = PairedTrainingSample.serializeFloats(keypoints),
        keypointConfidencesBytes = PairedTrainingSample.serializeFloats(keypointConfidences),
        overallConfidence = overallConfidence,
        numCameraFrames = numCameraFrames,
        windowStartMs = windowStartMs,
        windowEndMs = windowEndMs,
        screenMongoId = screenMongoId,
        venueType = venueType,
        cameraModelVersion = cameraModelVersion
    )
}

internal fun PairedTrainingSampleEntity.toSample(): PairedTrainingSample {
    return PairedTrainingSample(
        sampleId = sampleId,
        csiWindowBytes = csiWindowBytes,
        subcarrierCount = subcarrierCount,
        keypoints = PairedTrainingSample.deserializeFloats(keypointsBytes),
        keypointConfidences = PairedTrainingSample.deserializeFloats(keypointConfidencesBytes),
        overallConfidence = overallConfidence,
        numCameraFrames = numCameraFrames,
        windowStartMs = windowStartMs,
        windowEndMs = windowEndMs,
        screenMongoId = screenMongoId,
        venueType = venueType,
        cameraModelVersion = cameraModelVersion
    )
}
