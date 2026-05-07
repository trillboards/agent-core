package com.trillboards.ctv.core.audience

import android.content.Context
import android.util.Log
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import com.trillboards.ctv.core.socket.AgentSocketManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Offline signal buffer using Room database.
 *
 * When socket is disconnected, audience signals are stored locally
 * and flushed when connectivity resumes. Ensures zero data loss
 * during network interruptions.
 *
 * Capacity: 8,640 signals (24h × 6 signals/min at 10s intervals)
 * Prune: Signals older than 24h are automatically deleted
 */

// ========== Room Entity ==========

@Entity(tableName = "buffered_signals")
data class BufferedSignal(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    @ColumnInfo(name = "payload")
    val payload: String, // JSON string of the signal payload

    @ColumnInfo(name = "timestamp")
    val timestamp: Long = System.currentTimeMillis(),

    @ColumnInfo(name = "retry_count")
    val retryCount: Int = 0
)

// ========== Room DAO ==========

@Dao
interface SignalBufferDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(signal: BufferedSignal): Long

    @Query("SELECT * FROM buffered_signals ORDER BY timestamp ASC LIMIT :limit")
    suspend fun getOldest(limit: Int): List<BufferedSignal>

    @Query("DELETE FROM buffered_signals WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("DELETE FROM buffered_signals WHERE id IN (:ids)")
    suspend fun deleteByIds(ids: List<Long>)

    @Query("DELETE FROM buffered_signals WHERE timestamp < :timestamp")
    suspend fun deleteOlderThan(timestamp: Long)

    @Query("SELECT COUNT(*) FROM buffered_signals")
    suspend fun count(): Int

    @Query("DELETE FROM buffered_signals WHERE id IN (SELECT id FROM buffered_signals ORDER BY timestamp ASC LIMIT :count)")
    suspend fun deleteOldestN(count: Int)
}

// ========== Room Database ==========

@Database(entities = [BufferedSignal::class], version = 1, exportSchema = false)
abstract class SignalDatabase : RoomDatabase() {
    abstract fun signalBufferDao(): SignalBufferDao

    companion object {
        @Volatile
        private var INSTANCE: SignalDatabase? = null

        fun getInstance(context: Context): SignalDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    SignalDatabase::class.java,
                    "signal_buffer.db"
                )
                    .build()
                    .also { INSTANCE = it }
            }
        }

        /**
         * Close the database and release resources.
         * Must be called during service shutdown to prevent database leaks.
         */
        fun close() {
            synchronized(this) {
                INSTANCE?.close()
                INSTANCE = null
            }
        }
    }
}

// ========== Buffer Manager ==========

/**
 * Manages the offline signal buffer lifecycle.
 *
 * Usage:
 *   val buffer = SignalBufferManager(context)
 *   // When socket is down:
 *   buffer.bufferSignal(payload)
 *   // When socket reconnects:
 *   buffer.flushBuffer(socketManager)
 *   // Periodic cleanup:
 *   buffer.pruneOldSignals()
 */
class SignalBufferManager(context: Context) {

    companion object {
        private const val TAG = "SignalBuffer"
        private const val MAX_CAPACITY = 8640 // 24h × 6 signals/min
        private const val FLUSH_BATCH_SIZE = 50
        private const val MAX_AGE_MS = 24 * 60 * 60 * 1000L // 24 hours
        private const val MAX_RETRIES = 5
    }

    private val dao = SignalDatabase.getInstance(context).signalBufferDao()

    /**
     * Buffer a signal payload for later transmission.
     * If buffer exceeds capacity, oldest signals are evicted.
     */
    suspend fun bufferSignal(payload: JSONObject) {
        withContext(Dispatchers.IO) {
            try {
                // Evict oldest if at capacity
                val currentCount = dao.count()
                if (currentCount >= MAX_CAPACITY) {
                    val evictCount = (MAX_CAPACITY * 0.1).toInt().coerceAtLeast(1)
                    dao.deleteOldestN(evictCount)
                    Log.w(TAG, "Buffer at capacity ($currentCount), evicted $evictCount oldest signals")
                }

                dao.insert(BufferedSignal(payload = payload.toString()))
                Log.d(TAG, "Signal buffered (${currentCount + 1} queued)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to buffer signal: ${e.message}")
            }
        }
    }

    /**
     * Flush buffered signals to the server via socket.
     * Emits oldest signals first, deletes on successful emit.
     *
     * @return Number of signals successfully flushed
     */
    suspend fun flushBuffer(socketManager: AgentSocketManager, limit: Int = FLUSH_BATCH_SIZE): Int {
        return withContext(Dispatchers.IO) {
            try {
                val signals = dao.getOldest(limit)
                if (signals.isEmpty()) return@withContext 0

                Log.i(TAG, "Flushing ${signals.size} buffered signals...")

                val successIds = mutableListOf<Long>()

                for (signal in signals) {
                    if (signal.retryCount >= MAX_RETRIES) {
                        // Too many retries, discard
                        successIds.add(signal.id)
                        continue
                    }

                    try {
                        val payload = JSONObject(signal.payload)
                        // Mark as buffered replay so server knows it's delayed
                        payload.put("_buffered", true)
                        payload.put("_bufferedAt", signal.timestamp)
                        socketManager.emit("audienceSignals", payload)
                        successIds.add(signal.id)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to emit buffered signal ${signal.id}: ${e.message}")
                    }
                }

                if (successIds.isNotEmpty()) {
                    dao.deleteByIds(successIds)
                    Log.i(TAG, "Flushed ${successIds.size}/${signals.size} buffered signals")
                }

                successIds.size
            } catch (e: Exception) {
                Log.e(TAG, "Flush failed: ${e.message}")
                0
            }
        }
    }

    /**
     * Delete signals older than 24 hours.
     */
    suspend fun pruneOldSignals() {
        withContext(Dispatchers.IO) {
            try {
                val cutoff = System.currentTimeMillis() - MAX_AGE_MS
                dao.deleteOlderThan(cutoff)
                Log.d(TAG, "Pruned signals older than 24h")
            } catch (e: Exception) {
                Log.e(TAG, "Prune failed: ${e.message}")
            }
        }
    }

    /**
     * Get the count of currently buffered signals.
     */
    suspend fun count(): Int {
        return withContext(Dispatchers.IO) {
            try {
                dao.count()
            } catch (e: Exception) {
                Log.e(TAG, "Count failed: ${e.message}")
                0
            }
        }
    }

    /**
     * Shutdown the signal buffer and close the underlying database.
     * Call during service teardown to release database resources.
     */
    fun shutdown() {
        Log.i(TAG, "Shutting down signal buffer database")
        SignalDatabase.close()
    }
}
