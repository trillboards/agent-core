package com.trillboards.ctv.core.rendering

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.Collections
import java.util.LinkedHashMap
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class ImpressionTracker(
    private val context: Context,
    private val config: AgentConfig,
    private val scope: CoroutineScope
) {
    companion object {
        private const val TAG = "ImpressionTracker"
        private const val PREFS_NAME = "impression_tracker"
        private const val PREF_PENDING_KEY = "pending_impressions"
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
        private val renderCfg get() = SensingConfig.get().rendering
        private val MAX_PENDING get() = renderCfg.maxPendingImpressions
        private val BATCH_SIZE get() = renderCfg.impressionBatchSize
        private val MAX_DEDUP_IDS get() = renderCfg.maxDedupIds
        private val PERSIST_INTERVAL get() = renderCfg.impressionPersistInterval
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(renderCfg.impressionConnectTimeoutS, TimeUnit.SECONDS)
        .readTimeout(renderCfg.impressionReadTimeoutS, TimeUnit.SECONDS)
        .build()

    private val pendingImpressions = ConcurrentLinkedQueue<ImpressionRecord>()

    // Bounded LRU set for deduplication — evicts oldest entries when full
    private val sentImpressionIds: MutableSet<String> = Collections.newSetFromMap(
        object : LinkedHashMap<String, Boolean>(MAX_DEDUP_IDS, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Boolean>?): Boolean {
                return size > MAX_DEDUP_IDS
            }
        }
    )
    // Synchronize access since sentImpressionIds is accessed from IO + Main dispatchers
    private val dedupLock = Any()

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // Debounce persistence — only persist every N enqueues
    private val enqueueCounter = AtomicInteger(0)

    data class ImpressionRecord(
        val impressionId: String,
        val adId: String,
        val screenId: String,
        val fingerprint: String,
        val eventType: String,
        val timestamp: Long = System.currentTimeMillis(),
        val metadata: Map<String, Any> = emptyMap()
    )

    init {
        restorePending()
    }

    fun trackImpression(adId: String, impressionId: String?, screenId: String, fingerprint: String) {
        val impId = impressionId ?: UUID.randomUUID().toString()
        synchronized(dedupLock) {
            if (sentImpressionIds.contains(impId)) {
                Log.d(TAG, "Duplicate impression skipped: $impId")
                return
            }
        }

        val record = ImpressionRecord(
            impressionId = impId,
            adId = adId,
            screenId = screenId,
            fingerprint = fingerprint,
            eventType = "impression"
        )
        enqueue(record)
        fireImpression(record)
    }

    fun trackEvent(
        adId: String,
        impressionId: String?,
        screenId: String,
        fingerprint: String,
        eventType: String,
        metadata: Map<String, Any> = emptyMap()
    ) {
        val record = ImpressionRecord(
            impressionId = impressionId ?: UUID.randomUUID().toString(),
            adId = adId,
            screenId = screenId,
            fingerprint = fingerprint,
            eventType = eventType,
            metadata = metadata
        )
        enqueue(record)
        fireEvent(record)
    }

    fun trackVastRequest(
        screenId: String,
        fingerprint: String,
        sourceName: String,
        vastUrl: String,
        result: String,
        latencyMs: Long
    ) {
        scope.launch(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("screenId", screenId)
                    put("fingerprint", fingerprint)
                    put("source", sourceName)
                    put("vastUrl", vastUrl)
                    put("result", result)
                    put("latencyMs", latencyMs)
                    put("renderingMode", "native")
                    put("timestamp", System.currentTimeMillis())
                }

                val request = Request.Builder()
                    .url("${config.apiBaseUrl}/openrtb/v1/vast-request")
                    .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
                    .build()

                httpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.w(TAG, "VAST request tracking failed: ${response.code}")
                    }
                }
            } catch (e: IOException) {
                Log.w(TAG, "VAST request tracking network error", e)
            }
        }
    }

    fun flushBatch() {
        if (pendingImpressions.isEmpty()) return

        scope.launch(Dispatchers.IO) {
            val batch = mutableListOf<ImpressionRecord>()
            repeat(BATCH_SIZE) {
                pendingImpressions.poll()?.let { batch.add(it) } ?: return@repeat
            }
            if (batch.isEmpty()) return@launch

            try {
                val impressionsArray = JSONArray()
                for (record in batch) {
                    impressionsArray.put(JSONObject().apply {
                        put("impressionId", record.impressionId)
                        put("adId", record.adId)
                        put("screenId", record.screenId)
                        put("fingerprint", record.fingerprint)
                        put("eventType", record.eventType)
                        put("timestamp", record.timestamp)
                        put("renderingMode", "native")
                    })
                }

                val body = JSONObject().apply {
                    put("impressions", impressionsArray)
                }

                val request = Request.Builder()
                    .url("${config.apiBaseUrl}/v1/partner/impressions/batch")
                    .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
                    .build()

                httpClient.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        synchronized(dedupLock) {
                            batch.forEach { sentImpressionIds.add(it.impressionId) }
                        }
                        Log.i(TAG, "Batch sent: ${batch.size} impressions")
                    } else {
                        // Re-enqueue failed batch
                        batch.forEach { pendingImpressions.offer(it) }
                        Log.w(TAG, "Batch failed: ${response.code}")
                    }
                }
            } catch (e: IOException) {
                // Re-enqueue on network failure
                batch.forEach { pendingImpressions.offer(it) }
                Log.w(TAG, "Batch network error — re-enqueued ${batch.size}", e)
            }
        }
    }

    private fun enqueue(record: ImpressionRecord) {
        // Drop oldest if at capacity (O(1) for ConcurrentLinkedQueue.poll)
        while (pendingImpressions.size >= MAX_PENDING) {
            pendingImpressions.poll()
        }
        pendingImpressions.offer(record)

        // Debounce persistence — only write to SharedPreferences every N enqueues
        if (enqueueCounter.incrementAndGet() % PERSIST_INTERVAL == 0) {
            persistPending()
        }
    }

    private fun fireImpression(record: ImpressionRecord) {
        scope.launch(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("adid", record.adId)
                    put("impid", record.impressionId)
                    put("screenId", record.screenId)
                    put("fingerprint", record.fingerprint)
                    put("renderingMode", "native")
                    put("timestamp", record.timestamp)
                }

                val request = Request.Builder()
                    .url("${config.apiBaseUrl}/openrtb/v1/impression")
                    .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
                    .build()

                httpClient.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        synchronized(dedupLock) {
                            sentImpressionIds.add(record.impressionId)
                        }
                        pendingImpressions.remove(record)
                    }
                }
            } catch (e: IOException) {
                Log.d(TAG, "Impression fire-and-forget failed — will retry in batch")
            }
        }
    }

    private fun fireEvent(record: ImpressionRecord) {
        scope.launch(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("adid", record.adId)
                    put("impid", record.impressionId)
                    put("event", record.eventType)
                    put("screenId", record.screenId)
                    put("fingerprint", record.fingerprint)
                    put("renderingMode", "native")
                    put("timestamp", record.timestamp)
                    for ((key, value) in record.metadata) {
                        put(key, value)
                    }
                }

                val request = Request.Builder()
                    .url("${config.apiBaseUrl}/openrtb/v1/event")
                    .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
                    .build()

                httpClient.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        pendingImpressions.remove(record)
                    }
                }
            } catch (e: IOException) {
                Log.d(TAG, "Event fire-and-forget failed — will retry in batch")
            }
        }
    }

    private fun persistPending() {
        try {
            val jsonArray = JSONArray()
            for (record in pendingImpressions.take(MAX_PENDING)) {
                jsonArray.put(JSONObject().apply {
                    put("impressionId", record.impressionId)
                    put("adId", record.adId)
                    put("screenId", record.screenId)
                    put("fingerprint", record.fingerprint)
                    put("eventType", record.eventType)
                    put("timestamp", record.timestamp)
                })
            }
            prefs.edit().putString(PREF_PENDING_KEY, jsonArray.toString()).apply()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to persist pending impressions", e)
        }
    }

    private fun restorePending() {
        try {
            val stored = prefs.getString(PREF_PENDING_KEY, null) ?: return
            val jsonArray = JSONArray(stored)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.optJSONObject(i) ?: continue
                pendingImpressions.offer(
                    ImpressionRecord(
                        impressionId = obj.optString("impressionId"),
                        adId = obj.optString("adId"),
                        screenId = obj.optString("screenId"),
                        fingerprint = obj.optString("fingerprint"),
                        eventType = obj.optString("eventType", "impression"),
                        timestamp = obj.optLong("timestamp", System.currentTimeMillis())
                    )
                )
            }
            if (pendingImpressions.isNotEmpty()) {
                Log.i(TAG, "Restored ${pendingImpressions.size} pending impressions")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to restore pending impressions", e)
        }
    }
}
