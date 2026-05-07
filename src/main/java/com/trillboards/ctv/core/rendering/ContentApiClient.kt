package com.trillboards.ctv.core.rendering

import android.util.Log
import com.trillboards.ctv.core.AgentConfig
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.rendering.models.ContentParser
import com.trillboards.ctv.core.rendering.models.ContentResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class ContentApiClient(private val config: AgentConfig) {

    companion object {
        private const val TAG = "ContentApiClient"
        private val renderCfg get() = SensingConfig.get().rendering
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(renderCfg.contentConnectTimeoutS, TimeUnit.SECONDS)
        .readTimeout(renderCfg.contentReadTimeoutS, TimeUnit.SECONDS)
        .build()

    suspend fun fetchContent(screenId: String): ContentResponse? = withContext(Dispatchers.IO) {
        try {
            val url = "${config.apiBaseUrl}/v2/earner/alot-advertisement-list/$screenId"
            val request = Request.Builder()
                .url(url)
                .get()
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "Content fetch failed: ${response.code}")
                    return@withContext null
                }

                val body = response.body?.string() ?: return@withContext null
                val json = JSONObject(body)
                val etag = response.header("ETag")
                ContentParser.parseContentResponse(json, etag)
            }
        } catch (e: IOException) {
            Log.e(TAG, "Network error fetching content", e)
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing content", e)
            null
        }
    }

    suspend fun fetchContentIfChanged(screenId: String, etag: String?): ContentResponse? =
        withContext(Dispatchers.IO) {
            try {
                val url = "${config.apiBaseUrl}/v2/earner/alot-advertisement-list/$screenId"
                val requestBuilder = Request.Builder().url(url).get()

                if (!etag.isNullOrBlank()) {
                    requestBuilder.header("If-None-Match", etag)
                }

                httpClient.newCall(requestBuilder.build()).execute().use { response ->
                    when (response.code) {
                        304 -> {
                            Log.d(TAG, "Content not modified (304)")
                            null
                        }
                        200 -> {
                            val body = response.body?.string() ?: return@withContext null
                            val json = JSONObject(body)
                            val newEtag = response.header("ETag")
                            ContentParser.parseContentResponse(json, newEtag)
                        }
                        else -> {
                            Log.e(TAG, "Content fetch failed: ${response.code}")
                            null
                        }
                    }
                }
            } catch (e: IOException) {
                Log.e(TAG, "Network error fetching content", e)
                null
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing content", e)
                null
            }
        }
}
