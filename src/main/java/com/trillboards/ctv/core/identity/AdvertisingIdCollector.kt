package com.trillboards.ctv.core.identity

import android.content.Context
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

data class AdvertisingIdResult(
    val id: String,
    val type: String = "gaid",
    val isLat: Boolean
)

object AdvertisingIdCollector {

    private const val TAG = "AdvertisingIdCollector"
    private val identityCfg get() = SensingConfig.get().identity
    private val TIMEOUT_MS get() = identityCfg.advertisingIdTimeoutMs
    private val CACHE_TTL_MS get() = identityCfg.advertisingIdCacheTtlMs

    @Volatile
    private var cachedResult: AdvertisingIdResult? = null
    @Volatile
    private var cachedAt: Long = 0L

    suspend fun collect(context: Context): AdvertisingIdResult? {
        // Return cached result if still within TTL
        cachedResult?.let {
            if (System.currentTimeMillis() - cachedAt < CACHE_TTL_MS) return it
        }

        return withContext(Dispatchers.IO) {
            try {
                withTimeoutOrNull(TIMEOUT_MS) {
                    val adInfo = com.google.android.gms.ads.identifier.AdvertisingIdClient
                        .getAdvertisingIdInfo(context.applicationContext)

                    // Never collect when user has opted out (LAT enabled)
                    if (adInfo.isLimitAdTrackingEnabled) {
                        Log.d(TAG, "LAT enabled — not collecting advertising ID")
                        return@withTimeoutOrNull null
                    }

                    val id = adInfo.id
                    if (id.isNullOrBlank() || id == "00000000-0000-0000-0000-000000000000") {
                        Log.d(TAG, "Invalid or zeroed advertising ID")
                        return@withTimeoutOrNull null
                    }

                    val result = AdvertisingIdResult(
                        id = id,
                        type = "gaid",
                        isLat = false
                    )
                    cachedResult = result
                    cachedAt = System.currentTimeMillis()
                    Log.d(TAG, "Collected GAID: ${id.take(8)}...")
                    result
                }
            } catch (e: Exception) {
                // GMS unavailable (Fire TV AOSP without Play Services), timeout, etc.
                Log.w(TAG, "Failed to collect advertising ID: ${e.message}")
                null
            }
        }
    }

    fun clearCache() {
        cachedResult = null
        cachedAt = 0L
    }
}
