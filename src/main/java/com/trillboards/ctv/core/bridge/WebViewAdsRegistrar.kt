package com.trillboards.ctv.core.bridge

import android.util.Log
import android.webkit.WebView
import com.trillboards.ctv.core.consent.ConsentManager

/**
 * Registers a WebView with Google's Mobile Ads SDK for the WebView API for Ads.
 *
 * This enables Google's IMA SDK (running inside the WebView) to automatically
 * access native app signals (e.g., device identifiers, consent state) without
 * any custom bridge code on the IMA side.
 *
 * Must be called BEFORE WebView.loadUrl().
 *
 * Gated on [ConsentManager.canRegisterWebView]: registerWebView causes the
 * Mobile Ads SDK to enumerate installed apps for ad-targeting, which Play
 * Policy requires preceded by Prominent Disclosure + affirmative consent.
 *
 * See: https://developers.google.com/ad-manager/mobile-ads-sdk/android/browser/webview/api-for-ads
 */
object WebViewAdsRegistrar {

    private const val TAG = "WebViewAdsRegistrar"

    /**
     * Register the WebView with Google Mobile Ads SDK if the user has
     * consented (via UMP). No-op otherwise — IMA inside the WebView falls
     * back to "browser mode" (less measurement / lower CPMs) but the app
     * still functions. Safe to call even if GMS is unavailable (catches
     * all exceptions, including NoClassDefFoundError on Fire TV AOSP).
     */
    fun register(webView: WebView) {
        if (!ConsentManager.canRegisterWebView()) {
            Log.i(TAG, "Skipping registerWebView — consent not granted (UMP gate)")
            return
        }
        try {
            com.google.android.gms.ads.MobileAds.registerWebView(webView)
            Log.i(TAG, "WebView registered with Google Mobile Ads SDK")
        } catch (e: Throwable) {
            // GMS unavailable (Fire TV AOSP), SDK not found, etc.
            Log.w(TAG, "Failed to register WebView with Mobile Ads SDK: ${e.message}")
        }
    }
}
