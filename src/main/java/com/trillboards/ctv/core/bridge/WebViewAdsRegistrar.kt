package com.trillboards.ctv.core.bridge

import android.util.Log
import android.webkit.WebView

/**
 * Registers a WebView with Google's Mobile Ads SDK for the WebView API for Ads.
 *
 * This enables Google's IMA SDK (running inside the WebView) to automatically
 * access native app signals (e.g., device identifiers, consent state) without
 * any custom bridge code on the IMA side.
 *
 * Must be called BEFORE WebView.loadUrl().
 *
 * See: https://developers.google.com/ad-manager/mobile-ads-sdk/android/browser/webview/api-for-ads
 */
object WebViewAdsRegistrar {

    private const val TAG = "WebViewAdsRegistrar"

    /**
     * Register the WebView with Google Mobile Ads SDK.
     * Safe to call even if GMS is unavailable (catches all exceptions).
     */
    fun register(webView: WebView) {
        try {
            com.google.android.gms.ads.MobileAds.registerWebView(webView)
            Log.i(TAG, "WebView registered with Google Mobile Ads SDK")
        } catch (e: Exception) {
            // GMS unavailable (Fire TV AOSP), SDK not found, etc.
            Log.w(TAG, "Failed to register WebView with Mobile Ads SDK: ${e.message}")
        }
    }
}
