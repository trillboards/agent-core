package com.trillboards.ctv.core.activity

import android.view.View
import android.webkit.WebView

/**
 * Decouples agent-core's `BaseAgentActivity` from each platform's generated `ActivityMainBinding`.
 *
 * Each platform has its own ViewBinding generated from its `activity_main.xml` layout. The base
 * activity wires lifecycle / setup against this interface; the subclass returns a binding adapter
 * that exposes the live binding's children.
 *
 * `touchInterceptorOrNull` is non-null only on tablet (used for kiosk long-press); TV returns null
 * and routes kiosk entry through the MENU key instead.
 */
interface AgentBindingAdapter {
    val webView: WebView
    val blackoutView: View
    val loadingIndicator: View
    val touchInterceptorOrNull: View?
}
