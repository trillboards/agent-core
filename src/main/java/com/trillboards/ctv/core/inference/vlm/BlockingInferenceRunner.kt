package com.trillboards.ctv.core.inference.vlm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runInterruptible

/**
 * Runs a blocking VLM SDK call on an interruptible worker thread so coroutine
 * timeout cancellation can actually stop the wait and release the camera/VLM
 * admission gate.
 */
internal suspend fun <T> runBlockingInference(block: () -> T): T {
    return runInterruptible(Dispatchers.IO, block)
}
