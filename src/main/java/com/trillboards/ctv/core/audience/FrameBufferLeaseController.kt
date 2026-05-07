package com.trillboards.ctv.core.audience

/**
 * Coordinates shared bitmap access between the camera analysis thread and
 * lifecycle teardown. Camera buffers are reused across frames, so teardown must
 * wait until active readers are done before recycling them.
 */
class FrameBufferLeaseController {

    private val lock = Object()
    private var activeLeases = 0
    private var releasing = false

    fun <T> withLease(block: () -> T): T? {
        synchronized(lock) {
            if (releasing) {
                return null
            }
            activeLeases += 1
        }

        return try {
            block()
        } finally {
            synchronized(lock) {
                activeLeases -= 1
                if (activeLeases == 0) {
                    lock.notifyAll()
                }
            }
        }
    }

    fun releaseAndWait(timeoutMs: Long = 2_000L, releaseAction: () -> Unit): Boolean {
        val deadlineAtMs = System.currentTimeMillis() + timeoutMs
        synchronized(lock) {
            releasing = true
            while (activeLeases > 0) {
                val remainingMs = deadlineAtMs - System.currentTimeMillis()
                if (remainingMs <= 0L) {
                    return false
                }
                lock.wait(remainingMs)
            }
        }
        releaseAction()
        return true
    }

    fun reset() {
        synchronized(lock) {
            releasing = false
        }
    }
}
