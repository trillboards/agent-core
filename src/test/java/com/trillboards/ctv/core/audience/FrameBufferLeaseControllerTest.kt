package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class FrameBufferLeaseControllerTest {

    @Test
    fun `release waits for active lease to complete`() {
        val controller = FrameBufferLeaseController()
        val enteredLease = CountDownLatch(1)
        val releaseLease = CountDownLatch(1)
        val releaseCompleted = AtomicBoolean(false)

        val worker = Thread {
            controller.withLease {
                enteredLease.countDown()
                assertTrue(releaseLease.await(2, TimeUnit.SECONDS))
            }
        }
        worker.start()

        assertTrue(enteredLease.await(2, TimeUnit.SECONDS))

        val releaser = Thread {
            val released = controller.releaseAndWait(timeoutMs = 2_000L) {
                releaseCompleted.set(true)
            }
            assertTrue(released)
        }
        releaser.start()

        Thread.sleep(150L)
        assertTrue("release should still be blocked by the active lease", !releaseCompleted.get())

        releaseLease.countDown()

        worker.join(2_000L)
        releaser.join(2_000L)

        assertTrue(releaseCompleted.get())
    }

    @Test
    fun `new leases are rejected while release is in progress and allowed again after reset`() {
        val controller = FrameBufferLeaseController()

        controller.releaseAndWait(timeoutMs = 100L) { }

        assertNull(controller.withLease { "should_not_run" })

        controller.reset()

        assertEquals("ok", controller.withLease { "ok" })
    }

    @Test
    fun `release timeout does not recycle buffers while a lease is still active`() {
        val controller = FrameBufferLeaseController()
        val enteredLease = CountDownLatch(1)
        val releaseLease = CountDownLatch(1)
        val releaseCompleted = AtomicBoolean(false)

        val worker = Thread {
            controller.withLease {
                enteredLease.countDown()
                assertTrue(releaseLease.await(2, TimeUnit.SECONDS))
            }
        }
        worker.start()

        assertTrue(enteredLease.await(2, TimeUnit.SECONDS))

        val released = controller.releaseAndWait(timeoutMs = 100L) {
            releaseCompleted.set(true)
        }

        assertFalse(released)
        assertFalse(releaseCompleted.get())

        releaseLease.countDown()
        worker.join(2_000L)

        controller.reset()
        assertEquals("ok", controller.withLease { "ok" })
    }
}
