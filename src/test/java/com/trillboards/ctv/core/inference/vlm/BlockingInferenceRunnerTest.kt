package com.trillboards.ctv.core.inference.vlm

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.TimeUnit
import kotlin.system.measureNanoTime

class BlockingInferenceRunnerTest {

    @Test
    fun `runBlockingInference returns blocking result`() = runBlocking {
        val result = runBlockingInference {
            Thread.sleep(25)
            "ok"
        }

        assertEquals("ok", result)
    }

    @Test
    fun `runBlockingInference respects coroutine timeout`() = runBlocking {
        val elapsedMs = measureNanoTime {
            val result = withTimeoutOrNull(75) {
                runBlockingInference {
                    Thread.sleep(5_000)
                    "too-late"
                }
            }

            assertNull(result)
        }.let { TimeUnit.NANOSECONDS.toMillis(it) }

        assertTrue("timeout should stop the wait quickly, elapsed=${elapsedMs}ms", elapsedMs < 1_000L)
    }
}
