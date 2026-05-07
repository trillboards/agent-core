package com.trillboards.ctv.core.inference

import com.google.mediapipe.tasks.core.Delegate
import com.trillboards.ctv.core.audience.DeviceProfile.ChipsetVendor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class DelegateSelectorTest {

    // --- Qualcomm ---

    @Test
    fun `qualcomm vision models get GPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.QUALCOMM, "vision")
        assertEquals(Delegate.GPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.GPU, rec.acceleratorType)
    }

    @Test
    fun `qualcomm audio models get CPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.QUALCOMM, "audio")
        assertEquals(Delegate.CPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.CPU, rec.acceleratorType)
    }

    @Test
    fun `qualcomm pose models get GPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.QUALCOMM, "pose")
        assertEquals(Delegate.GPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
    }

    // --- MediaTek ---

    @Test
    fun `mediatek vision models get GPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.MEDIATEK, "vision")
        assertEquals(Delegate.GPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.GPU, rec.acceleratorType)
    }

    @Test
    fun `mediatek audio models get CPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.MEDIATEK, "audio")
        assertEquals(Delegate.CPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
    }

    // --- Exynos ---

    @Test
    fun `exynos vision models get GPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.EXYNOS, "vision")
        assertEquals(Delegate.GPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.GPU, rec.acceleratorType)
    }

    @Test
    fun `exynos audio models get CPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.EXYNOS, "audio")
        assertEquals(Delegate.CPU, rec.primary)
    }

    // --- Rockchip ---

    @Test
    fun `rockchip always gets CPU delegate for vision`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.ROCKCHIP, "vision")
        assertEquals(Delegate.CPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.CPU, rec.acceleratorType)
    }

    @Test
    fun `rockchip always gets CPU delegate for audio`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.ROCKCHIP, "audio")
        assertEquals(Delegate.CPU, rec.primary)
    }

    // --- Amlogic ---

    @Test
    fun `amlogic always gets CPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.AMLOGIC, "vision")
        assertEquals(Delegate.CPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.CPU, rec.acceleratorType)
    }

    // --- Unknown ---

    @Test
    fun `unknown vendor always gets CPU delegate`() {
        val rec = DelegateSelector.recommend(ChipsetVendor.UNKNOWN, "vision")
        assertEquals(Delegate.CPU, rec.primary)
        assertEquals(Delegate.CPU, rec.fallback)
        assertEquals(DelegateSelector.AcceleratorType.CPU, rec.acceleratorType)
    }

    // --- VLM special case ---

    @Test
    fun `vlm model type always gets GPU first regardless of vendor`() {
        for (vendor in ChipsetVendor.values()) {
            val rec = DelegateSelector.recommend(vendor, "vlm")
            assertEquals("VLM should use GPU for vendor $vendor",
                Delegate.GPU, rec.primary)
            assertEquals("VLM fallback should be CPU for vendor $vendor",
                Delegate.CPU, rec.fallback)
            assertEquals(DelegateSelector.AcceleratorType.GPU, rec.acceleratorType)
        }
    }

    // --- Recommendation always has a reason ---

    @Test
    fun `all recommendations have non-empty reason`() {
        for (vendor in ChipsetVendor.values()) {
            for (modelType in listOf("vision", "audio", "pose", "vlm")) {
                val rec = DelegateSelector.recommend(vendor, modelType)
                assertNotNull("Reason should not be null for $vendor/$modelType", rec.reason)
                assert(rec.reason.isNotEmpty()) {
                    "Reason should not be empty for $vendor/$modelType"
                }
            }
        }
    }

    // --- withFallback ---

    @Test
    fun `withFallback uses primary when it succeeds`() {
        var usedDelegate: Delegate? = null
        DelegateSelector.withFallback(Delegate.GPU, Delegate.CPU) { delegate ->
            usedDelegate = delegate
        }
        assertEquals(Delegate.GPU, usedDelegate)
    }

    @Test
    fun `withFallback falls back to CPU when primary fails`() {
        var usedDelegate: Delegate? = null
        var callCount = 0
        DelegateSelector.withFallback(Delegate.GPU, Delegate.CPU) { delegate ->
            callCount++
            if (delegate == Delegate.GPU) {
                throw RuntimeException("GPU init failed")
            }
            usedDelegate = delegate
        }
        assertEquals(Delegate.CPU, usedDelegate)
        assertEquals(2, callCount)
    }

    @Test(expected = RuntimeException::class)
    fun `withFallback rethrows when both primary and fallback fail`() {
        DelegateSelector.withFallback(Delegate.GPU, Delegate.CPU) { delegate ->
            throw RuntimeException("$delegate init failed")
        }
    }

    @Test(expected = RuntimeException::class)
    fun `withFallback rethrows immediately when primary equals fallback and fails`() {
        var callCount = 0
        DelegateSelector.withFallback(Delegate.CPU, Delegate.CPU) { _ ->
            callCount++
            throw RuntimeException("CPU init failed")
        }
        // Should only be called once since primary == fallback
        assertEquals(1, callCount)
    }
}
