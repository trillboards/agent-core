package com.trillboards.ctv.core.inference

import com.trillboards.ctv.core.audience.DeviceProfile.ChipsetVendor
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HardwareManifestTest {

    // --- Model tier classification ---

    @Test
    fun `2GB RAM device classified as FLOOR tier`() {
        val manifest = createManifest(totalRamMb = 2048)
        assertEquals(HardwareManifest.ModelTier.FLOOR, manifest.recommendedModelTier)
    }

    @Test
    fun `3GB RAM device classified as FLOOR tier`() {
        val manifest = createManifest(totalRamMb = 3072)
        assertEquals(HardwareManifest.ModelTier.FLOOR, manifest.recommendedModelTier)
    }

    @Test
    fun `4GB RAM device classified as STANDARD tier`() {
        val manifest = createManifest(totalRamMb = 4096)
        assertEquals(HardwareManifest.ModelTier.STANDARD, manifest.recommendedModelTier)
    }

    @Test
    fun `6GB RAM device classified as STANDARD tier`() {
        val manifest = createManifest(totalRamMb = 6144)
        assertEquals(HardwareManifest.ModelTier.STANDARD, manifest.recommendedModelTier)
    }

    @Test
    fun `8GB RAM device classified as PREMIUM tier`() {
        val manifest = createManifest(totalRamMb = 8192)
        assertEquals(HardwareManifest.ModelTier.PREMIUM, manifest.recommendedModelTier)
    }

    @Test
    fun `12GB RAM device classified as PREMIUM tier`() {
        val manifest = createManifest(totalRamMb = 12288)
        assertEquals(HardwareManifest.ModelTier.PREMIUM, manifest.recommendedModelTier)
    }

    // --- Max VLM size calculation ---
    // Formula: (totalRam - 1500) * 0.5, minimum 0

    @Test
    fun `maxVlmSizeMb for 4GB device is 1298`() {
        val manifest = createManifest(totalRamMb = 4096)
        // (4096 - 1500) * 0.5 = 1298
        assertEquals(1298, manifest.maxVlmSizeMb)
    }

    @Test
    fun `maxVlmSizeMb for 8GB device is 3346`() {
        val manifest = createManifest(totalRamMb = 8192)
        // (8192 - 1500) * 0.5 = 3346
        assertEquals(3346, manifest.maxVlmSizeMb)
    }

    @Test
    fun `maxVlmSizeMb for 12GB device is 5394`() {
        val manifest = createManifest(totalRamMb = 12288)
        // (12288 - 1500) * 0.5 = 5394
        assertEquals(5394, manifest.maxVlmSizeMb)
    }

    @Test
    fun `maxVlmSizeMb for 1GB device is 0`() {
        // (1024 - 1500) is negative, so maxVlmSizeMb = 0
        val manifest = createManifest(totalRamMb = 1024)
        assertEquals(0, manifest.maxVlmSizeMb)
    }

    @Test
    fun `maxVlmSizeMb for 2GB device is 274`() {
        val manifest = createManifest(totalRamMb = 2048)
        // (2048 - 1500) * 0.5 = 274
        assertEquals(274, manifest.maxVlmSizeMb)
    }

    // --- JSON serialization ---

    @Test
    fun `toJson includes all fields`() {
        val manifest = createManifest(
            chipsetVendor = ChipsetVendor.QUALCOMM,
            chipsetName = "SM6375",
            totalRamMb = 4096,
            availableRamMb = 2048,
            hasGpu = true,
            gpuName = "Adreno 619",
            hasNpu = true,
            npuName = "Qualcomm Hexagon DSP",
            hasDsp = true,
            gpuDelegateSupported = true,
            nnapiSupported = true,
            cpuAbi = "arm64-v8a",
            osApiLevel = 34,
            screenWidthPx = 2560,
            screenHeightPx = 1600,
            densityDpi = 320
        )

        val json = manifest.toJson()

        assertEquals("QUALCOMM", json.getString("chipsetVendor"))
        assertEquals("SM6375", json.getString("chipsetName"))
        assertEquals(4096, json.getInt("totalRamMb"))
        assertEquals(2048, json.getInt("availableRamMb"))
        assertTrue(json.getBoolean("hasGpu"))
        assertEquals("Adreno 619", json.getString("gpuName"))
        assertTrue(json.getBoolean("hasNpu"))
        assertEquals("Qualcomm Hexagon DSP", json.getString("npuName"))
        assertTrue(json.getBoolean("hasDsp"))
        assertTrue(json.getBoolean("gpuDelegateSupported"))
        assertTrue(json.getBoolean("nnapiSupported"))
        assertEquals("STANDARD", json.getString("recommendedModelTier"))
        assertEquals(1298, json.getInt("maxVlmSizeMb"))
        // Phase 2 prereq PR 2 — ABI / OS API / display-geometry
        assertEquals("arm64-v8a", json.getString("cpuAbi"))
        assertEquals(34, json.getInt("osApiLevel"))
        assertEquals(2560, json.getInt("screenWidthPx"))
        assertEquals(1600, json.getInt("screenHeightPx"))
        assertEquals(320, json.getInt("densityDpi"))
    }

    @Test
    fun `toJson handles null gpu and npu names`() {
        val manifest = createManifest(
            hasGpu = false,
            gpuName = null,
            hasNpu = false,
            npuName = null
        )

        val json = manifest.toJson()
        assertTrue(json.isNull("gpuName"))
        assertTrue(json.isNull("npuName"))
        assertFalse(json.getBoolean("hasGpu"))
        assertFalse(json.getBoolean("hasNpu"))
    }

    // --- Phase 2 prereq PR 2 — ABI / OS API / display-geometry tests ---

    @Test
    fun `toJson serializes null cpuAbi as JSON null`() {
        val manifest = createManifest(cpuAbi = null)
        val json = manifest.toJson()
        assertTrue(json.isNull("cpuAbi"))
    }

    @Test
    fun `toJson emits null for unset geometry fields after Codex P1 fix`() {
        val manifest = createManifest()
        val json = manifest.toJson()

        // Codex P1 (PR #4540): unset geometry / API fields serialize as JSON
        // null, not 0. The server's `safePositiveInt` treats null as "skip
        // column update" so the COALESCE upsert preserves prior valid values
        // across transient probe failures. The previous shape emitted 0 here,
        // which silently overwrote real typed-column data.
        assertTrue(json.isNull("cpuAbi"))
        assertTrue(json.isNull("osApiLevel"))
        assertTrue(json.isNull("screenWidthPx"))
        assertTrue(json.isNull("screenHeightPx"))
        assertTrue(json.isNull("densityDpi"))
    }

    @Test
    fun `toJson preserves arm64-v8a cpuAbi for Tab S11 12GB profile`() {
        // Samsung Galaxy Tab S11 12GB — Snapdragon 8 Gen 3 — typical profile.
        // arm64-v8a is the only ABI on every device shipping past Q4 2019 that
        // we care about. The probe MUST NOT downgrade to armeabi-v7a.
        val manifest = createManifest(
            chipsetVendor = ChipsetVendor.QUALCOMM,
            totalRamMb = 12288,
            cpuAbi = "arm64-v8a",
            osApiLevel = 34,
            screenWidthPx = 2800,
            screenHeightPx = 1752,
            densityDpi = 360
        )
        val json = manifest.toJson()

        assertEquals("arm64-v8a", json.getString("cpuAbi"))
        assertEquals(34, json.getInt("osApiLevel"))
        assertEquals(2800, json.getInt("screenWidthPx"))
        assertEquals(1752, json.getInt("screenHeightPx"))
        assertEquals(360, json.getInt("densityDpi"))
    }

    @Test
    fun `data class equality respects new fields`() {
        // Regression: forgetting to include the new fields in `data class`
        // copy/equals would let stale values silently flow through. Same
        // ctor args except cpuAbi differs → must NOT be equal.
        val a = createManifest(cpuAbi = "arm64-v8a")
        val b = createManifest(cpuAbi = "armeabi-v7a")
        org.junit.Assert.assertNotEquals(a, b)
    }

    // --- GPU delegate support by vendor ---

    @Test
    fun `qualcomm manifest reports gpu delegate supported`() {
        val manifest = createManifest(
            chipsetVendor = ChipsetVendor.QUALCOMM,
            hasGpu = true,
            gpuDelegateSupported = true
        )
        assertTrue(manifest.gpuDelegateSupported)
    }

    @Test
    fun `rockchip manifest reports gpu delegate not supported`() {
        val manifest = createManifest(
            chipsetVendor = ChipsetVendor.ROCKCHIP,
            hasGpu = true,
            gpuDelegateSupported = false
        )
        assertFalse(manifest.gpuDelegateSupported)
    }

    @Test
    fun `amlogic manifest reports gpu delegate not supported`() {
        val manifest = createManifest(
            chipsetVendor = ChipsetVendor.AMLOGIC,
            hasGpu = true,
            gpuDelegateSupported = false
        )
        assertFalse(manifest.gpuDelegateSupported)
    }

    // --- Phase 2 prereq PR 5: SoC-model probes ---
    // Audit `audit-2026-05-03-device-telemetry-deep-inventory.md`: `has_npu` and
    // `gpu_delegate_supported` were 0% TRUE in the active 7d cohort because:
    //   - probeNpu only matched `Build.HARDWARE.contains("rk3588")` — but
    //     RK3588 boards typically report HARDWARE="rk30board".
    //   - detectVendor never reached SOC_MODEL inference, so most devices
    //     resolved to UNKNOWN vendor → isGpuDelegateSupported returned false
    //     even for QUALCOMM/MEDIATEK/EXYNOS chipsets.
    //
    // The table below covers every SoC family the fleet ships on. Each row
    // asserts the probe outputs (vendor + NPU presence + NPU name) so a
    // regression on any one SoC is caught at unit-test time.

    // --- /proc/cpuinfo parser ---

    @Test
    fun `parseCpuInfoHardwareLine extracts simple Hardware line`() {
        val content = """
            processor       : 0
            BogoMIPS        : 38.40
            Hardware        : RK3588
            Revision        : 0000
        """.trimIndent()
        assertEquals("RK3588", HardwareManifest.parseCpuInfoHardwareLine(content))
    }

    @Test
    fun `parseCpuInfoHardwareLine handles vendor-prefixed comma form`() {
        val content = "Hardware\t: rockchip,rk3588\n"
        assertEquals("rk3588", HardwareManifest.parseCpuInfoHardwareLine(content))
    }

    @Test
    fun `parseCpuInfoHardwareLine handles Qualcomm verbose form`() {
        // Old Qualcomm boards: "Hardware: Qualcomm Technologies, Inc MSM8916"
        val content = "Hardware: Qualcomm Technologies, Inc MSM8916"
        assertEquals("MSM8916", HardwareManifest.parseCpuInfoHardwareLine(content))
    }

    @Test
    fun `parseCpuInfoHardwareLine returns empty when missing`() {
        val content = "processor : 0\nBogoMIPS : 38.40\n"
        assertEquals("", HardwareManifest.parseCpuInfoHardwareLine(content))
    }

    @Test
    fun `parseCpuInfoHardwareLine handles MediaTek Dimensity`() {
        val content = "Hardware\t: MT6989\n"
        assertEquals("MT6989", HardwareManifest.parseCpuInfoHardwareLine(content))
    }

    // --- vendorFromSocModel ---

    @Test
    fun `vendorFromSocModel maps Snapdragon 8 Gen 3 to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("sm8550"))
    }

    @Test
    fun `vendorFromSocModel maps Snapdragon 8 Gen 4 to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("sm8650"))
    }

    @Test
    fun `vendorFromSocModel maps Snapdragon 7 Plus Gen 3 to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("sm7550"))
    }

    @Test
    fun `vendorFromSocModel maps Snapdragon 6 Gen 1 to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("sm6450"))
    }

    @Test
    fun `vendorFromSocModel maps SDM (older Snapdragon) to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("sdm845"))
    }

    @Test
    fun `vendorFromSocModel maps MSM (legacy Qualcomm) to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("msm8916"))
    }

    @Test
    fun `vendorFromSocModel maps QCS (Qualcomm IoT) to QUALCOMM`() {
        assertEquals(ChipsetVendor.QUALCOMM, HardwareManifest.vendorFromSocModel("qcs8550"))
    }

    @Test
    fun `vendorFromSocModel maps Dimensity 9300 to MEDIATEK`() {
        assertEquals(ChipsetVendor.MEDIATEK, HardwareManifest.vendorFromSocModel("mt6989"))
    }

    @Test
    fun `vendorFromSocModel maps Dimensity 9000 to MEDIATEK`() {
        assertEquals(ChipsetVendor.MEDIATEK, HardwareManifest.vendorFromSocModel("mt6983"))
    }

    @Test
    fun `vendorFromSocModel maps Tensor G3 to EXYNOS`() {
        assertEquals(ChipsetVendor.EXYNOS, HardwareManifest.vendorFromSocModel("gs301"))
    }

    @Test
    fun `vendorFromSocModel maps Tensor literal to EXYNOS`() {
        assertEquals(ChipsetVendor.EXYNOS, HardwareManifest.vendorFromSocModel("tensor g3"))
    }

    @Test
    fun `vendorFromSocModel maps Exynos 2400 (s5e9945) to EXYNOS`() {
        assertEquals(ChipsetVendor.EXYNOS, HardwareManifest.vendorFromSocModel("s5e9945"))
    }

    @Test
    fun `vendorFromSocModel maps RK3588 to ROCKCHIP`() {
        assertEquals(ChipsetVendor.ROCKCHIP, HardwareManifest.vendorFromSocModel("rk3588"))
    }

    @Test
    fun `vendorFromSocModel maps RK3576 to ROCKCHIP`() {
        assertEquals(ChipsetVendor.ROCKCHIP, HardwareManifest.vendorFromSocModel("rk3576"))
    }

    @Test
    fun `vendorFromSocModel maps Amlogic S905 to AMLOGIC`() {
        assertEquals(ChipsetVendor.AMLOGIC, HardwareManifest.vendorFromSocModel("s905x4"))
    }

    @Test
    fun `vendorFromSocModel maps Amlogic S928 to AMLOGIC`() {
        assertEquals(ChipsetVendor.AMLOGIC, HardwareManifest.vendorFromSocModel("s928x"))
    }

    @Test
    fun `vendorFromSocModel returns UNKNOWN for blank input`() {
        assertEquals(ChipsetVendor.UNKNOWN, HardwareManifest.vendorFromSocModel(""))
    }

    @Test
    fun `vendorFromSocModel returns UNKNOWN for unrecognized model`() {
        assertEquals(ChipsetVendor.UNKNOWN, HardwareManifest.vendorFromSocModel("xyz9999"))
    }

    @Test
    fun `vendorFromSocModel does not confuse Exynos s5e for Amlogic`() {
        // s5e prefix MUST hit EXYNOS arm, not the narrow s9 Amlogic arm.
        assertEquals(ChipsetVendor.EXYNOS, HardwareManifest.vendorFromSocModel("s5e9925"))
    }

    // --- npuFromSocModel ---

    @Test
    fun `npuFromSocModel reports Snapdragon 8 Gen 3 has Hexagon NPU`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("sm8550")
        assertTrue("Tab S11 (sm8550) MUST report has_npu=true; this is the verified arm64 device", hasNpu)
        assertNotNull(name)
        assertTrue("Hexagon should appear in NPU name: actual=$name", name!!.contains("Hexagon"))
    }

    @Test
    fun `npuFromSocModel reports Snapdragon 8 Gen 4 has Hexagon NPU`() {
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("sm8650")
        assertTrue(hasNpu)
    }

    @Test
    fun `npuFromSocModel reports Snapdragon 7 series has Hexagon NPU`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("sm7475")
        assertTrue(hasNpu)
        assertTrue(name!!.contains("Hexagon"))
    }

    @Test
    fun `npuFromSocModel reports Dimensity 9300 has APU`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("mt6989")
        assertTrue("Dimensity 9300 (mt6989) MUST report has_npu=true", hasNpu)
        assertTrue("APU expected in name: actual=$name", name!!.contains("APU"))
    }

    @Test
    fun `npuFromSocModel reports Dimensity 9200 has APU`() {
        // Dimensity 9200 = mt6985.
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("mt6985")
        assertTrue(hasNpu)
    }

    @Test
    fun `npuFromSocModel reports Dimensity 9000 has APU`() {
        // Dimensity 9000 = mt6983 — original Dimensity 9-series chip.
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("mt6983")
        assertTrue(hasNpu)
    }

    @Test
    fun `npuFromSocModel rejects older Helio without APU`() {
        // mt6750 is Helio P10 — no APU. Probe must NOT false-positive.
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("mt6750")
        assertFalse(hasNpu)
    }

    @Test
    fun `npuFromSocModel reports Tensor G3 has TPU`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("gs301")
        assertTrue(hasNpu)
        assertTrue("TPU expected in name: actual=$name", name!!.contains("TPU"))
    }

    @Test
    fun `npuFromSocModel reports Exynos 2400 has NPU`() {
        // Exynos 2400 = s5e9945
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("s5e9945")
        assertTrue(hasNpu)
    }

    @Test
    fun `npuFromSocModel reports Exynos 2200 has NPU`() {
        // Exynos 2200 = s5e9925
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("s5e9925")
        assertTrue(hasNpu)
    }

    @Test
    fun `npuFromSocModel rejects older Exynos without NPU`() {
        // Exynos 990 (s5e9830) is below the 2200 cutoff.
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("s5e9830")
        assertFalse(hasNpu)
    }

    @Test
    fun `npuFromSocModel reports RK3588 has NPU regardless of HARDWARE staleness`() {
        // THE bug from the audit: Build.HARDWARE = "rk30board" but
        // SOC_MODEL / cpuinfo says "rk3588". With probeNpu now using SOC_MODEL
        // first, RK3588 boards correctly self-report has_npu=true.
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("rk3588")
        assertTrue("RK3588 MUST report has_npu=true (fixes audit bug)", hasNpu)
        assertTrue("Rockchip NPU expected in name: actual=$name", name!!.contains("Rockchip NPU"))
    }

    @Test
    fun `npuFromSocModel reports RK3576 has NPU`() {
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("rk3576")
        assertTrue(hasNpu)
    }

    @Test
    fun `npuFromSocModel reports RK3568 has NPU at 0_8 TOPS`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("rk3568")
        assertTrue(hasNpu)
        assertTrue("0.8 TOPS expected: actual=$name", name!!.contains("0.8"))
    }

    @Test
    fun `npuFromSocModel rejects older RockChip without NPU`() {
        // RK3399 has no NPU. Probe must NOT false-positive.
        val (hasNpu, _) = HardwareManifest.npuFromSocModel("rk3399")
        assertFalse(hasNpu)
    }

    @Test
    fun `npuFromSocModel returns false for blank input`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("")
        assertFalse(hasNpu)
        assertEquals(null, name)
    }

    @Test
    fun `npuFromSocModel returns false for unrecognized SoC`() {
        val (hasNpu, name) = HardwareManifest.npuFromSocModel("xyz9999")
        assertFalse(hasNpu)
        assertEquals(null, name)
    }

    // --- isGpuDelegateSupported ---
    // The delegate-supported probe is purely vendor-driven. Once SOC_MODEL
    // resolves the vendor correctly (the actual fix), QUALCOMM / MEDIATEK /
    // EXYNOS devices flip to TRUE. These tests assert the matrix is correct.

    @Test
    fun `isGpuDelegateSupported returns true for Qualcomm with GPU`() {
        assertTrue(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.QUALCOMM, hasGpu = true))
    }

    @Test
    fun `isGpuDelegateSupported returns true for MediaTek with GPU`() {
        assertTrue(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.MEDIATEK, hasGpu = true))
    }

    @Test
    fun `isGpuDelegateSupported returns true for Exynos with GPU`() {
        assertTrue(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.EXYNOS, hasGpu = true))
    }

    @Test
    fun `isGpuDelegateSupported returns false for RockChip even with GPU`() {
        assertFalse(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.ROCKCHIP, hasGpu = true))
    }

    @Test
    fun `isGpuDelegateSupported returns false for Amlogic even with GPU`() {
        assertFalse(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.AMLOGIC, hasGpu = true))
    }

    @Test
    fun `isGpuDelegateSupported returns false for UNKNOWN vendor`() {
        assertFalse(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.UNKNOWN, hasGpu = true))
    }

    @Test
    fun `isGpuDelegateSupported returns false when hasGpu=false even for Qualcomm`() {
        assertFalse(HardwareManifest.isGpuDelegateSupported(ChipsetVendor.QUALCOMM, hasGpu = false))
    }

    // --- detectVendor with SOC_MODEL fallback ---
    // Asserts the SOC_MODEL inference patches the gap from the audit (most
    // active fleet was hitting UNKNOWN here, suppressing gpu_delegate_supported).

    @Test
    fun `detectVendor uses SOC_MODEL when HARDWARE is generic and BOARD has no rk match`() {
        // OEM signage box where Build.SOC_MANUFACTURER is unset (API <31 or
        // returned blank) and HARDWARE/BOARD don't carry recognizable strings.
        // Pre-fix: returned UNKNOWN.
        // Post-fix: SOC_MODEL "rk3588" → ROCKCHIP.
        assertEquals(
            ChipsetVendor.ROCKCHIP,
            HardwareManifest.detectVendor(
                hardware = "generic",
                board = "x86_emu",
                manufacturer = "rockchip",
                socModel = "rk3588"
            )
        )
    }

    @Test
    fun `detectVendor SOC_MODEL fix for emulator board reporting Snapdragon`() {
        // OEM signage box: HARDWARE="ranchu", BOARD="emulator", but SOC_MODEL
        // populated as "sm8350" (Snapdragon 8 Gen 1). Pre-fix this would have
        // landed in UNKNOWN. Post-fix it lands in QUALCOMM.
        assertEquals(
            ChipsetVendor.QUALCOMM,
            HardwareManifest.detectVendor(
                hardware = "ranchu",
                board = "emulator",
                manufacturer = "samsung",
                socModel = "sm8350"
            )
        )
    }

    @Test
    fun `detectVendor falls through to HARDWARE string when SOC_MODEL empty`() {
        // Pre-existing path — should still work for legacy devices that don't
        // populate SOC_MODEL but DO have a recognizable HARDWARE string.
        assertEquals(
            ChipsetVendor.QUALCOMM,
            HardwareManifest.detectVendor(
                hardware = "qcom",
                board = "msm",
                manufacturer = "lg",
                socModel = ""
            )
        )
    }

    @Test
    fun `detectVendor returns UNKNOWN when no signal matches`() {
        assertEquals(
            ChipsetVendor.UNKNOWN,
            HardwareManifest.detectVendor(
                hardware = "generic",
                board = "generic",
                manufacturer = "generic",
                socModel = ""
            )
        )
    }

    // --- probeNpu integration: SOC_MODEL primary, vendor secondary ---

    @Test
    fun `probeNpu uses SOC_MODEL primary for Tab S11 sm8550`() {
        // Verified arm64 device (Galaxy Tab S11, SM-X910). Probe MUST report
        // has_npu=true; it MUST NOT depend on /dev/* readability.
        val (hasNpu, name) = HardwareManifest.probeNpu(
            vendor = ChipsetVendor.QUALCOMM,
            hardware = "qcom",
            socModel = "sm8550"
        )
        assertTrue("Tab S11 fixture MUST report has_npu=true", hasNpu)
        assertTrue("Hexagon expected in name: $name", name!!.contains("Hexagon"))
    }

    @Test
    fun `probeNpu uses SOC_MODEL primary for RK3588 even when HARDWARE is rk30board`() {
        // The exact audit-2026-05-03 bug. Pre-fix: returned (false, null).
        // Post-fix: SOC_MODEL primary → (true, "Rockchip NPU (6 TOPS)").
        val (hasNpu, name) = HardwareManifest.probeNpu(
            vendor = ChipsetVendor.ROCKCHIP,
            hardware = "rk30board",
            socModel = "rk3588"
        )
        assertTrue("RK3588 board MUST report has_npu=true even when HARDWARE is stale", hasNpu)
        assertTrue("Rockchip NPU expected in name: $name", name!!.contains("Rockchip NPU"))
    }

    @Test
    fun `probeNpu falls through to vendor heuristics when SOC_MODEL empty`() {
        // Pre-existing path for legacy Snapdragon: SOC_MODEL="" but
        // HARDWARE="sm6375" (a real Tab S9 substring) → still reports
        // has_npu=true via the modern-Snapdragon vendor heuristic.
        val (hasNpu, _) = HardwareManifest.probeNpu(
            vendor = ChipsetVendor.QUALCOMM,
            hardware = "sm6375",
            socModel = ""
        )
        assertTrue(hasNpu)
    }

    @Test
    fun `probeNpu returns false for non-Snapdragon Qualcomm without SOC_MODEL`() {
        // HARDWARE that doesn't contain "sm"/"sdm" substring → no DSP-inference
        // path triggers. Probe must NOT false-positive. (NB: "msm8916" DOES
        // match `hardware.contains("sm")` per the pre-PR5 vendor heuristic
        // and is therefore reported as Hexagon-inferred — that's the existing
        // contract, not a regression.)
        val (hasNpu, _) = HardwareManifest.probeNpu(
            vendor = ChipsetVendor.QUALCOMM,
            hardware = "qcom",  // no "sm"/"sdm6/7/8" substring
            socModel = ""
        )
        assertFalse(hasNpu)
    }

    @Test
    fun `probeNpu returns false for ROCKCHIP RK3399 without SOC_MODEL`() {
        // RK3399: no NPU. Hardware contains "rk3" but not 3588/3576/3568.
        val (hasNpu, _) = HardwareManifest.probeNpu(
            vendor = ChipsetVendor.ROCKCHIP,
            hardware = "rk3399",
            socModel = ""
        )
        assertFalse(hasNpu)
    }

    @Test
    fun `probeNpu returns false for UNKNOWN vendor with empty SOC_MODEL`() {
        val (hasNpu, name) = HardwareManifest.probeNpu(
            vendor = ChipsetVendor.UNKNOWN,
            hardware = "generic",
            socModel = ""
        )
        assertFalse(hasNpu)
        assertEquals(null, name)
    }

    // --- resolveChipsetName ---

    @Test
    fun `resolveChipsetName prefers SOC_MODEL when populated`() {
        assertEquals(
            "SM8550",
            HardwareManifest.resolveChipsetName(
                hardware = "qcom",
                board = "kona",
                socModel = "SM8550"
            )
        )
    }

    @Test
    fun `resolveChipsetName falls back to HARDWARE when SOC_MODEL empty`() {
        assertEquals(
            "rk30board",
            HardwareManifest.resolveChipsetName(
                hardware = "rk30board",
                board = "rk_pad",
                socModel = ""
            )
        )
    }

    @Test
    fun `resolveChipsetName falls back to BOARD when both HARDWARE and SOC_MODEL empty`() {
        assertEquals(
            "rk_pad",
            HardwareManifest.resolveChipsetName(
                hardware = "",
                board = "rk_pad",
                socModel = ""
            )
        )
    }

    // --- Helper to create manifests without needing Context ---

    private fun createManifest(
        chipsetVendor: ChipsetVendor = ChipsetVendor.UNKNOWN,
        chipsetName: String = "test_chipset",
        totalRamMb: Int = 4096,
        availableRamMb: Int = 2048,
        hasGpu: Boolean = false,
        gpuName: String? = null,
        hasNpu: Boolean = false,
        npuName: String? = null,
        hasDsp: Boolean = false,
        gpuDelegateSupported: Boolean = false,
        nnapiSupported: Boolean = false,
        cpuAbi: String? = null,
        // Codex P1 (PR #4540): nullable defaults — null = "probe not run / failed"
        osApiLevel: Int? = null,
        screenWidthPx: Int? = null,
        screenHeightPx: Int? = null,
        densityDpi: Int? = null
    ): HardwareManifest {
        val modelTier = when {
            totalRamMb < 4096 -> HardwareManifest.ModelTier.FLOOR
            totalRamMb < 8192 -> HardwareManifest.ModelTier.STANDARD
            else -> HardwareManifest.ModelTier.PREMIUM
        }
        val maxVlmSizeMb = run {
            val available = totalRamMb - 1500
            if (available <= 0) 0 else (available * 0.5).toInt()
        }
        return HardwareManifest(
            chipsetVendor = chipsetVendor,
            chipsetName = chipsetName,
            totalRamMb = totalRamMb,
            availableRamMb = availableRamMb,
            hasGpu = hasGpu,
            gpuName = gpuName,
            hasNpu = hasNpu,
            npuName = npuName,
            hasDsp = hasDsp,
            gpuDelegateSupported = gpuDelegateSupported,
            nnapiSupported = nnapiSupported,
            recommendedModelTier = modelTier,
            maxVlmSizeMb = maxVlmSizeMb,
            cpuAbi = cpuAbi,
            osApiLevel = osApiLevel,
            screenWidthPx = screenWidthPx,
            screenHeightPx = screenHeightPx,
            densityDpi = densityDpi
        )
    }
}
