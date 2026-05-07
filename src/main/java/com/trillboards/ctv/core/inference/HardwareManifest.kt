package com.trillboards.ctv.core.inference

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import com.trillboards.ctv.core.SensingConfig
import com.trillboards.ctv.core.audience.DeviceProfile
import com.trillboards.ctv.core.audience.DeviceProfile.ChipsetVendor
import org.json.JSONObject
import java.io.File

/**
 * Hardware manifest for device capability reporting.
 *
 * Provides a comprehensive snapshot of the device's ML inference capabilities
 * including GPU/NPU/DSP availability, recommended model tier, and maximum
 * VLM model size. This data is sent in heartbeat payloads to enable fleet-wide
 * hardware capability tracking and model deployment planning.
 *
 * Detection approach:
 * - SoC model: `Build.SOC_MODEL` (API 31+) → fallback to `/proc/cpuinfo`
 *   `Hardware:` line on lower API. SOC_MODEL is the most reliable signal
 *   because OEMs (esp. Rockchip signage boards) often leave `Build.HARDWARE`
 *   as a stale value like "rk30board" while the true chipset is RK3588.
 * - Chipset vendor: from `Build.SOC_MANUFACTURER` (API 31+) → SOC_MODEL
 *   inference → `Build.HARDWARE/BOARD/MANUFACTURER` strings.
 * - RAM: from ActivityManager.MemoryInfo
 * - GPU: check for /dev/mali* (ARM Mali) or /dev/kgsl* (Qualcomm Adreno)
 * - NPU: SoC-model table (Snapdragon 8 series → Hexagon, Dimensity 9000+/
 *   8000+ Gen 3+ → APU, Tensor G2/G3+ → TPU, Exynos 2200+/2400+ → NPU,
 *   Rockchip RK3588/RK3576/RK3568 → NPU). Falls back to `/dev/npu*` and
 *   `/dev/adsprpc-smd` device-file probing.
 * - GPU delegate support: inferred from vendor + known compatibility.
 *   Once vendor is known (SOC_MODEL fix), QUALCOMM/MEDIATEK/EXYNOS report
 *   TRUE — that's where the previously-reported 0%-TRUE bug lived.
 *
 * Note: File-based hardware detection (/dev/mali*, /dev/kgsl*) may not be
 * readable on all devices due to SELinux policies. The manifest falls back
 * to chipset vendor heuristics when direct probing fails. `/proc/cpuinfo`
 * is publicly readable on every Android API level we ship to.
 */
data class HardwareManifest(
    val chipsetVendor: ChipsetVendor,
    val chipsetName: String,
    val totalRamMb: Int,
    val availableRamMb: Int,
    val hasGpu: Boolean,
    val gpuName: String?,
    val hasNpu: Boolean,
    val npuName: String?,
    val hasDsp: Boolean,
    val gpuDelegateSupported: Boolean,
    val nnapiSupported: Boolean,
    val recommendedModelTier: ModelTier,
    val maxVlmSizeMb: Int,
    // Phase 2 prereq PR 2 — ABI / OS API / display-geometry fields
    // surfaced from Build + DisplayMetrics so the server can persist them
    // alongside the existing MLCapabilities columns. Server-side typed
    // columns + indices were landed in migration 20260503082254.
    //
    // Codex P1 (PR #4540): nullable so that `probeDisplayMetrics` /
    // `probeCpuAbi` can return null when the probe genuinely failed (vs. 0
    // which the previous shape forced even on probe failure). Server-side
    // `safePositiveInt` then treats null as "no data" → COALESCE preserves
    // the prior value rather than the agent overwriting good data with
    // fail-zero defaults.
    val cpuAbi: String? = null,
    val osApiLevel: Int? = null,
    val screenWidthPx: Int? = null,
    val screenHeightPx: Int? = null,
    val densityDpi: Int? = null
) {
    /**
     * Model tier classification based on available device RAM.
     * Determines which VLM/LLM models a device can realistically run.
     */
    enum class ModelTier {
        FLOOR,      // <4GB RAM: Moondream 0.5B, SmolVLM-256M
        STANDARD,   // 4-8GB RAM: Gemma 3n E2B
        PREMIUM     // >8GB RAM: Gemma 3n E2B, Qwen3-VL 2B
    }

    /**
     * Serialize to JSON for heartbeat payload inclusion.
     *
     * Codex P1 (PR #4540): nullable Int probe outputs serialize as
     * JSONObject.NULL when the underlying probe failed. Server-side
     * `safePositiveInt` treats null as "no data" so the dual-write upsert's
     * COALESCE preserves prior valid values across transient probe failures.
     */
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("chipsetVendor", chipsetVendor.name)
            put("chipsetName", chipsetName)
            put("totalRamMb", totalRamMb)
            put("availableRamMb", availableRamMb)
            put("hasGpu", hasGpu)
            put("gpuName", gpuName ?: JSONObject.NULL)
            put("hasNpu", hasNpu)
            put("npuName", npuName ?: JSONObject.NULL)
            put("hasDsp", hasDsp)
            put("gpuDelegateSupported", gpuDelegateSupported)
            put("nnapiSupported", nnapiSupported)
            put("recommendedModelTier", recommendedModelTier.name)
            put("maxVlmSizeMb", maxVlmSizeMb)
            // Phase 2 prereq PR 2 — ABI / OS API / display-geometry. Null
            // signals "probe failed" so the server can preserve prior data.
            put("cpuAbi", cpuAbi ?: JSONObject.NULL)
            put("osApiLevel", osApiLevel ?: JSONObject.NULL)
            put("screenWidthPx", screenWidthPx ?: JSONObject.NULL)
            put("screenHeightPx", screenHeightPx ?: JSONObject.NULL)
            put("densityDpi", densityDpi ?: JSONObject.NULL)
        }
    }

    companion object {
        private const val TAG = "HardwareManifest"

        // OS + app + other models baseline reservation (MB)
        private val OS_RESERVATION_MB: Int get() = SensingConfig.get().hardware.osReservationMb
        // Use half of remaining memory for VLM to leave room for other tasks
        private val VLM_MEMORY_FRACTION: Double get() = SensingConfig.get().hardware.vlmMemoryFraction

        /**
         * Auto-detect hardware manifest from device.
         * Uses DeviceProfile's existing vendor detection plus additional
         * hardware probing for GPU/NPU/DSP presence.
         */
        fun detect(context: Context): HardwareManifest {
            val hardware = Build.HARDWARE.lowercase()
            val board = Build.BOARD.lowercase()
            val manufacturer = Build.MANUFACTURER.lowercase()

            // Resolve SoC model FIRST — it's the most reliable signal and feeds
            // both vendor + NPU probes. On API 31+, `Build.SOC_MODEL` is
            // canonical; on lower API we fall back to `/proc/cpuinfo`'s
            // `Hardware:` line which RockChip / Amlogic firmware populate even
            // when `Build.HARDWARE` is stale (e.g. "rk30board"). Returned in
            // its source casing for `soc_name` persistence; downstream
            // matching helpers lower-case as needed.
            val socModelRaw = probeSocModel()
            val socModel = socModelRaw.lowercase()

            // Use DeviceProfile's existing detection for vendor + capabilities
            val runtimeCaps = DeviceProfile.detectRuntimeCapabilities(context)
            val vendor = detectVendor(hardware, board, manufacturer, socModel)
            // Persist original-cased SOC_MODEL when present so downstream
            // dashboards see "SM8550" rather than "sm8550".
            val chipsetName = resolveChipsetName(hardware, board, socModelRaw)

            // Memory info
            val totalRamMb = runtimeCaps.totalRamMb
            val availableRamMb = runtimeCaps.availableRamMb

            // GPU detection
            val gpuProbe = probeGpu(vendor, hardware)

            // NPU detection (SoC model is the primary signal; vendor + hardware
            // are only consulted when SOC_MODEL doesn't match the table).
            val npuProbe = probeNpu(vendor, hardware, socModel)

            // DSP detection (Qualcomm Hexagon)
            val hasDsp = probeDsp(vendor)

            // GPU delegate support is based on vendor + known compatibility
            val gpuDelegateSupported = isGpuDelegateSupported(vendor, gpuProbe.first)

            // NNAPI support: available on API 27+ with compatible hardware
            val nnapiSupported = Build.VERSION.SDK_INT >= 27 && vendor in setOf(
                ChipsetVendor.QUALCOMM,
                ChipsetVendor.MEDIATEK,
                ChipsetVendor.EXYNOS
            )

            // Model tier based on RAM
            val modelTier = classifyModelTier(totalRamMb)

            // Max VLM size: (totalRam - 1500MB OS reserve) * 0.5
            val maxVlmSizeMb = calculateMaxVlmSize(totalRamMb)

            // Phase 2 prereq PR 2 — ABI / OS API / display-geometry probes.
            // Build.SUPPORTED_ABIS is non-null on every Android API level we
            // ship to (minSdk=26), but defensively fall back to null on the
            // empty-array edge case. Build.VERSION.SDK_INT is always defined.
            //
            // Codex P1 (PR #4540): probeDisplayMetrics returns null on probe
            // failure; we propagate the null end-to-end so the server-side
            // COALESCE preserves prior valid values rather than overwriting
            // them with fail-zero defaults.
            val cpuAbi = probeCpuAbi()
            val osApiLevel: Int = Build.VERSION.SDK_INT
            val displayMetrics = probeDisplayMetrics(context)
            val widthPx: Int? = displayMetrics?.widthPixels?.takeIf { it > 0 }
            val heightPx: Int? = displayMetrics?.heightPixels?.takeIf { it > 0 }
            val dpi: Int? = displayMetrics?.densityDpi?.takeIf { it > 0 }

            val manifest = HardwareManifest(
                chipsetVendor = vendor,
                chipsetName = chipsetName,
                totalRamMb = totalRamMb,
                availableRamMb = availableRamMb,
                hasGpu = gpuProbe.first,
                gpuName = gpuProbe.second,
                hasNpu = npuProbe.first,
                npuName = npuProbe.second,
                hasDsp = hasDsp,
                gpuDelegateSupported = gpuDelegateSupported,
                nnapiSupported = nnapiSupported,
                recommendedModelTier = modelTier,
                maxVlmSizeMb = maxVlmSizeMb,
                cpuAbi = cpuAbi,
                osApiLevel = osApiLevel,
                screenWidthPx = widthPx,
                screenHeightPx = heightPx,
                densityDpi = dpi
            )

            Log.i(TAG, "Hardware manifest detected: vendor=$vendor, chipset=$chipsetName, " +
                "socModel=${socModelRaw.ifBlank { "unknown" }}, " +
                "ram=${totalRamMb}MB, gpu=${gpuProbe.second ?: "none"}, " +
                "npu=${npuProbe.second ?: "none"}, dsp=$hasDsp, " +
                "gpuDelegate=$gpuDelegateSupported, nnapi=$nnapiSupported, " +
                "tier=$modelTier, maxVlm=${maxVlmSizeMb}MB, " +
                "abi=${cpuAbi ?: "unknown"}, api=$osApiLevel, " +
                "display=${widthPx ?: "?"}x${heightPx ?: "?"}@${dpi ?: "?"}dpi")

            return manifest
        }

        /**
         * Probe the SoC model identifier.
         *
         * Resolution order:
         *   1. `Build.SOC_MODEL` (API 31+) when populated and not "unknown".
         *   2. `/proc/cpuinfo` `Hardware:` line — RockChip / Amlogic /
         *      Allwinner firmware populate this with the actual chipset
         *      (e.g. "rk3588") even when `Build.HARDWARE` is "rk30board"
         *      and `Build.SOC_MODEL` is unavailable.
         *
         * Returns trimmed string in its source casing (so dashboards see
         * `"SM8350"` not `"sm8350"`), or empty when no source yields a
         * usable value. Empty signals "unknown" to downstream probes — they
         * fall through to vendor / hardware-string heuristics.
         *
         * `/proc/cpuinfo` is publicly readable on every Android API level we
         * ship to (minSdk=26), no SELinux gate, no permission required.
         */
        internal fun probeSocModel(): String {
            // API 31+ SOC_MODEL — fastest, most reliable.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val socModel = runCatching { Build.SOC_MODEL }.getOrNull()
                if (!socModel.isNullOrBlank() && !socModel.equals("unknown", ignoreCase = true)) {
                    return socModel.trim()
                }
            }
            // /proc/cpuinfo fallback — read first 4 KB (Hardware: line is at
            // top of every modern arm64 cpuinfo dump). Capped read avoids
            // pathological cases on devices with many cores.
            return runCatching {
                File("/proc/cpuinfo").bufferedReader().use { reader ->
                    parseCpuInfoHardwareLine(reader.readText().take(4096))
                }
            }.getOrDefault("")
        }

        /**
         * Parse the `Hardware:` line from a `/proc/cpuinfo` chunk and
         * return the chipset model token in source casing.
         *
         * Examples:
         *   - `"Hardware\t: RK3588"`             → `"RK3588"`
         *   - `"Hardware\t: rockchip,rk3588"`    → `"rk3588"`
         *   - `"Hardware: Qualcomm Technologies, Inc MSM8916"` → `"MSM8916"`
         *   - `"Hardware\t: MT6989\n"`           → `"MT6989"`
         *
         * Returns "" when no `Hardware:` line is present.
         *
         * Extracted as a pure function for unit testability — no Build, no
         * file I/O, no Android dependencies.
         */
        internal fun parseCpuInfoHardwareLine(content: String): String {
            for (rawLine in content.lineSequence()) {
                val line = rawLine.trim()
                if (line.startsWith("Hardware", ignoreCase = true)) {
                    val value = line.substringAfter(':', "").trim()
                    if (value.isEmpty()) continue
                    // "rockchip,rk3588" → "rk3588"
                    val afterComma = value.substringAfterLast(',').trim()
                    // "Qualcomm Technologies, Inc MSM8916" (after comma → "Inc
                    // MSM8916") → take the last whitespace-token.
                    val finalToken = afterComma.substringAfterLast(' ').trim()
                    if (finalToken.isNotBlank()) return finalToken
                }
            }
            return ""
        }

        /**
         * Probe the primary CPU ABI from `Build.SUPPORTED_ABIS[0]`.
         *
         * Returns null in the impossible-but-defended edge case of an empty
         * array (e.g. an emulator built with no native bridge). Server-side
         * `cpu_abi` column tolerates NULL and skips the COALESCE write in
         * that case.
         */
        private fun probeCpuAbi(): String? {
            return runCatching {
                Build.SUPPORTED_ABIS.firstOrNull()?.takeIf { it.isNotBlank() }
            }.getOrNull()
        }

        /**
         * Snapshot the device's primary display metrics.
         *
         * Uses `Resources.getDisplayMetrics()` from the supplied Context.
         * On most CTV deployments this is an Application context, which gives
         * the configuration-aware metrics — matches what the kiosk WebView /
         * native renderer sees. Multi-display devices (foldables, external
         * HDMI) report the default display, which is the display the agent
         * is bound to.
         *
         * Codex P1 (PR #4540): returns NULL when the probe fails (rather
         * than a zero-default DisplayMetrics that would have us emit
         * widthPixels=0/heightPixels=0/densityDpi=0 to the server). The
         * server's `safePositiveInt` would have caught these as fail-zeros
         * anyway, but returning null here makes the failure mode explicit
         * end-to-end.
         */
        private fun probeDisplayMetrics(context: Context): android.util.DisplayMetrics? {
            return runCatching {
                context.resources.displayMetrics
            }.getOrNull()
        }

        /**
         * Detect chipset vendor from Build properties + SoC model.
         * Mirrors DeviceProfile.detectVendor but is accessible without context
         * dependency.
         *
         * Resolution ladder (most-specific first):
         *   1. `Build.SOC_MANUFACTURER` (API 31+) — vendor name, unambiguous.
         *   2. SoC model regex on `socModel` — covers boards where
         *      SOC_MANUFACTURER is "unknown" but SOC_MODEL says "RK3588".
         *      This is THE fix for the audit's gpu_delegate_supported=0% bug:
         *      most active fleet entries were resolving to UNKNOWN here.
         *   3. `Build.HARDWARE`/`Build.BOARD`/`Build.MANUFACTURER` substrings
         *      — last-resort heuristics for legacy devices.
         *
         * @param socModel lower-cased SoC model (from `probeSocModel()`); may
         *                 be empty when neither SOC_MODEL nor `/proc/cpuinfo`
         *                 yielded a usable string.
         */
        @Suppress("ReturnCount")
        internal fun detectVendor(
            hardware: String,
            board: String,
            manufacturer: String,
            socModel: String = ""
        ): ChipsetVendor {
            // Check SOC_MANUFACTURER on API 31+ for the most reliable signal.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val socManufacturer = runCatching { Build.SOC_MANUFACTURER.lowercase() }.getOrDefault("")
                when {
                    socManufacturer.contains("qualcomm") || socManufacturer.contains("qcom") -> return ChipsetVendor.QUALCOMM
                    socManufacturer.contains("mediatek") || socManufacturer.contains("mtk") -> return ChipsetVendor.MEDIATEK
                    socManufacturer.contains("samsung") && (hardware.contains("exynos") || socManufacturer.contains("slsi")) -> return ChipsetVendor.EXYNOS
                    socManufacturer.contains("rockchip") -> return ChipsetVendor.ROCKCHIP
                    socManufacturer.contains("amlogic") -> return ChipsetVendor.AMLOGIC
                    // Google Tensor (Pixel) uses a Samsung-fabbed die with a
                    // Mali GPU — group with EXYNOS for delegate semantics
                    // (both ship Mali-Gxx GPUs that support TFLite GPU
                    // delegate; TPU is reported separately via NPU probe).
                    socManufacturer.contains("google") -> return ChipsetVendor.EXYNOS
                }
            }

            // SOC_MODEL inference — patches the gap where SOC_MANUFACTURER is
            // empty/"unknown" on signage boards but SOC_MODEL is populated.
            val vendorFromSocModel = vendorFromSocModel(socModel)
            if (vendorFromSocModel != ChipsetVendor.UNKNOWN) return vendorFromSocModel

            // Fallback to hardware/board string matching (same logic as DeviceProfile).
            return when {
                hardware.contains("mt") || hardware.contains("mtk") ||
                    board.contains("mt") || board.contains("mtk") -> ChipsetVendor.MEDIATEK
                hardware.contains("qcom") || hardware.contains("sdm") ||
                    hardware.contains("sm") || board.contains("msm") ||
                    board.contains("sdm") || board.contains("qcom") -> ChipsetVendor.QUALCOMM
                hardware.contains("exynos") || board.contains("exynos") ||
                    (manufacturer.contains("samsung") && hardware.contains("samsungexynos")) -> ChipsetVendor.EXYNOS
                hardware.contains("rk") || board.contains("rk3") -> ChipsetVendor.ROCKCHIP
                hardware.contains("amlogic") || board.contains("s905") ||
                    board.contains("s912") || board.contains("s922") -> ChipsetVendor.AMLOGIC
                else -> ChipsetVendor.UNKNOWN
            }
        }

        /**
         * Map a SoC model string to a vendor.
         *
         * Patterns (lower-cased input):
         *   - `sm8…` / `sm7…` / `sm6…` / `sm4…` / `sdm…` / `msm…` / `qcs…` → QUALCOMM
         *   - `mt…` → MEDIATEK (Dimensity / Helio / Kompanio)
         *   - `gs101` / `gs201` / `gs301` / `tensor…` → EXYNOS (Mali GPU)
         *   - `s5e…` / `exynos…` → EXYNOS
         *   - `rk…` → ROCKCHIP
         *   - `s905` / `s912` / `s922` / `s928` → AMLOGIC
         */
        internal fun vendorFromSocModel(socModel: String): ChipsetVendor {
            if (socModel.isBlank()) return ChipsetVendor.UNKNOWN
            return when {
                socModel.startsWith("sm") || socModel.startsWith("sdm") ||
                    socModel.startsWith("msm") || socModel.startsWith("qcs") -> ChipsetVendor.QUALCOMM
                socModel.startsWith("mt") -> ChipsetVendor.MEDIATEK
                // Google Tensor (Pixel `gs101`/`gs201`/`gs301`) uses a
                // Samsung-fabbed die with a Mali GPU — bucket with EXYNOS
                // (Mali GPU = same TFLite GPU delegate path as Samsung).
                socModel.startsWith("gs") || socModel.contains("tensor") -> ChipsetVendor.EXYNOS
                socModel.startsWith("s5e") || socModel.contains("exynos") -> ChipsetVendor.EXYNOS
                socModel.startsWith("rk") -> ChipsetVendor.ROCKCHIP
                // Amlogic SoCs: s905/s912/s922/s928 — TV-box / set-top family.
                // Narrower than `startsWith("s9")` to avoid accidental matches.
                socModel.startsWith("s905") || socModel.startsWith("s912") ||
                    socModel.startsWith("s922") || socModel.startsWith("s928") ->
                    ChipsetVendor.AMLOGIC
                else -> ChipsetVendor.UNKNOWN
            }
        }

        /**
         * Resolve a human-readable chipset name.
         *
         * Uses the previously-probed SoC model first (covers both API 31+
         * `Build.SOC_MODEL` and the `/proc/cpuinfo` fallback path), then
         * falls back to `Build.HARDWARE`/`Build.BOARD`. Keeps source casing
         * so dashboards see `"SM8550"` rather than `"sm8550"`.
         */
        internal fun resolveChipsetName(hardware: String, board: String, socModel: String): String {
            if (socModel.isNotBlank()) return socModel
            return hardware.ifBlank { board }
        }

        /**
         * Probe for GPU presence and identify the GPU name.
         * Checks device files and falls back to vendor heuristics.
         *
         * @return Pair of (hasGpu, gpuName)
         */
        private fun probeGpu(vendor: ChipsetVendor, hardware: String): Pair<Boolean, String?> {
            // Try device file probing (may fail due to SELinux)
            val maliPresent = probeDeviceFiles("/dev/mali")
            if (maliPresent) {
                val gpuName = inferMaliGpuName(vendor, hardware)
                return Pair(true, gpuName)
            }

            val kgslPresent = probeDeviceFiles("/dev/kgsl")
            if (kgslPresent) {
                val gpuName = inferAdrenoGpuName(hardware)
                return Pair(true, gpuName)
            }

            // Fallback: infer GPU presence from vendor (all modern mobile SoCs have GPUs)
            return when (vendor) {
                ChipsetVendor.QUALCOMM -> Pair(true, inferAdrenoGpuName(hardware))
                ChipsetVendor.MEDIATEK -> Pair(true, inferMaliGpuName(vendor, hardware))
                ChipsetVendor.EXYNOS -> Pair(true, inferMaliGpuName(vendor, hardware))
                ChipsetVendor.ROCKCHIP -> Pair(true, "Mali (Rockchip)")
                ChipsetVendor.AMLOGIC -> Pair(true, "Mali (Amlogic)")
                ChipsetVendor.UNKNOWN -> Pair(false, null)
            }
        }

        /**
         * Probe for NPU/AI accelerator presence.
         *
         * Primary signal: `socModel` (table lookup — see [npuFromSocModel]).
         * Secondary: `/dev/npu*`, `/dev/adsprpc-smd` device files (often
         * SELinux-blocked, so best-effort only).
         * Tertiary: `Build.HARDWARE` substring matches — kept for backward
         * compatibility with API <31 devices that don't expose SOC_MODEL
         * and that have well-known HARDWARE values.
         *
         * @return Pair of (hasNpu, npuName)
         */
        internal fun probeNpu(
            vendor: ChipsetVendor,
            hardware: String,
            socModel: String = ""
        ): Pair<Boolean, String?> {
            // Primary: SoC-model table lookup. Most reliable when populated.
            val socNpu = npuFromSocModel(socModel)
            if (socNpu.first) return socNpu

            // Secondary: /dev/npu* device file (RockChip exports this on
            // RK3588/RK3576, but readability depends on SELinux policy).
            val npuDevice = probeDeviceFiles("/dev/npu")
            if (npuDevice) {
                return Pair(true, "NPU (detected via /dev/npu)")
            }

            // Tertiary: vendor + hardware-string heuristics. Same logic as the
            // pre-PR5 code — preserved verbatim for devices that didn't yield
            // a SOC_MODEL and don't expose /dev/npu, so we don't regress
            // already-working detection (mainly modern Snapdragon Sm-prefix
            // hardware strings on API <31 — Tab S9 etc.).
            return when (vendor) {
                ChipsetVendor.QUALCOMM -> {
                    val hexagonPresent = probeDeviceFiles("/dev/adsprpc-smd") ||
                        probeDeviceFiles("/dev/cdsprpc-smd")
                    if (hexagonPresent) {
                        Pair(true, "Qualcomm Hexagon DSP")
                    } else {
                        val isModernSnapdragon = hardware.contains("sm") || hardware.contains("sdm6") ||
                            hardware.contains("sdm7") || hardware.contains("sdm8")
                        if (isModernSnapdragon) {
                            Pair(true, "Qualcomm Hexagon DSP (inferred)")
                        } else {
                            Pair(false, null)
                        }
                    }
                }
                ChipsetVendor.MEDIATEK -> {
                    val isDimensity = hardware.contains("mt689") || hardware.contains("mt699") ||
                        hardware.contains("mt698") || hardware.contains("dimensity")
                    if (isDimensity) {
                        Pair(true, "MediaTek APU")
                    } else {
                        Pair(false, null)
                    }
                }
                ChipsetVendor.ROCKCHIP -> {
                    // Pre-SOC_MODEL fallback. Build.HARDWARE typically reads
                    // "rk30board" on signage boxes — that's the bug PR5 fixes
                    // by making SOC_MODEL the primary signal above. We keep
                    // the substring check for the rare case where SOC_MODEL
                    // is empty but HARDWARE happens to carry a real chipset.
                    if (hardware.contains("rk3588") || hardware.contains("rk3576") ||
                        hardware.contains("rk3568")) {
                        Pair(true, "Rockchip NPU")
                    } else {
                        Pair(false, null)
                    }
                }
                ChipsetVendor.EXYNOS -> {
                    Pair(false, null)
                }
                else -> Pair(false, null)
            }
        }

        /**
         * Map a SoC model to NPU presence + display name.
         *
         * Coverage matrix (lower-cased input):
         *
         * | SoC family | Examples | NPU |
         * |---|---|---|
         * | Snapdragon 8 series  | sm8350 (8 Gen 1), sm8450 (8 Gen 2), sm8550 (8 Gen 3), sm8650 (8 Gen 4) | Hexagon |
         * | Snapdragon 7+ Gen N  | sm7325 (7 Gen 1), sm7475 (7+ Gen 2), sm7550 (7+ Gen 3) | Hexagon |
         * | Snapdragon 6 series  | sm6225, sm6375, sm6450 | Hexagon (lower TOPS) |
         * | Dimensity 9000+      | mt6985 (9200), mt6989 (9300), mt6991 (9400) | APU |
         * | Dimensity 8000+ G3+  | mt6886 (8200), mt6896 (8300), mt6897 (8400) | APU |
         * | Tensor G2+           | gs201 (Tensor G2), gs301 (Tensor G3), gs401 (G4) | TPU |
         * | Exynos 2200+         | s5e9925 (2200), s5e9935 (2300), s5e9945 (2400) | NPU |
         * | RockChip RK3588      | rk3588, rk3588s, rk3588m | NPU (6 TOPS) |
         * | RockChip RK3576      | rk3576 | NPU (6 TOPS) |
         * | RockChip RK3568      | rk3568, rk3566 | NPU (0.8 TOPS) |
         *
         * Returns `(false, null)` when SoC model is empty or unrecognized
         * (callers fall through to /dev/npu and vendor-string heuristics).
         */
        @Suppress("ReturnCount")
        internal fun npuFromSocModel(socModel: String): Pair<Boolean, String?> {
            if (socModel.isBlank()) return Pair(false, null)

            // Snapdragon 8 / 7+ Gen / 6 / 4 — all carry Hexagon. The 4-series
            // (sm4350 etc.) has a smaller Hexagon block but it's still
            // NNAPI-accessible, so we report TRUE. False-positive risk: nil —
            // any Snapdragon labeled with this "sm" prefix has Hexagon.
            if (socModel.startsWith("sm")) {
                val tier = when {
                    socModel.startsWith("sm8") -> "8 series"
                    socModel.startsWith("sm7") -> "7 series"
                    socModel.startsWith("sm6") -> "6 series"
                    socModel.startsWith("sm4") -> "4 series"
                    else -> "modern"
                }
                return Pair(true, "Qualcomm Hexagon ($tier)")
            }
            // Older Qualcomm naming.
            if (socModel.startsWith("sdm") || socModel.startsWith("msm") || socModel.startsWith("qcs")) {
                return Pair(true, "Qualcomm Hexagon")
            }

            // MediaTek Dimensity 9000+ (mt698x, mt699x), 8000+ G3+ (mt6886+).
            // Lower-numbered Helio chips don't carry APU, so we filter.
            if (socModel.startsWith("mt")) {
                // Strip the "mt" prefix and look at the digit block.
                val tail = socModel.removePrefix("mt").take(4)
                val num = tail.toIntOrNull() ?: 0
                return when {
                    // Dimensity 9000 (mt6983), 9200 (mt6985), 9300 (mt6989),
                    // 9400 (mt6991) — all carry APU.
                    num in 6983..6999 -> Pair(true, "MediaTek APU (Dimensity 9xxx)")
                    // Dimensity 8200 (mt6886), 8300 (mt6896), 8400 (mt6897).
                    num in 6886..6899 -> Pair(true, "MediaTek APU (Dimensity 8xxx Gen 3+)")
                    else -> Pair(false, null) // Helio / older Dimensity — no public APU
                }
            }

            // Google Tensor — every shipped variant has the Tensor TPU.
            // gs101 (G1), gs201 (G2), gs301 (G3), gs401 (G4)…
            if (socModel.startsWith("gs") || socModel.contains("tensor")) {
                return Pair(true, "Google Tensor TPU")
            }

            // Samsung Exynos 2200+ — s5e9925 / s5e9935 / s5e9945.
            if (socModel.startsWith("s5e")) {
                val tail = socModel.removePrefix("s5e").take(4)
                val num = tail.toIntOrNull() ?: 0
                if (num >= 9925) {
                    return Pair(true, "Samsung Exynos NPU")
                }
                return Pair(false, null)
            }
            if (socModel.contains("exynos")) {
                // "exynos2200", "exynos 2400" — strip non-digits and compare.
                val numStr = socModel.filter { it.isDigit() }
                val num = numStr.toIntOrNull() ?: 0
                if (num >= 2200) {
                    return Pair(true, "Samsung Exynos NPU")
                }
                return Pair(false, null)
            }

            // RockChip RK3588 / RK3576 / RK3568 / RK3566 — all expose the NPU.
            if (socModel.startsWith("rk")) {
                return when {
                    socModel.startsWith("rk3588") -> Pair(true, "Rockchip NPU (6 TOPS)")
                    socModel.startsWith("rk3576") -> Pair(true, "Rockchip NPU (6 TOPS)")
                    socModel.startsWith("rk3568") || socModel.startsWith("rk3566") ->
                        Pair(true, "Rockchip NPU (0.8 TOPS)")
                    else -> Pair(false, null)
                }
            }

            return Pair(false, null)
        }

        /**
         * Probe for DSP (Digital Signal Processor) -- primarily Qualcomm Hexagon.
         */
        private fun probeDsp(vendor: ChipsetVendor): Boolean {
            if (vendor != ChipsetVendor.QUALCOMM) return false

            // Check for Qualcomm ADSP/CDSP device files
            return probeDeviceFiles("/dev/adsprpc-smd") ||
                probeDeviceFiles("/dev/cdsprpc-smd") ||
                probeDeviceFiles("/dev/adsp")
        }

        /**
         * Check if a device file path exists (or any files matching the prefix).
         * May fail on many devices due to SELinux restrictions, so this is
         * best-effort only.
         */
        private fun probeDeviceFiles(pathPrefix: String): Boolean {
            return try {
                val file = File(pathPrefix)
                if (file.exists()) return true

                // Check for numbered variants (e.g., /dev/mali0, /dev/kgsl-3d0)
                val parent = file.parentFile ?: return false
                val baseName = file.name
                parent.listFiles()?.any { it.name.startsWith(baseName) } ?: false
            } catch (e: SecurityException) {
                // SELinux blocks access -- cannot determine
                false
            } catch (e: Exception) {
                false
            }
        }

        /**
         * Infer Mali GPU model name from vendor and hardware string.
         */
        private fun inferMaliGpuName(vendor: ChipsetVendor, hardware: String): String {
            return when (vendor) {
                ChipsetVendor.MEDIATEK -> {
                    when {
                        hardware.contains("mt699") || hardware.contains("mt689") -> "Mali-G720 (Dimensity)"
                        hardware.contains("mt681") || hardware.contains("mt697") -> "Mali-G77 (Dimensity)"
                        hardware.contains("mt6768") || hardware.contains("mt6769") -> "Mali-G52 (Helio)"
                        else -> "Mali (MediaTek)"
                    }
                }
                ChipsetVendor.EXYNOS -> "Mali (Exynos)"
                ChipsetVendor.ROCKCHIP -> "Mali (Rockchip)"
                ChipsetVendor.AMLOGIC -> "Mali (Amlogic)"
                else -> "Mali"
            }
        }

        /**
         * Infer Adreno GPU model name from hardware string.
         */
        private fun inferAdrenoGpuName(hardware: String): String {
            return when {
                hardware.contains("sm8") -> "Adreno 7xx (Snapdragon 8 Gen)"
                hardware.contains("sm7") -> "Adreno 6xx (Snapdragon 7xx)"
                hardware.contains("sdm845") -> "Adreno 630"
                hardware.contains("sdm7") || hardware.contains("sm6") -> "Adreno 619"
                hardware.contains("sdm6") -> "Adreno 6xx"
                else -> "Adreno (Qualcomm)"
            }
        }

        /**
         * Determine if GPU delegate is supported and reliable for this vendor.
         *
         * This is a conservative check — we only enable GPU delegate where
         * MediaPipe / TFLite GPU inference is known to work reliably. The
         * audit of 2026-05-03 found 0% TRUE in production because almost
         * every device was resolving to UNKNOWN vendor (the SOC_MANUFACTURER
         * fallback didn't catch them). PR5's `vendorFromSocModel` patches
         * that — once the vendor is correctly classified, the existing
         * compatibility table just works.
         */
        internal fun isGpuDelegateSupported(vendor: ChipsetVendor, hasGpu: Boolean): Boolean {
            if (!hasGpu) return false

            return when (vendor) {
                ChipsetVendor.QUALCOMM -> true   // Adreno: well-tested with TFLite GPU delegate
                ChipsetVendor.MEDIATEK -> true   // Mali: works on Dimensity/Helio G-series
                ChipsetVendor.EXYNOS -> true     // Mali (Samsung + Google Tensor): well-tested
                ChipsetVendor.ROCKCHIP -> false  // Mali drivers unreliable for ML on signage boards
                ChipsetVendor.AMLOGIC -> false   // Mali drivers unreliable on TV boxes
                ChipsetVendor.UNKNOWN -> false   // Unknown hardware: don't risk it
            }
        }

        /**
         * Classify the model tier based on total RAM.
         */
        private fun classifyModelTier(totalRamMb: Int): ModelTier {
            val cfg = SensingConfig.get().hardware
            return when {
                totalRamMb < cfg.floorTierMaxRamMb -> ModelTier.FLOOR
                totalRamMb < cfg.standardTierMaxRamMb -> ModelTier.STANDARD
                else -> ModelTier.PREMIUM
            }
        }

        /**
         * Calculate maximum VLM model size this device can handle.
         * Formula: (totalRam - 1500MB OS/app reserve) * 0.5
         * Minimum: 0 (device cannot run VLM models)
         */
        private fun calculateMaxVlmSize(totalRamMb: Int): Int {
            val available = totalRamMb - OS_RESERVATION_MB
            if (available <= 0) return 0
            return (available * VLM_MEMORY_FRACTION).toInt()
        }
    }
}
