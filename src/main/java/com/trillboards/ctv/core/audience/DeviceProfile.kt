package com.trillboards.ctv.core.audience

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import android.util.Size
import com.trillboards.ctv.core.stability.AttenuationThresholds
import com.trillboards.ctv.core.stability.MemoryAttenuationManager.AttenuationTier

/**
 * Device profiling for adaptive audience sensing with capability matrix.
 *
 * Different chipsets have different camera HAL behaviors:
 * - MediaTek: Allocates new buffer per frame (memory-heavy)
 * - Qualcomm: Efficient buffer pool reuse
 * - Exynos: Moderate buffer management
 * - Rockchip RK3588: 6 TOPS NPU, hardware ML acceleration
 *
 * The DeviceCapabilityMatrix is the single source of truth for what models
 * a device runs, at what cadence, and when to shed them under memory pressure.
 * No ad-hoc `if (ramMb > 2048)` checks scattered across the codebase.
 */
object DeviceProfile {
    private const val TAG = "DeviceProfile"

    /**
     * Chipset vendor categories.
     */
    enum class ChipsetVendor {
        MEDIATEK,   // mt6xxx, Dimensity - aggressive memory allocation
        QUALCOMM,   // qcom, sdm, sm - efficient buffer reuse
        EXYNOS,     // exynos - Samsung's ARM chips
        ROCKCHIP,   // rk3xxx - common in signage
        AMLOGIC,    // s905, s912 - common in TV boxes
        UNKNOWN     // fallback to conservative settings
    }

    /**
     * Device capability tier. Determines which ML models run and at what fidelity.
     */
    enum class CapabilityTier(val level: Int) {
        TIER_1(1),  // RK3588 signage: 6 TOPS NPU, 8GB RAM — all models always ON
        TIER_2(2),  // Snapdragon 6xx+: decent GPU, 4-8GB RAM — most models ON
        TIER_3(3),  // MediaTek Helio: no NPU, 4GB RAM — basic models, aggressive shedding
        TIER_4(4)   // Legacy/Lite: display-only, no ML
    }

    /**
     * Defines when a model should be shed under memory pressure.
     * ALWAYS_ON means never shed. OFF means never loaded.
     */
    enum class ModelAvailability {
        ALWAYS_ON,       // Never shed regardless of memory pressure
        SHED_AT_MEDIUM,  // Shed when attenuation reaches MEDIUM tier
        SHED_AT_HIGH,    // Shed when attenuation reaches HIGH tier
        OFF              // Never loaded on this device tier
    }

    /**
     * Complete capability matrix for a device. This is the single source of truth
     * for what a device does — replaces scattered ad-hoc capability checks.
     */
    data class DeviceCapabilityMatrix(
        // Identity
        val tier: CapabilityTier,
        val chipsetVendor: ChipsetVendor,
        val chipsetName: String,
        val description: String,

        // Camera settings
        val targetFps: Int,
        val resolution: Size,

        // ML model availability per attenuation tier
        val mlKitFace: ModelAvailability,
        val yamnetAudio: ModelAvailability,
        val whisperAsr: ModelAvailability,       // Moonshine ASR (replacing Whisper)
        val ferEmotion: ModelAvailability,
        val poseEngagement: ModelAvailability,
        val ageGender: ModelAvailability,
        val gazeTracking: ModelAvailability,

        // Sensor stack
        val ambientLightSensor: Boolean,

        // Gemini Vision capture cadence (minutes between captures)
        val geminiVisionCadenceMinutes: Int,

        // Federated learning
        val federatedLearningEnabled: Boolean,
        val federatedSparseUpload: Boolean,      // true = top 5% only (Tier 3)

        // Memory attenuation thresholds (RAM % remaining triggers)
        val attenuationEnabled: Boolean,
        val attenuationMediumThreshold: Int,      // % RAM free to trigger MEDIUM
        val attenuationHighThreshold: Int,        // % RAM free to trigger HIGH
        val attenuationCriticalThreshold: Int,    // % RAM free to trigger CRITICAL

        // Recovery ladder scope
        val maxRecoveryLevel: Int,               // 1-6, how high the recovery ladder goes

        // Memory threshold before reducing settings (legacy compat)
        val memoryThresholdMb: Int
    ) {
        /**
         * Check if a specific model should be active at the given attenuation tier.
         */
        fun isModelActive(model: String, currentTier: AttenuationTier): Boolean {
            val availability = when (model) {
                "face" -> mlKitFace
                "yamnet" -> yamnetAudio
                "whisper" -> whisperAsr
                "emotion" -> ferEmotion
                "pose" -> poseEngagement
                "ageGender" -> ageGender
                "gaze" -> gazeTracking
                else -> ModelAvailability.OFF
            }
            return when (availability) {
                ModelAvailability.ALWAYS_ON -> true
                ModelAvailability.SHED_AT_MEDIUM -> currentTier.ordinal < AttenuationTier.MEDIUM.ordinal
                ModelAvailability.SHED_AT_HIGH -> currentTier.ordinal < AttenuationTier.HIGH.ordinal
                ModelAvailability.OFF -> false
            }
        }

        /**
         * Convenience: true when attenuation is disabled (e.g., Tier 1 with 8GB RAM).
         */
        val attenuationDisabled: Boolean get() = !attenuationEnabled

        /**
         * Get AttenuationThresholds object for MemoryAttenuationManager.
         */
        val attenuationThresholds: AttenuationThresholds get() = AttenuationThresholds(
            mediumPercent = attenuationMediumThreshold.toFloat(),
            highPercent = attenuationHighThreshold.toFloat(),
            criticalPercent = attenuationCriticalThreshold.toFloat()
        )

        /**
         * Get ModelAvailability for a specific model by name.
         */
        fun getModelAvailability(model: String): ModelAvailability {
            return when (model) {
                "face" -> mlKitFace
                "yamnet" -> yamnetAudio
                "whisper" -> whisperAsr
                "emotion" -> ferEmotion
                "pose" -> poseEngagement
                "ageGender" -> ageGender
                "gaze" -> gazeTracking
                "footTraffic" -> ModelAvailability.SHED_AT_MEDIUM // Foot traffic always sheddable
                else -> ModelAvailability.OFF
            }
        }

        /**
         * Get the ordered list of models to shed under memory pressure.
         * Models are shed in order: least important first.
         */
        fun getShedOrder(): List<String> {
            val shedOrder = mutableListOf<String>()
            // Order: footTraffic → ageGender → gaze → pose → emotion → whisper → yamnet → face
            val models = listOf(
                "ageGender" to ageGender,
                "gaze" to gazeTracking,
                "pose" to poseEngagement,
                "emotion" to ferEmotion,
                "whisper" to whisperAsr,
                "yamnet" to yamnetAudio,
                "face" to mlKitFace
            )
            for ((name, availability) in models) {
                if (availability != ModelAvailability.ALWAYS_ON && availability != ModelAvailability.OFF) {
                    shedOrder.add(name)
                }
            }
            return shedOrder
        }
    }

    /**
     * Legacy Profile for backward compatibility.
     */
    data class Profile(
        val chipsetVendor: ChipsetVendor,
        val chipsetName: String,
        val targetFps: Int,
        val resolution: Size,
        val enableEmotionalEngagement: Boolean,
        val memoryThresholdMb: Int,
        val description: String
    )

    /**
     * Runtime capabilities measured from actual hardware, not regex.
     */
    data class RuntimeCapabilities(
        val totalRamMb: Int,
        val availableRamMb: Int,
        val cpuCoreCount: Int,
        val largeHeapMb: Int,
        val thermalStatus: Int  // PowerManager.THERMAL_STATUS_* (API 29+), -1 if unavailable
    )

    // Cached results
    private var cachedProfile: Profile? = null
    private var cachedCapabilityMatrix: DeviceCapabilityMatrix? = null

    /**
     * Detect runtime capabilities from actual hardware APIs.
     * This provides ground-truth values for tier classification.
     */
    fun detectRuntimeCapabilities(context: Context): RuntimeCapabilities {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager?.getMemoryInfo(memInfo)

        val totalRamMb = (memInfo.totalMem / (1024 * 1024)).toInt()
        val availableRamMb = (memInfo.availMem / (1024 * 1024)).toInt()
        val cpuCoreCount = Runtime.getRuntime().availableProcessors()
        val largeHeapMb = (activityManager?.largeMemoryClass ?: 256)

        val thermalStatus = try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
                pm?.currentThermalStatus ?: -1
            } else {
                -1
            }
        } catch (e: Exception) {
            -1
        }

        return RuntimeCapabilities(
            totalRamMb = totalRamMb,
            availableRamMb = availableRamMb,
            cpuCoreCount = cpuCoreCount,
            largeHeapMb = largeHeapMb,
            thermalStatus = thermalStatus
        )
    }

    /**
     * Compute capability tier from runtime-measured hardware capabilities.
     *
     * This replaces the regex-only approach. A Samsung Galaxy Tab A9 with
     * 3.4GB RAM and 8 cores correctly lands in TIER_2 instead of being
     * penalized to TIER_3 just because it has a MediaTek chipset.
     *
     * Chipset regex is still used as a secondary signal for TIER_1 (RK3588 NPU).
     */
    private fun computeTierFromCapabilities(
        capabilities: RuntimeCapabilities,
        hardware: String,
        board: String
    ): CapabilityTier {
        // TIER_1: RK3588 with 6 TOPS NPU needs special delegate config — chipset regex is required
        if (isRk3588(hardware, board) && capabilities.totalRamMb >= 6144) {
            return CapabilityTier.TIER_1
        }

        // TIER_2: 3GB+ RAM and 4+ cores — capable of full ML stack
        // This correctly promotes Tab A9 (3.4GB, 8 cores) from TIER_3 to TIER_2
        if (capabilities.totalRamMb >= 3072 && capabilities.cpuCoreCount >= 4) {
            return CapabilityTier.TIER_2
        }

        // TIER_3: 2GB+ RAM — basic models with aggressive shedding
        if (capabilities.totalRamMb >= 2048) {
            return CapabilityTier.TIER_3
        }

        // TIER_4: <2GB RAM — display only, no ML
        return CapabilityTier.TIER_4
    }

    /**
     * Detect device capability matrix based on hardware characteristics.
     * This is the preferred method — returns the full capability matrix.
     *
     * Uses runtime capability detection as the PRIMARY signal for tier classification.
     * Chipset regex is used only for TIER_1 NPU-specific config (RK3588).
     */
    fun detectCapabilityMatrix(context: Context): DeviceCapabilityMatrix {
        cachedCapabilityMatrix?.let { return it }

        val hardware = Build.HARDWARE.lowercase()
        val board = Build.BOARD.lowercase()
        val manufacturer = Build.MANUFACTURER.lowercase()
        val ramMb = getTotalRamMb(context)
        val capabilities = detectRuntimeCapabilities(context)
        val runtimeTier = computeTierFromCapabilities(capabilities, hardware, board)

        Log.i(TAG, "Detecting device capability matrix...")
        Log.d(TAG, "Hardware: $hardware, Board: $board, Manufacturer: $manufacturer, RAM: ${ramMb}MB")
        Log.i(TAG, "Runtime: totalRam=${capabilities.totalRamMb}MB, cores=${capabilities.cpuCoreCount}, " +
                "largeHeap=${capabilities.largeHeapMb}MB, thermal=${capabilities.thermalStatus}")
        Log.i(TAG, "Runtime tier: $runtimeTier (chipset regex would give: ${getChipsetOnlyTier(hardware, board, manufacturer)})")

        val vendor = detectVendor(hardware, board, manufacturer)

        val matrix = when (runtimeTier) {
            CapabilityTier.TIER_1 -> createRk3588Matrix(hardware, ramMb)
            CapabilityTier.TIER_2 -> {
                // Use the appropriate Tier 2 constructor based on chipset for description
                when {
                    isQualcommHighEnd(hardware, board) -> createQualcommHighEndMatrix(hardware, ramMb)
                    else -> createTier2Matrix(hardware, vendor, ramMb)
                }
            }
            CapabilityTier.TIER_3 -> createTier3Matrix(hardware, vendor, ramMb)
            CapabilityTier.TIER_4 -> createTier4Matrix(hardware, vendor, ramMb)
        }

        cachedCapabilityMatrix = matrix
        logMatrix(matrix)
        return matrix
    }

    /**
     * What the old regex-only approach would have returned (for logging comparison).
     */
    private fun getChipsetOnlyTier(hardware: String, board: String, manufacturer: String): CapabilityTier {
        return when {
            isRk3588(hardware, board) -> CapabilityTier.TIER_1
            isQualcommHighEnd(hardware, board) -> CapabilityTier.TIER_2
            isQualcomm(hardware, board) || isExynos(hardware, board, manufacturer) -> CapabilityTier.TIER_2
            else -> CapabilityTier.TIER_3  // MediaTek, Unknown — the old misclassification
        }
    }

    /**
     * Detect device profile (legacy method, for backward compatibility).
     */
    fun detect(context: Context): Profile {
        cachedProfile?.let { return it }

        val matrix = detectCapabilityMatrix(context)
        val profile = Profile(
            chipsetVendor = matrix.chipsetVendor,
            chipsetName = matrix.chipsetName,
            targetFps = matrix.targetFps,
            resolution = matrix.resolution,
            enableEmotionalEngagement = matrix.ferEmotion != ModelAvailability.OFF,
            memoryThresholdMb = matrix.memoryThresholdMb,
            description = matrix.description
        )

        cachedProfile = profile
        return profile
    }

    // ---- RK3588 Tier 1: Full ML stack, no attenuation ----

    private fun isRk3588(hardware: String, board: String): Boolean =
        hardware.contains("rk3588") || board.contains("rk3588")

    private fun createRk3588Matrix(hardware: String, ramMb: Int): DeviceCapabilityMatrix =
        DeviceCapabilityMatrix(
            tier = CapabilityTier.TIER_1,
            chipsetVendor = ChipsetVendor.ROCKCHIP,
            chipsetName = hardware,
            description = "RK3588 signage — 6 TOPS NPU, all models always ON",
            targetFps = 12,
            resolution = Size(480, 360),
            mlKitFace = ModelAvailability.ALWAYS_ON,
            yamnetAudio = ModelAvailability.ALWAYS_ON,
            whisperAsr = ModelAvailability.ALWAYS_ON,
            ferEmotion = ModelAvailability.ALWAYS_ON,
            poseEngagement = ModelAvailability.ALWAYS_ON,
            ageGender = ModelAvailability.ALWAYS_ON,
            gazeTracking = ModelAvailability.ALWAYS_ON,
            ambientLightSensor = true,
            geminiVisionCadenceMinutes = 3,
            federatedLearningEnabled = true,
            federatedSparseUpload = false,
            attenuationEnabled = false,  // 8GB RAM, no need for attenuation
            attenuationMediumThreshold = 25,
            attenuationHighThreshold = 15,
            attenuationCriticalThreshold = 10,
            maxRecoveryLevel = 6,
            memoryThresholdMb = if (ramMb > 8192) 1024 else 512
        )

    // ---- Qualcomm Tier 2: Most models ON, shed under pressure ----

    private fun isQualcommHighEnd(hardware: String, board: String): Boolean =
        hardware.contains("sm8") || hardware.contains("sdm8") ||
        hardware.contains("sm7") || hardware.contains("sdm7")

    private fun isQualcomm(hardware: String, board: String): Boolean =
        hardware.contains("qcom") || hardware.contains("sdm") ||
        hardware.contains("sm") || board.contains("msm") ||
        board.contains("sdm") || board.contains("qcom")

    private fun createQualcommHighEndMatrix(hardware: String, ramMb: Int): DeviceCapabilityMatrix =
        DeviceCapabilityMatrix(
            tier = CapabilityTier.TIER_2,
            chipsetVendor = ChipsetVendor.QUALCOMM,
            chipsetName = hardware,
            description = "Qualcomm flagship — efficient buffer pooling",
            targetFps = 10,
            resolution = Size(480, 360),
            mlKitFace = ModelAvailability.ALWAYS_ON,
            yamnetAudio = ModelAvailability.ALWAYS_ON,
            whisperAsr = ModelAvailability.ALWAYS_ON,
            ferEmotion = ModelAvailability.SHED_AT_HIGH,
            poseEngagement = ModelAvailability.SHED_AT_HIGH,
            ageGender = ModelAvailability.SHED_AT_MEDIUM,
            gazeTracking = ModelAvailability.SHED_AT_HIGH,
            ambientLightSensor = true,
            geminiVisionCadenceMinutes = 5,
            federatedLearningEnabled = true,
            federatedSparseUpload = false,
            attenuationEnabled = true,
            attenuationMediumThreshold = 40,
            attenuationHighThreshold = 25,
            attenuationCriticalThreshold = 15,
            maxRecoveryLevel = 6,
            memoryThresholdMb = if (ramMb > 8192) 768 else 384
        )

    // ---- Generic Tier 2: Exynos, mid-range Qualcomm ----

    private fun isExynos(hardware: String, board: String, manufacturer: String): Boolean =
        hardware.contains("exynos") || board.contains("exynos") ||
        (manufacturer.contains("samsung") && hardware.contains("samsungexynos"))

    private fun createTier2Matrix(hardware: String, vendor: ChipsetVendor, ramMb: Int): DeviceCapabilityMatrix =
        DeviceCapabilityMatrix(
            tier = CapabilityTier.TIER_2,
            chipsetVendor = vendor,
            chipsetName = hardware,
            description = "${vendor.name} mid-range — moderate buffer management",
            targetFps = 8,
            resolution = Size(480, 360),
            mlKitFace = ModelAvailability.ALWAYS_ON,
            yamnetAudio = ModelAvailability.ALWAYS_ON,
            whisperAsr = ModelAvailability.ALWAYS_ON,
            ferEmotion = ModelAvailability.SHED_AT_HIGH,
            poseEngagement = ModelAvailability.SHED_AT_HIGH,
            ageGender = ModelAvailability.SHED_AT_MEDIUM,
            gazeTracking = ModelAvailability.SHED_AT_HIGH,
            ambientLightSensor = true,
            geminiVisionCadenceMinutes = 5,
            federatedLearningEnabled = true,
            federatedSparseUpload = false,
            attenuationEnabled = true,
            attenuationMediumThreshold = 40,
            attenuationHighThreshold = 25,
            attenuationCriticalThreshold = 15,
            maxRecoveryLevel = 6,
            memoryThresholdMb = if (ramMb > 8192) 640 else 320
        )

    // ---- Other Rockchip (non-RK3588): Tier 2 or 3 ----

    private fun isRockchip(hardware: String, board: String): Boolean =
        hardware.contains("rk") || board.contains("rk3")

    private fun createRockchipGenericMatrix(hardware: String, ramMb: Int): DeviceCapabilityMatrix {
        // RK3566/RK3568 have smaller NPUs but still decent
        val isMidRange = ramMb >= 4096
        return if (isMidRange) {
            createTier2Matrix(hardware, ChipsetVendor.ROCKCHIP, ramMb)
        } else {
            createTier3Matrix(hardware, ChipsetVendor.ROCKCHIP, ramMb)
        }
    }

    // ---- Amlogic: Tier 2-3 depending on RAM ----

    private fun isAmlogic(hardware: String, board: String): Boolean =
        hardware.contains("amlogic") || board.contains("s905") ||
        board.contains("s912") || board.contains("s922")

    private fun createAmlogicMatrix(hardware: String, ramMb: Int): DeviceCapabilityMatrix {
        return if (ramMb >= 4096) {
            createTier2Matrix(hardware, ChipsetVendor.AMLOGIC, ramMb)
        } else {
            createTier3Matrix(hardware, ChipsetVendor.AMLOGIC, ramMb)
        }
    }

    // ---- MediaTek / Unknown: Tier 3 ----

    private fun isMediaTek(hardware: String, board: String): Boolean =
        hardware.contains("mt") || hardware.contains("mtk") ||
        board.contains("mt") || board.contains("mtk")

    private fun createMediaTekMatrix(hardware: String, ramMb: Int): DeviceCapabilityMatrix =
        createTier3Matrix(hardware, ChipsetVendor.MEDIATEK, ramMb)

    private fun createUnknownMatrix(hardware: String, ramMb: Int): DeviceCapabilityMatrix =
        createTier3Matrix(hardware, ChipsetVendor.UNKNOWN, ramMb)

    private fun createTier3Matrix(hardware: String, vendor: ChipsetVendor, ramMb: Int): DeviceCapabilityMatrix =
        DeviceCapabilityMatrix(
            tier = CapabilityTier.TIER_3,
            chipsetVendor = vendor,
            chipsetName = hardware,
            description = "${vendor.name} budget — basic models, aggressive shedding",
            targetFps = 4,
            resolution = Size(320, 240),
            mlKitFace = ModelAvailability.ALWAYS_ON,
            yamnetAudio = ModelAvailability.ALWAYS_ON,
            whisperAsr = ModelAvailability.SHED_AT_MEDIUM,
            ferEmotion = ModelAvailability.SHED_AT_MEDIUM,
            poseEngagement = ModelAvailability.OFF,
            ageGender = ModelAvailability.OFF,
            gazeTracking = ModelAvailability.OFF,
            ambientLightSensor = true,
            geminiVisionCadenceMinutes = 5,
            federatedLearningEnabled = true,
            federatedSparseUpload = true,  // Top 5% only to save bandwidth
            attenuationEnabled = true,
            attenuationMediumThreshold = 40,
            attenuationHighThreshold = 25,
            attenuationCriticalThreshold = 15,
            maxRecoveryLevel = 5,  // No OTA rollback on budget devices
            memoryThresholdMb = 256
        )

    // ---- Tier 4: Display-only, no ML ----

    private fun createTier4Matrix(hardware: String, vendor: ChipsetVendor, ramMb: Int): DeviceCapabilityMatrix =
        DeviceCapabilityMatrix(
            tier = CapabilityTier.TIER_4,
            chipsetVendor = vendor,
            chipsetName = hardware,
            description = "${vendor.name} lite — display only, no ML",
            targetFps = 0,
            resolution = Size(320, 240),
            mlKitFace = ModelAvailability.OFF,
            yamnetAudio = ModelAvailability.OFF,
            whisperAsr = ModelAvailability.OFF,
            ferEmotion = ModelAvailability.OFF,
            poseEngagement = ModelAvailability.OFF,
            ageGender = ModelAvailability.OFF,
            gazeTracking = ModelAvailability.OFF,
            ambientLightSensor = false,
            geminiVisionCadenceMinutes = 0,
            federatedLearningEnabled = false,
            federatedSparseUpload = false,
            attenuationEnabled = false,
            attenuationMediumThreshold = 40,
            attenuationHighThreshold = 25,
            attenuationCriticalThreshold = 15,
            maxRecoveryLevel = 0,
            memoryThresholdMb = 128
        )

    // ---- Helpers ----

    private fun detectVendor(hardware: String, board: String, manufacturer: String): ChipsetVendor =
        when {
            isMediaTek(hardware, board) -> ChipsetVendor.MEDIATEK
            isQualcomm(hardware, board) -> ChipsetVendor.QUALCOMM
            isExynos(hardware, board, manufacturer) -> ChipsetVendor.EXYNOS
            isRockchip(hardware, board) -> ChipsetVendor.ROCKCHIP
            isAmlogic(hardware, board) -> ChipsetVendor.AMLOGIC
            else -> ChipsetVendor.UNKNOWN
        }

    private fun getTotalRamMb(context: Context): Int {
        return try {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            activityManager?.getMemoryInfo(memInfo)
            (memInfo.totalMem / (1024 * 1024)).toInt()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get RAM size: ${e.message}")
            4096 // Assume 4GB if we can't detect
        }
    }

    private fun logMatrix(matrix: DeviceCapabilityMatrix) {
        Log.i(TAG, "========================================")
        Log.i(TAG, "DEVICE CAPABILITY MATRIX")
        Log.i(TAG, "========================================")
        Log.i(TAG, "Tier: ${matrix.tier} (${matrix.description})")
        Log.i(TAG, "Chipset: ${matrix.chipsetVendor} (${matrix.chipsetName})")
        Log.i(TAG, "FPS: ${matrix.targetFps}, Resolution: ${matrix.resolution.width}x${matrix.resolution.height}")
        Log.i(TAG, "Models: face=${matrix.mlKitFace}, yamnet=${matrix.yamnetAudio}, asr=${matrix.whisperAsr}")
        Log.i(TAG, "        emotion=${matrix.ferEmotion}, pose=${matrix.poseEngagement}, age=${matrix.ageGender}, gaze=${matrix.gazeTracking}")
        Log.i(TAG, "Gemini cadence: ${matrix.geminiVisionCadenceMinutes}min")
        Log.i(TAG, "Federated: enabled=${matrix.federatedLearningEnabled}, sparse=${matrix.federatedSparseUpload}")
        Log.i(TAG, "Attenuation: enabled=${matrix.attenuationEnabled}, thresholds=${matrix.attenuationMediumThreshold}/${matrix.attenuationHighThreshold}/${matrix.attenuationCriticalThreshold}")
        Log.i(TAG, "Recovery: maxLevel=${matrix.maxRecoveryLevel}")
        Log.i(TAG, "Shed order: ${matrix.getShedOrder()}")
        Log.i(TAG, "========================================")
    }

    fun clearCache() {
        cachedProfile = null
        cachedCapabilityMatrix = null
    }

    /**
     * Predicate: does this device have separate (non-combo) Bluetooth and WiFi
     * radios? This gates `SCAN_MODE_LOW_LATENCY` BLE scanning in
     * `BleBeaconScanner` — devices with combo radios (single chip shared
     * between BT + WiFi) can experience WiFi-throughput degradation when BT
     * scans at LOW_LATENCY duty cycle, so they stay on LOW_POWER.
     *
     * Allow-list approach. Known-good split-radio designs in the Trillboards
     * fleet:
     *   - Rockchip RK3588 (TIER_1): external WiFi module on a separate die.
     *   - Qualcomm flagship (sm7/sm8/sdm7/sdm8): FastConnect 6900/7800 with
     *     dedicated BT+WiFi cores.
     *   - Exynos 2200+ TIER_1/TIER_2: Broadcom BCM4389/BCM4398 split cores.
     *   - MediaTek Dimensity 9xxx flagship (mt6989 / mt6991 / mt6993): MT79xx
     *     WiFi companion chip on a separate die. Samsung Galaxy Tab S11
     *     (Build.HARDWARE = mt6991, board = gts11wifi) lands here. Verified
     *     2026-05-23 on live ADB — the Tab S11 is MediaTek-based despite the
     *     planning doc's earlier assumption it was Exynos.
     *
     * Anything not in the allowlist (mid/budget MediaTek combo radios,
     * Amlogic Fire TV combo radios, UNKNOWN, TIER_3/TIER_4 devices) defaults
     * to false — keep LOW_POWER. This is the conservative safe choice from
     * the plan: "If the DeviceCapability check is uncertain, default to
     * LOW_POWER to avoid regression."
     */
    fun hasSeparateBtWifiRadios(matrix: DeviceCapabilityMatrix): Boolean {
        // Only TIER_1 + TIER_2 devices get LOW_LATENCY — TIER_3/TIER_4 are
        // memory-constrained budget devices, almost universally combo radios.
        if (matrix.tier != CapabilityTier.TIER_1 && matrix.tier != CapabilityTier.TIER_2) {
            return false
        }
        return when (matrix.chipsetVendor) {
            // Rockchip RK3588 (TIER_1 only) — external WiFi module, split design.
            ChipsetVendor.ROCKCHIP -> matrix.tier == CapabilityTier.TIER_1
            // Qualcomm flagship FastConnect chips on the high-end Snapdragon
            // SoCs — known split BT + WiFi cores. `chipsetName` is the lower-
            // cased Build.HARDWARE string; the regex helper accepts it for
            // both the hardware and board arguments since we only have one
            // source string at this layer.
            ChipsetVendor.QUALCOMM -> isQualcommHighEnd(matrix.chipsetName, matrix.chipsetName)
            // Exynos 2200+ (recent Samsung phones) use Broadcom BCM4389 /
            // BCM4398 — split BT + WiFi cores.
            ChipsetVendor.EXYNOS -> true
            // MediaTek Dimensity 9xxx flagships (mt6989 / mt6991 / mt6993)
            // ship with MT79xx WiFi companion chips — split radio. Mid/budget
            // MediaTek (Helio, lower Dimensity) stays combo, conservative.
            ChipsetVendor.MEDIATEK -> isMediaTekFlagship(matrix.chipsetName)
            // Amlogic + Unknown vendors: combo radios are the documented
            // default — conservative LOW_POWER.
            ChipsetVendor.AMLOGIC,
            ChipsetVendor.UNKNOWN -> false
        }
    }

    /**
     * Predicate: MediaTek Dimensity 9xxx flagship SoCs that ship with split
     * BT + WiFi radios (MT79xx companion chip on a separate die). The
     * `mt6989` / `mt6991` / `mt6993` family numbers map to Dimensity 9300 /
     * 9400+ / 9500 respectively. Older Helio + lower-Dimensity SKUs stay
     * combo-radio and fall through to false.
     */
    private fun isMediaTekFlagship(hardware: String): Boolean {
        val h = hardware.lowercase()
        return h.contains("mt6989") || h.contains("mt6991") || h.contains("mt6993")
    }
}
