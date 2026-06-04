plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.kapt")
    `maven-publish`
    // kotlinx.serialization compiler plugin — required by the Kotlin DTOs
    // emitted from `:agent-core:generateOpenApiTypes` (see below). The DTOs
    // carry `@Serializable` + `@SerialName` annotations to map snake_case
    // JSON keys onto camelCase Kotlin properties for the conformance tests.
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.24"
    // OpenAPI Kotlin codegen — emits DTOs from the merged partner OpenAPI spec
    // into `src/generated/kotlin/com/trillboards/api/types/`. Single source of
    // truth: `trillboard-api/docs/openapi/merged/partner-api.json` (regenerated
    // from the Zod registry by `scripts/build-openapi.js`).
    //
    // Plugin version 7.10.0 — last 7.x release; 7.x line is mature for Kotlin
    // jvm generation and avoids the 8.x KSP-only churn.
    id("org.openapi.generator") version "7.10.0"
}

android {
    namespace = "com.trillboards.ctv.core"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        targetSdk = 35
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

// Maven publication coordinate for the Trillboards Sensing SDK.
// The artifact is published to GitHub Packages by `.github/workflows/publish-agent-core.yml`
// when a tag matching `agent-core-v*` is pushed. Partners install via:
//   implementation("com.trillboards:agent-core:<version>")
// CI auths via ${{ secrets.GITHUB_TOKEN }} (same pattern as publish-edge-sdk.yml's GHCR push).
publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.trillboards"
            artifactId = "agent-core"
            version = project.findProperty("agentCoreVersion") as String? ?: "1.0.0"

            afterEvaluate {
                from(components["release"])
            }

            pom {
                name.set("Trillboards Sensing SDK (agent-core)")
                description.set(
                    "Edge AI audience sensing for Android — face detection (FaceXFormer), " +
                        "audio classification (MediaPipe AudioClassifier), speech recognition " +
                        "(Moonshine ASR), audio diarization, BLE / WiFi / mDNS / SSDP / ARP / " +
                        "HTTP probe / UWB / Auracast / Channel Sounding discovery, native sensors, " +
                        "and Vertex multimodal embeddings. On-device privacy: no PII or images leave " +
                        "the device."
                )
                url.set("https://trillboards.com/support/developers")
                // 2026 is when the Maven publication of this module starts; matches
                // the Apache 2.0 LICENSE copyright year in trillboard-ctv/agent-core/LICENSE.
                // The agent-core source has older history but the public artifact is new.
                inceptionYear.set("2026")
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                developers {
                    developer {
                        id.set("trillboards")
                        name.set("Trillboards Engineering")
                        email.set("engineering@trillboards.com")
                    }
                }
                scm {
                    url.set("https://github.com/trillboards/packages")
                    connection.set("scm:git:https://github.com/trillboards/packages.git")
                    developerConnection.set("scm:git:git@github.com:trillboards/packages.git")
                }
            }
        }
    }
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/trillboards/packages")
            credentials {
                username = System.getenv("GITHUB_ACTOR") ?: project.findProperty("gpr.user") as String?
                password = System.getenv("GITHUB_TOKEN") ?: project.findProperty("gpr.token") as String?
            }
        }
    }
}

dependencies {
    // Multi-protocol device discovery — Phase 1 of the Rust-backed rewrite.
    // The Rust `mdns-sd` core (compiled to per-ABI .so via the rust-android
    // gradle plugin) is shipped as a pre-built AAR at `libs/discovery-core-1.0.0.aar`.
    //
    // Marked `compileOnly` (mirroring the sherpa-onnx pattern below) so the
    // published agent-core POM does NOT declare a runtime dep on a flatDir-only
    // artifact. Partners who want Rust-backed mDNS drop the .aar into their
    // own libs/ + add `flatDir { dirs("libs") }` to their settings.gradle.kts;
    // partners who skip it use Android's NsdManager (the legacy path) and the
    // `MdnsDiscovery` shim falls back gracefully via reflection at first
    // adapter resolution.
    compileOnly("trillboards-flatdir:discovery-core:1.0.0@aar")

    // AndroidX
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")
    // Activity Result APIs + ComponentActivity for BaseAgentActivity (PR 4/5).
    // Pre-PR 4 the per-platform MainActivity files declared these locally; the
    // base class moves the permission-launcher registration into agent-core, so
    // the ComponentActivity superclass + ActivityResultContracts must be on the
    // module classpath here. Same-version pin as the per-platform agents.
    implementation("androidx.activity:activity-ktx:1.8.2")
    // appcompat for AlertDialog (kiosk PIN + Settings dialogs in BaseAgentActivity).
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Networking
    implementation("io.socket:socket.io-client:2.1.0") {
        exclude(group = "org.json", module = "json")
    }
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.11.0")

    // ML Kit for audience measurement (face detection)
    implementation("com.google.mlkit:face-detection:16.1.7")

    // MediaPipe for pose/body engagement detection
    // Upgraded from 0.10.9 → 0.10.33 to fix Android 16 (SDK 36) crash:
    // "ToTensorConverter: input data size does not match expected size"
    // Root causes fixed in upstream:
    //   - 0.10.26: 16KB page alignment for Android 15+/16 (issue #6028)
    //   - 0.10.32: "Do not assume images are RGB" — Samsung Android 16 uses
    //     non-RGB bitmap formats (RGBA_1010102/wide gamut), causing data size
    //     mismatch in ToTensorConverter when it assumed 3 bytes/pixel
    //   - 0.10.33: latest stable (March 2025)
    implementation("com.google.mediapipe:tasks-vision:0.10.33")

    // MediaPipe Text Classifier for on-device intent classification
    implementation("com.google.mediapipe:tasks-text:0.10.33")

    // CameraX — bumped 1.3.1 → 1.5.3 (latest stable as of 2026-01-28) to enable
    // VideoCapture concurrent with ImageCapture and ImageAnalysis (Phase 4
    // multimodal clips PR 3). The `camera-video` artifact is added now so
    // downstream PRs can pull it in without another lockstep dependency change.
    //
    // No API breakages relative to 1.3.1 for the call sites we own
    // (AudienceAnalyzer.kt, FrameCaptureManager.kt): ImageCapture, ImageAnalysis,
    // ImageProxy, ImageCaptureException, CameraSelector, ProcessCameraProvider
    // signatures are unchanged across 1.3.x → 1.5.x. We do not use
    // CameraController.getTapToFocusState() (deprecated in 1.5.x in favor of
    // getTapToFocusInfoState()), so the deprecation does not affect us.
    //
    // VideoCapture concurrent with ImageAnalysis + ImageCapture binding has
    // been GA since CameraX 1.3.0; this PR is dependency-bump only, the use
    // case wiring lands in Phase 4 PR 3.
    implementation("androidx.camera:camera-core:1.5.3")
    implementation("androidx.camera:camera-camera2:1.5.3")
    implementation("androidx.camera:camera-lifecycle:1.5.3")
    implementation("androidx.camera:camera-view:1.5.3")
    implementation("androidx.camera:camera-video:1.5.3")

    // TensorFlow Lite for AgeGender + Emotion model Interpreter usage.
    // Bumped 2.14.0 → 2.17.0 for 16 KB native-page-size alignment (Android 15+
    // Play Store policy — older versions ship libtensorflowlite_jni.so with
    // 4 KB LOAD segment alignment which fails Play Console upload as of
    // 2026-05-04). 2.17.0 was the first 16 KB-aligned release; subsequent
    // versions remain compatible.
    //
    // tensorflow-lite-api EXCLUDED because MediaPipe `tasks-audio` 0.10.33
    // pulls in `com.google.ai.edge.litert:litert-api` (LiteRT is Google's
    // rebrand of TFLite — same `org.tensorflow.lite.*` package, identical
    // class names). Without this exclude, AGP fails with `Duplicate class
    // org.tensorflow.lite.DataType found in modules litert-api / tensorflow-
    // lite-api`. LiteRT is the canonical successor; let it win.
    implementation("org.tensorflow:tensorflow-lite:2.17.0") {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
    }
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4") {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
        exclude(group = "org.tensorflow", module = "tensorflow-lite")
    }
    // Audio classification migrated from `org.tensorflow:tensorflow-lite-
    // task-audio:0.4.4` to MediaPipe AudioClassifier. The TFLite Task Library
    // was deprecated and the last release (0.4.4) ships
    // libtask_audio_jni.so with 4 KB segment alignment — no 16 KB-aligned
    // successor exists. MediaPipe `tasks-audio` ships pre-aligned and uses
    // the same YAMNet .tflite model file in assets.
    implementation("com.google.mediapipe:tasks-audio:0.10.33")

    // LiteRT-LM for on-device VLM inference (Gemma 3n, Gemma 4, etc.)
    // Android SDK v0.9.0 (latest Android release as of April 3, 2026).
    // Note: v0.10.1 is CLI/desktop only — Android Maven artifact stays at 0.9.0.
    // The .litertlm model format is shared across versions, so Gemma 4 .litertlm
    // files from HuggingFace should load with the 0.9.0 Android SDK.
    //
    // runtimeOnly: The SDK is compiled with Kotlin 2.3+ / Java 21, but this project uses
    // Kotlin 1.9.24 / Java 17. Using `implementation` would break kapt (class version 65.0
    // vs 61.0). runtimeOnly bundles the SDK classes + native libs (liblitertlm_jni.so) in
    // the APK without exposing them to the compiler. LiteRTLMEngine uses reflection to call
    // the SDK at runtime, which works because the classes ARE in the APK classloader.
    //
    // Exclude kotlin-stdlib/reflect: the SDK transitively pulls kotlin-stdlib which
    // has metadata version incompatible with Room's kapt (kotlinx-metadata-jvm).
    // Excluding lets the project's own kotlin-stdlib 1.9.x win.
    // LiteRT-LM disabled: replaced by llama.cpp native JNI (SIGSEGV after 3-5 inferences).
    // LiteRT-LM SDK for on-device VLM inference with GPU delegate.
    // KV cache crash fixed via createFreshConversation() per inference in LiteRTLMEngine.
    // Server selects litert-lm format for GPU-capable devices (MediaTek, Qualcomm).
    runtimeOnly("com.google.ai.edge.litertlm:litertlm-android:0.9.0") {
        exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib")
        exclude(group = "org.jetbrains.kotlin", module = "kotlin-reflect")
        exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib-jdk8")
        exclude(group = "org.jetbrains.kotlin", module = "kotlin-stdlib-jdk7")
        exclude(group = "org.jetbrains.kotlinx")
    }

    // sherpa-onnx for Moonshine ASR (replacing Whisper TFLite)
    //
    // Marked `compileOnly` so the published agent-core POM does NOT declare
    // a runtime dep on a flatDir-only artifact. sherpa-onnx-1.12.26.aar is
    // resolved from `agent-core/libs/` at compile time inside this monorepo
    // (via tablet-agent / fire-tv-agent / android-tv-agent settings.gradle.kts
    // `flatDir { dirs(... agent-core/libs) }`). Maven Central / GitHub
    // Packages does not host an `sherpa-onnx:sherpa-onnx-1.12.26@aar`
    // coordinate, so an `implementation()` declaration would emit it into
    // the published POM and break downstream Gradle resolution at the
    // partner side.
    //
    // Speech recognition (Moonshine ASR) becomes a BYO dependency for
    // partners — they can drop the .aar into their own libs/ if they want
    // it. The runtime guard is `hasSpeechModel` in
    // AudienceSensingService.detectCapabilities() (checks for the
    // `moonshine-tiny/tokens.txt` asset); partners who don't bundle the
    // model never load any sherpa-onnx classes at runtime.
    //
    // Follow-up: publish a Trillboards-bundled sherpa-onnx Maven artifact
    // in v1.1 so partners can `implementation("com.trillboards.bundled:
    // sherpa-onnx:1.12.26")` without sourcing the .aar themselves.
    compileOnly("sherpa-onnx:sherpa-onnx-1.12.26@aar")

    // Room for offline signal buffering
    implementation("androidx.room:room-runtime:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")

    // Lifecycle for camera management
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    // ProcessLifecycleOwner — used by `TrillboardsSensingSdk.start()` so
    // partners can initialize the SDK from `Application.onCreate()` without
    // needing to declare a foreground Service. The owner survives the full
    // app process lifetime and binds CameraX correctly.
    implementation("androidx.lifecycle:lifecycle-process:2.6.2")
    // LifecycleService — base class for BaseDeviceAgentService (PR 3). Each
    // platform's DeviceAgentService extends this to inherit the full
    // foreground-service skeleton + Adam's three production-bug fixes.
    implementation("androidx.lifecycle:lifecycle-service:2.6.2")

    // Encrypted SharedPreferences for VAS attestation key persistence
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // Media3 ExoPlayer (native video playback — replaces WebView for video ads)
    implementation("androidx.media3:media3-exoplayer:1.5.1")
    implementation("androidx.media3:media3-exoplayer-hls:1.5.1")
    implementation("androidx.media3:media3-exoplayer-dash:1.5.1")
    implementation("androidx.media3:media3-ui:1.5.1")
    implementation("androidx.media3:media3-datasource:1.5.1")
    implementation("androidx.media3:media3-database:1.5.1")

    // IMA Extension (VAST ad insertion via Google IMA SDK)
    compileOnly("androidx.media3:media3-exoplayer-ima:1.5.1")

    // Coil (image loading for native image carousels — replaces WebView <img>)
    implementation("io.coil-kt:coil:2.5.0")

    // Google Play Services Ads Identifier (GAID for cross-device identity resolution)
    implementation("com.google.android.gms:play-services-ads-identifier:18.0.1")

    // Google Mobile Ads SDK Lite (WebView API for Ads — registerWebView, no ad serving)
    compileOnly("com.google.android.gms:play-services-ads-lite:23.6.0")

    // Firebase Installation ID — survives reinstalls, used for cross-install device identity.
    implementation(platform("com.google.firebase:firebase-bom:33.7.0"))
    implementation("com.google.firebase:firebase-installations")

    // kotlinx.serialization runtime — paired with the
    // `org.jetbrains.kotlin.plugin.serialization` Kotlin compiler plugin
    // declared in the `plugins {}` block above. Required by the generated
    // OpenAPI DTOs (`com.trillboards.api.types.*`); also exposed to the
    // conformance test suite for fixture round-trips.
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20231013")
    testImplementation("io.mockk:mockk:1.13.5")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAPI Kotlin codegen
// ─────────────────────────────────────────────────────────────────────────────
//
// Reads the merged partner OpenAPI spec (regenerated from the Zod registry by
// `trillboard-api/scripts/build-openapi.js`) and emits Kotlin DTOs into
// `src/generated/kotlin/com/trillboards/api/types/`.
//
// Output is committed to git — partner-visible diffs whenever the OpenAPI
// surface drifts. CI gate `trillboard-api/scripts/ci/check-openapi-drift.js`
// regenerates these and runs `git diff --exit-code` to catch unchecked changes.
//
// Single task — no client / api / test artifacts. Producer-side code in
// agent-core builds wire payloads through `HeartbeatPayloadFactory` (which
// imports from `com.trillboards.api.types`), so we only need the data classes.
//
// Path computation:
//   $rootProject.rootDir = trillboard-ctv/tablet-agent (the Gradle composite root)
//   merged spec lives at: trillboard-api/docs/openapi/merged/partner-api.json
//   relative to tablet-agent: ../../trillboard-api/docs/openapi/merged/partner-api.json
val mergedPartnerApiSpec =
    rootProject.layout.projectDirectory.file("../../trillboard-api/docs/openapi/merged/partner-api.json")
val openApiGenerateOutputDir =
    layout.projectDirectory.dir("src/generated/kotlin")

tasks.register<org.openapitools.generator.gradle.plugin.tasks.GenerateTask>("generateOpenApiTypes") {
    group = "openapi"
    description =
        "Emit Kotlin DTOs from the merged partner OpenAPI spec into " +
        "src/generated/kotlin/com/trillboards/api/types/."

    // Skip silently when the merged spec is not present on disk (standalone
    // builds — JitPack, downstream consumers, the public agent-core repo —
    // ship pre-generated DTOs at src/generated/kotlin/ and do not need to
    // regenerate from the monorepo OpenAPI spec).
    onlyIf { mergedPartnerApiSpec.asFile.exists() }

    generatorName.set("kotlin")
    inputSpec.set(mergedPartnerApiSpec.asFile.absolutePath)
    outputDir.set(openApiGenerateOutputDir.asFile.absolutePath)

    // The merged spec carries a few inherited validation warnings (operationId
    // collisions, attribute type quirks) from upstream Zod generation that the
    // partner API-client codegen also tolerates. Disable validation so the
    // openapi-generator does not refuse what `npm run build:openapi` already
    // produced and committed.
    validateSpec.set(false)

    // Models-only — no API clients, no tests, no docs. Producer-side code
    // assembles wire payloads via the hand-written HeartbeatPayloadFactory;
    // there is no need for openapi-generator to emit a Retrofit-style client.
    globalProperties.set(mapOf(
        "models" to "",
        "modelDocs" to "false",
        "modelTests" to "false",
        "apis" to "false",
        "apiDocs" to "false",
        "apiTests" to "false",
        "supportingFiles" to "false",
    ))

    // kotlinx.serialization → matches the JSON-on-the-wire contract; the data
    // classes carry @Serializable + @SerialName for snake_case key mapping.
    additionalProperties.set(mapOf(
        "library" to "jvm-ktor",          // jvm target (no Android Retrofit deps)
        "serializationLibrary" to "kotlinx_serialization",
        "dateLibrary" to "string",        // RFC3339 datetimes as String (no java.time)
        "modelPackage" to "com.trillboards.api.types",
        "apiPackage" to "com.trillboards.api.api",
        "packageName" to "com.trillboards.api",
        "useCoroutines" to "true",
        "enumPropertyNaming" to "UPPERCASE",
    ))

    // Map free-form `object` properties (Zod's `z.record(z.unknown())` for
    // metadata / payload bags) to `JsonObject` so kotlinx.serialization has
    // a built-in serializer. Without this, openapi-generator emits
    // `Map<String, Any>` which lacks a compile-time serializer and fails the
    // Kotlin compile step. The matching `importMappings` replaces the default
    // generator import with the real kotlinx.serialization symbol. BigDecimal
    // → Double for free-form numeric (no java.math.BigDecimal in Android).
    typeMappings.set(mapOf(
        // Free-form `object` (Zod's `z.record(z.unknown())`) → JsonElement.
        // We use JsonElement (not JsonObject) so a Map<String, JsonElement>
        // can hold strings, numbers, booleans, AND nested objects — matches
        // the partner-emitted metadata bag where `{"make": "Samsung"}` is a
        // primitive string value, not a sub-object. JsonObject would force
        // every value to itself be `{...}`.
        "AnyType" to "JsonElement",
        "object" to "JsonElement",
        // openapi-generator's kotlin templates emit `java.math.BigDecimal` for
        // `format: number` without an explicit `type: integer`. kotlinx-
        // serialization can't auto-serialize java.math.BigDecimal; remap to
        // kotlin.Double which has a built-in Double.serializer.
        "BigDecimal" to "kotlin.Double",
        "decimal" to "kotlin.Double",
        "number" to "kotlin.Double",
    ))
    importMappings.set(mapOf(
        "JsonElement" to "kotlinx.serialization.json.JsonElement",
    ))

    // Treat input as up-to-date relative to the spec file. Re-runs only when
    // the spec changes — keeps `compileKotlin` fast.
    inputs.file(mergedPartnerApiSpec)
    outputs.dir(openApiGenerateOutputDir)
}

// Wire every Kotlin compile/kapt task → generateOpenApiTypes so a fresh
// checkout (or `clean`) produces the generated sources before kapt and
// Kotlin try to resolve imports against the source set. AGP creates these
// per-variant (`compileDebugKotlin`, `kaptGenerateStubsDebugKotlin`, etc.),
// so we tag them all up-front.
tasks.matching {
    it.name.startsWith("compile") && it.name.endsWith("Kotlin")
        || it.name.startsWith("kaptGenerateStubs")
        || it.name.startsWith("kapt")
}.configureEach {
    dependsOn("generateOpenApiTypes")
}

// Add the generated dir to the main source set so Kotlin sees the DTOs.
// openapi-generator emits its output as `<outputDir>/src/main/kotlin/<package>`
// regardless of the configured outputDir; we point Android's main source set
// at the inner `src/main/kotlin` directly so Kotlin discovers the DTOs at
// `com.trillboards.api.types.*`.
android {
    sourceSets.getByName("main") {
        java.srcDirs("src/generated/kotlin/src/main/kotlin")
    }
}
