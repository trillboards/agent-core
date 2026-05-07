// Standalone Gradle settings for the public Trillboards Sensing SDK
// (agent-core). Mirrors the monorepo's tablet-agent/settings.gradle.kts
// pluginManagement block so the AGP + Kotlin + OpenAPI Generator plugins
// resolve identically across composite and standalone builds.
//
// JitPack uses this file when building from the public repo
// (https://jitpack.io/#trillboards/agent-core).

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        id("com.android.library") version "8.7.3"
        id("org.jetbrains.kotlin.android") version "1.9.24"
        id("org.jetbrains.kotlin.kapt") version "1.9.24"
        id("org.jetbrains.kotlin.plugin.serialization") version "1.9.24"
        id("org.openapi.generator") version "7.10.0"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        google()
        mavenCentral()
        // Vendored AARs that don't have public Maven coordinates:
        //   - sherpa-onnx-1.12.26.aar  (Moonshine ASR runtime)
        //   - discovery-core-1.0.0.aar (Rust-backed mDNS via UniFFI)
        // Both are marked `compileOnly` in build.gradle.kts so the published
        // agent-core POM does not reference these flatDir-only artifacts.
        flatDir { dirs("libs") }
    }
}

rootProject.name = "agent-core"
