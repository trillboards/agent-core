package com.trillboards.ctv.core.audience

import org.junit.Test
import org.junit.Assert.assertNotNull

/**
 * Locks in the CameraX dependency contract for Phase 4 multimodal clips.
 *
 * The Phase 4 plan requires VideoCapture concurrent with ImageCapture and
 * ImageAnalysis. This test fails at runtime if any of the required CameraX
 * classes are missing from the classpath — i.e. if a future change drops
 * `camera-video` (or any other camera-* artifact) from
 * agent-core/build.gradle.kts.
 *
 * It does NOT exercise camera lifecycle binding or class static init
 * (that requires an Android device or Robolectric and is the job of
 * integration tests). We use `Class.forName(name, initialize=false, ...)`
 * because CameraX classes' static initializers transitively call into
 * `android.hardware.camera2`, which is not present on the JVM unit-test
 * classpath. `initialize=false` proves the .class is on the classpath
 * (which is the only contract we want to lock in here) without
 * triggering Android-specific init.
 */
class CameraXVideoCaptureClasspathTest {

    private fun assertOnClasspath(fqcn: String) {
        // `initialize=false` skips static init — we only want to assert the
        // class file exists in the test runtime classpath. ClassLoader is
        // the unit test's own loader, which is the same one Gradle uses to
        // expose the merged AAR JARs.
        val cls = Class.forName(fqcn, /* initialize = */ false, javaClass.classLoader)
        assertNotNull("$fqcn must resolve from CameraX artifacts on the test classpath", cls)
    }

    @Test
    fun `camera-video VideoCapture class is on the classpath`() {
        assertOnClasspath("androidx.camera.video.VideoCapture")
    }

    @Test
    fun `camera-video Recorder class is on the classpath`() {
        // Recorder is the standard VideoOutput implementation that pairs
        // with VideoCapture. Phase 4 PR 3 will use Recorder + VideoCapture.
        assertOnClasspath("androidx.camera.video.Recorder")
    }

    @Test
    fun `camera-video QualitySelector class is on the classpath`() {
        // QualitySelector picks the resolution profile (HD, FHD, etc.).
        // Required to instantiate Recorder.
        assertOnClasspath("androidx.camera.video.QualitySelector")
    }

    @Test
    fun `existing CameraX core use-case classes still resolve after bump`() {
        // Smoke-test that the ImageAnalysis and ImageCapture classes used
        // by AudienceAnalyzer.kt are still on the classpath after the
        // 1.3.1 → 1.5.3 bump. If a future bump accidentally drops
        // camera-core or camera-camera2, this test surfaces it as an
        // explicit failure rather than as a confusing call-site error.
        assertOnClasspath("androidx.camera.core.ImageAnalysis")
        assertOnClasspath("androidx.camera.core.ImageCapture")
        assertOnClasspath("androidx.camera.lifecycle.ProcessCameraProvider")
        assertOnClasspath("androidx.camera.core.CameraSelector")
    }

    @Test
    fun `ResolutionSelector API surface from camera-core is on the classpath`() {
        // 1.5.x deprecates setTargetResolution(Size); AudienceAnalyzer.kt
        // now uses ResolutionSelector + ResolutionStrategy. Lock those
        // two classes in so a future accidental drop of camera-core fails
        // here instead of in a runtime call-site.
        assertOnClasspath("androidx.camera.core.resolutionselector.ResolutionSelector")
        assertOnClasspath("androidx.camera.core.resolutionselector.ResolutionStrategy")
    }
}
