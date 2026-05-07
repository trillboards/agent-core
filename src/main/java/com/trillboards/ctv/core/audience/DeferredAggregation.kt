package com.trillboards.ctv.core.audience

data class DeferredAggregationPlan(
    val shouldDefer: Boolean,
    val audioSnapshotsToRetain: List<AudioMetrics> = emptyList(),
    val speechSnapshotsToRetain: List<SpeechInsights> = emptyList()
)

internal fun buildDeferredAggregationPlan(
    faceSnapshotsEmpty: Boolean,
    currentViewerCount: Int,
    audioSnapshots: List<AudioMetrics>,
    speechSnapshots: List<SpeechInsights>
): DeferredAggregationPlan {
    val shouldDefer = faceSnapshotsEmpty && currentViewerCount > 0
    return if (shouldDefer) {
        DeferredAggregationPlan(
            shouldDefer = true,
            audioSnapshotsToRetain = audioSnapshots,
            speechSnapshotsToRetain = speechSnapshots
        )
    } else {
        DeferredAggregationPlan(shouldDefer = false)
    }
}

internal fun <T> restoreDeferredSnapshots(
    buffer: ArrayDeque<T>,
    snapshots: List<T>,
    maxSize: Int
) {
    for (snapshot in snapshots) {
        while (buffer.size >= maxSize) {
            buffer.removeFirst()
        }
        buffer.addLast(snapshot)
    }
}
