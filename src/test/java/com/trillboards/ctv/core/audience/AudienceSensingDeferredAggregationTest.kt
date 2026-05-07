package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AudienceSensingDeferredAggregationTest {

    @Test
    fun `deferred aggregation retains speech snapshots when face samples have not arrived yet`() {
        val speech = SpeechInsights(
            purchaseIntent = PurchaseIntent.COMPARING,
            shoppingContexts = listOf("recipient_son"),
            purchaseJourney = PurchaseJourney(stage = "active_comparison"),
            confidence = 0.6f
        )

        val plan = buildDeferredAggregationPlan(
            faceSnapshotsEmpty = true,
            currentViewerCount = 1,
            audioSnapshots = emptyList(),
            speechSnapshots = listOf(speech)
        )

        assertNotNull(plan)
        assertTrue(plan!!.shouldDefer)
        assertEquals(listOf(speech), plan.speechSnapshotsToRetain)
        assertTrue(plan.audioSnapshotsToRetain.isEmpty())
    }

    @Test
    fun `deferred aggregation does not trigger when no active viewer is present`() {
        val speech = SpeechInsights(
            purchaseIntent = PurchaseIntent.CONSIDERING,
            shoppingContexts = listOf("gift_mission"),
            purchaseJourney = PurchaseJourney(stage = "repeat_consideration"),
            confidence = 0.6f
        )

        val plan = buildDeferredAggregationPlan(
            faceSnapshotsEmpty = true,
            currentViewerCount = 0,
            audioSnapshots = emptyList(),
            speechSnapshots = listOf(speech)
        )

        assertFalse(plan.shouldDefer)
        assertTrue(plan.speechSnapshotsToRetain.isEmpty())
        assertTrue(plan.audioSnapshotsToRetain.isEmpty())
    }
}
