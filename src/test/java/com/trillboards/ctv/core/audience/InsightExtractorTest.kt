package com.trillboards.ctv.core.audience

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class InsightExtractorTest {

    @Before
    fun resetExtractorState() {
        InsightExtractor.setDynamicBrands(emptyList())
        InsightExtractor.setCompetitorBrands(emptyList())
    }

    @Test
    fun `extract normalizes fuzzy brand mentions into canonical names`() {
        InsightExtractor.setDynamicBrands(listOf("Trillboards"))

        val insights = InsightExtractor.extract(
            "I saw the trillboard ad next to the samson TV."
        )

        assertTrue(insights.brandMentions.contains("samsung"))
        assertTrue(insights.brandMentions.contains("trillboards"))
        assertTrue(
            insights.brandMatches.any {
                it.canonicalName == "samsung" &&
                    it.observedText == "samson" &&
                    it.matchType == "fuzzy"
            }
        )
        assertTrue(
            insights.brandMatches.any {
                it.canonicalName == "trillboards" &&
                    it.observedText == "trillboard" &&
                    it.matchType == "fuzzy"
            }
        )
    }

    @Test
    fun `extract captures shopping context beyond fixed product categories`() {
        val insights = InsightExtractor.extract("I need a birthday gift for my daughter.")

        assertEquals(PurchaseIntent.CONSIDERING, insights.purchaseIntent)
        assertTrue(insights.shoppingContexts.contains("gift_mission"))
        assertTrue(insights.shoppingContexts.contains("occasion_birthday"))
        assertTrue(insights.shoppingContexts.contains("recipient_daughter"))
        assertTrue(insights.purchaseJourney.journeySignals.contains("mission_driven"))
        assertNotNull(insights.speechSemantics)
        assertEquals("active_consideration", insights.speechSemantics?.journeyState)
        assertTrue(insights.speechSemantics?.shoppingContexts?.contains("gift_mission") == true)
    }

    @Test
    fun `extract preserves compatibility objections while emitting richer themes`() {
        val insights = InsightExtractor.extract(
            "I love this but my wife would kill me and it's the third time I've looked at this."
        )

        assertEquals(PurchaseIntent.CONSIDERING, insights.purchaseIntent)
        assertTrue(insights.objections.contains(Objection.NEED_TO_THINK))
        assertEquals("repeat_consideration", insights.purchaseJourney.stage)
        assertTrue(insights.purchaseJourney.journeySignals.contains("repeat_consideration"))
        assertTrue(
            insights.objectionInsights.any {
                it.theme == "social_approval" &&
                    it.summaryLabel == Objection.NEED_TO_THINK.name
            }
        )
        assertNotNull(insights.speechSemantics)
        assertTrue(insights.speechSemantics?.reasons?.any { it.type == "social_approval" } == true)
        assertEquals("repeat_consideration", insights.speechSemantics?.journeyState)
        assertTrue(
            insights.speechSemantics?.evidencePhrases?.contains("third time i've looked") == true ||
                insights.speechSemantics?.evidencePhrases?.contains("third time i have looked") == true
        )
    }
}
