from __future__ import annotations

from ..domain.models import StoryCard, StoryUnit, SummaryItem


def summaries_to_story_units(summaries: list[SummaryItem]) -> list[StoryUnit]:
    story_units: list[StoryUnit] = []
    for summary in summaries:
        story_card = StoryCard(
            url=summary.url,
            origin_url=summary.origin_url,
            source_name=summary.source_name,
            title_raw=summary.title,
            published_at=summary.date,
            published_at_inferred=summary.date_inferred,
            raw_text=summary.summary_tr,
            content_type="legacy/summary",
            crawl_quality_flags=["legacy_summary"],
            blocked_or_partial=False,
            source_family=summary.source_name.casefold().replace(" ", "-"),
            story_title_tr=summary.title,
            story_type="legacy_summary",
            key_facts=[summary.summary_tr],
            must_keep_entities=[],
            must_keep_facts=[summary.summary_tr],
            why_it_matters_tr=summary.why_it_matters_tr,
            technical_relevance=0.0,
            strategic_relevance=0.0,
            confidence=summary.confidence,
        )
        story_units.append(
            StoryUnit.from_story_cards(
                story_cards=[story_card],
                primary_url=story_card.url,
                merge_relation="legacy_summary",
            )
        )
    return story_units
