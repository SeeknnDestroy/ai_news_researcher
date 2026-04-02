from __future__ import annotations

from datetime import date

from src.domain.models import StoryCard, StoryUnit


def _story_card(
    url: str, *, title: str, confidence: float, key_fact: str, entity: str
) -> StoryCard:
    return StoryCard(
        url=url,
        origin_url=url,
        source_name="Example",
        title_raw=title,
        published_at=date(2026, 3, 12),
        published_at_inferred=False,
        raw_text=f"Raw text for {title}",
        content_type="text/html",
        crawl_quality_flags=[],
        blocked_or_partial=False,
        source_family="example",
        story_title_tr=title,
        story_type="product_launch",
        key_facts=[key_fact],
        must_keep_entities=[entity],
        must_keep_facts=[key_fact],
        why_it_matters_tr=f"{title} matters.",
        technical_relevance=0.7,
        strategic_relevance=0.8,
        confidence=confidence,
    )


def test_story_unit_from_story_cards_uses_primary_first_precedence():
    primary_card = _story_card(
        "https://example.com/primary",
        title="Birincil Hikaye",
        confidence=0.65,
        key_fact="Primary fact",
        entity="OpenAI",
    )
    supporting_card = _story_card(
        "https://example.com/supporting",
        title="Destekleyici Hikaye",
        confidence=0.9,
        key_fact="Supporting fact",
        entity="Microsoft",
    )

    story_unit = StoryUnit.from_story_cards(
        story_cards=[supporting_card, primary_card],
        primary_url=primary_card.url,
        merge_relation="same_event_supporting",
    )

    assert story_unit.primary_url == primary_card.url
    assert story_unit.supporting_url == supporting_card.url
    assert story_unit.canonical_title == "Birincil Hikaye"
    assert story_unit.key_facts == ["Primary fact", "Supporting fact"]
    assert story_unit.must_keep_entities == ["OpenAI", "Microsoft"]
    assert story_unit.news_urls_included == [primary_card.url, supporting_card.url]
