from __future__ import annotations

from datetime import date

import pytest

from src.application.content_tasks import split_newsletter_items_async
from src.application.report_tasks import (
    apply_repair_plan,
    build_candidate_pairs,
    densify_gelisme,
    select_primary_story_card,
    validate_outline_structure,
)
from src.application.report_workflow import ReportWorkflowService
from src.domain.contracts import (
    DenseGelismePayload,
    DraftOutline,
    DraftOutlineArticle,
    DraftOutlineTheme,
    IntroPayload,
    RepairPlan,
    ThemeAssignmentPlan,
)
from src.domain.models import CrawlItem, StoryCard, StoryUnit


class FakeLLM:
    def __init__(self, *, structured: dict[str, list] | None = None) -> None:
        self._structured = {key: list(value) for key, value in (structured or {}).items()}
        self.generated_structured_prompts: list[tuple[str, str]] = []

    async def generate_structured(self, *, system: str, user: str, schema, task_name: str):
        del system
        self.generated_structured_prompts.append((task_name, user))
        try:
            result = self._structured[task_name].pop(0)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"unexpected structured task: {task_name}") from exc
        if isinstance(result, Exception):
            raise result
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    async def generate_text(self, *, system: str, user: str) -> str:  # pragma: no cover - unused
        del system, user
        raise AssertionError("unexpected text generation call")


class NeverCalledLLM:
    async def generate_structured(self, *args, **kwargs):  # pragma: no cover - not used
        raise AssertionError("should not call generate_structured")

    async def generate_text(self, *args, **kwargs):  # pragma: no cover - not used
        raise AssertionError("should not call generate_text")


def _newsletter_item(url: str, text: str) -> CrawlItem:
    return CrawlItem(url=url, text=text, metadata={}, title="newsletter", origin_url=url)


def _story_card(
    url: str,
    *,
    title: str,
    entities: list[str],
    facts: list[str],
    confidence: float = 0.8,
    blocked_or_partial: bool = False,
    published_at_inferred: bool = False,
    raw_text: str = "Raw text with enough context for the article.",
) -> StoryCard:
    return StoryCard(
        url=url,
        origin_url=url,
        source_name="Example",
        title_raw=title,
        published_at=date(2026, 3, 12),
        published_at_inferred=published_at_inferred,
        raw_text=raw_text,
        content_type="text/html",
        crawl_quality_flags=[],
        blocked_or_partial=blocked_or_partial,
        source_family="example",
        story_title_tr=title,
        story_type="product_update",
        key_facts=facts,
        must_keep_entities=entities,
        must_keep_facts=facts,
        why_it_matters_tr=f"{title} matters.",
        technical_relevance=0.7,
        strategic_relevance=0.8,
        confidence=confidence,
    )


def _story_unit(story_card: StoryCard) -> StoryUnit:
    return StoryUnit.from_story_cards(
        story_cards=[story_card],
        primary_url=story_card.url,
        merge_relation="single_source",
    )


@pytest.mark.asyncio
async def test_split_newsletter_not_applicable_prepares_item_metadata():
    item = _newsletter_item("https://example.com/article", "just a regular article text")
    result = await split_newsletter_items_async(NeverCalledLLM(), item)

    assert result.strategy == "not_applicable"
    assert result.items[0].source_name == "Example"
    assert result.items[0].title_raw == "newsletter"
    assert "short_text" in result.items[0].crawl_quality_flags


def test_build_candidate_pairs_caps_pair_count_and_prefers_strong_overlap():
    alpha = _story_card(
        "https://example.com/a",
        title="OpenAI launches agent platform",
        entities=["OpenAI", "Agents SDK"],
        facts=["OpenAI launched Agents SDK"],
    )
    beta = _story_card(
        "https://example.com/b",
        title="Agents SDK launch expands OpenAI platform",
        entities=["OpenAI", "Agents SDK"],
        facts=["OpenAI launched Agents SDK"],
    )
    gamma = _story_card(
        "https://example.com/c",
        title="Anthropic ships coding update",
        entities=["Anthropic", "Claude"],
        facts=["Anthropic shipped a coding update"],
    )

    pairs = build_candidate_pairs([alpha, beta, gamma], max_pairs_per_card=1)

    assert [(pair.left_url, pair.right_url) for pair in pairs] == [(alpha.url, beta.url)]


def test_select_primary_story_card_prefers_full_crawl_then_confidence():
    weaker_card = _story_card(
        "https://example.com/weaker",
        title="Weaker",
        entities=["A"],
        facts=["Fact A"],
        confidence=0.95,
        blocked_or_partial=True,
        raw_text="short text",
    )
    stronger_card = _story_card(
        "https://example.com/stronger",
        title="Stronger",
        entities=["B"],
        facts=["Fact B"],
        confidence=0.7,
        blocked_or_partial=False,
        raw_text=(
            "a much longer raw text body that should still lose only "
            "if the earlier rules are tied"
        ),
    )

    primary_card = select_primary_story_card([weaker_card, stronger_card])

    assert primary_card.url == stronger_card.url


def test_validate_outline_structure_rejects_duplicate_and_missing_story_units():
    first_unit = _story_unit(
        _story_card(
            "https://example.com/first",
            title="First Story",
            entities=["OpenAI"],
            facts=["First fact"],
        )
    )
    second_unit = _story_unit(
        _story_card(
            "https://example.com/second",
            title="Second Story",
            entities=["Anthropic"],
            facts=["Second fact"],
        )
    )
    outline = DraftOutline(
        report_title="Weekly",
        themes=[
            DraftOutlineTheme(
                theme_name="1. Tema",
                articles=[
                    DraftOutlineArticle(
                        heading="First story",
                        primary_url=first_unit.primary_url,
                        news_urls_included=[first_unit.primary_url],
                        content_plan="Plan",
                    ),
                    DraftOutlineArticle(
                        heading="Duplicate first story",
                        primary_url=first_unit.primary_url,
                        news_urls_included=[first_unit.primary_url],
                        content_plan="Plan",
                    ),
                ],
            )
        ],
    )

    validation_result = validate_outline_structure(outline, [first_unit, second_unit])

    assert any("appears multiple times" in error for error in validation_result.errors)
    assert any("missing" in error.lower() for error in validation_result.errors)
    assert second_unit.story_unit_id in validation_result.failed_story_unit_ids


def test_apply_repair_plan_splits_story_unit_locally():
    primary_card = _story_card(
        "https://example.com/primary",
        title="Primary Story",
        entities=["OpenAI"],
        facts=["Primary fact"],
    )
    supporting_card = _story_card(
        "https://example.com/supporting",
        title="Supporting Story",
        entities=["Microsoft"],
        facts=["Supporting fact"],
    )
    merged_unit = StoryUnit.from_story_cards(
        story_cards=[primary_card, supporting_card],
        primary_url=primary_card.url,
        merge_relation="same_event_supporting",
    )
    theme_plan = ThemeAssignmentPlan.model_validate(
        {
            "report_title": "Weekly",
            "themes": [
                {
                    "theme_name": "1. Tema",
                    "story_unit_ids": [merged_unit.story_unit_id],
                }
            ],
        }
    )
    repair_plan = RepairPlan.model_validate(
        {
            "operations": [
                {
                    "operation": "split_story_unit",
                    "story_unit_id": merged_unit.story_unit_id,
                }
            ]
        }
    )

    updated_theme_plan, updated_story_units, title_overrides = apply_repair_plan(
        theme_plan,
        [merged_unit],
        repair_plan,
    )

    assert title_overrides == {}
    assert len(updated_story_units) == 2
    assert len(updated_theme_plan.themes[0].story_unit_ids) == 2


@pytest.mark.asyncio
async def test_densify_gelisme_enforces_target_word_band():
    story_card = _story_card(
        "https://example.com/story",
        title="OpenAI launches agent platform",
        entities=["OpenAI", "Agents SDK"],
        facts=["OpenAI launched Agents SDK", "The release targets enterprise orchestration"],
    )
    story_unit = _story_unit(story_card)
    dense_text = (
        "OpenAI launched Agents SDK duyurusu, Agents SDK ve enterprise "
        "orchestration hedefini ayni hikaye icinde netlestirdi; The release "
        "targets enterprise orchestration ifadesiyle platform katmaninin "
        "agent workflow tasarimi, entegrasyon koordinasyonu, operasyonel "
        "kontrol, production benzeri yurutum, gelistirici akisi uyumu ve "
        "kurumsal kullanim senaryolari icin daha uygulanabilir bir delivery "
        "zemini sundugu acik bicimde tarif edildi."
    )
    llm = FakeLLM(
        structured={
            "cod_gelisme": [
                DenseGelismePayload(
                    missing_entities=[],
                    gelisme=dense_text,
                )
            ]
        }
    )

    gelisme = await densify_gelisme(llm, "OpenAI platformunu duyurdu.", story_unit)

    assert 50 <= len(gelisme.split()) <= 65
    assert "OpenAI" in gelisme
    assert "Agents SDK" in gelisme


@pytest.mark.asyncio
async def test_report_workflow_repairs_locally_and_generates_final_report():
    first_unit = _story_unit(
        _story_card(
            "https://example.com/first",
            title="OpenAI launches agent platform",
            entities=["OpenAI", "Agents SDK"],
            facts=["OpenAI launched Agents SDK"],
        )
    )
    second_unit = _story_unit(
        _story_card(
            "https://example.com/second",
            title="Anthropic updates coding workflow",
            entities=["Anthropic", "Claude Code"],
            facts=["Anthropic updated Claude Code"],
        )
    )
    llm = FakeLLM(
        structured={
            "theme_assignment": [
                ThemeAssignmentPlan.model_validate(
                    {
                        "report_title": "Weekly",
                        "introduction_signal": "Signals",
                        "themes": [
                            {
                                "theme_name": "1. Agentic Delivery",
                                "story_unit_ids": [first_unit.story_unit_id],
                            }
                        ],
                    }
                )
            ],
            "repair_planner": [
                RepairPlan.model_validate(
                    {
                        "operations": [
                            {
                                "operation": "assign_missing_story_unit",
                                "story_unit_id": second_unit.story_unit_id,
                                "target_theme_name": "1. Agentic Delivery",
                            }
                        ]
                    }
                )
            ],
            "intro_writer": [
                IntroPayload(
                    introduction_commentary=(
                        "Bu hafta agentic delivery ve kodlama odakli "
                        "iki net urun sinyali öne cikti."
                    )
                )
            ],
            "story_article": [
                {
                    "gelisme": "OpenAI Agents SDK ile yeni platform katmanini duyurdu.",
                    "neden_onemli": "Kurumsal agent workflow tasarimini hizlandiriyor.",
                },
                {
                    "gelisme": (
                        "Anthropic Claude Code tarafinda kodlama workflow "
                        "guncellemesi yayimladi."
                    ),
                    "neden_onemli": "Gelistirici akisini daha kontrollu hale getiriyor.",
                },
            ],
            "cod_gelisme": [
                {
                    "missing_entities": [],
                    "gelisme": (
                        "OpenAI launched Agents SDK ifadesiyle duyurulan "
                        "guncelleme, Agents SDK ve platform orkestrasyonu "
                        "odakli agent workflow tasarimini daha uretim hazir "
                        "hale getirdi; degisiklik, entegrasyon, koordinasyon, "
                        "operasyonel kontrol ve kurumsal delivery hattini tek "
                        "hikaye icinde daha net ele alarak ekiplere daha "
                        "uygulanabilir bir production zemini sundu ve "
                        "gelistirici ekiplerin production benzeri akislara "
                        "daha dusuk belirsizlikle gecmesini kolaylastirdi."
                    ),
                },
                {
                    "missing_entities": [],
                    "gelisme": (
                        "Anthropic updated Claude Code ifadesiyle paylasilan "
                        "guncelleme, Claude Code odakli kodlama workflow "
                        "akisini daha sistematik hale getirerek gelistirici "
                        "ekiplerin kod uretimi, yinelemeli duzeltme, uygulama "
                        "ici kontrol ve gunluk agentic coding akisini daha "
                        "uygulanabilir sekilde yonetmesine yardim eden pratik "
                        "bir iyilestirme sundu; degisiklik, ekiplerin araci "
                        "daha tutarli ve daha uretim odakli sekilde "
                        "kullanmasini kolaylastirdi."
                    ),
                },
            ],
        }
    )
    service = ReportWorkflowService(llm_client=llm)

    result = await service.run(
        story_units=[first_unit, second_unit],
        excluded=[],
        eval_enabled=False,
    )

    assert result.revision_count == 1
    assert "Deterministic outline validation failed" in result.critique_history[0]
    assert "## 1. Agentic Delivery" in result.final_report
    assert "OpenAI launches agent platform" in result.final_report
    assert "Anthropic updates coding workflow" in result.final_report
