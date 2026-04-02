import pytest
from pydantic import ValidationError

from src.domain.contracts import (
    DenseGelismePayload,
    DraftOutline,
    DraftOutlineArticle,
    JudgeEvaluation,
    MergeDecisionPayload,
    MergePlanPayload,
    RepairPlan,
    StoryCardPayload,
    SummaryPayload,
    ThemeAssignmentPlan,
)


def test_summary_payload_requires_grounded_fields():
    with pytest.raises(ValidationError):
        SummaryPayload.model_validate(
            {
                "title": "Title",
                "source_name": "Example",
                "why_it_matters_tr": "Important because of execution speed.",
            }
        )


def test_draft_outline_requires_report_title():
    with pytest.raises(ValidationError):
        DraftOutline.model_validate(
            {
                "introduction_commentary": "Intro",
                "themes": [],
            }
        )


def test_judge_evaluation_normalizes_fix_list():
    evaluation = JudgeEvaluation.model_validate(
        {
            "critique": "Needs tighter headings.",
            "specific_fixes_required": [" Fix title ", "", "Add stronger ordering"],
            "passes_criteria": False,
        }
    )

    assert evaluation.specific_fixes_required == ["Fix title", "Add stronger ordering"]


def test_draft_outline_article_requires_primary_url_in_news_urls():
    with pytest.raises(ValidationError):
        DraftOutlineArticle.model_validate(
            {
                "heading": "Heading",
                "primary_url": "https://example.com/a",
                "news_urls_included": ["https://example.com/b"],
                "content_plan": "Plan",
            }
        )


def test_draft_outline_article_rejects_more_than_two_urls():
    with pytest.raises(ValidationError):
        DraftOutlineArticle.model_validate(
            {
                "heading": "Heading",
                "primary_url": "https://example.com/a",
                "news_urls_included": [
                    "https://example.com/a",
                    "https://example.com/b",
                    "https://example.com/c",
                ],
                "content_plan": "Plan",
            }
        )


def test_story_card_payload_requires_authoritative_fields():
    with pytest.raises(ValidationError):
        StoryCardPayload.model_validate(
            {
                "story_title_tr": "Baslik",
                "story_type": "",
                "key_facts": ["Fact"],
                "must_keep_entities": ["OpenAI"],
                "must_keep_facts": ["GPT-5.4 released"],
                "why_it_matters_tr": "Onemli.",
            }
        )


def test_merge_decision_payload_limits_decision_labels():
    with pytest.raises(ValidationError):
        MergeDecisionPayload.model_validate({"decision": "merge_them_all"})


def test_merge_plan_payload_rejects_same_url_pairs():
    with pytest.raises(ValidationError):
        MergePlanPayload.model_validate(
            {
                "merges": [
                    {
                        "primary_url": "https://example.com/a",
                        "supporting_url": "https://example.com/a",
                        "decision": "same_story",
                    }
                ]
            }
        )


def test_theme_assignment_plan_requires_report_title():
    with pytest.raises(ValidationError):
        ThemeAssignmentPlan.model_validate(
            {
                "themes": [
                    {
                        "theme_name": "1. Tema",
                        "story_unit_ids": ["story-1"],
                    }
                ]
            }
        )


def test_repair_plan_accepts_narrow_operations():
    repair_plan = RepairPlan.model_validate(
        {
            "critique": "Move the misplaced story.",
            "operations": [
                {
                    "operation": "move_story_unit",
                    "story_unit_id": "story-1",
                    "theme_name": "1. Tema",
                    "target_theme_name": "2. Tema",
                }
            ],
        }
    )

    assert repair_plan.operations[0].operation == "move_story_unit"


def test_dense_gelisme_payload_limits_missing_entity_count():
    payload = DenseGelismePayload.model_validate(
        {
            "missing_entities": ["A", "B", "C"],
            "gelisme": "Kisa ve yogun bir gelisme metni.",
        }
    )

    assert payload.missing_entities == ["A", "B"]
