import pytest
from pydantic import ValidationError

from src.domain.contracts import DraftOutline, DraftOutlineArticle, JudgeEvaluation, SummaryPayload


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
