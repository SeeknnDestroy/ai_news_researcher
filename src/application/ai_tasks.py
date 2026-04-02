from __future__ import annotations

from ..agents._summary_compat import summaries_to_story_units
from ..domain.contracts import DraftOutline
from ..domain.models import ExcludedItem, SummaryItem
from .content_tasks import (
    extract_story_card_async,
    split_newsletter_items_async,
    summarize_article_async,
)
from .report_tasks import (
    assign_themes,
    build_candidate_pairs,
    build_outline,
    build_story_set,
    classify_story_merges,
    densify_gelisme,
    generate_intro,
    plan_repairs,
    render_final_report,
    render_legacy_final_report,
    write_story_articles,
)
from .report_tasks import (
    evaluate_draft_outline as _evaluate_draft_outline,
)


async def generate_draft_outline(
    client,
    summaries: list[SummaryItem],
    *,
    critique: str = "",
    previous_draft: DraftOutline | dict | None = None,
) -> DraftOutline:
    del previous_draft
    story_units = summaries_to_story_units(summaries)
    theme_plan = await assign_themes(client, story_units, critique=critique)
    return build_outline(theme_plan, story_units)


async def evaluate_draft_outline(
    client,
    outline: DraftOutline | dict,
    *,
    previous_critiques: str = "",
):
    outline_model = DraftOutline.model_validate(outline)
    return await _evaluate_draft_outline(
        client,
        outline_model,
        summaries_to_story_units([]),
        previous_critiques=previous_critiques,
    )


async def generate_final_report(
    client,
    outline: DraftOutline | dict,
    summaries: list[SummaryItem],
    excluded: list[ExcludedItem],
    *,
    critique: str = "",
) -> str:
    del client, critique
    outline_model = DraftOutline.model_validate(outline)
    return render_legacy_final_report(outline_model, summaries, excluded)

__all__ = [
    "assign_themes",
    "build_candidate_pairs",
    "build_outline",
    "build_story_set",
    "classify_story_merges",
    "densify_gelisme",
    "evaluate_draft_outline",
    "extract_story_card_async",
    "generate_draft_outline",
    "generate_final_report",
    "generate_intro",
    "plan_repairs",
    "render_final_report",
    "split_newsletter_items_async",
    "summarize_article_async",
    "write_story_articles",
]
