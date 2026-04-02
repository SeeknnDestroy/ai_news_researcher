from __future__ import annotations

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
    evaluate_draft_outline,
    generate_intro,
    plan_repairs,
    render_final_report,
    write_story_articles,
)

generate_final_report = render_final_report

__all__ = [
    "assign_themes",
    "build_candidate_pairs",
    "build_outline",
    "build_story_set",
    "classify_story_merges",
    "densify_gelisme",
    "evaluate_draft_outline",
    "extract_story_card_async",
    "generate_final_report",
    "generate_intro",
    "plan_repairs",
    "render_final_report",
    "split_newsletter_items_async",
    "summarize_article_async",
    "write_story_articles",
]
