from __future__ import annotations

from ..application.report_tasks import render_final_report
from ..domain.contracts import DraftOutline, FinalReportArticlePayload
from ..domain.models import ExcludedItem, SummaryItem
from ._summary_compat import summaries_to_story_units


# Compatibility shim: keep the legacy agent module surface while the canonical
# implementation lives under the StoryCard workflow.
async def generate_final_report(
    config,
    outline: dict,
    summaries: list[SummaryItem],
    excluded: list[ExcludedItem],
    critique: str = "",
) -> str:
    del config, critique
    story_units = summaries_to_story_units(summaries)
    article_payloads = {
        summary.url: FinalReportArticlePayload(
            gelisme=summary.summary_tr,
            neden_onemli=summary.why_it_matters_tr,
        )
        for summary in summaries
    }
    return render_final_report(
        DraftOutline.model_validate(outline),
        story_units,
        article_payloads,
        excluded,
    )
