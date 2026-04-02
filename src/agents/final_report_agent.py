from __future__ import annotations

from ..application.report_tasks import render_legacy_final_report
from ..domain.contracts import DraftOutline
from ..domain.models import ExcludedItem, SummaryItem


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
    outline_model = DraftOutline.model_validate(outline)
    return render_legacy_final_report(outline_model, summaries, excluded)
