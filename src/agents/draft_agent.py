from __future__ import annotations

from ..application.report_tasks import assign_themes, build_outline
from ..domain.models import SummaryItem
from ..infrastructure.llm_client import OpenAILLMClient
from ..llm import OpenAIConfig
from ._summary_compat import summaries_to_story_units


# Compatibility shim: keep the legacy agent module surface while the canonical
# implementation lives under the StoryCard workflow.
async def generate_draft_outline(
    config: OpenAIConfig,
    summaries: list[SummaryItem],
    critique: str = "",
    previous_draft: dict | None = None,
) -> dict:
    del previous_draft
    story_units = summaries_to_story_units(summaries)
    llm_client = OpenAILLMClient(config)
    theme_plan = await assign_themes(llm_client, story_units, critique=critique)
    outline = build_outline(theme_plan, story_units)
    return outline.model_dump(mode="json")
