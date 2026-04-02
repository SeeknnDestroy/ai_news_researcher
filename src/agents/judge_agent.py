from __future__ import annotations

from ..application.report_tasks import evaluate_draft_outline as _evaluate_draft_outline
from ..domain.contracts import DraftOutline
from ..infrastructure.llm_client import OpenAILLMClient
from ..llm import OpenAIConfig
from ._summary_compat import summaries_to_story_units


# Compatibility shim: keep the legacy agent module surface while the canonical
# implementation lives under the StoryCard workflow.
async def evaluate_draft_outline_legacy(
    config: OpenAIConfig, draft_outline: dict, previous_critiques: str = ""
) -> dict:
    outline = DraftOutline.model_validate(draft_outline)
    story_units = summaries_to_story_units([])
    evaluation = await _evaluate_draft_outline(
        OpenAILLMClient(config),
        outline,
        story_units,
        previous_critiques=previous_critiques,
    )
    return evaluation.model_dump(mode="json")


evaluate_draft_outline = evaluate_draft_outline_legacy
