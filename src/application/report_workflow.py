from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import DraftWorkflowResult, ExcludedItem, SummaryItem
from ..infrastructure.events import EventSink, NullEventSink, PipelineEvent
from ..infrastructure.llm_client import LLMClient
from .report_tasks import evaluate_draft_outline, generate_draft_outline, generate_final_report


@dataclass(slots=True)
class ReportWorkflowService:
    llm_client: LLMClient
    event_sink: EventSink | None = None

    async def run(
        self,
        *,
        summaries: list[SummaryItem],
        excluded: list[ExcludedItem],
        eval_enabled: bool,
    ) -> DraftWorkflowResult:
        sink = self.event_sink or NullEventSink()
        sink.emit(PipelineEvent(stage="DRAFT_AGENT", message=f"generating outline from {len(summaries)} summaries"))
        draft_outline = await generate_draft_outline(self.llm_client, summaries)

        max_retries = 1 if eval_enabled else 0
        retries = 0
        critique = ""
        critique_history: list[str] = []

        while True:
            evaluation = await evaluate_draft_outline(
                self.llm_client,
                draft_outline,
                previous_critiques="\n\n".join(critique_history),
            )
            critique = evaluation.critique
            sink.emit(
                PipelineEvent(
                    stage="JUDGE_AGENT",
                    message=f"attempt={retries + 1}/{max_retries + 1} pass={evaluation.passes_criteria}",
                )
            )

            if evaluation.passes_criteria or retries >= max_retries:
                break

            critique_entry = critique
            if evaluation.specific_fixes_required:
                critique_entry += "\n\nSpecific fixes required:\n" + "\n".join(
                    f"- {item}" for item in evaluation.specific_fixes_required
                )
            critique_history.append(f"Attempt {retries + 1} critique:\n{critique_entry}")
            sink.emit(PipelineEvent(stage="REVISION", message="regenerating draft with judge feedback"))
            draft_outline = await generate_draft_outline(
                self.llm_client,
                summaries,
                critique=critique_entry,
                previous_draft=draft_outline,
            )
            retries += 1

        final_report = await generate_final_report(
            self.llm_client,
            draft_outline,
            summaries,
            excluded,
            critique=critique,
        )
        sink.emit(PipelineEvent(stage="FINAL_AGENT", message="final report generated"))
        return DraftWorkflowResult(
            outline=draft_outline,
            evaluation=evaluation,
            final_report=final_report,
            critique=critique,
            revision_count=retries,
            critique_history=critique_history,
        )
