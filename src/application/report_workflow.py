from __future__ import annotations

from dataclasses import dataclass

from ..domain.contracts import DraftOutline, JudgeEvaluation
from ..domain.models import DraftWorkflowResult, ExcludedItem, OutlineValidationResult, StoryUnit
from ..infrastructure.events import EventSink, NullEventSink, PipelineEvent
from ..infrastructure.llm_client import LLMClient
from .report_tasks import (
    apply_repair_plan,
    assign_themes,
    build_outline,
    evaluate_draft_outline,
    generate_intro,
    plan_repairs,
    render_final_report,
    validate_outline_structure,
    write_story_articles,
)


@dataclass(slots=True)
class ReportWorkflowService:
    llm_client: LLMClient
    event_sink: EventSink | None = None

    async def run(
        self,
        *,
        story_units: list[StoryUnit],
        excluded: list[ExcludedItem],
        eval_enabled: bool,
    ) -> DraftWorkflowResult:
        sink = self.event_sink or NullEventSink()
        sink.emit(
            PipelineEvent(
                stage="THEME_ASSIGNER", message=f"assigning {len(story_units)} story units"
            )
        )
        theme_plan = await assign_themes(self.llm_client, story_units)

        revision_count = 0
        critique = ""
        critique_history: list[str] = []
        title_overrides: dict[str, str] = {}
        current_story_units = list(story_units)
        evaluation = JudgeEvaluation(critique="", specific_fixes_required=[], passes_criteria=True)
        outline = build_outline(theme_plan, current_story_units, title_overrides=title_overrides)

        while True:
            validation_result = validate_outline_structure(outline, current_story_units)
            critique, evaluation, passes_review = await self._evaluate_current_outline(
                outline=outline,
                story_units=current_story_units,
                validation_result=validation_result,
                critique_history=critique_history,
                eval_enabled=eval_enabled,
            )
            sink.emit(
                PipelineEvent(
                    stage="JUDGE_AGENT",
                    message=(
                        f"attempt={revision_count + 1}/3 "
                        f"pass={not validation_result.errors and passes_review}"
                    ),
                )
            )

            if not validation_result.errors and passes_review:
                break

            if revision_count >= 2:
                if validation_result.errors:
                    raise ValueError(critique or "Deterministic outline validation failed")
                break

            critique_history.append(f"Attempt {revision_count + 1} critique:\n{critique}")
            sink.emit(PipelineEvent(stage="REPAIR_PLANNER", message="repairing outline locally"))
            repair_plan = await plan_repairs(
                self.llm_client,
                outline,
                current_story_units,
                validation_result,
                critique=critique,
            )
            theme_plan, current_story_units, title_overrides = apply_repair_plan(
                theme_plan,
                current_story_units,
                repair_plan,
                title_overrides=title_overrides,
            )
            outline = build_outline(
                theme_plan, current_story_units, title_overrides=title_overrides
            )
            revision_count += 1

        intro = await generate_intro(self.llm_client, theme_plan, current_story_units)
        outline = build_outline(
            theme_plan,
            current_story_units,
            introduction_commentary=intro,
            title_overrides=title_overrides,
        )
        article_payloads = await write_story_articles(self.llm_client, outline, current_story_units)
        final_report = render_final_report(outline, current_story_units, article_payloads, excluded)
        sink.emit(PipelineEvent(stage="FINAL_AGENT", message="final report generated"))
        return DraftWorkflowResult(
            outline=outline,
            theme_plan=theme_plan,
            evaluation=evaluation,
            final_report=final_report,
            critique=critique,
            revision_count=revision_count,
            critique_history=critique_history,
        )

    async def _evaluate_current_outline(
        self,
        *,
        outline: DraftOutline,
        story_units: list[StoryUnit],
        validation_result: OutlineValidationResult,
        critique_history: list[str],
        eval_enabled: bool,
    ) -> tuple[str, JudgeEvaluation, bool]:
        if validation_result.errors:
            critique = _deterministic_validation_feedback(validation_result)
            evaluation = JudgeEvaluation(
                critique="", specific_fixes_required=[], passes_criteria=False
            )
            return critique, evaluation, False

        if not eval_enabled:
            evaluation = JudgeEvaluation(
                critique="", specific_fixes_required=[], passes_criteria=True
            )
            return "", evaluation, True

        evaluation = await evaluate_draft_outline(
            self.llm_client,
            outline,
            story_units,
            previous_critiques="\n\n".join(critique_history),
        )
        critique = _combine_outline_critiques(validation_result, evaluation)
        return critique, evaluation, evaluation.passes_criteria


def _combine_outline_critiques(
    validation_result: OutlineValidationResult, evaluation: JudgeEvaluation
) -> str:
    critique_parts: list[str] = []
    if validation_result.errors:
        critique_parts.append(_deterministic_validation_feedback(validation_result))
    if evaluation.critique:
        critique_parts.append(evaluation.critique)
    if evaluation.specific_fixes_required:
        fixes = "\n".join(f"- {item}" for item in evaluation.specific_fixes_required)
        critique_parts.append(f"Specific fixes required:\n{fixes}")
    return "\n\n".join(part for part in critique_parts if part)


def _deterministic_validation_feedback(validation_result: OutlineValidationResult) -> str:
    deterministic_feedback = ["Deterministic outline validation failed:"]
    deterministic_feedback.extend(f"- {item}" for item in validation_result.errors)
    if validation_result.failed_story_unit_ids:
        deterministic_feedback.append(
            "- Failed story units: " + ", ".join(validation_result.failed_story_unit_ids)
        )
    deterministic_feedback.append("- Repair only the listed local issues.")
    deterministic_feedback.append(
        "- Keep each story unit assigned exactly once and keep one primary URL per unit."
    )
    return "\n".join(deterministic_feedback)
