from __future__ import annotations

from dataclasses import dataclass

from ..domain.contracts import DraftOutline, JudgeEvaluation
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

        max_retries = 2
        retries = 0
        critique = ""
        critique_history: list[str] = []

        while True:
            validation_errors = validate_outline_structure(draft_outline, summaries)
            if validation_errors:
                evaluation = JudgeEvaluation(
                    critique="",
                    specific_fixes_required=[],
                    passes_criteria=False,
                )
                critique = _deterministic_validation_feedback(validation_errors)
                passes_validation = False
                passes_review = False
            else:
                evaluation = await self._evaluate_outline(
                    outline=draft_outline,
                    previous_critiques="\n\n".join(critique_history),
                    eval_enabled=eval_enabled,
                )
                critique = _combine_outline_critiques(validation_errors, evaluation)
                passes_validation = True
                passes_review = evaluation.passes_criteria
            sink.emit(
                PipelineEvent(
                    stage="JUDGE_AGENT",
                    message=f"attempt={retries + 1}/{max_retries + 1} pass={passes_validation and passes_review}",
                )
            )

            if passes_validation and passes_review:
                break
            if retries >= max_retries:
                if not passes_validation:
                    raise ValueError(critique or "Deterministic outline validation failed")
                break

            critique_entry = critique
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

    async def _evaluate_outline(
        self,
        *,
        outline: DraftOutline,
        previous_critiques: str,
        eval_enabled: bool,
    ) -> JudgeEvaluation:
        if not eval_enabled:
            return JudgeEvaluation(critique="", specific_fixes_required=[], passes_criteria=True)

        return await evaluate_draft_outline(
            self.llm_client,
            outline,
            previous_critiques=previous_critiques,
        )


def validate_outline_structure(outline: DraftOutline, summaries: list[SummaryItem]) -> list[str]:
    summary_urls = {item.url for item in summaries}
    article_count = 0
    multi_url_article_count = 0
    seen_urls: set[str] = set()
    errors: list[str] = []

    for theme in outline.themes:
        for article in theme.articles:
            article_count += 1
            if len(article.news_urls_included) > 2:
                errors.append(f"{article.heading}: article block contains more than two URLs")
            if article.primary_url not in article.news_urls_included:
                errors.append(f"{article.heading}: primary_url must be included in news_urls_included")
            if len(article.news_urls_included) > 1:
                multi_url_article_count += 1

            for url in article.news_urls_included:
                if url not in summary_urls:
                    errors.append(f"{article.heading}: URL is not present in the included summaries: {url}")
                    continue
                if url in seen_urls:
                    errors.append(f"{article.heading}: URL appears multiple times across outline: {url}")
                    continue
                seen_urls.add(url)

    missing_urls = summary_urls - seen_urls
    if missing_urls:
        errors.append(f"Outline is missing {len(missing_urls)} included summary URLs")

    included_summary_count = len(summaries)
    if included_summary_count >= 20:
        minimum_article_count = int((included_summary_count * 0.7) + 0.9999)
        if article_count < minimum_article_count:
            errors.append(
                f"Outline must contain at least {minimum_article_count} article blocks for {included_summary_count} summaries"
            )
        maximum_multi_url_articles = int(article_count * 0.3)
        if multi_url_article_count > maximum_multi_url_articles:
            errors.append("Outline groups too many summaries into multi-URL article blocks")

    return errors


def _combine_outline_critiques(validation_errors: list[str], evaluation: JudgeEvaluation) -> str:
    critique_parts: list[str] = []
    if validation_errors:
        critique_parts.append(_deterministic_validation_feedback(validation_errors))
    if evaluation.critique:
        critique_parts.append(evaluation.critique)
    if evaluation.specific_fixes_required:
        fixes = "\n".join(f"- {item}" for item in evaluation.specific_fixes_required)
        critique_parts.append(f"Specific fixes required:\n{fixes}")
    return "\n\n".join(part for part in critique_parts if part)


def _deterministic_validation_feedback(validation_errors: list[str]) -> str:
    critique_parts: list[str] = []
    deterministic_feedback = ["Deterministic outline validation failed:"]
    deterministic_feedback.extend(f"- {item}" for item in validation_errors)
    deterministic_feedback.append("- Regenerate the outline so every included summary URL appears exactly once.")
    deterministic_feedback.append("- Keep one primary story per article block and use at most one supporting URL.")
    return "\n".join(deterministic_feedback)
