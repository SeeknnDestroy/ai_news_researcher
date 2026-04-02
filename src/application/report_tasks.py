from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from datetime import date
from itertools import combinations

from ..domain.contracts import (
    DenseGelismePayload,
    DraftOutline,
    DraftOutlineArticle,
    DraftOutlineTheme,
    FinalReportArticlePayload,
    IntroPayload,
    JudgeEvaluation,
    MergePlanPayload,
    RepairPlan,
    ThemeAssignmentPlan,
)
from ..domain.models import (
    CandidatePair,
    ExcludedItem,
    MergeDecision,
    OutlineValidationResult,
    StoryCard,
    StorySetResult,
    StoryUnit,
    SummaryItem,
)
from ..infrastructure.llm_client import LLMClient
from ..templates.prompts import (
    COD_GELISME_SYSTEM_PROMPT,
    INTRO_WRITER_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,
    MERGE_CLASSIFIER_SYSTEM_PROMPT,
    REPAIR_PLANNER_SYSTEM_PROMPT,
    STORY_ARTICLE_SYSTEM_PROMPT,
    THEME_ASSIGNER_SYSTEM_PROMPT,
    cod_gelisme_user_prompt,
    intro_writer_user_prompt,
    judge_agent_user_prompt,
    merge_classifier_user_prompt,
    repair_planner_user_prompt,
    story_article_user_prompt,
    theme_assignment_user_prompt,
)
from ..utils import format_date


async def classify_story_merges(
    client: LLMClient,
    story_cards: list[StoryCard],
    candidate_pairs: list[CandidatePair] | None = None,
) -> list[MergeDecision]:
    del candidate_pairs
    if len(story_cards) < 2:
        return []

    story_cards_json = json.dumps(
        [
            _story_card_prompt_payload(story_card, include_raw_text=False)
            for story_card in story_cards
        ],
        ensure_ascii=False,
        indent=2,
    )
    prompt = merge_classifier_user_prompt(story_cards_json)
    try:
        payload = await client.generate_structured(
            system=MERGE_CLASSIFIER_SYSTEM_PROMPT,
            user=prompt,
            schema=MergePlanPayload,
            task_name="merge_classifier",
        )
    except Exception:
        # Merge planning is best-effort. If it fails, keep every URL separate.
        return []

    return _merge_decisions_from_plan(payload, story_cards)


def build_candidate_pairs(
    story_cards: list[StoryCard], *, max_pairs_per_card: int = 4
) -> list[CandidatePair]:
    sorted_cards = sorted(story_cards, key=lambda card: card.url)
    scored_pairs: list[tuple[int, CandidatePair]] = []
    for left_card, right_card in combinations(sorted_cards, 2):
        reason_codes, score = _candidate_reason_codes(left_card, right_card)
        if score < 3:
            continue
        scored_pairs.append(
            (
                score,
                CandidatePair(
                    left_url=left_card.url,
                    right_url=right_card.url,
                    reason_codes=reason_codes,
                ),
            )
        )

    scored_pairs.sort(key=lambda item: (-item[0], item[1].left_url, item[1].right_url))
    selected_pairs: list[CandidatePair] = []
    pair_counts: defaultdict[str, int] = defaultdict(int)
    for _, pair in scored_pairs:
        if pair_counts[pair.left_url] >= max_pairs_per_card:
            continue
        if pair_counts[pair.right_url] >= max_pairs_per_card:
            continue
        selected_pairs.append(pair)
        pair_counts[pair.left_url] += 1
        pair_counts[pair.right_url] += 1
    return selected_pairs


def build_story_set(
    story_cards: list[StoryCard],
    candidate_pairs: list[CandidatePair],
    merge_decisions: list[MergeDecision],
) -> StorySetResult:
    story_card_map = {card.url: card for card in story_cards}
    candidate_priority = {
        (pair.left_url, pair.right_url): index for index, pair in enumerate(candidate_pairs)
    }
    accepted_decisions = [
        decision
        for decision in merge_decisions
        if decision.decision in {"same_story", "same_event_supporting"}
    ]
    accepted_decisions.sort(
        key=lambda decision: (
            0 if decision.decision == "same_story" else 1,
            candidate_priority.get((decision.left_url, decision.right_url), 9999),
            decision.left_url,
            decision.right_url,
        )
    )

    assigned_urls: set[str] = set()
    story_units: list[StoryUnit] = []
    for decision in accepted_decisions:
        if decision.left_url in assigned_urls or decision.right_url in assigned_urls:
            continue
        left_card = story_card_map[decision.left_url]
        right_card = story_card_map[decision.right_url]
        story_unit = StoryUnit.from_story_cards(
            story_cards=[left_card, right_card],
            primary_url=decision.left_url,
            merge_relation=decision.decision,
        )
        story_units.append(story_unit)
        assigned_urls.add(decision.left_url)
        assigned_urls.add(decision.right_url)

    for story_card in sorted(story_cards, key=_story_card_sort_key):
        if story_card.url in assigned_urls:
            continue
        story_units.append(
            StoryUnit.from_story_cards(
                story_cards=[story_card],
                primary_url=story_card.url,
                merge_relation="single_source",
            )
        )

    story_units.sort(key=_story_unit_sort_key)
    return StorySetResult(
        story_units=story_units,
        candidate_pairs=candidate_pairs,
        merge_decisions=merge_decisions,
    )


def select_primary_story_card(story_cards: list[StoryCard]) -> StoryCard:
    return min(
        story_cards,
        key=lambda card: (
            card.blocked_or_partial,
            card.published_at_inferred,
            -card.confidence,
            -len(card.raw_text.split()),
            card.url,
        ),
    )


async def assign_themes(
    client: LLMClient,
    story_units: list[StoryUnit],
    *,
    critique: str = "",
) -> ThemeAssignmentPlan:
    story_units_json = json.dumps(
        [_story_unit_prompt_payload(story_unit) for story_unit in story_units],
        ensure_ascii=False,
        indent=2,
    )
    prompt = theme_assignment_user_prompt(story_units_json, critique=critique)
    return await client.generate_structured(
        system=THEME_ASSIGNER_SYSTEM_PROMPT,
        user=prompt,
        schema=ThemeAssignmentPlan,
        task_name="theme_assignment",
    )


def build_outline(
    theme_plan: ThemeAssignmentPlan,
    story_units: list[StoryUnit],
    *,
    introduction_commentary: str = "",
    title_overrides: dict[str, str] | None = None,
) -> DraftOutline:
    story_unit_map = {story_unit.story_unit_id: story_unit for story_unit in story_units}
    seen_story_units: set[str] = set()
    outline_themes: list[DraftOutlineTheme] = []
    title_lookup = title_overrides or {}

    for theme in theme_plan.themes:
        articles: list[DraftOutlineArticle] = []
        for story_unit_id in theme.story_unit_ids:
            story_unit = story_unit_map.get(story_unit_id)
            if story_unit is None:
                continue
            if story_unit_id in seen_story_units:
                continue
            heading = title_lookup.get(story_unit.story_unit_id, story_unit.canonical_title)
            content_plan = ". ".join(story_unit.key_facts[:3]).strip()
            articles.append(
                DraftOutlineArticle(
                    heading=heading,
                    primary_url=story_unit.primary_url,
                    news_urls_included=story_unit.news_urls_included,
                    content_plan=content_plan,
                )
            )
            seen_story_units.add(story_unit.story_unit_id)

        if not articles:
            continue
        outline_themes.append(
            DraftOutlineTheme(
                theme_name=theme.theme_name,
                theme_commentary=theme.theme_commentary,
                articles=articles,
            )
        )

    return DraftOutline(
        report_title=theme_plan.report_title,
        introduction_commentary=introduction_commentary,
        themes=outline_themes,
    )


def validate_outline_structure(
    outline: DraftOutline, story_units: list[StoryUnit]
) -> OutlineValidationResult:
    expected_story_units = {story_unit.story_unit_id for story_unit in story_units}
    url_to_story_unit_id = {
        story_card.url: story_unit.story_unit_id
        for story_unit in story_units
        for story_card in story_unit.story_cards
    }
    story_unit_primary_urls = {
        story_unit.story_unit_id: story_unit.primary_url for story_unit in story_units
    }
    expected_urls = set(url_to_story_unit_id)
    article_count = 0
    multi_url_article_count = 0
    seen_urls: set[str] = set()
    covered_story_unit_ids: set[str] = set()
    errors: list[str] = []
    failed_story_unit_ids: set[str] = set()
    failed_theme_names: set[str] = set()

    for theme in outline.themes:
        for article in theme.articles:
            article_count += 1
            if len(article.news_urls_included) > 2:
                errors.append(f"{article.heading}: article block contains more than two URLs")
                failed_theme_names.add(theme.theme_name)
            if article.primary_url not in article.news_urls_included:
                errors.append(
                    f"{article.heading}: primary_url must be included in news_urls_included"
                )
                failed_theme_names.add(theme.theme_name)
            if len(article.news_urls_included) > 1:
                multi_url_article_count += 1

            story_unit_ids_in_article = {
                url_to_story_unit_id[url]
                for url in article.news_urls_included
                if url in url_to_story_unit_id
            }
            if len(story_unit_ids_in_article) > 1:
                errors.append(f"{article.heading}: article block spans multiple story units")
                failed_story_unit_ids.update(story_unit_ids_in_article)
                failed_theme_names.add(theme.theme_name)

            for url in article.news_urls_included:
                if url not in expected_urls:
                    errors.append(
                        f"{article.heading}: URL is not present in the included story units: {url}"
                    )
                    failed_theme_names.add(theme.theme_name)
                    continue
                if url in seen_urls:
                    errors.append(
                        f"{article.heading}: URL appears multiple times across outline: {url}"
                    )
                    failed_story_unit_ids.add(url_to_story_unit_id[url])
                    failed_theme_names.add(theme.theme_name)
                    continue
                seen_urls.add(url)
                covered_story_unit_ids.add(url_to_story_unit_id[url])

            if len(story_unit_ids_in_article) == 1:
                story_unit_id = next(iter(story_unit_ids_in_article))
                expected_primary_url = story_unit_primary_urls[story_unit_id]
                if article.primary_url != expected_primary_url:
                    errors.append(
                        f"{article.heading}: primary_url does not match the story unit primary"
                    )
                    failed_story_unit_ids.add(story_unit_id)
                    failed_theme_names.add(theme.theme_name)

    missing_urls = expected_urls - seen_urls
    if missing_urls:
        errors.append(f"Outline is missing {len(missing_urls)} included story unit URLs")
        failed_story_unit_ids.update(url_to_story_unit_id[url] for url in missing_urls)

    missing_story_units = expected_story_units - covered_story_unit_ids
    if missing_story_units:
        failed_story_unit_ids.update(missing_story_units)

    included_story_unit_count = len(story_units)
    if included_story_unit_count >= 20:
        minimum_article_count = int((included_story_unit_count * 0.7) + 0.9999)
        if article_count < minimum_article_count:
            errors.append(
                "Outline must contain at least "
                f"{minimum_article_count} article blocks for "
                f"{included_story_unit_count} story units"
            )
        maximum_multi_url_articles = int(article_count * 0.3)
        if multi_url_article_count > maximum_multi_url_articles:
            errors.append("Outline groups too many story units into multi-URL article blocks")

    return OutlineValidationResult(
        errors=errors,
        failed_story_unit_ids=sorted(failed_story_unit_ids),
        failed_theme_names=sorted(failed_theme_names),
    )


async def evaluate_draft_outline(
    client: LLMClient,
    outline: DraftOutline,
    story_units: list[StoryUnit],
    *,
    previous_critiques: str = "",
) -> JudgeEvaluation:
    compact_story_index = json.dumps(
        [_story_unit_prompt_payload(story_unit) for story_unit in story_units],
        ensure_ascii=False,
        indent=2,
    )
    prompt = judge_agent_user_prompt(
        outline.model_dump_json(indent=2),
        compact_story_index,
        previous_critiques,
    )
    return await client.generate_structured(
        system=JUDGE_AGENT_SYSTEM_PROMPT,
        user=prompt,
        schema=JudgeEvaluation,
        task_name="judge_evaluation",
    )


async def plan_repairs(
    client: LLMClient,
    outline: DraftOutline,
    story_units: list[StoryUnit],
    validation_result: OutlineValidationResult,
    *,
    critique: str = "",
) -> RepairPlan:
    story_units_json = json.dumps(
        [_story_unit_prompt_payload(story_unit) for story_unit in story_units],
        ensure_ascii=False,
        indent=2,
    )
    validation_errors = "\n".join(f"- {item}" for item in validation_result.errors) or "- none"
    prompt = repair_planner_user_prompt(
        outline.model_dump_json(indent=2),
        story_units_json,
        validation_errors,
        critique=critique,
    )
    return await client.generate_structured(
        system=REPAIR_PLANNER_SYSTEM_PROMPT,
        user=prompt,
        schema=RepairPlan,
        task_name="repair_planner",
    )


def apply_repair_plan(
    theme_plan: ThemeAssignmentPlan,
    story_units: list[StoryUnit],
    repair_plan: RepairPlan,
    *,
    title_overrides: dict[str, str] | None = None,
) -> tuple[ThemeAssignmentPlan, list[StoryUnit], dict[str, str]]:
    working_payload = theme_plan.model_dump(mode="json")
    working_themes = [dict(theme) for theme in working_payload["themes"]]
    working_story_units = {story_unit.story_unit_id: story_unit for story_unit in story_units}
    working_title_overrides = dict(title_overrides or {})

    for operation in repair_plan.operations:
        if operation.operation == "assign_missing_story_unit":
            _remove_story_unit_from_themes(working_themes, operation.story_unit_id)
            target_theme = _ensure_theme(
                working_themes, operation.target_theme_name or operation.theme_name or "Eklenenler"
            )
            target_theme["story_unit_ids"].append(operation.story_unit_id)
            continue

        if operation.operation == "move_story_unit":
            _remove_story_unit_from_themes(working_themes, operation.story_unit_id)
            target_theme = _ensure_theme(
                working_themes, operation.target_theme_name or operation.theme_name or "Eklenenler"
            )
            target_theme["story_unit_ids"].append(operation.story_unit_id)
            continue

        if operation.operation == "rename_theme":
            theme = _find_theme(working_themes, operation.theme_name)
            if theme is not None:
                theme["theme_name"] = (
                    operation.new_value or operation.target_theme_name or theme["theme_name"]
                )
            continue

        if operation.operation == "reorder_story_units":
            theme = _find_theme(working_themes, operation.theme_name)
            if theme is None:
                continue
            existing_story_unit_ids = list(theme["story_unit_ids"])
            ordered_story_unit_ids = [
                story_unit_id
                for story_unit_id in operation.ordered_story_unit_ids
                if story_unit_id in existing_story_unit_ids
            ]
            trailing_story_unit_ids = [
                story_unit_id
                for story_unit_id in existing_story_unit_ids
                if story_unit_id not in ordered_story_unit_ids
            ]
            theme["story_unit_ids"] = ordered_story_unit_ids + trailing_story_unit_ids
            continue

        if operation.operation == "retitle_story_unit":
            if operation.story_unit_id:
                working_title_overrides[operation.story_unit_id] = operation.new_value
            continue

        if operation.operation == "set_primary_url":
            story_unit = working_story_units.get(operation.story_unit_id)
            if story_unit is None:
                continue
            if operation.new_value not in {story_card.url for story_card in story_unit.story_cards}:
                continue
            updated_story_unit = StoryUnit.from_story_cards(
                story_cards=story_unit.story_cards,
                primary_url=operation.new_value,
                merge_relation=story_unit.merge_relation,
            )
            working_story_units.pop(story_unit.story_unit_id)
            working_story_units[updated_story_unit.story_unit_id] = updated_story_unit
            _replace_story_unit_id(
                working_themes, story_unit.story_unit_id, [updated_story_unit.story_unit_id]
            )
            if story_unit.story_unit_id in working_title_overrides:
                working_title_overrides[updated_story_unit.story_unit_id] = (
                    working_title_overrides.pop(story_unit.story_unit_id)
                )
            continue

        if operation.operation == "split_story_unit":
            story_unit = working_story_units.pop(operation.story_unit_id, None)
            if story_unit is None:
                continue
            split_story_units = [
                StoryUnit.from_story_cards(
                    story_cards=[story_card],
                    primary_url=story_card.url,
                    merge_relation="split_repair",
                )
                for story_card in story_unit.story_cards
            ]
            for split_story_unit in split_story_units:
                working_story_units[split_story_unit.story_unit_id] = split_story_unit
            _replace_story_unit_id(
                working_themes,
                operation.story_unit_id,
                [story_unit.story_unit_id for story_unit in split_story_units],
            )
            working_title_overrides.pop(operation.story_unit_id, None)

    normalized_theme_plan = ThemeAssignmentPlan.model_validate(
        {
            "report_title": working_payload["report_title"],
            "introduction_signal": working_payload["introduction_signal"],
            "themes": working_themes,
        }
    )
    updated_story_units = sorted(working_story_units.values(), key=_story_unit_sort_key)
    return normalized_theme_plan, updated_story_units, working_title_overrides


async def write_story_articles(
    client: LLMClient,
    outline: DraftOutline,
    story_units: list[StoryUnit],
) -> dict[str, FinalReportArticlePayload]:
    story_unit_map = {story_unit.primary_url: story_unit for story_unit in story_units}
    article_tasks: list[asyncio.Task] = []
    article_keys: list[str] = []

    for theme in outline.themes:
        for article in theme.articles:
            story_unit = story_unit_map[article.primary_url]
            article_keys.append(article.primary_url)
            article_tasks.append(
                asyncio.create_task(_write_story_article(client, story_unit, theme.theme_name))
            )

    first_pass_payloads = await asyncio.gather(*article_tasks)
    final_payloads: dict[str, FinalReportArticlePayload] = {}
    for article_key, first_pass_payload in zip(
        article_keys,
        first_pass_payloads,
        strict=True,
    ):
        story_unit = story_unit_map[article_key]
        dense_gelisme = await densify_gelisme(client, first_pass_payload.gelisme, story_unit)
        final_payloads[article_key] = FinalReportArticlePayload(
            gelisme=dense_gelisme,
            neden_onemli=first_pass_payload.neden_onemli,
        )
    return final_payloads


async def generate_intro(
    client: LLMClient,
    theme_plan: ThemeAssignmentPlan,
    story_units: list[StoryUnit],
) -> str:
    story_unit_map = {story_unit.story_unit_id: story_unit for story_unit in story_units}
    ordered_story_units: list[dict[str, object]] = []
    for theme in theme_plan.themes:
        for story_unit_id in theme.story_unit_ids:
            story_unit = story_unit_map.get(story_unit_id)
            if story_unit is None:
                continue
            ordered_story_units.append(_story_unit_prompt_payload(story_unit))
            if len(ordered_story_units) >= 5:
                break
        if len(ordered_story_units) >= 5:
            break

    prompt = intro_writer_user_prompt(
        theme_plan.model_dump_json(indent=2),
        json.dumps(ordered_story_units, ensure_ascii=False, indent=2),
    )
    payload = await client.generate_structured(
        system=INTRO_WRITER_SYSTEM_PROMPT,
        user=prompt,
        schema=IntroPayload,
        task_name="intro_writer",
    )
    return payload.introduction_commentary


def render_final_report(
    outline: DraftOutline,
    story_units: list[StoryUnit],
    article_payloads: dict[str, FinalReportArticlePayload],
    excluded: list[ExcludedItem],
) -> str:
    story_unit_map = {story_unit.primary_url: story_unit for story_unit in story_units}
    report_parts: list[str] = []
    if outline.introduction_commentary:
        report_parts.append(f"# {outline.report_title}\n\n{outline.introduction_commentary}\n")
    else:
        report_parts.append(f"# {outline.report_title}\n")

    for theme in outline.themes:
        report_parts.append(f"## {theme.theme_name}\n")
        if theme.theme_commentary:
            report_parts.append(f"\n{theme.theme_commentary}\n")

        for article in theme.articles:
            story_unit = story_unit_map[article.primary_url]
            primary_story_card = _primary_story_card(story_unit)
            article_payload = article_payloads[article.primary_url]
            report_parts.append(
                "\n".join(
                    [
                        f"### <u>**{article.heading}**</u>",
                        f"- **Tarih:** {format_date(primary_story_card.published_at)}",
                        "- **Kaynak:** "
                        f"[[{primary_story_card.source_name}]({primary_story_card.url})]",
                        f"- **Gelişme:** {article_payload.gelisme}",
                        f"- **Neden Önemli:** {article_payload.neden_onemli}",
                    ]
                )
            )
            report_parts.append("\n")

    report = "\n".join(report_parts)
    if excluded:
        report += "\n\n## Kullanilamayan Kaynaklar\n"
        for item in excluded:
            report += f"- {item.url} - {item.reason}\n"
    return report


def render_legacy_final_report(
    outline: DraftOutline,
    summaries: list[SummaryItem],
    excluded: list[ExcludedItem],
) -> str:
    summary_map = {summary.url: summary for summary in summaries}
    report_parts: list[str] = []
    if outline.introduction_commentary:
        report_parts.append(f"# {outline.report_title}\n\n{outline.introduction_commentary}\n")
    else:
        report_parts.append(f"# {outline.report_title}\n")

    for theme in outline.themes:
        report_parts.append(f"## {theme.theme_name}\n")
        if theme.theme_commentary:
            report_parts.append(f"\n{theme.theme_commentary}\n")

        for article in theme.articles:
            source_summary = _legacy_source_summary(article, summaries, summary_map)
            if source_summary is None:
                continue
            report_parts.append(
                "\n".join(
                    [
                        f"### <u>**{article.heading}**</u>",
                        f"- **Tarih:** {format_date(source_summary.date)}",
                        f"- **Kaynak:** [[{source_summary.source_name}]({source_summary.url})]",
                        f"- **Gelişme:** {source_summary.summary_tr}",
                        f"- **Neden Önemli:** {source_summary.why_it_matters_tr}",
                    ]
                )
            )
            report_parts.append("\n")

    report = "\n".join(report_parts)
    if excluded:
        report += "\n\n## Kullanilamayan Kaynaklar\n"
        for item in excluded:
            report += f"- {item.url} - {item.reason}\n"
    return report


def _merge_decisions_from_plan(
    payload: MergePlanPayload,
    story_cards: list[StoryCard],
) -> list[MergeDecision]:
    story_card_urls = {story_card.url for story_card in story_cards}
    used_urls: set[str] = set()
    merge_decisions: list[MergeDecision] = []

    for item in payload.merges:
        if item.primary_url not in story_card_urls or item.supporting_url not in story_card_urls:
            return []
        if item.primary_url == item.supporting_url:
            return []
        if item.primary_url in used_urls or item.supporting_url in used_urls:
            return []

        used_urls.add(item.primary_url)
        used_urls.add(item.supporting_url)
        merge_decisions.append(
            MergeDecision(
                left_url=item.primary_url,
                right_url=item.supporting_url,
                decision=item.decision,
                rationale=item.rationale,
            )
        )

    return merge_decisions


async def _write_story_article(
    client: LLMClient,
    story_unit: StoryUnit,
    theme_name: str,
) -> FinalReportArticlePayload:
    prompt = story_article_user_prompt(
        theme_name,
        json.dumps(
            _story_unit_prompt_payload(story_unit, include_story_cards=True),
            ensure_ascii=False,
            indent=2,
        ),
    )
    return await client.generate_structured(
        system=STORY_ARTICLE_SYSTEM_PROMPT,
        user=prompt,
        schema=FinalReportArticlePayload,
        task_name="story_article",
    )


async def densify_gelisme(
    client: LLMClient,
    first_pass_gelisme: str,
    story_unit: StoryUnit,
) -> str:
    primary_story_card = _primary_story_card(story_unit)
    primary_story_card_json = json.dumps(
        _story_card_prompt_payload(primary_story_card, include_raw_text=False),
        ensure_ascii=False,
        indent=2,
    )

    for restart_index in range(2):
        current_gelisme = _normalize_text(first_pass_gelisme)
        for _round_index in range(4):
            current_word_count = _word_count(current_gelisme)
            missing_items = _missing_density_items(current_gelisme, story_unit)
            if not missing_items and 50 <= current_word_count <= 65:
                return current_gelisme

            payload = await client.generate_structured(
                system=COD_GELISME_SYSTEM_PROMPT,
                user=cod_gelisme_user_prompt(
                    current_gelisme=current_gelisme,
                    story_card_json=primary_story_card_json,
                    primary_raw_text=primary_story_card.raw_text,
                    required_additions=missing_items[:2],
                ),
                schema=DenseGelismePayload,
                task_name="cod_gelisme",
            )
            current_gelisme = _normalize_text(payload.gelisme)

        normalized_gelisme = _force_gelisme_word_band(current_gelisme, story_unit)
        if 50 <= _word_count(normalized_gelisme) <= 65:
            return normalized_gelisme

        if restart_index == 0:
            continue

    return _force_gelisme_word_band(first_pass_gelisme, story_unit)


def _candidate_reason_codes(left_card: StoryCard, right_card: StoryCard) -> tuple[list[str], int]:
    reasons: list[str] = []
    score = 0
    if left_card.origin_url == right_card.origin_url:
        reasons.append("same_origin_url")
        score += 1

    if left_card.source_family == right_card.source_family:
        reasons.append("same_source_family")
        score += 1

    if _within_days(left_card.published_at, right_card.published_at, 2):
        reasons.append("close_publish_date")
        score += 1

    shared_entities = _shared_normalized_items(
        left_card.must_keep_entities, right_card.must_keep_entities
    )
    if shared_entities:
        reasons.append("shared_entities")
        score += 2 if len(shared_entities) == 1 else 3

    title_overlap = _shared_normalized_items(_title_tokens(left_card), _title_tokens(right_card))
    if len(title_overlap) >= 2:
        reasons.append("title_overlap")
        score += 2

    if left_card.story_type == right_card.story_type:
        reasons.append("same_story_type")
        score += 1

    shared_facts = _shared_normalized_items(left_card.must_keep_facts, right_card.must_keep_facts)
    if shared_facts:
        reasons.append("shared_must_keep_facts")
        score += 2

    return reasons, score


def _within_days(left_date: date | None, right_date: date | None, max_days: int) -> bool:
    if left_date is None or right_date is None:
        return False
    return abs((left_date - right_date).days) <= max_days


def _shared_normalized_items(left_values: list[str], right_values: list[str]) -> set[str]:
    left_set = {str(value).strip().casefold() for value in left_values if str(value).strip()}
    right_set = {str(value).strip().casefold() for value in right_values if str(value).strip()}
    return left_set & right_set


def _title_tokens(story_card: StoryCard) -> list[str]:
    title = f"{story_card.story_title_tr} {story_card.title_raw}"
    return [token for token in re.findall(r"[a-zA-Z0-9]+", title.casefold()) if len(token) > 2]


def _story_card_sort_key(story_card: StoryCard) -> tuple[object, ...]:
    published_at = story_card.published_at or date.min
    return (
        -published_at.toordinal(),
        -story_card.strategic_relevance,
        -story_card.technical_relevance,
        story_card.url,
    )


def _story_unit_sort_key(story_unit: StoryUnit) -> tuple[object, ...]:
    primary_story_card = _primary_story_card(story_unit)
    published_at = primary_story_card.published_at or date.min
    return (
        -published_at.toordinal(),
        -story_unit.strategic_relevance,
        -story_unit.technical_relevance,
        story_unit.primary_url,
    )


def _primary_story_card(story_unit: StoryUnit) -> StoryCard:
    return next(
        story_card
        for story_card in story_unit.story_cards
        if story_card.url == story_unit.primary_url
    )


def _story_card_prompt_payload(
    story_card: StoryCard, *, include_raw_text: bool = False
) -> dict[str, object]:
    payload = {
        "url": story_card.url,
        "origin_url": story_card.origin_url,
        "source_name": story_card.source_name,
        "source_family": story_card.source_family,
        "title_raw": story_card.title_raw,
        "published_at": format_date(story_card.published_at),
        "published_at_inferred": story_card.published_at_inferred,
        "content_type": story_card.content_type,
        "crawl_quality_flags": story_card.crawl_quality_flags,
        "blocked_or_partial": story_card.blocked_or_partial,
        "story_title_tr": story_card.story_title_tr,
        "story_type": story_card.story_type,
        "key_facts": story_card.key_facts,
        "must_keep_entities": story_card.must_keep_entities,
        "must_keep_facts": story_card.must_keep_facts,
        "why_it_matters_tr": story_card.why_it_matters_tr,
        "technical_relevance": story_card.technical_relevance,
        "strategic_relevance": story_card.strategic_relevance,
        "confidence": story_card.confidence,
    }
    if include_raw_text:
        payload["raw_text"] = story_card.raw_text
    return payload


def _legacy_source_summary(
    article: DraftOutlineArticle,
    summaries: list[SummaryItem],
    summary_map: dict[str, SummaryItem],
) -> SummaryItem | None:
    primary_summary = summary_map.get(article.primary_url)
    if primary_summary is not None:
        return primary_summary

    for url in article.news_urls_included:
        fallback_summary = summary_map.get(url)
        if fallback_summary is not None:
            return fallback_summary

    if summaries:
        return summaries[0]

    return None


def _story_unit_prompt_payload(
    story_unit: StoryUnit,
    *,
    include_story_cards: bool = False,
) -> dict[str, object]:
    payload = {
        "story_unit_id": story_unit.story_unit_id,
        "primary_url": story_unit.primary_url,
        "supporting_url": story_unit.supporting_url,
        "merge_relation": story_unit.merge_relation,
        "canonical_title": story_unit.canonical_title,
        "canonical_story_type": story_unit.canonical_story_type,
        "news_urls_included": story_unit.news_urls_included,
        "key_facts": story_unit.key_facts,
        "must_keep_entities": story_unit.must_keep_entities,
        "must_keep_facts": story_unit.must_keep_facts,
        "why_it_matters_tr": story_unit.why_it_matters_tr,
        "technical_relevance": story_unit.technical_relevance,
        "strategic_relevance": story_unit.strategic_relevance,
        "confidence": story_unit.confidence,
    }
    if include_story_cards:
        payload["story_cards"] = [
            _story_card_prompt_payload(story_card, include_raw_text=False)
            for story_card in story_unit.story_cards
        ]
    return payload


def _find_theme(
    working_themes: list[dict[str, object]], theme_name: str
) -> dict[str, object] | None:
    for theme in working_themes:
        if theme["theme_name"] == theme_name:
            return theme
    return None


def _ensure_theme(working_themes: list[dict[str, object]], theme_name: str) -> dict[str, object]:
    theme = _find_theme(working_themes, theme_name)
    if theme is not None:
        return theme
    new_theme = {
        "theme_name": theme_name,
        "theme_commentary": "",
        "story_unit_ids": [],
    }
    working_themes.append(new_theme)
    return new_theme


def _remove_story_unit_from_themes(
    working_themes: list[dict[str, object]], story_unit_id: str
) -> None:
    for theme in working_themes:
        theme["story_unit_ids"] = [
            current_story_unit_id
            for current_story_unit_id in theme["story_unit_ids"]
            if current_story_unit_id != story_unit_id
        ]


def _replace_story_unit_id(
    working_themes: list[dict[str, object]],
    source_story_unit_id: str,
    replacement_story_unit_ids: list[str],
) -> None:
    for theme in working_themes:
        updated_story_unit_ids: list[str] = []
        replaced = False
        for story_unit_id in theme["story_unit_ids"]:
            if story_unit_id != source_story_unit_id:
                updated_story_unit_ids.append(story_unit_id)
                continue
            updated_story_unit_ids.extend(replacement_story_unit_ids)
            replaced = True
        if replaced:
            theme["story_unit_ids"] = updated_story_unit_ids
            return


def _missing_density_items(gelisme: str, story_unit: StoryUnit) -> list[str]:
    normalized_gelisme = gelisme.casefold()
    required_items = story_unit.must_keep_entities + story_unit.must_keep_facts
    missing_items = [
        item
        for item in required_items
        if str(item).strip() and str(item).casefold() not in normalized_gelisme
    ]
    return missing_items


def _force_gelisme_word_band(gelisme: str, story_unit: StoryUnit) -> str:
    normalized_gelisme = _normalize_text(gelisme)
    words = normalized_gelisme.split()
    if len(words) > 65:
        return " ".join(words[:65]).strip()

    if len(words) >= 50:
        return normalized_gelisme

    fallback_fragments = [
        item
        for item in story_unit.must_keep_facts
        if item.casefold() not in normalized_gelisme.casefold()
    ]
    fallback_fragments.extend(
        item
        for item in story_unit.key_facts
        if item.casefold() not in normalized_gelisme.casefold()
    )

    expanded_text = normalized_gelisme
    for fragment in fallback_fragments:
        candidate_text = _normalize_text(f"{expanded_text} {fragment}.")
        if _word_count(candidate_text) > 65:
            break
        expanded_text = candidate_text
        if _word_count(expanded_text) >= 50:
            return expanded_text

    return expanded_text


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _word_count(value: str) -> int:
    normalized_value = _normalize_text(value)
    if not normalized_value:
        return 0
    return len(normalized_value.split())
