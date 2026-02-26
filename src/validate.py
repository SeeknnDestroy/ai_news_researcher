from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Set

from .config import ExcludedItem, SummaryItem
from .llm import XAIConfig, LLMError, generate_json, generate_text


RUBRIC_WEIGHTS: Dict[str, float] = {
    "Structure & Template Compliance": 0.12,
    "Citation Correctness & Coverage": 0.18,
    "Date Handling & Labels": 0.08,
    "Gelişme Quality": 0.18,
    "Neden Önemli Relevance": 0.10,
    "Turkish Clarity & Mixed-Audience Fit": 0.12,
    "Theme Coherence & Grouping": 0.10,
    "Scannability & Brevity": 0.07,
    "Grounding / No Hallucinations": 0.05,
}

EVAL_WEIGHTS: Dict[str, float] = {
    "rubric": 0.25,
    "section": 0.15,
    "groundedness": 0.25,
    "coverage": 0.20,
    "source_quality": 0.15,
}

DEFAULT_TRIALS = 2
REVISION_THRESHOLD = 75.0
GROUNDEDNESS_THRESHOLD = 70.0
COVERAGE_THRESHOLD = 70.0
MAX_REPORT_CHARS = 12000
MAX_FACTCHECK_ITEMS = 8


@dataclass
class Issue:
    severity: str
    message: str


@dataclass
class RubricItem:
    score: float
    notes: str


@dataclass
class SectionScore:
    section: str
    score: float
    notes: str


@dataclass
class RubricEval:
    rubric_scores: Dict[str, RubricItem]
    section_scores: List[SectionScore]
    issues: List[Issue]
    rubric_score: float
    section_score_average: float
    trials: List[dict]


@dataclass
class GraderEval:
    score: float
    notes: str
    items: List[dict]
    trials: List[dict]


@dataclass
class EvalResult:
    overall_score: float
    rubric: RubricEval
    groundedness: GraderEval
    coverage: GraderEval
    source_quality: GraderEval
    issues: List[Issue]


def deterministic_checks(
    report_text: str,
    items: Sequence[SummaryItem],
    input_urls: Sequence[str],
    excluded: Sequence[ExcludedItem],
) -> List[str]:
    issues: List[str] = []

    required_sections = ["# Haftalık GenAI Raporu", "## Özet", "## Haftanın Temaları", "## Kaynaklar"]
    for section in required_sections:
        if section not in report_text:
            issues.append(f"Missing section: {section}")

    if "## Tema:" not in report_text:
        issues.append("Missing theme sections (## Tema: ...)")

    if excluded and "## Kullanılamayan Kaynaklar" not in report_text:
        issues.append("Missing section: ## Kullanılamayan Kaynaklar")

    for item in items:
        link_token = f"]({item.url})"
        if link_token not in report_text:
            issues.append(f"Missing inline link for {item.url}")

    input_set = set(input_urls)
    included_set = {item.origin_url for item in items}
    excluded_set = {entry.url for entry in excluded}
    if input_set != included_set.union(excluded_set):
        missing = input_set - included_set - excluded_set
        if missing:
            issues.append(f"Missing URL accounting for: {', '.join(sorted(missing))}")

    return issues


def evaluate_report(
    config: XAIConfig,
    report_text: str,
    allowed_urls: Sequence[str],
    source_texts: Dict[str, str],
    trials: int = DEFAULT_TRIALS,
) -> EvalResult:
    report_items = _extract_report_items(report_text)
    rubric = _run_rubric_trials(config, report_text, allowed_urls, trials)
    groundedness = _run_groundedness_trials(config, report_items, source_texts, trials)
    coverage = _run_coverage_trials(config, report_items, source_texts, trials)
    source_quality = _run_source_quality_trials(config, allowed_urls, trials)

    overall = compute_overall_score(
        rubric_score=rubric.rubric_score,
        section_score=rubric.section_score_average,
        groundedness_score=groundedness.score,
        coverage_score=coverage.score,
        source_quality_score=source_quality.score,
    )

    issues = list(rubric.issues)
    if groundedness.score < GROUNDEDNESS_THRESHOLD:
        issues.append(Issue(severity="high", message="Low groundedness score"))
    if coverage.score < COVERAGE_THRESHOLD:
        issues.append(Issue(severity="medium", message="Low coverage score"))

    return EvalResult(
        overall_score=overall,
        rubric=rubric,
        groundedness=groundedness,
        coverage=coverage,
        source_quality=source_quality,
        issues=issues,
    )


def revise_report(
    config: XAIConfig,
    report_text: str,
    eval_result: EvalResult,
    allowed_urls: Sequence[str],
) -> str:
    system_prompt = (
        "You are a careful Turkish technical editor for a bank engineering report. "
        "Return ONLY the revised report in Markdown. "
        "Keep the template and remain concise, factual, and non-promotional."
    )

    issue_lines = "\n".join(
        f"- ({issue.severity}) {issue.message}" for issue in eval_result.issues
    ) or "- (none)"

    rubric_lines = "\n".join(
        f"- {name}: {item.score}/5" for name, item in eval_result.rubric.rubric_scores.items()
    )

    allowed_list = "\n".join(f"- {url}" for url in allowed_urls)

    user_prompt = f"""
Revise the report to address the issues and improve low-scoring areas.
Constraints:
- Do NOT add new sources.
- Use only the allowed URLs below.
- Preserve inline links and dates (DD Month YYYY).
- Keep the same template structure and headings.
- Maintain scannability; avoid long paragraphs.

Issues:
{issue_lines}

Rubric summary:
{rubric_lines}

Allowed URLs:
{allowed_list}

Report:
"""
    user_prompt = user_prompt + report_text

    return generate_text(config=config, system=system_prompt, user=user_prompt)


def compute_overall_score(
    rubric_score: float,
    section_score: float,
    groundedness_score: float,
    coverage_score: float,
    source_quality_score: float,
    weights: Dict[str, float] = EVAL_WEIGHTS,
) -> float:
    return round(
        rubric_score * weights["rubric"]
        + section_score * weights["section"]
        + groundedness_score * weights["groundedness"]
        + coverage_score * weights["coverage"]
        + source_quality_score * weights["source_quality"],
        2,
    )


def needs_revision(result: EvalResult, threshold: float = REVISION_THRESHOLD) -> bool:
    if result.overall_score < threshold:
        return True
    for issue in result.issues:
        if issue.severity.lower() == "high":
            return True
    return False


def eval_result_to_dict(result: EvalResult | None) -> Dict[str, object] | None:
    if result is None:
        return None
    return {
        "overall_score": result.overall_score,
        "rubric": {
            "score": result.rubric.rubric_score,
            "section_score_average": result.rubric.section_score_average,
            "rubric_scores": {
                name: {"score": item.score, "notes": item.notes}
                for name, item in result.rubric.rubric_scores.items()
            },
            "section_scores": [
                {"section": item.section, "score": item.score, "notes": item.notes}
                for item in result.rubric.section_scores
            ],
            "issues": [
                {"severity": issue.severity, "message": issue.message}
                for issue in result.rubric.issues
            ],
            "trials": result.rubric.trials,
        },
        "groundedness": {
            "score": result.groundedness.score,
            "notes": result.groundedness.notes,
            "items": result.groundedness.items,
            "trials": result.groundedness.trials,
        },
        "coverage": {
            "score": result.coverage.score,
            "notes": result.coverage.notes,
            "items": result.coverage.items,
            "trials": result.coverage.trials,
        },
        "source_quality": {
            "score": result.source_quality.score,
            "notes": result.source_quality.notes,
            "items": result.source_quality.items,
            "trials": result.source_quality.trials,
        },
        "issues": [
            {"severity": issue.severity, "message": issue.message}
            for issue in result.issues
        ],
    }


def find_link_urls(report_text: str) -> Set[str]:
    return set(re.findall(r"\]\((https?://[^)]+)\)", report_text))


# ---- Rubric grader ----


def _run_rubric_trials(
    config: XAIConfig,
    report_text: str,
    allowed_urls: Sequence[str],
    trials: int,
) -> RubricEval:
    trial_results: List[dict] = []
    for _ in range(max(1, trials)):
        trial_results.append(_run_rubric_trial(config, report_text, allowed_urls))

    rubric_scores = _aggregate_rubric_scores(trial_results)
    section_scores = _aggregate_section_scores(trial_results)
    issues = _aggregate_issues(trial_results)
    rubric_score = _rubric_score(rubric_scores)
    section_avg = _section_score_average(section_scores)

    return RubricEval(
        rubric_scores=rubric_scores,
        section_scores=section_scores,
        issues=issues,
        rubric_score=rubric_score,
        section_score_average=section_avg,
        trials=trial_results,
    )


def _run_rubric_trial(
    config: XAIConfig,
    report_text: str,
    allowed_urls: Sequence[str],
) -> dict:
    system_prompt = (
        "You are a strict QA editor for a Turkish weekly GenAI report used by a bank technology team. "
        "Return ONLY valid JSON, exactly in the requested schema. "
        "Notes <= 20 words."
    )

    allowed_list = "\n".join(f"- {url}" for url in allowed_urls)
    user_prompt = f"""
Evaluate the report below and return JSON with this exact schema:
{{
  "rubric_scores": {{
    "Structure & Template Compliance": {{"score": 0-5, "notes": "..."}},
    "Citation Correctness & Coverage": {{"score": 0-5, "notes": "..."}},
    "Date Handling & Labels": {{"score": 0-5, "notes": "..."}},
    "Gelişme Quality": {{"score": 0-5, "notes": "..."}},
    "Neden Önemli Relevance": {{"score": 0-5, "notes": "..."}},
    "Turkish Clarity & Mixed-Audience Fit": {{"score": 0-5, "notes": "..."}},
    "Theme Coherence & Grouping": {{"score": 0-5, "notes": "..."}},
    "Scannability & Brevity": {{"score": 0-5, "notes": "..."}},
    "Grounding / No Hallucinations": {{"score": 0-5, "notes": "..."}}
  }},
  "section_scores": [
    {{"section": "Özet", "score": 0-5, "notes": "..."}}
  ],
  "issues": [
    {{"severity": "low|medium|high", "message": "..."}}
  ]
}}

Checks (keep it strict):
- Turkish language, mixed-audience clarity
- Scannable structure and compliance with headings
- Inline hyperlinks for each item
- Theme coherence
- "Gelişme" is concise but complete (no critical omissions)
- "Neden Önemli" relevance (SDLC only if applicable)
- No marketing-hype tone
- No missing sections
- No new sources beyond allowed URLs

Section scoring rules:
- Include scores for: Özet, Haftanın Temaları, each "Tema: <name>" block, Kaynaklar.
- Include Kullanılamayan Kaynaklar if present.

Allowed URLs:
{allowed_list}

Report:
"""
    compact_report = _truncate_text(report_text, MAX_REPORT_CHARS)
    user_prompt = user_prompt + compact_report

    try:
        return generate_json(config=config, system=system_prompt, user=user_prompt)
    except Exception as exc:
        return {
            "rubric_scores": _default_rubric_scores(),
            "section_scores": [],
            "issues": [{"severity": "high", "message": str(exc)}],
        }


# ---- Groundedness grader ----


def _run_groundedness_trials(
    config: XAIConfig,
    report_items: List[dict],
    source_texts: Dict[str, str],
    trials: int,
) -> GraderEval:
    trial_results: List[dict] = []
    for _ in range(max(1, trials)):
        trial_results.append(_run_groundedness_trial(config, report_items, source_texts))

    aggregated = _aggregate_simple_trials(trial_results)
    return GraderEval(
        score=aggregated["score"],
        notes=aggregated["notes"],
        items=aggregated["items"],
        trials=trial_results,
    )


def _run_groundedness_trial(
    config: XAIConfig,
    report_items: List[dict],
    source_texts: Dict[str, str],
) -> dict:
    if not report_items:
        return {"score": 0.0, "notes": "No items", "items": []}

    sample = report_items[:MAX_FACTCHECK_ITEMS]
    sampled_note = ""
    if len(report_items) > len(sample):
        sampled_note = f"sampled {len(sample)}/{len(report_items)}"

    payload_lines = []
    for item in sample:
        excerpt = _truncate_text(source_texts.get(item["url"], ""), 1800)
        payload_lines.append(
            "\n".join(
                [
                    f"URL: {item['url']}",
                    f"GELISME: {item['gelisme']}",
                    f"NEDEN_ONEMLI: {item['neden_onemli']}",
                    f"SOURCE_EXCERPT: {excerpt}",
                ]
            )
        )

    system_prompt = (
        "You are a strict fact-checker for technical report quality control. "
        "Return ONLY valid JSON."
    )
    user_prompt = """
Check whether each item's GELISME and NEDEN_ONEMLI are supported by SOURCE_EXCERPT.
Return JSON:
{
  "items": [
    {"url": "...", "grounded": true/false, "notes": "short"}
  ]
}
Rules:
- grounded = true only if both statements are supported.
- If support is partial or uncertain, set grounded = false.
- notes <= 15 words.
"""
    user_prompt += "\n\n" + "\n\n---\n\n".join(payload_lines)

    try:
        data = generate_json(config=config, system=system_prompt, user=user_prompt)
    except Exception as exc:
        return {"score": 0.0, "notes": str(exc), "items": []}

    items = []
    grounded_count = 0
    for entry in data.get("items", []):
        url = str(entry.get("url", "")).strip()
        grounded = bool(entry.get("grounded"))
        notes = str(entry.get("notes", "")).strip()
        items.append({"url": url, "grounded": grounded, "notes": notes})
        if grounded:
            grounded_count += 1

    total = max(1, len(items))
    score = round((grounded_count / total) * 100.0, 2)
    base_note = f"grounded {grounded_count}/{total}"
    notes = f"{base_note}; {sampled_note}".strip("; ")
    return {"score": score, "notes": notes, "items": items}


# ---- Coverage grader ----


def _run_coverage_trials(
    config: XAIConfig,
    report_items: List[dict],
    source_texts: Dict[str, str],
    trials: int,
) -> GraderEval:
    trial_results: List[dict] = []
    for _ in range(max(1, trials)):
        trial_results.append(_run_coverage_trial(config, report_items, source_texts))

    aggregated = _aggregate_simple_trials(trial_results)
    return GraderEval(
        score=aggregated["score"],
        notes=aggregated["notes"],
        items=aggregated["items"],
        trials=trial_results,
    )


def _run_coverage_trial(
    config: XAIConfig,
    report_items: List[dict],
    source_texts: Dict[str, str],
) -> dict:
    if not report_items:
        return {"score": 0.0, "notes": "No items", "items": []}

    sample = report_items[:MAX_FACTCHECK_ITEMS]
    sampled_note = ""
    if len(report_items) > len(sample):
        sampled_note = f"sampled {len(sample)}/{len(report_items)}"

    payload_lines = []
    for item in sample:
        excerpt = _truncate_text(source_texts.get(item["url"], ""), 1800)
        payload_lines.append(
            "\n".join(
                [
                    f"URL: {item['url']}",
                    f"GELISME: {item['gelisme']}",
                    f"SOURCE_EXCERPT: {excerpt}",
                ]
            )
        )

    system_prompt = (
        "You are a strict coverage grader for technical summaries. "
        "Return ONLY valid JSON."
    )
    user_prompt = """
Evaluate coverage of each GELISME against SOURCE_EXCERPT.
Return JSON:
{
  "items": [
    {"url": "...", "score": 0-5, "notes": "missing key facts"}
  ]
}
Rules:
- Score 5 if key facts are captured succinctly.
- Penalize missing metrics/numbers when they exist in the excerpt.
- notes <= 15 words.
"""
    user_prompt += "\n\n" + "\n\n---\n\n".join(payload_lines)

    try:
        data = generate_json(config=config, system=system_prompt, user=user_prompt)
    except Exception as exc:
        return {"score": 0.0, "notes": str(exc), "items": []}

    items = []
    scores = []
    for entry in data.get("items", []):
        url = str(entry.get("url", "")).strip()
        score = _coerce_score(entry.get("score", 0))
        notes = str(entry.get("notes", "")).strip()
        items.append({"url": url, "score": score, "notes": notes})
        scores.append(score)

    avg = sum(scores) / max(1, len(scores))
    score = round((avg / 5.0) * 100.0, 2)
    base_note = f"avg {round(avg, 2)}/5"
    notes = f"{base_note}; {sampled_note}".strip("; ")
    return {"score": score, "notes": notes, "items": items}


# ---- Source quality grader ----


def _run_source_quality_trials(
    config: XAIConfig,
    allowed_urls: Sequence[str],
    trials: int,
) -> GraderEval:
    trial_results: List[dict] = []
    for _ in range(max(1, trials)):
        trial_results.append(_run_source_quality_trial(config, allowed_urls))

    aggregated = _aggregate_simple_trials(trial_results)
    return GraderEval(
        score=aggregated["score"],
        notes=aggregated["notes"],
        items=aggregated["items"],
        trials=trial_results,
    )


def _run_source_quality_trial(
    config: XAIConfig,
    allowed_urls: Sequence[str],
) -> dict:
    if not allowed_urls:
        return {"score": 0.0, "notes": "No sources", "items": []}

    system_prompt = "You are a strict source quality grader. Return ONLY valid JSON."
    urls = "\n".join(f"- {url}" for url in allowed_urls)
    user_prompt = f"""
Classify each URL as primary, secondary, or low quality.
Return JSON:
{{
  "items": [
    {{"url": "...", "quality": "primary|secondary|low", "notes": "short"}}
  ]
}}
Rules:
- primary: official org/blog or direct source.
- secondary: reputable reporting / newsletters.
- low: SEO farms or unclear provenance.
- notes <= 10 words.

URLs:
{urls}
"""

    try:
        data = generate_json(config=config, system=system_prompt, user=user_prompt)
    except Exception as exc:
        return {"score": 0.0, "notes": str(exc), "items": []}

    items = []
    scores = []
    for entry in data.get("items", []):
        url = str(entry.get("url", "")).strip()
        quality = str(entry.get("quality", "secondary")).strip().lower()
        notes = str(entry.get("notes", "")).strip()
        score = {"primary": 5.0, "secondary": 3.0, "low": 1.0}.get(quality, 3.0)
        items.append({"url": url, "quality": quality, "score": score, "notes": notes})
        scores.append(score)

    avg = sum(scores) / max(1, len(scores))
    score = round((avg / 5.0) * 100.0, 2)
    notes = f"avg {round(avg, 2)}/5"
    return {"score": score, "notes": notes, "items": items}


# ---- Helpers ----


def _extract_report_items(report_text: str) -> List[dict]:
    block_pattern = re.compile(r"###\s+(?P<title>.+?)\n(?P<body>.*?)(?=\n###\s+|\Z)", re.DOTALL)
    line_pattern = re.compile(
        r"^[\-\*]\s+(?:\*\*)?(?P<label>Tarih|Kaynak|Gelişme|Neden Önemli)(?:\*\*)?:\s*(?P<value>.*)$",
        re.MULTILINE,
    )
    url_pattern = re.compile(r"\[(?:.*?)\]\((https?://[^)]+)\)")

    items: List[dict] = []
    for block in block_pattern.finditer(report_text):
        title = block.group("title").strip()
        body = block.group("body")

        fields: Dict[str, str] = {}
        for entry in line_pattern.finditer(body):
            label = entry.group("label").strip().lower()
            value = entry.group("value").strip()
            fields[label] = value

        kaynak = fields.get("kaynak", "")
        url_match = url_pattern.search(kaynak)
        gelisme = fields.get("gelişme", "")
        neden = fields.get("neden önemli", "")

        if not url_match:
            continue

        items.append(
            {
                "title": title,
                "url": url_match.group(1).strip(),
                "gelisme": gelisme,
                "neden_onemli": neden,
            }
        )
    return items


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _aggregate_rubric_scores(trials: List[dict]) -> Dict[str, RubricItem]:
    results: Dict[str, List[float]] = {name: [] for name in RUBRIC_WEIGHTS}
    notes_map: Dict[str, str] = {name: "" for name in RUBRIC_WEIGHTS}

    for trial in trials:
        for category in RUBRIC_WEIGHTS:
            entry = trial.get("rubric_scores", {}).get(category, {})
            results[category].append(_coerce_score(entry.get("score", 0)))
            if not notes_map[category]:
                notes_map[category] = str(entry.get("notes", "")).strip()

    aggregated = {}
    for category, scores in results.items():
        avg = sum(scores) / max(1, len(scores))
        aggregated[category] = RubricItem(score=round(avg, 2), notes=notes_map[category])

    return aggregated


def _aggregate_section_scores(trials: List[dict]) -> List[SectionScore]:
    scores_by_section: Dict[str, List[float]] = {}
    notes_by_section: Dict[str, str] = {}

    for trial in trials:
        for entry in trial.get("section_scores", []):
            section = str(entry.get("section", "")).strip()
            if not section:
                continue
            scores_by_section.setdefault(section, []).append(_coerce_score(entry.get("score", 0)))
            notes_by_section.setdefault(section, str(entry.get("notes", "")).strip())

    aggregated = []
    for section, scores in scores_by_section.items():
        avg = sum(scores) / max(1, len(scores))
        aggregated.append(SectionScore(section=section, score=round(avg, 2), notes=notes_by_section.get(section, "")))

    return aggregated


def _aggregate_issues(trials: List[dict]) -> List[Issue]:
    issues: List[Issue] = []
    seen = set()
    for trial in trials:
        for entry in trial.get("issues", []):
            severity = str(entry.get("severity", "medium")).strip().lower()
            message = str(entry.get("message", "")).strip()
            key = (severity, message)
            if message and key not in seen:
                seen.add(key)
                issues.append(Issue(severity=severity, message=message))
    return issues


def _aggregate_simple_trials(trials: List[dict]) -> dict:
    scores = [float(t.get("score", 0.0)) for t in trials]
    avg_score = round(sum(scores) / max(1, len(scores)), 2)
    items = trials[0].get("items", []) if trials else []
    notes = trials[0].get("notes", "") if trials else ""
    return {"score": avg_score, "notes": notes, "items": items}


def _section_score_average(section_scores: Sequence[SectionScore]) -> float:
    if not section_scores:
        return 0.0
    total = sum(_clamp_score(item.score) for item in section_scores)
    return round((total / (len(section_scores) * 5.0)) * 100.0, 2)


def _rubric_score(rubric_scores: Dict[str, RubricItem]) -> float:
    total = 0.0
    for category, weight in RUBRIC_WEIGHTS.items():
        score = rubric_scores.get(category, RubricItem(score=0.0, notes="")).score
        score = _clamp_score(score)
        total += (score / 5.0) * weight
    return round(total * 100.0, 2)


def _coerce_score(value: object) -> float:
    try:
        return _clamp_score(float(value))
    except (TypeError, ValueError):
        return 0.0


def _clamp_score(value: float) -> float:
    return max(0.0, min(5.0, value))


def _default_rubric_scores() -> Dict[str, dict]:
    return {name: {"score": 0.0, "notes": ""} for name in RUBRIC_WEIGHTS}
