from __future__ import annotations

import re
from typing import List, Optional

from .config import CrawlItem
from .llm import XAIConfig, LLMError, generate_json
from .utils import clamp_text, log_stage


def split_newsletter_items(
    config: XAIConfig,
    item: CrawlItem,
    max_items: int = 6,
) -> List[CrawlItem]:
    if not _is_newsletter(item.text, item.url):
        log_stage("SPLIT_DEBUG", f"Skipping {item.url} - identified as NOT a newsletter")
        return [item]

    log_stage("SPLIT_DEBUG", f"Attempting to split newsletter: {item.url}")
    trimmed = _trim_newsletter_text(item.text)
    trimmed = clamp_text(trimmed, 18000)
    extracted = _llm_split_markers(config, trimmed, max_items)
    if not extracted:
        log_stage("SPLIT_DEBUG", "LLM failed to return valid split markers")
    
    segments = _segments_from_markers(trimmed, extracted, max_items)

    if not segments:
        log_stage("SPLIT_DEBUG", "Marker extraction failed, falling back to heuristic extraction")
        segments = _extract_items(trimmed, max_items)

    results: List[CrawlItem] = []
    for entry in segments:
        title = entry.get("title") or item.title or "Haber"
        url = entry.get("url") or _first_link(entry.get("text", "")) or item.url
        text = entry.get("text") or ""
        if not text.strip():
            continue
        results.append(
            CrawlItem(
                url=url,
                text=text.strip(),
                metadata={},
                title=str(title).strip(),
                origin_url=item.origin_url or item.url,
            )
        )

    return results or [item]


def _is_newsletter(text: str, url: str) -> bool:
    if not text:
        return False
    normalized_text = text.lower().replace("’", "'")
    has_required_phrase = "in today's edition of data points" in normalized_text
    has_required_domain = "deeplearning.ai" in url.lower()
    return has_required_phrase and has_required_domain


_SPLIT_SYSTEM_PROMPT = (
    "You are a precise newsletter segmenter. "
    "Work extractively: copy markers exactly from text, never paraphrase. "
    "Return ONLY valid JSON in the requested schema."
)


def _trim_newsletter_text(text: str) -> str:
    if not text:
        return ""

    stop_markers = [
        "want to know more",
        "a special offer",
        "subscribe",
        "try pro membership",
        "enroll now",
        "special offer",
        "subscribe to data points",
    ]
    lowered = text.lower()
    for marker in stop_markers:
        idx = lowered.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def _llm_split_markers(config: XAIConfig, text: str, max_items: int) -> List[dict]:
    user_prompt = f"""
Identify up to {max_items} article blocks in the newsletter below.
Return ONLY JSON with this schema:
{{
  "items": [
    {{"title": "...", "start_marker": "...", "end_marker": "...", "url": "https://..." }}
  ]
}}

Rules:
- Use ONLY substrings copied from the text for start_marker/end_marker.
- start_marker should be a 5-20 word phrase near the beginning of the article.
- end_marker should be a 5-20 word phrase near the end of the article (can be empty).
- Do NOT rewrite content or summarize.
- Skip ads, subscription offers, and promos.
- If URL is unclear, leave it empty.
- Do not fabricate markers or URLs.

Newsletter text:
"""
    user_prompt = user_prompt + text

    try:
        data = generate_json(config=config, system=_SPLIT_SYSTEM_PROMPT, user=user_prompt)
    except LLMError:
        return []

    if not isinstance(data, dict):
        return []
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [entry for entry in items if isinstance(entry, dict)]


def _segments_from_markers(text: str, items: List[dict], max_items: int) -> List[dict]:
    if not items:
        return []

    found: List[dict] = []
    for entry in items:
        if len(found) >= max_items:
            break
        title = str(entry.get("title", "")).strip()
        start_marker = str(entry.get("start_marker", "")).strip()
        end_marker = str(entry.get("end_marker", "")).strip()
        url = str(entry.get("url", "")).strip()
        start_idx = _find_marker(text, start_marker)
        if start_idx is None:
            log_stage("SPLIT_DEBUG", f"Marker NOT found. Marker: '{start_marker}'")
            log_stage("SPLIT_DEBUG", f"Text snippet (first 200 chars): {text[:200].replace(chr(10), ' ')}")
            continue
        found.append(
            {
                "title": title,
                "url": url,
                "start_idx": start_idx,
                "end_marker": end_marker,
            }
        )

    if not found:
        return []

    found.sort(key=lambda x: x["start_idx"])
    segments: List[dict] = []
    total_len = len(text)
    for idx, entry in enumerate(found):
        start_idx = entry["start_idx"]
        end_idx = None
        if entry["end_marker"]:
            end_idx = _find_marker(text, entry["end_marker"], start=start_idx + 1)
        if end_idx is None:
            next_start = found[idx + 1]["start_idx"] if idx + 1 < len(found) else total_len
            end_idx = next_start
        if end_idx <= start_idx:
            continue
        block_text = text[start_idx:end_idx].strip()
        if len(block_text.split()) < 25:
            continue
        segments.append(
            {
                "title": entry.get("title") or block_text.splitlines()[0].strip(),
                "url": entry.get("url"),
                "text": block_text,
            }
        )
        if len(segments) >= max_items:
            break

    return segments


def _find_marker(text: str, marker: str, start: int = 0) -> Optional[int]:
    if not marker:
        return None
    
    # 1. Try exact match with flexible whitespace
    pattern = re.escape(marker)
    pattern = pattern.replace(r"\ ", r"\s+")
    match = re.search(pattern, text[start:], flags=re.IGNORECASE)
    if match:
        return start + match.start()
        
    # 2. Try normalizing quotes (smart -> straight)
    norm_text = text[start:].replace("’", "'").replace("“", '"').replace("”", '"')
    norm_marker = marker.replace("’", "'").replace("“", '"').replace("”", '"')
    
    pattern = re.escape(norm_marker)
    pattern = pattern.replace(r"\ ", r"\s+")
    match = re.search(pattern, norm_text, flags=re.IGNORECASE)
    if match:
        return start + match.start()

    # 3. Try partial match (first 8 words) if marker is long
    words = marker.split()
    if len(words) > 8:
        short_marker = " ".join(words[:8])
        pattern = re.escape(short_marker)
        pattern = pattern.replace(r"\ ", r"\s+")
        match = re.search(pattern, text[start:], flags=re.IGNORECASE)
        if match:
             return start + match.start()

    return None


def _extract_items(text: str, max_items: int) -> List[dict]:
    if not text:
        return []

    stop_markers = [
        "want to know more",
        "a special offer",
        "subscribe",
        "try pro membership",
        "enroll now",
    ]
    lowered = text.lower()
    for marker in stop_markers:
        idx = lowered.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Prefer markdown bold headings as item delimiters.
    headings = list(re.finditer(r"\*\*(.+?)\*\*", text))
    items: List[dict] = []

    if headings:
        for i, match in enumerate(headings):
            title = match.group(1).strip()
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            url = _first_link(body)
            items.append({"title": title, "url": url, "text": body})
            if len(items) >= max_items:
                break
        return items

    # Fallback: split by double newline blocks that look like articles
    blocks = [b.strip() for b in text.split("\\n\\n") if b.strip()]
    for block in blocks:
        if len(items) >= max_items:
            break
        if len(block.split()) < 30:
            continue
        title = block.split("\\n", 1)[0].strip()
        url = _first_link(block)
        items.append({"title": title, "url": url, "text": block})

    return items


def _first_link(text: str) -> str:
    match = re.search(r"\\((https?://[^)]+)\\)", text)
    if match:
        return match.group(1)
    match = re.search(r"(https?://\\S+)", text)
    return match.group(1) if match else ""
