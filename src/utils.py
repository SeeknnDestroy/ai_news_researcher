from __future__ import annotations

from urllib.parse import urlparse
import hashlib
import re
import time
from datetime import datetime


def source_name_from_url(url: str) -> str:
    netloc = urlparse(url).netloc or url
    if netloc.startswith("www."):
        netloc = netloc[4:]
    parts = [p for p in netloc.split(".") if p]
    if len(parts) >= 2:
        base = parts[-2]
    else:
        base = parts[0] if parts else netloc
    return base.replace("-", " ").title()


def clamp_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_date(value) -> str:
    if value is None:
        return "Bilinmiyor"
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_name = month_names[value.month - 1]
    return f"{value.day:02d} {month_name} {value.year}"


def slugify_url(url: str, max_len: int = 120) -> str:
    parsed = urlparse(url)
    base = f"{parsed.netloc}{parsed.path}"
    base = base or url
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    if not slug:
        slug = "url"
    slug = slug[:max_len]
    short_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{short_hash}"


_RUN_START = time.perf_counter()


def log_stage(stage: str, message: str = "") -> None:
    elapsed = time.perf_counter() - _RUN_START
    timestamp = datetime.now().strftime("%H:%M:%S")
    payload = f"[{timestamp}] {stage}"
    if message:
        payload += f" | {message}"
    payload += f" (+{elapsed:.1f}s)"
    print(payload, flush=True)


def log_progress(stage: str, current: int, total: int, message: str = "") -> None:
    elapsed = time.perf_counter() - _RUN_START
    timestamp = datetime.now().strftime("%H:%M:%S")
    if total > 0:
        payload = f"[{timestamp}] {stage} {current}/{total}"
    else:
        payload = f"[{timestamp}] {stage} {current}"
    if message:
        payload += f" | {message}"
    payload += f" (+{elapsed:.1f}s)"
    print(payload, flush=True)
