from __future__ import annotations

import difflib
from pathlib import Path
from typing import Iterable


def drafts_dir(out_path: str, run_id: str) -> Path:
    return Path(out_path).parent.parent / "artifacts" / "drafts" / run_id


def write_draft(out_path: str, run_id: str, name: str, content: str) -> Path:
    directory = drafts_dir(out_path, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def write_diff(out_path: str, run_id: str, draft_1: str, draft_2: str) -> Path:
    directory = drafts_dir(out_path, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    diff_path = directory / "draft_diff.txt"

    diff_lines: Iterable[str] = difflib.unified_diff(
        draft_1.splitlines(keepends=True),
        draft_2.splitlines(keepends=True),
        fromfile="draft_1.md",
        tofile="draft_2.md",
    )
    diff_path.write_text("".join(diff_lines), encoding="utf-8")
    return diff_path
