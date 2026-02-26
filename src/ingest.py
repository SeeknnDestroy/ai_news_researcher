from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from .config import InputData


class InputError(ValueError):
    pass


def load_input(path: str | Path) -> InputData:
    data = _read_yaml(path)

    if not isinstance(data, dict):
        raise InputError("Input YAML must be a mapping with urls.")

    if "urls" not in data:
        raise InputError("Input YAML must include 'urls'.")

    urls = data["urls"]
    if not isinstance(urls, list) or not urls:
        raise InputError("'urls' must be a non-empty list of strings.")

    clean_urls: List[str] = []
    for item in urls:
        if not isinstance(item, str) or not item.strip():
            raise InputError("All urls must be non-empty strings.")
        clean_urls.append(item.strip())

    eval_enabled = data.get("evaluation", True)
    if not isinstance(eval_enabled, bool):
        raise InputError("'evaluation' must be a boolean if provided.")

    return InputData(urls=clean_urls, eval_enabled=eval_enabled)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise InputError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

