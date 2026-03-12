from __future__ import annotations

from datetime import date
from pathlib import Path


def month_slug(target_date: date) -> str:
    return target_date.strftime("%Y-%m")


def dated_input_path(target_date: date, base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path()
    return root / "inputs" / month_slug(target_date) / f"links_{target_date.strftime('%d-%m-%Y')}.yaml"


def legacy_input_path(target_date: date, base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path()
    return root / "inputs" / f"links_{target_date.strftime('%d-%m-%Y')}.yaml"


def resolve_input_path(target_date: date, base_dir: str | Path | None = None) -> Path:
    monthly_path = dated_input_path(target_date, base_dir=base_dir)
    if monthly_path.exists():
        return monthly_path

    flat_path = legacy_input_path(target_date, base_dir=base_dir)
    if flat_path.exists():
        return flat_path

    return monthly_path


def dated_report_path(target_date: date, base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path()
    return root / "reports" / month_slug(target_date) / f"{target_date.strftime('%d-%m-%Y')}_weekly.md"


def artifacts_root_for_output(out_path: str | Path) -> Path:
    path = Path(out_path)
    for parent in path.parents:
        if parent.name == "reports":
            return parent.parent / "artifacts"

    return path.parent / "artifacts"
