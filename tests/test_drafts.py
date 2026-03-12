from pathlib import Path

from src.drafts import write_diff, write_draft


def test_write_diff(tmp_path: Path):
    out_path = tmp_path / "reports" / "2026-03" / "dummy.md"
    run_id = "test_run"

    draft_1 = "Line 1\nLine 2\n"
    draft_2 = "Line 1\nLine 3\n"

    diff_path = write_diff(str(out_path), run_id, draft_1, draft_2)
    assert diff_path.exists()
    assert diff_path == tmp_path / "artifacts" / "drafts" / run_id / "draft_diff.txt"
    content = diff_path.read_text(encoding="utf-8")
    assert "draft_1.md" in content
    assert "draft_2.md" in content
