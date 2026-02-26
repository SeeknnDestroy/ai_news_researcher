from src.validate import (
    EvalResult,
    GraderEval,
    Issue,
    RubricEval,
    RubricItem,
    RUBRIC_WEIGHTS,
    SectionScore,
    compute_overall_score,
    needs_revision,
)


def _make_rubric(score: float):
    return {name: RubricItem(score=score, notes="") for name in RUBRIC_WEIGHTS}


def test_compute_overall_score_full():
    overall = compute_overall_score(100, 100, 100, 100, 100)
    assert overall == 100.0


def test_compute_overall_score_half():
    overall = compute_overall_score(50, 50, 50, 50, 50)
    assert overall == 50.0


def test_compute_overall_score_with_sections():
    overall = compute_overall_score(80, 100, 90, 70, 60)
    assert overall > 70.0


def test_needs_revision_threshold():
    rubric = RubricEval(
        rubric_scores=_make_rubric(4),
        section_scores=[],
        issues=[],
        rubric_score=80.0,
        section_score_average=80.0,
        trials=[],
    )
    grader = GraderEval(score=80.0, notes="", items=[], trials=[])
    result = EvalResult(
        overall_score=74.0,
        rubric=rubric,
        groundedness=grader,
        coverage=grader,
        source_quality=grader,
        issues=[],
    )
    assert needs_revision(result) is True


def test_needs_revision_high_issue():
    rubric = RubricEval(
        rubric_scores=_make_rubric(5),
        section_scores=[],
        issues=[],
        rubric_score=100.0,
        section_score_average=100.0,
        trials=[],
    )
    issues = [Issue(severity="high", message="Problem")]
    grader = GraderEval(score=100.0, notes="", items=[], trials=[])
    result = EvalResult(
        overall_score=90.0,
        rubric=rubric,
        groundedness=grader,
        coverage=grader,
        source_quality=grader,
        issues=issues,
    )
    assert needs_revision(result) is True


def test_needs_revision_pass():
    rubric = RubricEval(
        rubric_scores=_make_rubric(5),
        section_scores=[],
        issues=[],
        rubric_score=100.0,
        section_score_average=100.0,
        trials=[],
    )
    grader = GraderEval(score=100.0, notes="", items=[], trials=[])
    result = EvalResult(
        overall_score=90.0,
        rubric=rubric,
        groundedness=grader,
        coverage=grader,
        source_quality=grader,
        issues=[],
    )
    assert needs_revision(result) is False
