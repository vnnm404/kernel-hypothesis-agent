import copy
import math

import pytest

from kernel_agent.protocol import promotion, validate_evaluation, validate_outcome, validate_plan


def evaluation():
    return {
        "cases": [
            {
                "name": name,
                "correct": True,
                "supported": True,
                "candidate_ms": [2.0, 2.0, 2.0],
                "reference_ms": [1.0, 1.0, 1.0],
            }
            for name in ["a", "b"]
        ]
    }


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "wrong", "unsupported", "nan", "zero", "repeats", "inf"]
)
def test_incomplete_or_invalid_evaluation_is_rejected(fault):
    data = evaluation()
    if fault == "missing":
        data["cases"].pop()
    elif fault == "duplicate":
        data["cases"][1] = copy.deepcopy(data["cases"][0])
    elif fault == "wrong":
        data["cases"][0]["correct"] = False
    elif fault == "unsupported":
        data["cases"][0]["supported"] = False
    elif fault == "repeats":
        data["cases"][0]["candidate_ms"].pop()
    else:
        data["cases"][0]["candidate_ms"][0] = {"nan": math.nan, "zero": 0, "inf": math.inf}[fault]
    with pytest.raises(ValueError):
        validate_evaluation(data, ["a", "b"], 3)


def test_improved_mean_cannot_hide_case_regression():
    cfg = {"cases": ["a", "b"], "repeats": 3, "min_gain": 0.02, "max_regression": 0.03}
    parent, candidate = evaluation(), evaluation()
    candidate["cases"][0]["candidate_ms"] = [1, 1, 1]
    candidate["cases"][1]["candidate_ms"] = [2.2, 2.2, 2.2]
    result = promotion(parent, candidate, cfg)
    assert not result["accepted"]
    assert result["regressions"] == ["b"]
    candidate["cases"][1]["candidate_ms"] = [1.95, 1.95, 1.95]
    assert promotion(parent, candidate, cfg)["accepted"]


def test_single_fast_outlier_does_not_promote():
    parent, candidate = evaluation(), evaluation()
    for row in candidate["cases"]:
        row["candidate_ms"] = [0.01, 2, 2]
    cfg = {"cases": ["a", "b"], "repeats": 3, "min_gain": 0.02, "max_regression": 0.03}
    assert not promotion(parent, candidate, cfg)["accepted"]


def test_plateau_requires_competing_test_and_revision(tmp_path):
    plan = {"kind": "optimization", "question": "q", "change": "c", "expected_result": "e"}
    validate_plan(plan, False)
    with pytest.raises(ValueError):
        validate_plan(plan, True)
    plan.update(
        kind="diagnostic",
        hypotheses=[
            {"id": k, "mechanism": k, "prediction": k, "falsifier": "no change", "conditions": "fixed layout"}
            for k in ["h1", "h2"]
        ],
        discriminating_test="ablation",
        decision_rule="switch if disproved",
    )
    validate_plan(plan, True)
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/raw.txt").write_text("observed value")
    outcome = {
        "observation": "o",
        "interpretation": "i",
        "next_action": "n",
        "evidence_files": ["evidence/raw.txt"],
        "hypothesis_updates": [],
    }
    with pytest.raises(ValueError):
        validate_outcome(outcome, plan, tmp_path, True)
    outcome["hypothesis_updates"] = [
        {
            "id": k,
            "status": "inconclusive",
            "evidence": "raw.txt",
            "conditions": "fixed layout",
            "revisit_when": "layout changes",
        }
        for k in ["h1", "h2"]
    ]
    validate_outcome(outcome, plan, tmp_path, True)
    outcome["evidence_files"] = ["../missing.txt"]
    with pytest.raises(ValueError):
        validate_outcome(outcome, plan, tmp_path, True)


def test_empty_stderr_is_allowed_but_empty_diagnostic_is_not(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/stderr.log").touch()
    (tmp_path / "evidence/stdout.log").write_text("observed permission failure")
    plan = {"kind": "diagnostic"}
    outcome = dict(
        observation="o",
        interpretation="i",
        next_action="n",
        evidence_files=["evidence/stdout.log", "evidence/stderr.log"],
    )
    validate_outcome(outcome, plan, tmp_path, False)
    (tmp_path / "evidence/stdout.log").write_text("")
    with pytest.raises(ValueError, match="all empty"):
        validate_outcome(outcome, plan, tmp_path, False)
