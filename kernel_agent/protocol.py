from __future__ import annotations

import math
import statistics
from pathlib import Path

from .common import geomean


def validate_evaluation(data, cases, repeats):
    """Reject missing, duplicate, unsupported, nonfinite, or incorrect cases; never drop rows."""
    rows = data.get("cases", [])
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("evaluator did not return the exact case count")
    if {r.get("name") for r in rows} != set(cases):
        raise ValueError("evaluator returned a different case set")
    for r in rows:
        if r.get("correct") is not True or r.get("supported") is not True:
            raise ValueError(f"case failed or unsupported: {r.get('name')}")
        for key in ("candidate_ms", "reference_ms"):
            samples = r.get(key, [])
            if not isinstance(samples, list) or len(samples) != repeats:
                raise ValueError(f"missing repeats for {r['name']}/{key}")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError(f"invalid timing for {r['name']}/{key}")
    return data


def score(data):
    return geomean(
        statistics.median(r["reference_ms"]) / statistics.median(r["candidate_ms"]) for r in data["cases"]
    )


def promotion(parent, candidate, cfg):
    validate_evaluation(parent, cfg["cases"], cfg["repeats"])
    validate_evaluation(candidate, cfg["cases"], cfg["repeats"])
    old = {r["name"]: statistics.median(r["candidate_ms"]) for r in parent["cases"]}
    relative = {r["name"]: old[r["name"]] / statistics.median(r["candidate_ms"]) for r in candidate["cases"]}
    gain = geomean(relative.values())
    regressions = [name for name, speedup in relative.items() if 1 / speedup > 1 + cfg["max_regression"]]
    accepted = gain >= 1 + cfg["min_gain"] and not regressions
    return {
        "accepted": accepted,
        "relative_speedup": gain,
        "per_case_speedup": relative,
        "regressions": regressions,
        "reason": "accepted"
        if accepted
        else "per-case regression"
        if regressions
        else "gain below threshold",
    }


def nonempty(obj, fields):
    for field in fields:
        if not isinstance(obj.get(field), str) or not obj[field].strip():
            raise ValueError(f"nonempty string required: {field}")


def validate_plan(plan, diagnostic_required):
    if plan.get("kind") not in {"optimization", "diagnostic"}:
        raise ValueError("plan.kind must be optimization or diagnostic")
    nonempty(plan, ("question", "change", "expected_result"))
    if diagnostic_required:
        if plan["kind"] != "diagnostic":
            raise ValueError("plateau requires a diagnostic experiment before another optimization")
        hypotheses = plan.get("hypotheses", [])
        if not isinstance(hypotheses, list) or len(hypotheses) < 2:
            raise ValueError("at least two competing hypotheses are required")
        ids = set()
        for h in hypotheses:
            nonempty(h, ("id", "mechanism", "prediction", "falsifier", "conditions"))
            if h["id"] in ids:
                raise ValueError("hypothesis IDs must be unique")
            ids.add(h["id"])
        nonempty(plan, ("discriminating_test", "decision_rule"))
    return plan


def validate_outcome(outcome, plan, work, require_revision):
    nonempty(outcome, ("observation", "interpretation", "next_action"))
    if plan["kind"] == "diagnostic":
        files = outcome.get("evidence_files", [])
        if not files:
            raise ValueError("diagnostic needs saved measurement evidence")
        has_content = False
        for name in files:
            p = (Path(work) / name).resolve()
            if not p.is_relative_to((Path(work) / "evidence").resolve()) or not p.is_file():
                raise ValueError(f"missing or invalid evidence file: {name}")
            has_content |= p.stat().st_size > 0
        if not has_content:
            raise ValueError("diagnostic evidence files are all empty")
    if require_revision:
        updates = outcome.get("hypothesis_updates", [])
        expected = {h["id"] for h in plan["hypotheses"]}
        if len(updates) != len(expected) or {u.get("id") for u in updates} != expected:
            raise ValueError("outcome must update every preregistered hypothesis")
        for u in updates:
            if u.get("status") not in {"supported", "refuted", "inconclusive"}:
                raise ValueError("invalid hypothesis status")
            nonempty(u, ("evidence", "conditions", "revisit_when"))
    return outcome
