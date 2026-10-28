"""Evaluator subprocess: owns correctness and timing. Never import GPU packages in the controller."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import ROOT, digest, read_json, write_json


def flash(cfg, candidate, output):
    sys.path.insert(0, str(ROOT / "vendor/flash_cuda"))
    import torch
    from bench import eval as E
    from bench import scenarios as S

    torch.cuda.set_device(0)
    target = E.TARGETS[cfg["target"]]
    major, _ = torch.cuda.get_device_capability(0)
    if major != {"hopper": 9, "blackwell": 10}[cfg["target"]]:
        raise RuntimeError(f"{cfg['target']} task on wrong GPU architecture: {major}")
    all_configs = {c.name: c for c in S.get_configs("all")}
    missing = set(cfg["cases"]) - all_configs.keys()
    if missing:
        raise ValueError(f"unknown scenario names: {sorted(missing)}")
    if not E.baseline_available(target):
        raise RuntimeError(
            f"required {target.baseline_label} reference unavailable: {E.baseline_import_error(target)}"
        )
    namespace = "_kha_" + digest(candidate)[:16]
    E.build_candidate(str(candidate), namespace, str(output.parent / "build"), target)
    rows = []
    # All requested shapes and input seeds must pass before ANY timing is considered.
    for name in cfg["cases"]:
        checks = [
            E.check_correctness(
                all_configs[name], namespace, target, gpu_id=0, seed=seed, with_cudnn=False, with_fi=False
            )
            for seed in cfg["correctness_seeds"]
        ]
        good = all(r.correct and not r.unsup_ours and not r.unsup_baseline and r.baseline for r in checks)
        rows.append(
            {
                "name": name,
                "correct": good,
                "supported": all(not r.unsup_ours and not r.unsup_baseline and r.baseline for r in checks),
                "candidate_ms": [],
                "reference_ms": [],
                "notes": [r.note for r in checks],
            }
        )
    if all(r["correct"] for r in rows):
        for row in rows:
            for _ in range(cfg["repeats"]):
                measured = E.measure(
                    all_configs[row["name"]], namespace, 0, target, with_cudnn=False, with_fi=False
                )
                row["candidate_ms"].append(measured.cand_ms)
                row["reference_ms"].append(measured.base_ms)
    return {
        "cases": rows,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "namespace": namespace,
        "timing": dict(S.PERF),
        "metric": "reference/candidate dispatch-inclusive CUDA-event latency",
        "correctness_seeds": cfg["correctness_seeds"],
    }


def fk_case_id(row):
    """Stable across report ordering; same identity used when preparing the task."""
    return json.dumps([row["op"], row["shape"], row["dtype"]], separators=(",", ":"))


def fastkernels(cfg, candidate, output):
    import shutil

    folder = output.parent / "fk-candidate" / f"L{cfg['level']}"
    folder.mkdir(parents=True)
    shutil.copy2(candidate, folder / f"{cfg['stem']}.py")
    env = os.environ | {
        "FASTKERNELS_CANDIDATE_DIR": str(folder.parent),
        "FASTKERNELS_RESULTS_DIR": str(output.parent / "fk-results"),
    }
    rows = {
        name: {"name": name, "correct": True, "supported": True, "candidate_ms": [], "reference_ms": []}
        for name in cfg["cases"]
    }
    for i in range(cfg["repeats"]):
        raw = output.parent / f"fastkernels-{i}.json"
        # CLI owns its numerical checks. No claim that our input-seed knob controls FastKernels.
        cmd = [
            sys.executable,
            "-m",
            "fastkernels",
            "bench",
            "--target",
            cfg["stem"],
            "--captures",
            cfg["captures"],
            "--gpus",
            "0",
            "--output",
            str(raw),
        ]
        subprocess.run(cmd, cwd=cfg["fk_root"], env=env, check=True)
        report = read_json(raw)
        scenarios = report.get("scenarios", [])
        names = [fk_case_id(r) for r in scenarios]
        if len(names) != len(set(names)) or set(names) != set(rows):
            raise ValueError(
                "FastKernels changed workload coverage; prepare task from a fresh reference report"
            )
        for r in scenarios:
            dst = rows[fk_case_id(r)]
            dst["correct"] &= r.get("status") == "PASSED"
            dst["supported"] &= r.get("status") != "SKIPPED"
            dst["candidate_ms"].append(r.get("candidate_ms"))
            dst["reference_ms"].append(r.get("baseline_ms"))
    return {
        "cases": list(rows.values()),
        "metric": "FastKernels native candidate/reference latency",
        "correctness_seeds": "owned by the pinned FastKernels evaluator",
    }


def synthetic(cfg, candidate, output):
    """Deterministic plumbing fixture. Not GPU measurement and not evidence of agent efficacy."""
    data = read_json(candidate)
    return {
        "synthetic": True,
        "cases": [
            {
                "name": c,
                "correct": data.get("correct") is True,
                "supported": True,
                "candidate_ms": [data["latency_ms"]] * cfg["repeats"],
                "reference_ms": [1.0] * cfg["repeats"],
            }
            for c in cfg["cases"]
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    cfg = read_json(a.config)
    data = {"schema_version": 1, "candidate_sha256": digest(a.candidate)}
    try:
        data.update(
            {"flash": flash, "fastkernels": fastkernels, "synthetic": synthetic}[cfg["adapter"]](
                cfg, a.candidate.resolve(), a.output.resolve()
            )
        )
    except Exception as e:
        data.update(error=f"{type(e).__name__}: {e}", cases=[])
        write_json(a.output, data)
        raise
    write_json(a.output, data)


if __name__ == "__main__":
    main()
