from __future__ import annotations

import json
import os
from pathlib import Path

from .common import read_json, run_process, write_json


def command(backend, model, extra):
    if not model:
        raise ValueError("choose an explicit model; no silently changing default")
    if backend == "claude":
        return [
            "claude",
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            model,
            "--allowedTools",
            "Read,Write,Edit,Bash,Glob,Grep",
            *extra,
        ]
    if backend == "codex":
        return [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--model",
            model,
            *extra,
            "-",
        ]
    raise ValueError(f"unsupported backend: {backend}")


def usage_from_events(path, backend):
    """Report unknown usage as null, never as zero. Keep raw events for provider-specific accounting."""
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "cache_creation_tokens": 0}
    seen = False
    cost = None
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(e, dict):
            continue
        if backend == "codex" and e.get("type") == "turn.completed":
            u = e.get("usage", {})
            if not u:
                continue
            seen = True
            for k in ("input_tokens", "output_tokens", "cached_input_tokens"):
                totals[k] += int(u.get(k, 0))
        if backend == "claude" and e.get("type") == "result" and e.get("usage"):
            u = e["usage"]
            seen = True
            # Claude input excludes cache reads/writes; normalize input to include them.
            totals = {
                "input_tokens": sum(
                    int(u.get(k, 0))
                    for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                ),
                "output_tokens": int(u.get("output_tokens", 0)),
                "cached_input_tokens": int(u.get("cache_read_input_tokens", 0)),
                "cache_creation_tokens": int(u.get("cache_creation_input_tokens", 0)),
            }
            cost = e.get("total_cost_usd")
    return (
        (totals | {"total_tokens": totals["input_tokens"] + totals["output_tokens"], "cost_usd": cost})
        if seen
        else None
    )


def fake(phase, work, diagnostic_required):
    """Explicit synthetic backend for integration tests only."""
    if phase == "plan":
        write_json(
            work / "plan.json",
            {
                "kind": "diagnostic" if diagnostic_required else "optimization",
                "question": "Synthetic fixture question",
                "change": "fixture action",
                "expected_result": "deterministic fixture observation",
                "hypotheses": [
                    {
                        "id": k,
                        "mechanism": k,
                        "prediction": k,
                        "falsifier": "opposite observation",
                        "conditions": "synthetic fixture",
                    }
                    for k in ("h1", "h2")
                ],
                "discriminating_test": "fixture only",
                "decision_rule": "fixture only",
            },
        )
    else:
        plan = read_json(work / "plan.json")
        candidate = next(work.glob("candidate.*"))
        state = read_json(work / "state.json")
        if plan["kind"] == "optimization":
            v = read_json(candidate)
            # First two attempts stall; after diagnosis later attempts improve in BOTH arms.
            if state["round"] > 2:
                v["latency_ms"] *= 0.9
            write_json(candidate, v)
        (work / "evidence/fixture.txt").write_text("SYNTHETIC observation; no GPU measurement.\n")
        write_json(
            work / "outcome.json",
            {
                "observation": "fixture observation",
                "interpretation": "fixture interpretation",
                "next_action": "continue fixture",
                "evidence_files": ["evidence/fixture.txt"],
                "hypothesis_updates": [
                    {
                        "id": h["id"],
                        "status": "inconclusive",
                        "evidence": "fixture.txt",
                        "conditions": "synthetic",
                        "revisit_when": "new fixture",
                    }
                    for h in plan.get("hypotheses", [])
                ],
            },
        )


def invoke(backend, model, extra, phase, work, log_dir, prompt, timeout, diagnostic_required):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.txt").write_text(prompt)
    if backend == "fake":
        fake(phase, work, diagnostic_required)
        write_json(log_dir / "synthetic.json", {"synthetic": True, "phase": phase})
        return {"returncode": 0, "timed_out": False, "wall_seconds": 0, "usage": None}
    argv = command(backend, model, extra)
    write_json(log_dir / "command.json", argv)
    # GPU access is via the supplied lease wrapper, not the reasoning process.
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    result = run_process(
        argv, work, env, timeout, log_dir / "events.jsonl", log_dir / "stderr.log", stdin=prompt
    )
    result["usage"] = usage_from_events(log_dir / "events.jsonl", backend)
    for line in (log_dir / "events.jsonl").read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and (
            event.get("type") == "turn.failed"
            or (event.get("type") == "result" and event.get("is_error") is True)
        ):
            result["returncode"] = result["returncode"] or 1
            result["agent_reported_error"] = True
    write_json(log_dir / "invocation.json", result)
    return result
