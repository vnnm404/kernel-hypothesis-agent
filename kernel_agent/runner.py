from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .backends import invoke
from .common import ROOT, append_event, digest, gpu_lease, read_json, run_process, write_json
from .protocol import promotion, score, validate_evaluation, validate_outcome, validate_plan


def fingerprint():
    paths = [
        *sorted((ROOT / "kernel_agent").glob("*.py")),
        *sorted(
            p for p in (ROOT / "vendor").rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".cu", ".h"}
        ),
        *sorted((ROOT / "prompts").glob("*.md")),
        ROOT / "skills/test-kernel-hypotheses/SKILL.md",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def evaluate(cfg, candidate, dest, gpu, locks, deadline):
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "config.json", cfg)
    # Evaluate a controller-owned immutable snapshot, not a file being edited by an agent.
    snapshot = dest / ("candidate" + Path(candidate).suffix)
    shutil.copy2(candidate, snapshot)
    before = digest(snapshot)
    started = time.monotonic()
    lease = contextlib.nullcontext() if cfg["adapter"] == "synthetic" else gpu_lease(gpu, locks, deadline)
    with lease:
        argv = [
            sys.executable,
            "-m",
            "kernel_agent.evaluator",
            "--config",
            str(dest / "config.json"),
            "--candidate",
            str(snapshot),
            "--output",
            str(dest / "result.json"),
        ]
        env = os.environ | cfg["environment"] | {"CUDA_VISIBLE_DEVICES": gpu}
        proc = run_process(
            argv,
            ROOT,
            env,
            min(cfg["eval_timeout"], deadline - time.monotonic()),
            dest / "stdout.log",
            dest / "stderr.log",
        )
    record = {
        "process": proc,
        "wall_seconds": time.monotonic() - started,
        "candidate_sha256": before,
        "valid": False,
        "score": None,
    }
    try:
        if proc["returncode"]:
            raise ValueError(f"evaluation process failed ({proc['returncode']}); see stderr.log")
        if digest(snapshot) != before:
            raise ValueError("candidate changed during evaluation")
        data = read_json(dest / "result.json")
        if data.get("candidate_sha256") != before:
            raise ValueError("evaluation candidate hash mismatch")
        validate_evaluation(data, cfg["cases"], cfg["repeats"])
        record.update(valid=True, score=score(data), data=data)
    except (ValueError, OSError, KeyError, TypeError) as e:
        record["error"] = str(e)
    write_json(dest / "evaluation.json", record)
    return record


def tool_version(name):
    try:
        p = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=10)
        return p.stdout.strip()[:1000] or p.stderr.strip()[:1000]
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def environment_record(cfg):
    """Versions only; deliberately excludes environment variables and package source URLs."""
    data = {
        "packages": sorted(
            {
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            }
        ),
        "nvcc": tool_version("nvcc"),
        "ncu": tool_version("ncu"),
    }
    if cfg.get("fk_root"):
        result = subprocess.run(
            ["git", "-C", cfg["fk_root"], "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        data["fastkernels_commit"] = result.stdout.strip() or "unknown"
    return data


def workspace(dest, cfg, gpu, locks):
    work = dest / "workspace"
    work.mkdir()
    (work / "evidence").mkdir()
    (work / "history").mkdir()
    write_json(work / "task.json", cfg)
    # Do not expose retrospective traces or successful patches as agent resources.
    shared = ROOT / "vendor/ncu-report-skill"
    if shared.exists():
        shutil.copytree(shared, work / "resources/ncu-report-skill")
    for source in cfg.get("resources", []):
        src = Path(source).resolve()
        if not src.is_dir():
            raise ValueError(f"missing shared resource directory: {src}")
        shutil.copytree(src, work / "resources" / src.name, ignore=shutil.ignore_patterns(".git"))
    text = (ROOT / "prompts/common.md").read_text()
    (work / "AGENTS.md").write_text(text)
    (work / "CLAUDE.md").write_text(text)
    # Python argv, never shell interpolation. Wrapper logs every leased diagnostic command.
    wrapper = f"""import sys
from kernel_agent.cli import main
raise SystemExit(main(["gpu", "--gpu", {gpu!r}, "--lock-dir", {str(locks)!r},
    "--timeout", "600", "--out", "evidence", "--", *sys.argv[1:]]))
"""
    (work / "with_gpu.py").write_text(wrapper)
    (work / "profile.py").write_text("""import sys
from kernel_agent.probe import main
main()
""")
    return work


def run_trial(
    cfg,
    dest,
    *,
    arm,
    backend,
    model,
    extra=(),
    gpu="0",
    locks="/tmp/kha-gpu-locks",
    repetition=0,
    token_limit=None,
):
    if arm not in {"baseline", "hypothesis", "rethink"}:
        raise ValueError("unknown arm")
    if (backend == "fake") != (cfg["adapter"] == "synthetic"):
        raise ValueError("fake backend and synthetic evaluator must be used together")
    if not model and backend != "fake":
        raise ValueError("explicit --model required")
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + cfg["wall_seconds"]
    events = dest / "events.jsonl"
    state = {
        "round": 0,
        "plateau_count": 0,
        "diagnostic_due": False,
        "best_score": None,
        "best_candidate": None,
        "completed_rounds": 0,
        "status": "running",
        "usage_known": True,
        "total_tokens": 0,
        "arm": arm,
        "synthetic": backend == "fake",
    }
    pinned = fingerprint()
    manifest = {
        "schema_version": 1,
        "config": cfg,
        "seed_sha256": digest(cfg["seed"]),
        "arm": arm,
        "backend": backend,
        "model": model,
        "agent_args": list(extra),
        "repetition": repetition,
        "gpu": gpu,
        "lock_dir": str(locks),
        "token_limit": token_limit,
        "python": sys.version,
        "platform": platform.platform(),
        "cli_version": tool_version(backend) if backend != "fake" else "synthetic",
        "source_hashes": pinned,
        "synthetic": backend == "fake",
        "started_unix": time.time(),
    }
    write_json(dest / "manifest.json", manifest)
    write_json(dest / "environment.json", environment_record(cfg))
    work = workspace(dest, cfg, gpu, locks)
    candidate_name = "candidate" + Path(cfg["seed"]).suffix
    best = dest / ("best" + Path(cfg["seed"]).suffix)
    shutil.copy2(cfg["seed"], best)

    def emit(kind, **fields):
        append_event(
            events,
            {"type": kind, "elapsed_seconds": time.monotonic() - start, "round": state["round"], **fields},
        )
        write_json(dest / "state.json", state)

    try:
        initial = evaluate(cfg, best, dest / "evaluations/seed", gpu, locks, deadline)
        emit("seed_evaluation", evaluation=initial)
        if not initial["valid"]:
            state["status"] = "invalid_seed"
            return state
        state.update(best_score=initial["score"], best_candidate=best.name)
        write_json(work / "history/seed.json", initial)
        emit("baseline", score=initial["score"], candidate_sha256=digest(best))
        for number in range(1, cfg["rounds"] + 1):
            if time.monotonic() >= deadline:
                state["status"] = "wall_budget"
                break
            state["round"] = number
            rd = dest / "rounds" / f"{number:03}"
            rd.mkdir(parents=True)
            shutil.copy2(best, work / candidate_name)
            before = digest(best)
            for name in ("plan.json", "outcome.json"):
                (work / name).unlink(missing_ok=True)
            diagnostic_required = arm == "hypothesis" and state["diagnostic_due"]
            public_state = {k: v for k, v in state.items() if k not in {"arm", "synthetic"}}
            write_json(work / "state.json", public_state | {"round": number, "candidate": candidate_name})
            base_prompt = (ROOT / "prompts/common.md").read_text()
            base_prompt += "\nTASK CONFIG:\n" + json.dumps(cfg, indent=2)
            base_prompt += f"\nCandidate: {candidate_name}. Read state.json and history/ for prior results.\n"
            intervention = ""
            if diagnostic_required:
                intervention = (ROOT / "skills/test-kernel-hypotheses/SKILL.md").read_text()
            elif arm == "rethink" and state["diagnostic_due"]:
                intervention = (
                    "Progress has stalled. Reconsider your assumptions and try a different approach."
                )
            emit("round_started", diagnostic_required=diagnostic_required, parent_sha256=before)
            try:
                for phase in ("plan", "execute"):
                    if token_limit and state["total_tokens"] >= token_limit:
                        state["status"] = "token_budget"
                        break
                    prompt = (
                        base_prompt + "\n" + intervention + "\n" + (ROOT / f"prompts/{phase}.md").read_text()
                    )
                    call = invoke(
                        backend,
                        model,
                        extra,
                        phase,
                        work,
                        rd / phase,
                        prompt,
                        min(cfg["agent_timeout"], deadline - time.monotonic()),
                        diagnostic_required,
                    )
                    usage = call.get("usage")
                    if usage is None:
                        state["usage_known"] = False
                    else:
                        state["total_tokens"] += usage["total_tokens"]
                    emit("agent_call", phase=phase, invocation=call)
                    if call["returncode"]:
                        raise ValueError(f"{phase} agent failed; see rounds/{number:03}/{phase}/")
                    if fingerprint() != pinned:
                        state["status"] = "source_modified"
                        raise RuntimeError("runner/evaluator source changed; run is invalid")
                    if digest(best) != before:
                        state["status"] = "source_modified"
                        raise RuntimeError("controller incumbent was modified outside promotion")
                    if token_limit and usage is None:
                        state["status"] = "usage_unavailable"
                        break
                    if phase == "plan":
                        if digest(work / candidate_name) != before:
                            raise ValueError("candidate changed before plan was preregistered")
                        plan = validate_plan(read_json(work / "plan.json"), diagnostic_required)
                        write_json(rd / "plan.json", plan)
                        emit("plan_registered", plan=plan)
                    else:
                        if read_json(work / "plan.json") != plan:
                            raise ValueError("preregistered plan was modified during execution")
                        outcome = validate_outcome(
                            read_json(work / "outcome.json"), plan, work, diagnostic_required
                        )
                        write_json(rd / "outcome.json", outcome)
                if state["status"] != "running":
                    break
                record = {"plan": plan, "outcome": outcome}
                if plan["kind"] == "diagnostic":
                    # Diagnostic code may be intentionally incorrect. Never promote it.
                    record["decision"] = {"accepted": False, "reason": "diagnostic only"}
                    state["diagnostic_due"] = False
                    state["plateau_count"] = 0
                else:
                    snapshot = rd / candidate_name
                    shutil.copy2(work / candidate_name, snapshot)
                    if digest(snapshot) == before:
                        record["decision"] = {"accepted": False, "reason": "candidate unchanged"}
                    else:
                        # Re-measure incumbent and candidate on the same GPU, alternating order.
                        results = {}
                        order = ["parent", "candidate"] if number % 2 else ["candidate", "parent"]
                        for label in order:
                            source = best if label == "parent" else snapshot
                            results[label] = evaluate(
                                cfg, source, dest / f"evaluations/r{number:03}-{label}", gpu, locks, deadline
                            )
                            emit("evaluation", role=label, evaluation=results[label])
                            if fingerprint() != pinned:
                                state["status"] = "source_modified"
                                raise RuntimeError("source changed during evaluation; run is invalid")
                        record["evaluations"] = results
                        if not all(v["valid"] for v in results.values()):
                            record["decision"] = {"accepted": False, "reason": "evaluation invalid"}
                        else:
                            record["decision"] = promotion(
                                results["parent"]["data"], results["candidate"]["data"], cfg
                            )
                            if record["decision"]["accepted"]:
                                shutil.copy2(snapshot, best)
                                state["best_score"] = results["candidate"]["score"]
                    if record["decision"]["accepted"]:
                        state["plateau_count"] = 0
                        state["diagnostic_due"] = False
                    else:
                        state["plateau_count"] += 1
                        state["diagnostic_due"] = state["plateau_count"] >= cfg["plateau_rounds"]
                state["completed_rounds"] += 1
                record.update(best_score=state["best_score"], best_sha256=digest(best))
                write_json(rd / "record.json", record)
                write_json(work / "history" / f"round-{number:03}.json", record)
                emit("round_completed", **record)
            except (ValueError, OSError, KeyError, TypeError) as e:
                # Failed plans/calls consume rounds and wall budget; no hidden free retries.
                emit("round_failed", error=str(e))
                write_json(work / "history" / f"round-{number:03}-failure.json", {"error": str(e)})
            finally:
                # Preserve raw observations even when parsing or validation failed.
                if (work / candidate_name).is_file():
                    shutil.copy2(work / candidate_name, rd / ("attempted" + Path(candidate_name).suffix))
                for artifact in ("plan.json", "outcome.json"):
                    if (work / artifact).is_file():
                        shutil.copy2(work / artifact, rd / ("workspace-" + artifact))
                if (work / "evidence").exists():
                    shutil.move(str(work / "evidence"), str(rd / "evidence"))
                    (work / "evidence").mkdir()
                    (work / "history" / f"evidence-{number:03}").symlink_to(
                        os.path.relpath(rd / "evidence", work / "history"), target_is_directory=True
                    )
            if token_limit and not state["usage_known"]:
                state["status"] = "usage_unavailable"
                break
        if state["status"] == "running":
            state["status"] = "round_budget"
    except TimeoutError as e:
        state["status"] = "wall_budget"
        emit("budget_exhausted", error=str(e))
    except KeyboardInterrupt:
        state["status"] = "interrupted"
    except Exception as e:
        state["status"] = "failed" if state["status"] == "running" else state["status"]
        emit("fatal_error", error=f"{type(e).__name__}: {e}")
    finally:
        state["elapsed_seconds"] = time.monotonic() - start
        emit("run_finished", **state)
        write_json(dest / "summary.json", state)
    return state


def campaign(cfg, dest, *, arms, repetitions, order_seed, **kwargs):
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    rng = random.Random(order_seed)
    schedule = []
    for rep in range(repetitions):
        order = list(arms)
        rng.shuffle(order)
        schedule += [{"arm": arm, "repetition": rep} for arm in order]
    write_json(dest / "schedule.json", {"order_seed": order_seed, "schedule": schedule})
    states = []
    for item in schedule:
        trial = dest / f"rep-{item['repetition']:02}-{item['arm']}"
        print(f"Starting {trial.name}", flush=True)
        states.append(run_trial(cfg, trial, **item, **kwargs))
    write_json(dest / "campaign.json", states)
    return states
