from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from .common import ROOT, gpu_lease, load_config, read_json, run_process, write_json
from .evaluator import fk_case_id
from .report import build_report
from .runner import campaign, evaluate, run_trial, tool_version


def doctor(cfg, backend, gpu):
    checks = {
        "python": sys.version,
        "backend": backend,
        "cli": tool_version(backend),
        "nvcc": tool_version("nvcc"),
        "ncu": tool_version("ncu"),
        "gpu": gpu,
    }
    if cfg["adapter"] == "synthetic":
        return checks | {"ok": backend == "fake", "synthetic": True}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import torch

        checks.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
        )
        if cfg["adapter"] == "flash":
            sys.path.insert(0, str(ROOT / "vendor/flash_cuda"))
            from bench import eval as E
            from bench import scenarios as S

            checks["reference_available"] = E.baseline_available(E.TARGETS[cfg["target"]])
            checks["missing_cases"] = sorted(set(cfg["cases"]) - {c.name for c in S.get_configs("all")})
            checks["architecture_matches"] = (
                checks["capability"][0] == {"hopper": 9, "blackwell": 10}[cfg["target"]]
            )
            checks["ok"] = (
                checks["reference_available"]
                and checks["architecture_matches"]
                and not checks["missing_cases"]
                and shutil.which("nvcc") is not None
            )
        else:
            import fastkernels

            checks["fastkernels"] = str(Path(fastkernels.__file__).resolve())
            checks["ok"] = Path(cfg["captures"]).is_dir() and Path(cfg["fk_root"]).is_dir()
        checks["ok"] &= shutil.which(backend) is not None
    except Exception as e:
        checks.update(ok=False, error=f"{type(e).__name__}: {e}")
    return checks


def prepare_fk(args):
    # Capture report is used only to freeze case identities, not to initialize a claimed score.
    rows = read_json(args.report).get("scenarios", [])
    prefix = f"fastkernels.tasks.baseline.L{args.level}.{args.stem}:"
    rows = [r for r in rows if r.get("op", "").startswith(prefix)]
    if not rows:
        raise ValueError("no matching scenarios in report")
    cases = [fk_case_id(r) for r in rows]
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate case identities; choose a report with unique op/shape/dtype rows")
    cfg = {
        "name": f"fk-L{args.level}-{args.stem}",
        "adapter": "fastkernels",
        "level": args.level,
        "stem": args.stem,
        "fk_root": str(args.fk.resolve()),
        "captures": str(args.captures.resolve()),
        "seed": str(args.seed.resolve()),
        "cases": cases,
        "rounds": 8,
        "wall_seconds": 14400,
        "repeats": 3,
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
    }
    if args.out.exists():
        raise ValueError("output already exists")
    write_json(args.out, cfg)
    load_config(args.out)
    print(args.out)


def parser():
    p = argparse.ArgumentParser(description="Controlled hypothesis-driven kernel optimization")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "campaign", "evaluate", "doctor"):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--gpu", default=os.environ.get("KHA_GPU", "0"))
        q.add_argument("--lock-dir", default=os.environ.get("KHA_GPU_LOCK_DIR", "/tmp/kha-gpu-locks"))
        if name != "evaluate":
            q.add_argument("--backend", choices=["claude", "codex", "fake"], default="claude")
        if name in {"run", "campaign"}:
            q.add_argument("--model", default=os.environ.get("KHA_MODEL"))
            q.add_argument(
                "--agent-arg", action="append", default=[], help="CLI argument; use --agent-arg=--flag"
            )
            q.add_argument(
                "--token-limit", type=int, help="soft cap checked between calls; unknown usage stops run"
            )
        if name != "doctor":
            q.add_argument("--out", required=True, type=Path)
        if name == "run":
            q.add_argument("--arm", choices=["baseline", "hypothesis", "rethink"], default="hypothesis")
        if name == "campaign":
            q.add_argument(
                "--arms",
                nargs="+",
                choices=["baseline", "hypothesis", "rethink"],
                default=["baseline", "hypothesis"],
            )
            q.add_argument("--repetitions", type=int, default=3)
            q.add_argument("--order-seed", type=int, default=2027)
        if name == "evaluate":
            q.add_argument("--candidate", type=Path)
    q = sub.add_parser("smoke", help="no GPU, model, credentials or network calls")
    q.add_argument("--out", type=Path, default=Path("runs/smoke"))
    q = sub.add_parser("report")
    q.add_argument("path", type=Path)
    q = sub.add_parser("gpu", help="lease a GPU for one diagnostic command")
    q.add_argument("--gpu", required=True)
    q.add_argument("--lock-dir", default="/tmp/kha-gpu-locks")
    q.add_argument("--timeout", type=int, default=600)
    q.add_argument("--out", type=Path, default=Path("evidence"))
    q.add_argument("command", nargs=argparse.REMAINDER)
    q = sub.add_parser("prepare-fk", help="freeze FastKernels task identities from an existing bench report")
    for flag in ("fk", "captures", "seed", "report", "out"):
        q.add_argument(f"--{flag}", required=True, type=Path)
    q.add_argument("--level", required=True, type=int, choices=[1, 2, 3, 4])
    q.add_argument("--stem", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.cmd == "report":
            print(build_report(a.path))
            return 0
        if a.cmd == "prepare-fk":
            prepare_fk(a)
            return 0
        if a.cmd == "gpu":
            command = a.command[1:] if a.command[:1] == ["--"] else a.command
            if not command or a.timeout < 1:
                raise ValueError("provide a command and positive timeout")
            output = a.out.resolve() / f"command-{uuid.uuid4().hex[:12]}"
            output.mkdir(parents=True)
            write_json(output / "command.json", command)
            deadline = time.monotonic() + a.timeout
            with gpu_lease(a.gpu, a.lock_dir, deadline):
                result = run_process(
                    command,
                    Path.cwd(),
                    os.environ | {"CUDA_VISIBLE_DEVICES": a.gpu},
                    deadline - time.monotonic(),
                    output / "stdout.log",
                    output / "stderr.log",
                )
            write_json(output / "process.json", result)
            print((output / "stdout.log").read_text(errors="replace"))
            print((output / "stderr.log").read_text(errors="replace"), file=sys.stderr)
            print(f"Evidence: {output}", file=sys.stderr)
            return result["returncode"]
        if a.cmd == "smoke":
            cfg = load_config(ROOT / "experiments/synthetic.json")
            campaign(
                cfg,
                a.out,
                arms=["baseline", "hypothesis", "rethink"],
                repetitions=1,
                order_seed=2027,
                backend="fake",
                model="synthetic-fixture",
            )
            print(build_report(a.out))
            summaries = list(a.out.glob("*/summary.json"))
            good = all(read_json(p)["completed_rounds"] == cfg["rounds"] for p in summaries)
            return 0 if len(summaries) == 3 and good else 1
        cfg = load_config(a.config)
        if a.cmd == "doctor":
            result = doctor(cfg, a.backend, a.gpu)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if a.cmd == "evaluate":
            result = evaluate(
                cfg,
                a.candidate or cfg["seed"],
                a.out.resolve(),
                a.gpu,
                a.lock_dir,
                time.monotonic() + cfg["eval_timeout"],
            )
            print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
            return 0 if result["valid"] else 1
        if not a.model and a.backend != "fake":
            raise ValueError("set KHA_MODEL or pass --model with your provider's explicit model ID")
        if a.token_limit is not None and a.token_limit <= 0:
            raise ValueError("token limit must be positive")
        opts = dict(
            backend=a.backend,
            model=a.model,
            extra=a.agent_arg,
            gpu=a.gpu,
            locks=a.lock_dir,
            token_limit=a.token_limit,
        )
        if a.cmd == "run":
            state = run_trial(cfg, a.out, arm=a.arm, **opts)
            print(build_report(a.out))
            return 0 if state["status"] in {"round_budget", "wall_budget", "token_budget"} else 1
        if a.repetitions < 1 or len(set(a.arms)) != len(a.arms):
            raise ValueError("positive repetitions and unique arms required")
        states = campaign(cfg, a.out, arms=a.arms, repetitions=a.repetitions, order_seed=a.order_seed, **opts)
        print(build_report(a.out))
        return 0 if all(s["status"] in {"round_budget", "wall_budget", "token_budget"} for s in states) else 1
    except (ValueError, OSError, TimeoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
