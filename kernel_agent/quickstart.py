"""Small, bounded entry points for first-time GPU users."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .cli import main as cli
from .common import ROOT, load_config, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description="GPU check, short agent pilot, or full campaign")
    p.add_argument("mode", choices=["check", "pilot", "campaign", "smoke"])
    p.add_argument("--config", type=Path, default=ROOT / "experiments/blackwell.json")
    p.add_argument("--backend", choices=["claude", "codex"], default=os.environ.get("BACKEND", "codex"))
    p.add_argument("--model", default=os.environ.get("KHA_MODEL"))
    p.add_argument("--gpu", default=os.environ.get("KHA_GPU", "0"))
    p.add_argument("--out", type=Path)
    p.add_argument("--rounds", type=int, default=3, help="pilot only; 3 reaches diagnosis after 2 stalls")
    p.add_argument("--minutes", type=int, default=15, help="pilot wall-clock budget")
    p.add_argument("--repetitions", type=int, default=3, help="campaign only")
    p.add_argument("--agent-arg", action="append", default=[])
    a = p.parse_args(argv)
    if a.mode in {"pilot", "campaign"} and not a.model:
        p.error("set KHA_MODEL or pass --model; use your already-authenticated coding CLI")
    if min(a.rounds, a.minutes, a.repetitions) < 1:
        p.error("rounds, minutes, repetitions must be positive")
    out = (a.out or ROOT / "runs" / f"{a.mode}-{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    if a.mode == "smoke":
        return cli(["smoke", "--out", str(out)])
    if out.exists():
        p.error(f"output already exists: {out}")
    cfg = load_config(a.config)
    if a.mode == "pilot":
        cfg.update(rounds=a.rounds, wall_seconds=60 * a.minutes, agent_timeout=240, eval_timeout=600)
    out.mkdir(parents=True)
    config = out / "config.json"
    write_json(config, cfg)
    base = ["--config", str(config), "--gpu", a.gpu]
    if a.mode == "check":
        return cli(["evaluate", *base, "--out", str(out / "seed")])
    if cli(["doctor", *base, "--backend", a.backend]):
        return 1
    args = ["--backend", a.backend, "--model", a.model]
    args += [f"--agent-arg={v}" for v in a.agent_arg]
    if a.mode == "pilot":
        return cli(["run", *base, *args, "--arm", "hypothesis", "--out", str(out / "trial")])
    return cli(
        ["campaign", *base, *args, "--repetitions", str(a.repetitions), "--out", str(out / "campaign")]
    )


if __name__ == "__main__":
    raise SystemExit(main())
