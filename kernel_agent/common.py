from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_event(path, event):
    with Path(path).open("a") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def geomean(values):
    values = list(values)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("geomean requires nonempty finite positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def run_process(argv, cwd, env, timeout, stdout, stderr, stdin=None):
    """No shell interpolation; kill this invocation's process group on timeout/interruption."""
    if timeout <= 0:
        raise TimeoutError("run deadline exhausted")
    started = time.monotonic()
    with Path(stdout).open("w") as out, Path(stderr).open("w") as err:
        p = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        timed_out = False

        def terminated(signum, frame):
            raise InterruptedError("process interrupted by SIGTERM")

        old_handler = signal.signal(signal.SIGTERM, terminated)
        try:
            p.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also reap stray children after a parent exits normally (e.g. background profiling).
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            # Give nested lease wrappers time to clean up their own subprocesses,
            # even when the immediate CLI parent has already exited.
            grace = time.monotonic() + 2
            while time.monotonic() < grace:
                try:
                    os.killpg(p.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            signal.signal(signal.SIGTERM, old_handler)
    return {
        "returncode": 124 if timed_out else p.returncode,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
    }


@contextlib.contextmanager
def gpu_lease(gpu, lock_dir, deadline):
    """Cooperative per-device lease, shared by evaluation and diagnostic commands."""
    if not gpu or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in gpu
    ):
        raise ValueError("gpu must be a single physical index or GPU UUID")
    path = Path(lock_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"gpu_{gpu}.lock").open("a") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU lease deadline exhausted")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    for field in ("name", "adapter", "seed", "cases"):
        if not cfg.get(field):
            raise ValueError(f"missing config field: {field}")
    if cfg["adapter"] not in {"flash", "fastkernels", "synthetic"}:
        raise ValueError("unknown adapter")
    for field in ("seed", "captures", "fk_root"):
        if cfg.get(field):
            p = Path(os.path.expandvars(cfg[field])).expanduser()
            cfg[field] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    if not Path(cfg["seed"]).is_file():
        raise ValueError(f"missing seed: {cfg['seed']}")
    if not isinstance(cfg["cases"], list) or len(cfg["cases"]) != len(set(cfg["cases"])):
        raise ValueError("cases must be a nonempty unique list")
    defaults = {
        "rounds": 8,
        "wall_seconds": 14400,
        "agent_timeout": 1200,
        "eval_timeout": 1800,
        "repeats": 3,
        "correctness_seeds": [0, 1, 2],
        "plateau_rounds": 2,
        "min_gain": 0.02,
        "max_regression": 0.03,
        "target_speedup": 1.0,
        "environment": {},
    }
    cfg = defaults | cfg
    for field in ("rounds", "wall_seconds", "agent_timeout", "eval_timeout", "repeats", "plateau_rounds"):
        if not isinstance(cfg[field], int) or isinstance(cfg[field], bool) or cfg[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("min_gain", "max_regression"):
        if not isinstance(cfg[field], (int, float)) or not 0 <= cfg[field] < 1:
            raise ValueError(f"{field} must be in [0, 1)")
    if not math.isfinite(cfg["target_speedup"]) or cfg["target_speedup"] <= 0:
        raise ValueError("target_speedup must be finite and positive")
    if not cfg["correctness_seeds"] or any(type(s) is not int for s in cfg["correctness_seeds"]):
        raise ValueError("correctness_seeds must be a nonempty integer list")
    if cfg["adapter"] == "flash" and cfg.get("target") not in {"hopper", "blackwell"}:
        raise ValueError("flash target must be hopper or blackwell")
    cfg["resources"] = [
        str((path.parent / Path(os.path.expandvars(p)).expanduser()).resolve())
        for p in cfg.get("resources", [])
    ]
    # Only kernel toggles belong in this file; credentials are inherited, never serialized.
    if any(
        not k.startswith(("FA3_", "FA4_")) or not isinstance(v, str) for k, v in cfg["environment"].items()
    ):
        raise ValueError("environment supports only string FA3_/FA4_ kernel toggles")
    return cfg
