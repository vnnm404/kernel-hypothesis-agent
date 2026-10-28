import json
import os
import subprocess
import sys
import time

import pytest

from kernel_agent import backends, runner
from kernel_agent.common import ROOT, load_config, read_json, run_process, write_json


def config():
    return load_config(ROOT / "experiments/synthetic.json")


def events(out):
    return [json.loads(s) for s in (out / "events.jsonl").read_text().splitlines()]


def test_plateau_intervention_diagnostic_not_promoted_and_history_preserved(tmp_path):
    out = tmp_path / "trial"
    state = runner.run_trial(config(), out, arm="hypothesis", backend="fake", model="fixture")
    assert state["completed_rounds"] == 5
    records = [e for e in events(out) if e["type"] == "round_completed"]
    assert records[2]["plan"]["kind"] == "diagnostic"
    assert not records[2]["decision"]["accepted"]
    assert records[3]["decision"]["accepted"]
    assert (out / "rounds/003/evidence/fixture.txt").is_file()
    assert (out / "workspace/history/round-003.json").is_file()
    assert state["usage_known"] is False  # no pretend token usage for fake backend
    with pytest.raises(FileExistsError):
        runner.run_trial(config(), out, arm="baseline", backend="fake", model="fixture")


def test_incorrect_seed_stops_before_any_agent_call(tmp_path):
    cfg = config()
    seed = tmp_path / "bad.json"
    write_json(seed, {"correct": False, "latency_ms": 0.01})
    cfg["seed"] = str(seed)
    state = runner.run_trial(cfg, tmp_path / "run", arm="baseline", backend="fake", model="fixture")
    assert state["status"] == "invalid_seed"
    assert not any(e["type"] == "agent_call" for e in events(tmp_path / "run"))


def test_rewriting_preregistered_plan_invalidates_round(tmp_path, monkeypatch):
    original = backends.fake

    def dishonest(phase, work, diagnostic_required):
        original(phase, work, diagnostic_required)
        if phase == "execute":
            plan = read_json(work / "plan.json")
            plan["change"] = "changed after observing result"
            write_json(work / "plan.json", plan)

    monkeypatch.setattr(backends, "fake", dishonest)
    cfg = config() | {"rounds": 1}
    state = runner.run_trial(cfg, tmp_path / "run", arm="baseline", backend="fake", model="fixture")
    assert state["completed_rounds"] == 0
    assert any(e["type"] == "round_failed" for e in events(tmp_path / "run"))


def test_diagnostic_fast_invalid_code_cannot_be_promoted(tmp_path, monkeypatch):
    original = backends.fake

    def fake_diagnostic(phase, work, diagnostic_required):
        original(phase, work, diagnostic_required)
        if phase == "execute" and diagnostic_required:
            write_json(work / "candidate.json", {"correct": False, "latency_ms": 0.001})

    monkeypatch.setattr(backends, "fake", fake_diagnostic)
    state = runner.run_trial(
        config() | {"rounds": 3}, tmp_path / "run", arm="hypothesis", backend="fake", model="fixture"
    )
    assert state["best_score"] == 0.5
    assert read_json(tmp_path / "run/best.json")["correct"] is True


def test_token_cap_stops_when_usage_is_unknown(tmp_path):
    state = runner.run_trial(
        config(), tmp_path / "run", arm="baseline", backend="fake", model="fixture", token_limit=100
    )
    assert state["status"] == "usage_unavailable"


def test_timeout_kills_owned_child_process(tmp_path):
    pidfile = tmp_path / "pid"
    program = "import subprocess,time,pathlib; p=subprocess.Popen(['sleep','30']); pathlib.Path('pid').write_text(str(p.pid)); time.sleep(30)"
    result = run_process(
        [sys.executable, "-c", program],
        tmp_path,
        os.environ.copy(),
        0.5,
        tmp_path / "stdout",
        tmp_path / "stderr",
    )
    assert result["timed_out"]
    pid = int(pidfile.read_text())
    # A zombie may still await OS reaping; it must no longer be running.
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert not result.stdout.strip() or result.stdout.strip().startswith("Z")


def test_gpu_lease_timeout(tmp_path):
    import fcntl

    from kernel_agent.common import gpu_lease

    with (tmp_path / "gpu_0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            with gpu_lease("0", tmp_path, time.monotonic() + 0.1):
                pytest.fail("must not enter an occupied lease")
