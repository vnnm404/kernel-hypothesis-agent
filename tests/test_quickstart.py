from pathlib import Path

import pytest

from kernel_agent import quickstart
from kernel_agent.common import read_json


def test_check_needs_no_agent_or_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(quickstart, "cli", lambda args: calls.append(args) or 0)
    assert quickstart.main(["check", "--out", str(tmp_path / "run")]) == 0
    assert len(calls) == 1 and calls[0][0] == "evaluate"
    assert "--backend" not in calls[0]
    cfg = read_json(tmp_path / "run/config.json")
    assert len(cfg["cases"]) == 3 and cfg["correctness_seeds"] == [0, 1, 2]
    assert Path(cfg["seed"]).is_absolute()


def test_pilot_budget_and_provider_args(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(quickstart, "cli", lambda args: calls.append(args) or 0)
    assert (
        quickstart.main(
            [
                "pilot",
                "--model",
                "explicit-test",
                "--rounds",
                "1",
                "--minutes",
                "10",
                "--agent-arg=--profile",
                "--agent-arg=team",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    cfg = read_json(tmp_path / "run/config.json")
    assert cfg["rounds"] == 1 and cfg["wall_seconds"] == 600
    assert [c[0] for c in calls] == ["doctor", "run"]
    assert "--agent-arg=--profile" in calls[1]
    assert "--agent-arg=team" in calls[1]


def test_missing_cli_stops_pilot(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(quickstart, "cli", lambda args: calls.append(args) or 1)
    assert quickstart.main(["pilot", "--model", "explicit-test", "--out", str(tmp_path / "run")]) == 1
    assert len(calls) == 1


def test_no_silent_model_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KHA_MODEL", raising=False)
    with pytest.raises(SystemExit):
        quickstart.main(["pilot", "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()
