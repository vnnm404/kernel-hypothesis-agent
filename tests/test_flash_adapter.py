"""Exercise real adapter ordering/coverage with a simulated Torch/bench API (no GPU claims)."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from kernel_agent.evaluator import flash


def modules(monkeypatch, *, wrong=False, baseline=True):
    calls = []
    torch = ModuleType("torch")
    torch.__version__ = "fixture"
    torch.version = SimpleNamespace(cuda="fixture")
    torch.cuda = SimpleNamespace(
        set_device=lambda n: None,
        get_device_capability=lambda n: (10, 0),
        get_device_name=lambda n: "fixture",
    )
    bench = ModuleType("bench")
    target = SimpleNamespace(baseline_label="FA4")

    def check(cfg, namespace, target, **kwargs):
        calls.append(("check", cfg.name, kwargs["seed"]))
        return SimpleNamespace(
            correct=not wrong, unsup_ours=False, unsup_baseline=False, baseline=True, note=""
        )

    def measure(cfg, namespace, gpu, target, **kwargs):
        calls.append(("measure", cfg.name))
        return SimpleNamespace(cand_ms=2.0, base_ms=1.0)

    bench.eval = SimpleNamespace(
        TARGETS={"blackwell": target},
        baseline_available=lambda t: baseline,
        baseline_import_error=lambda t: "missing",
        check_correctness=check,
        measure=measure,
        build_candidate=lambda *a: calls.append(("build",)),
    )
    bench.scenarios = SimpleNamespace(
        get_configs=lambda s: [SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        PERF={"warmup_iters": 20},
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "bench", bench)
    return calls


def config():
    return {"target": "blackwell", "cases": ["a", "b"], "correctness_seeds": [0, 1], "repeats": 3}


def test_every_seed_and_case_checked_before_timing(tmp_path, monkeypatch):
    calls = modules(monkeypatch)
    seed = tmp_path / "candidate.cu"
    seed.write_text("fixture")
    result = flash(config(), seed, tmp_path / "result.json")
    assert calls[:5] == [
        ("build",),
        ("check", "a", 0),
        ("check", "a", 1),
        ("check", "b", 0),
        ("check", "b", 1),
    ]
    assert len([c for c in calls if c[0] == "measure"]) == 6
    assert result["cases"][0]["candidate_ms"] == [2.0, 2.0, 2.0]


def test_wrong_kernel_is_never_timed(tmp_path, monkeypatch):
    calls = modules(monkeypatch, wrong=True)
    seed = tmp_path / "candidate.cu"
    seed.write_text("fixture")
    result = flash(config(), seed, tmp_path / "result.json")
    assert not any(c[0] == "measure" for c in calls)
    assert result["cases"][0]["correct"] is False


def test_reference_unavailable_fails_before_build(tmp_path, monkeypatch):
    calls = modules(monkeypatch, baseline=False)
    with pytest.raises(RuntimeError, match="reference unavailable"):
        flash(config(), Path("unused"), tmp_path / "result.json")
    assert not calls


def test_unknown_case_fails_instead_of_silently_filtering(tmp_path, monkeypatch):
    modules(monkeypatch)
    with pytest.raises(ValueError, match="unknown scenario"):
        flash(config() | {"cases": ["a", "typo"]}, Path("unused"), tmp_path / "result.json")
