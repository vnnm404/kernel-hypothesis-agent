from pathlib import Path
from types import SimpleNamespace

from kernel_agent import build_cache


def test_cache_reuses_binary_but_rebuilds_changed_or_corrupt_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(build_cache.subprocess, "check_output", lambda *a, **k: "nvcc test")
    builds, loads = [], []

    def compile(source, namespace, folder, target):
        builds.append(Path(source).read_text())
        (Path(folder) / f"{namespace}.so").write_text("compiled " + Path(source).read_text())

    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="13.0"),
        ops=SimpleNamespace(load_library=loads.append),
    )
    evaluator = SimpleNamespace(build_candidate=compile)
    source = tmp_path / "candidate.cu"
    source.write_text("first")
    args = (torch, evaluator, source, "_test", "target", tmp_path / "cache")
    a = build_cache.build(*args)
    b = build_cache.build(*args)
    assert not a["hit"] and b["hit"] and a["key"] == b["key"]
    assert len(builds) == 1 and len(loads) == 1
    Path(loads[0]).write_text("corrupt")
    assert not build_cache.build(*args)["hit"]
    source.write_text("second")
    c = build_cache.build(*args)
    assert not c["hit"] and c["key"] != a["key"]
    torch.__version__ = "other-version"
    assert build_cache.build(*args)["key"] != c["key"]
    assert builds == ["first", "first", "second", "second"]
