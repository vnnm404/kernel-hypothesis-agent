"""Reuse compiled kernels across fresh evaluator processes, never reuse measurements."""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .common import ROOT, digest, read_json, write_json


def build(torch, evaluator, candidate, namespace, target, cache):
    candidate = Path(candidate)
    identity = {
        "source": digest(candidate),
        "namespace": namespace,
        "target": repr(target),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "harness": {
            name: digest(ROOT / "vendor/flash_cuda/bench" / name) for name in ("eval.py", "scenarios.py")
        },
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(cache).resolve() / key
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / f"{namespace}.so"
    manifest = folder / "artifact.json"
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists() and library.exists():
            record = read_json(manifest)
            if record.get("identity") == identity and record.get("sha256") == digest(library):
                torch.ops.load_library(str(library))
                return {"key": key, "hit": True, "library_sha256": digest(library)}
        # A mismatched artifact must not be reused by Ninja either.
        library.unlink(missing_ok=True)
        source = folder / "candidate.cu"
        shutil.copy2(candidate, source)
        evaluator.build_candidate(str(source), namespace, str(folder), target)
        write_json(manifest, {"identity": identity, "sha256": digest(library)})
    return {"key": key, "hit": False, "library_sha256": digest(library)}
