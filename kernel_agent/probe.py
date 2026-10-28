"""FlashAttention profiling driver; run through with_gpu.py and NCU."""

import argparse
import os
import sys
from pathlib import Path

from .common import ROOT, digest, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--case", required=True)
    p.add_argument("--which", choices=["candidate", "reference"], default="candidate")
    a = p.parse_args()
    cfg = read_json(a.config)
    os.environ.update(cfg.get("environment", {}))
    if cfg["adapter"] != "flash" or a.case not in cfg["cases"]:
        raise ValueError("profile driver requires a configured FlashAttention case")
    sys.path.insert(0, str(ROOT / "vendor/flash_cuda"))
    import torch
    from bench import eval as E
    from bench import scenarios as S

    torch.cuda.set_device(0)
    target = E.TARGETS[cfg["target"]]
    case = next(c for c in S.get_configs("all") if c.name == a.case)
    inp = S.make_inputs(case, seed=0, device="cuda:0")
    if a.which == "candidate":
        namespace = "_kha_probe_" + digest(a.candidate)[:16]
        E.build_candidate(
            str(a.candidate.resolve()), namespace, str(Path("evidence/build").resolve()), target
        )

        def run():
            S.call_fwd(namespace, inp)
    else:
        if not E.baseline_available(target):
            raise RuntimeError("reference unavailable")

        def run():
            E.call_baseline(inp, target, normalize_lse=False)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
