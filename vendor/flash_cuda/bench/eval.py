"""Self-contained A/B evaluator for the two hand-written attention kernels in this repo.

Builds a candidate `.cu` ONCE into a torch op `.so`, then reports correctness AND performance
against the reference implementation for that architecture, over the shared `bench.scenarios` matrix.
`--target` selects which kernel/baseline pair to run and is REQUIRED:

  hopper     kernels/fa3_hopper.cu     (sm90a)  vs FA3 -- optional `_vllm_fa3_C` torch op (install
                                                 vLLM separately if you want that baseline). Same
                                                 37-arg schema as our op, so both go through `S.call_fwd`.
  blackwell  kernels/fa4_blackwell.cu  (sm100a) vs FA4 -- upstream Dao-AILab CuteDSL `_flash_attn_fwd`
                                                 (`flash-attn-4[cu13]`), via the oracle adapter below.

    cd /home/yak/fa3-cuda
    python -m bench.eval --target hopper                    # all visible GPUs, in parallel
    python -m bench.eval --target hopper --gpus 0,1          # only these two
    python -m bench.eval --target hopper --gpu 0             # single GPU (serial, in-process)
    python -m bench.eval --target hopper --no-perf           # build + correctness only
    python -m bench.eval --target blackwell --gpu 0          # default claim set, ≤15 min warm B200
    python -m bench.eval --target hopper --configs attn_dense_hd128_causal,attn_dense_hd128_fp8
    python -m bench.eval --target blackwell --set all         # union; writes 4 plots (claim/attntest/edge/real)
    python -m bench.eval --target blackwell                 # build fa4_blackwell.cu, A/B vs FA4
    python -m bench.eval --target blackwell --set edge --no-perf   # masking/boundary grid
    python -m bench.eval --target blackwell --baseline-only  # NO build: baseline vs fp32 ref
    python -m bench.eval --target hopper --gpu 1 --cand kernels/fa3_hopper.cu

Scheduling. By default every visible GPU runs scenarios concurrently, one worker process per GPU
(`spawn`, so each has its own CUDA context and its own interpreter -- threads would share the GIL,
and since the timed window includes host-side dispatch, GIL contention would corrupt the numbers).
Workers pull from a shared queue, so a GPU picks up the next scenario as soon as it is free rather
than being handed a fixed shard. **At most one scenario per GPU at any moment**, and a scenario's
baseline and candidate are timed back-to-back on that same GPU -- so the A/B pair never spans devices
and never shares a device with other work. `--gpus`/`--gpu` narrows the set; a single GPU runs
serially in-process, exactly as before parallelism existed. Both paths evaluate a scenario through
the same `_eval_one`, so parallelism changes where and when work runs, never what is measured.

Correctness is judged against the baseline AND a pure-PyTorch fp32 reference (the 2x-error criterion
from `bench.scenarios`, plus coarse allclose/bias/LSE gates). Configs whose baseline cannot run fall
back to an absolute fp32-ref tolerance with perf skipped, tagged `[no-baseline]`: for hopper that is
all-or-nothing (`_vllm_fa3_C` importable or not); for blackwell it is also per-config (e.g.
`cache_leftpad` has no FA4 argument). Homogeneous FP8 (`fp8e4m3`) is supported by the upstream FA4
oracle on SM100 + cutlass-dsl ≥4.6; cuDNN skips homogeneous FP8 (`sdpa_fp8` segfaults here).
cuDNN and FlashInfer are timed extra baselines (never OK/FAIL gates); disable with `--no-cudnn` /
`--no-flashinfer` (or `FA_NO_CUDNN` / `FA_NO_FLASHINFER`). FlashInfer uses direct
`trtllm_batch_*` APIs only (`trtllm-gen` on SM100+, `xqa` on SM90).

Perf is wall-clock A/B (L2-flushed CUDA events). The FA4 reference is JIT-compiled and
python-dispatched, so its launch overhead is inside the measured window -- the same convention as the
hopper side, which times the torch-op wall clock.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import multiprocessing as mp
import os
import queue
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch

from . import scenarios as S

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EPS = 1e-12


def _cuda_is_healthy(device: int) -> bool:
    """True iff a tiny device op + sync succeeds (False after illegal-address poison)."""
    try:
        with torch.cuda.device(device):
            torch.zeros(1, device=f"cuda:{device}").add_(1)
            torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def _cuda_device_reset(device: int) -> None:
    """Hard-reset the CUDA primary context on `device` after an illegal-address poison.

    Destroys every allocation on that device; callers must drop cached device tensors/handles
    and rebuild inputs before continuing work on this GPU.
    """
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    # Prefer the process's libcudart (CUDA 13 on this machine); fall back to unversioned.
    lib = None
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None or not hasattr(lib, "cudaDeviceReset"):
        raise RuntimeError("libcudart.cudaDeviceReset unavailable; cannot recover poisoned GPU")
    lib.cudaSetDevice(ctypes.c_int(int(device)))
    err = int(lib.cudaDeviceReset())
    if err != 0:
        raise RuntimeError(f"cudaDeviceReset({device}) returned {err}")
    torch.cuda.init()
    torch.cuda.set_device(device)
    _fi_clear_device(device)
    _cudnn_clear_device(device)


def _cuda_recover_if_poisoned(device: int) -> bool:
    """If `device` is poisoned, reset it. Returns True when a reset was performed."""
    if _cuda_is_healthy(device):
        return False
    _cuda_device_reset(device)
    return True


# --------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------
# tcgen05 needs the arch-conditional 'a' virtual arch, hence compute_100a (not compute_100).
_GENCODE_BLACKWELL = "arch=compute_100a,code=sm_100a"
_NVCC_FLAGS_BLACKWELL = (
    "-O3", "-std=c++17", "--use_fast_math",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-gencode", _GENCODE_BLACKWELL,
    "-lineinfo",
)


@dataclass(frozen=True)
class Target:
    """Everything that differs between the two kernel/baseline pairs."""
    name: str                       # "hopper" | "blackwell"
    gencode: str                    # printed in the [target] line
    nvcc_flags: tuple               # base nvcc flags, including -gencode
    cand_default: str               # default candidate .cu
    namespace: str                  # default torch op namespace
    trials_dir: str                 # build dir under the repo root
    baseline_label: str             # "FA3" | "FA4", used in row notes and messages
    baseline_desc: str              # human description for the [target] line
    gpu_env: str                    # env var holding the default GPU index
    semantic_defs: tuple = ()       # extra -D defines (FA3 build-surface parity)
    cutlass_includes: bool = False  # add the reference/cutlass -I paths
    extra_cflags: tuple = ()


TARGETS = {
    "hopper": Target(
        name="hopper",
        gencode=S.GPU_GENCODE,
        # Toolkit-only flags from scenarios.py -- mirrors how the kernel was built when the
        # per-scenario numbers were produced. scenarios.py stays authoritative for hopper.
        nvcc_flags=tuple(S.NVCC_BASE_FLAGS),
        semantic_defs=tuple(S.SEMANTIC_DEFS),
        cand_default=os.path.join(_REPO_ROOT, "kernels", "fa3_hopper.cu"),
        namespace="_fa3bench_C",
        trials_dir=".fa3_trials",
        baseline_label="FA3",
        baseline_desc="the FA3 `_vllm_fa3_C` op (optional: pip install vllm)",
        gpu_env="FA3_EVAL_GPU",
    ),
    "blackwell": Target(
        name="blackwell",
        gencode=_GENCODE_BLACKWELL,
        nvcc_flags=_NVCC_FLAGS_BLACKWELL,
        # CUTLASS include paths are harmless for the hand-written kernel; they are needed by a
        # kernels/fa4_cutlass.cu that instantiates the 77_blackwell_fmha collective.
        cutlass_includes=True,
        extra_cflags=("-ftemplate-backtrace-limit=0",),
        cand_default=os.path.join(_REPO_ROOT, "kernels", "fa4_blackwell.cu"),
        namespace="_fa4bench_C",
        trials_dir=".fa4_trials",
        baseline_label="FA4",
        baseline_desc="upstream Dao-AILab FA4 CuteDSL (`flash-attn-4[cu13]`, `_flash_attn_fwd`)",
        gpu_env="FA4_EVAL_GPU",
    ),
}


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------
def _driver_ldflags() -> list[str]:
    """Link libcuda (driver stub) so host code can call cuTensorMapEncodeTiled (TMA)."""
    flags: list[str] = []
    try:
        from torch.utils.cpp_extension import CUDA_HOME  # type: ignore
    except Exception:
        CUDA_HOME = None
    cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    for sub in ("lib64/stubs", "lib/stubs", "targets/x86_64-linux/lib/stubs"):
        if cuda_home:
            d = os.path.join(cuda_home, *sub.split("/"))
            if os.path.isdir(d):
                flags.append(f"-L{d}")
    flags.append("-lcuda")
    return flags


def _cutlass_include_flags() -> list[str]:
    """-I flags for cutlass+cute, cutlass/util, and the 77_blackwell_fmha example headers."""
    _cut = os.path.join(_REPO_ROOT, "reference", "cutlass")
    dirs = (os.path.join(_cut, "include"),
            os.path.join(_cut, "tools", "util", "include"),
            os.path.join(_cut, "examples", "77_blackwell_fmha"))
    return [f"-I{d}" for d in dirs if os.path.isdir(d)]


def build_candidate(cu_path: str, namespace: str, work_dir: str, target: Target) -> str:
    """Compile `cu_path` into a torch op library registering `namespace`. Returns the .so path.

    Raises RuntimeError on build failure or if the op fails to register.
    """
    from torch.utils.cpp_extension import load as cpp_load

    if not os.path.isfile(cu_path):
        raise RuntimeError(f"source not found: {cu_path}")
    os.makedirs(work_dir, exist_ok=True)

    extra_cuda_cflags = (
        list(target.nvcc_flags)
        + [f"-DFA3_LIB_NAME={namespace}"]
        + [f"-D{d}" for d in target.semantic_defs]
    )
    if target.cutlass_includes:
        extra_cuda_cflags += _cutlass_include_flags()
    extra_cuda_cflags += list(target.extra_cflags)

    cpp_load(
        name=namespace,
        sources=[cu_path],
        is_python_module=False,
        build_directory=work_dir,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=_driver_ldflags(),
        verbose=True,
    )
    ns = getattr(torch.ops, namespace, None)
    if ns is None or getattr(ns, "fwd", None) is None:
        raise RuntimeError(f"built {namespace}.so but torch.ops.{namespace}.fwd not registered")
    return os.path.join(work_dir, f"{namespace}.so")


# --------------------------------------------------------------------------------------
# baselines: FA3 (a torch op) and FA4 (the CuteDSL oracle)
# --------------------------------------------------------------------------------------
class BaselineUnsupported(RuntimeError):
    """The target's baseline cannot run this config on this machine (fp8, leftpad, ...)."""


OracleUnsupported = BaselineUnsupported   # pre-merge name, kept as an alias


def load_baseline() -> bool:
    """Import FA3 (`_vllm_fa3_C`) if available. Optional — not a required dependency."""
    if hasattr(torch.ops, S.BASELINE_NS) and hasattr(getattr(torch.ops, S.BASELINE_NS), "fwd"):
        return True
    try:
        import vllm.vllm_flash_attn._vllm_fa3_C  # noqa: F401
        return True
    except Exception:
        return False


# --- the FA4 (CuteDSL) oracle adapter -------------------------------------------------
# FA4 has no torch op: the reference is the JIT-compiled CuteDSL forward `_flash_attn_fwd` from
# upstream Dao-AILab flash-attention (`flash_attn.cute.interface`, pip package `flash-attn-4[cu13]`).
# The adapter maps a harness `Inputs` (the SAME tensors that drive our candidate) onto that
# Python function and returns `(out, softmax_lse)` in the exact layout `S.call_fwd` yields -- so
# the two can be diffed apples-to-apples.
#
# Known oracle limits (cutlass-dsl ≥4.6, B200 SM100):
#   * Homogeneous FP8 e4m3: pass q/k/v_descale as (batch, hkv) fp32 when the path supports them.
#   * Mixed bf16Q+fp8KV is dequantized to bf16 on the host (FA4 requires homogeneous QKV dtypes).
#   * `cache_leftpad` has no argument in `_flash_attn_fwd` -- no oracle for the leftpad config.
_FWD = None
_IMPORT_ERR: Optional[str] = None


def _get_fwd():
    """Import (once) the cute `_flash_attn_fwd`. Returns the callable or None if unavailable."""
    global _FWD, _IMPORT_ERR
    if _FWD is not None or _IMPORT_ERR is not None:
        return _FWD
    try:
        from flash_attn.cute.interface import _flash_attn_fwd  # type: ignore
        _FWD = _flash_attn_fwd
    except Exception as e:  # pragma: no cover - environment dependent
        _IMPORT_ERR = f"{type(e).__name__}: {e}"
    return _FWD


def oracle_available() -> bool:
    return _get_fwd() is not None


def import_error() -> Optional[str]:
    _get_fwd()
    return _IMPORT_ERR


def _none_if(val, sentinel):
    return None if val == sentinel else val


def call_oracle(inp: S.Inputs, *, normalize_lse: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FA4 cute forward on `inp` and return (out (total_q,hq,hdv), lse (hq,total_q) fp32).

    Raises BaselineUnsupported when the reference cannot handle the config on this machine.

    `normalize_lse=False` skips host-side LSE layout fixups (transpose/contiguous/reshape). Use that
    in the timed path: FA4 still computes LSE in-kernel (`return_lse=True`), matching our op, but
    does not pay an extra copy kernel that ours never runs.
    """
    if os.environ.get("FA4_NO_ORACLE"):
        # correctness-only fast path: skip the (slow-to-JIT) FA4 reference; check_correctness falls
        # back to the fp32 reference. Useful for the dense edge/masking grid where the
        # per-unique-shape CuteDSL JIT dominates runtime and only our-kernel-vs-ref matters.
        raise BaselineUnsupported("FA4_NO_ORACLE set: using fp32 reference only")
    fwd = _get_fwd()
    if fwd is None:
        raise BaselineUnsupported(f"cute _flash_attn_fwd not importable: {import_error()}")

    cfg = inp.cfg
    if cfg.is_scalar_fp8_scales:
        inp = S.expand_fp8_descales(inp)
    if cfg.is_bf16q_fp8kv:
        # FA4 requires homogeneous QKV dtypes; materialize bf16 KV (same interim as call_fwd).
        inp = S._dequant_fp8_kv_to_bf16(inp)
    if cfg.leftpad and int(cfg.leftpad) > 0:
        raise BaselineUnsupported("cache_leftpad: no argument in cute _flash_attn_fwd")

    softcap = float(cfg.softcap) if cfg.softcap and cfg.softcap > 0 else None
    wl = _none_if(int(cfg.window_left), -1)
    wr = _none_if(int(cfg.window_right), -1)
    # causal is canonicalized by the reference to window_right=0; pass causal flag + raw windows.
    sink = inp.s_aux if (cfg.sink and inp.s_aux is not None) else None
    if sink is not None and sink.dtype != torch.bfloat16:
        sink = sink.to(torch.bfloat16)

    # Fairness vs fa4_blackwell.cu: non-paged `launch_ws` ignores num_splits (single-pass KV).
    # Passing forced splits>1 would tax only FA4 (SplitKV+combine) on e.g. attn_dense_hd128_splits3.
    ns = int(cfg.num_splits)
    if not cfg.paged and ns > 1:
        ns = 1

    kwargs = dict(
        softmax_scale=float(inp.softmax_scale),
        causal=bool(cfg.causal),
        softcap=softcap,
        window_size_left=wl,
        window_size_right=wr,
        learnable_sink=sink,
        num_splits=ns,   # 0 -> reference auto-heuristic; >=1 forced (clamped for non-paged)
        return_lse=True,
    )
    if inp.q_descale is not None or inp.k_descale is not None or inp.v_descale is not None:
        kwargs.update(
            q_descale=inp.q_descale,
            k_descale=inp.k_descale,
            v_descale=inp.v_descale,
        )
    if inp.page_table is not None:
        # paged KV: k/v are (num_blocks, block, hkv, hd/hdv); q is varlen-packed.
        kwargs.update(
            page_table=inp.page_table,
            seqused_k=inp.seqused_k,
            cu_seqlens_q=inp.cu_seqlens_q,
            max_seqlen_q=int(inp.max_seqlen_q),
            max_seqlen_k=int(inp.max_seqlen_k),
        )
    else:
        # contiguous varlen KV: both q and k/v packed with cu_seqlens.
        kwargs.update(
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=int(inp.max_seqlen_q),
            max_seqlen_k=int(inp.max_seqlen_k),
        )

    try:
        res = fwd(inp.q, inp.k, inp.v, **kwargs)
    except Exception as e:
        # Surface anything the reference itself can't compile/run as "unsupported" so the harness
        # degrades to fp32-ref instead of marking the config failed.
        raise BaselineUnsupported(f"oracle raised: {type(e).__name__}: {str(e)[:160]}") from e

    if isinstance(res, (tuple, list)):
        out = res[0]
        lse = res[1] if len(res) > 1 else None
    else:
        out, lse = res, None

    if not normalize_lse:
        # Timed path: keep whatever LSE FA4 returned; do not launch layout fixup kernels.
        if lse is None:
            lse = torch.empty(0, device=inp.q.device, dtype=torch.float32)
        return out, lse

    # Normalize LSE to (hq, total_q) fp32 to match S.call_fwd's contract.
    # FA4 FP8 returns LSE as (batch, hq, seq) for dense or (hq, total_q) for varlen — handle both.
    if lse is not None:
        lse = lse.float()
        total_q, hq = inp.q.shape[0], inp.q.shape[1]
        if lse.shape == (hq, total_q):
            pass
        elif lse.dim() == 2 and lse.shape == (total_q, hq):
            lse = lse.transpose(0, 1).contiguous()
        elif lse.dim() == 3 and lse.shape[1] == hq:
            # (batch, hq, seq) dense layout from FA4 — flatten to (hq, total_q)
            lse = lse.transpose(0, 1).contiguous().reshape(hq, -1)
        # else: leave as-is; check_correctness compares finite entries with tolerance.
    else:
        lse = torch.empty(0, device=inp.q.device, dtype=torch.float32)
    return out, lse


# --- the cuDNN fused-SDPA third implementation ----------------------------------------
# cuDNN 9.x exposes a fused scaled-dot-product-attention through the graph API
# (`import cudnn`, the nvidia-cudnn-frontend bindings). It is measured on every run alongside our
# kernel and the target's native reference, but it is NOT a correctness gate -- see
# `check_correctness`, which only sanity-checks it against the fp32 reference.
#
# Layout fairness: cuDNN reads the SAME packed buffers our kernel and FA3/FA4 read -- no copies, no
# repacking. Our varlen-packed q (total_q, hq, hd) is handed over as a strided view [b, hq, s, d]
# (plus `set_ragged_offset` when lengths differ), and a paged cache (num_blocks, block, hkv, hd) is
# handed over as a cuDNN paged *container* [num_blocks, hkv, block, d] with the page table. A
# reported cuDNN time is therefore directly comparable, not a layout-advantaged number.
#
# Masking maps onto cuDNN's diagonal band, which was verified to reproduce
# `use_causal_mask_bottom_right` exactly. Our reference masks key j for query i (qpos = kl-ql+i) when
#   (causal and j > qpos) or (wl >= 0 and j < qpos - wl) or (wr >= 0 and j > qpos + wr)
# so with BOTTOM_RIGHT alignment (our append semantics): right bound = 0 when causal or wr == 0 else
# wr, and left bound = wl + 1. Both bounds were pinned down empirically -- window_left+0 and +2 give
# ~0.4 error where +1 gives 9e-3.
#
# `generate_stats=True` is deliberate: our kernel and FA3/FA4 always write softmax_lse, so letting
# cuDNN skip the statistics would hand it work our kernels are doing.
#
# Known gaps on cuDNN 9.19 (each raises CudnnUnsupported -> the row shows `[no-cudnn]`):
#   * max_s_q > max_s_kv -- cuDNN rejects it on both the causal and the diagonal-band paths
#   * fp8e4m3 -- refused up front (`sdpa_fp8` heuristics segfault on this Blackwell stack)
#   * softcap -- `score_mod` is mutually exclusive with the mask subgraphs, and ours are causal
#   * cache_leftpad -- no equivalent argument
class CudnnUnsupported(BaselineUnsupported):
    """cuDNN's fused SDPA cannot express / safely run this config."""


_CUDNN = None
_CUDNN_ERR: Optional[str] = None
_CUDNN_HANDLES: dict = {}
_CUDNN_PLANS: dict = {}


def _cudnn_clear_device(device: int) -> None:
    """Drop cuDNN handles/plans bound to a device whose CUDA context was reset."""
    _CUDNN_HANDLES.pop(int(device), None)
    dead = [k for k in _CUDNN_PLANS if k[1] == int(device)]
    for k in dead:
        _CUDNN_PLANS.pop(k, None)


def _cudnn_mod():
    """Import (once) the cudnn frontend bindings. Returns the module or None."""
    global _CUDNN, _CUDNN_ERR
    if _CUDNN is not None or _CUDNN_ERR is not None:
        return _CUDNN
    try:
        import cudnn  # type: ignore
        _CUDNN = cudnn
    except Exception as e:  # pragma: no cover - environment dependent
        _CUDNN_ERR = f"{type(e).__name__}: {e}"
    return _CUDNN


def cudnn_available() -> bool:
    return _cudnn_mod() is not None


def cudnn_import_error() -> Optional[str]:
    _cudnn_mod()
    return _CUDNN_ERR


def _cudnn_handle(device: Optional[int] = None):
    """Lazily create a cuDNN handle per device (never at import: the parent stays CUDA-clean).

    Keyed by device because a handle is bound to the device current when it was created, and the plan
    cache is device-keyed too -- a single global would silently execute graphs against the wrong
    device if one process ever evaluated two.
    """
    dev = torch.cuda.current_device() if device is None else device
    h = _CUDNN_HANDLES.get(dev)
    if h is None:
        h = _CUDNN_HANDLES[dev] = _cudnn_mod().create_handle()
    return h


def _cudnn_build(inp: S.Inputs):
    """Build the graph + cache the derived tensors and output buffers for one config.

    Everything that is not q/k/v is prepared here, so the timed path is just a dict build plus
    `graph.execute`.
    """
    cudnn = _cudnn_mod()
    cfg = inp.cfg
    if cfg.dtype == "fp8e4m3":
        # Homogeneous FP8: sdpa_fp8 heuristics segfault in EngineConfig on this stack.
        raise CudnnUnsupported(
            "fp8e4m3: cuDNN sdpa_fp8 heuristics segfault on this stack; use FA4 oracle")
    if cfg.is_scalar_fp8_scales:
        inp = S.expand_fp8_descales(inp)
    if cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
        # cuDNN has no asymmetric Q/KV SDPA; judge the dequantized bf16 problem (same interim as
        # call_fwd / FA oracle).
        inp = S._dequant_fp8_kv_to_bf16(inp)
        cfg = inp.cfg
    dtypes = {"bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF,
              "bf16q_fp8kv": cudnn.data_type.BFLOAT16}
    # After dequant, tensors are bf16 even when cfg.dtype is still bf16q_fp8kv.
    io_key = "bf16" if cfg.is_bf16q_fp8kv else cfg.dtype
    if io_key not in dtypes:
        raise CudnnUnsupported(f"{cfg.dtype}: no cuDNN mapping")
    if cfg.leftpad and int(cfg.leftpad) > 0:
        raise CudnnUnsupported("cache_leftpad: no equivalent cuDNN argument")
    if cfg.softcap and cfg.softcap > 0:
        raise CudnnUnsupported("softcap: score_mod is exclusive with the mask subgraphs")

    reqs = cfg.requests
    b = len(reqs)
    hq, hkv, d, dv = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv
    q_lens = [q for q, _ in reqs]
    k_lens = [k for _, k in reqs]
    sq, sk = max(q_lens), max(k_lens)
    if sq > sk:
        raise CudnnUnsupported(f"max_s_q ({sq}) > max_s_kv ({sk}): rejected by cuDNN")
    ragged = len(set(q_lens)) != 1 or len(set(k_lens)) != 1
    dev, total_q = inp.q.device, inp.q.shape[0]

    g = cudnn.pygraph(handle=_cudnn_handle(), io_data_type=dtypes[io_key],
                      intermediate_data_type=cudnn.data_type.FLOAT,
                      compute_data_type=cudnn.data_type.FLOAT)
    I32 = cudnn.data_type.INT32
    # (graph tensor, fn(Inputs) -> device tensor). Keeping the builder lets _cudnn_plan rebind every
    # non-q/k/v input when a different Inputs shows up for this config, instead of silently reusing
    # the page tables / seq lens / descales captured at build time.
    builders: list = []

    def bind(h, build):
        builders.append((h, build))
        return h

    def i32(name, build):
        t = build(inp)
        h = g.tensor(name=name, dim=[t.shape[0], 1, 1, 1], stride=[1, 1, 1, 1], data_type=I32)
        return bind(h, build)

    # q: our packed (total_q, hq, d) buffer, viewed as [b, hq, sq, d]
    tq = g.tensor(name="q", dim=[b, hq, sq, d], stride=[sq * hq * d, d, hq * d, 1])
    if ragged:
        tq.set_ragged_offset(i32("off_q", lambda i: (i.cu_seqlens_q.int() * (hq * d)).contiguous()))

    if cfg.paged:
        nb, blk = inp.k.shape[0], cfg.block_size
        tk = g.tensor(name="k", dim=[nb, hkv, blk, d], stride=[blk * hkv * d, d, hkv * d, 1])
        tv = g.tensor(name="v", dim=[nb, hkv, blk, dv], stride=[blk * hkv * dv, dv, hkv * dv, 1])
        mb = inp.page_table.shape[1]
        page_tbl = lambda i: i.page_table.int().contiguous()
        tpk = bind(g.tensor(name="pt_k", dim=[b, 1, mb, 1], stride=[mb, mb, 1, 1], data_type=I32),
                   page_tbl)
        tpv = bind(g.tensor(name="pt_v", dim=[b, 1, mb, 1], stride=[mb, mb, 1, 1], data_type=I32),
                   page_tbl)
        paged = dict(paged_attention_k_table=tpk, paged_attention_v_table=tpv,
                     paged_attention_max_seq_len_kv=sk)
    else:
        tk = g.tensor(name="k", dim=[b, hkv, sk, d], stride=[sk * hkv * d, d, hkv * d, 1])
        tv = g.tensor(name="v", dim=[b, hkv, sk, dv], stride=[sk * hkv * dv, dv, hkv * dv, 1])
        paged = {}
        if ragged:
            tk.set_ragged_offset(i32("off_k", lambda i: (i.cu_seqlens_k.int() * (hkv * d)).contiguous()))
            tv.set_ragged_offset(i32("off_v", lambda i: (i.cu_seqlens_k.int() * (hkv * dv)).contiguous()))

    kw = dict(attn_scale=inp.softmax_scale, generate_stats=True, **paged)
    if ragged or cfg.paged:
        kw.update(use_padding_mask=True,
                  seq_len_q=i32("len_q", lambda i: torch.tensor(q_lens, device=dev, dtype=torch.int32)),
                  seq_len_kv=i32("len_k", lambda i: torch.tensor(k_lens, device=dev, dtype=torch.int32)))

    # Masking as a diagonal band anchored bottom-right (our append semantics).
    wl, wr = int(cfg.window_left), int(cfg.window_right)
    if cfg.causal or wr == 0:
        kw["diagonal_band_right_bound"] = 0
    elif wr > 0:
        kw["diagonal_band_right_bound"] = wr
    if wl >= 0:
        kw["diagonal_band_left_bound"] = wl + 1
    kw["diagonal_alignment"] = cudnn.diagonal_alignment.BOTTOM_RIGHT

    if cfg.sink and inp.s_aux is not None:
        # sink_token must be fp32 and shaped [1, hq, 1, 1]; other layouts fault at execute.
        mk_sink = lambda i: i.s_aux.reshape(1, hq, 1, 1).float().contiguous()
        sink = mk_sink(inp)
        kw["sink_token"] = bind(
            g.tensor(name="sink", dim=list(sink.shape), stride=list(sink.stride()),
                     data_type=cudnn.data_type.FLOAT), mk_sink)

    to, tstats = g.sdpa(q=tq, k=tk, v=tv, **kw)
    out_dtype = inp.q.dtype

    to.set_output(True).set_dim([b, hq, sq, dv]).set_stride([sq * hq * dv, dv, hq * dv, 1])
    tstats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    if ragged:
        to.set_ragged_offset(i32("off_o", lambda i: (i.cu_seqlens_q.int() * (hq * dv)).contiguous()))

    g.build([cudnn.heur_mode.A])

    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    out = torch.zeros(total_q, hq, dv, device=dev, dtype=out_dtype)
    stats = torch.empty(b, hq, sq, 1, device=dev, dtype=torch.float32)
    return dict(g=g, tq=tq, tk=tk, tv=tv, to=to, tstats=tstats, ws=ws, out=out, stats=stats,
                derived={h: build(inp) for h, build in builders}, builders=builders, src=inp,
                ragged=ragged, b=b, sq=sq, hq=hq)


def _cudnn_plan(inp: S.Inputs):
    """Per-process cache of built graphs, keyed by config name + device.

    The cached `derived` tensors (page tables, seq lens, descales) are rebuilt whenever a *different*
    `Inputs` arrives for the same config: they were derived from whichever Inputs built the graph, and
    `check_correctness` and `measure` each construct their own. Rebinding happens once per Inputs, not
    per call, so it stays out of the timed loop.
    """
    key = (inp.cfg.name, inp.q.device.index)
    plan = _CUDNN_PLANS.get(key)
    if plan is None:
        if not cudnn_available():
            raise CudnnUnsupported(f"cudnn frontend not importable: {cudnn_import_error()}")
        try:
            plan = _cudnn_build(inp)
        except CudnnUnsupported:
            raise
        except Exception as e:
            raise CudnnUnsupported(f"graph build failed: {type(e).__name__}: {str(e)[:140]}") from e
        _CUDNN_PLANS[key] = plan
    elif plan["src"] is not inp:
        plan["derived"] = {h: build(inp) for h, build in plan["builders"]}
        plan["src"] = inp
    return plan


def _cudnn_evict(cfg_name: str, device: Optional[int]) -> None:
    """Drop a config's cached plan. Each config is evaluated once per run, and a plan pins its
    output buffer, stats buffer and cuDNN workspace -- for a 16k-token config that output alone is
    >100 MB, so keeping every config's plan alive would cost a worker hundreds of MB for nothing."""
    _CUDNN_PLANS.pop((cfg_name, device), None)


def call_cudnn(inp: S.Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cuDNN's fused SDPA on `inp`; returns (out (total_q,hq,hdv), empty lse).

    Raises CudnnUnsupported when cuDNN cannot express the config.

    The timed path is deliberately just a dict build plus `graph.execute`. cuDNN's softmax statistics
    are still requested (our kernels always write softmax_lse, so skipping them would hand cuDNN free
    work) but are NOT repacked into our (hq, total_q) layout here: that permute+reshape cannot be a
    view, so it allocated and launched a copy kernel on every one of the ~800 warmup+timed iterations,
    a cost neither the baseline nor the candidate paid -- it inflated every reported cuDNN time. No
    caller ever used the result, since cuDNN is not a correctness gate.
    """
    if os.environ.get("FA_NO_CUDNN"):
        raise CudnnUnsupported("FA_NO_CUDNN set")
    if inp.cfg.is_scalar_fp8_scales:
        inp = S.expand_fp8_descales(inp)
    if inp.cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
        inp = S._dequant_fp8_kv_to_bf16(inp)
    p = _cudnn_plan(inp)
    pack = {p["tq"]: inp.q, p["tk"]: inp.k, p["tv"]: inp.v,
            p["to"]: p["out"], p["tstats"]: p["stats"], **p["derived"]}
    try:
        p["g"].execute(pack, p["ws"], handle=_cudnn_handle())
    except Exception as e:
        raise CudnnUnsupported(f"execute failed: {type(e).__name__}: {str(e)[:140]}") from e
    return p["out"], torch.empty(0, device=inp.q.device, dtype=torch.float32)


# --- FlashInfer (fourth measured implementation) --------------------------------------
# Like cuDNN: timed + sanity-checked vs fp32, never part of the OK/FAIL verdict.
#
# Direct TRT-LLM APIs only (vLLM competitive path) — no Batch*Wrapper / fa2|fa3|auto:
#   * SM100+: ``trtllm-gen`` via ``trtllm_batch_{decode,context}_with_kv_cache`` (HND KV)
#   * SM90:   ``xqa`` via the same direct APIs
#   * Override with FA_FLASHINFER_BACKEND=trtllm-gen|xqa
#   * Non-paged inputs are packed to paged HND *outside* the timed window (no ragged path).
#   * Softcap / hd≠hdv / per-head FP8 / leftpad / window_right>0 → [no-fi] (no FA2 fallback).
#
# Head-dim allowlist: hd=96 illegally-addresses on this stack.
class FlashInferUnsupported(BaselineUnsupported):
    """FlashInfer cannot express / safely run this config."""


_FI = None
_FI_ERR: Optional[str] = None
_FI_WS: dict = {}            # device_index -> workspace uint8
_FI_PLAN: dict = {}          # (cfg_name, device, kind, backend) -> id(inp)
_FI_KV: dict = {}            # same key -> prepared tensors for the timed run()
_FI_OK_HD = frozenset({64, 128, 192, 256})
_FI_PAGE = 64                # page size used when packing ragged → paged for trtllm
_FI_BACKENDS = frozenset({"trtllm-gen", "xqa"})


def _fi_clear_device(device: int) -> None:
    """Drop FlashInfer workspaces/plans bound to a reset CUDA context."""
    d = int(device)
    _FI_WS.pop(d, None)
    for k in [k for k in _FI_PLAN if k[1] == d]:
        _FI_PLAN.pop(k, None)
        _FI_KV.pop(k, None)


def _fi_mod():
    global _FI, _FI_ERR
    if _FI is not None or _FI_ERR is not None:
        return _FI
    try:
        import flashinfer  # type: ignore
        _FI = flashinfer
    except Exception as e:  # pragma: no cover
        _FI_ERR = f"{type(e).__name__}: {e}"
    return _FI


def flashinfer_available() -> bool:
    return _fi_mod() is not None


def flashinfer_import_error() -> Optional[str]:
    _fi_mod()
    return _FI_ERR


def _fi_device_key(device) -> int:
    d = torch.device(device)
    return d.index if d.index is not None else 0


def _fi_backend(device) -> str:
    """Direct-API backend for this GPU: trtllm-gen (SM100+) or xqa (SM90)."""
    env = (os.environ.get("FA_FLASHINFER_BACKEND") or "").strip()
    if env:
        if env not in _FI_BACKENDS:
            raise FlashInferUnsupported(
                f"FA_FLASHINFER_BACKEND={env!r}: only {sorted(_FI_BACKENDS)} "
                f"(Batch*Wrapper / fa2|fa3|auto removed)")
        return env
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 10:
        return "trtllm-gen"
    if major == 9:
        return "xqa"
    raise FlashInferUnsupported(
        f"SM{major}x: FlashInfer direct path needs SM90 (xqa) or SM100+ (trtllm-gen)")


def _fi_workspace(device) -> torch.Tensor:
    """128 MiB workspace per device (FlashInfer / vLLM recommended size)."""
    key = _fi_device_key(device)
    ws = _FI_WS.get(key)
    if ws is None:
        ws = _FI_WS[key] = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return ws


def _fi_pack_ragged_to_paged(inp: S.Inputs, page_size: int = _FI_PAGE):
    """Pack contiguous/varlen K/V into a paged NHD cache (host setup; not on the timed path).

    Needed because the direct trtllm APIs have no ragged varlen path.
    """
    if inp.cu_seqlens_k is None:
        raise FlashInferUnsupported("non-paged requires cu_seqlens_k")
    cfg = inp.cfg
    hkv, hd, hdv = cfg.num_heads_kv, cfg.head_dim, cfg.hdv
    device = inp.k.device
    cu = inp.cu_seqlens_k.tolist()
    b = len(cu) - 1
    lens = [cu[i + 1] - cu[i] for i in range(b)]
    nblocks = [(L + page_size - 1) // page_size for L in lens]
    total = sum(nblocks)
    if total <= 0:
        raise FlashInferUnsupported("ragged pack produced zero pages")
    k_p = torch.zeros(total, page_size, hkv, hd, device=device, dtype=inp.k.dtype)
    v_p = torch.zeros(total, page_size, hkv, hdv, device=device, dtype=inp.v.dtype)
    indices: list[int] = []
    last: list[int] = []
    indptr = [0]
    page_id = 0
    for i, L in enumerate(lens):
        off = cu[i]
        for j in range(nblocks[i]):
            t0 = j * page_size
            t1 = min(L, t0 + page_size)
            n = t1 - t0
            k_p[page_id, :n].copy_(inp.k[off + t0:off + t1])
            v_p[page_id, :n].copy_(inp.v[off + t0:off + t1])
            indices.append(page_id)
            page_id += 1
        rem = L % page_size
        last.append(page_size if (rem == 0 and L > 0) else rem)
        indptr.append(indptr[-1] + nblocks[i])
    return (
        k_p, v_p,
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last, dtype=torch.int32, device=device),
        page_size,
    )


def _fi_nhd_to_hnd(k: torch.Tensor, v: torch.Tensor):
    """NHD [pages, page, hkv, d] → HND [pages, hkv, page, d] (contiguous; setup only)."""
    return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()


def _fi_block_table_from_csr(indptr, indices, batch: int):
    """Dense [B, max_pages] page table from CSR indptr/indices (vLLM block_tables shape)."""
    nblocks = [int(indptr[i + 1] - indptr[i]) for i in range(batch)]
    max_p = max(nblocks) if nblocks else 0
    if max_p <= 0:
        raise FlashInferUnsupported("zero pages in block table")
    bt = torch.zeros(batch, max_p, dtype=torch.int32, device=indices.device)
    for i in range(batch):
        s, e = int(indptr[i]), int(indptr[i + 1])
        bt[i, : e - s] = indices[s:e]
    return bt


def _fi_prepare_trtllm(inp: S.Inputs, page_size: int = _FI_PAGE):
    """Build HND KV + block_tables + seq metadata for direct trtllm APIs (setup only)."""
    cfg = inp.cfg
    device = inp.q.device
    if cfg.paged:
        assert inp.page_table is not None and inp.seqused_k is not None
        k_hnd, v_hnd = _fi_nhd_to_hnd(inp.k, inp.v)
        block_tables = inp.page_table.contiguous()
        seq_lens = inp.seqused_k.to(torch.int32).contiguous()
        page = int(cfg.block_size)
        b = int(seq_lens.shape[0])
        nblocks = (seq_lens + (page - 1)) // page
    else:
        k_nhd, v_nhd, indptr, indices, _last, page = _fi_pack_ragged_to_paged(
            inp, page_size=page_size)
        k_hnd, v_hnd = _fi_nhd_to_hnd(k_nhd, v_nhd)
        b = int(indptr.shape[0] - 1)
        block_tables = _fi_block_table_from_csr(indptr, indices, b)
        if inp.cu_seqlens_k is None:
            raise FlashInferUnsupported("non-paged requires cu_seqlens_k")
        cu = inp.cu_seqlens_k
        seq_lens = (cu[1:] - cu[:-1]).to(torch.int32).contiguous()
        nblocks = (seq_lens + (page - 1)) // page

    # vLLM passes cum_seq_lens_kv as cumulative *block* counts for the context API.
    cum_kv = torch.zeros(b + 1, dtype=torch.int32, device=device)
    cum_kv[1:] = torch.cumsum(nblocks, dim=0)
    cum_q = inp.cu_seqlens_q.to(torch.int32).contiguous()
    return {
        "k": k_hnd, "v": v_hnd,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "cum_q": cum_q,
        "cum_kv": cum_kv,
        "page": page,
        "batch": b,
        "max_q": int(inp.max_seqlen_q),
        "max_kv": int(inp.max_seqlen_k),
    }


def _fi_scalar_fp8_bmm_scales(inp: S.Inputs) -> tuple[float, float]:
    """vLLM-style (bmm1, bmm2) for trtllm: sm*q*k and v. Requires scalar descales."""
    cfg = inp.cfg
    sm = float(inp.softmax_scale)

    def _f(t: Optional[torch.Tensor], default: float = 1.0) -> float:
        if t is None:
            return default
        if t.numel() != 1:
            raise FlashInferUnsupported(
                f"FlashInfer FP8 needs scalar scales, got shape {tuple(t.shape)}")
        return float(t.reshape(-1)[0].item())

    q_s = _f(inp.q_descale, 1.0) if cfg.is_fp8_qkv else 1.0
    k_s = _f(inp.k_descale, 1.0)
    v_s = _f(inp.v_descale, 1.0)
    return sm * q_s * k_s, v_s


def call_flashinfer(inp: S.Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlashInfer attention on `inp` via direct trtllm_batch_* APIs. Returns (out, empty_lse).

    Raises FlashInferUnsupported when FlashInfer cannot express the config fairly.
    """
    if os.environ.get("FA_NO_FLASHINFER"):
        raise FlashInferUnsupported("FA_NO_FLASHINFER set")
    if _fi_mod() is None:
        raise FlashInferUnsupported(f"flashinfer not importable: {flashinfer_import_error()}")

    cfg = inp.cfg
    if cfg.leftpad and int(cfg.leftpad) > 0:
        raise FlashInferUnsupported("cache_leftpad: no FlashInfer argument")
    if int(cfg.window_right) > 0:
        raise FlashInferUnsupported("window_right>0: FlashInfer has window_left only")
    if cfg.has_fp8_kv and not cfg.is_scalar_fp8_scales:
        raise FlashInferUnsupported(
            f"{cfg.dtype}: FlashInfer trtllm needs scalar FP8 scales "
            f"(set fp8_scales='scalar'); per-head descales unsupported")
    hd, hdv = cfg.head_dim, cfg.hdv
    if hd not in _FI_OK_HD or hdv not in _FI_OK_HD:
        raise FlashInferUnsupported(
            f"head_dim/hdv ({hd},{hdv}): not in FlashInfer allowlist "
            f"{sorted(_FI_OK_HD)} (others illegal-address on this stack)")
    if hd != hdv:
        raise FlashInferUnsupported(
            f"head_dim_v={hdv} != head_dim={hd}: trtllm direct path needs equal dims")

    qdt = inp.q.dtype
    ok_q = (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
    if qdt not in ok_q:
        raise FlashInferUnsupported(f"{qdt}: only bf16/fp16/fp8e4m3 wired")
    if qdt == torch.float8_e4m3fn and not cfg.is_fp8_qkv:
        raise FlashInferUnsupported("fp8 Q requires dtype=fp8e4m3")
    if cfg.softcap and cfg.softcap > 0:
        raise FlashInferUnsupported(
            "logits_soft_cap: unsupported by trtllm/xqa direct path")

    backend = _fi_backend(inp.q.device)
    wl = int(cfg.window_left)
    sm = float(inp.softmax_scale)
    hq, hkv = cfg.num_heads_q, cfg.num_heads_kv
    causal = bool(cfg.causal)
    if hq % hkv != 0:
        raise FlashInferUnsupported(
            f"GQA {hq}/{hkv}: trtllm requires num_qo_heads % num_kv_heads == 0")

    sink = None
    if cfg.sink and inp.s_aux is not None:
        sink = inp.s_aux.float().reshape(-1).contiguous()
        if sink.numel() != hq:
            raise FlashInferUnsupported(f"sinks shape {tuple(sink.shape)} != [{hq}]")

    dkey = _fi_device_key(inp.q.device)
    bmm1, bmm2 = (sm, 1.0)
    if cfg.has_fp8_kv:
        bmm1, bmm2 = _fi_scalar_fp8_bmm_scales(inp)

    # Short uniform causal queries → decode API; else context/prefill.
    q_lens = [q for q, _ in cfg.requests]
    uniform_q = len(set(q_lens)) == 1
    use_decode = bool(causal and uniform_q and int(q_lens[0]) <= 8)
    # trtllm-gen context has no bf16-Q + fp8-KV cubin (vLLM dequants; we refuse).
    if cfg.is_bf16q_fp8kv and not use_decode:
        raise FlashInferUnsupported(
            "bf16q_fp8kv prefill: no trtllm-gen context cubin (use fp8e4m3 scalar or decode)")

    kind = f"trtllm-{'decode' if use_decode else 'context'}"
    plan_key = (cfg.name, dkey, kind, backend)
    need_plan = _FI_PLAN.get(plan_key) != id(inp)
    try:
        if need_plan:
            prep = _fi_prepare_trtllm(inp)
            _FI_PLAN[plan_key] = id(inp)
            _FI_KV[plan_key] = prep
        prep = _FI_KV[plan_key]
        ws = _fi_workspace(inp.q.device)
        out_kw = dict(out_dtype=torch.bfloat16) if cfg.has_fp8_kv else {}
        if use_decode:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache
            out = trtllm_batch_decode_with_kv_cache(
                query=inp.q,
                kv_cache=(prep["k"], prep["v"]),
                workspace_buffer=ws,
                block_tables=prep["block_tables"],
                seq_lens=prep["seq_lens"],
                max_seq_len=prep["max_kv"],
                bmm1_scale=bmm1,
                bmm2_scale=bmm2,
                window_left=wl,
                sinks=sink,
                kv_layout="HND",
                backend=backend,
                q_len_per_req=int(q_lens[0]),
                **out_kw,
            )
        else:
            from flashinfer.prefill import trtllm_batch_context_with_kv_cache
            out = trtllm_batch_context_with_kv_cache(
                query=inp.q,
                kv_cache=(prep["k"], prep["v"]),
                workspace_buffer=ws,
                block_tables=prep["block_tables"],
                seq_lens=prep["seq_lens"],
                max_q_len=prep["max_q"],
                max_kv_len=prep["max_kv"],
                bmm1_scale=bmm1,
                bmm2_scale=bmm2,
                batch_size=prep["batch"],
                cum_seq_lens_q=prep["cum_q"],
                cum_seq_lens_kv=prep["cum_kv"],
                window_left=wl,
                sinks=sink,
                kv_layout="HND",
                causal=causal,
                **out_kw,
            )
        torch.cuda.synchronize(inp.q.device)
    except FlashInferUnsupported:
        raise
    except Exception as e:
        _FI_PLAN.pop(plan_key, None)
        _FI_KV.pop(plan_key, None)
        raise FlashInferUnsupported(f"{type(e).__name__}: {str(e)[:160]}") from e

    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dtype == torch.float8_e4m3fn:
        out = out.to(torch.bfloat16)
    return out, torch.empty(0, device=inp.q.device, dtype=torch.float32)


def _fi_evict(cfg_name: str, device) -> None:
    """Drop plan-cache entries for this config so the next Inputs re-plans cleanly."""
    dkey = _fi_device_key(device if isinstance(device, torch.device)
                          else torch.device(f"cuda:{device}"))
    dead = [k for k in _FI_PLAN if k[0] == cfg_name and k[1] == dkey]
    for k in dead:
        _FI_PLAN.pop(k, None)
        _FI_KV.pop(k, None)


# --- per-target baseline dispatch -----------------------------------------------------
def baseline_available(target: Target) -> bool:
    """Is the target's baseline importable at all? (Per-config support is separate.)"""
    return load_baseline() if target.name == "hopper" else oracle_available()


def baseline_import_error(target: Target) -> Optional[str]:
    return None if target.name == "hopper" else import_error()


def call_baseline(inp: S.Inputs, target: Target, *, normalize_lse: bool = True
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, lse) from the target's baseline. Raises BaselineUnsupported for configs it can't run."""
    if target.name == "hopper":
        return S.call_fwd(S.BASELINE_NS, inp)
    return call_oracle(inp, normalize_lse=normalize_lse)


def _baseline_out(inp: S.Inputs, target: Target):
    """(out, lse) from the baseline, or None if it cannot run this config."""
    try:
        return call_baseline(inp, target)
    except BaselineUnsupported:
        return None


# --------------------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------------------
def _maxabs(t: torch.Tensor) -> float:
    return 0.0 if t.numel() == 0 else float(t.abs().max().item())


class _Result:
    __slots__ = ("name", "correct", "unsup", "unsup_ours", "unsup_baseline", "baseline_note",
                 "err_ratio", "ratio", "base_ms", "cand_ms", "note",
                 "baseline", "lse_skipped", "cudnn_ms", "cudnn_ran", "cudnn_note", "cudnn_suspect",
                 "fi_ms", "fi_ran", "fi_note", "fi_suspect")

    def __init__(self, name):
        self.name = name
        self.correct = False
        self.unsup = False          # ours unsupported (kept for counts / plot JSON)
        self.unsup_ours = False
        self.unsup_baseline = False  # FA3/FA4 oracle cannot run this config
        self.baseline_note = ""      # why the baseline could not run (when unsup_baseline)
        self.err_ratio = 0.0
        self.ratio: Optional[float] = None
        self.base_ms: Optional[float] = None
        self.cand_ms: Optional[float] = None
        self.note = ""
        self.baseline = False     # was the baseline available for this config?
        self.lse_skipped = False  # baseline gave no comparable LSE -> that gate did not run
        # cuDNN / FlashInfer are extra *measurements*, never part of the verdict.
        self.cudnn_ms: Optional[float] = None
        self.cudnn_ran = False    # did cuDNN execute at all for this config?
        self.cudnn_note = ""      # why cuDNN could not run, ONLY when it could not
        self.cudnn_suspect = False  # cuDNN ran but disagreed with the fp32 reference
        self.fi_ms: Optional[float] = None
        self.fi_ran = False
        self.fi_note = ""
        self.fi_suspect = False

    @property
    def cudnn_ratio(self) -> Optional[float]:
        """cuDNN's speedup over the target's native reference, same convention as `ratio`."""
        if self.base_ms and self.cudnn_ms:
            return self.base_ms / self.cudnn_ms
        return None

    @property
    def fi_ratio(self) -> Optional[float]:
        """FlashInfer's speedup over the target's native reference, same convention as `ratio`."""
        if self.base_ms and self.fi_ms:
            return self.base_ms / self.fi_ms
        return None

    def unsup_cudnn(self, *, cudnn_enabled: bool) -> bool:
        """True when cuDNN was probed for this run but could not express/run the config."""
        return bool(cudnn_enabled and not self.cudnn_ran)

    def unsup_fi(self, *, fi_enabled: bool) -> bool:
        """True when FlashInfer was probed for this run but could not express/run the config."""
        return bool(fi_enabled and not self.fi_ran)

    def unsup_frameworks(self, baseline_label: str, *, cudnn_enabled: bool,
                         fi_enabled: bool = False) -> list[str]:
        """Frameworks that cannot run this scenario (ours / FA3|FA4 / cuDNN / FlashInfer)."""
        fw: list[str] = []
        if self.unsup_ours:
            fw.append("ours")
        if self.unsup_baseline:
            fw.append(baseline_label)
        if self.unsup_cudnn(cudnn_enabled=cudnn_enabled):
            fw.append("cuDNN")
        if self.unsup_fi(fi_enabled=fi_enabled):
            fw.append("FlashInfer")
        return fw

    def status_label(self, baseline_label: str, *, cudnn_enabled: bool,
                     fi_enabled: bool = False) -> str:
        """OK / FAIL / UNSUP({ours, FA4, cuDNN, FlashInfer}) — lists every unsupported framework."""
        if self.unsup_ours:
            fw = self.unsup_frameworks(baseline_label, cudnn_enabled=cudnn_enabled,
                                       fi_enabled=fi_enabled)
            return f"UNSUP({{{', '.join(fw)}}})"
        if self.correct:
            return "OK"
        return "FAIL"


def check_correctness(cfg: S.Config, namespace: Optional[str], target: Target,
                      gpu_id: int = 0, seed: int = 0, with_cudnn: bool = True,
                      with_fi: bool = True) -> _Result:
    """Judge the candidate with the Dao-AILab flash-attention forward criterion.

    Always builds a pure-PyTorch fp32 reference (`ref_fp32`). When the arch baseline is
    available it plays the role of `out_pt` in FA's tests:

        fwd_atol = 2 * |(out_ref + 0.3 - 0.3) - out_ref|
        |out - out_ref| <= rtol * |out_pt - out_ref| + fwd_atol
        rtol = 2 (3 if softcap > 0)

    Plus coarse allclose / mean-bias / LSE gates vs the baseline. `namespace is None` is the
    --baseline-only path (baseline vs fp32 ref only).
    """
    res = _Result(cfg.name)
    inp = S.make_inputs(cfg, seed, device=f"cuda:{gpu_id}")
    ref = S.ref_fp32(inp).float()
    # FA numerical floor: any bf16 arithmetic on out_ref can move values by this much.
    fwd_atol = 2.0 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    err_mult = S.ERR_MULT_SOFTCAP if (cfg.softcap and cfg.softcap > 0) else S.ERR_MULT

    # cuDNN / FlashInfer: sanity only. Extra measurements, never part of the verdict -- their own
    # deviation from fp32 sits near bf16 atol, so gating would fail our kernel for their
    # accumulation order. A disagreement only tags the row [cudnn-suspect] / [fi-suspect].
    if with_cudnn:
        try:
            cudnn_out = call_cudnn(inp)[0].float()
            torch.cuda.synchronize(gpu_id)
            atol_c = S.ALLCLOSE_TOL[cfg.dtype][0]
            c_err = _maxabs(cudnn_out - ref)
            res.cudnn_ran = True
            if not (torch.isfinite(cudnn_out).all() and c_err < max(atol_c, 5e-2) * 2):
                res.cudnn_suspect = True
        except BaselineUnsupported as e:
            res.cudnn_note = str(e)[:120]
        except Exception as e:
            res.cudnn_note = f"{type(e).__name__}: {e}"[:120]

    if with_fi:
        try:
            fi_out = call_flashinfer(inp)[0].float()
            atol_f = S.ALLCLOSE_TOL[cfg.dtype][0]
            f_err = _maxabs(fi_out - ref)
            res.fi_ran = True
            if not (torch.isfinite(fi_out).all() and f_err < max(atol_f, 5e-2) * 2):
                res.fi_suspect = True
        except BaselineUnsupported as e:
            res.fi_note = str(e)[:120]
        except Exception as e:
            res.fi_note = f"{type(e).__name__}: {e}"[:120]

    # A poisoned probe (esp. FlashInfer async illegal-address) must not cascade into FA4/ours
    # for this scenario, or into every later scenario on this worker.
    if not _cuda_is_healthy(gpu_id):
        try:
            _cuda_device_reset(gpu_id)
        except Exception as e:
            res.note = f"CUDA context poisoned and reset failed: {type(e).__name__}: {e}"[:200]
            return res
        inp = S.make_inputs(cfg, seed, device=f"cuda:{gpu_id}")
        ref = S.ref_fp32(inp).float()
        fwd_atol = 2.0 * (ref + 0.3 - 0.3 - ref).abs().max().item()

    try:
        base = call_baseline(inp, target)
        res.baseline = True
    except BaselineUnsupported as e:
        base = None
        res.baseline = False
        res.unsup_baseline = True
        res.baseline_note = str(e)[:160]
    except Exception as e:
        base = None
        res.baseline = False
        res.unsup_baseline = True
        res.baseline_note = f"{type(e).__name__}: {e}"[:160]

    if namespace is None:  # --baseline-only: judge the reference itself vs the fp32 ref
        if base is None:
            res.unsup = True
            res.unsup_baseline = True
            res.note = res.baseline_note or f"{target.baseline_label} baseline unsupported for this config"
            return res
        bf = base[0].float()
        atol, rtol = S.ALLCLOSE_TOL[cfg.dtype]
        err = _maxabs(bf - ref)
        res.err_ratio = err
        res.correct = bool(torch.isfinite(bf).all() and err < max(atol, 5e-2) * 4)
        res.note = "" if res.correct else f"{target.baseline_label} max|Δref|={err:.3e}"
        return res

    try:
        cand_out, cand_lse = S.call_fwd(namespace, inp)
    except Exception as e:
        res.note = f"candidate fwd raised: {type(e).__name__}: {e}"[:200]
        if "unsupported" in res.note.lower():
            res.unsup = True
            res.unsup_ours = True
        return res

    cand_out_f = cand_out.float()
    cand_lse_f = cand_lse.float()
    if not torch.isfinite(cand_out_f).all():
        res.note = "candidate out has NaN/inf"
        return res

    if base is not None:
        base_out, base_lse = base
        base_out_f = base_out.float()
        base_lse_f = base_lse.float()

        if cand_out_f.shape != base_out_f.shape:
            res.note = (f"out shape {tuple(cand_out_f.shape)} != "
                        f"{target.baseline_label} {tuple(base_out_f.shape)}")
            return res

        # FA criterion: cand vs fp32 ref, scaled by baseline-vs-ref (+ fwd_atol floor).
        baseline_err = _maxabs(base_out_f - ref)
        out_err = _maxabs(cand_out_f - ref)
        res.err_ratio = out_err / (baseline_err + _EPS)
        ratio_ok = out_err <= err_mult * baseline_err + fwd_atol

        bias = float((cand_out_f - base_out_f).abs().mean().item()) if cand_out_f.numel() else 0.0
        atol, rtol = S.ALLCLOSE_TOL[cfg.dtype]
        allclose_ok = torch.allclose(cand_out_f, base_out_f, atol=atol, rtol=rtol)
        bias_ok = bias <= S.MEAN_ABS_TOL.get(cfg.dtype, 1e-2)

        lse_ok = True
        if base_lse_f.numel() and cand_lse_f.numel() and base_lse_f.shape == cand_lse_f.shape:
            finite = torch.isfinite(cand_lse_f) & torch.isfinite(base_lse_f)
            lse_atol, lse_rtol = S.LSE_TOL
            lse_ok = (not finite.any()) or torch.allclose(
                cand_lse_f[finite], base_lse_f[finite], atol=lse_atol, rtol=lse_rtol)
        else:
            res.lse_skipped = True

        res.correct = bool(ratio_ok and allclose_ok and lse_ok and bias_ok)
        if not res.correct:
            fails = []
            if not ratio_ok:
                fails.append(f"FA-crit err={out_err:.3e} > {err_mult}×base={baseline_err:.3e}+atol={fwd_atol:.3e}")
            if not allclose_ok:
                fails.append(f"allclose fail (atol={atol},rtol={rtol})")
            if not bias_ok:
                fails.append(f"bias={bias:.3e}>{S.MEAN_ABS_TOL.get(cfg.dtype)}")
            if not lse_ok:
                fails.append("lse_err>tol")
            res.note = "; ".join(fails)[:200]
    else:
        # No baseline: absolute fp32-ref tolerance (approximate; bf16/fp16/fp8 are noisy).
        atol, rtol = S.ALLCLOSE_TOL[cfg.dtype]
        out_err = _maxabs(cand_out_f - ref)
        res.err_ratio = out_err
        res.correct = bool(torch.allclose(cand_out_f, ref, atol=max(atol, 5e-2), rtol=max(rtol, 5e-2)))
        res.note = ("" if res.correct else f"fp32-ref max|Δ|={out_err:.3e}") + " (no-baseline)"
    return res



# --------------------------------------------------------------------------------------
# perf (interleaved CUDA-event timing; clocks unlockable -> ratio only)
# --------------------------------------------------------------------------------------
def _time(call, flush, start, end) -> list[float]:
    """Timing samples for a 0-arg callable, L2-flushed and synchronized per iteration."""
    timed, trials = int(S.PERF["timed_iters"]), int(S.PERF["num_trials"])
    samples = []
    for _ in range(trials):
        for _ in range(timed):
            flush.zero_()
            torch.cuda.synchronize()
            start.record()
            call()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    return samples


def measure(cfg: S.Config, namespace: str, gpu_id: int, target: Target,
            with_cudnn: bool = True, with_fi: bool = True,
            skip_cand: bool = False) -> _Result:
    """Time reference, ours, cuDNN and FlashInfer back-to-back on `gpu_id`, from one set of inputs.

    `skip_cand=True` when ours cannot run the config (e.g. homogeneous FP8) but FA4/FI still should
    be timed against each other.
    """
    res = _Result(cfg.name)
    res.correct = True
    device = torch.device(f"cuda:{gpu_id}")
    inp = S.make_inputs(cfg, seed=0, device=f"cuda:{gpu_id}")
    # FlashInfer keeps native FP8 tensors. FA4/ours/cuDNN interim path for bf16q_fp8kv dequants to
    # bf16 — do that *once outside the timed window* so we time attention, not host dequant.
    inp_fi = inp
    inp_attn = inp
    if cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
        if cfg.is_scalar_fp8_scales:
            inp_attn = S.expand_fp8_descales(inp)
        inp_attn = S._dequant_fp8_kv_to_bf16(inp_attn)
    flush = torch.empty(int(S.PERF["l2_flush_mb"]) * 1024 * 1024, dtype=torch.int8, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def base_call():
        # Timed FA4 path: still compute LSE in-kernel, but skip host LSE layout fixups that ours
        # never pays (transpose/contiguous). Correctness uses the default normalize_lse=True.
        return call_baseline(inp_attn, target, normalize_lse=False)

    def cand_call():
        return S.call_fwd(namespace, inp_attn)

    def cudnn_call():
        return call_cudnn(inp_attn)

    def fi_call():
        return call_flashinfer(inp_fi)

    # Is cuDNN / FlashInfer in play for this config? Probe (and plan/build) outside the timed window.
    have_cudnn = with_cudnn
    try:
        if not have_cudnn:
            raise CudnnUnsupported("disabled (--no-cudnn)")
        cudnn_call()
    except BaselineUnsupported as e:
        have_cudnn = False
        res.cudnn_note = str(e)[:120]
    except Exception as e:
        have_cudnn = False
        res.cudnn_note = f"{type(e).__name__}: {e}"[:120]

    have_fi = with_fi
    try:
        if not have_fi:
            raise FlashInferUnsupported("disabled (--no-flashinfer)")
        fi_call()
    except BaselineUnsupported as e:
        have_fi = False
        res.fi_note = str(e)[:120]
    except Exception as e:
        have_fi = False
        res.fi_note = f"{type(e).__name__}: {e}"[:120]

    if not _cuda_is_healthy(gpu_id):
        # Probe poisoned the device — heal and rebuild inputs before timing ours/FA4.
        _cuda_device_reset(gpu_id)
        inp = S.make_inputs(cfg, seed=0, device=f"cuda:{gpu_id}")
        inp_fi = inp
        inp_attn = inp
        if cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
            if cfg.is_scalar_fp8_scales:
                inp_attn = S.expand_fp8_descales(inp)
            inp_attn = S._dequant_fp8_kv_to_bf16(inp_attn)
        flush = torch.empty(int(S.PERF["l2_flush_mb"]) * 1024 * 1024, dtype=torch.int8, device=device)
        have_cudnn = False
        have_fi = False

    for _ in range(int(S.PERF["warmup_iters"])):
        base_call()
        if not skip_cand:
            cand_call()
        if have_cudnn:
            try:
                cudnn_call()
            except Exception as e:      # never let the third implementation cost us our own numbers
                have_cudnn = False
                res.cudnn_note = f"{type(e).__name__}: {e}"[:120]
        if have_fi:
            try:
                fi_call()
            except Exception as e:
                have_fi = False
                res.fi_note = f"{type(e).__name__}: {e}"[:120]
    torch.cuda.synchronize()

    base = _time(base_call, flush, start, end)
    res.base_ms = statistics.median(base) if base else 0.0
    if skip_cand:
        res.cand_ms = None
        res.ratio = None
    else:
        cand = _time(cand_call, flush, start, end)
        res.cand_ms = statistics.median(cand) if cand else 0.0
        res.ratio = ((res.base_ms / res.cand_ms)
                     if (res.base_ms > 0 and res.cand_ms > 0) else 0.0)
    if have_cudnn:
        # Guarded for the same reason: cuDNN is timed AFTER ours, and a failure here must not throw
        # away the ours-vs-baseline result already in `res`.
        try:
            cud = _time(cudnn_call, flush, start, end)
            res.cudnn_ms = statistics.median(cud) if cud else None
            res.cudnn_ran = True
        except Exception as e:
            res.cudnn_note = f"{type(e).__name__}: {e}"[:120]
    if have_fi:
        try:
            fi = _time(fi_call, flush, start, end)
            res.fi_ms = statistics.median(fi) if fi else None
            res.fi_ran = True
        except Exception as e:
            res.fi_note = f"{type(e).__name__}: {e}"[:120]
    return res


# --------------------------------------------------------------------------------------
# scheduling: one scenario at a time per GPU; a scenario's A/B pair stays on one GPU
# --------------------------------------------------------------------------------------
def _eval_one(cfg: S.Config, namespace: Optional[str], target: Target, gpu_id: int,
              no_perf: bool, with_cudnn: bool = True, with_fi: bool = True) -> _Result:
    """Correctness then (optionally) the perf A/B for ONE scenario, entirely on `gpu_id`.

    The single place a scenario is evaluated: the serial and multi-GPU paths both go through here,
    so parallelism cannot change what is measured -- only where and when it runs. `measure` times the
    baseline and the candidate back-to-back on this same device.
    """
    try:
        r = check_correctness(cfg, namespace, target, gpu_id=gpu_id, with_cudnn=with_cudnn,
                              with_fi=with_fi)
    except Exception as e:
        r = _Result(cfg.name)
        r.note = f"verify error: {type(e).__name__}: {e}"[:200]
    if (not no_perf and namespace is not None and r.baseline
            and (r.correct or r.unsup_ours)):
        try:
            m = measure(cfg, namespace, gpu_id, target, with_cudnn=with_cudnn, with_fi=with_fi,
                        skip_cand=bool(r.unsup_ours and not r.correct))
            if m.ratio is not None:
                r.ratio = m.ratio
            if m.base_ms is not None:
                r.base_ms = m.base_ms
            if m.cand_ms is not None:
                r.cand_ms = m.cand_ms
            r.cudnn_ms = m.cudnn_ms
            r.cudnn_ran = r.cudnn_ran or m.cudnn_ran
            if m.cudnn_note and not r.cudnn_note:
                r.cudnn_note = m.cudnn_note
            r.fi_ms = m.fi_ms
            r.fi_ran = r.fi_ran or m.fi_ran
            if m.fi_note and not r.fi_note:
                r.fi_note = m.fi_note
        except Exception as e:
            r.note = f"perf error: {type(e).__name__}: {e}"[:120]
    # Each config is evaluated once per run; a cached plan pins its output/stats/workspace buffers
    # (>100 MB for the 16k-token configs), so release it rather than accumulating ~80 of them.
    _cudnn_evict(cfg.name, gpu_id)
    _fi_evict(cfg.name, gpu_id)
    # Last line of defense: if ours/FA4/cuDNN poisoned the context, heal before the next scenario.
    try:
        _cuda_recover_if_poisoned(gpu_id)
    except Exception:
        pass
    return r


def _resolve_gpus(gpus_arg: Optional[str], gpu_arg: Optional[int], target: Target) -> list[int]:
    """Which devices to schedule on. --gpus > --gpu > $FA3_EVAL_GPU/$FA4_EVAL_GPU > every visible."""
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA devices visible")
    if gpus_arg:
        want = [int(x) for x in gpus_arg.replace(",", " ").split()]
    elif gpu_arg is not None:
        want = [gpu_arg]
    elif os.environ.get(target.gpu_env):
        want = [int(os.environ[target.gpu_env])]
    else:
        want = list(range(n))
    bad = [g for g in want if not 0 <= g < n]
    if bad:
        raise RuntimeError(f"GPU index out of range: {bad} (visible: 0..{n - 1})")
    seen, out = set(), []
    for g in want:            # dedupe, preserve order
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _gpu_worker(gpu: int, spec: dict, cfgs: list, task_q, res_q) -> None:
    """One process pinned to one GPU: pull scenario indices off `task_q`, evaluate, push results.

    Pulling (rather than a precomputed shard) is what makes a GPU pick up the next scenario the
    moment it goes idle. Only one scenario is in flight per worker, so only one per GPU.
    """
    try:
        torch.cuda.set_device(gpu)
        target = TARGETS[spec["target"]]
        if target.name == "blackwell":
            os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
        if spec["so_path"]:
            torch.ops.load_library(spec["so_path"])
        # Do the baseline import here, before any timing, so its cost never lands in a sample.
        baseline_available(target)
    except Exception as e:
        res_q.put(("fatal", gpu, f"{type(e).__name__}: {e}", None))
        return

    while True:
        i = task_q.get()
        if i is None:
            break
        try:
            r = _eval_one(cfgs[i], spec["namespace"], target, gpu, spec["no_perf"],
                          with_cudnn=spec.get("with_cudnn", True),
                          with_fi=spec.get("with_fi", True))
        except Exception as e:
            r = _Result(cfgs[i].name)
            r.note = f"worker error: {type(e).__name__}: {e}"[:200]
        res_q.put(("row", gpu, i, r))


def _run_parallel(cfgs: list, namespace: Optional[str], target: Target, gpus: list[int],
                  no_perf: bool, so_path: Optional[str], on_row,
                  with_cudnn: bool = True, with_fi: bool = True) -> list[_Result]:
    """Run every scenario across `gpus`, one at a time per GPU. Returns rows in config order."""
    # spawn, not fork: the parent already holds a CUDA context and CUDA is not fork-safe.
    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    for i in range(len(cfgs)):
        task_q.put(i)
    for _ in gpus:
        task_q.put(None)          # one sentinel per worker

    spec = {"target": target.name, "namespace": namespace,
            "so_path": so_path, "no_perf": no_perf, "with_cudnn": with_cudnn, "with_fi": with_fi}
    procs = [ctx.Process(target=_gpu_worker, args=(g, spec, cfgs, task_q, res_q), daemon=True)
             for g in gpus]
    for p in procs:
        p.start()

    rows: list[Optional[_Result]] = [None] * len(cfgs)
    got = 0
    try:
        while got < len(cfgs):
            try:
                kind, gpu, a, b = res_q.get(timeout=30)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    break             # every worker gone; the missing-rows check below reports it
                continue
            if kind == "fatal":
                raise RuntimeError(f"worker on GPU {gpu} failed to start: {a}")
            rows[a] = b
            got += 1
            on_row(b, gpu)
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    missing = [cfgs[i].name for i, r in enumerate(rows) if r is None]
    if missing:
        raise RuntimeError(f"{len(missing)} scenarios never returned (worker died?): "
                           f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    return rows


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS),
                    help="which kernel/baseline pair to evaluate (required)")
    ap.add_argument("--cand", default=None,
                    help="path to the candidate .cu (default: the target's kernel)")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU indices to schedule scenarios on, one scenario at a "
                         "time each (default: every visible GPU)")
    ap.add_argument("--gpu", type=int, default=None,
                    help="shorthand for --gpus <N>: a single GPU, run serially in-process")
    ap.add_argument("--configs", default=None, help="comma-separated subset of scenario names")
    ap.add_argument("--set", dest="cfgset", default="claim",
                    help="config set: claim (default, ≤15 min) | attntest | edge | real | all | known_bad")
    ap.add_argument("--no-perf", action="store_true", help="build + correctness only")
    ap.add_argument("--baseline-only", "--oracle-only", dest="baseline_only", action="store_true",
                    help="no build; judge the target's baseline against the fp32 reference")
    ap.add_argument("--namespace", default=None,
                    help="candidate torch op namespace (default: the target's)")
    ap.add_argument("--no-cudnn", action="store_true",
                    help="skip the cuDNN third implementation entirely (also FA_NO_CUDNN=1)")
    ap.add_argument("--no-flashinfer", action="store_true",
                    help="skip the FlashInfer fourth implementation (also FA_NO_FLASHINFER=1)")
    args = ap.parse_args(argv)

    # defaults that depend on --target
    tgt = TARGETS[args.target]
    cand = args.cand or tgt.cand_default
    namespace = args.namespace or tgt.namespace
    try:
        gpus = _resolve_gpus(args.gpus, args.gpu, tgt)
    except (RuntimeError, ValueError) as e:
        print(f"[error] {e}")
        return 1

    torch.cuda.set_device(gpus[0])
    if tgt.name == "blackwell":
        os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

    cfgs = S.get_configs(args.cfgset)
    if args.configs:
        want = {s.strip() for s in args.configs.split(",") if s.strip()}
        cfgs = [c for c in cfgs if c.name in want]
        if not cfgs:
            print(f"no scenarios match {sorted(want)}")
            return 1

    print(f"[target] {tgt.name} ({tgt.gencode}) vs {tgt.baseline_desc}", flush=True)

    # Build ONCE in the parent; parallel workers just load the resulting .so.
    ns, so_path = None, None
    if not args.baseline_only:
        work_dir = os.path.join(_REPO_ROOT, tgt.trials_dir, namespace)
        print(f"[build] {cand} -> ns={namespace}", flush=True)
        t0 = time.time()
        try:
            so_path = build_candidate(cand, namespace, work_dir, tgt)
        except Exception as e:
            print(f"[build] FAILED: {e}")
            return 1
        print(f"[build] ok in {time.time() - t0:.1f}s", flush=True)
        ns = namespace

    # cuDNN / FlashInfer are extra measurements; --baseline-only measures nothing, so skip them.
    with_cudnn = not args.no_cudnn and not args.baseline_only and not os.environ.get("FA_NO_CUDNN")
    if with_cudnn and not cudnn_available():
        # Say so once. Otherwise a missing frontend is indistinguishable from "cuDNN cannot express
        # these configs" -- every row would just read [no-cudnn].
        print(f"[warn] cuDNN frontend not importable ({cudnn_import_error()}) -> no cuDNN column.",
              flush=True)
        with_cudnn = False

    with_fi = (not args.no_flashinfer and not args.baseline_only
               and not os.environ.get("FA_NO_FLASHINFER"))
    if with_fi and not flashinfer_available():
        print(f"[warn] flashinfer not importable ({flashinfer_import_error()}) "
              f"-> no FlashInfer column.", flush=True)
        with_fi = False
    elif with_fi:
        try:
            _fib = _fi_backend(torch.device(f"cuda:{gpus[0]}"))
            print(f"[flashinfer] backend={_fib} via direct trtllm_batch_* "
                  f"(override with FA_FLASHINFER_BACKEND=trtllm-gen|xqa)", flush=True)
        except FlashInferUnsupported as e:
            print(f"[warn] FlashInfer backend unavailable ({e}) -> no FlashInfer column.",
                  flush=True)
            with_fi = False

    have_baseline = baseline_available(tgt)
    if not have_baseline:
        err = baseline_import_error(tgt)
        print(f"[warn] {tgt.baseline_label} baseline not importable"
              f"{f' ({err})' if err else ''} -> fp32-ref-only correctness, perf skipped.", flush=True)
        args.no_perf = True

    parallel = len(gpus) > 1
    mode = (f"{tgt.baseline_label}-vs-fp32ref" if args.baseline_only
            else f"candidate vs {tgt.baseline_label} + fp32 ref" if have_baseline
            else "fp32 ref only")
    where = (f"{len(gpus)} GPUs [{','.join(map(str, gpus))}] in parallel, "
             f"1 scenario at a time each" if parallel else f"GPU {gpus[0]}")
    print(f"[run] {len(cfgs)} scenarios on {where} ({mode})...", flush=True)

    def emit(r: _Result, gpu: int) -> None:
        st = r.status_label(tgt.baseline_label, cudnn_enabled=with_cudnn, fi_enabled=with_fi)
        perf = f"{r.ratio:5.3f}x" if (r.ratio is not None) else "  -  "
        tag = "" if (r.baseline or args.baseline_only) else " [no-baseline]"
        cr = r.cudnn_ratio
        if cr:
            tag += f" [cudnn {cr:.3f}x]"
        elif r.cudnn_note and not r.cudnn_ran:
            tag += " [no-cudnn]"      # only when cuDNN genuinely could not run
        if r.cudnn_suspect:
            tag += " [cudnn-suspect]"
        fr = r.fi_ratio
        if fr:
            tag += f" [fi {fr:.3f}x]"
        elif r.fi_note and not r.fi_ran:
            tag += " [no-fi]"
        if r.fi_suspect:
            tag += " [fi-suspect]"
        extra = ""
        if r.correct and r.base_ms:
            extra = (f"  ({tgt.baseline_label} {r.base_ms*1e3:7.1f}us"
                     f" / ours {r.cand_ms*1e3:7.1f}us")
            if r.cudnn_ms:
                extra += f" / cuDNN {r.cudnn_ms*1e3:7.1f}us"
            if r.fi_ms:
                extra += f" / FI {r.fi_ms*1e3:7.1f}us"
            extra += ")"
        elif r.note:
            extra = "  " + r.note[:90]
        gputag = f" [gpu{gpu}]" if parallel else ""
        print(f"  {st:22s} {r.name:32s} {perf:>7s}{tag}{extra}{gputag}", flush=True)

    t_run = time.time()
    if parallel:
        try:
            rows = _run_parallel(cfgs, ns, tgt, gpus, args.no_perf, so_path, emit,
                                 with_cudnn=with_cudnn, with_fi=with_fi)
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
    else:
        rows = []
        for cfg in cfgs:
            r = _eval_one(cfg, ns, tgt, gpus[0], args.no_perf, with_cudnn=with_cudnn,
                          with_fi=with_fi)
            rows.append(r)
            emit(r, gpus[0])
    elapsed = time.time() - t_run

    n_correct = sum(1 for r in rows if r.correct)
    n_wrong = sum(1 for r in rows if not r.correct and not r.unsup)
    n_unsup = sum(1 for r in rows if r.unsup)
    n_lse_skipped = sum(1 for r in rows if r.lse_skipped)
    ratios = sorted(r.ratio for r in rows if r.correct and r.ratio)
    print("\n" + "-" * 80)
    print(f"correct: {n_correct}/{len(rows)}   unsupported: {n_unsup}   wrong: {n_wrong}"
          f"   lse-skipped: {n_lse_skipped}   ({elapsed:.1f}s on {len(gpus)} GPU"
          f"{'s' if len(gpus) > 1 else ''})")
    if ratios:
        print(f"perf over {len(ratios)} configs:  geomean={_gm(ratios):.3f}x  "
              f"min={ratios[0]:.3f}x  median={ratios[len(ratios)//2]:.3f}x  max={ratios[-1]:.3f}x")
        print(f"  >=0.95x: {sum(1 for x in ratios if x>=0.95)}   "
              f"0.85-0.95x: {sum(1 for x in ratios if 0.85<=x<0.95)}   "
              f"<0.85x: {sum(1 for x in ratios if x<0.85)}")

    # cuDNN / FlashInfer: reported separately so the line above keeps meaning ours-vs-reference
    # (tools/avo scrapes that geomean as its fitness score).
    n_cudnn = sum(1 for r in rows if r.cudnn_ratio)
    n_cudnn_sus = sum(1 for r in rows if r.cudnn_suspect)
    n_cudnn_no = sum(1 for r in rows if not r.cudnn_ran)
    if n_cudnn:
        cr = sorted(r.cudnn_ratio for r in rows if r.cudnn_ratio)
        ours_where_cudnn = [r.ratio for r in rows if r.cudnn_ratio and r.ratio]
        print(f"cuDNN over {n_cudnn} configs:  geomean={_gm(cr):.3f}x vs {tgt.baseline_label}  "
              f"(ours {_gm(ours_where_cudnn):.3f}x on the same {len(ours_where_cudnn)})  "
              f"min={cr[0]:.3f}x  max={cr[-1]:.3f}x")
        print(f"  ours faster than cuDNN on {sum(1 for r in rows if r.cudnn_ratio and r.ratio and r.ratio > r.cudnn_ratio)}"
              f"/{n_cudnn}")
    if not with_cudnn:
        print("cuDNN: disabled for this run")
    elif n_cudnn_no or n_cudnn_sus:
        print(f"  cudnn: {n_cudnn} measured, {n_cudnn_no} unsupported, {n_cudnn_sus} suspect")

    n_fi = sum(1 for r in rows if r.fi_ratio)
    n_fi_sus = sum(1 for r in rows if r.fi_suspect)
    n_fi_no = sum(1 for r in rows if not r.fi_ran)
    if n_fi:
        fr = sorted(r.fi_ratio for r in rows if r.fi_ratio)
        ours_where_fi = [r.ratio for r in rows if r.fi_ratio and r.ratio]
        print(f"FlashInfer over {n_fi} configs:  geomean={_gm(fr):.3f}x vs {tgt.baseline_label}  "
              f"(ours {_gm(ours_where_fi):.3f}x on the same {len(ours_where_fi)})  "
              f"min={fr[0]:.3f}x  max={fr[-1]:.3f}x")
        print(f"  ours faster than FlashInfer on "
              f"{sum(1 for r in rows if r.fi_ratio and r.ratio and r.ratio > r.fi_ratio)}/{n_fi}")
    if not with_fi:
        print("FlashInfer: disabled for this run")
    elif n_fi_no or n_fi_sus:
        print(f"  flashinfer: {n_fi} measured, {n_fi_no} unsupported, {n_fi_sus} suspect")

    # Per-framework unsupported coverage (ours / FA3|FA4 / cuDNN / FlashInfer).
    unsup_rows = [
        r for r in rows
        if r.unsup_frameworks(tgt.baseline_label, cudnn_enabled=with_cudnn, fi_enabled=with_fi)
    ]
    if unsup_rows:
        bl = tgt.baseline_label
        print(f"\nUnsupported scenarios ({len(unsup_rows)}; ● = cannot run):")
        hdr = f"| scenario | ours | {bl}"
        sep = "|---|:-:|:-:"
        if with_cudnn:
            hdr += " | cuDNN"
            sep += "|:-:"
        if with_fi:
            hdr += " | FlashInfer"
            sep += "|:-:"
        hdr += " | note |"
        sep += "|---|"
        print(hdr)
        print(sep)
        for r in unsup_rows:
            o = "●" if r.unsup_ours else ""
            b = "●" if r.unsup_baseline else ""
            cells = [f"`{r.name}`", o, b]
            if with_cudnn:
                cells.append("●" if r.unsup_cudnn(cudnn_enabled=True) else "")
            if with_fi:
                cells.append("●" if r.unsup_fi(fi_enabled=True) else "")
            bits = []
            if r.unsup_ours and r.note:
                bits.append(r.note)
            if r.unsup_baseline and r.baseline_note:
                bits.append(r.baseline_note)
            if with_cudnn and r.unsup_cudnn(cudnn_enabled=True) and r.cudnn_note:
                bits.append(r.cudnn_note)
            if with_fi and r.unsup_fi(fi_enabled=True) and r.fi_note:
                bits.append(r.fi_note)
            if not bits and r.note:
                bits.append(r.note)
            note = "; ".join(bits).replace("\n", " ")
            if len(note) > 72:
                note = note[:69] + "…"
            cells.append(note)
            print("| " + " | ".join(cells) + " |")

    # Per-set CSV breakdown + markdown table(s) to stdout; then plots when ratios exist.
    plot_sets = S.plot_sets_for(args.cfgset)
    if plot_sets and rows:
        try:
            from bench import plot as P
            for path in P.write_breakdown_csvs(
                    rows, arch=tgt.name, cfgset=args.cfgset, ref_label=tgt.baseline_label,
                    cudnn_enabled=with_cudnn, fi_enabled=with_fi):
                print(f"[csv] {path}", flush=True)
            md = P.format_breakdown_markdown(
                rows, arch=tgt.name, cfgset=args.cfgset, ref_label=tgt.baseline_label,
                cudnn_enabled=with_cudnn, fi_enabled=with_fi)
            if md:
                print("\n" + md, flush=True)
            if any(r.ratio for r in rows):
                results_path = os.path.join(_REPO_ROOT, "docs", f"last_results_{tgt.name}.json")
                P.write_measured(results_path, rows, arch=tgt.name, cfgset=args.cfgset)
                print(f"[plot] wrote {results_path}", flush=True)
                for p in P.plot_cfgset(tgt.name, args.cfgset, P.load_measured(results_path)):
                    print(f"[plot] {p}", flush=True)
        except Exception as e:
            print(f"[report] skipped: {type(e).__name__}: {e}", flush=True)

    return 0 if n_wrong == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
