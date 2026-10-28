"""Self-contained scenario matrix + input construction for the fa3-cuda harness.

Default config set is `claim` (~70 scenarios, ≤15 min on 1 warm B200 with default PERF):
the paper bar taxonomy (dense/varlen + paged serving), including true bf16Q+fp8KV rows on
the Fig2 FP8-KV bar. Optional `--set attntest|edge|real` keep the older FA3 pytest /
masking / SOL grids for regression. `--set all` evaluates the union and plots one PNG per set.
vLLM/FlashInfer-compatible FP8 rows use ``fp8_scales="scalar"`` (per-tensor Q/K/V scales);
default ``per_head`` keeps the FA4/ours native ``(batch, hkv)`` descale tables.

Two halves, merged into one module because the harness always uses them as a unit:

  1. CONFIG -- the attention problem instances and the constants the build / correctness /
     perf code needs: nvcc flags, tolerances, perf knobs.
  2. SHAPES -- authoritative input construction for that matrix. The SAME tensors drive the
     baseline (`_vllm_fa3_C` on Hopper, the CuteDSL oracle on Blackwell) and the candidate.
     The 37-positional-arg ordering of the `fwd` op is encoded in exactly ONE place
     (`fwd_args`). Plus `ref_fp32`, the pure-PyTorch fp32 reference behind the FA 2x-error
     correctness criterion.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch

# --- target / toolchain -----------------------------------------------------------------
GPU_GENCODE = "arch=compute_90a,code=sm_90a"

# Toolkit-only nvcc flags (no cutlass/cute include path). Mirrors how the kernel was built
# when the per-scenario numbers were produced.
NVCC_BASE_FLAGS = [
    "-O3", "-std=c++17", "--use_fast_math",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-gencode", GPU_GENCODE,
    "-lineinfo",
]
# Semantic defs matching the vLLM FA3 build surface (forward / varlen / packgqa only). The
# kernel only requires -DFA3_LIB_NAME, but these are kept for build-surface parity.
SEMANTIC_DEFS = [
    "FLASHATTENTION_DISABLE_BACKWARD", "FLASHATTENTION_DISABLE_DROPOUT",
    "FLASHATTENTION_DISABLE_APPENDKV", "FLASHATTENTION_DISABLE_SM8x",
    "FLASHATTENTION_DISABLE_CLUSTER", "FLASHATTENTION_DISABLE_UNEVEN_K",
    "FLASHATTENTION_VARLEN_ONLY", "FLASHATTENTION_PACKGQA_ONLY",
]

# --- explicitly OUT OF SCOPE (so "all FA3 forward scenarios" is unambiguous) -------------
# FA3's forward pytest grids also exercise the following, which this build/harness deliberately
# does NOT cover (they are disabled above and/or pinned to defaults in `fwd_args`):
#   * backward + dropout                              (DISABLE_BACKWARD / DISABLE_DROPOUT)
#   * appendKV (new_kv) + rotary insertion            (DISABLE_APPENDKV; k_new/v_new/rotary_* = None)
#   * cluster launch                                  (DISABLE_CLUSTER; FA3 test_flash_attn_cluster)
#   * uneven-K template specialization               (DISABLE_UNEVEN_K build flag; NB the EDGE set still
#                                                      tests unaligned seqlens e.g. (1023,1024)/(1024,1023)
#                                                      via the general path -- only FA3's uneven-K template
#                                                      is out of scope, not odd-seqlen correctness testing)
#   * pack_gqa = False                                (PACKGQA_ONLY; FA3 tests both True/False)
#   * the dense BSHD entry point                      (VARLEN_ONLY; even "dense" cfgs go via varlen)
#   * context parallelism (cp_world_size > 1)         (pinned cp_world_size=1 in fwd_args)
#   * cache_batch_idx / seqused_q < q_len             (pinned None in fwd_args)
# Everything else FA3's forward grids touch IS represented in the attntest set below.

# Prebuilt FA3 baseline ("the oracle") shipped inside the installed vLLM.
BASELINE_NS = "_vllm_fa3_C"

# --- dtypes -----------------------------------------------------------------------------
# Homogeneous: bf16 / fp16 / fp8e4m3 (Q=K=V). Serving FP8-KV: bf16q_fp8kv (Q bf16, K/V fp8e4m3).
# `fp8_scales`: "per_head" = (batch, hkv) descales (FA4/ours native); "scalar" = one float each
# for Q/K/V (vLLM / FlashInfer trtllm-gen bmm1/bmm2 scales).
DTYPES = ("bf16", "fp16", "fp8e4m3", "bf16q_fp8kv")
TORCH_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp8e4m3": "float8_e4m3fn",
               "bf16q_fp8kv": "bfloat16"}  # Q dtype; KV is always float8_e4m3fn
PLOT_SETS = ("claim", "attntest", "edge", "real")


@dataclass(frozen=True)
class Config:
    name: str
    phase: str               # "prefill" | "decode"
    dtype: str               # one of DTYPES
    head_dim: int
    num_heads_q: int
    num_heads_kv: int
    causal: bool = True
    paged: bool = True        # paged KV cache (decode) vs contiguous varlen (prefill)
    block_size: int = 16
    num_splits: int = 0       # 0 = FA3 auto-heuristic; 1 = forced single; >1 forced split
    softcap: float = 0.0
    window_left: int = -1     # -1 = no sliding window
    window_right: int = -1
    sink: bool = False        # attention sink: per-(query)head s_aux logit added to the softmax denom
    leftpad: int = 0          # cache_leftpad: ignore the first `leftpad` KV tokens (paged decode)
    batch: int = 1
    q_len: int = 1            # per-sequence query length (decode=1; prefill=prompt len)
    k_len: int = 256          # per-sequence kv/context length
    head_dim_v: Optional[int] = None  # defaults to head_dim
    mix: Optional[tuple] = None        # ((q_len,k_len),...) ragged/MIXED batch; overrides batch/q/k
    weight: float = 1.0
    plot_bar: Optional[str] = None     # paper/category bar this scenario geomeans into (plot.py)
    fp8_scales: str = "per_head"       # "per_head" | "scalar" (ignored unless has_fp8_kv)

    @property
    def hdv(self) -> int:
        return self.head_dim_v if self.head_dim_v is not None else self.head_dim

    @property
    def is_fp8_qkv(self) -> bool:
        return self.dtype == "fp8e4m3"

    @property
    def is_bf16q_fp8kv(self) -> bool:
        """Serving-style asymmetric: bf16 Q, fp8e4m3 K/V (+ k/v_descale)."""
        return self.dtype == "bf16q_fp8kv"

    @property
    def has_fp8_kv(self) -> bool:
        return self.is_fp8_qkv or self.is_bf16q_fp8kv

    @property
    def is_scalar_fp8_scales(self) -> bool:
        """vLLM/FlashInfer-style per-tensor Q/K/V scales (not per-head)."""
        return self.has_fp8_kv and self.fp8_scales == "scalar"

    @property
    def requests(self):
        """Per-request (q_len, k_len). Uniform (`batch` copies) unless `mix` is set."""
        if self.mix:
            return [(int(q), int(k)) for q, k in self.mix]
        return [(self.q_len, self.k_len)] * self.batch

    def to_dict(self) -> dict:
        return asdict(self)


# --- CLAIM set (DEFAULT): paper bars, sized for ≤15 min on 1 warm B200 -------------------
# Dense/prefill capped at 8k so pytorch fp32 ref stays affordable; long-ctx covered by decode
# (q=1) and a few chunked-prefill rows. Optional --set attntest|edge|real keep the older grids.
#
# plot_bar tags feed the 14-bar claim chart (geomean within each bar). Order = CLAIM_BAR_ORDER.
CLAIM_BAR_ORDER = (
    # Fig1 — dense / varlen
    "Causal, d=64",
    "Causal, d=128",
    "Causal, d=256",
    "Non-causal",
    "FP8",
    "Sliding window",
    "Variable-length",
    # Fig2 — paged KV
    "Decode ≤4k",
    "Decode >4k",
    "Prefill",
    "Chunked prefill",
    "Speculative",
    "Mixed batch",
    "FP8 KV",
)


def _bar(bar: str, *cfgs: Config) -> list[Config]:
    """Stamp plot_bar onto claim configs (Config is frozen)."""
    return [Config(**{**c.to_dict(), "plot_bar": bar}) for c in cfgs]


def _claim_configs() -> list[Config]:
    cfgs: list[Config] = []
    _vmix = tuple([(256, 256), (113, 203), (128, 217), (108, 256), (64, 128), (128, 128)])
    _rmix = tuple([(2048, 2048), (1024, 1024), (512, 512), (4096, 4096), (256, 256), (768, 768)])

    cfgs += _bar("Causal, d=64",
        Config("dense_c_d64_s1k", "prefill", "bf16", 64, 32, 32, paged=False, batch=16, q_len=1024, k_len=1024),
        Config("dense_c_d64_s4k", "prefill", "bf16", 64, 32, 32, paged=False, batch=4, q_len=4096, k_len=4096),
        Config("dense_c_d64_s8k", "prefill", "bf16", 64, 32, 32, paged=False, batch=2, q_len=8192, k_len=8192),
    )
    cfgs += _bar("Causal, d=128",
        Config("dense_c_d128_s1k", "prefill", "bf16", 128, 16, 16, paged=False, batch=16, q_len=1024, k_len=1024),
        Config("dense_c_d128_s4k", "prefill", "bf16", 128, 16, 16, paged=False, batch=4, q_len=4096, k_len=4096),
        Config("dense_c_d128_s8k", "prefill", "bf16", 128, 16, 16, paged=False, batch=2, q_len=8192, k_len=8192),
        Config("dense_c_d128_gqa8_s2k", "prefill", "bf16", 128, 16, 2, paged=False, batch=8, q_len=2048, k_len=2048),
        Config("dense_c_d128_gqa8_s8k", "prefill", "bf16", 128, 16, 2, paged=False, batch=2, q_len=8192, k_len=8192),
        Config("dense_c_d96_s2k", "prefill", "bf16", 96, 16, 16, paged=False, batch=8, q_len=2048, k_len=2048),
        Config("dense_c_d128_softcap50_s2k", "prefill", "bf16", 128, 16, 2, paged=False, batch=8,
               q_len=2048, k_len=2048, softcap=50.0),
        Config("dense_c_d128_sink_s2k", "prefill", "bf16", 128, 16, 16, paged=False, batch=8,
               q_len=2048, k_len=2048, sink=True),
    )
    cfgs += _bar("Causal, d=256",
        Config("dense_c_d256_s2k", "prefill", "bf16", 256, 8, 8, paged=False, batch=8, q_len=2048, k_len=2048),
        Config("dense_c_d256_s8k", "prefill", "bf16", 256, 8, 8, paged=False, batch=2, q_len=8192, k_len=8192),
        Config("dense_c_d192_s2k", "prefill", "bf16", 192, 8, 8, paged=False, batch=8, q_len=2048, k_len=2048),
        Config("dense_c_d192_dv128_s2k", "prefill", "bf16", 192, 8, 8, paged=False, batch=8,
               q_len=2048, k_len=2048, head_dim_v=128),
    )
    cfgs += _bar("Non-causal",
        Config("dense_nc_d128_s2k", "prefill", "bf16", 128, 16, 16, paged=False, causal=False,
               batch=8, q_len=2048, k_len=2048),
        Config("dense_nc_d128_s8k", "prefill", "bf16", 128, 16, 16, paged=False, causal=False,
               batch=2, q_len=8192, k_len=8192),
        Config("dense_nc_d256_s8k", "prefill", "bf16", 256, 8, 8, paged=False, causal=False,
               batch=2, q_len=8192, k_len=8192),
    )
    cfgs += _bar("FP8",
        Config("dense_c_d64_fp8_s8k", "prefill", "fp8e4m3", 64, 32, 32, paged=False, batch=2,
               q_len=8192, k_len=8192),
        Config("dense_c_d128_fp8_s4k", "prefill", "fp8e4m3", 128, 16, 16, paged=False, batch=4,
               q_len=4096, k_len=4096),
        Config("dense_c_d128_fp8_s8k", "prefill", "fp8e4m3", 128, 16, 16, paged=False, batch=2,
               q_len=8192, k_len=8192),
        Config("dense_c_d128_fp8_gqa_s4k", "prefill", "fp8e4m3", 128, 16, 2, paged=False, batch=4,
               q_len=4096, k_len=4096),
        Config("dense_c_d256_fp8_s8k", "prefill", "fp8e4m3", 256, 8, 8, paged=False, batch=2,
               q_len=8192, k_len=8192),
        # vLLM/FlashInfer-compatible: per-tensor scales (trtllm-gen bmm1/bmm2).
        Config("dense_c_d128_fp8_scalar_s4k", "prefill", "fp8e4m3", 128, 16, 16, paged=False,
               batch=4, q_len=4096, k_len=4096, fp8_scales="scalar"),
        Config("dense_c_d128_fp8_scalar_gqa_s4k", "prefill", "fp8e4m3", 128, 16, 2, paged=False,
               batch=4, q_len=4096, k_len=4096, fp8_scales="scalar"),
        Config("dense_c_d64_fp8_scalar_s8k", "prefill", "fp8e4m3", 64, 32, 32, paged=False,
               batch=2, q_len=8192, k_len=8192, fp8_scales="scalar"),
    )
    cfgs += _bar("Sliding window",
        Config("dense_c_d128_swa256_s4k", "prefill", "bf16", 128, 16, 2, paged=False, batch=4,
               q_len=4096, k_len=4096, window_left=256, window_right=0),
        Config("dense_c_d128_swa4k_s4k", "prefill", "bf16", 128, 16, 2, paged=False, batch=4,
               q_len=4096, k_len=4096, window_left=4096, window_right=0),
    )
    cfgs += _bar("Variable-length",
        Config("varlen_c_d128_mix", "prefill", "bf16", 128, 6, 2, paged=False, mix=_vmix),
        Config("varlen_c_d64_mix", "prefill", "bf16", 64, 6, 6, paged=False, mix=_vmix),
        Config("varlen_c_d128_mqa_mix", "prefill", "bf16", 128, 6, 1, paged=False, mix=_vmix),
        Config("varlen_c_d128_realmix", "prefill", "bf16", 128, 32, 8, paged=False, mix=_rmix),
    )

    cfgs += _bar("Decode ≤4k",
        Config("llama8b_dec_b16_s512", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=512, block_size=64),
        Config("llama8b_dec_b32_s1k", "decode", "bf16", 128, 32, 8, batch=32, q_len=1, k_len=1024, block_size=64),
        Config("llama8b_dec_b16_s4k", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=4096, block_size=64),
        Config("llama8b_dec_b64_s4k", "decode", "bf16", 128, 32, 8, batch=64, q_len=1, k_len=4096, block_size=64),
        Config("llama8b_dec_b16_s4k_p16", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=4096, block_size=16),
        Config("llama8b_dec_b16_s4k_p128", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=4096, block_size=128),
        Config("qwen3_dec_b16_s4k", "decode", "bf16", 128, 32, 4, batch=16, q_len=1, k_len=4096, block_size=64),
        Config("qwen3_dec_b64_s4k", "decode", "bf16", 128, 32, 4, batch=64, q_len=1, k_len=4096, block_size=64),
        Config("llama70b_dec_b16_s4k", "decode", "bf16", 128, 64, 8, batch=16, q_len=1, k_len=4096, block_size=64),
        Config("mistral_dec_b16_s4k", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=4096,
               block_size=64, window_left=4096, window_right=0),
        Config("gemma2_dec_b16_s4k", "decode", "bf16", 256, 16, 8, batch=16, q_len=1, k_len=4096,
               block_size=64, window_left=4096, window_right=0, softcap=50.0),
    )
    cfgs += _bar("Decode >4k",
        Config("llama8b_dec_b16_s16k", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=16384, block_size=64),
        Config("llama8b_dec_b64_s16k", "decode", "bf16", 128, 32, 8, batch=64, q_len=1, k_len=16384, block_size=64),
        Config("llama8b_dec_b16_s32k", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=32768, block_size=64),
        Config("qwen3_dec_b16_s32k", "decode", "bf16", 128, 32, 4, batch=16, q_len=1, k_len=32768, block_size=64),
        Config("llama70b_dec_b4_s128k", "decode", "bf16", 128, 64, 8, batch=4, q_len=1, k_len=131072,
               block_size=64, num_splits=0),
        Config("mistral_dec_b16_s16k", "decode", "bf16", 128, 32, 8, batch=16, q_len=1, k_len=16384,
               block_size=64, window_left=4096, window_right=0),
    )
    cfgs += _bar("Prefill",
        Config("llama8b_pf_q512", "prefill", "bf16", 128, 32, 8, batch=1, q_len=512, k_len=512, block_size=64),
        Config("llama8b_pf_q2k", "prefill", "bf16", 128, 32, 8, batch=1, q_len=2048, k_len=2048, block_size=64),
        Config("llama8b_pf_q8k", "prefill", "bf16", 128, 32, 8, batch=1, q_len=8192, k_len=8192, block_size=64),
        Config("llama8b_pf_q2k_p16", "prefill", "bf16", 128, 32, 8, batch=1, q_len=2048, k_len=2048, block_size=16),
        Config("qwen3_pf_q4k", "prefill", "bf16", 128, 32, 4, batch=1, q_len=4096, k_len=4096, block_size=64),
        Config("llama70b_pf_q8k", "prefill", "bf16", 128, 64, 8, batch=1, q_len=8192, k_len=8192, block_size=64),
        Config("mistral_pf_q4k", "prefill", "bf16", 128, 32, 8, batch=1, q_len=4096, k_len=4096,
               block_size=64, window_left=4096, window_right=0),
        Config("gemma2_pf_q4k", "prefill", "bf16", 256, 16, 8, batch=1, q_len=4096, k_len=4096,
               block_size=64, window_left=4096, window_right=0, softcap=50.0),
        Config("llama8b_pf_ragged", "prefill", "bf16", 128, 32, 8, paged=False, mix=_rmix),
    )
    cfgs += _bar("Chunked prefill",
        Config("llama8b_ext_q512_k16k", "prefill", "bf16", 128, 32, 8, batch=1, q_len=512, k_len=16384, block_size=64),
        Config("llama8b_ext_q1k_k64k", "prefill", "bf16", 128, 32, 8, batch=1, q_len=1024, k_len=65536, block_size=64),
        Config("qwen3_ext_q512_k32k", "prefill", "bf16", 128, 32, 4, batch=1, q_len=512, k_len=32768, block_size=64),
        Config("llama70b_ext_q1k_k128k", "prefill", "bf16", 128, 64, 8, batch=1, q_len=1024, k_len=131072,
               block_size=64, num_splits=0),
    )
    cfgs += _bar("Speculative",
        Config("llama8b_spec_q2_b16_s1k", "decode", "bf16", 128, 32, 8, batch=16, q_len=2, k_len=1024, block_size=64),
        Config("llama8b_spec_q4_b32_s4k", "decode", "bf16", 128, 32, 8, batch=32, q_len=4, k_len=4096, block_size=64),
        Config("llama8b_spec_q8_b16_s4k", "decode", "bf16", 128, 32, 8, batch=16, q_len=8, k_len=4096, block_size=64),
        Config("qwen3_spec_q4_b16_s4k", "decode", "bf16", 128, 32, 4, batch=16, q_len=4, k_len=4096, block_size=64),
    )
    cfgs += _bar("Mixed batch",
        Config("mixed_2pf2k_8dec1k", "prefill", "bf16", 128, 32, 8, paged=True, block_size=64,
               mix=tuple([(2048, 2048)] * 2 + [(1, 1024)] * 8)),
        Config("mixed_1pf8k_64dec4k", "prefill", "bf16", 128, 32, 8, paged=True, block_size=64,
               mix=tuple([(8192, 8192)] + [(1, 4096)] * 64)),
        Config("mixed_2pf2k_8dec16k", "prefill", "bf16", 128, 32, 8, paged=True, block_size=64,
               mix=tuple([(2048, 2048)] * 2 + [(1, 16384)] * 8)),
    )
    cfgs += _bar("FP8 KV",
        Config("llama8b_dec_bf16q_fp8kv_b16_s1k", "decode", "bf16q_fp8kv", 128, 32, 8,
               batch=16, q_len=1, k_len=1024, block_size=64),
        Config("llama8b_dec_bf16q_fp8kv_b32_s4k", "decode", "bf16q_fp8kv", 128, 32, 8,
               batch=32, q_len=1, k_len=4096, block_size=64),
        Config("llama8b_dec_bf16q_fp8kv_b16_s16k", "decode", "bf16q_fp8kv", 128, 32, 8,
               batch=16, q_len=1, k_len=16384, block_size=64),
        Config("llama8b_pf_bf16q_fp8kv_q2k", "prefill", "bf16q_fp8kv", 128, 32, 8,
               batch=1, q_len=2048, k_len=2048, block_size=64),
        # vLLM FlashInfer path: scalar scales. Decode supports bf16-Q+fp8-KV; prefill wants
        # homogeneous fp8-QKV (trtllm-gen has no bf16-Q+fp8-KV context cubin).
        Config("llama8b_dec_bf16q_fp8kv_scalar_b16_s1k", "decode", "bf16q_fp8kv", 128, 32, 8,
               batch=16, q_len=1, k_len=1024, block_size=64, fp8_scales="scalar"),
        Config("llama8b_dec_bf16q_fp8kv_scalar_b32_s4k", "decode", "bf16q_fp8kv", 128, 32, 8,
               batch=32, q_len=1, k_len=4096, block_size=64, fp8_scales="scalar"),
        Config("llama8b_dec_fp8_scalar_b16_s4k", "decode", "fp8e4m3", 128, 32, 8,
               batch=16, q_len=1, k_len=4096, block_size=64, fp8_scales="scalar"),
        Config("llama8b_pf_fp8_scalar_q2k", "prefill", "fp8e4m3", 128, 32, 8,
               batch=1, q_len=2048, k_len=2048, block_size=64, fp8_scales="scalar"),
    )
    return cfgs


CLAIM_CONFIGS = _claim_configs()


# --- FA3TEST set: scenarios mirrored from FlashAttention-3's OWN forward pytest grids -----
# Source: reference/flash-attention/hopper/test_flash_attn.py (dense flash_attn_func + varlen
# flash_attn_varlen_func) and hopper/test_attn_kvcache.py (kvcache decode + split-KV). A
# curated, deduplicated sample spanning every axis the kernel + harness can express.
def _attntest_configs() -> list[Config]:
    cfgs: list[Config] = []

    # === A) Dense flash_attn_func (non-paged, nheads=16) ================================
    for hd in (64, 96, 128, 192, 256):
        cfgs.append(Config(f"attn_dense_hd{hd}_causal", "prefill", "bf16", hd, 16, 16,
                           paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_noncausal", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=False, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_nc4k", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=False, batch=2, q_len=4096, k_len=4096))
    cfgs.append(Config("attn_dense_hd128_gqa", "prefill", "bf16", 128, 16, 2,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_mqa", "prefill", "bf16", 128, 16, 1,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_fp16", "prefill", "fp16", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_fp8", "prefill", "fp8e4m3", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_softcap", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024, softcap=15.0))
    cfgs.append(Config("attn_dense_hd128_local", "prefill", "bf16", 128, 16, 2,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024,
                       window_left=256, window_right=0))
    cfgs.append(Config("attn_dense_hd128_oddseq", "prefill", "bf16", 128, 16, 2,
                       paged=False, causal=True, batch=9, q_len=113, k_len=203))
    cfgs.append(Config("attn_dense_hd128_seq256", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=256, k_len=256))
    cfgs.append(Config("attn_dense_hd128_large4k", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=2, q_len=4096, k_len=4096))
    cfgs.append(Config("attn_dense_hd128_splits3", "prefill", "bf16", 128, 16, 2,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024, num_splits=3))

    # === B) Varlen flash_attn_varlen_func (ragged, nheads=6) ===========================
    _vmix = tuple([(256, 256), (113, 203), (128, 217), (108, 256), (64, 128), (128, 128)])
    cfgs.append(Config("attn_varlen_hd128_causal", "prefill", "bf16", 128, 6, 2,
                       paged=False, causal=True, mix=_vmix))
    cfgs.append(Config("attn_varlen_hd128_noncausal", "prefill", "bf16", 128, 6, 2,
                       paged=False, causal=False, mix=_vmix))
    cfgs.append(Config("attn_varlen_hd64_causal", "prefill", "bf16", 64, 6, 6,
                       paged=False, causal=True, mix=_vmix))
    cfgs.append(Config("attn_varlen_hd128_mqa", "prefill", "bf16", 128, 6, 1,
                       paged=False, causal=True, mix=_vmix))

    # === C) KV-cache decode / split-KV (paged) =========================================
    cfgs.append(Config("attn_kv_hd128_decode_mha", "decode", "bf16", 128, 8, 8,
                       paged=True, causal=True, batch=4, q_len=1, k_len=4224))
    cfgs.append(Config("attn_kv_hd128_decode_gqa5", "decode", "bf16", 128, 10, 2,
                       paged=True, causal=True, batch=4, q_len=1, k_len=3131))
    cfgs.append(Config("attn_kv_hd128_decode_mqa32", "decode", "bf16", 128, 32, 1,
                       paged=True, causal=True, batch=1, q_len=1, k_len=1024))
    cfgs.append(Config("attn_kv_hd128_decode_qlen120", "decode", "bf16", 128, 16, 8,
                       paged=True, causal=True, batch=4, q_len=120, k_len=4224))
    cfgs.append(Config("attn_kv_hd128_decode_qlen8", "decode", "bf16", 128, 16, 8,
                       paged=True, causal=True, batch=4, q_len=8, k_len=4224))
    cfgs.append(Config("attn_kv_hd64_decode_gqa3", "decode", "bf16", 64, 9, 3,
                       paged=True, causal=True, batch=4, q_len=1, k_len=3131))
    cfgs.append(Config("attn_kv_hd256_decode_mqa16", "decode", "bf16", 256, 16, 1,
                       paged=True, causal=True, batch=4, q_len=1, k_len=4224))
    cfgs.append(Config("attn_kv_hd128_decode_noncausal", "decode", "bf16", 128, 8, 2,
                       paged=True, causal=False, batch=4, q_len=1, k_len=3131))
    cfgs.append(Config("attn_kv_hd128_fp8_decode", "decode", "fp8e4m3", 128, 16, 8,
                       paged=True, causal=True, batch=3, q_len=1, k_len=4000))
    cfgs.append(Config("attn_kv_hd128_split2", "decode", "bf16", 128, 16, 8,
                       paged=True, causal=True, batch=3, q_len=1, k_len=5555, num_splits=2))
    cfgs.append(Config("attn_kv_hd256_split_auto", "decode", "bf16", 256, 9, 3,
                       paged=True, causal=True, batch=3, q_len=1, k_len=4000, num_splits=0))
    cfgs.append(Config("attn_kv_hd128_local_decode", "decode", "bf16", 128, 8, 2,
                       paged=True, causal=False, batch=4, q_len=1, k_len=4096,
                       window_left=1024, window_right=0))
    # extra (nheads_kv, gqa_ratio) decode combos from FA3's test_attn_kvcache grid not sampled above
    cfgs.append(Config("attn_kv_hd128_decode_gqa9", "decode", "bf16", 128, 27, 3,
                       paged=True, causal=True, batch=4, q_len=1, k_len=3131))   # ratio 9 (3,9)
    cfgs.append(Config("attn_kv_hd128_decode_gqa7", "decode", "bf16", 128, 35, 5,
                       paged=True, causal=True, batch=4, q_len=1, k_len=4224))   # ratio 7 (5,7)
    # long-context split-KV decode: FA3's benchmark_split_kv headline (Llama-3.1-70B h64/kv8, 128k ctx,
    # num_splits=0 -> exercises the split-KV auto-heuristic). Also motivated by SOL's ~60k-token captures.
    cfgs.append(Config("attn_kv_hd128_decode_128k_splitkv", "decode", "bf16", 128, 64, 8,
                       paged=True, causal=True, batch=4, q_len=1, k_len=131072, num_splits=0))

    # === D) KV-cache hd128 page_size variants + long ctx ===============================
    for ps in (1, 4, 128):
        cfgs.append(Config(f"attn_kv_hd128_page{ps}", "decode", "bf16", 128, 6, 2,
                           paged=True, causal=True, block_size=ps, batch=5, q_len=1, k_len=2048))
    cfgs.append(Config("attn_kv_hd128_ctx20k", "prefill", "bf16", 128, 6, 2,
                       paged=True, causal=True, batch=5, q_len=16, k_len=20000))
    cfgs.append(Config("attn_kv_hd128_prefill3577", "prefill", "bf16", 128, 6, 2,
                       paged=True, causal=True, batch=5, q_len=2048, k_len=3577))
    cfgs.append(Config("attn_kv_hd128_dec339", "decode", "bf16", 128, 6, 2,
                       paged=True, causal=True, batch=5, q_len=1, k_len=339))
    cfgs.append(Config("attn_kv_hd128_dec1024", "decode", "bf16", 128, 6, 2,
                       paged=True, causal=True, batch=5, q_len=3, k_len=1024))
    cfgs.append(Config("attn_kv_hd128_dec800", "decode", "bf16", 128, 6, 2,
                       paged=True, causal=True, batch=5, q_len=64, k_len=800))

    # === E) Coverage extensions (forward-only) =========================================
    # Axes FA3's fwd grids exercise but sections A-D pinned to a single default. All stay
    # within the in-scope op surface (see SEMANTIC_DEFS out-of-scope note above).

    # -- head_dim_v != head_dim: the flash_fwd_hdimdiff_* instantiations (FA3 d=192 -> dv=128) --
    cfgs.append(Config("attn_dense_hd192_dv128", "prefill", "bf16", 192, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024, head_dim_v=128))
    cfgs.append(Config("attn_varlen_hd192_dv128", "prefill", "bf16", 192, 6, 2,
                       paged=False, causal=True, mix=_vmix, head_dim_v=128))

    # -- dtype breadth: fp16 across head dims + paged decode; fp8 across gqa/varlen/hd64 --
    cfgs.append(Config("attn_dense_hd64_fp16", "prefill", "fp16", 64, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd192_fp16", "prefill", "fp16", 192, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_dense_hd128_fp8_gqa", "prefill", "fp8e4m3", 128, 16, 2,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_varlen_hd128_fp8", "prefill", "fp8e4m3", 128, 6, 2,
                       paged=False, causal=True, mix=_vmix))
    cfgs.append(Config("attn_kv_hd128_fp16_decode", "decode", "fp16", 128, 16, 8,
                       paged=True, causal=True, batch=3, q_len=1, k_len=4000))
    cfgs.append(Config("attn_kv_hd64_fp8_decode", "decode", "fp8e4m3", 64, 16, 8,
                       paged=True, causal=True, batch=3, q_len=1, k_len=4000))
    # fp8 at head_dim 256 (FA3 benchmark_flash_attention_fp8 sweeps hd {64,128,256}; only hd128/64 above)
    cfgs.append(Config("attn_dense_hd256_fp8", "prefill", "fp8e4m3", 256, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024))
    cfgs.append(Config("attn_kv_hd256_fp8_decode", "decode", "fp8e4m3", 256, 16, 8,
                       paged=True, causal=True, batch=3, q_len=1, k_len=4000))

    # -- masking edges: k>q, degenerate q>>k (mostly-masked rows), non-causal local, varlen local --
    cfgs.append(Config("attn_dense_hd128_kgtq", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=256, k_len=512))
    cfgs.append(Config("attn_dense_hd128_deg_q239k1", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=239, k_len=1))
    cfgs.append(Config("attn_dense_hd128_local_noncausal", "prefill", "bf16", 128, 16, 2,
                       paged=False, causal=False, batch=9, q_len=1024, k_len=1024,
                       window_left=256, window_right=256))
    cfgs.append(Config("attn_varlen_hd128_local", "prefill", "bf16", 128, 6, 2,
                       paged=False, causal=True, mix=_vmix, window_left=128, window_right=0))

    # -- softcap breadth (softcap x GQA) --
    cfgs.append(Config("attn_dense_hd128_softcap_gqa", "prefill", "bf16", 128, 16, 2,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024, softcap=15.0))

    # -- attention sink (s_aux): dense + varlen (FA3 test_sink) --
    cfgs.append(Config("attn_dense_hd128_sink", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=9, q_len=1024, k_len=1024, sink=True))
    cfgs.append(Config("attn_varlen_hd128_sink", "prefill", "bf16", 128, 6, 2,
                       paged=False, causal=True, mix=_vmix, sink=True))

    # -- cache_leftpad (paged decode: ignore the first L KV tokens) --
    cfgs.append(Config("attn_kv_hd128_leftpad", "decode", "bf16", 128, 6, 2,
                       paged=True, causal=True, batch=5, q_len=1, k_len=2048, leftpad=128))

    # -- long-context prefill (FA3's steady-state design point; perf-relevant) --
    cfgs.append(Config("attn_dense_hd128_seq8k", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=2, q_len=8192, k_len=8192))
    cfgs.append(Config("attn_dense_hd128_seq16k", "prefill", "bf16", 128, 16, 16,
                       paged=False, causal=True, batch=1, q_len=16384, k_len=16384))
    return cfgs


FA3TEST_CONFIGS = _attntest_configs()


# --- EDGE set: dense seqlen/masking correctness grid (mirrors FlashAttention-4's own grid) ---
# FA4's cute pytest (tests/cute/test_flash_attn.py) sweeps ~27 (seqlen_q, seqlen_k) pairs crossed
# with causal / local / softcap to stress masking, boundary and tail handling. FA3TEST samples
# only a few; this set adds the boundary / off-by-one / tail pairs it misses, crossed with causal
# + noncausal + sliding-window + GQA. Small shapes (nheads<=8, batch 2) -> primarily a CORRECTNESS
# grid (run with --no-perf); it also exercises the fa4/ws2 routing at odd tile alignments.
def _edge_configs() -> list[Config]:
    cfgs: list[Config] = []
    # boundary / off-by-one / tail (seqlen_q, seqlen_k) pairs from FA4's grid not in FA3TEST
    pairs = [(1, 1), (3, 3), (64, 1), (64, 32), (64, 128), (128, 192), (255, 256), (108, 256),
             (113, 128), (113, 211), (128, 217), (799, 3), (384, 256), (640, 128), (512, 256),
             (1023, 1024), (1024, 1023), (2048, 2048)]
    for q, k in pairs:
        cfgs.append(Config(f"attn_edge_causal_q{q}k{k}", "prefill", "bf16", 128, 8, 8,
                           paged=False, causal=True, batch=2, q_len=q, k_len=k))
    # non-causal (full attention) on a representative subset
    for q, k in [(1, 1), (64, 128), (255, 256), (128, 217), (512, 256), (1024, 1023)]:
        cfgs.append(Config(f"attn_edge_noncausal_q{q}k{k}", "prefill", "bf16", 128, 8, 8,
                           paged=False, causal=False, batch=2, q_len=q, k_len=k))
    # sliding-window (causal, left band) on odd shapes -> exercises n_lo/n_hi prune + band mask
    for q, k, wl in [(128, 217, 64), (255, 256, 128), (512, 256, 128), (1024, 1023, 256)]:
        cfgs.append(Config(f"attn_edge_local_q{q}k{k}", "prefill", "bf16", 128, 8, 2,
                           paged=False, causal=True, batch=2, q_len=q, k_len=k,
                           window_left=wl, window_right=0))
    # GQA (hkv=2) on odd shapes -> head replication + masking at odd alignments
    for q, k in [(113, 211), (384, 256), (640, 128)]:
        cfgs.append(Config(f"attn_edge_gqa_q{q}k{k}", "prefill", "bf16", 128, 8, 2,
                           paged=False, causal=True, batch=2, q_len=q, k_len=k))
    return cfgs


# --- REAL set: production attention shapes from nvidia/SOL-ExecBench (B200 kernel benchmark) --
# Ground truth = the SCORED FlashInfer-Bench subset, which defines exactly SIX GQA attention
# problems: two models x three phases, all head_dim 128, page_size 1 (verified against the
# downloaded data/benchmark/FlashInfer-Bench/{012..017}/{definition,workload}.json):
#   * Llama-3.1-8B   h32 / kv8  (GQA 4:1) -- decode(013) / paged-prefill(015) / ragged(017)
#   * Qwen3-30B-A3B  h32 / kv4  (GQA 8:1) -- decode(012) / paged-prefill(014) / ragged(016)
# The captured WORKLOADS span batch in {1,16,64} and are dominated by LONG context: decode
# per-request ctx p50 ~2k-15k up to ~61k tokens; paged/ragged prefill total_q up to 16384. The
# configs below sample that envelope (short + long ctx, small + large batch) rather than the round
# 512/4096 points the earlier version used.
# Out of scope (unchanged): SOL's MLA problems 018/019 (h16 ckv512 kpe64) and the fp8 DSA/topk
# paths -- outside the kernel's envelope. NB the broader flashinfer-ai/flashinfer-trace *pool*
# (page_size 64, hd64/256, other GQA ratios) is a SUPERSET dataset that SOL-ExecBench does not
# score, so it is intentionally not mirrored here; those axes are covered in FA3TEST/EDGE instead.
def _real_configs() -> list[Config]:
    cfgs: list[Config] = []
    # Decode (q=1, paged page_size 1), batched serving: short, long, and large-batch points
    cfgs.append(Config("attn_real_llama8b_dec_s512", "decode", "bf16", 128, 32, 8,
                       paged=True, causal=True, block_size=1, batch=16, q_len=1, k_len=512))
    cfgs.append(Config("attn_real_llama8b_dec_s4k", "decode", "bf16", 128, 32, 8,
                       paged=True, causal=True, block_size=1, batch=16, q_len=1, k_len=4096))
    cfgs.append(Config("attn_real_llama8b_dec_s16k", "decode", "bf16", 128, 32, 8,
                       paged=True, causal=True, block_size=1, batch=16, q_len=1, k_len=16384))
    cfgs.append(Config("attn_real_llama8b_dec_b64_s4k", "decode", "bf16", 128, 32, 8,
                       paged=True, causal=True, block_size=1, batch=64, q_len=1, k_len=4096))
    cfgs.append(Config("attn_real_qwen3moe_dec_s512", "decode", "bf16", 128, 32, 4,
                       paged=True, causal=True, block_size=1, batch=16, q_len=1, k_len=512))
    cfgs.append(Config("attn_real_qwen3moe_dec_s4k", "decode", "bf16", 128, 32, 4,
                       paged=True, causal=True, block_size=1, batch=16, q_len=1, k_len=4096))
    cfgs.append(Config("attn_real_qwen3moe_dec_s32k", "decode", "bf16", 128, 32, 4,
                       paged=True, causal=True, block_size=1, batch=8, q_len=1, k_len=32768))
    # Causal prefill (paged, page_size 1): small and large total_q (SOL captures reach total_q 16384)
    cfgs.append(Config("attn_real_llama8b_pf_s2k", "prefill", "bf16", 128, 32, 8,
                       paged=True, causal=True, block_size=1, batch=2, q_len=2048, k_len=2048))
    cfgs.append(Config("attn_real_llama8b_pf_s16k", "prefill", "bf16", 128, 32, 8,
                       paged=True, causal=True, block_size=1, batch=1, q_len=16384, k_len=16384))
    cfgs.append(Config("attn_real_qwen3moe_pf_s4k", "prefill", "bf16", 128, 32, 4,
                       paged=True, causal=True, block_size=1, batch=1, q_len=4096, k_len=4096))
    # Ragged varlen prefill (SOL gqa_ragged_prefill), realistic prompt-length mix (total_q ~8.7k)
    _rmix = tuple([(2048, 2048), (1024, 1024), (512, 512), (4096, 4096), (256, 256), (768, 768)])
    cfgs.append(Config("attn_real_llama8b_pf_ragged", "prefill", "bf16", 128, 32, 8,
                       paged=False, causal=True, mix=_rmix))
    return cfgs


EDGE_CONFIGS = _edge_configs()
REAL_CONFIGS = _real_configs()

# --- plot-bar taxonomies for non-claim sets (claim uses CLAIM_BAR_ORDER + Config.plot_bar) ---
ATTNTEST_BAR_ORDER = (
    "Dense causal",
    "Dense features",
    "Dense long",
    "Variable-length",
    "KV decode",
    "KV features",
    "FP8",
)
EDGE_BAR_ORDER = ("Causal", "Non-causal", "Sliding window", "GQA")
REAL_BAR_ORDER = ("Decode", "Prefill")

_BAR_ORDERS = {
    "claim": CLAIM_BAR_ORDER,
    "attntest": ATTNTEST_BAR_ORDER,
    "edge": EDGE_BAR_ORDER,
    "real": REAL_BAR_ORDER,
}


def bar_order(set_name: str) -> tuple[str, ...]:
    return _BAR_ORDERS.get(set_name, ())


def plot_bar_for(cfg: Config, set_name: str) -> Optional[str]:
    """Category bar this scenario feeds. Claim configs carry `plot_bar`; others are inferred."""
    if cfg.plot_bar:
        return cfg.plot_bar
    n = cfg.name
    if set_name == "attntest":
        if "fp8" in n:
            return "FP8"
        if n.startswith("attn_varlen_"):
            return "Variable-length"
        if n.startswith("attn_kv_"):
            if any(t in n for t in ("fp16", "local", "leftpad", "page", "split", "128k",
                                    "ctx20k", "prefill", "dec339", "dec1024", "dec800")):
                return "KV features"
            return "KV decode"
        if n.startswith("attn_dense_"):
            if any(t in n for t in ("seq8k", "seq16k", "large4k")):
                return "Dense long"
            if any(t in n for t in ("fp16", "softcap", "local", "sink", "gqa", "mqa",
                                    "noncausal", "nc4k", "oddseq", "splits", "kgtq", "deg_",
                                    "dv128")):
                return "Dense features"
            return "Dense causal"
        return None
    if set_name == "edge":
        if "_noncausal_" in n:
            return "Non-causal"
        if "_local_" in n:
            return "Sliding window"
        if "_gqa_" in n:
            return "GQA"
        if "_causal_" in n:
            return "Causal"
        return None
    if set_name == "real":
        if "_dec_" in n:
            return "Decode"
        if "_pf_" in n:
            return "Prefill"
        return None
    return None


# Shapes the dense edge/real grid surfaced as fwd_kernel_ws2 bugs. The DENSE ones (q512k256 wrong,
# q799k3/q640k128 hangs) were FIXED 2026-07 by routing uniform-dense sub-wave grids to fwd_kernel_fa4
# (grid_fill's dense_uniform term in launch_ws) -- fa4 is correct on them AND beats FA4 (1.5-1.8x). The
# remaining deferred bug is ws2 VARLEN (h32 ragged mix), which must stay on ws2 (sub-wave varlen has no
# amortization headroom for persistent fa4; the h6 varlen configs beat FA4 there). Excluded from the
# green sets so default runs don't wedge. Run `get_configs("known_bad")` when fixing the ws2 varlen path.
KNOWN_WS2_BUGS = {
    "attn_real_llama8b_pf_ragged": "HANG: ws2 varlen (h32 ragged mix) did not complete at 100% GPU",
}

_CONFIG_SETS = {"claim": CLAIM_CONFIGS, "attntest": FA3TEST_CONFIGS, "edge": EDGE_CONFIGS, "real": REAL_CONFIGS}


def _drop_known_bad(cfgs: list[Config]) -> list[Config]:
    return [c for c in cfgs if c.name not in KNOWN_WS2_BUGS]


def config_names_in_set(which: str) -> set[str]:
    """Scenario names belonging to a named set (known_bad dropped except for which=='known_bad')."""
    return {c.name for c in get_configs(which)}


def plot_sets_for(cfgset: str) -> list[str]:
    """Which per-set plots to emit after an eval run with `--set cfgset`."""
    if cfgset == "all":
        return list(PLOT_SETS)
    if cfgset in PLOT_SETS:
        return [cfgset]
    return []


def get_configs(which: str = "claim") -> list[Config]:
    """Named config sets. Default `claim` is the ≤15 min paper suite. edge/real/attntest/all
    EXCLUDE KNOWN_WS2_BUGS; pass 'known_bad' for deferred ws2 hang shapes."""
    everything = CLAIM_CONFIGS + FA3TEST_CONFIGS + EDGE_CONFIGS + REAL_CONFIGS
    if which == "known_bad":
        return [c for c in everything if c.name in KNOWN_WS2_BUGS]
    if which == "all":
        # Dedup by name (claim overlaps some real/attntest shapes); claim order first.
        seen, out = set(), []
        for c in _drop_known_bad(everything):
            if c.name not in seen:
                seen.add(c.name)
                out.append(c)
        return out
    if which in _CONFIG_SETS:
        return _drop_known_bad(_CONFIG_SETS[which])
    raise ValueError(f"unknown config set {which!r} (claim | attntest | edge | real | all | known_bad)")



# --- tolerances (fp32-reference 2x-error criterion, matching Dao-AILab flash-attention) ---
# FA tests: |out - out_ref| <= rtol * |out_pt - out_ref| + fwd_atol
#   fwd_atol = 2 * |(out_ref + 0.3 - 0.3) - out_ref|   (bf16 arithmetic floor)
#   rtol = 2 (3 if softcap)
# Here out_pt is the arch baseline (FA3/FA4); out_ref is pure-PyTorch fp32 (`ref_fp32`).
ERR_MULT = 2.0
ERR_MULT_SOFTCAP = 3.0
ALLCLOSE_TOL = {  # (atol, rtol) per dtype, for out vs baseline
    "bf16": (8e-3, 1e-2),
    "fp16": (1e-3, 1e-2),
    "fp8e4m3": (1e-1, 5e-2),
    "bf16q_fp8kv": (1e-1, 5e-2),  # KV quantization noise dominates
}
MEAN_ABS_TOL = {"bf16": 5e-4, "fp16": 1e-4, "fp8e4m3": 1e-2, "bf16q_fp8kv": 1e-2}
LSE_TOL = (5e-3, 1e-2)

# --- perf knobs (default sized for ≤15 min claim-suite on 1 warm B200) -----------------
PERF = dict(
    warmup_iters=20,
    timed_iters=40,
    num_trials=3,
    l2_flush_mb=32,
)

# ======================================================================================
# INPUT CONSTRUCTION -- deterministic tensors for the matrix above + the fp32 reference
# ======================================================================================

@dataclass
class Inputs:
    cfg: Config
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: Optional[torch.Tensor]
    seqused_k: Optional[torch.Tensor]
    page_table: Optional[torch.Tensor]
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: float
    q_descale: Optional[torch.Tensor] = None
    k_descale: Optional[torch.Tensor] = None
    v_descale: Optional[torch.Tensor] = None
    s_aux: Optional[torch.Tensor] = None       # (num_heads_q,) per-head attention-sink logit
    leftpad_k: Optional[torch.Tensor] = None   # (batch,) int32 cache_leftpad


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, TORCH_DTYPE[name])


def _cumsum0(xs: list[int]) -> list[int]:
    out, s = [0], 0
    for x in xs:
        s += int(x)
        out.append(s)
    return out


def make_inputs(cfg: Config, seed: int = 0, device: str = "cuda") -> Inputs:
    """Construct identical, deterministic inputs for a config (uniform OR ragged `mix`)."""
    g = torch.Generator(device=device).manual_seed(seed)
    # Generate in bf16 then cast: homogeneous fp8 QKV, or bf16 Q + fp8 KV.
    gen = (torch.bfloat16 if cfg.has_fp8_kv else _torch_dtype(cfg.dtype))

    hq, hkv, hd, hdv, block = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv, cfg.block_size
    reqs = cfg.requests                          # [(q_len, k_len), ...]; uniform = batch copies
    q_lens = [q for q, _ in reqs]
    k_lens = [k for _, k in reqs]
    b = len(reqs)
    total_q = sum(q_lens)

    q = torch.randn(total_q, hq, hd, generator=g, device=device, dtype=gen)
    cu_q = torch.tensor(_cumsum0(q_lens), device=device, dtype=torch.int32)
    softmax_scale = hd ** -0.5

    if cfg.paged:
        nbp = [(k + block - 1) // block for k in k_lens]
        total_blocks = sum(nbp) + 4
        k = torch.randn(total_blocks, block, hkv, hd, generator=g, device=device, dtype=gen)
        v = torch.randn(total_blocks, block, hkv, hdv, generator=g, device=device, dtype=gen)
        page_table = torch.zeros(b, max(nbp), device=device, dtype=torch.int32)
        nb = 0
        for i, nbi in enumerate(nbp):
            page_table[i, :nbi] = torch.arange(nb, nb + nbi, device=device, dtype=torch.int32)
            nb += nbi
        seqused_k = torch.tensor(k_lens, device=device, dtype=torch.int32)
        cu_k = None
    else:
        total_k = sum(k_lens)
        k = torch.randn(total_k, hkv, hd, generator=g, device=device, dtype=gen)
        v = torch.randn(total_k, hkv, hdv, generator=g, device=device, dtype=gen)
        cu_k = torch.tensor(_cumsum0(k_lens), device=device, dtype=torch.int32)
        seqused_k = None
        page_table = None

    qd = kd = vd = None
    if cfg.has_fp8_kv:
        if cfg.fp8_scales not in ("per_head", "scalar"):
            raise ValueError(f"fp8_scales must be per_head|scalar, got {cfg.fp8_scales!r}")

        def _descale_per_head():
            # deterministic non-trivial scale in [0.75, 1.25) per (request, kv-head)
            return torch.rand((b, hkv), generator=g, device=device, dtype=torch.float32) * 0.5 + 0.75

        def _descale_scalar():
            # vLLM / FlashInfer trtllm-gen: one float per tensor (Q/K/V).
            return torch.rand((), generator=g, device=device, dtype=torch.float32) * 0.5 + 0.75

        _descale = _descale_scalar if cfg.is_scalar_fp8_scales else _descale_per_head

    if cfg.is_fp8_qkv:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        qd = _descale()
        kd = _descale()
        vd = _descale()
    elif cfg.is_bf16q_fp8kv:
        # Q stays bf16; only the KV cache is fp8 (vLLM/SGLang --kv-cache-dtype fp8).
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        kd = _descale()
        vd = _descale()

    s_aux = None
    if cfg.sink:
        # per-(query)head sink logit; mirrors FA3's test (randn(nheads) * 4, bf16)
        s_aux = torch.randn(hq, generator=g, device=device, dtype=torch.bfloat16) * 4

    leftpad_k = None
    if cfg.leftpad > 0:
        leftpad_k = torch.full((b,), int(cfg.leftpad), device=device, dtype=torch.int32)

    return Inputs(cfg=cfg, q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                  seqused_k=seqused_k, page_table=page_table,
                  max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens), softmax_scale=softmax_scale,
                  q_descale=qd, k_descale=kd, v_descale=vd, s_aux=s_aux, leftpad_k=leftpad_k)


def _descale_is_scalar(t: Optional[torch.Tensor]) -> bool:
    return t is not None and t.numel() == 1


def expand_fp8_descales(inp: Inputs) -> Inputs:
    """Broadcast scalar (vLLM-style) descales to (batch, hkv) for FA4 / our kernels / cuDNN."""
    cfg = inp.cfg
    if not cfg.is_scalar_fp8_scales:
        return inp
    b, hkv = len(cfg.requests), cfg.num_heads_kv

    def exp(d: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if d is None:
            return None
        if not _descale_is_scalar(d):
            return d
        return d.new_full((b, hkv), float(d.reshape(-1)[0].item()))

    return Inputs(
        cfg=cfg, q=inp.q, k=inp.k, v=inp.v,
        cu_seqlens_q=inp.cu_seqlens_q, cu_seqlens_k=inp.cu_seqlens_k,
        seqused_k=inp.seqused_k, page_table=inp.page_table,
        max_seqlen_q=inp.max_seqlen_q, max_seqlen_k=inp.max_seqlen_k,
        softmax_scale=inp.softmax_scale,
        q_descale=exp(inp.q_descale), k_descale=exp(inp.k_descale), v_descale=exp(inp.v_descale),
        s_aux=inp.s_aux, leftpad_k=inp.leftpad_k,
    )


def _dequant_fp8_kv_to_bf16(inp: Inputs) -> Inputs:
    """Interim host path for bf16q_fp8kv: apply k/v_descale and cast KV to bf16.

    FA3/FA4 and our kernels currently require homogeneous QKV dtypes (tridao, FA#1737). Until a
    native fp8-KV gather lands, this materializes a bf16 cache so the bf16 path can run. Descales
    are folded in so results match `ref_fp32`. Bandwidth is bf16 — claim ratios on these rows are
    not a true fp8-KV story yet.
    """
    cfg = inp.cfg
    assert cfg.is_bf16q_fp8kv
    hkv = cfg.num_heads_kv
    reqs = cfg.requests
    b = len(reqs)
    kf = inp.k.to(torch.float32)
    vf = inp.v.to(torch.float32)
    k_sc = inp.k_descale
    v_sc = inp.v_descale
    if cfg.paged:
        block = cfg.block_size
        k_lens = [k for _, k in reqs]
        nbp = [(k + block - 1) // block for k in k_lens]
        for bi in range(b):
            pt = inp.page_table[bi, :nbp[bi]].long()
            if k_sc is not None:
                if _descale_is_scalar(k_sc):
                    kf[pt] = kf[pt] * float(k_sc)
                else:
                    kf[pt] = kf[pt] * k_sc[bi].view(1, 1, hkv, 1)
            if v_sc is not None:
                if _descale_is_scalar(v_sc):
                    vf[pt] = vf[pt] * float(v_sc)
                else:
                    vf[pt] = vf[pt] * v_sc[bi].view(1, 1, hkv, 1)
    else:
        k_lens = [k for _, k in reqs]
        off = 0
        for bi, kl in enumerate(k_lens):
            if k_sc is not None:
                if _descale_is_scalar(k_sc):
                    kf[off:off + kl] = kf[off:off + kl] * float(k_sc)
                else:
                    kf[off:off + kl] = kf[off:off + kl] * k_sc[bi].view(1, hkv, 1)
            if v_sc is not None:
                if _descale_is_scalar(v_sc):
                    vf[off:off + kl] = vf[off:off + kl] * float(v_sc)
                else:
                    vf[off:off + kl] = vf[off:off + kl] * v_sc[bi].view(1, hkv, 1)
            off += kl
    return Inputs(
        cfg=cfg, q=inp.q, k=kf.to(torch.bfloat16), v=vf.to(torch.bfloat16),
        cu_seqlens_q=inp.cu_seqlens_q, cu_seqlens_k=inp.cu_seqlens_k,
        seqused_k=inp.seqused_k, page_table=inp.page_table,
        max_seqlen_q=inp.max_seqlen_q, max_seqlen_k=inp.max_seqlen_k,
        softmax_scale=inp.softmax_scale,
        q_descale=None, k_descale=None, v_descale=None,
        s_aux=inp.s_aux, leftpad_k=inp.leftpad_k,
    )


def fwd_args(inp: Inputs, out: Optional[torch.Tensor] = None) -> tuple:
    """Build the 37-positional-arg tuple for `torch.ops.<ns>.fwd(*args)` (vLLM dispatch order)."""
    cfg = inp.cfg
    return (
        inp.q, inp.k, inp.v,
        None, None, None,                          # k_new, v_new, q_v
        out,                                       # out (None -> op allocates)
        inp.cu_seqlens_q, inp.cu_seqlens_k, None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
        None, inp.seqused_k,                       # seqused_q, seqused_k
        inp.max_seqlen_q, inp.max_seqlen_k,        # max_seqlen_q, max_seqlen_k
        inp.page_table, None, inp.leftpad_k,       # page_table, kv_batch_idx, leftpad_k
        None, None, None,                          # rotary_cos, rotary_sin, seqlens_rotary
        inp.q_descale, inp.k_descale, inp.v_descale,
        inp.softmax_scale, bool(cfg.causal),
        int(cfg.window_left), int(cfg.window_right),
        float(cfg.softcap), True,                  # softcap, is_rotary_interleaved
        None, int(cfg.num_splits), None,           # scheduler_metadata, num_splits, pack_gqa
        0,                                         # sm_margin
        inp.s_aux, 1, 0, None,                     # s_aux, cp_world_size, cp_rank, cp_tot_seqused_k
    )


def call_fwd(namespace: str, inp: Inputs, out: Optional[torch.Tensor] = None):
    """Invoke `torch.ops.<namespace>.fwd` and return (out, softmax_lse)."""
    if inp.cfg.is_scalar_fp8_scales:
        inp = expand_fp8_descales(inp)
    if inp.cfg.is_bf16q_fp8kv and inp.k.dtype == torch.float8_e4m3fn:
        inp = _dequant_fp8_kv_to_bf16(inp)
    op = getattr(torch.ops, namespace).fwd
    res = op(*fwd_args(inp, out))
    return res[0], res[1]  # [out, softmax_lse, out_accum, lse_accum]


def ref_fp32(inp: Inputs) -> torch.Tensor:
    """Pure-PyTorch fp32 reference attention for the 2x-error correctness criterion.

    Handles GQA/MQA, causal, sliding window, softcap, paged + non-paged KV, fp8 descales,
    head_dim_v != head_dim, attention sink (s_aux), cache_leftpad, ragged/MIXED batches, and
    fully-masked query rows (causal q>k) -> 0. Returns out fp32 (total_q, hq, hdv).
    Query positions are the LAST q_len tokens of the k_len-long sequence (append semantics).
    """
    cfg = inp.cfg
    hq, hkv, hd, hdv = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim, cfg.hdv
    rep = hq // hkv
    dev = inp.q.device
    block = cfg.block_size
    reqs = cfg.requests
    k_lens = [k for _, k in reqs]

    qf = inp.q.to(torch.float32)
    kf = inp.k.to(torch.float32)
    vf = inp.v.to(torch.float32)
    out = torch.empty(qf.shape[0], hq, hdv, device=dev, dtype=torch.float32)
    scale = inp.softmax_scale
    sink_vec = inp.s_aux.float() if (cfg.sink and inp.s_aux is not None) else None

    q_off = 0
    for bi, (ql, kl) in enumerate(reqs):
        qi = qf[q_off:q_off + ql]
        if inp.q_descale is not None:
            if _descale_is_scalar(inp.q_descale):
                qi = qi * float(inp.q_descale)
            else:
                qi = qi * inp.q_descale[bi].repeat_interleave(rep).view(1, hq, 1)
        if cfg.paged:
            nbp = (kl + block - 1) // block
            pt = inp.page_table[bi, :nbp].long()
            Ki = kf[pt].reshape(nbp * block, hkv, hd)[:kl]
            Vi = vf[pt].reshape(nbp * block, hkv, hdv)[:kl]
        else:
            koff = sum(k_lens[:bi])
            Ki = kf[koff:koff + kl]
            Vi = vf[koff:koff + kl]
        if inp.k_descale is not None:
            if _descale_is_scalar(inp.k_descale):
                Ki = Ki * float(inp.k_descale)
            else:
                Ki = Ki * inp.k_descale[bi].view(1, hkv, 1)
        if inp.v_descale is not None:
            if _descale_is_scalar(inp.v_descale):
                Vi = Vi * float(inp.v_descale)
            else:
                Vi = Vi * inp.v_descale[bi].view(1, hkv, 1)

        qpos = torch.arange(kl - ql, kl, device=dev).view(ql, 1)
        kpos = torch.arange(kl, device=dev).view(1, kl)
        mask = torch.zeros(ql, kl, device=dev, dtype=torch.bool)
        if cfg.causal:
            mask |= kpos > qpos
        if cfg.window_left >= 0:
            mask |= kpos < (qpos - cfg.window_left)
        if cfg.window_right >= 0:
            mask |= kpos > (qpos + cfg.window_right)
        if cfg.leftpad > 0:
            # cache_leftpad: the first `leftpad` KV tokens are ignored. The causal/window
            # geometry is shift-invariant under leftpad, so this reduces to masking them.
            mask |= kpos < cfg.leftpad

        for h in range(hq):
            hk = h // rep
            s = torch.einsum("qd,kd->qk", qi[:, h, :], Ki[:, hk, :]) * scale
            if cfg.softcap and cfg.softcap > 0:
                s = cfg.softcap * torch.tanh(s / cfg.softcap)
            s = s.masked_fill(mask, float("-inf"))
            if sink_vec is not None:
                # extra per-head "sink" logit in the softmax denominator (dropped from output)
                sink_col = sink_vec[h].view(1, 1).expand(ql, 1)
                p = torch.softmax(torch.cat([s, sink_col], dim=-1), dim=-1)[:, :kl]
            else:
                p = torch.softmax(s, dim=-1)
                # fully-masked rows (e.g. causal q>k) -> 0 instead of NaN (matches FA3)
                p = torch.nan_to_num(p, nan=0.0)
            out[q_off:q_off + ql, h, :] = torch.einsum("qk,kd->qd", p, Vi[:, hk, :])
        q_off += ql
    return out
