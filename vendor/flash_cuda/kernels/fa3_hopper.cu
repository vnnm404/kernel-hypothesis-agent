// fa3-cuda: a self-contained FlashAttention-3 forward op for Hopper (sm_90a), pure CUDA + PTX,
// NO external libraries (no CUTLASS/CuTe/cuBLAS/cuDNN).  torch appears ONLY in the host mha_fwd +
// TORCH_LIBRARY registration (the allowed boundary).  Registers the exact vLLM `_vllm_fa3_C` op
// schema (fwd + get_scheduler_metadata) so it can A/B against, and ultimately replace, FA3.
//
// ARCHITECTURE: a general DISPATCHER.  A correct, fully-general scalar fallback (`attn_kernel`,
// one warp per (query,head), online softmax, fp32 accumulate) guarantees correct output for EVERY
// config -- bf16/fp16/fp8-e4m3, paged + non-paged KV, GQA/MQA, causal, sliding window, softcap,
// varlen, head_dim_v != head_dim.  Fast TMA+wgmma warp-specialized paths are wired in incrementally
// (prefill first, then decode) and selected per-config; anything not yet specialized falls back to
// the correct path, so correctness holds throughout the optimization.  The scalar fallback below
// began as the fa3agent bootstrap reference; the fast paths are the fa3-cuda contribution.
//
// Build with -DFA3_LIB_NAME=<namespace> to pick the torch op namespace
// (default _vllm_fa3_C for the real vLLM swap; _distill_fa3_C for in-process A/B).

#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda.h>            // driver API: CUtensorMap + cuTensorMapEncodeTiled (TMA), libcuda (-lcuda)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <vector>
#include <cstdint>
#include <cstdlib>          // getenv/atoi (perf-lever env toggles)
#include <limits>
#include <algorithm>
#include <type_traits>

#ifndef FA3_LIB_NAME
#define FA3_LIB_NAME _vllm_fa3_C
#endif

#define MAX_HD 256

// ---- scalar load/convert helpers -----------------------------------------------------
template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 x) { return float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Params: all strides are IN ELEMENTS.
template <typename TIn, typename TOut>
struct Params {
  const TIn* q; const TIn* k; const TIn* v;
  TOut* o; float* lse;
  const int* cu_q;          // (b+1) or null
  const int* cu_k;          // (b+1) or null (non-paged)
  const int* seqused_k;     // (b) or null (paged)
  const int* page_table;    // (b, max_blocks) or null
  const float* q_descale; const float* k_descale; const float* v_descale;  // (b, hkv) or null
  int ds_b, ds_h;           // descale strides IN ELEMENTS (broadcast-safe; 0 for stride-0 tensors)
  long q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head;
  int b, hq, hkv, hd, hdv, total_q, block_size, page_stride;
  float scale, softcap;
  int win_l, win_r; bool causal;
};

template <typename TIn, typename TOut>
__global__ void attn_kernel(Params<TIn, TOut> p) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const long total_pairs = (long)p.total_q * p.hq;
  if (warp >= total_pairs) return;
  const int q_idx = warp / p.hq;
  const int head  = warp % p.hq;
  const int hk    = head / (p.hq / p.hkv);

  // locate batch b s.t. cu_q[b] <= q_idx < cu_q[b+1]
  int b = 0;
  while (b + 1 <= p.b && p.cu_q[b + 1] <= q_idx) ++b;
  const int q_in_seq = q_idx - p.cu_q[b];
  const int sq = p.cu_q[b + 1] - p.cu_q[b];
  const int klen = p.seqused_k ? p.seqused_k[b] : (p.cu_k[b + 1] - p.cu_k[b]);
  const int q_abs = klen - sq + q_in_seq;   // causal alignment: last query <-> last key

  // Descale is stride-aware (vLLM passes a BROADCAST (b,hkv) tensor with stride [0,0] and one
  // physical element -- dense indexing would read out of bounds). Only non-null for fp8.
  const long ds_idx = (long)b * p.ds_b + (long)hk * p.ds_h;
  float qd = p.q_descale ? p.q_descale[ds_idx] : 1.f;
  float kd = p.k_descale ? p.k_descale[ds_idx] : 1.f;
  float vd = p.v_descale ? p.v_descale[ds_idx] : 1.f;
  const float qk_scale = p.scale * qd * kd;

  // load q row into per-lane registers
  const int per = (p.hd + 31) / 32;
  float qreg[(MAX_HD + 31) / 32];
  const TIn* qp = p.q + (long)q_idx * p.q_row + (long)head * p.q_head;
  #pragma unroll
  for (int e = 0; e < per; ++e) { int d = lane + e * 32; qreg[e] = d < p.hd ? to_float<TIn>(qp[d]) : 0.f; }

  const int perv = (p.hdv + 31) / 32;
  float acc[(MAX_HD + 31) / 32];
  #pragma unroll
  for (int e = 0; e < perv; ++e) acc[e] = 0.f;
  float m = -INFINITY, l = 0.f;

  for (int j = 0; j < klen; ++j) {
    bool masked = false;
    if (p.causal && j > q_abs) masked = true;
    if (p.win_l >= 0 && j < q_abs - p.win_l) masked = true;
    if (p.win_r >= 0 && j > q_abs + p.win_r) masked = true;
    if (masked) continue;

    long krow;
    if (p.page_table) {
      int blk = p.page_table[(long)b * p.page_stride + j / p.block_size];
      krow = ((long)blk * p.block_size + (j % p.block_size));
    } else {
      krow = (long)p.cu_k[b] + j;
    }
    const TIn* kp = p.k + krow * p.k_row + (long)hk * p.k_head;
    float dot = 0.f;
    #pragma unroll
    for (int e = 0; e < per; ++e) { int d = lane + e * 32; if (d < p.hd) dot += qreg[e] * to_float<TIn>(kp[d]); }
    float s = warp_reduce_sum(dot) * qk_scale;
    if (p.softcap > 0.f) s = p.softcap * tanhf(s / p.softcap);

    float m_new = fmaxf(m, s);
    float corr = __expf(m - m_new);
    float pj = __expf(s - m_new);
    l = l * corr + pj;
    const TIn* vp = p.v + krow * p.v_row + (long)hk * p.v_head;
    #pragma unroll
    for (int e = 0; e < perv; ++e) { int d = lane + e * 32; float vv = d < p.hdv ? to_float<TIn>(vp[d]) * vd : 0.f; acc[e] = acc[e] * corr + pj * vv; }
    m = m_new;
  }

  TOut* op = p.o + (long)q_idx * p.o_row + (long)head * p.o_head;
  float inv = l > 0.f ? 1.f / l : 0.f;
  #pragma unroll
  for (int e = 0; e < perv; ++e) { int d = lane + e * 32; if (d < p.hdv) op[d] = from_float<TOut>(acc[e] * inv); }
  if (lane == 0) p.lse[(long)head * p.total_q + q_idx] = (l > 0.f) ? (m + logf(l)) : (-INFINITY);
}

// ======================================================================================
// FAST PATH 1: prefill, head_dim=128, bf16 (paged+non-paged, pack-GQA, varlen, causal/non-causal)
//
// Reuses the validated hd128 wgmma+softmax consumer mainloop from fused_attn.cu (1 producer + 2
// consumer warpgroups, 128x128 tile, 3-stage K/V ring, intra-WG QK||PV overlap), but replaces the
// whole-tensor TMA load path with a cp.async GATHER that swizzle-stores packed-Q and paged/contig
// K/V into the SAME 128B-swizzle smem layout the descriptors expect (tma_swz_test{,_v}.cu proved
// this layout reproduces TMA byte-for-byte). Epilogue scatters O/LSE via the pack-GQA divmod.
// ======================================================================================
namespace pf {
// PF_CPASYNC=1: pipelined cp.async gather loaders (fast path). =0: synchronous uint4 stores (debug).
#ifndef PF_CPASYNC
#define PF_CPASYNC 1
#endif
constexpr int CTA_M = 128, CTA_N = 128, HD = 128, WG = 128;
constexpr int NThreads = 3 * WG, NumCons = 2 * WG, NumProd = WG, WS_STAGES = 3;
constexpr int KVSTRIDE = CTA_N * HD;            // one swizzled K/V tile (elems)

// ---- 128B swizzle (validated) ----
// 128B-swizzle of a [.,HD] K-major bf16 tile, byte-exact with cp.async.bulk.tensor (TMA): the column's
// bits 3-5 are XORed by the row phase (row&7). Verified against the TMA-landed layout (tma_probe.cu).
__device__ __forceinline__ int koff(int row, int c) { return (c >> 6) * 8192 + row * 64 + ((c & 63) ^ ((row & 7) << 3)); } // K-major (Q,K)
__device__ __forceinline__ int voff(int key, int dv) { return (dv >> 6) * 8192 + key * 64 + ((dv & 63) ^ ((key & 7) << 3)); } // MN-major (V)
// Row-count-generic swizzle: the [64-col] block stride is ROWS*64 (the # rows in the tile: CTA_M for
// Q, CN for K/V). ROWS=128 reproduces koff/voff exactly (8192 = 128*64).
template <int ROWS> __device__ __forceinline__ int koffR(int row, int c) { return (c >> 6) * (ROWS * 64) + row * 64 + ((c & 63) ^ ((row & 7) << 3)); }
template <int ROWS> __device__ __forceinline__ int voffR(int key, int dv) { return (dv >> 6) * (ROWS * 64) + key * 64 + ((dv & 63) ^ ((key & 7) << 3)); }
// Descriptor col-block step (16-byte units) for a ROWS-row tile: within a 64-col block step 16 cols
// (=2 units); crossing to the next 64-col block jumps ROWS*64 elems = ROWS*8 units. ROWS=128 -> 1024.
template <int ROWS> __device__ __forceinline__ uint64_t qk_doffR(int kb) { return (uint64_t)((kb & 3) * 2 + (kb >> 2) * (ROWS * 8)); }

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t a, uint32_t lbo, uint32_t sbo, uint32_t lt) {
    uint64_t d = 0;
    d |= (uint64_t)((a >> 4) & 0x3FFFu);
    d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
    d |= (uint64_t)(lt & 0x3u) << 62;
    return d;
}
__device__ __forceinline__ uint64_t desc_add(uint64_t base, uint32_t off) {
    uint32_t lo, hi;
    asm("mov.b64 {%0,%1}, %2;\n" : "=r"(lo), "=r"(hi) : "l"(base));
    lo += off;
    uint64_t r; asm("mov.b64 %0, {%1,%2};\n" : "=l"(r) : "r"(lo), "r"(hi));
    return r;
}
__device__ __forceinline__ uint64_t v_doff(int kb)  { return (uint64_t)(kb * 128); }
constexpr uint32_t SW_QK_LBO = 16, SW_QK_SBO = 1024;
constexpr uint32_t SW_V_SBO = 1024;

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat16 a = __float2bfloat16_rn(lo), b = __float2bfloat16_rn(hi);
    uint16_t ua, ub; memcpy(&ua, &a, 2); memcpy(&ub, &b, 2);
    return uint32_t(ua) | (uint32_t(ub) << 16);
}
// Pack two fp32 into the wgmma A/B register pair for the QKV element type T (bf16 or fp16). Both
// are 16-bit so the smem/swizzle layout is byte-identical; only the wgmma type token and pack differ.
template <typename T> __device__ __forceinline__ uint32_t pack_2x(float lo, float hi);
template <> __device__ __forceinline__ uint32_t pack_2x<__nv_bfloat16>(float lo, float hi) { return pack_bf16(lo, hi); }
template <> __device__ __forceinline__ uint32_t pack_2x<__half>(float lo, float hi) {
    __half a = __float2half_rn(lo), b = __float2half_rn(hi);
    uint16_t ua, ub; memcpy(&ua, &a, 2); memcpy(&ub, &b, 2);
    return uint32_t(ua) | (uint32_t(ub) << 16);
}
template <typename T> __device__ __forceinline__ constexpr bool is_f16() { return false; }
template <> __device__ __forceinline__ constexpr bool is_f16<__half>() { return true; }
__device__ __forceinline__ float ex2_approx(float x) {
    float r; asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }

template <int CN = CTA_N>
__device__ __forceinline__ float softmax_stats(float (&s)[CN/2], float* m_r, float* l_r, int r, int nb, float s2) {
    float b0 = -INFINITY, b1 = -INFINITY, b2 = -INFINITY, b3 = -INFINITY;
#pragma unroll
    for (int c = 0; c < CN / 8; c += 2) {
        b0 = fmaxf(b0, s[c*4+r*2+0]);     b1 = fmaxf(b1, s[c*4+r*2+1]);
        b2 = fmaxf(b2, s[(c+1)*4+r*2+0]); b3 = fmaxf(b3, s[(c+1)*4+r*2+1]);
    }
    float bm = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
    bm = quad_max(bm);
    float scale_f;
    if (nb == 0) { m_r[r] = bm; scale_f = 1.f; }
    else { float prev = m_r[r]; m_r[r] = fmaxf(m_r[r], bm);
           float cur = (m_r[r] == -INFINITY) ? 0.f : m_r[r];
           scale_f = ex2_approx((prev - cur) * s2); l_r[r] *= scale_f; }
    float max_scaled = (m_r[r] == -INFINITY) ? 0.f : m_r[r] * s2;
    float ls0 = 0.f, ls1 = 0.f, ls2 = 0.f, ls3 = 0.f;
#pragma unroll
    for (int c = 0; c < CN / 8; c += 2) {
        float p0 = ex2_approx(s[c*4+r*2+0]     * s2 - max_scaled);
        float p1 = ex2_approx(s[c*4+r*2+1]     * s2 - max_scaled);
        float p2 = ex2_approx(s[(c+1)*4+r*2+0] * s2 - max_scaled);
        float p3 = ex2_approx(s[(c+1)*4+r*2+1] * s2 - max_scaled);
        s[c*4+r*2+0] = p0;     s[c*4+r*2+1] = p1;
        s[(c+1)*4+r*2+0] = p2; s[(c+1)*4+r*2+1] = p3;
        ls0 += p0; ls1 += p1; ls2 += p2; ls3 += p3;
    }
    float lsum = (ls0 + ls1) + (ls2 + ls3);
    l_r[r] = (nb == 0) ? lsum : l_r[r] + lsum;
    return scale_f;
}
// Combined causal + key-length mask. r=0->qlo (row_lo), r=1->qhi (row_hi). col indices: s[c*4+r*2+p].
// Cheap causal/bounds-only mask (hot path: no sliding window).
template <int CN = CTA_N>
__device__ __forceinline__ void pf_mask(float (&s)[CN/2], int col0, int qlo, int qhi, int seqlen_k, int causal, int tg) {
    // Fast-out for fully-visible tiles (the common case in causal prefill: only the diagonal tile of
    // each work item is partially masked, every interior tile is entirely below the diagonal and within
    // seqlen_k). kmax = col0+CN-1 is the largest key index this CTA's tile spans; if it is < seqlen_k and
    // <= the smaller query position (qlo), no element of either row is masked -> skip the per-element ALU
    // that otherwise sits on the QK->next-QK critical path (stealing tensor-pipe issue slots while PV is
    // in flight). Conservative (tile-global kmax) so the branch is uniform across the warpgroup.
    const int kmax = col0 + CN - 1;
    if (kmax < seqlen_k && (!causal || kmax <= qlo)) return;
#pragma unroll
    for (int c = 0; c < CN / 8; ++c) {
        int k0 = col0 + 8 * c + 2 * tg;
        if (k0     >= seqlen_k || (causal && k0     > qlo)) s[c*4+0] = -INFINITY;
        if (k0 + 1 >= seqlen_k || (causal && k0 + 1 > qlo)) s[c*4+1] = -INFINITY;
        if (k0     >= seqlen_k || (causal && k0     > qhi)) s[c*4+2] = -INFINITY;
        if (k0 + 1 >= seqlen_k || (causal && k0 + 1 > qhi)) s[c*4+3] = -INFINITY;
    }
}
// Windowed mask (sliding window): valid <=> key<seqlen_k && !(causal&&key>qpos) &&
//   !(win_l>=0&&key<qpos-win_l) && !(win_r>=0&&key>qpos+win_r). Collapsed into a per-row valid key
// range [lo, hi] (inclusive) computed ONCE, then 2 comparisons/element instead of 4 conditions/element.
// The mask sits on the critical path between the QK wgmma result and the next wgmma issue, so halving
// its ALU recovers tensor-pipe utilization that softcap/SWA otherwise lose (NCU: tensor 23%->~).
template <int CN = CTA_N>
__device__ __forceinline__ void pf_mask_win(float (&s)[CN/2], int col0, int qlo, int qhi, int seqlen_k,
                                            int causal, int win_l, int win_r, int tg, int lpad = 0) {
    constexpr int NEG = -0x40000000;                                  // -inf sentinel for "no left bound"
    int lo0 = (win_l >= 0) ? (qlo - win_l) : NEG;
    int lo1 = (win_l >= 0) ? (qhi - win_l) : NEG;
    // cache_leftpad: keys [0, lpad) are ignored -> absolute lower bound on the valid key range (folds in
    // alongside the relative sliding-window left bound; lpad==0 leaves lo0/lo1 untouched, zero overhead).
    if (lpad > 0) { if (lpad > lo0) lo0 = lpad; if (lpad > lo1) lo1 = lpad; }
    int hi0 = seqlen_k - 1, hi1 = seqlen_k - 1;                       // upper bound folds in seqlen_k
    if (causal)     { if (qlo < hi0) hi0 = qlo;        if (qhi < hi1) hi1 = qhi; }
    if (win_r >= 0) { int t = qlo + win_r; if (t < hi0) hi0 = t;  t = qhi + win_r; if (t < hi1) hi1 = t; }
    // Fast-out for fully-visible tiles (same lever as pf_mask): if every key index this tile spans lies
    // within the most-restrictive [lo,hi] of both rows, nothing is masked -> skip the per-element ALU on
    // the QK->next-QK critical path. lo = max(lo0,lo1) (tightest left), hi = min(hi0,hi1) (tightest right).
    const int kmax = col0 + CN - 1;
    const int lo = lo0 > lo1 ? lo0 : lo1, hi = hi0 < hi1 ? hi0 : hi1;
    if (col0 >= lo && kmax <= hi) return;
#pragma unroll
    for (int c = 0; c < CN / 8; ++c) {
        int k0 = col0 + 8 * c + 2 * tg;
        if (k0     < lo0 || k0     > hi0) s[c*4+0] = -INFINITY;
        if (k0 + 1 < lo0 || k0 + 1 > hi0) s[c*4+1] = -INFINITY;
        if (k0     < lo1 || k0     > hi1) s[c*4+2] = -INFINITY;
        if (k0 + 1 < lo1 || k0 + 1 > hi1) s[c*4+3] = -INFINITY;
    }
}
// Hardware tanh approximation (matches cutlass::fast_tanh / FA3's softcap path): a single SFU
// instruction vs the ~20-instruction libdevice tanhf -- softcap feeds exp2 so approx is ample.
__device__ __forceinline__ float fast_tanh(float x) {
    float r; asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
// Logit softcap, applied to the RAW Q.K^T before masking. softmax_stats later multiplies by `scale`,
// so store softcap*tanh(raw*scale/softcap)/scale to get an effective logit of softcap*tanh(raw*scale/softcap).
template <int CN = CTA_N>
__device__ __forceinline__ void pf_softcap(float (&s)[CN/2], float scale, float softcap) {
    const float a = scale / softcap;       // raw -> tanh argument
    const float b = softcap / scale;       // tanh -> stored value (softmax_stats re-applies *scale)
#pragma unroll
    for (int i = 0; i < CN / 2; ++i) s[i] = fast_tanh(s[i] * a) * b;
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, int phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar), ok;
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(phase));
    return ok != 0;
}
__device__ __forceinline__ void producer_regs() { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory"); }
// Consumer register budget = the kernel's static per-thread allocation (set by __launch_bounds__
// min-blocks). 1 CTA/SM (MINB=1) affords 232; 2 CTA/SM (MINB=2, decode small-HDV occupancy path)
// caps the static alloc at 65536/(2*256)=128, so the consumer grabs 128.
template <int R = 232>
__device__ __forceinline__ void consumer_regs() {
    if constexpr (R <= 128)      asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n" ::: "memory");
    else if constexpr (R <= 168) asm volatile("setmaxnreg.inc.sync.aligned.u32 168;\n" ::: "memory");
    else                         asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
}
// TMA: arrive on `bar` and register `bytes` of pending tensor traffic (phase flips when the
// cp.async.bulk completions have delivered all expected bytes). One arrive per stage-fill.
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes) : "memory");
}
// 2D tiled TMA load: copy box at global (x=col, y=row) into swizzled smem `dst`; completion is
// tracked by `bar` (mbarrier::complete_tx::bytes). Single-thread issue.
__device__ __forceinline__ void tma_box(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(dst), b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" :: "r"(s), "l"(map), "r"(x), "r"(y), "r"(b) : "memory");
}
// 3D variant: coords {x (innermost),y,z}. Used for packed-GQA Q TMA -- a [64,rep,CM/rep] box over
// Q[total_q,hq,hd] lands at koffR<CM>(tok*rep+head, d) byte-exactly (the box's row-major smem order
// (tok*rep+head)*64+d == koffR's pre-swizzle term, and the 128B swizzle is address-based).
__device__ __forceinline__ void tma_box3(void* dst, const CUtensorMap* map, int x, int y, int z, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(dst), b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];\n" :: "r"(s), "l"(map), "r"(x), "r"(y), "r"(z), "r"(b) : "memory");
}
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem) : "memory");
}
// 4-byte cp.async (page-table int -> smem). .ca (cache-all): the page table is small + reused, so keep
// it in L1. Used by the decoupled page-row prefetch ring so the page LDG is off the gather critical path.
__device__ __forceinline__ void cp_async_4(uint32_t smem_addr, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_addr), "l"(gmem) : "memory");
}
// Tie this thread's prior (untied) cp.async to `bar`: an arrive-on fires when those copies complete.
// `.noinc` => the barrier's init count must already account for these arrives (it does: NumProd).
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory");
}
// Producer->consumer publish of a filled stage. cp.async path: non-blocking, completion-tracked
// (lets the producer race STAGES ahead). sync-store path: plain release arrive.
#if PF_CPASYNC
#define PF_PUBLISH(bar) cp_async_mbar_arrive(bar)
#else
#define PF_PUBLISH(bar) mbar_arrive(bar)
#endif

#define PF_ACC64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
"%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47," \
"%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define PF_ACC24 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23}"
#define PF_ACC32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define PF_ACC40 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
"%32,%33,%34,%35,%36,%37,%38,%39}"
#define PF_ACC48 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
"%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
"%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47}"
#define PF_OUT24(a) \
        "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]), \
        "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]), \
        "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23])
#define PF_OUT40(a) \
        "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]), \
        "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]), \
        "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]), \
        "+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]), \
        "+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39])
#define PF_OUT48(a) \
        "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]), \
        "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]), \
        "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]), \
        "+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]), \
        "+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]), \
        "+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47])
#define PF_OUT32(a, O) \
        "+f"(a[(O)+0]),"+f"(a[(O)+1]),"+f"(a[(O)+2]),"+f"(a[(O)+3]),"+f"(a[(O)+4]),"+f"(a[(O)+5]),"+f"(a[(O)+6]),"+f"(a[(O)+7]), \
        "+f"(a[(O)+8]),"+f"(a[(O)+9]),"+f"(a[(O)+10]),"+f"(a[(O)+11]),"+f"(a[(O)+12]),"+f"(a[(O)+13]),"+f"(a[(O)+14]),"+f"(a[(O)+15]), \
        "+f"(a[(O)+16]),"+f"(a[(O)+17]),"+f"(a[(O)+18]),"+f"(a[(O)+19]),"+f"(a[(O)+20]),"+f"(a[(O)+21]),"+f"(a[(O)+22]),"+f"(a[(O)+23]), \
        "+f"(a[(O)+24]),"+f"(a[(O)+25]),"+f"(a[(O)+26]),"+f"(a[(O)+27]),"+f"(a[(O)+28]),"+f"(a[(O)+29]),"+f"(a[(O)+30]),"+f"(a[(O)+31])
#define PF_OUT64(a) \
        "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]), \
        "+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]), \
        "+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]), \
        "+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]), \
        "+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]), \
        "+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]), \
        "+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]), \
        "+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63])
// 64-accumulator OUT with base offset O (for HDV>128 PV output boxes -> acc_o[64b .. 64b+63]).
#define PF_OUT64O(a, O) \
        "+f"(a[(O)+0]),"+f"(a[(O)+1]),"+f"(a[(O)+2]),"+f"(a[(O)+3]),"+f"(a[(O)+4]),"+f"(a[(O)+5]),"+f"(a[(O)+6]),"+f"(a[(O)+7]), \
        "+f"(a[(O)+8]),"+f"(a[(O)+9]),"+f"(a[(O)+10]),"+f"(a[(O)+11]),"+f"(a[(O)+12]),"+f"(a[(O)+13]),"+f"(a[(O)+14]),"+f"(a[(O)+15]), \
        "+f"(a[(O)+16]),"+f"(a[(O)+17]),"+f"(a[(O)+18]),"+f"(a[(O)+19]),"+f"(a[(O)+20]),"+f"(a[(O)+21]),"+f"(a[(O)+22]),"+f"(a[(O)+23]), \
        "+f"(a[(O)+24]),"+f"(a[(O)+25]),"+f"(a[(O)+26]),"+f"(a[(O)+27]),"+f"(a[(O)+28]),"+f"(a[(O)+29]),"+f"(a[(O)+30]),"+f"(a[(O)+31]), \
        "+f"(a[(O)+32]),"+f"(a[(O)+33]),"+f"(a[(O)+34]),"+f"(a[(O)+35]),"+f"(a[(O)+36]),"+f"(a[(O)+37]),"+f"(a[(O)+38]),"+f"(a[(O)+39]), \
        "+f"(a[(O)+40]),"+f"(a[(O)+41]),"+f"(a[(O)+42]),"+f"(a[(O)+43]),"+f"(a[(O)+44]),"+f"(a[(O)+45]),"+f"(a[(O)+46]),"+f"(a[(O)+47]), \
        "+f"(a[(O)+48]),"+f"(a[(O)+49]),"+f"(a[(O)+50]),"+f"(a[(O)+51]),"+f"(a[(O)+52]),"+f"(a[(O)+53]),"+f"(a[(O)+54]),"+f"(a[(O)+55]), \
        "+f"(a[(O)+56]),"+f"(a[(O)+57]),"+f"(a[(O)+58]),"+f"(a[(O)+59]),"+f"(a[(O)+60]),"+f"(a[(O)+61]),"+f"(a[(O)+62]),"+f"(a[(O)+63])
template <int N>
__device__ __forceinline__ void wg_fence_c(float (&c)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(c[i]) :: "memory");
}
template <int M = 32>
__device__ __forceinline__ void wg_fence_a(const uint32_t (&a)[M]) {
#pragma unroll
    for (int i = 0; i < M; ++i) asm volatile("" : "+r"(const_cast<uint32_t&>(a[i])) :: "memory");
}

// ---- head-dim-generic helpers (HD,HDV multiples of 64; QK is m64n{CN}k16 over CN keys, k-step count
// = HD/16; PV is split into HDV/64 independent m64n64k16 output boxes over CN keys). CN (the key-tile
// width) is a template param so large head dims can use CN=64 (more smem stages fit). The Q (A) tile
// always has CTA_M=128 rows while the K/V (B) tiles have CN rows => the per-k-block descriptor stride
// differs for A vs B once HD spans >1 col-block (qk_doffR<CTA_M> vs qk_doffR<CN>). ---------------
template <typename T, int HDt, int CN, int CM = CTA_M>
__device__ __forceinline__ void qk_batch_g(float (&s)[CN/2], uint64_t baseA, uint64_t baseB, int p0) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HDt / 16; ++kb) {
        uint64_t dA = desc_add(baseA, (uint32_t)qk_doffR<CM>(kb));
        uint64_t dB = desc_add(baseB, (uint32_t)qk_doffR<CN>(kb));
        int scaleD = (kb == 0) ? p0 : 1;
        if constexpr (CN == 48) {
            if constexpr (is_f16<T>())
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %26, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n48k16.f32.f16.f16 " PF_ACC24 ",%24,%25,p,%27,%28,%29,%30;\n}\n"
                : PF_OUT24(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
            else
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %26, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 " PF_ACC24 ",%24,%25,p,%27,%28,%29,%30;\n}\n"
                : PF_OUT24(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
        } else if constexpr (CN == 64) {
            if constexpr (is_f16<T>())
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " PF_ACC32 ",%32,%33,p,%35,%36,%37,%38;\n}\n"
                : PF_OUT32(s, 0) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
            else
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " PF_ACC32 ",%32,%33,p,%35,%36,%37,%38;\n}\n"
                : PF_OUT32(s, 0) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
        } else if constexpr (CN == 80) {
            if constexpr (is_f16<T>())
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %42, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n80k16.f32.f16.f16 " PF_ACC40 ",%40,%41,p,%43,%44,%45,%46;\n}\n"
                : PF_OUT40(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
            else
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %42, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 " PF_ACC40 ",%40,%41,p,%43,%44,%45,%46;\n}\n"
                : PF_OUT40(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
        } else if constexpr (CN == 96) {
            if constexpr (is_f16<T>())
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %50, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n96k16.f32.f16.f16 " PF_ACC48 ",%48,%49,p,%51,%52,%53,%54;\n}\n"
                : PF_OUT48(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
            else
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %50, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 " PF_ACC48 ",%48,%49,p,%51,%52,%53,%54;\n}\n"
                : PF_OUT48(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
        } else {
            if constexpr (is_f16<T>())
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 " PF_ACC64 ",%64,%65,p,%67,%68,%69,%70;\n}\n"
                : PF_OUT64(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
            else
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " PF_ACC64 ",%64,%65,p,%67,%68,%69,%70;\n}\n"
                : PF_OUT64(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1),"n"(0),"n"(0));
        }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// One 64-col output box: (CN/16)x m64n64k16 wgmma accumulating into acc[O..O+31] (O compile-time).
// NO fence/commit here -- pv_boxes wraps ALL boxes in ONE wgmma group (single fence + single commit) so
// the whole PV is a single in-flight group. That lets wgmma_wait1 leave the ENTIRE PV overlapping the
// softmax (FA3's 2-group QK/PV pipeline). Per-box commit (the old form) left only the LAST box in flight
// after wait1, so softmax overlapped just 1/N of the PV -> tensor pipe 35% vs FA3 56%.
template <typename T, int O, int N, int CN>
__device__ __forceinline__ void pv_box64_body(float (&acc)[N], const uint32_t (&P)[CN/4], uint64_t baseV) {
#pragma unroll
    for (int kb = 0; kb < CN / 16; ++kb) {
        uint64_t dB = desc_add(baseV, (uint32_t)v_doff(kb));
        if constexpr (is_f16<T>())
            asm volatile(
            "{\n.reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " PF_ACC32 ",{%32,%33,%34,%35},%36,p,%38,%39,%40;\n}\n"
            : PF_OUT32(acc, O)
            : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
        else
            asm volatile(
            "{\n.reg .pred p;\n setp.ne.b32 p, %37, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " PF_ACC32 ",{%32,%33,%34,%35},%36,p,%38,%39,%40;\n}\n"
            : PF_OUT32(acc, O)
            : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
    }
}
// One 128-col PV output box: (CN/16)x m64n128k16 wgmma into acc[O..O+63]. A=P (registers, K-major),
// B=V (MN-major, transB=1). Reads dv[0,64) from baseV and dv[64,128) via the descriptor LBO (== the
// dv-block stride == CN*128 bytes == +BOX), so ONE wgmma covers 128 dv -> half the PV wgmma of the 2x
// m64n64 form (matches FA3's 8x HGMMA.64x128x16 .tnspB, denser tensor feeding). voffR is already a
// canonical MN-major GMMA layout (verified vs cute make_gmma_desc<Major::MN>). No fence/commit here.
template <typename T, int O, int N, int CN>
__device__ __forceinline__ void pv_box128_body(float (&acc)[N], const uint32_t (&P)[CN/4], uint64_t baseV) {
#pragma unroll
    for (int kb = 0; kb < CN / 16; ++kb) {
        uint64_t dB = desc_add(baseV, (uint32_t)v_doff(kb));
        if constexpr (is_f16<T>())
            asm volatile(
            "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 " PF_ACC64 ",{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
            : PF_OUT64O(acc, O)
            : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
        else
            asm volatile(
            "{\n.reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " PF_ACC64 ",{%64,%65,%66,%67},%68,p,%70,%71,%72;\n}\n"
            : PF_OUT64O(acc, O)
            : "r"(P[kb*4+0]),"r"(P[kb*4+1]),"r"(P[kb*4+2]),"r"(P[kb*4+3]),"l"(dB),"r"(1),"n"(1),"n"(1),"n"(1));
    }
}
template <int N>
__device__ __forceinline__ void rescale_o_g(float (&acc_o)[N], const float* sc) {
#pragma unroll
    for (int c = 0; c < N / 4; ++c) {
        acc_o[c*4+0] *= sc[0]; acc_o[c*4+1] *= sc[0];
        acc_o[c*4+2] *= sc[1]; acc_o[c*4+3] *= sc[1];
    }
}
// PV: emit ceil(HDVsm/64) 64-col boxes. V box b starts CN*64 elems (one CN-row col-block) apart;
// the descriptor leading-byte-offset spans that same col-block (CN*64*2 = CN*128 bytes).
template <typename T, int HDV, int CN>
__device__ __forceinline__ void pv_boxes(float (&acc_o)[HDV/2], const uint32_t (&P)[CN/4], uint32_t sV_base) {
    constexpr uint32_t LBO = (uint32_t)(CN * 128), BOX = (uint32_t)(CN * 128);
    // ONE wgmma group for the whole PV: single fence, all boxes, single commit -> wgmma_wait1 (after the
    // next QK) leaves the FULL PV in flight to overlap the softmax (FA3-style QK/PV pipeline). Emit
    // HDV/128 m64n128 boxes (each spans 2 dv-blocks: dv[128b,128b+64) at sV_base+2b*BOX, dv[128b+64,+128)
    // via the descriptor LBO==dv-block stride==+BOX), + a trailing m64n64 box for HDV%128==64. This halves
    // the PV wgmma vs the old 2x-m64n64 form (matches FA3's 8x HGMMA.64x128x16 .tnspB) -> denser tensor
    // feeding, fewer issue slots stolen from softmax. Accumulator reg->(dv,row) mapping is identical.
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    if constexpr (HDV >= 128) pv_box128_body<T, 0,  HDV/2, CN>(acc_o, P, make_smem_desc(sV_base + 0*BOX, LBO, SW_V_SBO, 1));
    if constexpr (HDV >= 256) pv_box128_body<T, 64, HDV/2, CN>(acc_o, P, make_smem_desc(sV_base + 2*BOX, LBO, SW_V_SBO, 1));
    if constexpr (HDV == 64)  pv_box64_body <T, 0,  HDV/2, CN>(acc_o, P, make_smem_desc(sV_base + 0*BOX, LBO, SW_V_SBO, 1));
    if constexpr (HDV == 192) pv_box64_body <T, 64, HDV/2, CN>(acc_o, P, make_smem_desc(sV_base + 2*BOX, LBO, SW_V_SBO, 1));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// First K-block: QK only (s = Q.K^T), HD-generic.
template <typename T, int HDt, int CN, int CM = CTA_M>
__device__ __forceinline__ void qk_gemm_g(float (&s)[CN/2], uint32_t sQ_wg, uint32_t sK_base) {
    uint64_t baseA = make_smem_desc(sQ_wg,   SW_QK_LBO, SW_QK_SBO, 1);
    uint64_t baseB = make_smem_desc(sK_base, SW_QK_LBO, SW_QK_SBO, 1);
    qk_batch_g<T, HDt, CN, CM>(s, baseA, baseB, 0);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    wg_fence_c(s);
}
// Fused QK (next block) + rescale O + PV (current block), HD/HDV/CN-generic. acc_o has HDV/2 regs.
template <typename T, int HD, int HDV, int CN, int CM = CTA_M>
__device__ __forceinline__ void qk_rs_pv_g(float (&s)[CN/2], float (&acc_o)[HDV/2], const uint32_t (&P)[CN/4],
                                           uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base, const float* sc) {
    uint64_t baseA = make_smem_desc(sQ_wg,   SW_QK_LBO, SW_QK_SBO, 1);
    uint64_t baseB = make_smem_desc(sK_base, SW_QK_LBO, SW_QK_SBO, 1);
    qk_batch_g<T, HD, CN, CM>(s, baseA, baseB, 0);
    rescale_o_g(acc_o, sc);
    pv_boxes<T, HDV, CN>(acc_o, P, sV_base);
}
// Fused QK (next block) + PV (current block) with NO O-rescale between the two wgmma issues -- matches
// FA3's schedule (issue QK(i+1) and PV(i) back-to-back, both left in flight, then rescale O AFTER PV via
// rescale_o_g). Inserting the 64-ALU rescale between the issues (qk_rs_pv_g) delays PV and cuts the
// QK||PV tensor overlap; deferring it recovers tensor-pipe utilization. acc_o has HDV/2 regs.
template <typename T, int HD, int HDV, int CN, int CM = CTA_M>
__device__ __forceinline__ void qk_pv_g(float (&s)[CN/2], float (&acc_o)[HDV/2], const uint32_t (&P)[CN/4],
                                        uint32_t sQ_wg, uint32_t sK_base, uint32_t sV_base) {
    uint64_t baseA = make_smem_desc(sQ_wg,   SW_QK_LBO, SW_QK_SBO, 1);
    uint64_t baseB = make_smem_desc(sK_base, SW_QK_LBO, SW_QK_SBO, 1);
    qk_batch_g<T, HD, CN, CM>(s, baseA, baseB, 0);
    pv_boxes<T, HDV, CN>(acc_o, P, sV_base);
}
// Final PV (last block, after the loop), HDV/CN-generic.
template <typename T, int HDV, int CN>
__device__ __forceinline__ void pv_final_g(float (&acc_o)[HDV/2], const uint32_t (&P)[CN/4], uint32_t sV_base) {
    pv_boxes<T, HDV, CN>(acc_o, P, sV_base);
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    wg_fence_c(acc_o);
    wg_fence_a(P);
}

// ============================ native fp8 (e4m3) prefill path ============================
// QK and PV both run on the fp8 tensor core (2x bf16 throughput). The f32 accumulator layout for
// m64n128k32.e4m3 is identical to the bf16 m64n128k16 one, so softmax/mask/epilogue are unchanged.
// fp8 1-byte 128B swizzle: 128 fp8 = one 128B row; the 16-byte (16-fp8) chunk is swizzled by row&7.
// For hd128/CTA_N=128 every operand tile (Q,K,Vt,P) is a [128-row,128-col] K-major fp8 tile. The
// whole recipe (koff8 + LBO16/SBO1024/lt=1 + k32 step=kb*32 + V-transpose + P-in-smem SS-PV) is
// validated bit-exact vs a CPU fp8 GEMM in kernels/fp8_probe.cu.
// Row-count-generic fp8 128B swizzle: the col-block (c>=128) stride is 128*ROWS (# rows in the tile).
// ROWS=CTA_N reproduces the original koff8 exactly (and for HD<=128, c<128 so the col-block term is 0,
// making ROWS irrelevant -- byte-identical to the old koff8 on every existing hd128 path).
template <int ROWS> __device__ __forceinline__ int koff8R(int row, int c) {
    return (c >> 7) * (128 * ROWS) + row * 128 + ((c & 127) ^ ((row & 7) << 4));
}
__device__ __forceinline__ int koff8(int row, int c) { return koff8R<CTA_N>(row, c); }
constexpr uint32_t SW8_LBO = 16, SW8_SBO = 1024;
__device__ __forceinline__ void wg_barrier(int id) {
    asm volatile("bar.sync %0, 128;\n" :: "r"(id) : "memory");
}
// FA3 warp-scheduler ping-pong primitives (mainloop_fwd_sm90_tma_gmma_ws.hpp). A named barrier with
// an explicit arrival count: one consumer WG `bar.sync`s it (contributes its 128 threads + blocks)
// while the other WG `bar.arrive`s it (contributes 128, no block). With cnt = 256 the syncing WG
// unblocks only once the other WG arrives -> the two consumer WGs alternate their wgmma-issue phase
// by one step, so while WG_a runs softmax on the SFU, WG_b feeds the tensor core (and vice-versa).
__device__ __forceinline__ void nb_sync(int id, int cnt) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(cnt) : "memory");
}
__device__ __forceinline__ void nb_arrive(int id, int cnt) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(cnt) : "memory");
}
// QK: s[CN/2] = Q.K^T (m64n{CN}k32 e4m3 SS over HD; first k-block overwrites, rest accumulate). The Q(A)
// tile has CTA_M rows, the K(B) tile CN rows, so once HD spans >1 128-col block their per-k-block descriptor
// strides differ (128*CTA_M vs 128*CN) -- mirrors the bf16 qk_doffR<CM> vs qk_doffR<CN>. CN=CTA_N (hd<=128,
// kb<4) keeps offA==offB==kb*32 (col-block term 0) -> byte-identical to before. Validated in fp8_probe256.cu.
template <int HDt, int CN = CTA_N>
__device__ __forceinline__ void qk_fp8_ss(float (&s)[CN/2], uint32_t sQ_wg, uint32_t sK_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < HDt / 32; ++kb) {
        uint32_t offA = (uint32_t)((kb & 3) * 32 + (kb >> 2) * (128 * CTA_M));
        uint32_t offB = (uint32_t)((kb & 3) * 32 + (kb >> 2) * (128 * CN));
        uint64_t dA = make_smem_desc(sQ_wg + offA, SW8_LBO, SW8_SBO, 1);
        uint64_t dB = make_smem_desc(sK_base + offB, SW8_LBO, SW8_SBO, 1);
        int scaleD = (kb == 0) ? 0 : 1;
        if constexpr (CN == 128)
            asm volatile(
            "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " PF_ACC64 ",%64,%65,p,%67,%68;\n}\n"
            : PF_OUT64(s) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1));
        else
            asm volatile(
            "{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " PF_ACC32 ",%32,%33,p,%35,%36;\n}\n"
            : PF_OUT32(s, 0) : "l"(dA),"l"(dB),"r"(scaleD),"n"(1),"n"(1));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// PV: acc_o[HDV/2] += P.V (e4m3 SS over CTA_N keys; A=P[q][key], B=Vt[dv][key]). m64n128 for HDV=128,
// m64n64 for HDV=64 (hd64 fp8 decode -- acc_o[32]). The Vt koff8 layout + SW8 descriptor are HDV-agnostic
// (only the dv-row count differs), so n is the only change. Always accumulates (running online output).
// For HDV>128 the output is tiled into HDV/128 m64n128 boxes: box b reads Vt rows [128b,128b+128) at
// sV_base + b*(128*128) and accumulates acc_o[64b..64b+63]. acc_o is the persistent online output ->
// scaleD=1 always. CN keys -> CN/32 k-blocks. CN=CTA_N=128,HDV=128 reduces to the old single-box loop.
template <int HDV, int CN = CTA_N>
__device__ __forceinline__ void pv_fp8_ss(float (&acc_o)[HDV/2], uint32_t sP_wg, uint32_t sV_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    constexpr int NBOX = (HDV > 128) ? (HDV / 128) : 1;
#pragma unroll
    for (int box = 0; box < NBOX; ++box) {
        const uint32_t vboff = (uint32_t)(box * (128 * 128));
#pragma unroll
        for (int kb = 0; kb < CN / 32; ++kb) {
            uint32_t off = (uint32_t)(kb * 32);
            uint64_t dA = make_smem_desc(sP_wg + off, SW8_LBO, SW8_SBO, 1);
            uint64_t dB = make_smem_desc(sV_base + vboff + off, SW8_LBO, SW8_SBO, 1);
            if constexpr (HDV >= 128)
                asm volatile(
                "{\n.reg .pred p;\n setp.ne.b32 p, %66, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " PF_ACC64 ",%64,%65,p,%67,%68;\n}\n"
                : PF_OUT64O(acc_o, box * 64) : "l"(dA),"l"(dB),"r"(1),"n"(1),"n"(1));
            else
                asm volatile(
                "{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " PF_ACC32 ",%32,%33,p,%35,%36;\n}\n"
                : PF_OUT32(acc_o, 0) : "l"(dA),"l"(dB),"r"(1),"n"(1),"n"(1));
        }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// FA3 register-source P (MmaPV_is_RS): build the fp8 A-operand registers from the f32 softmax
// probabilities s[64] (C-accumulator layout) WITHOUT the smem scatter+barrier round-trip. Pack each
// c-group's 4 f32 ((p,r2) combined, convert_layout_acc_Aregs order) -> P16[c], then permute_Aregs_fp8
// (quad __shfl + byte_perm) fixes the register layout to the wgmma A operand. Validated bit-exact vs
// the SS path in kernels/fp8_probe.cu (PV-RS). tg=lane&3, group=lane>>2.
template <int CN = CTA_N>
__device__ __forceinline__ void build_P_rs_fp8(const float (&s)[CN/2], uint32_t (&P16)[CN/8], int tg, int group) {
    constexpr int NC = CN / 8;   // c-groups (16 for CN=128, 8 for CN=64); per-pair permute count = NC/2
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        __nv_fp8_e4m3 e0(s[c*4+0]), e1(s[c*4+1]), e2(s[c*4+2]), e3(s[c*4+3]);
        P16[c] = (uint32_t)e0.__x | ((uint32_t)e1.__x << 8) | ((uint32_t)e2.__x << 16) | ((uint32_t)e3.__x << 24);
    }
    const bool lane_03 = (tg == 0 || tg == 3);
    const int sel_up = lane_03 ? 0x5410 : 0x1054, sel_lo = lane_03 ? 0x7632 : 0x3276;
    const int up = (tg == 0) ? 0 : (tg == 1) ? 3 : (tg == 2) ? 1 : 2;   // upper_map[tg]
    const int src_hi = group * 4 + up, src_lo = group * 4 + (up ^ 1);
#pragma unroll
    for (int u = 0; u < NC / 2; ++u) {
        uint32_t upper = P16[2*u + 0], lower = P16[2*u + 1];
        uint32_t upper0 = lane_03 ? upper : lower;
        uint32_t lower0 = lane_03 ? lower : upper;
        upper0 = __shfl_sync(0xffffffffu, upper0, src_hi, 32);
        lower0 = __shfl_sync(0xffffffffu, lower0, src_lo, 32);
        P16[2*u + 0] = __byte_perm(upper0, lower0, sel_up);
        P16[2*u + 1] = __byte_perm(upper0, lower0, sel_lo);
    }
}
// PV with P in REGISTERS (RS A operand): acc_o += P.V over CN keys, A = P16[4kb..4kb+3] per k-block,
// B = Vt[dv][key] (smem descriptor). No smem-P scatter/barrier -> enables the QK(i+1)/PV(i) pipeline.
// HDV>128 is tiled into HDV/128 m64n128 boxes (box b reads Vt rows [128b,128b+128) at sV_base+b*128*128
// and accumulates acc_o[64b..64b+63]); box b reuses the SAME P16 (P is dv-independent). CN k-blocks =
// CN/32, P16 has CN/8 regs. HDV<128 -> single m64n64 box (acc_o[32]). Always accumulates (scaleD=1).
template <int HDV, int CN = CTA_N>
__device__ __forceinline__ void pv_fp8_rs(float (&acc_o)[HDV/2], const uint32_t (&P16)[CN/8], uint32_t sV_base) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    constexpr int NBOX = (HDV > 128) ? (HDV / 128) : 1;
#pragma unroll
    for (int box = 0; box < NBOX; ++box) {
        const uint32_t vboff = (uint32_t)(box * (128 * 128));
#pragma unroll
        for (int kb = 0; kb < CN / 32; ++kb) {
            uint32_t off = (uint32_t)(kb * 32);
            uint64_t dB = make_smem_desc(sV_base + vboff + off, SW8_LBO, SW8_SBO, 1);
            if constexpr (HDV >= 128)
                asm volatile(
                "{\n.reg .pred p;\n setp.ne.b32 p, %68, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " PF_ACC64 ",{%64,%65,%66,%67},%69,p,%70,%71;\n}\n"
                : PF_OUT64O(acc_o, box * 64)
                : "r"(P16[kb*4+0]),"r"(P16[kb*4+1]),"r"(P16[kb*4+2]),"r"(P16[kb*4+3]),
                  "r"(1),"l"(dB),"n"(1),"n"(1));
            else
                asm volatile(
                "{\n.reg .pred p;\n setp.ne.b32 p, %36, 0;\n"
                "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " PF_ACC32 ",{%32,%33,%34,%35},%37,p,%38,%39;\n}\n"
                : PF_OUT32(acc_o, 0)
                : "r"(P16[kb*4+0]),"r"(P16[kb*4+1]),"r"(P16[kb*4+2]),"r"(P16[kb*4+3]),
                  "r"(1),"l"(dB),"n"(1),"n"(1));
        }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
// Two adjacent key-cols share a byte pair => one st.shared.u16 (koff8 is even at col=8c+2tg).
template <int CN = CTA_N>
__device__ __forceinline__ void scatter_P_fp8(const float (&s)[CN/2], uint32_t sP_wg_addr,
                                              int warp, int group, int tg) {
#pragma unroll
    for (int c = 0; c < CN / 8; ++c)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int row = warp * 16 + group + r * 8;
            int col = 8 * c + 2 * tg;
            __nv_fp8_e4m3 a(s[c*4+r*2+0]), b(s[c*4+r*2+1]);
            uint16_t packed = (uint16_t)a.__x | ((uint16_t)b.__x << 8);
            uint32_t dst = sP_wg_addr + (uint32_t)koff8(row, col);
            asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(dst), "h"(packed) : "memory");
        }
}
// cp.async gather (fp8): the producer issues raw 16-byte copies and races STAGES ahead instead of
// blocking on each LDG (a synchronous uint4 gather exposes ~56% long_scoreboard at 1 CTA/SM since
// there is no other resident CTA to hide it). Completion is tied to an mbarrier via
// cp_async_mbar_arrive. Invalid rows (key>=seqlen_k
// or padded q) are zeroed synchronously: masked-out keys must still be finite (P=0 * V; a stale fp8
// NaN would make 0*NaN=NaN). cp.async can't convert/transpose, so K/Q go straight to the swizzled
// tile while V lands in a PLAIN [key][dv] staging tile that a later in-smem pass transposes.
template <int HDt, int ROWS = CTA_N>
__device__ __forceinline__ void gather_row_k8_cp(__nv_fp8_e4m3* sbase, uint32_t sbase_addr, int row,
                                                 const __nv_fp8_e4m3* src, bool valid) {
#pragma unroll
    for (int c16 = 0; c16 < HDt / 16; ++c16) {
        int o = koff8R<ROWS>(row, c16 * 16);
        if (valid) cp_async_16(sbase_addr + (uint32_t)o, src + c16 * 16);
        else       *reinterpret_cast<uint4*>(sbase + o) = make_uint4(0, 0, 0, 0);
    }
}
template <int HDt>
__device__ __forceinline__ void gather_row_plain8_cp(__nv_fp8_e4m3* sbase, uint32_t sbase_addr, int row,
                                                     const __nv_fp8_e4m3* src, bool valid) {
#pragma unroll
    for (int c16 = 0; c16 < HDt / 16; ++c16) {
        int o = row * HDt + c16 * 16;
        if (valid) cp_async_16(sbase_addr + (uint32_t)o, src + c16 * 16);
        else       *reinterpret_cast<uint4*>(sbase + o) = make_uint4(0, 0, 0, 0);
    }
}
// Transpose one key column of the PLAIN [key][dv] staging tile into the swizzled K-major sVt[dv][key]
// (smem->smem; the global load latency is now hidden by cp.async, this pass only touches smem).
template <int HDVt>
__device__ __forceinline__ void transpose_v_smem8(__nv_fp8_e4m3* sVt, const __nv_fp8_e4m3* sVst, int key) {
#pragma unroll
    for (int c16 = 0; c16 < HDVt / 16; ++c16) {
        uint4 chunk = *reinterpret_cast<const uint4*>(sVst + key * HDVt + c16 * 16);
        const __nv_fp8_e4m3* cv = reinterpret_cast<const __nv_fp8_e4m3*>(&chunk);
#pragma unroll
        for (int e = 0; e < 16; ++e) sVt[koff8(c16 * 16 + e, key)] = cv[e];
    }
}

extern __shared__ __align__(1024) __nv_bfloat16 pf_smem[];

// Gather one row (HDt elems) of K-major data from gmem into the swizzled smem tile (16-bit TC), OR
// zero it. TIn is the gmem element type: when TIn==TC (bf16/fp16) it is a raw 16-byte copy (cp.async
// or store); when TIn is fp8-e4m3 each 8-elem chunk is loaded, converted to TC (bf16) and stored
// (TMA/cp.async can't convert ⇒ fp8 always takes the synchronous convert-store path).
template <typename TIn, typename TC, int HDt, int ROWS = CTA_N>
__device__ __forceinline__ void gather_row_k(TC* sbase, uint32_t sbase_addr, int row,
                                             const TIn* src, bool valid) {
#pragma unroll
    for (int c8 = 0; c8 < HDt / 8; ++c8) {
        int o = koffR<ROWS>(row, c8 * 8);
        if (!valid) { *reinterpret_cast<uint4*>(sbase + o) = make_uint4(0, 0, 0, 0); continue; }
        if constexpr (std::is_same<TIn, TC>::value) {
#if PF_CPASYNC
            cp_async_16(sbase_addr + (uint32_t)(o * 2), src + c8 * 8);
#else
            *reinterpret_cast<uint4*>(sbase + o) = *reinterpret_cast<const uint4*>(src + c8 * 8);
#endif
        } else {
            uint2 raw = *reinterpret_cast<const uint2*>(src + c8 * 8);   // 8 fp8 in ONE coalesced load
            const TIn* f = reinterpret_cast<const TIn*>(&raw);
            TC tmp[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) tmp[e] = from_float<TC>(to_float<TIn>(f[e]));
            *reinterpret_cast<uint4*>(sbase + o) = *reinterpret_cast<const uint4*>(tmp);
        }
    }
}
template <typename TIn, typename TC, int HDVt, int ROWS = CTA_N>
__device__ __forceinline__ void gather_row_v(TC* sbase, uint32_t sbase_addr, int key,
                                             const TIn* src, bool valid) {
#pragma unroll
    for (int c8 = 0; c8 < HDVt / 8; ++c8) {
        int o = voffR<ROWS>(key, c8 * 8);
        if (!valid) { *reinterpret_cast<uint4*>(sbase + o) = make_uint4(0, 0, 0, 0); continue; }
        if constexpr (std::is_same<TIn, TC>::value) {
#if PF_CPASYNC
            cp_async_16(sbase_addr + (uint32_t)(o * 2), src + c8 * 8);
#else
            *reinterpret_cast<uint4*>(sbase + o) = *reinterpret_cast<const uint4*>(src + c8 * 8);
#endif
        } else {
            uint2 raw = *reinterpret_cast<const uint2*>(src + c8 * 8);   // 8 fp8 in ONE coalesced load
            const TIn* f = reinterpret_cast<const TIn*>(&raw);
            TC tmp[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) tmp[e] = from_float<TC>(to_float<TIn>(f[e]));
            *reinterpret_cast<uint4*>(sbase + o) = *reinterpret_cast<const uint4*>(tmp);
        }
    }
}

// ---- continuous-ring work-item decode (shared by the persistent producer & consumers) ----
// One linear work index -> (bi, hk, m_block) + the per-tile bounds. Producer and consumers call this
// with the SAME work index so they agree on n_cnt (and thus the global ring counter g advances in
// lock-step). Mirrors the per-tile prefill_kernel coordinate/bounds math, minus split-KV (prefill
// runs num_splits==1). m_block is LPT-reversed (largest causal tile first) for static-strided balance.
struct PfTile {
    bool valid;
    int q0, hk, seqlen_q, seqlen_k, Mrows, row0, delta, n_lo, n_cnt;
    float sc_eff, s2, vd;
};
template <int CN, int CM, bool EXTRA>
__device__ __forceinline__ PfTile pf_work(
        int work, int tiles_m, int hkv, int rep,
        const int* __restrict__ cu_q, const int* __restrict__ cu_k, const int* __restrict__ seqused_k,
        int causal, int win_l, int win_r, float scale,
        const float* __restrict__ q_descale, const float* __restrict__ k_descale,
        const float* __restrict__ v_descale, int ds_b, int ds_h) {
    PfTile t; t.valid = false;
    const int hb = tiles_m * hkv;
    const int bi = work / hb;
    const int rem = work - bi * hb;
    t.hk = rem / tiles_m;
    const int m_block = (tiles_m - 1) - (rem - t.hk * tiles_m);   // LPT: largest causal tile first
    t.q0 = cu_q[bi];
    t.seqlen_q = cu_q[bi + 1] - t.q0;
    t.seqlen_k = seqused_k ? seqused_k[bi] : (cu_k[bi + 1] - cu_k[bi]);
    t.Mrows = t.seqlen_q * rep;
    t.row0 = m_block * CM;
    if (t.row0 >= t.Mrows || t.seqlen_k <= 0) return t;            // padded m_block / empty seq
    t.delta = t.seqlen_k - t.seqlen_q;
    const long ds_idx = (long)bi * ds_b + (long)t.hk * ds_h;
    const float qd = q_descale ? q_descale[ds_idx] : 1.f;
    const float kd = k_descale ? k_descale[ds_idx] : 1.f;
    t.vd = v_descale ? v_descale[ds_idx] : 1.f;
    t.sc_eff = scale * qd * kd;
    t.s2 = t.sc_eff * float(M_LOG2E);
    const int top = (t.row0 + CM - 1 < t.Mrows - 1) ? (t.row0 + CM - 1) : (t.Mrows - 1);
    const int qabs_max = (top / rep) + t.delta;
    int n_max = causal ? ((qabs_max + 1 + CN - 1) / CN) : ((t.seqlen_k + CN - 1) / CN);
    int n_beg = 0;
    if constexpr (EXTRA) {
        if (win_l >= 0) { int lo = (t.row0 / rep) + t.delta - win_l; if (lo > 0) n_beg = lo / CN; }
        if (!causal && win_r >= 0) { int tt = (qabs_max + win_r) / CN + 1; if (tt < n_max) n_max = tt; }
    }
    if (n_beg >= n_max) return t;                                  // fully masked
    t.n_lo = n_beg; t.n_cnt = n_max - n_beg;
    t.valid = true;
    return t;
}

// bi is recovered from work where needed (page_table indexing); keep it out of PfTile to save a reg.
__device__ __forceinline__ int pf_bi(int work, int tiles_m, int hkv) { return work / (tiles_m * hkv); }

// Split-KV: an empty/masked/padded CTA (no key tiles to process) must still stamp -inf into its
// LSEpart rows so combine_kernel drops it. Doing it here lets LSEpart be a plain at::empty -- it
// replaces a per-call host-side at::full that memset the WHOLE scratch (an extra ~2us fill kernel +
// launch every decode call) even though the typical split has no empties. Only the consumer
// warpgroups carry rows; the (cwg/warp/group, tg==0) -> (qidx,qhead) mapping mirrors the epilogue.
__device__ __forceinline__ void emit_empty_lse(float* __restrict__ LSEpart, int tid, int split, int hq,
        int row0, int Mrows, int hk, int rep, int q0, int total_q) {
    if (tid < WG) return;                              // producer warpgroup carries no rows
    const int cwg = tid / WG - 1, wt = tid % WG, wp = wt >> 5, ln = wt & 31;
    if ((ln & 3) != 0) return;                         // only tg==0 writes the LSE (matches epilogue)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int idx = row0 + cwg * 64 + wp * 16 + (ln >> 2) + (r ? 8 : 0);
        if (idx < Mrows)
            LSEpart[((long)split * hq + hk * rep + idx % rep) * total_q + q0 + idx / rep] = -INFINITY;
    }
}

// Single-split (num_splits==1) analogue of emit_empty_lse for a fully-masked / empty / padded tile:
// write 0 to O and -inf to LSE for the tile's valid rows. Without this a fully-masked m-tile (causal
// q>>k: every row's qabs < 0 -> n_max==0 -> empty_cta) would `return` before the mainloop, leaving its
// O rows at the uninitialized at::empty value (garbage). FA3 returns 0 for such rows. The
// (cwg/warp/group, tg) -> (qidx,qhead,dv) mapping mirrors the epilogue exactly.
template <typename TC, int HDV>
__device__ __forceinline__ void emit_zero_o(TC* __restrict__ O, float* __restrict__ LSE, int tid,
        int row0, int Mrows, int hk, int rep, int q0, int total_q, long o_row, long o_head) {
    if (tid < WG) return;                              // producer warpgroup carries no rows
    const int cwg = tid / WG - 1, wt = tid % WG, wp = wt >> 5, ln = wt & 31;
    const int group = ln >> 2, tg = ln & 3;
    const int row_lo = cwg * 64 + wp * 16 + group, row_hi = row_lo + 8;
    const uint32_t z = pack_2x<TC>(0.f, 0.f);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int idx = row0 + (r == 0 ? row_lo : row_hi);
        if (idx >= Mrows) continue;
        const int token = idx / rep, qhead = hk * rep + idx % rep, qidx = q0 + token;
        const long obase = (long)qidx * o_row + (long)qhead * o_head;
#pragma unroll
        for (int c = 0; c < HDV / 8; ++c) *reinterpret_cast<uint32_t*>(&O[obase + 8 * c + 2 * tg]) = z;
        if (tg == 0) LSE[(long)qhead * total_q + qidx] = -INFINITY;
    }
}

template <typename TIn, typename TC, int HD, int HDV, bool USE_TMA, bool EXTRA, int STAGES, int CN = CTA_N,
          int CM = CTA_M, int MINB = 1>
__global__ __launch_bounds__(WG * (1 + CM / 64), MINB) void prefill_kernel(
        const TIn* __restrict__ Q, const TIn* __restrict__ K,
        const TIn* __restrict__ V, TC* __restrict__ O, float* __restrict__ LSE,
        const int* __restrict__ cu_q, const int* __restrict__ cu_k, const int* __restrict__ seqused_k,
        const int* __restrict__ page_table,
        long q_row, long q_head, long k_row, long k_head, long v_row, long v_head, long o_row, long o_head,
        int hkv, int rep, int total_q, int block_size, int page_stride,
        float scale, int causal, float softcap, int win_l, int win_r,
        int num_splits, float* __restrict__ Opart, float* __restrict__ LSEpart,
        const float* __restrict__ q_descale, const float* __restrict__ k_descale,
        const float* __restrict__ v_descale, int ds_b, int ds_h,
        const float* __restrict__ s_aux, const int* __restrict__ leftpad_k,
        const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV) {
    constexpr bool CONV = !std::is_same<TIn, TC>::value;     // fp8 gmem -> bf16 smem (no TMA, convert-store)
    // Smem/compute dims are HD/HDV rounded UP to a multiple of 64 (the 128B-swizzle col-block width):
    // head dims that aren't a multiple of 64 (e.g. 96) pad the tile to the next 64 so the validated
    // koff/voff swizzle still applies. Loads fill only the real HD/HDV cols; the QK k-loop runs HD/16
    // (never touches the pad), and PV computes HDVsm/64 boxes (the pad cols land in discarded acc lanes).
    constexpr int HDsm  = ((HD  + 63) / 64) * 64;
    constexpr int HDVsm = ((HDV + 63) / 64) * 64;
    constexpr int KSTRIDE = CN * HDsm;         // one swizzled K tile (elems)
    constexpr int VSTRIDE = CN * HDVsm;        // one swizzled V tile (elems)
    // Consumer warpgroups = CM/64 (each WG does 64 q-rows). Decode passes CM=64 (1 consumer WG) so the
    // 2nd-warpgroup compute that would be entirely masked for tiny-M decode is never launched, and the
    // halved Q smem buys extra K/V pipeline stages (the memory-latency-hiding lever). Prefill: CM=128.
    constexpr int NCONS = (CM / 64) * WG;      // consumer thread count (empty[] barrier arrival count)
    // LPT (longest-processing-time-first) m-block order for causal: the HW block scheduler dispatches CTAs
    // in linear blockIdx order, so putting the HEAVY diagonal tiles (high m_block, ~m+1 key-tiles) FIRST
    // leaves a LIGHT tail -> shorter makespan / less end-of-grid idle (FA3's tile scheduler does the same).
    // Non-causal tiles are all full-height (order-invariant), so keep the natural order there.
    const int split = blockIdx.x % num_splits;
    const int bx = blockIdx.x / num_splits;
    const int tiles_m_g = (int)gridDim.x / num_splits;
    const int m_block = causal ? (tiles_m_g - 1 - bx) : bx;
    const int hk = blockIdx.y, bi = blockIdx.z;
    const int q0 = cu_q[bi];
    const int seqlen_q = cu_q[bi + 1] - q0;
    const int seqlen_k = seqused_k ? seqused_k[bi] : (cu_k[bi + 1] - cu_k[bi]);
    const int Mrows = seqlen_q * rep;
    const int row0 = m_block * CM;
    if (row0 >= Mrows) return;                          // padded m_block (varlen/ragged): no valid rows
    const int delta = seqlen_k - seqlen_q;             // varlen causal append alignment
    // fp8 descale folding: Q,K,V are stored in smem as their RAW (fp8->bf16) values; the real values
    // are q*qd, k*kd, v*vd. exp(scale*Q_real.K_real) = exp((scale*qd*kd)*(q_raw.k_raw)) ⇒ fold qd*kd
    // into the softmax scale; v_descale is a constant factor ⇒ apply once to O at the epilogue.
    const long ds_idx = (long)bi * ds_b + (long)hk * ds_h;
    const float qd = q_descale ? q_descale[ds_idx] : 1.f;
    const float kd = k_descale ? k_descale[ds_idx] : 1.f;
    const float vd = v_descale ? v_descale[ds_idx] : 1.f;
    const float sc_eff = scale * qd * kd;
    const float s2 = sc_eff * float(M_LOG2E);
    const int lpad = leftpad_k ? leftpad_k[bi] : 0;    // cache_leftpad: ignore keys [0, lpad)
    const int top = (row0 + CM - 1 < Mrows - 1) ? (row0 + CM - 1) : (Mrows - 1);
    const int qabs_max = (top / rep) + delta;
    int n_max = causal ? ((qabs_max + 1 + CN - 1) / CN) : ((seqlen_k + CN - 1) / CN);
    // Sliding-window tile pruning: tiles whose keys are entirely outside [qabs-win_l, qabs+win_r] for
    // every row are fully masked -> skip them. Left bound uses the CTA's smallest qabs (row 0), right
    // bound (non-causal) its largest. Cheap causal/non-SWA path keeps n_beg=0 / n_max unchanged.
    int n_beg = 0;
    if constexpr (EXTRA) {
        if (win_l >= 0) { int lo = (row0 / rep) + delta - win_l; if (lo > 0) n_beg = lo / CN; }
        if (!causal && win_r >= 0) { int t = (qabs_max + win_r) / CN + 1; if (t < n_max) n_max = t; }
        if (lpad > 0) { int lb = lpad / CN; if (lb > n_beg) n_beg = lb; }   // skip fully-leftpad tiles
    }
    // split-KV: this CTA processes the absolute key-tile range [n_lo, n_lo+n_cnt). Ring/barrier phases
    // use a LOCAL counter (0..n_cnt-1) so they start at parity 0; key offsets/masks use the absolute nb.
    // An empty CTA (empty KV, fully window-masked, or a split past the last tile) does no compute -- for
    // split-KV it stamps -inf into its LSEpart rows (so combine drops it) instead of relying on a host
    // pre-fill; single-split just exits. Detected uniformly, before the mbar_init/__syncthreads below.
    int n_lo = n_beg, n_cnt = n_max - n_beg;
    bool empty_cta = (seqlen_k <= 0) || (n_beg >= n_max);
    if (!empty_cta && num_splits > 1) {
        const int per = (n_max - n_beg + num_splits - 1) / num_splits;
        n_lo = n_beg + split * per;
        if (n_lo >= n_max) empty_cta = true;
        else n_cnt = (n_max - n_lo < per) ? (n_max - n_lo) : per;
    }
    if (empty_cta) {
        if (num_splits > 1)
            emit_empty_lse(LSEpart, (int)threadIdx.x, split, hkv * rep, row0, Mrows, hk, rep, q0, total_q);
        else
            emit_zero_o<TC, HDV>(O, LSE, (int)threadIdx.x, row0, Mrows, hk, rep, q0, total_q, o_row, o_head);
        return;
    }

    TC* sQ = reinterpret_cast<TC*>(pf_smem);
    TC* sK0 = sQ + CM * HDsm;
    TC* sV0 = sK0 + STAGES * KSTRIDE;
    uint64_t* bars = reinterpret_cast<uint64_t*>(sV0 + STAGES * VSTRIDE);
    // HD==256 (SPLITKV): FA3-style SEPARATE K and V async pipelines (pipeline_k / pipeline_v). The consumer
    // releases K a full PV-gemm BEFORE V (right after the QK gemm), so the producer refills K deep enough to
    // hide the 80KB load at STAGES=2 -> the wide CN=80 tile fits (which single-ring STAGES=2 starved on).
    // `full`/`empty` double as full_k/empty_k here; full_v/empty_v are the V-side barriers. HD<=192 keeps
    // the original combined K+V ring (full_v/empty_v unused).
    // Gate tightly to the dense hd256 PREFILL instantiation (CM==128 + contiguous TMA). The decode-split
    // path also instantiates prefill_kernel<...,256,...> but with CM==64 and (often) the cp.async gather
    // producer, which arrives on the combined full[] -- if SPLITKV activated there the consumer's full_v[]
    // wait would never be signalled (deadlock). CM==128 && USE_TMA excludes it.
    constexpr bool SPLITKV = (HD == 256) && (CM == 128) && USE_TMA;
    uint64_t* full   = bars;                 // [STAGES] combined K+V (HD<=192) OR K-only (HD==256)
    uint64_t* empty  = bars + STAGES;        // [STAGES]
    uint64_t* full_v = bars + 2 * STAGES;    // [STAGES] V-only  (HD==256)
    uint64_t* empty_v= bars + 3 * STAGES;    // [STAGES] V-only  (HD==256)
    uint64_t* barQ   = bars + (SPLITKV ? 4 : 2) * STAGES;   // [1] count NumProd

    const int tid = threadIdx.x;
    if (tid == 0) {
        // TMA producer parallelizes K/V issue across `pages` lanes (one page each), so full[] takes one
        // expect_tx arrival per lane (= pages); the contiguous (non-paged) TMA path issues the whole tile
        // from one thread => 1 arrival; the non-TMA cp.async path keeps NumProd arrivals.
        const int fcnt = USE_TMA ? (block_size > 0 ? (CN / block_size) : 1) : NumProd;
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], fcnt); mbar_init(&empty[s], NCONS);
            if constexpr (SPLITKV) { mbar_init(&full_v[s], fcnt); mbar_init(&empty_v[s], NCONS); }
        }
        mbar_init(&barQ[0], NumProd);
    }
    __syncthreads();

    const int wg = tid / WG;
    if (wg == 0) {
        // ---------- producer ----------
        producer_regs();
        const int pt = tid;                            // 0..127
        // Q (packed-GQA) loaded once via cp.async gather; rows map to distinct heads, so there is no
        // simple TMA tile for Q. Q is loaded once per CTA (not per block) so its cost is amortized.
        // Only the first CM lanes own a Q row -- sQ is a CM-row buffer, so lanes pt>=CM MUST skip the
        // gather entirely (the !valid branch still writes zeros at koffR<CM>(pt,.) which is OOB for
        // pt>=CM). This is a no-op for CM==128 prefill (pt<128 always); it matters for CM==64 tiles that
        // pack Mrows>64 (multi-m-tile decode-split, e.g. q120) where pt 64..127 would corrupt smem.
        // Coalesced Q gather for the common pow2-rep GQA case (llama rep=4, qwen rep=8): 8 lanes/row x
        // KIT 16B chunks = 128-byte bursts (matches the K/V COOP granularity) instead of the per-row form's
        // scattered 256B. rep pow2 -> token/head via shift/mask (NOT the runtime idx/rep integer division,
        // which -- at 8 rows/lane -- swamped the coalescing gain when tried with real div). Non-pow2 rep or
        // fp8-convert path keeps the per-row gather.
        constexpr bool QCOOP = std::is_same<TIn, TC>::value && (HD % 64 == 0);
        const bool rep_pow2 = (rep & (rep - 1)) == 0;
        if (QCOOP && rep_pow2) {
            constexpr int QT = 8, QKIT = HD / 64, QRPP = WG / QT, QRPT = (CM + QRPP - 1) / QRPP;
            const int qct = pt % QT, qrs = pt / QT;
            const int rlog = 31 - __clz(rep), rmask = rep - 1;
            const uint32_t sQa = (uint32_t)__cvta_generic_to_shared(sQ);
#pragma unroll
            for (int qr = 0; qr < QRPT; ++qr) {
                const int row = qrs + qr * QRPP;
                if (row < CM) {
                    const int ii = row0 + row;
                    const bool qval = ii < Mrows;
                    const int token = qval ? (ii >> rlog) : 0;
                    const int qhead = hk * rep + (qval ? (ii & rmask) : 0);
                    const TIn* qsrc = Q + (long)(q0 + token) * q_row + (long)qhead * q_head;
#pragma unroll
                    for (int ki = 0; ki < QKIT; ++ki) {
                        const int col = ki * 64 + qct * 8;
                        const int o = koffR<CM>(row, col);
                        if (qval) cp_async_16(sQa + (uint32_t)(o * 2), qsrc + col);
                        else *reinterpret_cast<uint4*>(sQ + o) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
        } else {
            const int idx = row0 + pt;
            if (pt < CM) {
                const bool qval = idx < Mrows;
                const int token = qval ? idx / rep : 0;
                const int qhead = hk * rep + (qval ? idx % rep : 0);
                const TIn* qsrc = Q + (long)(q0 + token) * q_row + (long)qhead * q_head;
                gather_row_k<TIn, TC, HD, CM>(sQ, (uint32_t)__cvta_generic_to_shared(sQ), pt, qsrc, qval);
            }
        }
        if constexpr (CONV) mbar_arrive(&barQ[0]);      // convert-store path: plain release arrive
        else                PF_PUBLISH(&barQ[0]);       // non-blocking (cp.async completion-tracked)

        if constexpr (USE_TMA && !CONV) {
            // K/V via per-page [block_size,64]-box TMA, issued in PARALLEL across `pages` producer lanes:
            // lane `pt` (0..pages-1, all in producer warp 0 -> SIMT-lockstep, no drift) drives ONE page of
            // every K/V tile -- its page_table[] global read + (HD/64 + HDV/64) TMA boxes run concurrently
            // with the other lanes instead of serially on one thread, hiding the paged address-gen latency
            // (long-scoreboard) and TMA-issue cost that was starving the consumer. full[] is inited to
            // `pages` arrivals; each lane registers its own page's bytes via expect_tx, so the phase flips
            // after all pages arrive + the full kv_bytes land. Each [block_size,64] box lands at the SAME
            // swizzled offset the wgmma consumer reads (validated byte-exact, tma_probe16.cu):
            //   box bx of page p -> smem off = bx*8192 + p*block_size*64  ==> koff over CTA_N rows.
            constexpr int KBOX = HD / 64, VBOX = HDV / 64, CB = CN * 64;   // CB = one CN-row col-block
            const int xk = hk * HD, xv = hk * HDV;                      // head column base in [.,hkv*HD(V)]
            if (page_table != nullptr) {
                // paged: per-page [block_size,64] boxes, issued in PARALLEL across `pages` producer lanes.
                const int pages = CN / block_size;
                const uint32_t page_bytes = (uint32_t)((block_size * HD + block_size * HDV) * sizeof(TC));
                const int last_pg = (seqlen_k - 1) / block_size;        // clamp target: never read OOB pages
                if (pt < pages) {
                    const int boff = pt * block_size * 64;
                    for (int li = 0; li < n_cnt; ++li) {
                        const int nb = n_lo + li;
                        const int st = li % STAGES;
                        if (li >= STAGES) { int ep = ((li / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
                        int pg = nb * pages + pt; if (pg > last_pg) pg = last_pg;   // OOB pages -> masked
                        const int yrow = page_table[(long)bi * page_stride + pg] * block_size;
                        TC* sKp = sK0 + st * KSTRIDE;
                        TC* sVp = sV0 + st * VSTRIDE;
                        mbar_expect_tx(&full[st], page_bytes);
#pragma unroll
                        for (int bx = 0; bx < KBOX; ++bx) tma_box(sKp + boff + bx * CB, &mapK, xk + bx * 64, yrow, &full[st]);
#pragma unroll
                        for (int bx = 0; bx < VBOX; ++bx) tma_box(sVp + boff + bx * CB, &mapV, xv + bx * 64, yrow, &full[st]);
                    }
                }
            } else {
                // contiguous (non-paged): ONE [CN,64] box per col-block loads the full key tile; the whole
                // tile (all K+V boxes) is issued from a single thread with one expect_tx (fcnt==1). Partial
                // last tiles read OOB rows -> TMA bounds-fills them (full tx still completes) and the
                // consumer masks key>=seqlen_k. yk is the absolute key row for tile nb (varlen: + cu_k[bi]).
                const int ybase = cu_k ? cu_k[bi] : 0;
                const uint32_t tile_bytes = (uint32_t)((CN * HD + CN * HDV) * sizeof(TC));
                const uint32_t k_bytes = (uint32_t)(CN * HD * sizeof(TC));
                const uint32_t v_bytes = (uint32_t)(CN * HDV * sizeof(TC));
                if (pt == 0) {
                    for (int li = 0; li < n_cnt; ++li) {
                        const int nb = n_lo + li;
                        const int st = li % STAGES;
                        const int ep = ((li / STAGES) - 1) & 1;
                        const int yk = ybase + nb * CN;
                        TC* sKp = sK0 + st * KSTRIDE;
                        TC* sVp = sV0 + st * VSTRIDE;
                        if constexpr (SPLITKV) {
                            // K first: its empty[st] is released a PV-gemm earlier by the consumer, so this
                            // wait unblocks sooner -> K TMA issues sooner -> the 80KB load hides at STAGES=2.
                            if (li >= STAGES) { while (!mbar_try_wait(&empty[st], ep)) {} }
                            mbar_expect_tx(&full[st], k_bytes);
#pragma unroll
                            for (int bx = 0; bx < KBOX; ++bx) tma_box(sKp + bx * CB, &mapK, xk + bx * 64, yk, &full[st]);
                            if (li >= STAGES) { while (!mbar_try_wait(&empty_v[st], ep)) {} }
                            mbar_expect_tx(&full_v[st], v_bytes);
#pragma unroll
                            for (int bx = 0; bx < VBOX; ++bx) tma_box(sVp + bx * CB, &mapV, xv + bx * 64, yk, &full_v[st]);
                        } else {
                            if (li >= STAGES) { while (!mbar_try_wait(&empty[st], ep)) {} }
                            mbar_expect_tx(&full[st], tile_bytes);
#pragma unroll
                            for (int bx = 0; bx < KBOX; ++bx) tma_box(sKp + bx * CB, &mapK, xk + bx * 64, yk, &full[st]);
#pragma unroll
                            for (int bx = 0; bx < VBOX; ++bx) tma_box(sVp + bx * CB, &mapV, xv + bx * 64, yk, &full[st]);
                        }
                    }
                }
            }
            return;
        }

        // ---------- non-TMA: cp.async gather ----------
        // Issue loads and publish without waiting -> the producer races up to STAGES ahead of the
        // consumers (bounded only by empty[] backpressure), hiding cp.async latency.
        //
        // COALESCED COOPERATIVE gather (bf16/fp16, HD==HDV, TPR|WG): the naive "1 thread = 1 row" form
        // (fallback below) reads, per warp cp.async instruction, one 16B chunk from 32 DIFFERENT rows --
        // even when rows are physically contiguous the 256B row stride makes that 32 uncoalesced sector
        // requests, pinning L1TEX (NCU paged prefill: 74.6% mem-pipe, 40% stall on the load scoreboard,
        // DRAM only 4.6%). Here TPR=HD/8 consecutive lanes cooperate on ONE row (each issues one 16B
        // chunk), so a row's HD*2 bytes are one coalesced burst -> ~2x fewer L1TEX sectors. To avoid the
        // per-row dependent page_table LDG (an earlier inline-__shfl coop tried that and serialized 2x),
        // physical rows are PREFETCHED into smem in a coalesced pass, decoupled by a producer-WG barrier;
        // the coalesced load then reads sPR[] (short-scoreboard smem, no global dep). Matches vLLM/FA3's
        // PagedKVManager (kGmemThreadsPerRow + decoupled load_page_table). Writes remain koffR/voffR
        // (byte-identical to the wgmma consumer) and full[]=NumProd arrivals are unchanged.
        constexpr bool COOP = std::is_same<TIn, TC>::value && (HD == HDV) &&
                              (HD % 64 == 0) && (HD / 64 >= 1);
        if constexpr (COOP) {
            // FA3 PagedKVManager granularity: TPR=8 lanes/row (kGmemThreadsPerRow), each issuing KIT=HD/64
            // consecutive 16B (8-bf16) chunks -> each row is loaded as HD/64 x 128-byte coalesced bursts
            // (one cache line each), matching FA3's access pattern. RPP=16 rows resolved per producer pass;
            // each lane resolves ONE page entry (spread across all 128 lanes) and __shfl broadcasts the
            // physical row to the 8 lanes cooperating on it (barrier-free -- no smem prefetch, no page bar).
            constexpr int TPR = 8;
            constexpr int KIT = HD / 64;         // 16B chunks per lane per row (2 for hd128)
            constexpr int RPP = WG / TPR;        // rows resolved per pass (16)
            static_assert(CN % RPP == 0, "COOP gather requires RPP | CN");
            constexpr int RPT = CN / RPP;        // rows each lane strides over (also page entries/lane)
            const int col_th = pt % TPR;         // 0..7
            const int rowsub = pt / TPR;         // 0..RPP-1
            const int rsw = (pt & 31) / TPR;     // rowsub within warp (shfl source-lane base)
            const int kbase = (int)(cu_k ? cu_k[bi] : 0);
            // Paged block_size is a power of two (all vLLM/bench paged caches): resolve the page index /
            // offset with shift/mask instead of `key / block_size` + `key % block_size`. block_size is a
            // RUNTIME int, so nvcc otherwise emits the full ~15-20-instruction software idiv/imod PER KEY
            // (128 lanes each resolve 1 key/tile) in the producer -- pure issue-slot waste competing with
            // cp.async (same axis the offset-broadcast win freed). FA3 uses cutlass::FastDivmod here.
            const int bshift = block_size > 0 ? (31 - __clz((unsigned)block_size)) : 0;
            const int bmask  = block_size - 1;
            for (int li = 0; li < n_cnt; ++li) {
                const int nb = n_lo + li;
                const int st = li % STAGES;
                if (li >= STAGES) { int ep = ((li / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
                int ent = -1;                    // this lane's resolved physical row (for row rowsub+col_th*RPP)
                if (col_th < RPT) {
                    const int key = nb * CN + rowsub + col_th * RPP;
                    const bool kv = key < seqlen_k;
                    if (page_table) {
                        int blk = kv ? page_table[(long)bi * page_stride + (key >> bshift)] : -1;
                        ent = kv ? (blk << bshift) + (key & bmask) : -1;
                    } else {
                        ent = kv ? kbase + key : -1;
                    }
                }
                TC* sKp = sK0 + st * KSTRIDE; TC* sVp = sV0 + st * VSTRIDE;
                const uint32_t sKa = (uint32_t)__cvta_generic_to_shared(sKp);
                const uint32_t sVa = (uint32_t)__cvta_generic_to_shared(sVp);
                // FA3 PagedKVManager: the int64 K/V row OFFSET is computed ONCE by the row's resolving lane
                // (ent*stride, a 64-bit IMAD) and shfl-broadcast to the 8 cooperating lanes, instead of each
                // of the 8 recomputing ent*stride (8x redundant int64 muls competing with cp.async for issue
                // slots). Sentinel -1 => masked row (zero-fill). Broadcast the OFFSET (2x32-bit shfl) not row.
                const long koff_l = (ent >= 0) ? ((long)ent * k_row + (long)hk * k_head) : -1;
                const long voff_l = (ent >= 0) ? ((long)ent * v_row + (long)hk * v_head) : -1;
                #pragma unroll
                for (int r = 0; r < RPT; ++r) {
                    const int row = rowsub + r * RPP, src = rsw * TPR + r;
                    const long koff = (long)(((unsigned long)(unsigned)__shfl_sync(0xffffffffu, (int)((unsigned long)koff_l >> 32), src, 32) << 32)
                                             | (unsigned)__shfl_sync(0xffffffffu, (int)koff_l, src, 32));
                    const long voff = (long)(((unsigned long)(unsigned)__shfl_sync(0xffffffffu, (int)((unsigned long)voff_l >> 32), src, 32) << 32)
                                             | (unsigned)__shfl_sync(0xffffffffu, (int)voff_l, src, 32));
#pragma unroll
                    for (int ki = 0; ki < KIT; ++ki) {
                        const int col = ki * 64 + col_th * 8;
                        const int ok = koffR<CN>(row, col), ov = voffR<CN>(row, col);
                        if (koff >= 0) {
                            cp_async_16(sKa + (uint32_t)(ok * 2), K + koff + col);
                            cp_async_16(sVa + (uint32_t)(ov * 2), V + voff + col);
                        } else {
                            *reinterpret_cast<uint4*>(sKp + ok) = make_uint4(0, 0, 0, 0);
                            *reinterpret_cast<uint4*>(sVp + ov) = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
                PF_PUBLISH(&full[st]);           // COOP implies !CONV (is_same<TIn,TC>)
            }
            return;
        }
        for (int li = 0; li < n_cnt; ++li) {
            const int nb = n_lo + li;
            const int st = li % STAGES;
            if (li >= STAGES) { int ep = ((li / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
            // CN keys -> CN producer threads load one row each; the rest only arrive on the barrier.
            const int key = nb * CN + pt;
            const bool kval = (pt < CN) && (key < seqlen_k);
            if (pt < CN) {
                long krow;
                if (page_table) {
                    int blk = kval ? page_table[(long)bi * page_stride + key / block_size] : 0;
                    krow = (long)blk * block_size + (kval ? (key % block_size) : 0);
                } else {
                    krow = (long)(cu_k ? cu_k[bi] : 0) + (kval ? key : 0);
                }
                const TIn* ksrc = K + krow * k_row + (long)hk * k_head;
                const TIn* vsrc = V + krow * v_row + (long)hk * v_head;
                TC* sKp = sK0 + st * KSTRIDE; TC* sVp = sV0 + st * VSTRIDE;
                gather_row_k<TIn, TC, HD, CN>(sKp, (uint32_t)__cvta_generic_to_shared(sKp), pt, ksrc, kval);
                gather_row_v<TIn, TC, HDV, CN>(sVp, (uint32_t)__cvta_generic_to_shared(sVp), pt, vsrc, kval);
            }
            if constexpr (CONV) mbar_arrive(&full[st]);  // convert-store path: plain release arrive
            else                PF_PUBLISH(&full[st]);
        }
        return;
    }

    // ---------- consumers ----------
    consumer_regs<(MINB >= 2) ? 128 : 232>();
    const int cwg = wg - 1;
    // Inter-WG ping-pong only with 2 consumer WGs (CM==128 prefill); decode uses CM=64 (1 WG).
    // Gated to HD<=128: big heads (192/256) are already tensor-pipe-saturated per QK/PV tile, so
    // serializing the issue phase only adds barrier latency (measured -0.02 on hd192/hd256 causal),
    // whereas HD<=128 has enough softmax to overlap (chunk16k +0.025, hd96 +0.069, noncausal +0.031).
    constexpr bool PINGPONG = (CM == 128) && (HD <= 128);
    constexpr int PP_BASE = 8, PP_CNT = 2 * WG;   // barriers 8/9, 256 threads (both consumer WGs)
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const int qpos_lo = ((row0 + row_lo) / rep) + delta, qpos_hi = ((row0 + row_hi) / rep) + delta;
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 4096 * (int)sizeof(TC));
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t stK_bytes = (uint32_t)(KSTRIDE * (int)sizeof(TC));
    const uint32_t stV_bytes = (uint32_t)(VSTRIDE * (int)sizeof(TC));

    float acc_o[HDVsm / 2];
#pragma unroll
    for (int i = 0; i < HDVsm / 2; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float s[CN / 2]; uint32_t P[CN / 4]; float sc[2];

    // Break the ping-pong symmetry: WG0 pre-arrives its own scheduler barrier so it issues first.
    if constexpr (PINGPONG) { if (cwg == 0) nb_arrive(PP_BASE, PP_CNT); }

    while (!mbar_try_wait(&barQ[0], 0)) {}
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    while (!mbar_try_wait(&full[0], 0)) {}
    // K/V writes (TMA complete-tx OR cp.async.mbarrier.arrive -> full[]) are ordered before the wgmma
    // reads by the mbarrier completion itself (same async proxy), so NO per-tile fence.proxy.async is
    // needed on either path -- matches FA3/CUTLASS, whose cp.async consumer has no steady-state fence.
#pragma unroll
    for (int i = 0; i < CN / 2; ++i) s[i] = 0.f;
    qk_gemm_g<TC, HD, CN, CM>(s, sQ_wg, sK_a);
    if constexpr (SPLITKV) mbar_arrive(&empty[0]);   // K[0] consumed by QK(0) (qk_gemm_g fully waited) -> free it
    if constexpr (EXTRA) { if (softcap > 0.f) pf_softcap<CN>(s, scale, softcap);
                           pf_mask_win<CN>(s, n_lo * CN, qpos_lo, qpos_hi, seqlen_k, causal, win_l, win_r, tg, lpad); }
    else                   pf_mask<CN>(s, n_lo * CN, qpos_lo, qpos_hi, seqlen_k, causal, tg);
    sc[0] = softmax_stats<CN>(s, m_r, l_r, 0, 0, s2);
    sc[1] = softmax_stats<CN>(s, m_r, l_r, 1, 0, s2);
#pragma unroll
    for (int i = 0; i < CN / 4; ++i) P[i] = pack_2x<TC>(s[2*i], s[2*i+1]);

    for (int li = 1; li < n_cnt; ++li) {
        const int nb = n_lo + li;
        const int st = li % STAGES, pst = (li - 1) % STAGES;
        const int fp = (li / STAGES) & 1;
        if constexpr (PINGPONG) nb_sync(PP_BASE + cwg, PP_CNT);        // wait my turn to issue
        // FA3 gates the K/V-ready wait to the FIRST consumer WG only: WG1 is released by WG0's ping-pong
        // arrive (which fires AFTER WG0 confirmed full[st]), so K/V is already resident -> WG1 skips the
        // spin, freeing its issue slots. (Non-pingpong / decode: every WG waits.)
        if constexpr (!PINGPONG) { while (!mbar_try_wait(&full[st], fp)) {} }   // K[li] ready (SPLITKV) / K+V (combined)
        else if (cwg == 0)      { while (!mbar_try_wait(&full[st], fp)) {} }
        if constexpr (SPLITKV) { const int fpv = ((li - 1) / STAGES) & 1; while (!mbar_try_wait(&full_v[pst], fpv)) {} }  // V[li-1] ready
        // No fence.proxy.async here: the cp.async.mbarrier.arrive -> full[st] completion already orders the
        // gather writes before the wgmma reads (same async proxy), so the per-tile fence is redundant --
        // FA3's cp.async consumer has NO steady-state fence either. Removing it frees consumer issue slots.
        // QK(li) + PV(li-1) issued back-to-back (no rescale between) -> both wgmmas overlap; acc_o was
        // already rescaled to the running max at the END of the previous step (FA3 schedule).
        qk_pv_g<TC, HD, HDVsm, CN, CM>(s, acc_o, P, sQ_wg, sK_a + st * stK_bytes, sV_a + pst * stV_bytes);
        if constexpr (PINGPONG) nb_arrive(PP_BASE + (1 - cwg), PP_CNT); // release the other WG
        wgmma_wait1();
        // SPLITKV: QK(li) done at wgmma_wait1 (PV still in flight) -> release K[li] a full PV-gemm early so the
        // producer refills K sooner (FA3 pipeline_k.consumer_release after warpgroup_wait<1>). sK0[st] and
        // sV0[pst] are distinct buffers, so freeing K here does not touch the V the PV is still reading.
        if constexpr (SPLITKV) mbar_arrive(&empty[st]);
        if constexpr (EXTRA) { if (softcap > 0.f) pf_softcap<CN>(s, scale, softcap);
                               pf_mask_win<CN>(s, nb * CN, qpos_lo, qpos_hi, seqlen_k, causal, win_l, win_r, tg, lpad); }
        else                   pf_mask<CN>(s, nb * CN, qpos_lo, qpos_hi, seqlen_k, causal, tg);
        sc[0] = softmax_stats<CN>(s, m_r, l_r, 0, li, s2);
        sc[1] = softmax_stats<CN>(s, m_r, l_r, 1, li, s2);
        wgmma_wait0();
        if constexpr (SPLITKV) mbar_arrive(&empty_v[pst]);   // V[li-1] consumed by PV -> free V (FA3 releases V after PV)
        else mbar_arrive(&empty[pst]);   // combined ring: release stage pst NOW (fully consumed at wgmma_wait0;
                                    // rescale_o_g/pack touch only registers) so the producer refills sooner.
                                    // Matches FA3 (releases before convert/rescale). paged pf_s2k 0.919->0.925.
        rescale_o_g(acc_o, sc);                                        // rescale O AFTER PV, to the new max
#pragma unroll
        for (int i = 0; i < CN / 4; ++i) P[i] = pack_2x<TC>(s[2*i], s[2*i+1]);
    }
    // Final PV: acc_o already at the last step's max (rescaled at loop end) -> no rescale before it.
    // SPLITKV: the loop only waited V up to tile n_cnt-2; the last tile's V (read by pv_final) is on its own
    // full_v barrier and must be waited here (also covers the single-tile n_cnt==1 case where the loop is skipped).
    if constexpr (SPLITKV) { const int lst = (n_cnt - 1) % STAGES; while (!mbar_try_wait(&full_v[lst], ((n_cnt - 1) / STAGES) & 1)) {} }
    pv_final_g<TC, HDVsm, CN>(acc_o, P, sV_a + ((n_cnt - 1) % STAGES) * stV_bytes);
    if constexpr (SPLITKV) mbar_arrive(&empty_v[(n_cnt - 1) % STAGES]);
    else                   mbar_arrive(&empty[(n_cnt - 1) % STAGES]);

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        int idx = row0 + (r == 0 ? row_lo : row_hi);
        bool val = idx < Mrows;
        int token = idx / rep, qhead = hk * rep + idx % rep, qidx = q0 + token;
        float mr = m_r[r];
        // attention sink (s_aux): an extra per-head logit in the softmax DENOMINATOR (no value, so the
        // numerator/acc_o is untouched). FA3 finalize_aux: sum += exp2(log2e*s_aux - max_scaled), with a
        // fully-masked row clamping its max to 0. Only for the final (single-split) result.
        if (s_aux != nullptr && num_splits <= 1) {
            if (mr == -INFINITY) mr = 0.f;
            l_r[r] += ex2_approx(float(M_LOG2E) * s_aux[qhead] - mr * s2);
        }
        bool bad = (l_r[r] == 0.f || l_r[r] != l_r[r]);
        float inv = bad ? 0.f : 1.f / l_r[r];
        const float invvd = inv * vd;                  // v_descale folds into the normalized output (fp8)
        float lse_v = bad ? -INFINITY : mr * sc_eff + __logf(l_r[r]);
        if (num_splits > 1) {
            // write the NORMALIZED split partial (O_s, lse_s); combine_kernel does the cross-split reduce.
            const int hq = hkv * rep;
            float* op = Opart + (((long)split * total_q + qidx) * hq + qhead) * HDV;
#pragma unroll
            for (int c = 0; c < HDV / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                if (val) { op[dv0] = acc_o[c*4+r*2+0] * invvd; op[dv0+1] = acc_o[c*4+r*2+1] * invvd; }
            }
            if (tg == 0 && val) LSEpart[((long)split * hq + qhead) * total_q + qidx] = lse_v;
        } else {
            long obase = (long)qidx * o_row + (long)qhead * o_head;
#pragma unroll
            for (int c = 0; c < HDV / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                uint32_t o = pack_2x<TC>(acc_o[c*4+r*2+0] * invvd, acc_o[c*4+r*2+1] * invvd);
                if (val) *reinterpret_cast<uint32_t*>(&O[obase + dv0]) = o;
            }
            if (tg == 0 && val) LSE[(long)qhead * total_q + qidx] = lse_v;
        }
    }
}

// ===================== persistent continuous-ring prefill (hd128 bf16 TMA) =====================
// FA3-style persistent kernel: each CTA grabs work-items off a static stride (work = blockIdx.x +
// k*gridDim.x) and processes them in a loop. The win vs the per-tile prefill_kernel is that the K/V
// ring is ONE CONTINUOUS stream across work-items -- PipelineState (the ring index g + its phase) is
// declared once and NEVER reset per item, so the producer prefetches the next item's K/V into slots
// the consumer just freed, with no per-item pipeline-fill bubble. (The earlier naive per-item-reinit
// persistent attempt regressed precisely because it paid that fill + barrier-reinit cost every item.)
//
// Q is single-buffered with a barQf/barQe ping-pong (phase = witer&1): the producer waits barQe (the
// consumer freed the prior Q), reloads Q -- via ONE async TMA box-set (rep==1, mapQ) so the reload is
// cheap enough to hide behind the continuous ring; GQA/paged would fall back to the cp.async gather --
// then signals barQf. To keep the ring primed across the boundary the producer issues the item's FIRST
// K/V tile (empty-gated on the global g, NOT Q-gated) BEFORE the barQe wait, then floods the rest after
// Q. Stage index / phase everywhere use the GLOBAL counter g so
// producer and consumer stay in lock-step (both advance g by n_cnt per non-empty item, skip empties
// identically). num_splits is implicitly 1 (prefill); no split-KV epilogue. TMA-only (routed so).
template <typename TIn, typename TC, int HD, int HDV, bool USE_TMA, bool EXTRA, int STAGES, int CN = CTA_N,
          int CM = CTA_M, int MINB = 1, bool TMA_Q = USE_TMA>
__global__ __launch_bounds__(WG * (1 + CM / 64), MINB) void prefill_persist_kernel(
        const TIn* __restrict__ Q, const TIn* __restrict__ K,
        const TIn* __restrict__ V, TC* __restrict__ O, float* __restrict__ LSE,
        const int* __restrict__ cu_q, const int* __restrict__ cu_k, const int* __restrict__ seqused_k,
        const int* __restrict__ page_table,
        long q_row, long q_head, long k_row, long k_head, long v_row, long v_head, long o_row, long o_head,
        int hkv, int rep, int total_q, int block_size, int page_stride,
        float scale, int causal, float softcap, int win_l, int win_r,
        const float* __restrict__ q_descale, const float* __restrict__ k_descale,
        const float* __restrict__ v_descale, int ds_b, int ds_h,
        int tiles_m, int batch, int* __restrict__ work_ctr, const float* __restrict__ s_aux,
        const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
        const __grid_constant__ CUtensorMap mapQ) {
    constexpr int HDsm  = ((HD  + 63) / 64) * 64;
    constexpr int HDVsm = ((HDV + 63) / 64) * 64;
    constexpr int KSTRIDE = CN * HDsm;
    constexpr int VSTRIDE = CN * HDVsm;
    constexpr int NCONS = (CM / 64) * WG;

    constexpr int QBUF = 1;                    // Single Q buffer: rep==1 Q is reloaded via ONE async TMA
                                               // (mapQ), fast enough that the STAGES-deep continuous K/V
                                               // ring hides it across the item boundary. Double-Q hid the
                                               // reload too but forced STAGES=2 (smem) -> starved K/V.
    constexpr int SDEPTH = 2;                  // dynamic-scheduler work FIFO depth (producer lookahead)
    TC* sQ = reinterpret_cast<TC*>(pf_smem);
    TC* sK0 = sQ + QBUF * CM * HDsm;
    TC* sV0 = sK0 + STAGES * KSTRIDE;
    uint64_t* bars = reinterpret_cast<uint64_t*>(sV0 + STAGES * VSTRIDE);
    uint64_t* full = bars;                     // [STAGES] count 1 (TMA tx-barrier)
    uint64_t* empty = bars + STAGES;           // [STAGES] count NCONS
    uint64_t* barQf = bars + 2 * STAGES;       // [QBUF] Q ready (producer -> consumers), count NumProd
    uint64_t* barQe = bars + 2 * STAGES + QBUF;// [QBUF] Q free  (consumers -> producer), count NCONS
    uint64_t* schF = bars + 2 * STAGES + 2 * QBUF;          // [SDEPTH] work ready (producer -> consumers)
    uint64_t* schE = bars + 2 * STAGES + 2 * QBUF + SDEPTH; // [SDEPTH] slot free (consumers -> producer)
    int* schW = reinterpret_cast<int*>(bars + 2 * STAGES + 2 * QBUF + 2 * SDEPTH);  // [SDEPTH] grabbed work idx

    // Q via single async TMA when KV is contiguous (non-paged): rep==1 uses one 2D [CM,64] box per
    // col-block; packed-GQA (rep>1, rep|CM) uses one 3D [64,rep,CM/rep] box per col-block (mapQ is built
    // accordingly by the launcher). Both land in the same swizzled tile gather_row_k would have built, so
    // the per-item Q reload is one async TMA (not a 128-row cp.async gather) -- which is what makes the
    // persistent ring WIN for GQA/MQA (the gather reload serialized and lost to the per-tile grid).
    // Paged / rep-not-dividing-CM keep the cp.async gather.
    // Q is NEVER paged (only K/V) -> Q can use TMA even when K/V uses the cp.async COOP gather (USE_TMA=0).
    // TMA_Q decouples this (FA3: Use_TMA_Q independent of Use_TMA_KV): the paged persist launches TMA_Q=1
    // so the per-work-item Q reload is ONE async TMA box (cheap) instead of the 128-lane cp.async gather --
    // the reason the paged persist previously lost to the per-tile grid.
    const bool tmaq     = TMA_Q && (rep == 1);
    const bool tmaq_gqa = TMA_Q && (rep > 1) && (CM % rep == 0);
    const int tid = threadIdx.x;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], USE_TMA ? 1 : NumProd); mbar_init(&empty[s], NCONS); }
#pragma unroll
        for (int s = 0; s < QBUF; ++s) { mbar_init(&barQf[s], (tmaq || tmaq_gqa) ? 1 : NumProd); mbar_init(&barQe[s], NCONS); }
#pragma unroll
        for (int s = 0; s < SDEPTH; ++s) { mbar_init(&schF[s], 1); mbar_init(&schE[s], NCONS); }
    }
    __syncthreads();

    // Dynamic LPT-free balancing: instead of static work=blockIdx.x+k*gridDim.x (which strands the
    // causal triangular load -- early m-tiles are light, late ones heavy -- on fixed CTAs), each CTA
    // pulls the next tile from a global atomic counter. The producer's lead thread grabs + publishes the
    // index through a tiny smem FIFO (schW/schF/schE) so both warpgroups process the SAME stream while
    // the GPU load-balances across CTAs the way the HW block scheduler does for the per-tile grid.
    const int total_work = tiles_m * hkv * batch;
    const int wg = tid / WG;

    if (wg == 0) {
        // ---------- producer ----------
        producer_regs();
        const int pt = tid;
        constexpr int KBOX = HD / 64, VBOX = HDV / 64, CB = CN * 64;
        const bool np = (page_table == nullptr);                      // non-paged contiguous KV
        const int pages = np ? 0 : (CN / block_size);
        const uint32_t kv_bytes = (uint32_t)((CN * HD + CN * HDV) * sizeof(TC));
        // COOP gather constants (non-TMA paged persist). FA3 granularity: TPR=8 lanes/row, KIT=HD/64
        // 16B chunks per lane (128-byte coalesced bursts). Barrier-free __shfl page resolve (see prefill_kernel).
        constexpr int TPR = 8, KIT = HD / 64, RPP = WG / TPR, RPT = CN / RPP;
        const int col_th = pt % TPR, rowsub = pt / TPR;
        const int rsw = (pt & 31) / TPR;
        int g = 0, witer = 0;
        for (int it = 0; ; ++it) {
            // Dynamic work-stealing via a global atomic (greedy balance for the causal triangular load;
            // static round-robin regressed -- worse balance). pt==0 grabs + publishes via schW, the producer
            // warpgroup broadcasts it (bar.sync), and pt==0 signals schF for the consumers.
            const int sslot = it % SDEPTH;
            if (pt == 0 && it >= SDEPTH) { int ep = ((it / SDEPTH) - 1) & 1; while (!mbar_try_wait(&schE[sslot], ep)) {} }
            if (pt == 0) schW[sslot] = atomicAdd(work_ctr, 1);
            asm volatile("bar.sync 1, %0;\n" :: "r"(WG) : "memory");
            const int work = schW[sslot];
            if (pt == 0) { __threadfence_block(); mbar_arrive(&schF[sslot]); }
            if (work >= total_work) break;
            PfTile t = pf_work<CN, CM, EXTRA>(work, tiles_m, hkv, rep, cu_q, cu_k, seqused_k,
                                              causal, win_l, win_r, scale, q_descale, k_descale, v_descale, ds_b, ds_h);
            if (!t.valid) continue;
            const int gbase = g;
            g += t.n_cnt;
            const int bi = pf_bi(work, tiles_m, hkv);
            const int last_pg = np ? 0 : ((t.seqlen_k - 1) / block_size);
            const int ybase = np ? (cu_k ? cu_k[bi] : 0) : 0;         // non-paged: absolute key row of req bi
            const int xk = t.hk * HD, xv = t.hk * HDV;
            // load absolute key-tile nb into ring stage (gg % STAGES); pt==0 issues the TMA boxes.
            // empty-gate on the GLOBAL gg keeps the ring a single continuous stream across items.
            auto load_kv = [&](int gg, int nb) {
                const int st = gg % STAGES;
                if (gg >= STAGES) { int ep = ((gg / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
                TC* sKp = sK0 + st * KSTRIDE;
                TC* sVp = sV0 + st * VSTRIDE;
                mbar_expect_tx(&full[st], kv_bytes);
                if (np) {
                    // contiguous (non-paged): ONE [CN,64] box per col-block (same swizzled layout as paged).
                    const int yk = ybase + nb * CN;
#pragma unroll
                    for (int bx = 0; bx < KBOX; ++bx) tma_box(sKp + bx * CB, &mapK, xk + bx * 64, yk, &full[st]);
#pragma unroll
                    for (int bx = 0; bx < VBOX; ++bx) tma_box(sVp + bx * CB, &mapV, xv + bx * 64, yk, &full[st]);
                } else {
                    for (int p = 0; p < pages; ++p) {
                        int pg = nb * pages + p; if (pg > last_pg) pg = last_pg;
                        int yrow = page_table[(long)bi * page_stride + pg] * block_size;
                        int boff = p * block_size * 64;
#pragma unroll
                        for (int bx = 0; bx < KBOX; ++bx) tma_box(sKp + boff + bx * CB, &mapK, xk + bx * 64, yrow, &full[st]);
#pragma unroll
                        for (int bx = 0; bx < VBOX; ++bx) tma_box(sVp + boff + bx * CB, &mapV, xv + bx * 64, yrow, &full[st]);
                    }
                }
            };
            // Non-TMA paged variant: all 128 producer lanes cooperate on the barrier-free spread gather
            // (see prefill_kernel COOP). full[] is inited to NumProd for !USE_TMA, so each lane arrives via
            // cp_async_mbar_arrive (PF_PUBLISH). HD==HDV==128 only (COOP invariant); other HD stay on TMA.
            auto load_kv_coop = [&](int gg, int nb) {
                const int st = gg % STAGES;
                if (gg >= STAGES) { int ep = ((gg / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
                TC* sKp = sK0 + st * KSTRIDE; TC* sVp = sV0 + st * VSTRIDE;
                const uint32_t sKa = (uint32_t)__cvta_generic_to_shared(sKp);
                const uint32_t sVa = (uint32_t)__cvta_generic_to_shared(sVp);
                int ent = -1;
                if (col_th < RPT) {
                    const int key = nb * CN + rowsub + col_th * RPP;
                    const bool kv = key < t.seqlen_k;
                    if (page_table) {
                        int blk = kv ? page_table[(long)bi * page_stride + key / block_size] : -1;
                        ent = kv ? blk * block_size + (key % block_size) : -1;
                    } else {
                        ent = kv ? (ybase + key) : -1;
                    }
                }
#pragma unroll
                for (int r = 0; r < RPT; ++r) {
                    const int row = rowsub + r * RPP;
                    const long krow = __shfl_sync(0xffffffffu, ent, rsw * TPR + r, 32);
#pragma unroll
                    for (int ki = 0; ki < KIT; ++ki) {
                        const int cc = ki * 64 + col_th * 8;
                        const int ok = koffR<CN>(row, cc), ov = voffR<CN>(row, cc);
                        if (krow >= 0) {
                            cp_async_16(sKa + (uint32_t)(ok * 2), K + krow * k_row + (long)t.hk * k_head + cc);
                            cp_async_16(sVa + (uint32_t)(ov * 2), V + krow * v_row + (long)t.hk * v_head + cc);
                        } else {
                            *reinterpret_cast<uint4*>(sKp + ok) = make_uint4(0, 0, 0, 0);
                            *reinterpret_cast<uint4*>(sVp + ov) = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
                PF_PUBLISH(&full[st]);
            };
            // (1) FIRST K/V tile before the Q-free wait -> primes the ring across the item boundary.
            if constexpr (USE_TMA) { if (pt == 0) load_kv(gbase, t.n_lo); }
            else                   load_kv_coop(gbase, t.n_lo);
            // (2) Q into buffer slot witer%QBUF: wait the consumer freed it (item witer-QBUF), reload,
            //     signal. rep==1 reloads via ONE TMA box-set; GQA/paged fall back to the cp.async gather.
            const int qslot = witer % QBUF;
            const int qcyc = witer / QBUF;
            TC* sQs = sQ + qslot * (CM * HDsm);
            if (witer >= QBUF) { int p = (qcyc - 1) & 1; while (!mbar_try_wait(&barQe[qslot], p)) {} }
            if (tmaq) {
                if (pt == 0) {
                    mbar_expect_tx(&barQf[qslot], (uint32_t)(CM * HD * (int)sizeof(TC)));
                    const int yq = t.q0 + t.row0, xq = t.hk * HD;  // rep==1 -> qhead == hk
                    constexpr int QBOX = HD / 64;
#pragma unroll
                    for (int bx = 0; bx < QBOX; ++bx) tma_box(sQs + bx * (CM * 64), &mapQ, xq + bx * 64, yq, &barQf[qslot]);
                }
            } else if (tmaq_gqa) {
                if (pt == 0) {
                    mbar_expect_tx(&barQf[qslot], (uint32_t)(CM * HD * (int)sizeof(TC)));
                    // 3D box: token z = q0 + row0/rep (rep|CM => row0/rep is the tile's first token),
                    // head y = hk*rep, col x = bx*64. One box per col-block fills the [CM,64] swizzle tile.
                    const int zq = t.q0 + t.row0 / rep, yq = t.hk * rep;
                    constexpr int QBOX = HD / 64;
#pragma unroll
                    for (int bx = 0; bx < QBOX; ++bx) tma_box3(sQs + bx * (CM * 64), &mapQ, bx * 64, yq, zq, &barQf[qslot]);
                }
            } else if ((rep & (rep - 1)) == 0) {
                // Coalesced Q reload (pow2 rep -> shift/mask, no runtime div): 8 lanes/row x KIT 16B chunks
                // = 128-byte bursts (matches K/V COOP). This is the FIX that makes the persistent ring beat
                // the per-tile grid for paged prefill -- the uncoalesced per-work-item Q reload was its loss.
                const int rlog = 31 - __clz(rep), rmask = rep - 1;
                const uint32_t sQsa = (uint32_t)__cvta_generic_to_shared(sQs);
                constexpr int QRPT = (CM + RPP - 1) / RPP;
#pragma unroll
                for (int qr = 0; qr < QRPT; ++qr) {
                    const int row = rowsub + qr * RPP;
                    if (row < CM) {
                        const int ii = t.row0 + row;
                        const bool qval = ii < t.Mrows;
                        const int token = qval ? (ii >> rlog) : 0;
                        const int qhead = t.hk * rep + (qval ? (ii & rmask) : 0);
                        const TIn* qsrc = Q + (long)(t.q0 + token) * q_row + (long)qhead * q_head;
#pragma unroll
                        for (int ki = 0; ki < KIT; ++ki) {
                            const int cc = ki * 64 + col_th * 8;
                            const int o = koffR<CM>(row, cc);
                            if (qval) cp_async_16(sQsa + (uint32_t)(o * 2), qsrc + cc);
                            else *reinterpret_cast<uint4*>(sQs + o) = make_uint4(0, 0, 0, 0);
                        }
                    }
                }
                PF_PUBLISH(&barQf[qslot]);
            } else {
                const int idx = t.row0 + pt;
                const bool qval = idx < t.Mrows;
                const int token = qval ? idx / rep : 0;
                const int qhead = t.hk * rep + (qval ? idx % rep : 0);
                const TIn* qsrc = Q + (long)(t.q0 + token) * q_row + (long)qhead * q_head;
                gather_row_k<TIn, TC, HD, CM>(sQs, (uint32_t)__cvta_generic_to_shared(sQs), pt, qsrc, qval);
                PF_PUBLISH(&barQf[qslot]);
            }
            // (3) remaining K/V tiles flood in behind Q.
            if constexpr (USE_TMA) { if (pt == 0) for (int li = 1; li < t.n_cnt; ++li) load_kv(gbase + li, t.n_lo + li); }
            else                   for (int li = 1; li < t.n_cnt; ++li) load_kv_coop(gbase + li, t.n_lo + li);
            witer++;
        }
        return;
    }

    // ---------- consumers ----------
    // NOTE: tried lifting the consumer to 240 by ceding the producer to 24 (256*240+128*24 = the 64512
    // pool) -- it regressed broadly (gqa 0.91->0.87, large4k 0.82->0.78): the TMA producer needs its 40
    // regs to issue/schedule ahead, and starving it stalls the whole ring. Consumer stays at 232.
    consumer_regs<(MINB >= 2) ? 128 : 232>();
    const int cwg = wg - 1;
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const uint32_t sQ_base = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 4096 * (int)sizeof(TC));
    const uint32_t sQ_slot = (uint32_t)(CM * HDsm * (int)sizeof(TC));   // stride between Q double-buffers
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t stK_bytes = (uint32_t)(KSTRIDE * (int)sizeof(TC));
    const uint32_t stV_bytes = (uint32_t)(VSTRIDE * (int)sizeof(TC));

    // Inter-WG ping-pong (mirrors prefill_kernel): the two consumer WGs process the SAME work item in
    // lockstep on the shared K/V ring, so both hit the softmax (SFU) at the same time -> the tensor pipe
    // idles (NCU large4k: barrier 31%, tensor 57% vs FA3 70%). Serializing the wgmma-ISSUE phase via two
    // named barriers (8/9, cnt=256) staggers them by one step: while WG_a feeds the tensor core, WG_b runs
    // softmax, and vice-versa. The token stays balanced across work items -- both WGs run the same n_cnt
    // (same work index), so each does exactly one sync+arrive per inner-loop step. Gate: CM==128 (2
    // consumer WGs) && HD==128. Big heads (192/256) are already tensor-pipe-saturated per tile so
    // serializing the issue only adds barrier latency (-0.02 hd192/hd256). On the PERSIST path hd64
    // measured a ~4% regression (1.25->1.20: its short HD=64 QK/PV leaves little softmax to overlap, so
    // the extra barrier dominates) and hd96 was flat -- so unlike the per-tile kernel (HD<=128) the
    // persist ping-pong is scoped to HD==128, where the win is real (local 0.89->0.96, causal 0.95->0.97,
    // gqa 0.91->0.93).
    // hd256: ping-pong measured NEUTRAL here (the two consumer WGs' softmax stall isn't what bounds it),
    // so it stays off. hd192->dv128 (HDV=128, acc_o[64]) has more softmax relative to its smaller PV than
    // hd192/hdv192, so staggering the WGs' issue overlaps it -- enabled to give that borderline config margin.
    constexpr bool PINGPONG = (CM == 128) && (HD == 128 || (HD == 192 && HDV == 128));
    constexpr int PP_BASE = 8, PP_CNT = 2 * WG;   // barriers 8/9, 256 threads (both consumer WGs)
    if constexpr (PINGPONG) { if (cwg == 0) nb_arrive(PP_BASE, PP_CNT); }  // WG0 issues first

    int g = 0, witer = 0;
    for (int it = 0; ; ++it) {
        // Read the producer-grabbed work index for this slot, then free the slot for reuse.
        const int sslot = it % SDEPTH;
        { int p = (it / SDEPTH) & 1; while (!mbar_try_wait(&schF[sslot], p)) {} }
        const int work = schW[sslot];
        __threadfence_block();
        mbar_arrive(&schE[sslot]);
        if (work >= total_work) break;
        PfTile t = pf_work<CN, CM, EXTRA>(work, tiles_m, hkv, rep, cu_q, cu_k, seqused_k,
                                          causal, win_l, win_r, scale, q_descale, k_descale, v_descale, ds_b, ds_h);
        if (!t.valid) {                              // fully-masked / empty / padded item -> O=0, LSE=-inf
            emit_zero_o<TC, HDV>(O, LSE, tid, t.row0, t.Mrows, t.hk, rep, t.q0, total_q, o_row, o_head);
            continue;
        }
        const int gbase = g;
        g += t.n_cnt;
        const int qpos_lo = ((t.row0 + row_lo) / rep) + t.delta, qpos_hi = ((t.row0 + row_hi) / rep) + t.delta;

        float acc_o[HDVsm / 2];
#pragma unroll
        for (int i = 0; i < HDVsm / 2; ++i) acc_o[i] = 0.f;
        float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
        float s[CN / 2]; uint32_t P[CN / 4]; float sc[2];

        // Q ready (buffer slot witer%QBUF, phase = (witer/QBUF)&1).
        const int qslot = witer % QBUF;
        const uint32_t sQ_wg = sQ_base + (uint32_t)qslot * sQ_slot;
        { int p = (witer / QBUF) & 1; while (!mbar_try_wait(&barQf[qslot], p)) {} }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        // first key tile lives at the GLOBAL stage gbase%STAGES (phase from gbase), not stage 0.
        { int st0 = gbase % STAGES, fp0 = (gbase / STAGES) & 1; while (!mbar_try_wait(&full[st0], fp0)) {} }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#pragma unroll
        for (int i = 0; i < CN / 2; ++i) s[i] = 0.f;
        qk_gemm_g<TC, HD, CN, CM>(s, sQ_wg, sK_a + (gbase % STAGES) * stK_bytes);
        if constexpr (EXTRA) { if (softcap > 0.f) pf_softcap<CN>(s, scale, softcap);
                               pf_mask_win<CN>(s, t.n_lo * CN, qpos_lo, qpos_hi, t.seqlen_k, causal, win_l, win_r, tg); }
        else                   pf_mask<CN>(s, t.n_lo * CN, qpos_lo, qpos_hi, t.seqlen_k, causal, tg);
        sc[0] = softmax_stats<CN>(s, m_r, l_r, 0, 0, t.s2);
        sc[1] = softmax_stats<CN>(s, m_r, l_r, 1, 0, t.s2);
#pragma unroll
        for (int i = 0; i < CN / 4; ++i) P[i] = pack_2x<TC>(s[2*i], s[2*i+1]);

        for (int li = 1; li < t.n_cnt; ++li) {
            const int gg = gbase + li, nb = t.n_lo + li;
            const int st = gg % STAGES, pst = (gg - 1) % STAGES, fp = (gg / STAGES) & 1;
            if constexpr (PINGPONG) nb_sync(PP_BASE + cwg, PP_CNT);          // wait my turn to issue
            if constexpr (!PINGPONG) { while (!mbar_try_wait(&full[st], fp)) {} }   // FA3: only WG0 waits K/V
            else if (cwg == 0)      { while (!mbar_try_wait(&full[st], fp)) {} }
            // No per-tile fence.proxy.async: the full[st] mbarrier completion (TMA complete-tx / cp.async
            // arrive) already orders the K/V writes before the wgmma reads (same async proxy) -- matches the
            // per-tile kernel's fence removal (fe7ee46) and FA3's fence-free dense consumer. Frees issue slots.
            // QK(li)+PV(li-1) back-to-back (rescale deferred to after PV -> better wgmma overlap; FA3 sched).
            qk_pv_g<TC, HD, HDVsm, CN, CM>(s, acc_o, P, sQ_wg, sK_a + st * stK_bytes, sV_a + pst * stV_bytes);
            if constexpr (PINGPONG) nb_arrive(PP_BASE + (1 - cwg), PP_CNT);  // release the other WG
            wgmma_wait1();
            if constexpr (EXTRA) { if (softcap > 0.f) pf_softcap<CN>(s, scale, softcap);
                                   pf_mask_win<CN>(s, nb * CN, qpos_lo, qpos_hi, t.seqlen_k, causal, win_l, win_r, tg); }
            else                   pf_mask<CN>(s, nb * CN, qpos_lo, qpos_hi, t.seqlen_k, causal, tg);
            sc[0] = softmax_stats<CN>(s, m_r, l_r, 0, li, t.s2);
            sc[1] = softmax_stats<CN>(s, m_r, l_r, 1, li, t.s2);
            wgmma_wait0();
            mbar_arrive(&empty[pst]);   // release stage pst EARLY: fully consumed at wgmma_wait0 (QK read K,
                                        // PV read V); rescale/pack touch only registers, so signalling empty
                                        // before them lets the producer refill ~130cyc sooner (FA3 release-
                                        // before-pack). Critical at STAGES=2/CN=80 (hd256): the late release
                                        // was delaying the next TMA -> long_scoreboard starve (2.28->). Matches
                                        // the per-tile prefill_kernel consumer (same reordering).
            rescale_o_g(acc_o, sc);                                          // rescale O AFTER PV, to new max
#pragma unroll
            for (int i = 0; i < CN / 4; ++i) P[i] = pack_2x<TC>(s[2*i], s[2*i+1]);
        }
        pv_final_g<TC, HDVsm, CN>(acc_o, P, sV_a + ((gbase + t.n_cnt - 1) % STAGES) * stV_bytes);
        mbar_arrive(&empty[(gbase + t.n_cnt - 1) % STAGES]);
        // free this Q double-buffer slot (after the last QK read in the mainloop; pv_final_g + the
        // epilogue below touch no smem Q) so the producer can refill it for item witer+QBUF.
        mbar_arrive(&barQe[qslot]);

        // epilogue (num_splits==1: write O / LSE directly).
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            l_r[r] = quad_sum(l_r[r]);
            int idx = t.row0 + (r == 0 ? row_lo : row_hi);
            bool val = idx < t.Mrows;
            int token = idx / rep, qhead = t.hk * rep + idx % rep, qidx = t.q0 + token;
            float mr = m_r[r];
            if (s_aux != nullptr) {                       // attention sink: extra logit in the denom (see prefill_kernel)
                if (mr == -INFINITY) mr = 0.f;
                l_r[r] += ex2_approx(float(M_LOG2E) * s_aux[qhead] - mr * t.s2);
            }
            bool bad = (l_r[r] == 0.f || l_r[r] != l_r[r]);
            float inv = bad ? 0.f : 1.f / l_r[r];
            const float invvd = inv * t.vd;
            float lse_v = bad ? -INFINITY : mr * t.sc_eff + __logf(l_r[r]);
            long obase = (long)qidx * o_row + (long)qhead * o_head;
#pragma unroll
            for (int c = 0; c < HDV / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                uint32_t o = pack_2x<TC>(acc_o[c*4+r*2+0] * invvd, acc_o[c*4+r*2+1] * invvd);
                if (val) *reinterpret_cast<uint32_t*>(&O[obase + dv0]) = o;
            }
            if (tg == 0 && val) LSE[(long)qhead * total_q + qidx] = lse_v;
        }
        witer++;
    }
}

// ===================== native fp8 (e4m3) prefill kernel =====================
// Same warp-specialized shape as prefill_kernel (1 producer WG + 2 consumer WGs of 64 q-rows each)
// but the matmuls run natively on the fp8 tensor core (~2x bf16). Q/K are stored as raw fp8 in the
// 1-byte 128B-swizzle smem tile; V is transposed to the K-major [dv][key] tile the fp8 PV needs;
// the softmax probabilities P are downcast to fp8 and round-tripped through smem so PV is an SS GEMM
// (P and V both from smem descriptors -> no register-fragment puzzle). Output is bf16.
template <int HD, int HDV, bool EXTRA, int STAGES, bool USE_TMA = false, int CN = CTA_N>
__global__ __launch_bounds__(NThreads, 1) void prefill_fp8_kernel(
        const __nv_fp8_e4m3* __restrict__ Q, const __nv_fp8_e4m3* __restrict__ K,
        const __nv_fp8_e4m3* __restrict__ V, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
        const int* __restrict__ cu_q, const int* __restrict__ cu_k, const int* __restrict__ seqused_k,
        const int* __restrict__ page_table,
        long q_row, long q_head, long k_row, long k_head, long v_row, long v_head, long o_row, long o_head,
        int hkv, int rep, int total_q, int block_size, int page_stride,
        float scale, int causal, float softcap, int win_l, int win_r,
        int num_splits, float* __restrict__ Opart, float* __restrict__ LSEpart,
        const float* __restrict__ q_descale, const float* __restrict__ k_descale,
        const float* __restrict__ v_descale, int ds_b, int ds_h,
        const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapVst) {
    // K/Q smem are 128-wide regardless of HD: koff8 hardcodes a row*128 swizzle stride, so a 64-wide
    // (hd64) tile would index past CTA_N*HD -> OOB. Allocate 128-wide (HDQK) so koff8 fits; the QK still
    // contracts only the real HD cols (qk_fp8_ss<HD>), never reading the unused upper half.
    constexpr int HDQK = (HD < 128) ? 128 : HD;
    constexpr int KSTRIDE = CN * HDQK;         // one fp8 K tile (elems=bytes)
    constexpr int VSTRIDE = HDV * 128;         // one transposed fp8 V tile (Vt is 128-key-padded, koff8R)
    const int m_block = blockIdx.x / num_splits, split = blockIdx.x % num_splits, hk = blockIdx.y, bi = blockIdx.z;
    const int q0 = cu_q[bi];
    const int seqlen_q = cu_q[bi + 1] - q0;
    const int seqlen_k = seqused_k ? seqused_k[bi] : (cu_k[bi + 1] - cu_k[bi]);
    const int Mrows = seqlen_q * rep;
    const int row0 = m_block * CTA_M;
    if (row0 >= Mrows) return;                          // padded m_block: no valid rows
    const int delta = seqlen_k - seqlen_q;
    const long ds_idx = (long)bi * ds_b + (long)hk * ds_h;
    const float qd = q_descale ? q_descale[ds_idx] : 1.f;
    const float kd = k_descale ? k_descale[ds_idx] : 1.f;
    const float vd = v_descale ? v_descale[ds_idx] : 1.f;
    const float sc_eff = scale * qd * kd;
    const float s2 = sc_eff * float(M_LOG2E);
    const int top = (row0 + CTA_M - 1 < Mrows - 1) ? (row0 + CTA_M - 1) : (Mrows - 1);
    const int qabs_max = (top / rep) + delta;
    const int n_max = causal ? ((qabs_max + 1 + CN - 1) / CN) : ((seqlen_k + CN - 1) / CN);
    // Empty CTA stamps -inf into its split-KV LSEpart rows (see emit_empty_lse) so LSEpart can be a
    // plain at::empty; single-split exits. Detected uniformly, before the mbar_init/__syncthreads.
    int n_lo = 0, n_cnt = n_max;
    bool empty_cta = (seqlen_k <= 0) || (n_max <= 0);
    if (!empty_cta && num_splits > 1) {
        const int per = (n_max + num_splits - 1) / num_splits;
        n_lo = split * per;
        if (n_lo >= n_max) empty_cta = true;
        else n_cnt = (n_max - n_lo < per) ? (n_max - n_lo) : per;
    }
    if (empty_cta) {
        if (num_splits > 1)
            emit_empty_lse(LSEpart, (int)threadIdx.x, split, hkv * rep, row0, Mrows, hk, rep, q0, total_q);
        else
            emit_zero_o<__nv_bfloat16, HDV>(O, LSE, (int)threadIdx.x, row0, Mrows, hk, rep, q0, total_q, o_row, o_head);
        return;
    }

    // RS-PV path (hd256): P stays in registers (build_P_rs_fp8 -> pv_fp8_rs), so the smem P tile (sP) and
    // its scatter+barrier round-trip are gone -- freeing exactly the smem for CN=128/STAGES=2 (224KB) to
    // match FA3's kBlockN=128 hd256 fp8 tile, and enabling the QK(i+1)||PV(i) intra-WG overlap. hd128 fp8
    // keeps the SS path (sP scatter) unchanged.
    constexpr bool RS = (HD == 256);
    constexpr int VstSTRIDE = CN * HDV;        // one PLAIN [key][dv] V staging tile (cp.async target)
    __nv_fp8_e4m3* sQ = reinterpret_cast<__nv_fp8_e4m3*>(pf_smem);
    __nv_fp8_e4m3* sK0 = sQ + CTA_M * HDQK;
    __nv_fp8_e4m3* sV0 = sK0 + STAGES * KSTRIDE;
    __nv_fp8_e4m3* sP  = sV0 + STAGES * VSTRIDE;                       // SS only (unused when RS)
    __nv_fp8_e4m3* sVst0 = RS ? (sV0 + STAGES * VSTRIDE)              // RS: no sP tile
                              : (sP + CTA_M * 128);                    // SS: sP is 128-key-padded (key<128)
    uint64_t* bars = reinterpret_cast<uint64_t*>(sVst0 + STAGES * VstSTRIDE);
    uint64_t* full = bars;                 // [STAGES] count NumProd  (K cp.async done + Vt transposed)
    uint64_t* empty = bars + STAGES;       // [STAGES] count NumCons
    uint64_t* loaded = bars + 2 * STAGES;  // [STAGES] count NumProd  (K + V-staging cp.async delivered)
    uint64_t* barQ = bars + 3 * STAGES;    // [1] count NumProd

    const int tid = threadIdx.x;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], NumProd); mbar_init(&empty[s], NumCons); mbar_init(&loaded[s], USE_TMA ? 1 : NumProd); }
        mbar_init(&barQ[0], NumProd);
    }
    __syncthreads();

    const int wg = tid / WG;
    if (wg == 0) {
        // ---------- producer: cp.async Q/K (swizzled) + V (plain staging), then transpose V into sVt ----
        // cp.async lets the producer race STAGES ahead instead of blocking per-LDG. K/Q stream straight
        // into the swizzled wgmma tiles; V can't be transposed by cp.async so it lands in a plain [key][dv]
        // staging ring (loaded[]) and a pipelined in-smem pass transposes it into the K-major sVt (full[]).
        producer_regs();
        const int pt = tid;
        const int idx = row0 + pt;
        const bool qval = idx < Mrows;
        const int token = qval ? idx / rep : 0;
        const int qhead = hk * rep + (qval ? idx % rep : 0);
        const __nv_fp8_e4m3* qsrc = Q + (long)(q0 + token) * q_row + (long)qhead * q_head;
        gather_row_k8_cp<HD>(sQ, (uint32_t)__cvta_generic_to_shared(sQ), pt, qsrc, qval);
        cp_async_mbar_arrive(&barQ[0]);
        const uint32_t sVst_a = (uint32_t)__cvta_generic_to_shared(sVst0);
        // Issue loads for tile `li` into stage st (K -> swizzled sK, V -> plain staging). USE_TMA path:
        // pt==0 issues TMA boxes (paged: CN/block_size page boxes at page-table rows; contiguous: 1 box),
        // delivered against loaded[st] (one expect_tx covers K+Vst). Else: per-key cp.async gather.
        const int xk_tma = hk * HD, xv_tma = hk * HDV;
        const int last_pg = page_table ? ((seqlen_k - 1) / block_size) : 0;
        const long ybase_tma = (long)(cu_k ? cu_k[bi] : 0);
        auto issue_load = [&](int li) {
            const int st = li % STAGES;
            if (li >= STAGES) { int ep = ((li / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
            const int nb = n_lo + li;
            if constexpr (USE_TMA) {
                if (pt == 0) {
                    mbar_expect_tx(&loaded[st], (uint32_t)((CTA_N * HD + CTA_N * HDV)));
                    __nv_fp8_e4m3* sKp = sK0 + st * KSTRIDE;
                    __nv_fp8_e4m3* sVstp = sVst0 + st * VstSTRIDE;
                    if (page_table) {
                        const int pages = CTA_N / block_size;
                        for (int p = 0; p < pages; ++p) {
                            int pg = nb * pages + p; if (pg > last_pg) pg = last_pg;
                            int yrow = page_table[(long)bi * page_stride + pg] * block_size;
                            int boff = p * block_size * 128;
                            tma_box(sKp + boff, &mapK, xk_tma, yrow, &loaded[st]);
                            tma_box(sVstp + boff, &mapVst, xv_tma, yrow, &loaded[st]);
                        }
                    } else {
                        int yk = (int)ybase_tma + nb * CTA_N;
                        tma_box(sKp, &mapK, xk_tma, yk, &loaded[st]);
                        tma_box(sVstp, &mapVst, xv_tma, yk, &loaded[st]);
                    }
                }
                return;
            }
            // COALESCED COOP gather (fp8): TPR8=HD/16 consecutive lanes cooperate on ONE key's row (each a
            // 16B = 16-fp8 chunk), so a key's HD bytes are one coalesced burst -- vs the naive 1-thread-per-
            // key form, whose 32 rows-256B-apart per warp are 32 scattered sectors and pinned the fp8
            // prefill at ~19k long_scoreboard (uncoalesced loads). Barrier-free page resolve via __shfl
            // (mirrors the bf16 COOP). K -> swizzled koff8R<CN>; V -> plain [key][dv] staging (transposed later).
            // COOP8 requires TPR8=HD/16 to be a power of 2 (shfl width) dividing WG, and RPP8 | CN. hd96
            // (TPR8=6) fails this and keeps the per-key gather.
            constexpr int TPR8 = HD / 16, RPP8c = TPR8 ? (WG / TPR8) : 1;
            constexpr bool COOP8 = (HD % 16 == 0) && TPR8 > 0 && ((TPR8 & (TPR8 - 1)) == 0) &&
                                   (WG % TPR8 == 0) && (CN % RPP8c == 0) && (HD == HDV);
            if constexpr (COOP8) {
                constexpr int RPP8 = RPP8c, RPT8 = CN / RPP8;
                const int ct8 = pt % TPR8, ks8 = pt / TPR8, col8 = ct8 * 16;
                const int rsw8 = (pt & 31) / TPR8;
                const int kbase8 = (int)(cu_k ? cu_k[bi] : 0);
                __nv_fp8_e4m3* sKp = sK0 + st * KSTRIDE;
                __nv_fp8_e4m3* sVstp = sVst0 + st * VstSTRIDE;
                const uint32_t sKa = (uint32_t)__cvta_generic_to_shared(sKp);
                const uint32_t sVsa = (uint32_t)__cvta_generic_to_shared(sVstp);
                int ent8 = -1;                   // this lane's physical row (for key ks8 + ct8*RPP8)
                if (ct8 < RPT8) {
                    const int key = nb * CN + ks8 + ct8 * RPP8;
                    const bool kv = key < seqlen_k;
                    if (page_table) {
                        int blk = kv ? page_table[(long)bi * page_stride + key / block_size] : -1;
                        ent8 = kv ? blk * block_size + (key % block_size) : -1;
                    } else {
                        ent8 = kv ? kbase8 + key : -1;
                    }
                }
#pragma unroll
                for (int r = 0; r < RPT8; ++r) {
                    const int key = ks8 + r * RPP8;
                    const long krow = __shfl_sync(0xffffffffu, ent8, rsw8 * TPR8 + r, 32);
                    const int ok = koff8R<CN>(key, col8);
                    if (krow >= 0) {
                        cp_async_16(sKa + (uint32_t)ok, K + krow * k_row + (long)hk * k_head + col8);
                        cp_async_16(sVsa + (uint32_t)(key * HDV + col8), V + krow * v_row + (long)hk * v_head + col8);
                    } else {
                        *reinterpret_cast<uint4*>(sKp + ok) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(sVstp + key * HDV + col8) = make_uint4(0, 0, 0, 0);
                    }
                }
            } else {
                // Fallback (e.g. hd96): naive 1-thread-per-key gather.
                const int key = nb * CN + pt;
                const bool kval = (pt < CN) && (key < seqlen_k);
                if (pt < CN) {
                    long krow;
                    if (page_table) {
                        int blk = kval ? page_table[(long)bi * page_stride + key / block_size] : 0;
                        krow = (long)blk * block_size + (kval ? (key % block_size) : 0);
                    } else {
                        krow = (long)(cu_k ? cu_k[bi] : 0) + (kval ? key : 0);
                    }
                    const __nv_fp8_e4m3* ksrc = K + krow * k_row + (long)hk * k_head;
                    const __nv_fp8_e4m3* vsrc = V + krow * v_row + (long)hk * v_head;
                    __nv_fp8_e4m3* sKp = sK0 + st * KSTRIDE;
                    __nv_fp8_e4m3* sVstp = sVst0 + st * VstSTRIDE;
                    gather_row_k8_cp<HD, CN>(sKp, (uint32_t)__cvta_generic_to_shared(sKp), pt, ksrc, kval);
                    gather_row_plain8_cp<HDV>(sVstp, sVst_a + (uint32_t)(st * VstSTRIDE), pt, vsrc, kval);
                }
            }
            cp_async_mbar_arrive(&loaded[st]);
        };
        if (n_cnt > 0) issue_load(0);
        for (int li = 0; li < n_cnt; ++li) {
            const int st = li % STAGES;
            if (li + 1 < n_cnt) issue_load(li + 1);          // prefetch next tile while we transpose this one
            { int fp = (li / STAGES) & 1; while (!mbar_try_wait(&loaded[st], fp)) {} }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V-staging cp.async -> smem reads
            if (pt < CN) transpose_v_smem8<HDV>(sV0 + st * VSTRIDE, sVst0 + st * VstSTRIDE, pt);  // CN keys
            mbar_arrive(&full[st]);                          // K (cp.async, loaded) + Vt (transposed) ready
        }
        return;
    }

    // ---------- consumers ----------
    consumer_regs();
    const int cwg = wg - 1;
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const int qpos_lo = ((row0 + row_lo) / rep) + delta, qpos_hi = ((row0 + row_hi) / rep) + delta;
    const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(cwg * 8192);
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t sP_wg = (uint32_t)__cvta_generic_to_shared(sP) + (uint32_t)(cwg * 8192);
    const int barid = 1 + cwg;

    float acc_o[HDV / 2];
#pragma unroll
    for (int i = 0; i < HDV / 2; ++i) acc_o[i] = 0.f;
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float s[CN / 2]; float sc[2];

    while (!mbar_try_wait(&barQ[0], 0)) {}
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    for (int li = 0; li < n_cnt; ++li) {
        const int nb = n_lo + li;
        const int st = li % STAGES;
        const int fp = (li / STAGES) & 1;
        while (!mbar_try_wait(&full[st], fp)) {}
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        qk_fp8_ss<HD, CN>(s, sQ_wg, sK_a + st * (uint32_t)KSTRIDE);
        wgmma_wait0();
        wg_fence_c(s);
        if constexpr (EXTRA) { if (softcap > 0.f) pf_softcap<CN>(s, sc_eff, softcap);
                               pf_mask_win<CN>(s, nb * CN, qpos_lo, qpos_hi, seqlen_k, causal, win_l, win_r, tg); }
        else                   pf_mask<CN>(s, nb * CN, qpos_lo, qpos_hi, seqlen_k, causal, tg);
        sc[0] = softmax_stats<CN>(s, m_r, l_r, 0, li, s2);
        sc[1] = softmax_stats<CN>(s, m_r, l_r, 1, li, s2);
        rescale_o_g(acc_o, sc);
        if constexpr (RS) {
            // P in registers -> no smem scatter / no cross-WG barrier / no proxy fence (the SS path's
            // 3 serialization points). Frees the tensor pipe to overlap and matches FA3's RS PV.
            uint32_t P16[CN / 8];
            build_P_rs_fp8<CN>(s, P16, tg, group);
            pv_fp8_rs<HDV, CN>(acc_o, P16, sV_a + st * (uint32_t)VSTRIDE);
        } else {
            scatter_P_fp8<CN>(s, sP_wg, warp, group, tg);
            wg_barrier(barid);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            pv_fp8_ss<HDV, CN>(acc_o, sP_wg, sV_a + st * (uint32_t)VSTRIDE);
        }
        wgmma_wait0();
        wg_fence_c(acc_o);
        mbar_arrive(&empty[st]);
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] = quad_sum(l_r[r]);
        bool bad = (l_r[r] == 0.f || l_r[r] != l_r[r]);
        float inv = bad ? 0.f : 1.f / l_r[r];
        const float invvd = inv * vd;
        int idx = row0 + (r == 0 ? row_lo : row_hi);
        bool val = idx < Mrows;
        int token = idx / rep, qhead = hk * rep + idx % rep, qidx = q0 + token;
        float lse_v = bad ? -INFINITY : m_r[r] * sc_eff + __logf(l_r[r]);
        if (num_splits > 1) {
            const int hq = hkv * rep;
            float* op = Opart + (((long)split * total_q + qidx) * hq + qhead) * HDV;
#pragma unroll
            for (int c = 0; c < HDV / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                if (val) { op[dv0] = acc_o[c*4+r*2+0] * invvd; op[dv0+1] = acc_o[c*4+r*2+1] * invvd; }
            }
            if (tg == 0 && val) LSEpart[((long)split * hq + qhead) * total_q + qidx] = lse_v;
        } else {
            long obase = (long)qidx * o_row + (long)qhead * o_head;
#pragma unroll
            for (int c = 0; c < HDV / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * invvd, acc_o[c*4+r*2+1] * invvd);
                if (val) *reinterpret_cast<uint32_t*>(&O[obase + dv0]) = o;
            }
            if (tg == 0 && val) LSE[(long)qhead * total_q + qidx] = lse_v;
        }
    }
}

// ===================== persistent continuous-ring native fp8 prefill =====================
// fp8 analog of prefill_persist_kernel. K + V-staging now load via TMA (cp.async.bulk.tensor; fp8 is
// loaded as raw UINT8 -- no convert needed since it is already e4m3 in gmem), which offloads the
// per-tile gather from the LSU to the TMA engine and adds L2-promotion prefetch -- the fp8 dense
// config was long_scoreboard-bound on the cp.async gather. Q stays on the cp.async gather (one reload
// per work-item, cheap). V still transposes in-smem (fp8 PV needs a K-major B operand). Non-paged,
// num_splits==1; the per-tile prefill_fp8_kernel still covers paged/split/varlen (cp.async).
template <int HD, int HDV, bool EXTRA, int STAGES>
__global__ __launch_bounds__(NThreads, 1) void prefill_fp8_persist_kernel(
        const __nv_fp8_e4m3* __restrict__ Q, const __nv_fp8_e4m3* __restrict__ K,
        const __nv_fp8_e4m3* __restrict__ V, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
        const int* __restrict__ cu_q, const int* __restrict__ cu_k, const int* __restrict__ seqused_k,
        long q_row, long q_head, long k_row, long k_head, long v_row, long v_head, long o_row, long o_head,
        int hkv, int rep, int total_q,
        float scale, int causal, float softcap, int win_l, int win_r,
        const float* __restrict__ q_descale, const float* __restrict__ k_descale,
        const float* __restrict__ v_descale, int ds_b, int ds_h,
        int tiles_m, int batch, int* __restrict__ work_ctr,
        const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapVst) {
    constexpr int CN = CTA_N, CM = CTA_M;
    constexpr int HDQK = (HD < 128) ? 128 : HD;  // K/Q 128-wide so koff8's row*128 stride fits (see per-tile fp8)
    constexpr int KSTRIDE = CN * HDQK;         // one fp8 K tile (swizzled, elems=bytes)
    constexpr int VSTRIDE = HDV * CN;          // one transposed fp8 V tile
    constexpr int VstSTRIDE = CN * HDV;        // one PLAIN [key][dv] V staging tile
    constexpr int QBUF = 1;                    // single-Q reload (cp.async, hidden by the K/V ring)
    constexpr int SDEPTH = 2;                  // dynamic-scheduler work FIFO depth
    constexpr int SCHED_BAR = 3;               // producer scheduler bar.sync id (consumer WGs use 1,2)

    __nv_fp8_e4m3* sQ = reinterpret_cast<__nv_fp8_e4m3*>(pf_smem);
    __nv_fp8_e4m3* sK0 = sQ + QBUF * CM * HDQK;
    __nv_fp8_e4m3* sV0 = sK0 + STAGES * KSTRIDE;
    __nv_fp8_e4m3* sP  = sV0 + STAGES * VSTRIDE;
    __nv_fp8_e4m3* sVst0 = sP + CM * CN;
    uint64_t* bars = reinterpret_cast<uint64_t*>(sVst0 + STAGES * VstSTRIDE);
    uint64_t* full = bars;                      // [STAGES] count NumProd (K cp.async + Vt transposed)
    uint64_t* empty = bars + STAGES;            // [STAGES] count NumCons
    uint64_t* loaded = bars + 2 * STAGES;       // [STAGES] count NumProd (K + V-staging cp.async)
    uint64_t* barQf = bars + 3 * STAGES;        // [QBUF] Q ready (producer -> consumers), count NumProd
    uint64_t* barQe = bars + 3 * STAGES + QBUF; // [QBUF] Q free  (consumers -> producer), count NumCons
    uint64_t* schF = bars + 3 * STAGES + 2 * QBUF;          // [SDEPTH] work ready (producer -> consumers)
    uint64_t* schE = bars + 3 * STAGES + 2 * QBUF + SDEPTH; // [SDEPTH] slot free (consumers -> producer)
    int* schW = reinterpret_cast<int*>(bars + 3 * STAGES + 2 * QBUF + 2 * SDEPTH);  // [SDEPTH] grabbed work idx

    const int tid = threadIdx.x;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], NumProd); mbar_init(&empty[s], NumCons); mbar_init(&loaded[s], 1); }
#pragma unroll
        for (int s = 0; s < QBUF; ++s) { mbar_init(&barQf[s], NumProd); mbar_init(&barQe[s], NumCons); }
#pragma unroll
        for (int s = 0; s < SDEPTH; ++s) { mbar_init(&schF[s], 1); mbar_init(&schE[s], NumCons); }
    }
    __syncthreads();

    const int total_work = tiles_m * hkv * batch;
    const int wg = tid / WG;

    if (wg == 0) {
        // ---------- producer: cp.async Q/K (swizzled) + V (plain staging) -> in-smem transpose ----------
        producer_regs();
        const int pt = tid;
        const uint32_t sVst_a = (uint32_t)__cvta_generic_to_shared(sVst0);
        const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
        int g = 0, witer = 0;
        for (int it = 0; ; ++it) {
            const int sslot = it % SDEPTH;
            if (pt == 0 && it >= SDEPTH) { int ep = ((it / SDEPTH) - 1) & 1; while (!mbar_try_wait(&schE[sslot], ep)) {} }
            if (pt == 0) schW[sslot] = atomicAdd(work_ctr, 1);
            asm volatile("bar.sync %0, %1;\n" :: "r"(SCHED_BAR), "r"(WG) : "memory");
            const int work = schW[sslot];
            if (pt == 0) { __threadfence_block(); mbar_arrive(&schF[sslot]); }
            if (work >= total_work) break;
            PfTile t = pf_work<CN, CM, EXTRA>(work, tiles_m, hkv, rep, cu_q, cu_k, seqused_k,
                                              causal, win_l, win_r, scale, q_descale, k_descale, v_descale, ds_b, ds_h);
            if (!t.valid) continue;
            const int bi = pf_bi(work, tiles_m, hkv);
            const long ybase = (long)(cu_k ? cu_k[bi] : 0);            // non-paged: absolute key row of req bi
            const int xk = t.hk * HD, xv = t.hk * HDV;
            // TMA-load K (swizzled koff8) + V-staging (plain) for GLOBAL tile gg (key tile nb) into stage
            // gg%STAGES. pt==0 issues both boxes against loaded[st] (one expect_tx covers K+Vst bytes).
            auto issue_load = [&](int gg, int nb) {
                const int st = gg % STAGES;
                if (gg >= STAGES) { int ep = ((gg / STAGES) - 1) & 1; while (!mbar_try_wait(&empty[st], ep)) {} }
                if (pt == 0) {
                    const int yk = (int)ybase + nb * CN;
                    mbar_expect_tx(&loaded[st], (uint32_t)((CN * HD + CN * HDV)));   // 1-byte fp8 K + Vst
                    tma_box(sK0 + st * KSTRIDE, &mapK, xk, yk, &loaded[st]);         // [CN,128] -> koff8(key,hd)
                    tma_box(sVst0 + st * VstSTRIDE, &mapVst, xv, yk, &loaded[st]);   // [CN,128] plain [key][dv]
                }
            };
            // (1) prime the item's first K/V tile (continuous ring) before the Q-free wait.
            issue_load(g, t.n_lo);
            // (2) Q into the single buffer: wait the consumer freed it, reload, signal.
            const int qslot = witer % QBUF;
            if (witer >= QBUF) { int p = ((witer / QBUF) - 1) & 1; while (!mbar_try_wait(&barQe[qslot], p)) {} }
            {
                const int idx = t.row0 + pt;
                const bool qval = idx < t.Mrows;
                const int token = qval ? idx / rep : 0;
                const int qhead = t.hk * rep + (qval ? idx % rep : 0);
                const __nv_fp8_e4m3* qsrc = Q + (long)(t.q0 + token) * q_row + (long)qhead * q_head;
                gather_row_k8_cp<HD>(sQ + qslot * (CM * HD),
                                     (uint32_t)__cvta_generic_to_shared(sQ + qslot * (CM * HD)), pt, qsrc, qval);
                cp_async_mbar_arrive(&barQf[qslot]);
            }
            // (3) remaining K/V tiles flood in behind Q; transpose each staged tile into sVt.
            for (int li = 0; li < t.n_cnt; ++li) {
                if (li + 1 < t.n_cnt) issue_load(g + li + 1, t.n_lo + li + 1);   // prefetch next tile
                const int gg = g + li, st = gg % STAGES, fp = (gg / STAGES) & 1;
                while (!mbar_try_wait(&loaded[st], fp)) {}
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // V-staging cp.async -> smem reads
                transpose_v_smem8<HDV>(sV0 + st * VSTRIDE, sVst0 + st * VstSTRIDE, pt);
                mbar_arrive(&full[st]);                                          // K (cp.async) + Vt (transposed) ready
            }
            g += t.n_cnt;
            witer++;
        }
        return;
    }

    // ---------- consumers ----------
    consumer_regs();
    const int cwg = wg - 1;
    const int wtid = tid % WG, warp = wtid >> 5, lane = wtid & 31;
    const int group = lane >> 2, tg = lane & 3;
    const int row_lo = cwg * 64 + warp * 16 + group, row_hi = row_lo + 8;
    const uint32_t sK_a = (uint32_t)__cvta_generic_to_shared(sK0);
    const uint32_t sV_a = (uint32_t)__cvta_generic_to_shared(sV0);
    const uint32_t sP_base = (uint32_t)__cvta_generic_to_shared(sP) + (uint32_t)(cwg * 8192);
    const int barid = 1 + cwg;

    int g = 0, witer = 0;
    for (int it = 0; ; ++it) {
        const int sslot = it % SDEPTH;
        { int p = (it / SDEPTH) & 1; while (!mbar_try_wait(&schF[sslot], p)) {} }
        const int work = schW[sslot];
        __threadfence_block();
        mbar_arrive(&schE[sslot]);
        if (work >= total_work) break;
        PfTile t = pf_work<CN, CM, EXTRA>(work, tiles_m, hkv, rep, cu_q, cu_k, seqused_k,
                                          causal, win_l, win_r, scale, q_descale, k_descale, v_descale, ds_b, ds_h);
        if (!t.valid) {                              // fully-masked / empty / padded item -> O=0, LSE=-inf
            emit_zero_o<__nv_bfloat16, HDV>(O, LSE, tid, t.row0, t.Mrows, t.hk, rep, t.q0, total_q, o_row, o_head);
            continue;
        }
        const int gbase = g;
        g += t.n_cnt;
        const int qpos_lo = ((t.row0 + row_lo) / rep) + t.delta, qpos_hi = ((t.row0 + row_hi) / rep) + t.delta;

        float acc_o[HDV / 2];
#pragma unroll
        for (int i = 0; i < HDV / 2; ++i) acc_o[i] = 0.f;
        float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
        float s[64]; float sc[2];

        const int qslot = witer % QBUF;
        const uint32_t sQ_wg = (uint32_t)__cvta_generic_to_shared(sQ) + (uint32_t)(qslot * CM * HD + cwg * 8192);
        { int p = (witer / QBUF) & 1; while (!mbar_try_wait(&barQf[qslot], p)) {} }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        for (int li = 0; li < t.n_cnt; ++li) {
            const int nb = t.n_lo + li, gg = gbase + li;
            const int st = gg % STAGES, fp = (gg / STAGES) & 1;
            while (!mbar_try_wait(&full[st], fp)) {}
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            qk_fp8_ss<HD>(s, sQ_wg, sK_a + st * (uint32_t)KSTRIDE);
            wgmma_wait0();
            wg_fence_c(s);
            if constexpr (EXTRA) { if (softcap > 0.f) pf_softcap(s, t.sc_eff, softcap);
                                   pf_mask_win(s, nb * CN, qpos_lo, qpos_hi, t.seqlen_k, causal, win_l, win_r, tg); }
            else                   pf_mask(s, nb * CN, qpos_lo, qpos_hi, t.seqlen_k, causal, tg);
            sc[0] = softmax_stats(s, m_r, l_r, 0, li, t.s2);
            sc[1] = softmax_stats(s, m_r, l_r, 1, li, t.s2);
            rescale_o_g(acc_o, sc);
            scatter_P_fp8(s, sP_base, warp, group, tg);
            wg_barrier(barid);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            pv_fp8_ss<HDV>(acc_o, sP_base, sV_a + st * (uint32_t)VSTRIDE);
            wgmma_wait0();
            wg_fence_c(acc_o);
            mbar_arrive(&empty[st]);
        }
        // free this Q buffer (all qk reads done) so the producer can refill it for item witer+QBUF.
        mbar_arrive(&barQe[qslot]);

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            l_r[r] = quad_sum(l_r[r]);
            bool bad = (l_r[r] == 0.f || l_r[r] != l_r[r]);
            float inv = bad ? 0.f : 1.f / l_r[r];
            const float invvd = inv * t.vd;
            int idx = t.row0 + (r == 0 ? row_lo : row_hi);
            bool val = idx < t.Mrows;
            int token = idx / rep, qhead = t.hk * rep + idx % rep, qidx = t.q0 + token;
            float lse_v = bad ? -INFINITY : m_r[r] * t.sc_eff + __logf(l_r[r]);
            long obase = (long)qidx * o_row + (long)qhead * o_head;
#pragma unroll
            for (int c = 0; c < HDV / 8; ++c) {
                int dv0 = 8 * c + 2 * tg;
                uint32_t o = pack_bf16(acc_o[c*4+r*2+0] * invvd, acc_o[c*4+r*2+1] * invvd);
                if (val) *reinterpret_cast<uint32_t*>(&O[obase + dv0]) = o;
            }
            if (tg == 0 && val) LSE[(long)qhead * total_q + qidx] = lse_v;
        }
        witer++;
    }
}

// Build a 2D TMA tensor map over a paged KV cache [num_blocks, block_size, hkv, hd] viewed as
// [num_blocks*block_size (rows), hkv*hd (cols)], box [block_size, 64] with 128B swizzle. The fast
// path loads each page as 2 such boxes (cols [0,64),[64,128) of the head) -> composes byte-exactly
// into the 128-row swizzled tile the wgmma consumer reads (see tma_probe16.cu).
template <typename T> static constexpr CUtensorMapDataType tmap_dtype();
template <> constexpr CUtensorMapDataType tmap_dtype<__nv_bfloat16>() { return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; }
template <> constexpr CUtensorMapDataType tmap_dtype<__half>()        { return CU_TENSOR_MAP_DATA_TYPE_FLOAT16; }
template <> constexpr CUtensorMapDataType tmap_dtype<__nv_fp8_e4m3>() { return CU_TENSOR_MAP_DATA_TYPE_UINT8; }

// fp8 (1-byte) contiguous TMA map over [total_rows, heads, hd] viewed as [heads*hd, total_rows]. box
// [128, rows]: 128 fp8 = one 128-Byte swizzle row, so a single box loads a full hd128 K/Q tile into
// the koff8 layout (Swizzle<3,4,3>) the wgmma descriptor reads. swz=128B when `sw`, else NONE (the
// plain V-staging tile that transpose_v_smem8 consumes row-major). Heads contiguous (stride(1)==hd).
static CUtensorMap make_fp8_contig(const at::Tensor& t, int rows_box, bool sw) {
    CUtensorMap map{};
    int64_t total = t.size(0), heads = t.size(1), hd = t.size(2);
    uint64_t gdim[2]    = {(uint64_t)(heads * hd), (uint64_t)total};
    uint64_t gstride[1] = {(uint64_t)(t.stride(0))};                 // bytes per row (1-byte elems)
    uint32_t bdim[2]    = {128u, (uint32_t)rows_box};
    uint32_t estride[2] = {1u, 1u};
    CUresult cr = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(),
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        sw ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled (fp8) failed: ", (int)cr);
    return map;
}
// fp8 (1-byte) PAGED TMA map over [num_blocks, block_size, heads, hd] viewed as [heads*hd,
// num_blocks*block_size]. box [128, block_size]: a CN-key tile loads CN/block_size such page boxes
// (each at a page-table-looked-up block row) that stack into the koff8 tile. swz=128B (K) / NONE (Vst).
static CUtensorMap make_fp8_paged(const at::Tensor& t, int block_size, bool sw) {
    CUtensorMap map{};
    int64_t num_blocks = t.size(0), heads = t.size(2), hd = t.size(3);
    uint64_t gdim[2]    = {(uint64_t)(heads * hd), (uint64_t)(num_blocks * block_size)};
    uint64_t gstride[1] = {(uint64_t)(t.stride(1))};                 // bytes per (blk,key) row (1-byte)
    uint32_t bdim[2]    = {128u, (uint32_t)block_size};
    uint32_t estride[2] = {1u, 1u};
    CUresult cr = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(),
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        sw ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled (fp8 paged) failed: ", (int)cr);
    return map;
}
template <typename T>
static CUtensorMap make_kv_map(const at::Tensor& t, int block_size) {
    CUtensorMap map{};
    int64_t num_blocks = t.size(0), hkv = t.size(2), hd = t.size(3);
    uint64_t gdim[2]    = {(uint64_t)(hkv * hd), (uint64_t)(num_blocks * block_size)};
    uint64_t gstride[1] = {(uint64_t)(t.stride(1) * (int64_t)sizeof(T))};   // bytes per (blk,key) row
    uint32_t bdim[2]    = {64u, (uint32_t)block_size};
    uint32_t estride[2] = {1u, 1u};
    CUresult cr = cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 2, t.data_ptr(),
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)cr);
    return map;
}

// Build a 2D TMA tensor map over a CONTIGUOUS (non-paged) KV cache [total_k, hkv, hd] viewed as
// [total_k (rows), hkv*hd (cols)], box [cn, 64] with 128B swizzle. Mirrors make_kv_map but with no
// page blocking: one [cn,64] box loads a full CN-row key tile per head col-block, landing at the SAME
// swizzled smem offset (bx*CN*64) the wgmma consumer reads. A single [cn,64] box is byte-identical to
// the paged path's CN/block_size stacked [block_size,64] boxes because the 128B swizzle period is 8
// rows and the consumer tile starts at swizzle phase 0 (validated equivalence: koff over CTA_N rows).
// Requires heads contiguous (stride(1)==hd) so the hkv*hd column flatten (col = hk*hd + d) is dense.
template <typename T>
static CUtensorMap make_kv_map_contig(const at::Tensor& t, int cn) {
    CUtensorMap map{};
    int64_t total_k = t.size(0), hkv = t.size(1), hd = t.size(2);
    uint64_t gdim[2]    = {(uint64_t)(hkv * hd), (uint64_t)total_k};
    uint64_t gstride[1] = {(uint64_t)(t.stride(0) * (int64_t)sizeof(T))};   // bytes per key row
    uint32_t bdim[2]    = {64u, (uint32_t)cn};
    uint32_t estride[2] = {1u, 1u};
    CUresult cr = cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 2, t.data_ptr(),
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled (contig) failed: ", (int)cr);
    return map;
}

// Build a 3D TMA tensor map over packed-GQA Q [total_q, hq, hd], box [64, rep, CM/rep]. A single box
// per (col-block bx, head-group hk, m_block) loads CM/rep tokens x rep heads x 64 cols into the SAME
// swizzled [CM,64] tile the wgmma consumer reads: the box's row-major smem order (tok*rep+head)*64+d
// equals koffR<CM>(tok*rep+head,d)'s pre-swizzle term, and SWIZZLE_128B is applied on the smem address
// (period 8 rows) so it lands byte-exact -- this is the GQA analogue of the rep==1 contiguous Q box
// (validated by correctness). Requires rep | CM and heads contiguous (stride(2)==1, stride(1)==hd).
template <typename T>
static CUtensorMap make_q_map_gqa(const at::Tensor& q, int rep, int cm) {
    CUtensorMap map{};
    int64_t total_q = q.size(0), hq = q.size(1), hd = q.size(2);
    uint64_t gdim[3]    = {(uint64_t)hd, (uint64_t)hq, (uint64_t)total_q};            // inner -> outer
    uint64_t gstride[2] = {(uint64_t)(q.stride(1) * (int64_t)sizeof(T)),              // bytes per head
                           (uint64_t)(q.stride(0) * (int64_t)sizeof(T))};             // bytes per token
    uint32_t bdim[3]    = {64u, (uint32_t)rep, (uint32_t)(cm / rep)};
    uint32_t estride[3] = {1u, 1u, 1u};
    CUresult cr = cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 3, q.data_ptr(),
        gdim, gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled (gqa-q) failed: ", (int)cr);
    return map;
}

// NOTE (2026-07): per-page TMA for block_size==1 (a [64,1] box per key) was tried and is 1.6x SLOWER
// than the cp.async gather -- 128 tiny 128B TMA boxes/tile let TMA per-descriptor overhead dominate,
// which is exactly why the block_size%8==0 gate below exists. The coalesced cp.async gather (in the
// prefill_kernel producer) is the better page_size=1 feed and is what fixed this family.

template <typename TIn, typename TC, int HD, int HDV, bool TMA, bool EXTRA, int STAGES = WS_STAGES, int CN = CTA_N,
          int CM = CTA_M, int MINB = 1>
static void launch_prefill_t(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                             at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                             const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                             const std::optional<at::Tensor>& page_table, const CUtensorMap& mapK,
                             const CUtensorMap& mapV, int hkv, int rep, int total_q, int block_size,
                             int page_stride, int tiles_m, int b, double scale, bool causal,
                             double softcap, int win_l, int win_r,
                             long q_row, long q_head, long k_row, long k_head, long v_row, long v_head,
                             long o_row, long o_head, int num_splits, float* Opart, float* LSEpart,
                             const float* qd, const float* kd, const float* vd, int ds_b, int ds_h,
                             const float* s_aux, const int* leftpad_k,
                             cudaStream_t stream) {
    const bool paged = page_table.has_value();
    constexpr int HDsm = ((HD + 63) / 64) * 64, HDVsm = ((HDV + 63) / 64) * 64;
    constexpr int NTHR = WG * (1 + CM / 64);   // 1 producer WG + CM/64 consumer WGs

    // Persistent continuous-ring prefill: collapses the per-tile grid's ~8.7 waves into 1 (recovering
    // the 14%->18.7% inter-wave SM-residency gap that bounds the big-head / long-context dense configs).
    // The lever that makes it WIN (vs the earlier single/double-Q gather attempts that regressed): Q is
    // reloaded per item via ONE async TMA (rep==1 contiguous), fast enough that the STAGES-deep K/V ring
    // hides it -> single-Q fits the full ring (no double-Q smem squeeze that forced STAGES=2 and starved
    // the pipe). Scoped to rep==1 + contiguous Q/KV: the rep>1 cp.async-gather Q reload serializes on the
    // single-Q handshake and lost to the per-tile grid (gqa 0.82->0.78, mqa 0.82->0.77), so GQA/MQA/
    // paged/varlen stay on the per-tile grid.
    constexpr int PSTAGES = STAGES;
    constexpr bool PF_PERSIST = true;
    // CM==64 persist is enabled ONLY for big heads (HD>=192): the hd256 dense-prefill wide-CN path uses
    // a half-height Q tile so a wider K/V tile fits at STAGES=3 (the ping-pong is off for HD>128 anyway).
    // Scoping to HD>=192 keeps the CM=64 decode-split path (hd<=128, which also calls launch_prefill_t)
    // on its per-tile grid -- only the non-paged single-split big-head prefill reaches this branch.
    if constexpr (PF_PERSIST && HD != 256 && (CM == 128 || (CM == 64 && HD >= 192)) && TMA && std::is_same<TIn, TC>::value) {
        static int num_sm = 0;
        if (num_sm == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev); if (num_sm <= 0) num_sm = 132; }
        const long total_work = (long)tiles_m * hkv * b;
        const bool q_contig = (q.stride(2) == 1) && (q.stride(1) == (int64_t)HD);
        // rep==1 -> 2D contiguous Q box; packed-GQA (rep>1, rep|CM) -> 3D Q box. Both reload Q via one
        // async TMA so the persistent ring's per-item Q cost stays hidden (the gather reload regressed).
        const bool persist_q_ok = q_contig && (rep == 1 || (rep > 1 && (CM % rep == 0)));
        if (!paged && persist_q_ok && num_splits <= 1 && total_work >= 2L * num_sm) {
            CUtensorMap mapQ = (rep == 1) ? make_kv_map_contig<TC>(q, CM) : make_q_map_gqa<TC>(q, rep, CM);
            constexpr int QBUF = 1, SDEPTH = 2;   // must match prefill_persist_kernel
            int smem_p = (int)((QBUF * CM * HDsm + PSTAGES * CN * HDsm + PSTAGES * CN * HDVsm) * (int)sizeof(TC))
                         + (2 * PSTAGES + 2 * QBUF + 2 * SDEPTH) * (int)sizeof(uint64_t) + SDEPTH * (int)sizeof(int);
            auto wctr = at::empty({1}, q.options().dtype(at::kInt));
            cudaMemsetAsync(wctr.data_ptr<int>(), 0, sizeof(int), stream);
            static bool attr_set_p = false;
            if (!attr_set_p) { cudaFuncSetAttribute(prefill_persist_kernel<TIn, TC, HD, HDV, TMA, EXTRA, PSTAGES, CN, CM, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p); attr_set_p = true; }
            const unsigned gx = (unsigned)(total_work < (long)num_sm ? total_work : (long)num_sm);
            prefill_persist_kernel<TIn, TC, HD, HDV, TMA, EXTRA, PSTAGES, CN, CM, MINB><<<dim3(gx, 1, 1), NTHR, smem_p, stream>>>(
                (const TIn*)q.data_ptr(), (const TIn*)k.data_ptr(), (const TIn*)v.data_ptr(),
                (TC*)out.data_ptr(), (float*)lse.data_ptr(),
                (const int*)cu_q.data_ptr(), cu_k ? (const int*)cu_k->data_ptr() : nullptr,
                seqused_k ? (const int*)seqused_k->data_ptr() : nullptr,
                paged ? (const int*)page_table->data_ptr() : nullptr,
                q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,
                hkv, rep, total_q, block_size, page_stride, (float)scale, causal ? 1 : 0,
                (float)softcap, win_l, win_r, qd, kd, vd, ds_b, ds_h, tiles_m, b,
                wctr.data_ptr<int>(), s_aux, mapK, mapV, mapQ);
            return;
        }
    }
    // Paged persistent prefill (page_size not TMA-composable, e.g. page_size=1): the SAME continuous-ring
    // kernel with the COOP cp.async gather producer (USE_TMA=false path in prefill_persist_kernel). Collapses
    // the per-tile grid's ~8 waves into one persistent wave, removing the per-work-item Q-load + pipeline-
    // prime bubble that pinned paged prefill at ~0.67x (FA3's persistent scheduler runs the mainloop at ~56%
    // compute vs our per-tile ~34%). HD==HDV==128 (COOP invariant); no TMA maps (Q + K/V both gather).
    if constexpr (PF_PERSIST && CM == 128 && HD == 128 && HDV == 128 && !TMA && std::is_same<TIn, TC>::value) {
        static int num_sm_c = 0;
        if (num_sm_c == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&num_sm_c, cudaDevAttrMultiProcessorCount, dev); if (num_sm_c <= 0) num_sm_c = 132; }
        const long total_work = (long)tiles_m * hkv * b;
        // Paged persistent prefill with TMA_Q (K/V stays cp.async COOP). Q is NOT paged, so it reloads via
        // ONE async TMA box (rep==1 2D / packed-GQA rep>1 3D) -- the same validated map the dense GQA persist
        // uses -- while K/V uses the USE_TMA=0 COOP gather. This is FA3's Use_TMA_Q!=Use_TMA_KV split: the
        // continuous ring amortizes the per-work-item prologue that caps the per-tile grid's light causal
        // CTAs, and the cheap TMA Q reload is why this finally BEATS per-tile (the COOP Q reload did not).
        const bool q_contig = (q.stride(2) == 1) && (q.stride(1) == (int64_t)HD);
        const bool tmaq_ok = q_contig && (rep == 1 || (CM % rep == 0));
        // Paged persistent prefill with TMA_Q + PPS=2 stages (FA3's EXACT config: Q single-buffered 32KB +
        // 2*K + 2*V = 160KB, K/V via cp.async COOP). The continuous ring amortizes the per-work-item
        // prologue that caps the per-tile grid's light causal CTAs; TMA_Q makes the per-item Q reload one
        // cheap async box. An earlier PSTAGES=3 persist lost (0.78-0.82) -- FA3 proves 2 stages suffice and
        // the 3rd stage's 64KB only added smem pressure without deepening useful prefetch.
        constexpr int PPS = 3;
        // DISABLED (`&& false`): the paged persist loses to the per-tile+LPT grid in every measured variant
        // (PPS=2 starves the cp.async ring 0.62; PPS=3 0.80; prefetched-scheduler 0.72 AND regressed the
        // dense persist it shares). Root cause (ncu): my dense-causal per-tile mainloop already BEATS FA3
        // (1.05x, 60-74% tensor), so there is no codegen ceiling; the paged residual is the 128-thread COOP
        // producer's per-light-causal-m-tile pipeline-fill prologue that FA3's persist amortizes but my
        // persist's barrier overhead (ncu 1.46 vs 0.44) negates. per-tile+LPT is the paged path. See
        // .experimental_stash/ for the prefetch-scheduler attempt.
        if (paged && tmaq_ok && num_splits <= 1 && total_work >= 2L * num_sm_c && false) {
            CUtensorMap mapQ = (rep == 1) ? make_kv_map_contig<TC>(q, CM) : make_q_map_gqa<TC>(q, rep, CM);
            constexpr int QBUF = 1, SDEPTH = 2;   // must match prefill_persist_kernel
            int smem_p = (int)((QBUF * CM * HDsm + PPS * CN * HDsm + PPS * CN * HDVsm) * (int)sizeof(TC))
                         + (2 * PPS + 2 * QBUF + 2 * SDEPTH) * (int)sizeof(uint64_t) + SDEPTH * (int)sizeof(int);
            auto wctr = at::empty({1}, q.options().dtype(at::kInt));
            cudaMemsetAsync(wctr.data_ptr<int>(), 0, sizeof(int), stream);
            static bool attr_set_pc = false;
            if (!attr_set_pc) { cudaFuncSetAttribute(prefill_persist_kernel<TIn, TC, HD, HDV, TMA, EXTRA, PPS, CN, CM, MINB, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p); attr_set_pc = true; }
            const unsigned gx = (unsigned)(total_work < (long)num_sm_c ? total_work : (long)num_sm_c);
            CUtensorMap zmap{};   // K/V unused (COOP gather); pass dummy grid-constant maps
            prefill_persist_kernel<TIn, TC, HD, HDV, TMA, EXTRA, PPS, CN, CM, MINB, /*TMA_Q=*/true><<<dim3(gx, 1, 1), NTHR, smem_p, stream>>>(
                (const TIn*)q.data_ptr(), (const TIn*)k.data_ptr(), (const TIn*)v.data_ptr(),
                (TC*)out.data_ptr(), (float*)lse.data_ptr(),
                (const int*)cu_q.data_ptr(), cu_k ? (const int*)cu_k->data_ptr() : nullptr,
                seqused_k ? (const int*)seqused_k->data_ptr() : nullptr,
                paged ? (const int*)page_table->data_ptr() : nullptr,
                q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,
                hkv, rep, total_q, block_size, page_stride, (float)scale, causal ? 1 : 0,
                (float)softcap, win_l, win_r, qd, kd, vd, ds_b, ds_h, tiles_m, b,
                wctr.data_ptr<int>(), s_aux, zmap, zmap, mapQ);
            return;
        }
    }

    int smem = (int)((CM * HDsm + STAGES * CN * HDsm + STAGES * CN * HDVsm) * (int)sizeof(TC))
               + ((HD == 256 ? 4 : 2) * STAGES + 1) * (int)sizeof(uint64_t);   // HD==256: separate K/V pipelines (2x barriers)
    static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(prefill_kernel<TIn, TC, HD, HDV, TMA, EXTRA, STAGES, CN, CM, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr_set = true; }
    dim3 grid((unsigned)(tiles_m * num_splits), (unsigned)hkv, (unsigned)b);
    prefill_kernel<TIn, TC, HD, HDV, TMA, EXTRA, STAGES, CN, CM, MINB><<<grid, NTHR, smem, stream>>>(
        (const TIn*)q.data_ptr(), (const TIn*)k.data_ptr(), (const TIn*)v.data_ptr(),
        (TC*)out.data_ptr(), (float*)lse.data_ptr(),
        (const int*)cu_q.data_ptr(), cu_k ? (const int*)cu_k->data_ptr() : nullptr,
        seqused_k ? (const int*)seqused_k->data_ptr() : nullptr,
        paged ? (const int*)page_table->data_ptr() : nullptr,
        q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,
        hkv, rep, total_q, block_size, page_stride, (float)scale, causal ? 1 : 0,
        (float)softcap, win_l, win_r, num_splits, Opart, LSEpart,
        qd, kd, vd, ds_b, ds_h, s_aux, leftpad_k, mapK, mapV);
}

template <typename TIn, typename TC>
static void launch_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                           at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                           const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                           const std::optional<at::Tensor>& page_table,
                           const std::optional<at::Tensor>& qd_, const std::optional<at::Tensor>& kd_,
                           const std::optional<at::Tensor>& vd_,
                           int b, int hq, int hkv, int hd, int hdv, int total_q, int max_seqlen_q,
                           double scale, bool causal, double softcap, int win_l, int win_r,
                           const float* s_aux, cudaStream_t stream) {
    const bool paged = page_table.has_value();
    int rep = hq / hkv;
    long q_row = q.stride(0), q_head = q.stride(1);
    long k_row, k_head, v_row, v_head;
    if (paged) { k_row = k.stride(1); k_head = k.stride(2); v_row = v.stride(1); v_head = v.stride(2); }
    else       { k_row = k.stride(0); k_head = k.stride(1); v_row = v.stride(0); v_head = v.stride(1); }
    long o_row = out.stride(0), o_head = out.stride(1);
    int block_size = paged ? (int)k.size(1) : 0;
    int page_stride = paged ? (int)page_table->stride(0) : 0;
    int tiles_m = (max_seqlen_q * rep + CTA_M - 1) / CTA_M;
    const float* qd = qd_ ? (const float*)qd_->data_ptr() : nullptr;
    const float* kd = kd_ ? (const float*)kd_->data_ptr() : nullptr;
    const float* vd = vd_ ? (const float*)vd_->data_ptr() : nullptr;
    int ds_b = qd_ ? (int)qd_->stride(0) : 0, ds_h = qd_ ? (int)qd_->stride(1) : 0;

    // TMA-paged loader: requires paged KV with a swizzle-composable page (block_size | CTA_N and a
    // multiple of 8 so per-page boxes land at the right swizzle phase) and a contiguous [.,hkv*hd]
    // row stride. Disabled for fp8 (TIn!=TC ⇒ producer must convert, TMA copies raw bytes).
    // TMA needs 64-col-aligned per-page boxes (KBOX=HD/64); head dims that aren't a multiple of 64
    // (e.g. 96, padded to 128 in smem) can't tile cleanly -> fall back to the cp.async gather. The
    // key tile CN need NOT be a 64-multiple: pages of [block_size,64] boxes stack at row offsets
    // p*block_size inside a CN-row col-block, so CN%block_size==0 (block_size%8==0) suffices -- BUT the
    // contiguous one-[CN,64]-box path needs CN to compose byte-exactly with the 128B swizzle, which only
    // the validated 48/64/128 do: CN=72 (hd192, would fit at STAGES=3) loads a wrong swizzle -> NaN.
    // hd192/hd256 use CN=64/48 at STAGES=3 (smem-maxed). FA3's wider CN=96/80 at STAGES=2 regressed
    // even on the persistent ring (hd256 0.83->0.59, hd192 0.89->0.76): the 256-deep big-head TMA loads
    // need the 3rd stage to hide latency, cold-start or not. The persist path's win comes from the wave
    // collapse + dynamic balance, not the tile width.
    // NOTE: a deep (CN=64, STAGES=5) hd128 ring for long contexts (large4k) was tried to fight the
    // K/V-handshake barrier stall (tensor 57% vs FA3 70%); it regressed hard (large4k 0.82->0.72). The
    // narrower QK (N=64) wgmma loses more tensor throughput than the extra 2 stages recover -- large4k
    // wants the WIDE CN=128 tile, and its gap to FA3 is barrier/compute overlap, not ring depth.
    // Windowed (SWA) hd128 narrows to CN=64 (see the hd128 dispatch branch below) -> the TMA contig
    // map must be built at the matching key-tile width, so fold the window into cn_eff here.
    const bool pf_windowed = (win_l >= 0) || (win_r >= 0);
    const int cn_eff = (hd == 256) ? 80 : (hd >= 192) ? 64 : (hd == 128 && pf_windowed) ? 64 : 128;
    const bool tma_dt_hd = std::is_same<TIn, TC>::value && (hd % 64 == 0) && (hdv % 64 == 0);
    bool use_tma_paged = tma_dt_hd &&
                   paged && block_size > 0 && (cn_eff % block_size == 0) && (block_size % 8 == 0) &&
                   (k.stride(0) == (int64_t)block_size * k.stride(1)) &&
                   (v.stride(0) == (int64_t)block_size * v.stride(1)) &&
                   k.stride(3) == 1 && v.stride(3) == 1;
    // TMA-contiguous loader (non-paged dense/varlen): a 2D map over [hkv*hd, total_k] feeds the wgmma
    // mainloop via one [CN,64] box per col-block -- TMA hides the load latency that the cp.async gather
    // exposes (FA3 uses Use_TMA_KV for non-paged too). Requires the head dim contiguous (stride(2)==1)
    // and heads packed (stride(1)==hd) so the hkv*hd column flatten is dense.
    bool use_tma_contig = tma_dt_hd && !paged &&
                   k.stride(2) == 1 && v.stride(2) == 1 &&
                   k.stride(1) == (int64_t)hd && v.stride(1) == (int64_t)hdv;
    bool use_tma = use_tma_paged || use_tma_contig;
    CUtensorMap mapK{}, mapV{};
    if (use_tma_paged) { mapK = make_kv_map<TC>(k, block_size); mapV = make_kv_map<TC>(v, block_size); }
    else if (use_tma_contig) { mapK = make_kv_map_contig<TC>(k, cn_eff); mapV = make_kv_map_contig<TC>(v, cn_eff); }

#define PF_LAUNCH(HD_, HDV_, ST_, CN_, CM_, TMA_, EXTRA_)                                               \
    launch_prefill_t<TIn, TC, HD_, HDV_, TMA_, EXTRA_, ST_, CN_, CM_>(q, k, v, out, lse, cu_q, cu_k, seqused_k, \
        page_table, mapK, mapV, hkv, rep, total_q, block_size, page_stride,                             \
        (max_seqlen_q * rep + (CM_) - 1) / (CM_), b, scale,                                             \
        causal, softcap, win_l, win_r, q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,      \
        1, nullptr, nullptr, qd, kd, vd, ds_b, ds_h, s_aux, nullptr, stream)
#define PF_DISPATCH(HD_, HDV_, ST_, CN_, CM_)                                                           \
    do {                                                                                                \
        bool extra = (softcap > 0.0) || (win_l >= 0) || (win_r >= 0);                                   \
        if (use_tma) { if (extra) PF_LAUNCH(HD_, HDV_, ST_, CN_, CM_, true, true);  else PF_LAUNCH(HD_, HDV_, ST_, CN_, CM_, true, false); }   \
        else         { if (extra) PF_LAUNCH(HD_, HDV_, ST_, CN_, CM_, false, true); else PF_LAUNCH(HD_, HDV_, ST_, CN_, CM_, false, false); }  \
    } while (0)
    if (hd == 64 && hdv == 64)        PF_DISPATCH(64, 64, WS_STAGES, 128, 128);
    else if (hd == 96 && hdv == 96)   PF_DISPATCH(96, 96, WS_STAGES, 128, 128);
    // hd128: the windowed (SWA) EXTRA mask path's extra live registers tip the wide CM=128/CN=128
    // consumer into spill-RELOADS (NCU: local spillLD 4608 -> long_scoreboard 37%, vs gqa EXTRA=false
    // spillLD 0). A narrow CN=64 halves the score-register footprint s[CN/2] -> no spill, and since the
    // 256-wide window leaves the tensor pipe only ~31% utilized, the narrower QK wgmma is ~free while
    // finer 64-key tiles waste less compute past the window edge. STAGES=4 (CN=64 smem fits 5; the
    // window is only ~4 tiles, so a 4-deep K/V ring already covers it). Non-windowed hd128 keeps the
    // wide CN=128 tile (its gap is compute/barrier overlap, not register pressure -- CN=64 regressed it).
    else if (hd == 128 && hdv == 128) {
        if (pf_windowed && use_tma_contig) PF_DISPATCH(128, 128, 4, 64, 128);
        else                               PF_DISPATCH(128, 128, WS_STAGES, 128, 128);
    }
    else if (hd == 192 && hdv == 192) PF_DISPATCH(192, 192, 3, 64, 128);
    // hd192 -> dv128 (FA3 flash_fwd_hdimdiff): QK over HD=192, PV over HDV=128 (acc_o[64]). The smaller
    // V accumulator frees smem for a deeper STAGES=4 K/V ring at CN=64 (hd192/hdv192 is stuck at STAGES=3).
    else if (hd == 192 && hdv == 128) PF_DISPATCH(192, 128, 4, 64, 128);
    // hd256 (~0.89x): NCU-CONFIRMED the gap is tensor DENSITY, not occupancy/registers/barriers. FA3 and
    // this kernel BOTH run 1 CTA/SM at 18.75% occupancy (168 regs + 213KB smem each cap at 1 block, 384
    // threads, 12 warps -- identical), and BOTH stall similarly per issue (barrier ~2.0-2.3, wgmma-wait
    // ~1.3-1.5, long_scoreboard ~1.0). FA3 still extracts 59% SM-compute vs our 46% because its m64n80
    // QK/PV wgmma packs 1.67x the work/issue of our m64n48 (fewer, wider, denser tiles). FA3's CN=80 needs
    // kStages=2 (Q 64KB + 2*(K+V) 160KB = 224KB); OUR STAGES=2 STARVES -- the producer can't hide the 80KB
    // load at 2 stages (measured CN=80/STAGES=2 = 0.75x, long_scoreboard 0.94->2.28, barrier 2.0->3.62;
    // release-before-empty did NOT fix it, 0.746). CN=80 is numerically correct (swizzle valid) but 2
    // stages is the wall: STAGES=3 hides the load but the smem budget then caps CN at 48 (56/64 overflow
    // 227KB), and CN=48's narrow QK is the 46% ceiling. FA3 hides the 2-stage load via CUTLASS's async TMA
    // pipeline (single wide swizzled box + producer-acquire scheduling); our hand-issued HD/64 serial boxes
    // from one lane cannot match its prefetch. CM=64 (1 WG, STAGES=4) = 0.72 (loses the 2nd warpgroup);
    // ping-pong = neutral (WG softmax stall isn't the bound). STAGES=3/CN=48 is the faithful floor.
    else if (hd == 256 && hdv == 256) PF_DISPATCH(256, 256, 2, 80, 128);   // SPLITKV per-tile: separate K/V rings hide the 80KB load at STAGES=2 -> wide CN=80
    else TORCH_CHECK(false, "launch_prefill: unsupported hd/hdv ", hd, "/", hdv);
#undef PF_DISPATCH
#undef PF_LAUNCH
}

// Native fp8 (e4m3) prefill launcher (output bf16). Mirrors launch_prefill but routes to
// prefill_fp8_kernel: no TMA, V is transposed in-kernel, P round-trips through smem for SS PV.
template <int HD, int HDV, bool EXTRA, int STAGES = WS_STAGES, int CN = CTA_N>
static void launch_prefill_fp8_t(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                 at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                                 const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                                 const std::optional<at::Tensor>& page_table, int hkv, int rep, int total_q,
                                 int block_size, int page_stride, int tiles_m, int b, double scale, bool causal,
                                 double softcap, int win_l, int win_r,
                                 long q_row, long q_head, long k_row, long k_head, long v_row, long v_head,
                                 long o_row, long o_head, int num_splits, float* Opart, float* LSEpart,
                                 const float* qd, const float* kd, const float* vd, int ds_b, int ds_h,
                                 cudaStream_t stream) {
    const bool paged = page_table.has_value();
    // Persistent continuous-ring fp8 prefill: collapses the per-tile grid's ~8.7 waves into 1 (the fp8
    // dense config's 38% long_scoreboard was the per-CTA cold-start that the streamed K/V ring hides).
    // (A deeper PSTG=4 ring fits at 1-byte fp8 -- 16384*14=224KB -- but only moved 0.817->0.820, so the
    // remaining gap is the SS-PV smem round-trip / V-transpose, not ring depth; kept at STAGES=3.)
    // Scoped to non-paged + num_splits==1 (paged/split/varlen stay on the per-tile grid below). The
    // persistent ring is hd128-only (its TMA box is hardcoded 128-wide); if constexpr keeps
    // prefill_fp8_persist_kernel<256> from being instantiated at all.
    if constexpr (HD == 128 && HDV == 128) if (!paged && num_splits <= 1) {
        static int num_sm = 0;
        if (num_sm == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev); if (num_sm <= 0) num_sm = 132; }
        const long total_work = (long)tiles_m * hkv * b;
        if (total_work >= 2L * num_sm) {
            constexpr int QBUF = 1, SDEPTH = 2;   // must match prefill_fp8_persist_kernel
            constexpr int HDQK = (HD < 128) ? 128 : HD;   // K/Q 128-wide (koff8 row*128); see fp8 kernel
            int smem_p = (int)(QBUF * CTA_M * HDQK + STAGES * CTA_N * HDQK + STAGES * HDV * CTA_N + CTA_M * CTA_N
                               + STAGES * CTA_N * HDV)
                         + (3 * STAGES + 2 * QBUF + 2 * SDEPTH) * (int)sizeof(uint64_t) + SDEPTH * (int)sizeof(int);
            auto wctr = at::empty({1}, q.options().dtype(at::kInt));
            cudaMemsetAsync(wctr.data_ptr<int>(), 0, sizeof(int), stream);
            // fp8 TMA maps: K swizzled (koff8, [128,CTA_N] box) + V-staging plain ([128,CTA_N], no swz).
            // Requires contiguous heads (stride(1)==hd). Falls back below if not TMA-eligible.
            const bool fp8_tma_ok = (HD == 128 && HDV == 128) &&
                                    (k.stride(2) == 1 && v.stride(2) == 1 &&
                                     k.stride(1) == (int64_t)HD && v.stride(1) == (int64_t)HDV);
            if (fp8_tma_ok) {
                CUtensorMap mapK = make_fp8_contig(k, CTA_N, true);
                CUtensorMap mapVst = make_fp8_contig(v, CTA_N, false);
                static bool attr_set_p = false;
                if (!attr_set_p) { cudaFuncSetAttribute(prefill_fp8_persist_kernel<HD, HDV, EXTRA, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_p); attr_set_p = true; }
                const unsigned gx = (unsigned)(total_work < (long)num_sm ? total_work : (long)num_sm);
                prefill_fp8_persist_kernel<HD, HDV, EXTRA, STAGES><<<dim3(gx, 1, 1), NThreads, smem_p, stream>>>(
                    (const __nv_fp8_e4m3*)q.data_ptr(), (const __nv_fp8_e4m3*)k.data_ptr(), (const __nv_fp8_e4m3*)v.data_ptr(),
                    (__nv_bfloat16*)out.data_ptr(), (float*)lse.data_ptr(),
                    (const int*)cu_q.data_ptr(), cu_k ? (const int*)cu_k->data_ptr() : nullptr,
                    seqused_k ? (const int*)seqused_k->data_ptr() : nullptr,
                    q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,
                    hkv, rep, total_q, (float)scale, causal ? 1 : 0, (float)softcap, win_l, win_r,
                    qd, kd, vd, ds_b, ds_h, tiles_m, b, wctr.data_ptr<int>(), mapK, mapVst);
                return;
            }
            // !fp8_tma_ok -> fall through to the per-tile cp.async path below.
        }
    }
    // + STAGES*CTA_N*HDV for the plain V-staging ring (TMA/cp.async target, transposed in-smem to sVt);
    // + STAGES extra barriers (loaded[]) for the K/V delivered signal.
    constexpr int HDQK_ = (HD < 128) ? 128 : HD;   // K/Q 128-wide (koff8 row*128); see prefill_fp8_kernel
    // Vt and sP are 128-key-padded (koff8R, key<128); sK and V-staging shrink with CN. CN=CTA_N reduces
    // to the original hd128 sizing exactly. RS-PV (HD==256) drops the sP tile (CTA_M*128), which is what
    // lets CN=128/STAGES=2 hd256 fit 224KB (matching FA3's kBlockN=128 fp8 tile).
    constexpr int sP_bytes = (HD == 256) ? 0 : (CTA_M * 128);
    int smem = (int)(CTA_M * HDQK_ + STAGES * CN * HDQK_ + STAGES * HDV * 128 + sP_bytes
                     + STAGES * CN * HDV)
               + (3 * STAGES + 1) * (int)sizeof(uint64_t);
    dim3 grid((unsigned)(tiles_m * num_splits), (unsigned)hkv, (unsigned)b);
    // TMA-eligible? paged: block_size composes with the 128B swizzle (|CTA_N, %8) + contiguous heads;
    // non-paged: contiguous heads. TMA offloads the per-key cp.async gather (long_scoreboard-bound) to
    // the TMA engine -- the same fix that took fp8 dense 0.84->1.0; fp8 decode is load-bound too.
    // The fp8 TMA box is hardcoded 128-wide (128 fp8 = one 128B swizzle row), so it is valid ONLY for
    // HD==HDV==128; a 64-wide head would over-read into the next head / past the tensor. hd64 fp8 falls
    // to the cp.async gather (loads exactly HD cols into the HDQK=128-padded koff8 tile).
    bool fp8_tma_ok = (HD == 128 && HDV == 128) && (paged
        ? (block_size > 0 && (CTA_N % block_size == 0) && (block_size % 8 == 0) &&
           k.stride(3) == 1 && v.stride(3) == 1 && k.stride(2) == (int64_t)HD && v.stride(2) == (int64_t)HDV)
        : (k.stride(2) == 1 && v.stride(2) == 1 && k.stride(1) == (int64_t)HD && v.stride(1) == (int64_t)HDV));
#define FP8K_LAUNCH(TMA_, MK_, MV_)                                                                      \
    do {                                                                                                \
        static bool as_##TMA_ = false;                                                                  \
        if (!as_##TMA_) { cudaFuncSetAttribute(prefill_fp8_kernel<HD, HDV, EXTRA, STAGES, TMA_, CN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); as_##TMA_ = true; } \
        prefill_fp8_kernel<HD, HDV, EXTRA, STAGES, TMA_, CN><<<grid, NThreads, smem, stream>>>(         \
            (const __nv_fp8_e4m3*)q.data_ptr(), (const __nv_fp8_e4m3*)k.data_ptr(), (const __nv_fp8_e4m3*)v.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), (float*)lse.data_ptr(),                                     \
            (const int*)cu_q.data_ptr(), cu_k ? (const int*)cu_k->data_ptr() : nullptr,                 \
            seqused_k ? (const int*)seqused_k->data_ptr() : nullptr,                                    \
            paged ? (const int*)page_table->data_ptr() : nullptr,                                       \
            q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,                                 \
            hkv, rep, total_q, block_size, page_stride, (float)scale, causal ? 1 : 0,                   \
            (float)softcap, win_l, win_r, num_splits, Opart, LSEpart, qd, kd, vd, ds_b, ds_h, MK_, MV_); \
    } while (0)
    if (fp8_tma_ok) {
        CUtensorMap mapK = paged ? make_fp8_paged(k, block_size, true)  : make_fp8_contig(k, CTA_N, true);
        CUtensorMap mapVst = paged ? make_fp8_paged(v, block_size, false) : make_fp8_contig(v, CTA_N, false);
        FP8K_LAUNCH(true, mapK, mapVst);
    } else {
        CUtensorMap z{};
        FP8K_LAUNCH(false, z, z);
    }
#undef FP8K_LAUNCH
}

static void launch_prefill_fp8(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                               at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                               const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                               const std::optional<at::Tensor>& page_table,
                               const std::optional<at::Tensor>& qd_, const std::optional<at::Tensor>& kd_,
                               const std::optional<at::Tensor>& vd_,
                               int b, int hq, int hkv, int hd, int hdv, int total_q, int max_seqlen_q,
                               double scale, bool causal, double softcap, int win_l, int win_r, cudaStream_t stream) {
    const bool paged = page_table.has_value();
    int rep = hq / hkv;
    long q_row = q.stride(0), q_head = q.stride(1);
    long k_row, k_head, v_row, v_head;
    if (paged) { k_row = k.stride(1); k_head = k.stride(2); v_row = v.stride(1); v_head = v.stride(2); }
    else       { k_row = k.stride(0); k_head = k.stride(1); v_row = v.stride(0); v_head = v.stride(1); }
    long o_row = out.stride(0), o_head = out.stride(1);
    int block_size = paged ? (int)k.size(1) : 0;
    int page_stride = paged ? (int)page_table->stride(0) : 0;
    int tiles_m = (max_seqlen_q * rep + CTA_M - 1) / CTA_M;
    const float* qd = qd_ ? (const float*)qd_->data_ptr() : nullptr;
    const float* kd = kd_ ? (const float*)kd_->data_ptr() : nullptr;
    const float* vd = vd_ ? (const float*)vd_->data_ptr() : nullptr;
    int ds_b = qd_ ? (int)qd_->stride(0) : 0, ds_h = qd_ ? (int)qd_->stride(1) : 0;
    bool extra = (softcap > 0.0) || (win_l >= 0) || (win_r >= 0);
    TORCH_CHECK((hd == 128 && hdv == 128) || (hd == 256 && hdv == 256),
                "native fp8 prefill: hd/hdv in {128,256}, got ", hd, "/", hdv);
#define PF8_LAUNCH(HD_, HDV_, ST_, CN_, EXTRA_)                                                         \
    launch_prefill_fp8_t<HD_, HDV_, EXTRA_, ST_, CN_>(q, k, v, out, lse, cu_q, cu_k, seqused_k, page_table, hkv, \
        rep, total_q, block_size, page_stride, tiles_m, b, scale, causal, softcap, win_l, win_r,        \
        q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head, 1, nullptr, nullptr,                \
        qd, kd, vd, ds_b, ds_h, stream)
    // hd256 native fp8: CN=128/STAGES=2 RS-PV (matches FA3 kBlockN=128). RS drops sP so sK+Vt+Vst+sQ =
    // 224KB fits; acc_o[128]+s[64]+P16[16] fit the 232-reg consumer. (Was CN=64 SS-PV at ~0.66x -- the
    // scatter+barrier round-trip + 2x tiles were the bottleneck; see build_P_rs_fp8/pv_fp8_rs.)
    if (hd == 256) { if (extra) PF8_LAUNCH(256, 256, 2, 128, true); else PF8_LAUNCH(256, 256, 2, 128, false); }
    else           { if (extra) PF8_LAUNCH(128, 128, WS_STAGES, 128, true); else PF8_LAUNCH(128, 128, WS_STAGES, 128, false); }
#undef PF8_LAUNCH
}
} // namespace pf

// ======================================================================================
// FAST PATH 2: decode (q_len small), flash-decoding with split-KV.
//
// Memory-bound regime (the QK/PV are tiny GEMVs; the bottleneck is streaming the paged KV cache).
// One WARP per (batch, kv_head, kv_split) work-item; 128-thread blocks pack 4 independent warps.
// A warp processes ALL packed-GQA rows (M = q_len * rep) of its (b,hk) over its key slice, with
// lane L owning the contiguous head-dim chunk [L*per, L*per+per) so K/V loads are 256 B coalesced and
// the QK dot is a single warp reduction. Each warp writes a NORMALIZED partial (O_s, lse_s) to scratch;
// `combine_kernel` does the across-split flash reduction. Matches the scalar fallback semantics
// (qk_scale = scale*qd*kd, softcap after scale, causal + sliding-window, q_abs = klen - sq + qpos).
namespace dec {
constexpr int WARP = 32, WARPS_PER_BLOCK = 4, DEC_THREADS = WARP * WARPS_PER_BLOCK;

__device__ __forceinline__ float wreduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One warp = one (b, hk, split). MAXM bounds the packed-GQA rows; per/perv = HD/32, HDV/32 dims/lane.
template <typename TIn, typename TOut, int HD, int HDV, int MAXM>
__global__ void decode_kernel(
        const TIn* __restrict__ Q, const TIn* __restrict__ K, const TIn* __restrict__ V,
        float* __restrict__ Opart, float* __restrict__ LSEpart,
        const int* __restrict__ cu_q, const int* __restrict__ cu_k, const int* __restrict__ seqused_k,
        const int* __restrict__ page_table,
        const float* __restrict__ q_descale, const float* __restrict__ k_descale, const float* __restrict__ v_descale,
        int ds_b, int ds_h,
        long q_row, long q_head, long k_row, long k_head, long v_row, long v_head,
        int batch, int hq, int hkv, int rep, int total_q, int block_size, int page_stride,
        int num_splits, float scale, float softcap, int win_l, int win_r, int causal) {
    constexpr int per = HD / 32, perv = HDV / 32;
    const int witem = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (witem >= num_splits * hkv * batch) return;
    // work-item -> (b, hk, split)
    const int split = witem % num_splits;
    int tmp = witem / num_splits;
    const int hk = tmp % hkv;
    const int b = tmp / hkv;
    // b is bounded by the launch (grid sized to batch); guard:
    const int q0 = cu_q[b];
    const int sq = cu_q[b + 1] - q0;
    if (sq <= 0) return;
    const int klen = seqused_k ? seqused_k[b] : (cu_k[b + 1] - cu_k[b]);
    const int M = sq * rep;

    const long ds_idx = (long)b * ds_b + (long)hk * ds_h;
    const float qd = q_descale ? q_descale[ds_idx] : 1.f;
    const float kd = k_descale ? k_descale[ds_idx] : 1.f;
    const float vd = v_descale ? v_descale[ds_idx] : 1.f;
    const float qk_scale = scale * qd * kd;

    // split's key range
    const int span = (klen + num_splits - 1) / num_splits;
    const int jlo = split * span;
    const int jhi = min(jlo + span, klen);

    float qreg[MAXM][per];
    float acc[MAXM][perv];
    float mrow[MAXM], lrow[MAXM];
    int qabs[MAXM];
#pragma unroll
    for (int mi = 0; mi < MAXM; ++mi) {
        if (mi >= M) break;
        const int qpos = mi / rep, r = mi % rep;
        const int head = hk * rep + r;
        const TIn* qp = Q + (long)(q0 + qpos) * q_row + (long)head * q_head;
#pragma unroll
        for (int e = 0; e < per; ++e) qreg[mi][e] = to_float<TIn>(qp[lane * per + e]);
#pragma unroll
        for (int e = 0; e < perv; ++e) acc[mi][e] = 0.f;
        mrow[mi] = -INFINITY; lrow[mi] = 0.f;
        qabs[mi] = klen - sq + qpos;
    }

    // Software-pipelined K/V load: issue key (j+1)'s gmem loads (including the dependent page_table
    // lookup) while computing on key j, so the load latency overlaps the (short) softmax compute.
    auto load_key = [&](int j, float* kx, float* vx) {
        long krow;
        if (page_table) { int blk = page_table[(long)b * page_stride + j / block_size]; krow = (long)blk * block_size + (j % block_size); }
        else            { krow = (long)cu_k[b] + j; }
        const TIn* kp = K + krow * k_row + (long)hk * k_head;
        const TIn* vp = V + krow * v_row + (long)hk * v_head;
#pragma unroll
        for (int e = 0; e < per; ++e)  kx[e] = to_float<TIn>(kp[lane * per + e]);
#pragma unroll
        for (int e = 0; e < perv; ++e) vx[e] = to_float<TIn>(vp[lane * perv + e]);
    };
    float kxA[per], vxA[perv], kxB[per], vxB[perv];
    if (jlo < jhi) load_key(jlo, kxA, vxA);
    for (int j = jlo; j < jhi; ++j) {
        const bool A = ((j - jlo) & 1) == 0;
        float* kx = A ? kxA : kxB; float* vx = A ? vxA : vxB;
        if (j + 1 < jhi) load_key(j + 1, A ? kxB : kxA, A ? vxB : vxA);   // prefetch next
#pragma unroll
        for (int mi = 0; mi < MAXM; ++mi) {
            if (mi >= M) break;
            const int qa = qabs[mi];
            bool masked = (causal && j > qa) || (win_l >= 0 && j < qa - win_l) || (win_r >= 0 && j > qa + win_r);
            float dot = 0.f;
#pragma unroll
            for (int e = 0; e < per; ++e) dot += qreg[mi][e] * kx[e];
            float s = wreduce_sum(dot) * qk_scale;
            if (softcap > 0.f) s = softcap * tanhf(s / softcap);
            if (masked) continue;
            float mnew = fmaxf(mrow[mi], s);
            float corr = __expf(mrow[mi] - mnew), p = __expf(s - mnew);
            lrow[mi] = lrow[mi] * corr + p;
#pragma unroll
            for (int e = 0; e < perv; ++e) acc[mi][e] = acc[mi][e] * corr + p * (vx[e] * vd);
            mrow[mi] = mnew;
        }
    }

#pragma unroll
    for (int mi = 0; mi < MAXM; ++mi) {
        if (mi >= M) break;
        const int qpos = mi / rep, r = mi % rep;
        const int head = hk * rep + r, qidx = q0 + qpos;
        const float inv = lrow[mi] > 0.f ? 1.f / lrow[mi] : 0.f;
        float* op = Opart + (((long)split * total_q + qidx) * hq + head) * HDV;
#pragma unroll
        for (int e = 0; e < perv; ++e) op[lane * perv + e] = acc[mi][e] * inv;
        if (lane == 0)
            LSEpart[((long)split * hq + head) * total_q + qidx] =
                (lrow[mi] > 0.f) ? (mrow[mi] + __logf(lrow[mi])) : -INFINITY;
    }
}

// Flash-combine the per-split normalized partials into the final O/LSE.
// The old kernel ran one WARP per (qidx,head) looping serially over splits, so for small
// batch*heads only a few dozen warps were resident -> the cross-split reduce was DRAM-latency
// starved (~40% of decode time, DRAM <2% on hd256). Here:
//   * grid = (total_q*hq, NB): each (qidx,head) is split into NB ~64-elem HDV tiles -> NB x more
//     resident blocks so the strided per-split O loads have enough warps to hide latency.
//   * warp 0 builds the per-split softmax weights in smem (num_splits <= 64).
//   * the split loop is unrolled x4 with 4 independent accumulators -> 4 O loads in flight/thread
//     (memory-level parallelism), the real lever on a latency-bound reduce.
constexpr int COMBINE_THREADS = 128;
constexpr int combine_nb(int hdv) { return hdv / 64 > 0 ? hdv / 64 : 1; }
template <typename TOut, int HDV>
__global__ __launch_bounds__(COMBINE_THREADS) void combine_kernel(
        const float* __restrict__ Opart, const float* __restrict__ LSEpart,
        TOut* __restrict__ O, float* __restrict__ LSE,
        int num_splits, int total_q, int hq, long o_row, long o_head) {
    constexpr int NB = combine_nb(HDV);
    constexpr int TILE = HDV / NB;
    const int witem = blockIdx.x;
    if (witem >= total_q * hq) return;
    const int qidx = witem / hq, head = witem % hq;
    const int tile = blockIdx.y, e0 = tile * TILE;
    const int tid = threadIdx.x;

    __shared__ float sw[64];          // per-split softmax weight (num_splits <= 64)
    __shared__ float s_inv, s_lse;
    if (tid < 32) {                   // warp 0: lmax -> weights -> denom
        float lmax = -INFINITY;
        for (int s = tid; s < num_splits; s += 32)
            lmax = fmaxf(lmax, LSEpart[((long)s * hq + head) * total_q + qidx]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
        float denom = 0.f;
        for (int s = tid; s < num_splits; s += 32) {
            float lse = LSEpart[((long)s * hq + head) * total_q + qidx];
            float w = (lmax > -INFINITY) ? __expf(lse - lmax) : 0.f;
            sw[s] = w;
            denom += w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) denom += __shfl_xor_sync(0xffffffffu, denom, o);
        if (tid == 0) { s_inv = denom > 0.f ? 1.f / denom : 0.f;
                        s_lse = denom > 0.f ? (lmax + __logf(denom)) : -INFINITY; }
    }
    __syncthreads();
    const float inv = s_inv;
    const float* base = Opart + (((long)qidx * hq + head) * HDV);   // + s*sstride + e
    const long sstride = (long)total_q * hq * HDV;
    TOut* o = O + (long)qidx * o_row + (long)head * o_head;
    // A zero-weight split contributes nothing -- but an EMPTY split (num_splits not dividing the
    // key-tile count -> a trailing split starts past the last tile) never writes its Opart slice, so
    // that memory is uninitialized at::empty (clean zeros on a fresh alloc, but NaN/inf when the
    // caching allocator recycles a dirty pool). w==0 must therefore yield exactly 0, NOT 0*Opart
    // (0*NaN=NaN). The select gates the garbage out; non-empty splits always store finite Opart.
    for (int e = e0 + tid; e < e0 + TILE; e += COMBINE_THREADS) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int s = 0;
        for (; s + 4 <= num_splits; s += 4) {
            float w0 = sw[s + 0], w1 = sw[s + 1], w2 = sw[s + 2], w3 = sw[s + 3];
            a0 += w0 != 0.f ? w0 * base[(long)(s + 0) * sstride + e] : 0.f;
            a1 += w1 != 0.f ? w1 * base[(long)(s + 1) * sstride + e] : 0.f;
            a2 += w2 != 0.f ? w2 * base[(long)(s + 2) * sstride + e] : 0.f;
            a3 += w3 != 0.f ? w3 * base[(long)(s + 3) * sstride + e] : 0.f;
        }
        float acc = (a0 + a1) + (a2 + a3);
        for (; s < num_splits; ++s) { float w = sw[s]; acc += w != 0.f ? w * base[(long)s * sstride + e] : 0.f; }
        o[e] = from_float<TOut>(acc * inv);
    }
    if (tid == 0 && tile == 0) LSE[(long)head * total_q + qidx] = s_lse;
}

template <typename TIn, typename TOut, int HD, int HDV, int MAXM>
static void launch_decode_t(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                            at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                            const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                            const std::optional<at::Tensor>& page_table,
                            const std::optional<at::Tensor>& qd, const std::optional<at::Tensor>& kd,
                            const std::optional<at::Tensor>& vd,
                            int b, int hq, int hkv, int total_q, int cfg_splits, double scale,
                            double softcap, int win_l, int win_r, bool causal, int max_klen, cudaStream_t stream) {
    const bool paged = page_table.has_value();
    int rep = hq / hkv;
    long q_row = q.stride(0), q_head = q.stride(1);
    long k_row, k_head, v_row, v_head;
    if (paged) { k_row = k.stride(1); k_head = k.stride(2); v_row = v.stride(1); v_head = v.stride(2); }
    else       { k_row = k.stride(0); k_head = k.stride(1); v_row = v.stride(0); v_head = v.stride(1); }
    long o_row = out.stride(0), o_head = out.stride(1);
    int block_size = paged ? (int)k.size(1) : 0;
    int page_stride = paged ? (int)page_table->stride(0) : 0;

    // num_splits: forced when cfg>0, else split aggressively for occupancy (this warp path is 1 warp /
    // work-item, so more splits = more resident warps). Don't split finer than 64 keys/split.
    int num_splits;
    if (cfg_splits > 0) num_splits = cfg_splits;
    else {
        int dev = 0; cudaGetDevice(&dev);
        int sms = 132; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        int base = hkv * b;
        int want = (8 * sms * WARPS_PER_BLOCK + base - 1) / (base > 0 ? base : 1);
        int cap = (max_klen + 63) / 64;
        num_splits = std::max(1, std::min(std::min(want, cap), 64));
    }

    int ds_b = qd ? (int)qd->stride(0) : 0, ds_h = qd ? (int)qd->stride(1) : 0;
    auto fopt = q.options().dtype(at::kFloat);
    at::Tensor Opart = at::empty({(long)num_splits, total_q, hq, HDV}, fopt);
    at::Tensor LSEpart = at::empty({(long)num_splits, hq, total_q}, fopt);

    int witems = num_splits * hkv * b;
    int blocks = (witems + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid((unsigned)blocks, 1, 1);
    decode_kernel<TIn, TOut, HD, HDV, MAXM><<<grid, DEC_THREADS, 0, stream>>>(
        (const TIn*)q.data_ptr(), (const TIn*)k.data_ptr(), (const TIn*)v.data_ptr(),
        (float*)Opart.data_ptr(), (float*)LSEpart.data_ptr(),
        (const int*)cu_q.data_ptr(), cu_k ? (const int*)cu_k->data_ptr() : nullptr,
        seqused_k ? (const int*)seqused_k->data_ptr() : nullptr,
        paged ? (const int*)page_table->data_ptr() : nullptr,
        qd ? (const float*)qd->data_ptr() : nullptr, kd ? (const float*)kd->data_ptr() : nullptr,
        vd ? (const float*)vd->data_ptr() : nullptr, ds_b, ds_h,
        q_row, q_head, k_row, k_head, v_row, v_head,
        b, hq, hkv, rep, total_q, block_size, page_stride, num_splits,
        (float)scale, (float)softcap, win_l, win_r, causal ? 1 : 0);

    combine_kernel<TOut, HDV><<<dim3((unsigned)(total_q * hq), (unsigned)combine_nb(HDV)), COMBINE_THREADS, 0, stream>>>(
        (const float*)Opart.data_ptr(), (const float*)LSEpart.data_ptr(),
        (TOut*)out.data_ptr(), (float*)lse.data_ptr(), num_splits, total_q, hq, o_row, o_head);
}

// Dispatch on (HD,HDV) and the packed-GQA row bound MAXM (rounded to 4/8/16 to bound registers).
template <typename TIn, typename TOut>
static void launch_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                          at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                          const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                          const std::optional<at::Tensor>& page_table,
                          const std::optional<at::Tensor>& qd, const std::optional<at::Tensor>& kd,
                          const std::optional<at::Tensor>& vd,
                          int b, int hq, int hkv, int hd, int hdv, int total_q, int max_q, int cfg_splits,
                          double scale, double softcap, int win_l, int win_r, bool causal, int max_klen,
                          cudaStream_t stream) {
    int rep = hq / hkv;
    int maxM = max_q * rep;
    int mm = maxM <= 4 ? 4 : (maxM <= 8 ? 8 : 16);
    TORCH_CHECK(maxM <= 16, "decode: packed-GQA rows ", maxM, " > 16 unsupported");
#define DEC_DISP(HD_, HDV_, MM_)                                                                     \
    launch_decode_t<TIn, TOut, HD_, HDV_, MM_>(q, k, v, out, lse, cu_q, cu_k, seqused_k, page_table, \
        qd, kd, vd, b, hq, hkv, total_q, cfg_splits, scale, softcap, win_l, win_r, causal, max_klen, stream)
#define DEC_BY_MM(HD_, HDV_) do { if (mm == 4) DEC_DISP(HD_, HDV_, 4); else if (mm == 8) DEC_DISP(HD_, HDV_, 8); else DEC_DISP(HD_, HDV_, 16); } while (0)
    if (hd == 128 && hdv == 128)      DEC_BY_MM(128, 128);
    else if (hd == 64 && hdv == 64)   DEC_BY_MM(64, 64);
    else if (hd == 96 && hdv == 96)   DEC_BY_MM(96, 96);
    else if (hd == 192 && hdv == 192) DEC_BY_MM(192, 192);
    else if (hd == 256 && hdv == 256) DEC_BY_MM(256, 256);
    else if (hd == 192 && hdv == 128) DEC_BY_MM(192, 128);
    else TORCH_CHECK(false, "decode: unsupported hd/hdv ", hd, "/", hdv);
#undef DEC_BY_MM
#undef DEC_DISP
}
} // namespace dec

// Decode via the prefill wgmma mainloop + split-KV (bf16, hd==hdv in {64,128}, no softcap/SWA).
// Allocates per-split scratch (normalized O + lse), launches prefill_kernel with grid.x = num_splits,
// then dec::combine_kernel reduces across splits. num_splits auto-fills ~2 waves of CTAs, capped by
// n_max (key tiles) so no split is wasted; LSEpart is pre-filled -inf so empty/early-out splits drop.
// Largest stage count in [1,DS_MAX_STAGES] whose Q + STAGES*(K+V) smem fits the Hopper dynamic-smem cap
// (~227KB), for a CN-wide key tile and HDsm/HDVsm-padded head dims (TC is 16-bit here). Decode uses a
// single consumer WG (cm=64) so Q smem halves -> deeper K/V pipelines fit, hiding more load latency.
static constexpr int DS_MAX_STAGES = 5;
// occ = target CTAs/SM: occ=2 halves the smem budget so two CTAs co-reside (2x memory-level
// parallelism for DRAM-latency-bound decode). Returns the deepest pipeline that fits.
static constexpr int ds_fit_stages(int cn, int hdsm, int hdvsm, int cm = pf::CTA_M, int occ = 1) {
    const int per = (cn * hdsm + cn * hdvsm) * 2;
    const int qb  = cm * hdsm * 2;
    const int budget = (227 * 1024 - 64) / occ;
    int s = DS_MAX_STAGES;
    while (s > 1 && qb + s * per > budget) --s;
    return s;
}

template <typename TIn, typename TC, int HD, int HDV>
static void launch_decode_split(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                                const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                                const std::optional<at::Tensor>& page_table,
                                const std::optional<at::Tensor>& qd_, const std::optional<at::Tensor>& kd_,
                                const std::optional<at::Tensor>& vd_,
                                int b, int hq, int hkv, int total_q, int max_q, int max_klen, int cfg_splits,
                                double scale, bool causal, double softcap, int win_l, int win_r,
                                const int* leftpad_k, cudaStream_t stream) {
    using namespace pf;
    const bool paged = page_table.has_value();
    int rep = hq / hkv;
    long q_row = q.stride(0), q_head = q.stride(1);
    long k_row, k_head, v_row, v_head;
    if (paged) { k_row = k.stride(1); k_head = k.stride(2); v_row = v.stride(1); v_head = v.stride(2); }
    else       { k_row = k.stride(0); k_head = k.stride(1); v_row = v.stride(0); v_head = v.stride(1); }
    long o_row = out.stride(0), o_head = out.stride(1);
    int block_size = paged ? (int)k.size(1) : 0;
    int page_stride = paged ? (int)page_table->stride(0) : 0;
    const float* qd = qd_ ? (const float*)qd_->data_ptr() : nullptr;
    const float* kd = kd_ ? (const float*)kd_->data_ptr() : nullptr;
    const float* vd = vd_ ? (const float*)vd_->data_ptr() : nullptr;
    int ds_b = qd_ ? (int)qd_->stride(0) : 0, ds_h = qd_ ? (int)qd_->stride(1) : 0;

    // Key-tile width (DCN) + STAGES, chosen per EFFECTIVE context length. hd128 has two regimes:
    //   wide (DCN=128, STAGES=3): moderate ctx -> wgmma n-width + fewer/meatier splits win;
    //   deep (DCN=64,  STAGES=5): long ctx     -> deeper K/V pipeline hides DRAM latency.
    // Larger head dims are smem-forced narrow (192->64, 256->48) so wide==deep (single instantiation).
    constexpr int DHDsm = ((HD + 63) / 64) * 64, DHDVsm = ((HDV + 63) / 64) * 64;
    // hd256: non-windowed (deep) wants wider CN=64 (fewer/meatier tiles amortize the wasted-M
    // softmax; STAGES=3 still fits smem); windowed+softcap (gemma2, wide) keeps CN=48/STAGES=4
    // (the deeper K/V pipeline hides the extra tanh/mask latency better than wider tiles).
    constexpr int DCN_W = (HD == 256) ? 48 : (HD == 192) ? 64 : CTA_N;   // hd128 -> 128
    // hd128 long-ctx decode: FA3 (PagedKVManager, use_one_mma_wg) uses kBlockN=128 and hits ~89% DRAM BW
    // at the same 1-CTA/SM occupancy; our earlier CN=64/STAGES=5 "deep" pipeline sat at ~41% -- 2x more
    // tiles means 2x the per-tile pipeline-sync / online-softmax-rescale / m64n64 (vs m64n128) wgmma
    // overhead on the memory-latency-bound path. CN=128/STAGES=3 keeps MORE bytes in flight (3*128 vs
    // 5*64 keys) with half the tiles, so it hides DRAM latency better. (deep==wide for hd128 now.)
    constexpr int DCN_D = (HD == 256) ? 64 : (HD == 192) ? 64 : CTA_N;   // hd128 -> 128 (match FA3 kBlockN)
    // Deep path occupancy. NCU shows the deep decode is occupancy/latency-bound, NOT bandwidth-bound:
    // ~8% achieved occupancy (1 consumer WG == 4 active warps), DRAM only ~56% peak, top stalls
    // wait(wgmma)+long-scoreboard(loads). Adding warps that do WASTED-M work does not help: a 2nd
    // consumer WG (CM=128) regresses (it computes 64 dead rows that contend for issue slots), and a
    // 2nd CTA (DEEP2) costs pipeline depth on hd128. DEEP2 (2 CTA/SM) is kept ONLY for the compute-
    // heavier HD>=192 small-value decode (diffv: enough real wgmma work to overlap a 2nd CTA).
    // DCM=64 => single consumer warpgroup (decode is tiny-M; a 2nd WG would be fully masked).
    constexpr int DCM = 64;
    constexpr int DST_D2 = ds_fit_stages(DCN_D, DHDsm, DHDVsm, DCM, 2);  // 2 CTA/SM
    constexpr bool DEEP2 = (HDV <= 128) && (HD >= 192) && (DST_D2 >= 2);
    constexpr int DCM_DEEP = DCM, DCM_W = DCM;
    constexpr int DST_D = DEEP2 ? DST_D2 : ds_fit_stages(DCN_D, DHDsm, DHDVsm, DCM, 1);
    constexpr int MINB_D = DEEP2 ? 2 : 1;
    constexpr int DST_W = ds_fit_stages(DCN_W, DHDsm, DHDVsm, DCM);
    // Deep (narrow-CN, STAGES=5) for plain non-windowed hd128 -> hides DRAM latency. Windowed (SWA)
    // hd128 uses the wide path: split-KV over a sliding window leaves the deep pipeline mostly unfilled
    // (few in-window tiles/split), so wider tiles + GPU-filling split count measure better.
    // "deep" = the DRAM-latency-hiding regime (4-wave split target + >=2-tiles/split pipeline-fill
    // guard). hd128 long-ctx gets it via the narrow DCN (DCN_W!=DCN_D); the big head dims (>=192) are
    // ALREADY smem-forced narrow (DCN_W==DCN_D) and run 1 CTA/SM, so they want the same treatment
    // (fill the 4-5 stage K/V pipeline + spread enough CTAs across SMs). Windowed stays wide.
    const bool deep = ((DCN_W != DCN_D) || (HD >= 192)) && (win_l < 0) && (win_r < 0);
    const int DCN = deep ? DCN_D : DCN_W;
    const int DCM_RUN = deep ? DCM_DEEP : DCM_W;
    int tiles_m = (max_q * rep + DCM_RUN - 1) / DCM_RUN;
    // n_max key-tiles (causal: last query attends all keys -> ceil(klen/CN)); cap splits to it.
    int n_max_est = (max_klen + DCN - 1) / DCN;
    int dev = 0; cudaGetDevice(&dev);
    int sms = 132; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int base = tiles_m * hkv * b;   // resident CTAs (hkv*batch tiles) before splitting
    int num_splits;
    if (cfg_splits > 0) num_splits = cfg_splits;
    else {
        // Wave-quantization-aware split count. An NCU split sweep across the whole decode family
        // shows the optimum is the LARGEST split count that still fits in ~ONE occupancy wave
        // (base*splits ~ sms*bps): splitting past one wave adds a partial-wave tail (those SMs cost
        // a FULL wave of latency) plus per-split Q-load + combine overhead, which strictly regresses
        // -- e.g. gemma2 0.99@2 -> 0.79@3 -> 0.82@5, blk32 0.98@2 -> 0.80@12. The old 2-4 wave target
        // therefore over-split every shape. bps = co-resident CTAs/SM (DEEP2 packs 2/SM, else 1/SM),
        // so one wave is sms*bps CTAs. n_max_est (below) clamps to available key-tiles.
        int bps = deep ? MINB_D : 1;
        int want = std::max(1, (sms * bps) / (base > 0 ? base : 1));
        num_splits = std::min(want, deep ? 48 : 32);
    }
    if (num_splits > n_max_est) num_splits = std::max(1, n_max_est);

    bool use_tma = std::is_same<TIn, TC>::value && (HD % 64 == 0) && (HDV % 64 == 0) &&
                   paged && block_size > 0 && (DCN % block_size == 0) && (block_size % 8 == 0) &&
                   (k.stride(0) == (int64_t)block_size * k.stride(1)) &&
                   (v.stride(0) == (int64_t)block_size * v.stride(1)) &&
                   k.stride(3) == 1 && v.stride(3) == 1;
    CUtensorMap mapK{}, mapV{};
    if (use_tma) { mapK = make_kv_map<TC>(k, block_size); mapV = make_kv_map<TC>(v, block_size); }

    const bool extra = (softcap > 0.0) || (win_l >= 0) || (win_r >= 0) || (leftpad_k != nullptr);
#define DS_LAUNCH(TMA_, EXTRA_, DCN_, DST_, DCM_, MINB_, NS_, OP_, LP_)                                 \
    launch_prefill_t<TIn, TC, HD, HDV, TMA_, EXTRA_, DST_, DCN_, DCM_, MINB_>(q, k, v, out, lse, cu_q, cu_k, seqused_k, page_table, \
        mapK, mapV, hkv, rep, total_q, block_size, page_stride, tiles_m, b, scale, causal,              \
        softcap, win_l, win_r, q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head, NS_, OP_, LP_, \
        qd, kd, vd, ds_b, ds_h, nullptr, leftpad_k, stream)
#define DS_BY_TMA(EXTRA_, DCN_, DST_, DCM_, MINB_, NS_, OP_, LP_)                                       \
    do { if (use_tma) DS_LAUNCH(true, EXTRA_, DCN_, DST_, DCM_, MINB_, NS_, OP_, LP_);                  \
         else         DS_LAUNCH(false, EXTRA_, DCN_, DST_, DCM_, MINB_, NS_, OP_, LP_); } while (0)
#define DS_BY_EXTRA(NS_, OP_, LP_)                                                                      \
    do { if (deep) { if (extra) DS_BY_TMA(true, DCN_D, DST_D, DCM_DEEP, MINB_D, NS_, OP_, LP_); else DS_BY_TMA(false, DCN_D, DST_D, DCM_DEEP, MINB_D, NS_, OP_, LP_); } \
         else      { if (extra) DS_BY_TMA(true, DCN_W, DST_W, DCM_W, 1, NS_, OP_, LP_); else DS_BY_TMA(false, DCN_W, DST_W, DCM_W, 1, NS_, OP_, LP_); } } while (0)

    // NOTE (2026-07): 2-CTA/SM for tiny-M long-ctx hd128 decode was tried (env FA3_DEC2CTA) and is
    // NEUTRAL -- the decode is DRAM-latency-bound at ~40% BW, and halving STAGES (5->3) to fit a 2nd
    // CTA cancels the extra-warps benefit. Splitting past one wave also regresses (non-persistent tail).
    // The win for this family came from the coalesced K/V gather (see prefill_kernel producer).
    if (num_splits <= 1) {                  // single split -> write final O/LSE directly (no combine)
        DS_BY_EXTRA(1, nullptr, nullptr);
        return;
    }

    // LSEpart is left UNINITIALIZED (at::empty): empty/masked/padded split CTAs stamp their own -inf
    // via emit_empty_lse, so we skip the per-call at::full memset (an extra fill kernel + launch).
    auto fopt = q.options().dtype(at::kFloat);
    at::Tensor Opart = at::empty({(long)num_splits, total_q, hq, HDV}, fopt);
    at::Tensor LSEpart = at::empty({(long)num_splits, hq, total_q}, fopt);
    float* opd = (float*)Opart.data_ptr(); float* lpd = (float*)LSEpart.data_ptr();
    DS_BY_EXTRA(num_splits, opd, lpd);
#undef DS_BY_EXTRA
#undef DS_BY_TMA
#undef DS_LAUNCH

    dec::combine_kernel<TC, HDV><<<dim3((unsigned)(total_q * hq), (unsigned)dec::combine_nb(HDV)), dec::COMBINE_THREADS, 0, stream>>>(
        opd, lpd, (TC*)out.data_ptr(), (float*)lse.data_ptr(), num_splits, total_q, hq, o_row, o_head);
}

// Native fp8 (e4m3) decode via split-KV: same scratch/combine flow as launch_decode_split, but the
// per-split work runs on prefill_fp8_kernel (native fp8 wgmma, no convert pass). Output bf16.
template <int HD, int HDV>
static void launch_decode_split_fp8(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                    at::Tensor& out, at::Tensor& lse, const at::Tensor& cu_q,
                                    const std::optional<at::Tensor>& cu_k, const std::optional<at::Tensor>& seqused_k,
                                    const std::optional<at::Tensor>& page_table,
                                    const std::optional<at::Tensor>& qd_, const std::optional<at::Tensor>& kd_,
                                    const std::optional<at::Tensor>& vd_,
                                    int b, int hq, int hkv, int total_q, int max_q, int max_klen, int cfg_splits,
                                    double scale, bool causal, double softcap, int win_l, int win_r, cudaStream_t stream) {
    using namespace pf;
    const bool paged = page_table.has_value();
    int rep = hq / hkv;
    long q_row = q.stride(0), q_head = q.stride(1);
    long k_row, k_head, v_row, v_head;
    if (paged) { k_row = k.stride(1); k_head = k.stride(2); v_row = v.stride(1); v_head = v.stride(2); }
    else       { k_row = k.stride(0); k_head = k.stride(1); v_row = v.stride(0); v_head = v.stride(1); }
    long o_row = out.stride(0), o_head = out.stride(1);
    int block_size = paged ? (int)k.size(1) : 0;
    int page_stride = paged ? (int)page_table->stride(0) : 0;
    int tiles_m = (max_q * rep + CTA_M - 1) / CTA_M;
    const float* qd = qd_ ? (const float*)qd_->data_ptr() : nullptr;
    const float* kd = kd_ ? (const float*)kd_->data_ptr() : nullptr;
    const float* vd = vd_ ? (const float*)vd_->data_ptr() : nullptr;
    int ds_b = qd_ ? (int)qd_->stride(0) : 0, ds_h = qd_ ? (int)qd_->stride(1) : 0;

    // hd256 fp8 decode: CN=128/STAGES=2 (RS-PV dropped sP so 128-wide K/Vt/Vst + Q fit 224KB); wider CN
    // halves the tile count on the memory-bound decode (mirrors the bf16 decode CN=64->128 win).
    constexpr int FCN = (HD == 256) ? 128 : CTA_N;
    constexpr int FST = (HD == 256) ? 2 : WS_STAGES;   // hd256 fp8 decode: STAGES=2
    int n_max_est = (max_klen + FCN - 1) / FCN;
    int num_splits;
    if (cfg_splits > 0) num_splits = cfg_splits;
    else {
        int dev = 0; cudaGetDevice(&dev);
        int sms = 132; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        int base = tiles_m * hkv * b;   // resident CTAs before splitting (fp8 decode runs 1 CTA/SM)
        // Wave-quantization-aware: target the largest split count within ~one wave (sms CTAs); see
        // launch_decode_split for the NCU sweep rationale (over-splitting past one wave regresses).
        int want = std::max(1, sms / (base > 0 ? base : 1));
        num_splits = std::min(want, 32);
    }
    if (num_splits > n_max_est) num_splits = std::max(1, n_max_est);

    const bool extra = (softcap > 0.0) || (win_l >= 0) || (win_r >= 0);
#define DS8(NS_, OP_, LP_)                                                                              \
    do { if (extra) launch_prefill_fp8_t<HD, HDV, true, FST, FCN>(q, k, v, out, lse, cu_q, cu_k, seqused_k, \
                page_table, hkv, rep, total_q, block_size, page_stride, tiles_m, b, scale, causal,       \
                softcap, win_l, win_r, q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,       \
                NS_, OP_, LP_, qd, kd, vd, ds_b, ds_h, stream);                                          \
         else      launch_prefill_fp8_t<HD, HDV, false, FST, FCN>(q, k, v, out, lse, cu_q, cu_k, seqused_k, \
                page_table, hkv, rep, total_q, block_size, page_stride, tiles_m, b, scale, causal,       \
                softcap, win_l, win_r, q_row, q_head, k_row, k_head, v_row, v_head, o_row, o_head,       \
                NS_, OP_, LP_, qd, kd, vd, ds_b, ds_h, stream); } while (0)
    if (num_splits <= 1) { DS8(1, nullptr, nullptr); return; }
    // LSEpart uninitialized (at::empty): empty splits stamp -inf via emit_empty_lse (see launch_decode_split).
    auto fopt = q.options().dtype(at::kFloat);
    at::Tensor Opart = at::empty({(long)num_splits, total_q, hq, HDV}, fopt);
    at::Tensor LSEpart = at::empty({(long)num_splits, hq, total_q}, fopt);
    float* opd = (float*)Opart.data_ptr(); float* lpd = (float*)LSEpart.data_ptr();
    DS8(num_splits, opd, lpd);
#undef DS8
    dec::combine_kernel<__nv_bfloat16, HDV><<<dim3((unsigned)(total_q * hq), (unsigned)dec::combine_nb(HDV)), dec::COMBINE_THREADS, 0, stream>>>(
        opd, lpd, (__nv_bfloat16*)out.data_ptr(), (float*)lse.data_ptr(), num_splits, total_q, hq, o_row, o_head);
}

// ---- host -----------------------------------------------------------------------------
static inline const void* dptr(const at::Tensor& t) { return t.data_ptr(); }

template <typename TIn, typename TOut>
static void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   at::Tensor& out, at::Tensor& lse,
                   const at::Tensor& cu_q, const std::optional<at::Tensor>& cu_k,
                   const std::optional<at::Tensor>& seqused_k, const std::optional<at::Tensor>& page_table,
                   const std::optional<at::Tensor>& qd, const std::optional<at::Tensor>& kd, const std::optional<at::Tensor>& vd,
                   int b, int hq, int hkv, int hd, int hdv, int total_q,
                   double scale, double softcap, int win_l, int win_r, bool causal, cudaStream_t stream) {
  Params<TIn, TOut> p{};
  p.q = (const TIn*)dptr(q); p.k = (const TIn*)dptr(k); p.v = (const TIn*)dptr(v);
  p.o = (TOut*)out.data_ptr(); p.lse = (float*)lse.data_ptr();
  p.cu_q = (const int*)cu_q.data_ptr();
  p.cu_k = cu_k ? (const int*)cu_k->data_ptr() : nullptr;
  p.seqused_k = seqused_k ? (const int*)seqused_k->data_ptr() : nullptr;
  p.page_table = page_table ? (const int*)page_table->data_ptr() : nullptr;
  p.page_stride = page_table ? (int)page_table->stride(0) : 0;
  p.block_size = (page_table) ? (int)k.size(1) : 0;
  p.q_descale = qd ? (const float*)qd->data_ptr() : nullptr;
  p.k_descale = kd ? (const float*)kd->data_ptr() : nullptr;
  p.v_descale = vd ? (const float*)vd->data_ptr() : nullptr;
  // Descale strides IN ELEMENTS (broadcast-safe: a (b,hkv) tensor with stride [0,0] -> 0,0).
  p.ds_b = qd ? (int)qd->stride(0) : 0;
  p.ds_h = qd ? (int)qd->stride(1) : 0;
  p.q_row = q.stride(0); p.q_head = q.stride(1);
  // k/v paged (nblocks, block, hkv, hd): index krow over flattened (blk*block_size+off),
  // so row stride = stride(1), head stride = stride(2). Non-paged: row stride(0), head stride(1).
  if (page_table) { p.k_row = k.stride(1); p.k_head = k.stride(2); p.v_row = v.stride(1); p.v_head = v.stride(2); }
  else            { p.k_row = k.stride(0); p.k_head = k.stride(1); p.v_row = v.stride(0); p.v_head = v.stride(1); }
  p.o_row = out.stride(0); p.o_head = out.stride(1);
  p.b = b; p.hq = hq; p.hkv = hkv; p.hd = hd; p.hdv = hdv; p.total_q = total_q;
  p.scale = (float)scale; p.softcap = (float)softcap; p.win_l = win_l; p.win_r = win_r; p.causal = causal;
  long pairs = (long)total_q * hq;
  int threads = 128; long warps_per_block = threads / 32;
  long blocks = (pairs + warps_per_block - 1) / warps_per_block;
  attn_kernel<TIn, TOut><<<(unsigned)blocks, threads, 0, stream>>>(p);
}

static int mha_num_sm() {
  static int num_sm = 0;
  if (num_sm == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev); if (num_sm <= 0) num_sm = 132; }
  return num_sm;
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v,
    std::optional<at::Tensor> k_new, std::optional<at::Tensor> v_new, std::optional<at::Tensor> q_v,
    std::optional<at::Tensor> out_,
    std::optional<at::Tensor> cu_seqlens_q, std::optional<at::Tensor> cu_seqlens_k, std::optional<at::Tensor> cu_seqlens_k_new,
    std::optional<at::Tensor> seqused_q, std::optional<at::Tensor> seqused_k,
    std::optional<int64_t> max_seqlen_q, std::optional<int64_t> max_seqlen_k,
    std::optional<at::Tensor> page_table, std::optional<at::Tensor> kv_batch_idx, std::optional<at::Tensor> leftpad_k,
    std::optional<at::Tensor> rotary_cos, std::optional<at::Tensor> rotary_sin, std::optional<at::Tensor> seqlens_rotary,
    std::optional<at::Tensor> q_descale, std::optional<at::Tensor> k_descale, std::optional<at::Tensor> v_descale,
    double softmax_scale, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    double softcap, bool is_rotary_interleaved,
    std::optional<at::Tensor> scheduler_metadata, int64_t num_splits, std::optional<bool> pack_gqa, int64_t sm_margin,
    std::optional<at::Tensor> s_aux, int64_t cp_world_size, int64_t cp_rank, std::optional<at::Tensor> cp_tot_seqused_k) {
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  int total_q = (int)q.size(0);
  int hq = (int)q.size(1);
  int hd = (int)q.size(2);
  bool paged = page_table.has_value();
  int hkv = paged ? (int)k.size(2) : (int)k.size(1);
  int hdv = paged ? (int)v.size(3) : (int)v.size(2);
  int b = (int)cu_seqlens_q->size(0) - 1;

  bool is_fp8 = q.scalar_type() == at::kFloat8_e4m3fn;
  auto out_dtype = is_fp8 ? at::kBFloat16 : q.scalar_type();
  at::Tensor out = out_.has_value() ? *out_
                 : at::empty({total_q, hq, hdv}, q.options().dtype(out_dtype));
  at::Tensor lse = at::empty({hq, total_q}, q.options().dtype(at::kFloat));

  int win_l = (int)window_size_left, win_r = (int)window_size_right;

  // Attention sink (s_aux): per-(query)head logit added to the softmax denominator. Converted to a
  // float scratch (kept alive until mha_fwd returns; freed only after the same-stream kernel runs, so
  // the async launch is safe). cache_leftpad (leftpad_k): per-batch count of leading KV tokens to skip.
  at::Tensor s_aux_f;
  const float* s_aux_ptr = nullptr;
  if (s_aux.has_value()) { s_aux_f = s_aux->to(at::kFloat).contiguous(); s_aux_ptr = (const float*)s_aux_f.data_ptr(); }
  const int* leftpad_ptr = leftpad_k.has_value() ? (const int*)leftpad_k->data_ptr() : nullptr;
  bool causal = is_causal;

  // FAST PATH routing: two wgmma paths cover EVERY supported shape -- the split-KV decode kernel for
  // small/medium packed-M decode/append/spec (q <= 32 AND q*rep <= 64, i.e. one DCM=64 M-tile) and the
  // prefill kernel for everything else (larger q, incl. q >= 64 prompts and rep > 16 past 64 rows).
  // BOTH implement softcap + sliding-window via the EXTRA template, so neither is gated at dispatch.
  // `can_prefill` is defined AFTER the decode predicate below so it can defer to it (the prefill kernel
  // is correct for small q too, but split-KV decode parallelizes long context that prefill cannot).
  bool hd_ok = ((hd == hdv) && (hd == 64 || hd == 96 || hd == 128 || hd == 192 || hd == 256)) ||
               (hd == 192 && hdv == 128);  // FA3 flash_fwd_hdimdiff: d=192 -> dv=128 (prefill + decode)
  bool dt16_ok = (q.scalar_type() == at::kBFloat16) || (q.scalar_type() == at::kHalf);  // bf16/fp16 share layout
  bool dtwg_ok = dt16_ok || is_fp8;  // wgmma fast path: bf16/fp16 native, fp8 via convert->bf16 smem + descale

  // DECODE fast path: small q_len (decode/append/spec/medium-q). Flash-decoding + split-KV + combine.
  // Generic over dtype (bf16/fp16/fp8 descale-folded), GQA/MQA, softcap, sliding-window, paged.
  // The split-KV kernel runs the prefill mainloop with DCM=64 (one consumer warpgroup), so it packs
  // up to 64 q*rep rows into ONE M-tile -- the lever it adds over prefill is splitting a long KV
  // context across CTAs (+ combine), which is exactly what a small-q / long-context request needs and
  // the single-CTA prefill path cannot do. Cap at max_q<=32 (keep large-q prompts q>=64 on the CM=128
  // prefill path) and q*rep<=64 (one DCM tile, right-sized -- CM=128 would waste a warpgroup on masked
  // rows). Requires contiguous head-dim.
  bool dt_ok = (q.scalar_type() == at::kBFloat16) || (q.scalar_type() == at::kHalf) || is_fp8;
  bool dhd_ok = (hd % 32 == 0) && (hdv % 32 == 0) &&
                ((hd == 64 && hdv == 64) || (hd == 96 && hdv == 96) || (hd == 128 && hdv == 128) ||
                 (hd == 192 && hdv == 192) || (hd == 256 && hdv == 256) || (hd == 192 && hdv == 128));
  bool contig_hd = q.stride(2) == 1 &&
                   (paged ? (k.stride(3) == 1 && v.stride(3) == 1) : (k.stride(2) == 1 && v.stride(2) == 1));
  int rep = (hq % hkv == 0) ? hq / hkv : 0;
  int max_q = max_seqlen_q.has_value() ? (int)max_seqlen_q.value() : total_q;
  bool can_decode = dt_ok && dhd_ok && contig_hd && (hq % hkv == 0) &&
                    max_q >= 1 && max_q <= 32 && (max_q * rep) <= 64;
  int max_klen = max_seqlen_k.has_value() ? (int)max_seqlen_k.value() : 0;
  // FAST decode: the wgmma prefill mainloop + split-KV (reuses TMA/async-pipelined loads, softcap+SWA).
  // Limited to what that kernel supports (bf16/fp16/fp8-convert, hd==hdv in {64,128}); rest use warp path.
  bool can_decode_fast = can_decode && dtwg_ok &&
                         (((hd == hdv) && (hd == 64 || hd == 96 || hd == 128 || hd == 192 || hd == 256)) ||
                          (hd == 192 && hdv == 128));
  // Prefill wgmma path: ANY q >= 1 that the small-q decode path does not claim. Covers q >= 64 prompts,
  // the medium-q 5..63 band, and rep > 16 at small q -- the prefill kernel packs q*rep into its M tiles
  // (token = row/rep, qhead = hk*rep + row%rep) with no <=16-row cap, so arbitrary q/rep are correct.
  // softcap + sliding-window are handled by launch_prefill's EXTRA dispatch. Deferring via
  // !can_decode_fast keeps every existing decode config on the (faster) decode kernel -> no regression.
  bool can_prefill = dtwg_ok && hd_ok && (hq % hkv == 0) &&
                     max_seqlen_q.has_value() && max_seqlen_q.value() >= 1 && !can_decode_fast;

  // UNDERFILLED prefill -> split-KV decode. The CM=128 prefill grid is tiles_m*hkv*b CTAs (one wave,
  // num_splits=1); when batch*heads*m-tiles is small over a LONG context that leaves most SMs idle and
  // exposes the full DRAM latency (e.g. q120/k4224: 2*8*4 = 64 CTAs on 132 SMs -> 0.48 wave, ~0.81x).
  // The split-KV decode path runs the SAME prefill mainloop at CM=64 (2x m-tiles), picks a wave-filling
  // split count, and uses the deep STAGES=5 K/V pipeline -- exactly the long-context latency-hiding the
  // single-wave prefill lacks. Gated to the decode-path head dims + long ctx + modest q so well-filled
  // prefills (dense batch>=2, q>=256, or short ctx) are untouched (they measure better on CM=128).
  int pf_tiles_m = (max_q * rep + 128 - 1) / 128;
  long pf_base = (long)pf_tiles_m * (long)hkv * (long)b;
  bool decode_caps = ((hd == hdv) && (hd == 64 || hd == 96 || hd == 128 || hd == 192 || hd == 256)) ||
                     (hd == 192 && hdv == 128);
  bool pf_underfill = can_prefill && dhd_ok && contig_hd && decode_caps && rep >= 1 &&
                      (pf_base * 2 <= (long)mha_num_sm()) && max_klen >= 2048 && max_q > 32 && max_q <= 256;

  if (total_q > 0 && can_prefill && !pf_underfill) {
    // descale: nullopt for bf16/fp16 (vLLM passes a stride-0 broadcast that must be IGNORED); real
    // q/k/v_descale for fp8 (folded: qd*kd into the softmax scale, vd into O at the epilogue).
#define PF_RT(TIN_, TC_, QD_, KD_, VD_)                                                                 \
    pf::launch_prefill<TIN_, TC_>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k, page_table, \
        QD_, KD_, VD_, b, hq, hkv, hd, hdv, total_q, (int)max_seqlen_q.value(), softmax_scale, causal,  \
        softcap, win_l, win_r, s_aux_ptr, stream)
    if (q.scalar_type() == at::kBFloat16) PF_RT(__nv_bfloat16, __nv_bfloat16, std::nullopt, std::nullopt, std::nullopt);
    else if (q.scalar_type() == at::kHalf) PF_RT(__half, __half, std::nullopt, std::nullopt, std::nullopt);
    else if (hd == 128 || hd == 256)  // native e4m3 wgmma prefill (2x tensor); hd256 uses CN=64/STAGES=2
      pf::launch_prefill_fp8(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k, page_table,
        q_descale, k_descale, v_descale, b, hq, hkv, hd, hdv, total_q, (int)max_seqlen_q.value(),
        softmax_scale, causal, softcap, win_l, win_r, stream);
    else                                  PF_RT(__nv_fp8_e4m3, __nv_bfloat16, q_descale, k_descale, v_descale);
#undef PF_RT
  } else if (total_q > 0 && (can_decode_fast || pf_underfill)) {
#define DS_DEC(TIN_, TC_, HD_, HDV_, QD_, KD_, VD_)                                                    \
    launch_decode_split<TIN_, TC_, HD_, HDV_>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k, \
        page_table, QD_, KD_, VD_, b, hq, hkv, total_q, max_q, max_klen, (int)num_splits, softmax_scale, \
        causal, softcap, win_l, win_r, leftpad_ptr, stream)
#define DS_RT(TIN_, TC_, QD_, KD_, VD_) do {                                                            \
    if (hd == 128)                    DS_DEC(TIN_, TC_, 128, 128, QD_, KD_, VD_);                        \
    else if (hd == 96)                DS_DEC(TIN_, TC_, 96, 96, QD_, KD_, VD_);                          \
    else if (hd == 192 && hdv == 128) DS_DEC(TIN_, TC_, 192, 128, QD_, KD_, VD_);                       \
    else if (hd == 192)               DS_DEC(TIN_, TC_, 192, 192, QD_, KD_, VD_);                       \
    else if (hd == 256)               DS_DEC(TIN_, TC_, 256, 256, QD_, KD_, VD_);                       \
    else                              DS_DEC(TIN_, TC_, 64, 64, QD_, KD_, VD_);                          \
  } while (0)
    if (q.scalar_type() == at::kBFloat16) DS_RT(__nv_bfloat16, __nv_bfloat16, std::nullopt, std::nullopt, std::nullopt);
    else if (q.scalar_type() == at::kHalf) DS_RT(__half, __half, std::nullopt, std::nullopt, std::nullopt);
    else if (hd == 128)  // native e4m3 wgmma decode-split (no convert pass)
      launch_decode_split_fp8<128, 128>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k,
        page_table, q_descale, k_descale, v_descale, b, hq, hkv, total_q, max_q, max_klen,
        (int)num_splits, softmax_scale, causal, softcap, win_l, win_r, stream);
    // hd64/hd96 fp8 decode: NATIVE fp8 wgmma via cp.async (no 128-wide TMA box, which would over-read a
    // 64-wide head). The K/Q smem tiles are padded to HDQK=128 so koff8's row*128 swizzle stride fits;
    // the QK still contracts only the real HD cols. No fp8->bf16 cast (the cast was LSU-bound at ~14us).
    else if (hd == 64)
      launch_decode_split_fp8<64, 64>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k,
        page_table, q_descale, k_descale, v_descale, b, hq, hkv, total_q, max_q, max_klen,
        (int)num_splits, softmax_scale, causal, softcap, win_l, win_r, stream);
    else if (hd == 96)
      launch_decode_split_fp8<96, 96>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k,
        page_table, q_descale, k_descale, v_descale, b, hq, hkv, total_q, max_q, max_klen,
        (int)num_splits, softmax_scale, causal, softcap, win_l, win_r, stream);
    else if (hd == 256)   // native e4m3 hd256 decode-split (CN=64/STAGES=2, 2-box PV over HDV)
      launch_decode_split_fp8<256, 256>(q, k, v, out, lse, *cu_seqlens_q, cu_seqlens_k, seqused_k,
        page_table, q_descale, k_descale, v_descale, b, hq, hkv, total_q, max_q, max_klen,
        (int)num_splits, softmax_scale, causal, softcap, win_l, win_r, stream);
    else                                  DS_RT(__nv_fp8_e4m3, __nv_bfloat16, q_descale, k_descale, v_descale);
#undef DS_RT
#undef DS_DEC
  } else if (total_q > 0) {
    // The two wgmma paths above (split-KV decode for q<=4 & q*rep<=16; prefill for any other q>=1,
    // softcap + sliding-window included) cover every q/rep/causal/window/softcap combination. Reaching
    // here means the shape is outside the supported HEAD-DIM / DTYPE / LAYOUT envelope: hd==hdv in
    // {64,96,128,192,256} (or 192/128 for decode), bf16/fp16/fp8-e4m3, contiguous head dim, hq%hkv==0.
    TORCH_CHECK(false, "fa3-fwd: unsupported config (dtype=", q.scalar_type(), " hd=", hd, " hdv=", hdv,
                " max_seqlen_q=", (max_seqlen_q.has_value() ? (int)max_seqlen_q.value() : -1),
                " softcap=", softcap, " win=[", win_l, ",", win_r, "]); supported envelope: "
                "hd==hdv in {64,96,128,192,256} (or 192/128 decode), bf16/fp16/fp8, contiguous, hq%hkv==0.");
  }
  // return [out, softmax_lse, out_accum(empty), softmax_lse_accum(empty)]
  at::Tensor empty = at::empty({0}, q.options().dtype(at::kFloat));
  return {out, lse, empty, empty};
}

at::Tensor mha_fwd_get_scheduler_metadata(
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads, int64_t num_heads_k,
    int64_t headdim, int64_t headdim_v, at::ScalarType qkv_dtype, at::Tensor seqused_k,
    std::optional<at::Tensor> cu_seqlens_q, std::optional<at::Tensor> cu_seqlens_k, std::optional<at::Tensor> cu_seqlens_k_new,
    std::optional<at::Tensor> seqused_q, std::optional<at::Tensor> leftpad_k, std::optional<int64_t> page_size,
    int64_t max_seqlen_k_new, bool is_causal, int64_t window_size_left, int64_t window_size_right, bool has_softcap,
    int64_t num_splits, std::optional<bool> pack_gqa, int64_t sm_margin) {
  // Our fwd recomputes everything internally; return a benign tensor.
  return at::zeros({1}, seqused_k.options().dtype(at::kInt));
}

TORCH_LIBRARY(FA3_LIB_NAME, m) {
  m.def("fwd(Tensor! q, Tensor k, Tensor v, Tensor? k_new, Tensor? v_new, Tensor? q_v, "
        "Tensor!? out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
        "Tensor? seqused_q, Tensor? seqused_k, int? max_seqlen_q, int? max_seqlen_k, "
        "Tensor? page_table, Tensor? kv_batch_idx, Tensor? leftpad_k, Tensor? rotary_cos, "
        "Tensor? rotary_sin, Tensor? seqlens_rotary, Tensor? q_descale, Tensor? k_descale, "
        "Tensor? v_descale, float softmax_scale, bool is_causal, int window_size_left, "
        "int window_size_right, float softcap, bool is_rotary_interleaved, "
        "Tensor? scheduler_metadata, int num_splits, bool? pack_gqa, int sm_margin, "
        "Tensor? s_aux, int cp_world_size, int cp_rank, Tensor? cp_tot_seqused_k) -> Tensor[]");
  m.impl("fwd", torch::kCUDA, &mha_fwd);
  m.def("get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype, "
        "Tensor seqused_k, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
        "Tensor? seqused_q, Tensor? leftpad_k, int? page_size, int max_seqlen_k_new, "
        "bool is_causal, int window_size_left, int window_size_right, bool has_softcap, "
        "int num_splits, bool? pack_gqa, int sm_margin) -> Tensor");
  m.impl("get_scheduler_metadata", torch::kCUDA, &mha_fwd_get_scheduler_metadata);
}
