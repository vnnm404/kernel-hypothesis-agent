// fa4_blackwell.cu — dependency-free CUDA + tcgen05/tmem inline-PTX reimplementation of the
// FlashAttention-4 forward kernel for NVIDIA Blackwell (SM100 / sm_100a). No CUTLASS/CuTe/CuteDSL.
//
// Registers the same `_vllm_fa3_C` `fwd` op schema as kernels/fa3_hopper.cu so it A/Bs against the
// FA4 CuteDSL reference via `python -m bench.eval_fa4`.
//
// STATUS (correctness-first walking skeleton): the compute core (QK GEMM -> online softmax(ex2) ->
// PV GEMM, all in tensor memory, 128B swizzle) is the one validated bit-close in kernels/probes/
// (fa4_qk_gemm / fa4_attn_block / fa4_attn_stream). This host+kernel wraps it for real tensors:
// dense (non-paged) bf16/fp16, head_dim==head_dim_v==128, GQA/MQA/MHA, causal / non-causal /
// sliding-window. One CTA (single warpgroup) per (m_block, q_head, batch); plain swizzled smem
// stores (no TMA yet) — perf work (TMA, warp-specialization, pipeline) is a later phase. Configs
// outside this envelope raise "unsupported" (the dispatcher's correctness-first fallback rule).
//
// Build: -gencode arch=compute_100a,code=sm_100a  (tcgen05 requires the compute_100a virtual arch).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <optional>
#include <torch/library.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace fa4 {

// ------------------------- tcgen05 / tmem inline-PTX primitives -------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void tmem_alloc(uint32_t s,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(s),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
// cta_group::2 variants: collective across the 2-CTA pair (one warp per CTA, same dst_ptr + num_columns).
__device__ __forceinline__ void tmem_alloc2(uint32_t s,int n){ asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(s),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish2(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc2(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ uint32_t tmem_ld(uint32_t a){ uint32_t v; asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(v):"r"(a)); return v; }
__device__ __forceinline__ void tmem_st(uint32_t a, uint32_t v){ asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"::"r"(a),"r"(v)); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;"); }
// Vectorized tmem load/store: 32 columns (32 b32 regs) per instruction — far fewer instructions +
// waits than the .x1 path. Caller issues one tmem_wait_ld/st after a batch.
__device__ __forceinline__ void tmem_ld32(uint32_t a, uint32_t* r){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
        "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
        "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
        "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
      : "r"(a)); }
// Load 32 tmem cols DIRECTLY into f[OFF..OFF+31] (compile-time-const slots, no &f address-taking):
// lets nvcc SROA-split f[] into scalars that ptxas keeps in the 192-reg window -> spill-free 1-pass.
#define LD32_INTO(ADDR, F, OFF) \
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
        : "=r"(F[OFF+0]),"=r"(F[OFF+1]),"=r"(F[OFF+2]),"=r"(F[OFF+3]),"=r"(F[OFF+4]),"=r"(F[OFF+5]),"=r"(F[OFF+6]),"=r"(F[OFF+7]), \
          "=r"(F[OFF+8]),"=r"(F[OFF+9]),"=r"(F[OFF+10]),"=r"(F[OFF+11]),"=r"(F[OFF+12]),"=r"(F[OFF+13]),"=r"(F[OFF+14]),"=r"(F[OFF+15]), \
          "=r"(F[OFF+16]),"=r"(F[OFF+17]),"=r"(F[OFF+18]),"=r"(F[OFF+19]),"=r"(F[OFF+20]),"=r"(F[OFF+21]),"=r"(F[OFF+22]),"=r"(F[OFF+23]), \
          "=r"(F[OFF+24]),"=r"(F[OFF+25]),"=r"(F[OFF+26]),"=r"(F[OFF+27]),"=r"(F[OFF+28]),"=r"(F[OFF+29]),"=r"(F[OFF+30]),"=r"(F[OFF+31]) \
        : "r"(ADDR))
__device__ __forceinline__ void tmem_st32(uint32_t a, const uint32_t* r){
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
      :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
        "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
        "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
        "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]),
        "r"(a)); }
__device__ __forceinline__ void tmem_st16(uint32_t a, const uint32_t* r){
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
      :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
        "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
        "r"(a)); }
__device__ __forceinline__ void tmem_fence(){ asm volatile("tcgen05.fence::after_thread_sync;"); }
// PACKED-FP32 (Blackwell f32x2 SIMD) scalar broadcast + fused ops. FA4 does all its O-rescale / correction
// float math with FADD2/FFMA2 (2 fp32/inst); our old scalar FMUL/FADD ran 2x the fma-pipe instructions.
// The mov.b64 pack/unpack around each op become free register-pair aliasing in SASS (as in FA4's codegen).
__device__ __forceinline__ unsigned long long bcast2(float c){
    unsigned long long r; asm("mov.b64 %0, {%1,%1};":"=l"(r):"f"(c)); return r; }
// r[0],r[1] *= c2  (c2 = bcast2(corr)); 2 fp32 mul in one FMUL2.
__device__ __forceinline__ void pmul2(uint32_t& a, uint32_t& b, unsigned long long c2){
    asm("{ .reg .b64 v; mov.b64 v, {%0,%1}; mul.rn.f32x2 v, v, %2; mov.b64 {%0,%1}, v; }"
        : "+r"(a), "+r"(b) : "l"(c2)); }
// Vectorized online O-rescale: O[row,:] *= corr. Replaces HDV serial ld+st (256 tmem ops/row) with
// HDV/32 x {ld32, mul32, st32}. a0 = base + O_COL + (row<<16). Caller issues tmem_wait_st() after a batch.
template<int HDV>
__device__ __forceinline__ void tmem_o_rescale(uint32_t a0, float corr){
    // NB: do NOT #pragma unroll the g loop. Unrolling lets ptxas software-pipeline all HDV/32 chunks,
    // keeping 4x og[32]=128 regs live -> blows the softmax's 192-reg budget -> spills (the hot line-68
    // tmem_st32-operand LDL). A rolled loop reuses one og[32] (32 regs) and is spill-free.
    const unsigned long long c2=bcast2(corr);
    #pragma unroll 1
    for(int g=0; g<HDV; g+=32){ uint32_t og[32]; tmem_ld32(a0+(uint32_t)g, og); tmem_wait_ld();
        #pragma unroll
        for(int t=0;t<32;t+=2) pmul2(og[t],og[t+1],c2);   // packed fp32x2: 16 FMUL2 vs 32 scalar FMUL
        tmem_st32(a0+(uint32_t)g, og); }
}
// BATCHED O-rescale: issue ALL HDV/32 tmem loads up front, ONE wait, then mul+store. Overlaps the 4 tmem
// read latencies into a single exposure (vs the rolled 4x above). Safe ONLY when f[128] is NOT register-
// resident (i.e. the inline-PTX FULL softmax path, which frees its fragment before this runs); the og[HDV]
// fragment (128 regs) would otherwise blow the 192-reg softmax budget. Cuts long_scoreboard in the hot loop.
template<int HDV>
__device__ __forceinline__ void tmem_o_rescale_batch(uint32_t a0, float corr){
    uint32_t og[HDV];
    #pragma unroll
    for(int g=0; g<HDV; g+=32) tmem_ld32(a0+(uint32_t)g, og+g);
    tmem_wait_ld();
    #pragma unroll
    for(int t=0;t<HDV;++t) og[t]=__float_as_uint(__uint_as_float(og[t])*corr);
    #pragma unroll
    for(int g=0; g<HDV; g+=32) tmem_st32(a0+(uint32_t)g, og+g);
}
// ---- AVO co-design knobs (CORR path only; non-CORR paths untouched). Swept by /tmp/avo driver. ----
// The 3 register tiers MUST satisfy the setmaxnreg pool balance (baseline 128/thr, 512 thr, launch_bounds 512):
//   softmax claim (SM-128)*256  ==  release (128-CORR)*128 + (128-PROD)*128  <=>  CORR+PROD = 512 - 2*SM.
// AVO_RESCALE_DEPTH = #(32-col O chunks) batched per tmem_wait_ld in the correction warp (needs CORR >= ~32*D+overhead).
// AVO_BRANCHLESS = 1 removes the __all_sync(corr==1) skip (v20: always-rescale, no divergence).
#ifndef AVO_SM_REG
#define AVO_SM_REG 192
#endif
#ifndef AVO_CORR_REG
#define AVO_CORR_REG 64
#endif
#ifndef AVO_PROD_REG
#define AVO_PROD_REG 64
#endif
#ifndef AVO_RESCALE_DEPTH
#define AVO_RESCALE_DEPTH 1
#endif
#ifndef AVO_BRANCHLESS
#define AVO_BRANCHLESS 0
#endif
#define AVO_STR2(x) #x
#define AVO_STR(x) AVO_STR2(x)
// ---- register-resident ring/slot accessors -------------------------------------------------------
// `a[s]` with a RUNTIME s forces the array into LOCAL memory (SASS: a prologue `STL.128 [R1]` plus an
// `LDL [R+s*4]` per use). Every pipelined kernel here indexes its ring state -- K/V mbarriers, smem
// bases, MMA descriptors, phase bits, the correction WG's depth-2 C-channel tables -- by `s = i % KVS`
// or the depth-2 slot, all inside the per-K-block loop. NCU on attn_real_llama8b_pf_s16k
// (fwd_kernel_fa4_paged, page_size 1): 38.6M local-memory LOAD sectors against just 6.5M global ones,
// long_scoreboard 9.21/issue vs FA4's 1.95, tensor pipe 31.7% vs FA4's 71.6%, DRAM under 2% on both --
// the kernel is spill-bound, not bandwidth-bound. An unrolled compare-select chain keeps the tables in
// registers; at these depths (2) it is a single ISETP+SEL.
template<int N, class V> __device__ __forceinline__ V rsel(const V (&a)[N], int s){
    V r=a[0];
    #pragma unroll
    for(int t=1;t<N;++t) if(s==t) r=a[t];
    return r;
}
// Read-and-flip an mbarrier phase bit (`p=a[s]; a[s]^=1;`) without a local array.
template<int N> __device__ __forceinline__ int rflip(int (&a)[N], int s){
    int r=a[0];
    #pragma unroll
    for(int t=1;t<N;++t) if(s==t) r=a[t];
    #pragma unroll
    for(int t=0;t<N;++t) if(s==t) a[t]=r^1;
    return r;
}
// Depth-batched O-rescale: issue DEPTH tmem loads, ONE wait, packed-f32x2 mul, DEPTH stores. DEPTH=1 == rolled
// (32 live regs, spill-free at 64); DEPTH=2 (64 regs) overlaps 2 read latencies; DEPTH=4 == full batch (128 regs).
template<int HDV, int DEPTH>
__device__ __forceinline__ void tmem_o_rescale_d(uint32_t a0, float corr){
    const unsigned long long c2=bcast2(corr);
    #pragma unroll 1
    for(int bg=0; bg<HDV; bg += DEPTH*32){
        uint32_t og[DEPTH*32];
        #pragma unroll
        for(int d=0; d<DEPTH; ++d) tmem_ld32(a0+(uint32_t)(bg+d*32), og+d*32);
        tmem_wait_ld();
        #pragma unroll
        for(int t=0;t<DEPTH*32;t+=2) pmul2(og[t],og[t+1],c2);
        #pragma unroll
        for(int d=0; d<DEPTH; ++d) tmem_st32(a0+(uint32_t)(bg+d*32), og+d*32);
    }
}
__device__ __forceinline__ float ex2f(float x){ float r; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x)); return r; }

// 128B swizzle, ROWS=128 tile: koff = K-major (Q,K), voff = MN-major (V). Byte-identical for any
// 16-bit element (bf16/fp16). Reused from the validated FA3 layout.
__device__ __forceinline__ int koff(int r,int c){ return (c>>6)*8192 + r*64 + ((c&63)^((r&7)<<3)); }
__device__ __forceinline__ int voff(int key,int dv){ return (dv>>6)*8192 + key*64 + ((dv&63)^((key&7)<<3)); }
__device__ __forceinline__ uint64_t make_desc(uint32_t addr,uint32_t lbo,uint32_t sbo,int lt){
    uint64_t d=0; d|=(uint64_t)((addr>>4)&0x3FC0u); d|=(uint64_t)((lbo>>4)&0x3FFFu)<<16;
    d|=(uint64_t)((sbo>>4)&0x3FFFu)<<32; d|=(uint64_t)1u<<46; d|=(uint64_t)(lt&0x7u)<<61; return d; }
__device__ __forceinline__ uint64_t desc_add(uint64_t b,uint32_t off){
    uint32_t lo,hi; asm("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(b));
    lo+=off; uint64_t r; asm("mov.b64 %0,{%1,%2};":"=l"(r):"r"(lo),"r"(hi)); return r; }
__device__ __forceinline__ uint32_t qk_doff(int kb){ return (uint32_t)((kb&3)*2 + (kb>>2)*1024); }
__device__ __forceinline__ uint32_t v_doff(int kb){ return (uint32_t)(kb*0x80); }

// Pack two fp32 -> one 32b of two 16-bit elems (low=a, high=b) via a SINGLE packed convert instruction
// (cvt.rn.f16x2/bf16x2.f32) instead of 2 scalar cvts + PRMT. The P conversion (fp32 softmax -> bf16 for the
// PV MMA) was ~15% of steady-state time (cuda_bf16.hpp F2F + exp-store); the packed cvt halves it. RN
// (round-nearest-even) matches __float2*_rn, so bit-identical results. hi operand=b, lo operand=a.
template<class T> __device__ __forceinline__ uint32_t pack16(float a,float b);
template<> __device__ __forceinline__ uint32_t pack16<__nv_bfloat16>(float a,float b){
    uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(r):"f"(b),"f"(a)); return r; }
template<> __device__ __forceinline__ uint32_t pack16<__half>(float a,float b){
    uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;":"=r"(r):"f"(b),"f"(a)); return r; }
// AUTO-GENERATED (gen_softmax_ptx.py): spill-free 1-pass FULL softmax fragment (f<128> in PTX .reg).
__device__ __forceinline__ void softmax_full_ptx_bf16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp;\n"
        ".reg .pred pth;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f16, f17};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f108, f109};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.bf16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.bf16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.bf16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.bf16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.bf16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.bf16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.bf16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.bf16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.bf16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.bf16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.bf16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.bf16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.bf16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.bf16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.bf16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.bf16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.bf16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.bf16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.bf16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.bf16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.bf16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.bf16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.bf16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.bf16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.bf16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.bf16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.bf16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.bf16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.bf16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.bf16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.bf16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.bf16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.bf16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.bf16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.bf16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.bf16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.bf16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.bf16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.bf16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.bf16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.bf16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.bf16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.bf16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.bf16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.bf16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.bf16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.bf16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.bf16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.bf16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.bf16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.bf16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.bf16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.bf16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.bf16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.bf16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.bf16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.bf16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.bf16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.bf16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.bf16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.bf16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa) : "memory");
}
__device__ __forceinline__ void softmax_full_ptx_f16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp;\n"
        ".reg .pred pth;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f16, f17};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f108, f109};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.f16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.f16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.f16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.f16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.f16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.f16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.f16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.f16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.f16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.f16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.f16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.f16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.f16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.f16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.f16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.f16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.f16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.f16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.f16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.f16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.f16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.f16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.f16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.f16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.f16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.f16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.f16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.f16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.f16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.f16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.f16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.f16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.f16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.f16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.f16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.f16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.f16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.f16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.f16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.f16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.f16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.f16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.f16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.f16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.f16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.f16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.f16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.f16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.f16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.f16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.f16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.f16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.f16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.f16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.f16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.f16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.f16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.f16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.f16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.f16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.f16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa) : "memory");
}
// === BEGIN ORD SOFTMAX (generated) ===
__device__ __forceinline__ void softmax_full_ptx_ord_bf16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps, uint32_t orda){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp;\n"
        ".reg .pred pth;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f16, f17};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mbarrier.arrive.shared.b64 _, [%5];\n"
        "mov.b64 vp, {f108, f109};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.bf16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.bf16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.bf16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.bf16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.bf16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.bf16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.bf16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.bf16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.bf16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.bf16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.bf16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.bf16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.bf16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.bf16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.bf16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.bf16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.bf16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.bf16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.bf16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.bf16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.bf16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.bf16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.bf16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.bf16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.bf16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.bf16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.bf16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.bf16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.bf16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.bf16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.bf16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.bf16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.bf16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.bf16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.bf16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.bf16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.bf16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.bf16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.bf16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.bf16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.bf16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.bf16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.bf16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.bf16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.bf16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.bf16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.bf16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.bf16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.bf16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.bf16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.bf16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.bf16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.bf16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.bf16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.bf16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.bf16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.bf16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.bf16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.bf16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.bf16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.bf16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa), "r"(orda) : "memory");
}
__device__ __forceinline__ void softmax_full_ptx_ord_f16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps, uint32_t orda){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp;\n"
        ".reg .pred pth;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f16, f17};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mbarrier.arrive.shared.b64 _, [%5];\n"
        "mov.b64 vp, {f108, f109};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.f16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.f16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.f16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.f16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.f16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.f16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.f16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.f16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.f16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.f16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.f16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.f16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.f16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.f16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.f16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.f16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.f16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.f16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.f16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.f16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.f16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.f16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.f16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.f16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.f16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.f16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.f16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.f16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.f16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.f16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.f16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.f16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.f16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.f16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.f16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.f16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.f16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.f16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.f16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.f16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.f16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.f16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.f16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.f16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.f16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.f16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.f16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.f16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.f16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.f16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.f16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.f16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.f16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.f16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.f16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.f16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.f16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.f16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.f16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.f16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.f16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa), "r"(orda) : "memory");
}
// === BEGIN ORDCS SOFTMAX (generated) ===
__device__ __forceinline__ void softmax_full_ptx_ordcs_bf16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps, uint32_t orda, uint32_t scaddr, uint32_t csbar, float qsc){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp,cor;\n"
        ".reg .pred pth,pcs,pinf;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 qsc2;\n"
        ".reg .f32 qscr;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "mov.f32 qscr, %8;\n"
        "mov.b64 qsc2, {qscr, qscr};\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "mul.f32 mm, mm, qscr;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "setp.eq.f32 pinf, %3, 0fFF800000;\n"
        "sub.f32 cor, %3, mm;\n"
        "ex2.approx.ftz.f32 cor, cor;\n"
        "selp.f32 cor, 0f3F800000, cor, pinf;\n"
        "setp.ne.b32 pcs, %7, 0;\n"
        "@pcs st.shared.f32 [%6], cor;\n"
        "@pcs mbarrier.arrive.shared.b64 _, [%7];\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mbarrier.arrive.shared.b64 _, [%5];\n"
        "mov.b64 vp, {f16, f17};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f108, f109};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.bf16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.bf16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.bf16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.bf16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.bf16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.bf16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.bf16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.bf16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.bf16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.bf16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.bf16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.bf16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.bf16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.bf16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.bf16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.bf16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.bf16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.bf16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.bf16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.bf16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.bf16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.bf16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.bf16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.bf16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.bf16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.bf16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.bf16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.bf16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.bf16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.bf16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.bf16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.bf16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.bf16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.bf16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.bf16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.bf16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.bf16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.bf16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.bf16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.bf16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.bf16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.bf16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.bf16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.bf16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.bf16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.bf16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.bf16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.bf16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.bf16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.bf16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.bf16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.bf16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.bf16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.bf16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.bf16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.bf16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.bf16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.bf16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.bf16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.bf16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.bf16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa), "r"(orda), "r"(scaddr), "r"(csbar), "f"(qsc) : "memory");
}
__device__ __forceinline__ void softmax_full_ptx_ordcs_f16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps, uint32_t orda, uint32_t scaddr, uint32_t csbar, float qsc){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp,cor;\n"
        ".reg .pred pth,pcs,pinf;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 qsc2;\n"
        ".reg .f32 qscr;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "mov.f32 qscr, %8;\n"
        "mov.b64 qsc2, {qscr, qscr};\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "mul.f32 mm, mm, qscr;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "setp.eq.f32 pinf, %3, 0fFF800000;\n"
        "sub.f32 cor, %3, mm;\n"
        "ex2.approx.ftz.f32 cor, cor;\n"
        "selp.f32 cor, 0f3F800000, cor, pinf;\n"
        "setp.ne.b32 pcs, %7, 0;\n"
        "@pcs st.shared.f32 [%6], cor;\n"
        "@pcs mbarrier.arrive.shared.b64 _, [%7];\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mbarrier.arrive.shared.b64 _, [%5];\n"
        "mov.b64 vp, {f16, f17};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f108, f109};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "fma.rn.f32x2 dp, vp, qsc2, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.f16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.f16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.f16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.f16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.f16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.f16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.f16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.f16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.f16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.f16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.f16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.f16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.f16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.f16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.f16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.f16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.f16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.f16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.f16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.f16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.f16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.f16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.f16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.f16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.f16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.f16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.f16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.f16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.f16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.f16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.f16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.f16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.f16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.f16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.f16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.f16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.f16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.f16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.f16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.f16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.f16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.f16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.f16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.f16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.f16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.f16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.f16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.f16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.f16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.f16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.f16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.f16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.f16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.f16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.f16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.f16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.f16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.f16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.f16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.f16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.f16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa), "r"(orda), "r"(scaddr), "r"(csbar), "f"(qsc) : "memory");
}
template<class T> __device__ __forceinline__ void softmax_full_ptx_ordcs(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t orda,uint32_t scaddr,uint32_t csbar,float qsc);
template<> __device__ __forceinline__ void softmax_full_ptx_ordcs<__nv_bfloat16>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t orda,uint32_t scaddr,uint32_t csbar,float qsc){ softmax_full_ptx_ordcs_bf16(sa,pa,m_i,m_new,ps,orda,scaddr,csbar,qsc); }
template<> __device__ __forceinline__ void softmax_full_ptx_ordcs<__half>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t orda,uint32_t scaddr,uint32_t csbar,float qsc){ softmax_full_ptx_ordcs_f16(sa,pa,m_i,m_new,ps,orda,scaddr,csbar,qsc); }
// === END ORDCS SOFTMAX (generated) ===
template<class T> __device__ __forceinline__ void softmax_full_ptx_ord(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t orda);
template<> __device__ __forceinline__ void softmax_full_ptx_ord<__nv_bfloat16>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t orda){ softmax_full_ptx_ord_bf16(sa,pa,m_i,m_new,ps,orda); }
template<> __device__ __forceinline__ void softmax_full_ptx_ord<__half>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t orda){ softmax_full_ptx_ord_f16(sa,pa,m_i,m_new,ps,orda); }
// === END ORD SOFTMAX (generated) ===
__device__ __forceinline__ void softmax_split_ptx_bf16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps, uint32_t pfbar, uint32_t scaddr, uint32_t csbar){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp,cor;\n"
        ".reg .pred pth,pinf,pcs;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "setp.ne.b32 pcs, %7, 0;\n"
        "setp.eq.f32 pinf, %3, 0fFF800000;\n"
        "sub.f32 cor, %3, mm;\n"
        "ex2.approx.ftz.f32 cor, cor;\n"
        "selp.f32 cor, 0f3F800000, cor, pinf;\n"
        "@pcs st.shared.f32 [%6], cor;\n"
        "@pcs mbarrier.arrive.shared.b64 _, [%7];\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f16, f17};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f108, f109};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.bf16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.bf16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.bf16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.bf16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.bf16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.bf16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.bf16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.bf16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.bf16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.bf16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.bf16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.bf16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.bf16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.bf16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.bf16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.bf16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.bf16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.bf16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.bf16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.bf16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.bf16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.bf16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.bf16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.bf16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.bf16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.bf16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.bf16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.bf16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.bf16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.bf16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.bf16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.bf16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.bf16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.bf16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.bf16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.bf16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.bf16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.bf16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.bf16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.bf16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.bf16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.bf16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.bf16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.bf16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.bf16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.bf16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.bf16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "tcgen05.wait::st.sync.aligned;\n"
        "mbarrier.arrive.shared.b64 _, [%5];\n"
        "cvt.rn.bf16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.bf16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.bf16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.bf16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.bf16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.bf16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.bf16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.bf16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.bf16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.bf16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.bf16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.bf16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.bf16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.bf16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.bf16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.bf16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa), "r"(pfbar), "r"(scaddr), "r"(csbar) : "memory");
}
__device__ __forceinline__ void softmax_split_ptx_f16(uint32_t sa, uint32_t pa, float m_i, float& m_new, float& ps, uint32_t pfbar, uint32_t scaddr, uint32_t csbar){
    asm volatile(
        "{\n"
        ".reg .f32 f<128>;\n"
        ".reg .f32 m0,m1,m2,m3,mm,nm,d0,d1,cand,jmp,cor;\n"
        ".reg .pred pth,pinf,pcs;\n"
        ".reg .b64 vp,dp,ep,nm2,sp0,sp1,sp2,sp3;\n"
        ".reg .b64 mgc,pc0,pc1,pc2,pc3,e7,e8,e9,e10;\n"
        ".reg .b32 ei0,ei1,ef0,ef1;\n"
        ".reg .b32 a1,a2,a3,pa1,pa2,pa3,pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15;\n"
        "add.s32 a1, %2, 32;\n"
        "add.s32 a2, %2, 64;\n"
        "add.s32 a3, %2, 96;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31}, [%2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63}, [a1];\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95}, [a2];\n"
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127}, [a3];\n"
        "mov.f32 m0, f0; mov.f32 m1, f1; mov.f32 m2, f2; mov.f32 m3, f3;\n"
        "max.f32 m0, m0, f4;\n"
        "max.f32 m1, m1, f5;\n"
        "max.f32 m2, m2, f6;\n"
        "max.f32 m3, m3, f7;\n"
        "max.f32 m0, m0, f8;\n"
        "max.f32 m1, m1, f9;\n"
        "max.f32 m2, m2, f10;\n"
        "max.f32 m3, m3, f11;\n"
        "max.f32 m0, m0, f12;\n"
        "max.f32 m1, m1, f13;\n"
        "max.f32 m2, m2, f14;\n"
        "max.f32 m3, m3, f15;\n"
        "max.f32 m0, m0, f16;\n"
        "max.f32 m1, m1, f17;\n"
        "max.f32 m2, m2, f18;\n"
        "max.f32 m3, m3, f19;\n"
        "max.f32 m0, m0, f20;\n"
        "max.f32 m1, m1, f21;\n"
        "max.f32 m2, m2, f22;\n"
        "max.f32 m3, m3, f23;\n"
        "max.f32 m0, m0, f24;\n"
        "max.f32 m1, m1, f25;\n"
        "max.f32 m2, m2, f26;\n"
        "max.f32 m3, m3, f27;\n"
        "max.f32 m0, m0, f28;\n"
        "max.f32 m1, m1, f29;\n"
        "max.f32 m2, m2, f30;\n"
        "max.f32 m3, m3, f31;\n"
        "max.f32 m0, m0, f32;\n"
        "max.f32 m1, m1, f33;\n"
        "max.f32 m2, m2, f34;\n"
        "max.f32 m3, m3, f35;\n"
        "max.f32 m0, m0, f36;\n"
        "max.f32 m1, m1, f37;\n"
        "max.f32 m2, m2, f38;\n"
        "max.f32 m3, m3, f39;\n"
        "max.f32 m0, m0, f40;\n"
        "max.f32 m1, m1, f41;\n"
        "max.f32 m2, m2, f42;\n"
        "max.f32 m3, m3, f43;\n"
        "max.f32 m0, m0, f44;\n"
        "max.f32 m1, m1, f45;\n"
        "max.f32 m2, m2, f46;\n"
        "max.f32 m3, m3, f47;\n"
        "max.f32 m0, m0, f48;\n"
        "max.f32 m1, m1, f49;\n"
        "max.f32 m2, m2, f50;\n"
        "max.f32 m3, m3, f51;\n"
        "max.f32 m0, m0, f52;\n"
        "max.f32 m1, m1, f53;\n"
        "max.f32 m2, m2, f54;\n"
        "max.f32 m3, m3, f55;\n"
        "max.f32 m0, m0, f56;\n"
        "max.f32 m1, m1, f57;\n"
        "max.f32 m2, m2, f58;\n"
        "max.f32 m3, m3, f59;\n"
        "max.f32 m0, m0, f60;\n"
        "max.f32 m1, m1, f61;\n"
        "max.f32 m2, m2, f62;\n"
        "max.f32 m3, m3, f63;\n"
        "tcgen05.wait::ld.sync.aligned;\n"
        "max.f32 m0, m0, f64;\n"
        "max.f32 m1, m1, f65;\n"
        "max.f32 m2, m2, f66;\n"
        "max.f32 m3, m3, f67;\n"
        "max.f32 m0, m0, f68;\n"
        "max.f32 m1, m1, f69;\n"
        "max.f32 m2, m2, f70;\n"
        "max.f32 m3, m3, f71;\n"
        "max.f32 m0, m0, f72;\n"
        "max.f32 m1, m1, f73;\n"
        "max.f32 m2, m2, f74;\n"
        "max.f32 m3, m3, f75;\n"
        "max.f32 m0, m0, f76;\n"
        "max.f32 m1, m1, f77;\n"
        "max.f32 m2, m2, f78;\n"
        "max.f32 m3, m3, f79;\n"
        "max.f32 m0, m0, f80;\n"
        "max.f32 m1, m1, f81;\n"
        "max.f32 m2, m2, f82;\n"
        "max.f32 m3, m3, f83;\n"
        "max.f32 m0, m0, f84;\n"
        "max.f32 m1, m1, f85;\n"
        "max.f32 m2, m2, f86;\n"
        "max.f32 m3, m3, f87;\n"
        "max.f32 m0, m0, f88;\n"
        "max.f32 m1, m1, f89;\n"
        "max.f32 m2, m2, f90;\n"
        "max.f32 m3, m3, f91;\n"
        "max.f32 m0, m0, f92;\n"
        "max.f32 m1, m1, f93;\n"
        "max.f32 m2, m2, f94;\n"
        "max.f32 m3, m3, f95;\n"
        "max.f32 m0, m0, f96;\n"
        "max.f32 m1, m1, f97;\n"
        "max.f32 m2, m2, f98;\n"
        "max.f32 m3, m3, f99;\n"
        "max.f32 m0, m0, f100;\n"
        "max.f32 m1, m1, f101;\n"
        "max.f32 m2, m2, f102;\n"
        "max.f32 m3, m3, f103;\n"
        "max.f32 m0, m0, f104;\n"
        "max.f32 m1, m1, f105;\n"
        "max.f32 m2, m2, f106;\n"
        "max.f32 m3, m3, f107;\n"
        "max.f32 m0, m0, f108;\n"
        "max.f32 m1, m1, f109;\n"
        "max.f32 m2, m2, f110;\n"
        "max.f32 m3, m3, f111;\n"
        "max.f32 m0, m0, f112;\n"
        "max.f32 m1, m1, f113;\n"
        "max.f32 m2, m2, f114;\n"
        "max.f32 m3, m3, f115;\n"
        "max.f32 m0, m0, f116;\n"
        "max.f32 m1, m1, f117;\n"
        "max.f32 m2, m2, f118;\n"
        "max.f32 m3, m3, f119;\n"
        "max.f32 m0, m0, f120;\n"
        "max.f32 m1, m1, f121;\n"
        "max.f32 m2, m2, f122;\n"
        "max.f32 m3, m3, f123;\n"
        "max.f32 m0, m0, f124;\n"
        "max.f32 m1, m1, f125;\n"
        "max.f32 m2, m2, f126;\n"
        "max.f32 m3, m3, f127;\n"
        "max.f32 mm, m0, m1;\n"
        "max.f32 m2, m2, m3;\n"
        "max.f32 mm, mm, m2;\n"
        "max.f32 cand, mm, %3;\n"
        "sub.f32 jmp, cand, %3;\n"
        "setp.gt.f32 pth, jmp, 0f41000000;\n"
        "selp.f32 mm, cand, %3, pth;\n"
        "mov.f32 %0, mm;\n"
        "neg.f32 nm, mm;\n"
        "mov.b64 nm2, {nm, nm};\n"
        "setp.ne.b32 pcs, %7, 0;\n"
        "setp.eq.f32 pinf, %3, 0fFF800000;\n"
        "sub.f32 cor, %3, mm;\n"
        "ex2.approx.ftz.f32 cor, cor;\n"
        "selp.f32 cor, 0f3F800000, cor, pinf;\n"
        "@pcs st.shared.f32 [%6], cor;\n"
        "@pcs mbarrier.arrive.shared.b64 _, [%7];\n"
        "mov.b64 sp0, 0; mov.b64 sp1, 0; mov.b64 sp2, 0; mov.b64 sp3, 0;\n"
        "mov.b64 vp, {f0, f1};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f0, d0;\n"
        "ex2.approx.ftz.f32 f1, d1;\n"
        "mov.b64 ep, {f0, f1};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f2, f3};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f2, d0;\n"
        "ex2.approx.ftz.f32 f3, d1;\n"
        "mov.b64 ep, {f2, f3};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f4, f5};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f4, d0;\n"
        "ex2.approx.ftz.f32 f5, d1;\n"
        "mov.b64 ep, {f4, f5};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f6, f7};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f6, d0;\n"
        "ex2.approx.ftz.f32 f7, d1;\n"
        "mov.b64 ep, {f6, f7};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f8, f9};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f8, d0;\n"
        "ex2.approx.ftz.f32 f9, d1;\n"
        "mov.b64 ep, {f8, f9};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f10, f11};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f10, d0;\n"
        "ex2.approx.ftz.f32 f11, d1;\n"
        "mov.b64 ep, {f10, f11};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f12, f13};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f12, d0;\n"
        "ex2.approx.ftz.f32 f13, d1;\n"
        "mov.b64 ep, {f12, f13};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f14, f15};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f14, d0;\n"
        "ex2.approx.ftz.f32 f15, d1;\n"
        "mov.b64 ep, {f14, f15};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f16, f17};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f16, d0;\n"
        "ex2.approx.ftz.f32 f17, d1;\n"
        "mov.b64 ep, {f16, f17};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f18, f19};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f18, d0;\n"
        "ex2.approx.ftz.f32 f19, d1;\n"
        "mov.b64 ep, {f18, f19};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f20, f21};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f20, d0;\n"
        "ex2.approx.ftz.f32 f21, d1;\n"
        "mov.b64 ep, {f20, f21};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f22, f23};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f22, d0;\n"
        "ex2.approx.ftz.f32 f23, d1;\n"
        "mov.b64 ep, {f22, f23};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f24, f25};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f24, d0;\n"
        "ex2.approx.ftz.f32 f25, d1;\n"
        "mov.b64 ep, {f24, f25};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f26, f27};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f26, d0;\n"
        "ex2.approx.ftz.f32 f27, d1;\n"
        "mov.b64 ep, {f26, f27};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f28, f29};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f28, d0;\n"
        "ex2.approx.ftz.f32 f29, d1;\n"
        "mov.b64 ep, {f28, f29};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f30, f31};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f30, d0;\n"
        "ex2.approx.ftz.f32 f31, d1;\n"
        "mov.b64 ep, {f30, f31};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f32, f33};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f32, d0;\n"
        "ex2.approx.ftz.f32 f33, d1;\n"
        "mov.b64 ep, {f32, f33};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f34, f35};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f34, d0;\n"
        "ex2.approx.ftz.f32 f35, d1;\n"
        "mov.b64 ep, {f34, f35};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f36, f37};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f36, d0;\n"
        "ex2.approx.ftz.f32 f37, d1;\n"
        "mov.b64 ep, {f36, f37};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f38, f39};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f38, d0;\n"
        "ex2.approx.ftz.f32 f39, d1;\n"
        "mov.b64 ep, {f38, f39};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f40, f41};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f40, d0;\n"
        "ex2.approx.ftz.f32 f41, d1;\n"
        "mov.b64 ep, {f40, f41};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f42, f43};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f42, d0;\n"
        "ex2.approx.ftz.f32 f43, d1;\n"
        "mov.b64 ep, {f42, f43};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f44, f45};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f44, d0;\n"
        "ex2.approx.ftz.f32 f45, d1;\n"
        "mov.b64 ep, {f44, f45};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f46, f47};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f46, d0;\n"
        "ex2.approx.ftz.f32 f47, d1;\n"
        "mov.b64 ep, {f46, f47};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f48, f49};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f48, d0;\n"
        "ex2.approx.ftz.f32 f49, d1;\n"
        "mov.b64 ep, {f48, f49};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f50, f51};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f50, d0;\n"
        "ex2.approx.ftz.f32 f51, d1;\n"
        "mov.b64 ep, {f50, f51};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f52, f53};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f52, d0;\n"
        "ex2.approx.ftz.f32 f53, d1;\n"
        "mov.b64 ep, {f52, f53};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f54, f55};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f54, d0;\n"
        "ex2.approx.ftz.f32 f55, d1;\n"
        "mov.b64 ep, {f54, f55};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f56, f57};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f56, d0;\n"
        "ex2.approx.ftz.f32 f57, d1;\n"
        "mov.b64 ep, {f56, f57};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f58, f59};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f58, d0;\n"
        "ex2.approx.ftz.f32 f59, d1;\n"
        "mov.b64 ep, {f58, f59};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f60, f61};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f60, d0;\n"
        "ex2.approx.ftz.f32 f61, d1;\n"
        "mov.b64 ep, {f60, f61};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f62, f63};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f62, d0;\n"
        "ex2.approx.ftz.f32 f63, d1;\n"
        "mov.b64 ep, {f62, f63};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f64, f65};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f64, d0;\n"
        "ex2.approx.ftz.f32 f65, d1;\n"
        "mov.b64 ep, {f64, f65};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f66, f67};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f66, d0;\n"
        "ex2.approx.ftz.f32 f67, d1;\n"
        "mov.b64 ep, {f66, f67};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f68, f69};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f68, d0;\n"
        "ex2.approx.ftz.f32 f69, d1;\n"
        "mov.b64 ep, {f68, f69};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f70, f71};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f70, d0;\n"
        "ex2.approx.ftz.f32 f71, d1;\n"
        "mov.b64 ep, {f70, f71};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f72, f73};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f72, d0;\n"
        "ex2.approx.ftz.f32 f73, d1;\n"
        "mov.b64 ep, {f72, f73};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f74, f75};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f74, d0;\n"
        "ex2.approx.ftz.f32 f75, d1;\n"
        "mov.b64 ep, {f74, f75};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f76, f77};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f76, d0;\n"
        "ex2.approx.ftz.f32 f77, d1;\n"
        "mov.b64 ep, {f76, f77};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f78, f79};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f78, d0;\n"
        "ex2.approx.ftz.f32 f79, d1;\n"
        "mov.b64 ep, {f78, f79};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f80, f81};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f80, d0;\n"
        "ex2.approx.ftz.f32 f81, d1;\n"
        "mov.b64 ep, {f80, f81};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f82, f83};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f82, d0;\n"
        "ex2.approx.ftz.f32 f83, d1;\n"
        "mov.b64 ep, {f82, f83};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f84, f85};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f84, d0;\n"
        "ex2.approx.ftz.f32 f85, d1;\n"
        "mov.b64 ep, {f84, f85};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f86, f87};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f86, d0;\n"
        "ex2.approx.ftz.f32 f87, d1;\n"
        "mov.b64 ep, {f86, f87};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f88, f89};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f88, d0;\n"
        "ex2.approx.ftz.f32 f89, d1;\n"
        "mov.b64 ep, {f88, f89};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f90, f91};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f90, d0;\n"
        "ex2.approx.ftz.f32 f91, d1;\n"
        "mov.b64 ep, {f90, f91};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f92, f93};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f92, d0;\n"
        "ex2.approx.ftz.f32 f93, d1;\n"
        "mov.b64 ep, {f92, f93};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f94, f95};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f94, d0;\n"
        "ex2.approx.ftz.f32 f95, d1;\n"
        "mov.b64 ep, {f94, f95};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f96, f97};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f96, d0;\n"
        "ex2.approx.ftz.f32 f97, d1;\n"
        "mov.b64 ep, {f96, f97};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f98, f99};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f98, d0;\n"
        "ex2.approx.ftz.f32 f99, d1;\n"
        "mov.b64 ep, {f98, f99};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f100, f101};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f100, d0;\n"
        "ex2.approx.ftz.f32 f101, d1;\n"
        "mov.b64 ep, {f100, f101};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f102, f103};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f102, d0;\n"
        "ex2.approx.ftz.f32 f103, d1;\n"
        "mov.b64 ep, {f102, f103};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f104, f105};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f104, d0;\n"
        "ex2.approx.ftz.f32 f105, d1;\n"
        "mov.b64 ep, {f104, f105};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f106, f107};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f106, d0;\n"
        "ex2.approx.ftz.f32 f107, d1;\n"
        "mov.b64 ep, {f106, f107};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f108, f109};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f108, d0;\n"
        "ex2.approx.ftz.f32 f109, d1;\n"
        "mov.b64 ep, {f108, f109};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f110, f111};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f110, d0;\n"
        "ex2.approx.ftz.f32 f111, d1;\n"
        "mov.b64 ep, {f110, f111};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f112, f113};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f112, d0;\n"
        "ex2.approx.ftz.f32 f113, d1;\n"
        "mov.b64 ep, {f112, f113};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f114, f115};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f114, d0;\n"
        "ex2.approx.ftz.f32 f115, d1;\n"
        "mov.b64 ep, {f114, f115};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f116, f117};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f116, d0;\n"
        "ex2.approx.ftz.f32 f117, d1;\n"
        "mov.b64 ep, {f116, f117};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f118, f119};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f118, d0;\n"
        "ex2.approx.ftz.f32 f119, d1;\n"
        "mov.b64 ep, {f118, f119};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "mov.b64 vp, {f120, f121};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f120, d0;\n"
        "ex2.approx.ftz.f32 f121, d1;\n"
        "mov.b64 ep, {f120, f121};\n"
        "add.rn.f32x2 sp0, sp0, ep;\n"
        "mov.b64 vp, {f122, f123};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f122, d0;\n"
        "ex2.approx.ftz.f32 f123, d1;\n"
        "mov.b64 ep, {f122, f123};\n"
        "add.rn.f32x2 sp1, sp1, ep;\n"
        "mov.b64 vp, {f124, f125};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f124, d0;\n"
        "ex2.approx.ftz.f32 f125, d1;\n"
        "mov.b64 ep, {f124, f125};\n"
        "add.rn.f32x2 sp2, sp2, ep;\n"
        "mov.b64 vp, {f126, f127};\n"
        "add.rn.f32x2 dp, vp, nm2;\n"
        "mov.b64 {d0, d1}, dp;\n"
        "ex2.approx.ftz.f32 f126, d0;\n"
        "ex2.approx.ftz.f32 f127, d1;\n"
        "mov.b64 ep, {f126, f127};\n"
        "add.rn.f32x2 sp3, sp3, ep;\n"
        "add.rn.f32x2 sp0, sp0, sp1;\n"
        "add.rn.f32x2 sp2, sp2, sp3;\n"
        "add.rn.f32x2 sp0, sp0, sp2;\n"
        "mov.b64 {d0, d1}, sp0;\n"
        "add.f32 %1, d0, d1;\n"
        "add.s32 pa1, %4, 16; add.s32 pa2, %4, 32; add.s32 pa3, %4, 48;\n"
        "cvt.rn.f16x2.f32 pk0, f1, f0;\n"
        "cvt.rn.f16x2.f32 pk1, f3, f2;\n"
        "cvt.rn.f16x2.f32 pk2, f5, f4;\n"
        "cvt.rn.f16x2.f32 pk3, f7, f6;\n"
        "cvt.rn.f16x2.f32 pk4, f9, f8;\n"
        "cvt.rn.f16x2.f32 pk5, f11, f10;\n"
        "cvt.rn.f16x2.f32 pk6, f13, f12;\n"
        "cvt.rn.f16x2.f32 pk7, f15, f14;\n"
        "cvt.rn.f16x2.f32 pk8, f17, f16;\n"
        "cvt.rn.f16x2.f32 pk9, f19, f18;\n"
        "cvt.rn.f16x2.f32 pk10, f21, f20;\n"
        "cvt.rn.f16x2.f32 pk11, f23, f22;\n"
        "cvt.rn.f16x2.f32 pk12, f25, f24;\n"
        "cvt.rn.f16x2.f32 pk13, f27, f26;\n"
        "cvt.rn.f16x2.f32 pk14, f29, f28;\n"
        "cvt.rn.f16x2.f32 pk15, f31, f30;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%4], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f33, f32;\n"
        "cvt.rn.f16x2.f32 pk1, f35, f34;\n"
        "cvt.rn.f16x2.f32 pk2, f37, f36;\n"
        "cvt.rn.f16x2.f32 pk3, f39, f38;\n"
        "cvt.rn.f16x2.f32 pk4, f41, f40;\n"
        "cvt.rn.f16x2.f32 pk5, f43, f42;\n"
        "cvt.rn.f16x2.f32 pk6, f45, f44;\n"
        "cvt.rn.f16x2.f32 pk7, f47, f46;\n"
        "cvt.rn.f16x2.f32 pk8, f49, f48;\n"
        "cvt.rn.f16x2.f32 pk9, f51, f50;\n"
        "cvt.rn.f16x2.f32 pk10, f53, f52;\n"
        "cvt.rn.f16x2.f32 pk11, f55, f54;\n"
        "cvt.rn.f16x2.f32 pk12, f57, f56;\n"
        "cvt.rn.f16x2.f32 pk13, f59, f58;\n"
        "cvt.rn.f16x2.f32 pk14, f61, f60;\n"
        "cvt.rn.f16x2.f32 pk15, f63, f62;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa1], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "cvt.rn.f16x2.f32 pk0, f65, f64;\n"
        "cvt.rn.f16x2.f32 pk1, f67, f66;\n"
        "cvt.rn.f16x2.f32 pk2, f69, f68;\n"
        "cvt.rn.f16x2.f32 pk3, f71, f70;\n"
        "cvt.rn.f16x2.f32 pk4, f73, f72;\n"
        "cvt.rn.f16x2.f32 pk5, f75, f74;\n"
        "cvt.rn.f16x2.f32 pk6, f77, f76;\n"
        "cvt.rn.f16x2.f32 pk7, f79, f78;\n"
        "cvt.rn.f16x2.f32 pk8, f81, f80;\n"
        "cvt.rn.f16x2.f32 pk9, f83, f82;\n"
        "cvt.rn.f16x2.f32 pk10, f85, f84;\n"
        "cvt.rn.f16x2.f32 pk11, f87, f86;\n"
        "cvt.rn.f16x2.f32 pk12, f89, f88;\n"
        "cvt.rn.f16x2.f32 pk13, f91, f90;\n"
        "cvt.rn.f16x2.f32 pk14, f93, f92;\n"
        "cvt.rn.f16x2.f32 pk15, f95, f94;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa2], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "tcgen05.wait::st.sync.aligned;\n"
        "mbarrier.arrive.shared.b64 _, [%5];\n"
        "cvt.rn.f16x2.f32 pk0, f97, f96;\n"
        "cvt.rn.f16x2.f32 pk1, f99, f98;\n"
        "cvt.rn.f16x2.f32 pk2, f101, f100;\n"
        "cvt.rn.f16x2.f32 pk3, f103, f102;\n"
        "cvt.rn.f16x2.f32 pk4, f105, f104;\n"
        "cvt.rn.f16x2.f32 pk5, f107, f106;\n"
        "cvt.rn.f16x2.f32 pk6, f109, f108;\n"
        "cvt.rn.f16x2.f32 pk7, f111, f110;\n"
        "cvt.rn.f16x2.f32 pk8, f113, f112;\n"
        "cvt.rn.f16x2.f32 pk9, f115, f114;\n"
        "cvt.rn.f16x2.f32 pk10, f117, f116;\n"
        "cvt.rn.f16x2.f32 pk11, f119, f118;\n"
        "cvt.rn.f16x2.f32 pk12, f121, f120;\n"
        "cvt.rn.f16x2.f32 pk13, f123, f122;\n"
        "cvt.rn.f16x2.f32 pk14, f125, f124;\n"
        "cvt.rn.f16x2.f32 pk15, f127, f126;\n"
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [pa3], {pk0,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8,pk9,pk10,pk11,pk12,pk13,pk14,pk15};\n"
        "}\n"
        : "=f"(m_new), "=f"(ps)
        : "r"(sa), "f"(m_i), "r"(pa), "r"(pfbar), "r"(scaddr), "r"(csbar) : "memory");
}
template<class T> __device__ __forceinline__ void softmax_full_ptx(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps);
template<> __device__ __forceinline__ void softmax_full_ptx<__nv_bfloat16>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps){ softmax_full_ptx_bf16(sa,pa,m_i,m_new,ps); }
template<> __device__ __forceinline__ void softmax_full_ptx<__half>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps){ softmax_full_ptx_f16(sa,pa,m_i,m_new,ps); }
template<class T> __device__ __forceinline__ void softmax_split_ptx(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t pfbar,uint32_t scaddr,uint32_t csbar);
template<> __device__ __forceinline__ void softmax_split_ptx<__nv_bfloat16>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t pfbar,uint32_t scaddr,uint32_t csbar){ softmax_split_ptx_bf16(sa,pa,m_i,m_new,ps,pfbar,scaddr,csbar); }
template<> __device__ __forceinline__ void softmax_split_ptx<__half>(uint32_t sa,uint32_t pa,float m_i,float& m_new,float& ps,uint32_t pfbar,uint32_t scaddr,uint32_t csbar){ softmax_split_ptx_f16(sa,pa,m_i,m_new,ps,pfbar,scaddr,csbar); }

template<class T> __device__ __forceinline__ float to_f(T x);
template<> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x){ return __bfloat162float(x); }
template<> __device__ __forceinline__ float to_f<__half>(__half x){ return __half2float(x); }
template<class T> __device__ __forceinline__ T from_f(float x);
template<> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x){ return __float2bfloat16_rn(x); }
template<> __device__ __forceinline__ __half from_f<__half>(float x){ return __float2half_rn(x); }

__device__ __forceinline__ void mma_commit(uint32_t mbar){
    asm volatile("{ .reg .pred lead; elect.sync _|lead,-1;\n  @lead tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }"::"r"(mbar)); }
__device__ __forceinline__ void mma_wait0(uint32_t mbar){
    asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],0,10000000;\n @P bra D; bra LW;\nD: }"::"r"(mbar)); }
// ---- warp-specialized-pipeline mbarrier + TMA helpers (fast path) ----
__device__ __forceinline__ void mbar_init(uint32_t m,int c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(m),"r"(c)); }
__device__ __forceinline__ void mbar_arrive(uint32_t m){ asm volatile("mbarrier.arrive.shared.b64 _, [%0];"::"r"(m)); }
__device__ __forceinline__ void mbar_expect(uint32_t m,int bytes){ asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(m),"r"(bytes)); }
__device__ __forceinline__ void mbar_wait(uint32_t m,int ph){ asm volatile("{ .reg .pred P;\nLW: mbarrier.try_wait.parity.shared.b64 P,[%0],%1,10000000;\n @P bra D; bra LW;\nD: }"::"r"(m),"r"(ph)); }
// Defer an mbarrier arrival until every cp.async this THREAD has already issued has landed. `.noinc` means
// the pending count comes from mbarrier.init, so the barrier must be initialised with the load-thread count.
// This is what lets a cp.async producer run ahead: no cp.async.wait_group, no CTA rendezvous.
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t m){
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"::"r"(m):"memory"); }
__device__ __forceinline__ void tma_box(void* dst,const void* map,int x,int y,uint32_t bar){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
                 :: "r"(s),"l"(map),"r"(x),"r"(y),"r"(bar) : "memory"); }
// TMA tile::gather4 (SM100): gather FOUR non-contiguous source rows {y0..y3} at column x into a single 2D
// swizzled smem destination (4 consecutive dst rows). HW-async, consumes no warp issue slots. Empirically
// (gather4_probe): the 128B swizzle is applied by ABSOLUTE smem offset within the 1024B swizzle atom, so
// gathering keys {4g..4g+3} to dst=sK+cb*8192+4g*64 lands EXACTLY in the koff/voff layout the MMA reads.
// .shared::cta (single-CTA dst) is the sm_100 form; .shared::cluster gather4 needs sm_100a-only codegen.
__device__ __forceinline__ void tma_gather4(void* dst,const void* map,int x,int y0,int y1,int y2,int y3,uint32_t bar){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2,%3,%4,%5,%6}], [%7];"
                 :: "r"(s),"l"(map),"r"(x),"r"(y0),"r"(y1),"r"(y2),"r"(y3),"r"(bar) : "memory"); }
// ---- 2-SM (cta_group::2) cluster helpers (validated in kernels/probes/fa4_{qk,attn}_2sm.cu) ----
// Full-cluster execution+memory barrier (both CTAs, all threads). release/acquire orders cross-CTA smem.
__device__ __forceinline__ void cluster_sync(){
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;"   ::: "memory"); }
// Leader signals BOTH CTAs' mbar (same offset) after all prior cta_group::2 MMAs retire. mask=CTAs in cluster.
__device__ __forceinline__ void mma_commit2(uint32_t mbar, uint16_t mask){
    asm volatile("{ .reg .pred lead; elect.sync _|lead,-1;\n"
                 "  @lead tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1; }"
                 ::"r"(mbar),"h"(mask)); }
// 2-SM QK (SS: both operands from swizzled smem, K split by keys across the pair) -> S in distributed tmem.
__device__ __forceinline__ void mma_qk2(uint32_t tc,uint64_t da,uint64_t db,uint32_t idesc,uint32_t acc){
    asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%4,0;\n"
        "  @lead tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5,%6,%7,%8,%9,%10,%11,%12}, p; }"
        ::"r"(tc),"l"(da),"l"(db),"r"(idesc),"r"(acc),
          "r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u)); }
// 2-SM PV (TS: A=P from distributed tmem, B=V from smem MN-major split by HDV) -> O in distributed tmem.
__device__ __forceinline__ void mma_pv2(uint32_t tc,uint32_t ta,uint64_t db,uint32_t idesc,uint32_t acc){
    asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%4,0;\n"
        "  @lead tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5,%6,%7,%8,%9,%10,%11,%12}, p; }"
        ::"r"(tc),"r"(ta),"l"(db),"r"(idesc),"r"(acc),
          "r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u),"r"(0u)); }
// Remote mbarrier arrive: map local smem barrier addr into cta `rank`'s window, then arrive (cluster
// release/acquire orders that CTA's prior smem/tmem writes for the waiter). CUTLASS ClusterBarrier::arrive.
__device__ __forceinline__ void mbar_arrive_remote(uint32_t local_bar, uint32_t rank){
    asm volatile("{ .reg .u32 ra; mapa.shared::cluster.u32 ra, %0, %1;\n"
                 "  mbarrier.arrive.shared::cluster.b64 _, [ra]; }" ::"r"(local_bar),"r"(rank):"memory"); }
// 2-SM TMA (cute::SM100_TMA_2SM_LOAD_2D, minus the optional L2 hint): BOTH CTAs load their OWN half into
// their OWN smem, but the completion tx is routed to the LEADER (CTA0) barrier by clearing the peer bit
// (Sm100MmaPeerBitMask=0xFEFFFFFF). This is what makes the leader's cta_group::2 MMA correctly observe
// the PEER CTA's TMA-written operand (a per-CTA barrier + generic remote-arrive does NOT establish the
// async->tensor proxy ordering the UMMA needs). Only the leader arrive_and_expect_tx (DOUBLED bytes).
__device__ __forceinline__ void tma_box_2sm(void* dst,const void* map,int x,int y,uint32_t bar_local){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(dst); uint32_t m=bar_local & 0xFEFFFFFFu;
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2,%3}], [%4];" ::"r"(s),"l"(map),"r"(x),"r"(y),"r"(m):"memory"); }

constexpr int TM=128, TN=128;                 // Q rows / K cols per tile
// tmem column map compressed to 256 cols so TWO CTAs fit per SM (512-col tmem) -> 2x warps for
// latency hiding. S at 0..127; P (bf16, TN/2=64 cols) ALIASED into S's upper half at col 64 (S is
// fully read into registers before P is written, then P is consumed by PV before the next QK); O at 128.
constexpr int S_COL=0, P_COL=64, O_COL=128;
constexpr int TMEM_COLS=256;

// idesc: c_format=F32, a/b_format bf16=1 fp16=0, m_dim=M>>4=8. QK: K-major, n_dim=N>>3=TN>>3.
// PV: b(V) MN-major (bit16), n_dim=HDV>>3.
template<class T> __device__ __forceinline__ uint32_t idesc_base(){
    return (1u<<4) | ((std::is_same<T,__nv_bfloat16>::value?1u:0u)*(0x80u+0x400u)) | (8u<<24); }
template<class T,int N> __device__ __forceinline__ uint32_t qk_id(){ return idesc_base<T>() | ((uint32_t)(N>>3)<<17); }
template<class T,int N> __device__ __forceinline__ uint32_t pv_id(){ return qk_id<T,N>() | (1u<<16); }

// One CTA (128 threads = 1 warpgroup) computes O for a [TM rows] Q-tile of (batch b, q-head h).
// HD = QK contraction dim (K cols, mult of 64), HDV = value/output dim (mult of 64).
template<class T, int HD, int HDV>
__global__ void __launch_bounds__(128) fwd_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale,
    int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int paged, const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, int leftpad,
    int nB, int nsplits, float* __restrict__ o_part, float* __restrict__ lse_part) {
    constexpr int NKT_QK = HD/16;    // QK k-tiles (contract over HD)
    constexpr int NKT_PV = TN/16;    // PV k-tiles (contract over N=keys)
    extern __shared__ __nv_bfloat16 dyn_raw[];
    uint32_t rawp = smem_u32(dyn_raw);
    uint32_t pad = ((rawp+1023u)&~1023u) - rawp;
    T* dyn = (T*)((char*)dyn_raw + pad);
    T* sQ = dyn; T* sK = sQ + TM*HD; T* sV = sK + TN*HD;
    __shared__ __align__(8) uint32_t s_base; __shared__ __align__(8) uint64_t s_mbar;

    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y;
    const int b = blockIdx.z % nB, split_idx = blockIdx.z / nB;   // SplitKV: fold split into grid.z
    const int q0 = cu_q[b], q1 = cu_q[b+1];
    const int k0 = paged ? 0 : cu_k[b];
    const int seqlen_q = q1-q0;
    const int seqlen_k = paged ? seqused_k[b] : (cu_k[b+1]-cu_k[b]);
    const int row0 = mb*TM;
    if (row0 >= seqlen_q) return;                       // m-tile beyond this sequence
    const int hk = h / rep;
    const float LOG2E=1.4426950408889634f;
    const float inv_softcap = (softcap>0.f) ? 1.f/softcap : 0.f;

    // load Q tile [TM,HD] into swizzled smem; gmem uses the REAL head_dim (rhd), padded cols -> 0.
    for (int i=tid; i<TM*HD; i+=128){ int r=i/HD, c=i%HD; int qr=row0+r;
        sQ[koff(r,c)] = (qr<seqlen_q && c<rhd) ? q[(size_t)(q0+qr)*hq*rhd + (size_t)h*rhd + c] : from_f<T>(0.f); }
    asm volatile("fence.proxy.async.shared::cta;");
    if (warp==0){ tmem_alloc(smem_u32(&s_base),TMEM_COLS); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base=s_base, mbar=smem_u32(&s_mbar);

    // causal/local n-block range (append semantics: query_pos = seqlen_k - seqlen_q + qrow)
    const int nblocks = (seqlen_k + TN - 1)/TN;
    int n_lo=0, n_hi=nblocks;
    const int qp_max = seqlen_k - seqlen_q + row0 + (TM-1);
    const int qp_min = seqlen_k - seqlen_q + row0;
    if (is_causal) n_hi = min(nblocks, (qp_max)/TN + 1);
    if (win_r>=0)  n_hi = min(n_hi, (qp_max+win_r)/TN + 1);
    if (win_l>=0){ int lo=(qp_min-win_l); n_lo = (lo>0)? lo/TN : 0; }
    n_hi = max(n_hi, n_lo);
    if (nsplits>1){ int on_lo=n_lo, on_hi=n_hi, tot=on_hi-on_lo, per=(tot+nsplits-1)/nsplits;   // this split's K sub-range
        n_lo = min(on_hi, on_lo + split_idx*per); n_hi = min(on_hi, on_lo + (split_idx+1)*per); }

    float m_i=-INFINITY, l_i=0.f;
    float sreg[TN];
    const uint32_t QKI=qk_id<T,TN>(), PVI=pv_id<T,HDV>();
    int first_pv = 1;   // first PV MMA for this row initializes O (enable_d=0)

    for (int nb=n_lo; nb<n_hi; ++nb){
        const int key0 = nb*TN;
        // stage K,V block [TN,HD]/[TN,HDV] (keys past seqlen_k zero-filled)
        for (int i=tid;i<TN*HD;i+=128){ int r=i/HD,c=i%HD; int kk=key0+r;
            bool ok = (kk<seqlen_k && c<rhd);
            int prow = paged ? (page_table[b*pt_stride + kk/block_size]*block_size + kk%block_size) : (k0+kk);
            sK[koff(r,c)] = ok ? k[(size_t)prow*hkv*rhd + (size_t)hk*rhd + c] : from_f<T>(0.f); }
        for (int i=tid;i<TN*HDV;i+=128){ int key=i/HDV,dv=i%HDV; int kk=key0+key;
            bool ok = (kk<seqlen_k && dv<rhdv);
            int prow = paged ? (page_table[b*pt_stride + kk/block_size]*block_size + kk%block_size) : (k0+kk);
            sV[voff(key,dv)] = ok ? v[(size_t)prow*hkv*rhdv + (size_t)hk*rhdv + dv] : from_f<T>(0.f); }
        asm volatile("fence.proxy.async.shared::cta;");
        if (tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(mbar));
        __syncthreads();

        // QK -> S at S_COL
        if (warp==0){
            uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2), kd0=make_desc(smem_u32(sK),16,1024,2);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%3,0;\n"
                    "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+S_COL),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); }
            mma_commit(mbar);
        }
        mma_wait0(mbar); tmem_fence(); __syncthreads();

        // softmax step for row = tid (query pos qp). Work in the "logit" domain: logit = scale*S,
        // then optional softcap = softcap*tanh(logit/softcap); base for ex2 is LOG2E (scale folded in).
        // Single-pass: read S into registers first (P aliases S's upper half, so S must be fully read
        // before P is written).
        const int qp = seqlen_k - seqlen_q + row0 + tid;
        float rmax=-INFINITY;
        for(int c=0;c<TN;++c){ uint32_t bb=tmem_ld(base+S_COL+((uint32_t)tid<<16)+c); float sv; memcpy(&sv,&bb,4);
            float logit = scale*sv;
            if (softcap>0.f) logit = softcap * tanhf(logit*inv_softcap);
            int kk=key0+c; bool valid = (kk<seqlen_k) && (kk>=leftpad);
            if (is_causal && kk>qp) valid=false;
            if (win_l>=0 && kk < qp-win_l) valid=false;
            if (win_r>=0 && kk > qp+win_r) valid=false;
            logit = valid ? logit : -INFINITY; sreg[c]=logit; rmax=fmaxf(rmax,logit); }
        tmem_wait_ld();
        float m_new = fmaxf(m_i, rmax);
        float corr = (m_i==-INFINITY || m_new==-INFINITY) ? 1.f : ex2f((m_i-m_new)*LOG2E);
        float ps=0.f;
        for(int c=0;c<TN;++c){ float p = (sreg[c]==-INFINITY)?0.f:ex2f((sreg[c]-m_new)*LOG2E); sreg[c]=p; ps+=p; }
        l_i = l_i*corr + ps; m_i = m_new;
        for(int c=0;c<TN/2;++c) tmem_st(base+P_COL+((uint32_t)tid<<16)+c, pack16<T>(sreg[2*c],sreg[2*c+1]));
        if (!first_pv && !__all_sync(0xffffffffu,corr==1.0f)){ for(int d=0;d<HDV;++d){ uint32_t a=base+O_COL+((uint32_t)tid<<16)+d; uint32_t bb=tmem_ld(a); float o; memcpy(&o,&bb,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob);} }
        tmem_wait_st();
        if (tid==0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(mbar));
        __syncthreads();

        // PV: O += P·V
        if (warp==0){
            uint64_t vd0=make_desc(smem_u32(sV),16384,1024,2);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(!first_pv||kb>0)?1u:0u; uint32_t pa=base+P_COL+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred lead,p; elect.sync _|lead,-1; setp.ne.b32 p,%3,0;\n"
                    "  @lead tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+O_COL),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); }
            mma_commit(mbar);
        }
        mma_wait0(mbar); tmem_fence(); __syncthreads();
        first_pv = 0;
    }

    // epilogue: attention sink (s_aux) = an extra key with logit s_aux[h], value 0. Fold it into
    // the running max/sum here (contributes to the denominator, 0 to O). Then O/=l, LSE=ln(l)+m
    // (logit domain; scale already folded). tmem ld/wait are .sync.aligned -> run for ALL threads.
    const int qr = row0 + tid;
    if (nsplits>1) {
        // SplitKV: write this split's LOCALLY-normalized O and its LSE; the sink is applied once in
        // the combine kernel (folding it per-split would double-count). tmem ld runs for ALL threads.
        const int TQ = cu_q[nB];
        float invp = (l_i>0.f) ? 1.f/l_i : 0.f;
        for(int d=0;d<HDV;++d){ float o=0.f;
            if(!first_pv){ uint32_t bb=tmem_ld(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); memcpy(&o,&bb,4); }
            if (qr<seqlen_q && d<rhdv) o_part[(((size_t)split_idx*TQ + (q0+qr))*hq + h)*rhdv + d] = o*invp; }
        if (qr<seqlen_q) lse_part[((size_t)split_idx*hq + h)*TQ + (q0+qr)] = (l_i>0.f) ? (logf(l_i)+m_i) : -INFINITY;
    } else {
        // attention sink (s_aux) = an extra key with logit s_aux[h], value 0. Fold into running max/sum.
        float mfin = m_i, lfin = l_i, osc = 1.f;   // osc = extra O rescale if sink shifts the max
        if (s_aux != nullptr) {
            float sk = s_aux[h];
            float mn = fmaxf(m_i, sk);
            float c2 = (m_i==-INFINITY) ? 0.f : ex2f((m_i-mn)*LOG2E);
            lfin = l_i*c2 + ex2f((sk-mn)*LOG2E);
            osc = c2; mfin = mn;
        }
        float inv = (lfin>0.f) ? osc/lfin : 0.f;
        for(int d=0;d<HDV;++d){ float o=0.f;
            if(!first_pv){ uint32_t bb=tmem_ld(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); memcpy(&o,&bb,4); }
            if (qr < seqlen_q && d < rhdv) out[(size_t)(q0+qr)*hq*rhdv + (size_t)h*rhdv + d] = from_f<T>(o*inv); }
        if (qr < seqlen_q) lse[(size_t)h*cu_q[nB] + (q0+qr)] = (lfin>0.f) ? (logf(lfin)+mfin) : -INFINITY;
    }
    __syncthreads();
    if (warp==0) tmem_dealloc(base,TMEM_COLS);
}

// SplitKV combine: merge `nsplits` partial (O, LSE) into the final out/lse via global-max rescale.
// o_part [nsplits, total_q, hq, hdv] (each split locally normalized); lse_part [nsplits, hq, total_q].
// One thread per (q, head). The attention sink (extra key, logit s_aux[h], value 0) is applied here.
template<class T>
__global__ void combine_kernel(const float* __restrict__ o_part, const float* __restrict__ lse_part,
    T* __restrict__ out, float* __restrict__ lse, int total_q, int hq, int hdv, int rhdv,
    int nsplits, const float* __restrict__ s_aux) {
    const float LOG2E=1.4426950408889634f;
    // one thread per (q, head, d): coalesced o_part reads (adjacent d -> adjacent threads), fully
    // parallel. (Old: one thread per (q,head) looping d+splits -> underutilized + uncoalesced -> slow.)
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= total_q*hq*rhdv) return;
    int d = idx % rhdv; int qh = idx / rhdv; int h = qh % hq, qi = qh / hq;
    float G=-INFINITY;
    for(int s=0;s<nsplits;++s){ float ls=lse_part[((size_t)s*hq+h)*total_q+qi]; G=fmaxf(G,ls); }
    bool hs = (s_aux!=nullptr); float sk = hs ? s_aux[h] : 0.f;
    float g2 = hs ? fmaxf(G,sk) : G;
    if (g2==-INFINITY){ out[((size_t)qi*hq+h)*hdv+d]=from_f<T>(0.f); if(d==0) lse[(size_t)h*total_q+qi]=-INFINITY; return; }
    float L=0.f, acc=0.f;
    for(int s=0;s<nsplits;++s){ float ls=lse_part[((size_t)s*hq+h)*total_q+qi];
        if(ls>-INFINITY){ float w=ex2f((ls-g2)*LOG2E); L+=w; acc += w * o_part[(((size_t)s*total_q+qi)*hq+h)*hdv+d]; } }
    if (hs) L += ex2f((sk-g2)*LOG2E);
    out[((size_t)qi*hq+h)*hdv+d]=from_f<T>((L>0.f)?acc/L:0.f);
    if(d==0) lse[(size_t)h*total_q+qi] = g2 + logf(L);
}

// ================================ 2-SM (cta_group::2) CLUSTER PREFILL ================================
// A 2-CTA cluster collaborates on one 256-row attention tile: the pair's QK/PV are single cta_group::2
// UMMAs (M=256, distributed accumulator: rank0=rows[0:128], rank1=rows[128:256]). Each CTA loads only
// HALF of K (by keys) and V (by HDV cols) -> half the K/V smem + DRAM per CTA. Softmax/epilogue are
// per-CTA on the local 128 rows. This is the correctness-first SERIAL variant (validated compute core in
// kernels/probes/fa4_{qk,attn}_2sm.cu); a warp-specialized pipelined variant layers on top. Gated behind
// FA4_2SM for bf16/fp16 hd128|hd64 non-paged contiguous K/V. Cross-CTA sync: cluster_sync (K/V + P
// publish) + tcgen05.commit.cta_group::2.multicast (MMA-done, both CTAs).
template<class T, int HD, int HDV>
__global__ void __cluster_dims__(2,1,1) __launch_bounds__(128) fwd_kernel_2sm(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv) {
    constexpr int TM=128, TN=128, NH=TN/2, HDVH=HDV/2;   // per-CTA: TM Q rows, NH K keys, HDVH V cols
    constexpr int NKT_QK=HD/16, NKT_PV=TN/16;
    constexpr int S_COL=0, P_COL=64, O_COL=128;          // 256 tmem cols (S | P aliased@64 | O)
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK=sQ+TM*HD; T* sV=sK+(HD/64)*8192;    // sK uses koff-128 addressing (block stride 8192)
    __shared__ __align__(8) uint32_t s_base; __shared__ __align__(8) uint64_t s_mbar;
    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y, b=blockIdx.z, hk=h/rep;
    const int rank=mb&1; const bool leader=(rank==0);
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
    const int cl0=mb&~1, crow0=cl0*TM;                   // cluster's first Q-tile / first row
    if (crow0>=seqlen_q) return;                         // whole cluster is padding (uniform across the pair)
    const int row0=mb*TM;
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;

    // shared KV block range over the cluster's 2*TM rows (both CTAs compute the SAME range for the joint MMA)
    const int cqp_max=seqlen_k-seqlen_q+crow0+(2*TM-1), cqp_min=seqlen_k-seqlen_q+crow0;
    int n_lo=0, n_hi=(seqlen_k+TN-1)/TN;
    if(is_causal) n_hi=min(n_hi,cqp_max/TN+1);
    if(win_r>=0)  n_hi=min(n_hi,(cqp_max+win_r)/TN+1);
    if(win_l>=0){ int lo=cqp_min-win_l; n_lo=(lo>0)?lo/TN:0; }
    n_hi=max(n_hi,n_lo);

    // Q load: this CTA's TM rows into swizzled smem (OOB rows/cols -> 0)
    for(int i=tid;i<TM*HD;i+=128){ int r=i/HD,c=i%HD; int qr=row0+r;
        sQ[koff(r,c)]=(qr<seqlen_q && c<rhd)?q[(size_t)(q0+qr)*hq*rhd+(size_t)h*rhd+c]:from_f<T>(0.f); }
    if(warp==0){ tmem_alloc2(smem_u32(&s_base),256); tmem_relinquish2(); }   // collective across the pair
    if(tid==0) mbar_init(smem_u32(&s_mbar),1);
    asm volatile("fence.proxy.async.shared::cta;");
    cluster_sync();                                      // publish Q + tmem base + mbar across the pair
    const uint32_t base=s_base, mbar=smem_u32(&s_mbar);
    const uint32_t QKI2=(qk_id<T,TN>()&~(0x1Fu<<24))|(16u<<24);   // m_dim 8->16 (M=256 for cta_group::2)
    const uint32_t PVI2=(pv_id<T,HDV>()&~(0x1Fu<<24))|(16u<<24);

    float m_i=-INFINITY, l_i=0.f; float sreg[TN]; int first_pv=1, ph=0;
    const int qp = seqlen_k - seqlen_q + row0 + tid;     // this row's query position (causal/window mask)

    for(int nb=n_lo; nb<n_hi; ++nb){
        const int key0=nb*TN;
        // load THIS CTA's HALF of K (keys [rank*NH, +NH)) and V (dv-cols [rank*HDVH, +HDVH), all keys)
        for(int i=tid;i<NH*HD;i+=128){ int r=i/HD,c=i%HD; int kk=key0+rank*NH+r;
            sK[koff(r,c)]=(kk<seqlen_k && c<rhd)?k[(size_t)(k0+kk)*hkv*rhd+(size_t)hk*rhd+c]:from_f<T>(0.f); }
        for(int i=tid;i<TN*HDVH;i+=128){ int key=i/HDVH,dv=i%HDVH; int kk=key0+key; int gdv=rank*HDVH+dv;
            sV[voff(key,dv)]=(kk<seqlen_k && gdv<rhdv)?v[(size_t)(k0+kk)*hkv*rhdv+(size_t)hk*rhdv+gdv]:from_f<T>(0.f); }
        asm volatile("fence.proxy.async.shared::cta;");
        cluster_sync();                                  // both CTAs' K/V halves ready + cross-CTA visible

        if(leader && warp==0){                           // QK -> S (leader issues cta_group::2)
            uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2), kd0=make_desc(smem_u32(sK),16,1024,2);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb));
                mma_qk2(base+S_COL,qd,kd,QKI2,(kb>0)?1u:0u); }
            mma_commit2(mbar,0x3);
        }
        mbar_wait(mbar,ph); ph^=1; tmem_fence(); __syncthreads();

        // per-CTA online softmax over row=tid (read S fully before overwriting the aliased P half)
        float rmax=-INFINITY;
        for(int c=0;c<TN;++c){ uint32_t bb=tmem_ld(base+S_COL+((uint32_t)tid<<16)+c); float sv; memcpy(&sv,&bb,4);
            float logit=scale*sv;
            if(softcap>0.f) logit=softcap*tanhf(logit*inv_sc);
            int kk=key0+c; bool valid=(kk<seqlen_k);
            if(is_causal && kk>qp) valid=false;
            if(win_l>=0 && kk<qp-win_l) valid=false;
            if(win_r>=0 && kk>qp+win_r) valid=false;
            logit=valid?logit:-INFINITY; sreg[c]=logit; rmax=fmaxf(rmax,logit); }
        tmem_wait_ld();
        float m_new=fmaxf(m_i,rmax);
        float corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E);
        float ps=0.f;
        for(int c=0;c<TN;++c){ float p=(sreg[c]==-INFINITY)?0.f:ex2f((sreg[c]-m_new)*LOG2E); sreg[c]=p; ps+=p; }
        l_i=l_i*corr+ps; m_i=m_new;
        for(int c=0;c<TN/2;++c) tmem_st(base+P_COL+((uint32_t)tid<<16)+c, pack16<T>(sreg[2*c],sreg[2*c+1]));
        if(!first_pv && !__all_sync(0xffffffffu,corr==1.0f)){
            for(int d=0;d<HDV;++d){ uint32_t a=base+O_COL+((uint32_t)tid<<16)+d; uint32_t bb=tmem_ld(a); float o; memcpy(&o,&bb,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob); } }
        tmem_wait_st();
        cluster_sync();                                  // both CTAs' P (+rescaled O) ready + visible

        if(leader && warp==0){                           // PV -> O (leader issues cta_group::2, TS form)
            uint64_t vd0=make_desc(smem_u32(sV),16384,1024,2);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pa=base+P_COL+(uint32_t)(kb*8);
                mma_pv2(base+O_COL,pa,vd,PVI2,(!first_pv||kb>0)?1u:0u); }
            mma_commit2(mbar,0x3);
        }
        mbar_wait(mbar,ph); ph^=1; tmem_fence(); __syncthreads();
        first_pv=0;
    }

    // epilogue (this CTA's rows): attention sink (extra key logit s_aux[h], value 0) -> O/=l -> LSE
    const int qr=row0+tid;
    float mfin=m_i, lfin=l_i, osc=1.f;
    if(s_aux!=nullptr){ float sk=s_aux[h]; float mn=fmaxf(m_i,sk); float c2=(m_i==-INFINITY)?0.f:ex2f((m_i-mn)*LOG2E);
        lfin=l_i*c2+ex2f((sk-mn)*LOG2E); osc=c2; mfin=mn; }
    float inv=(lfin>0.f)?osc/lfin:0.f;
    for(int d=0;d<HDV;++d){ float o=0.f;
        if(!first_pv){ uint32_t bb=tmem_ld(base+O_COL+((uint32_t)tid<<16)+d); tmem_wait_ld(); memcpy(&o,&bb,4); }
        if(qr<seqlen_q && d<rhdv) out[(size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv+d]=from_f<T>(o*inv); }
    if(qr<seqlen_q) lse[(size_t)h*cu_q[gridDim.z]+(q0+qr)]=(lfin>0.f)?(logf(lfin)+mfin):-INFINITY;
    __syncthreads();
    if(warp==0) tmem_dealloc2(base,256);
}

// ================================ warp-specialized FAST PATH ================================
template<class T> static constexpr CUtensorMapDataType tmap_dtype();
template<> constexpr CUtensorMapDataType tmap_dtype<__nv_bfloat16>(){ return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; }
template<> constexpr CUtensorMapDataType tmap_dtype<__half>(){ return CU_TENSOR_MAP_DATA_TYPE_FLOAT16; }

// Blackwell-native pipeline (validated in kernels/probes/fa4_ws2.cu): 512 thr / 1 CTA/SM.
//   w0-3 softmax WG (setmaxnreg 192, no spill): read S -> online softmax (+mask/softcap/sink) ->
//        write P, rescale O, epilogue. w4 MMA warp: QK[i+1] issued ahead of PV[i] (overlap). w5
//        LOAD warp: TMA-streams K/V into a 2-deep ring (hardware swizzle, async, off MMA critical
//        path). Others idle. tmem: S0@0 S1@128 (double-buf) O@256 P0@384 P1@448 (512 cols).
// Non-paged contiguous K/V only (paged/splitkv/leftpad route to the serial fallback).
namespace ws {
constexpr int WTM=128, WTN=128, WST=2, WSTMAX=4;   // Q-tile rows, K-tile cols, KV ring depth (WSTMAX: max K/V stages for deeper-prefetch decode instantiations)
constexpr int FA4KVS=2;   // fwd_kernel_fa4 K/V ring depth. hd128: sQ(2 halves)=64KB + KVS*(K+V)=KVS*64KB must fit 227KB dyn smem; sO aliased into sQ. KVS=3 (192KB KV) overflows -> KVS=2 is the max that fits.
struct Bars { uint64_t kv_full[WSTMAX], kv_empty[WSTMAX], v_full[WSTMAX], v_empty[WSTMAX], s_full[2], s_empty[2], p_full[2], p_last[2], o_ready[2], corr_stats[2], o_resc[2]; };

template<class T, int HD, int HDV>
__global__ void __launch_bounds__(512,1) fwd_kernel_ws(
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
    const T* __restrict__ q, T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv) {
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64, VCB=HDV/64;
    constexpr int SCOL[2]={0,128}, OCOL=256, PCOL[2]={384,448};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[WST];
    { T* p=sQ+WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<WST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) Bars bars; __shared__ __align__(8) uint32_t s_base;
    __shared__ int prow_c[WST][WTN];   // page-row per key, precomputed once/block (breaks the page_table->K dep chain)
    __shared__ float sScale[2][2][WTM];   // corr per (tile, block-parity, row): softmax -> correction WG (w12-15)
    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y, b=blockIdx.z, hk=h/rep;
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
    const int row0=mb*WTM;
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    if (row0>=seqlen_q){  // still must partake in nothing; just return (whole CTA)
        return; }
    // n-block range (causal/local), append semantics
    const int qp_max=seqlen_k-seqlen_q+row0+(WTM-1), qp_min=seqlen_k-seqlen_q+row0;
    int n_lo=0, n_hi=(seqlen_k+WTN-1)/WTN;
    if(is_causal) n_hi=min(n_hi,(qp_max)/WTN+1);
    if(win_r>=0)  n_hi=min(n_hi,(qp_max+win_r)/WTN+1);
    if(win_l>=0){ int lo=qp_min-win_l; n_lo=(lo>0)?lo/WTN:0; }
    n_hi=max(n_hi,n_lo);
    const int NBk=n_hi-n_lo;

    if (warp<4) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n" ::: "memory");
    else        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;\n" ::: "memory");
    // Q load (all threads, plain swizzled; one-time)
    for(int i=tid;i<WTM*HD;i+=512){ int r=i/HD,c=i%HD; int qr=row0+r;
        sQ[koff(r,c)] = (qr<seqlen_q && c<rhd) ? q[(size_t)(q0+qr)*hq*rhd+(size_t)h*rhd+c] : from_f<T>(0.f); }
    if(warp==4){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.s_empty[j]),1);
                                      mbar_init(smem_u32(&bars.p_full[j]),128); mbar_init(smem_u32(&bars.o_ready[j]),1);} }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])}, SE[2]={smem_u32(&bars.s_empty[0]),smem_u32(&bars.s_empty[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])}, OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();

    if (warp==5) {                       // LOAD warp: TMA K/V ring
        int pe[WST]={0}; bool lead=(tid&31)==0;
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(KVE[s],pe[s]); pe[s]^=1;
            if(lead){ int y=k0+(n_lo+i)*WTN; uint32_t bar=KVF[s];
                mbar_expect(bar, WTN*HD*2 + WTN*HDV*2);
                for(int cb=0;cb<KCB;cb++) tma_box(sK[s]+cb*8192, &mapK, hk*rhd+cb*64, y, bar);
                for(int cb=0;cb<VCB;cb++) tma_box(sV[s]+cb*8192, &mapV, hk*rhdv+cb*64, y, bar);
            } }
    } else if (warp==4) {                // MMA warp
        int pf[WST]={0}, pse[2]={0,0}, ppf[2]={0,0};
        auto qk=[&](int i){ int bb=i&1,s=i%WST; uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2),kd0=make_desc(smem_u32(sK[s]),16,1024,2);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SCOL[bb]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); }
            mma_commit(SF[bb]); };
        auto pv=[&](int i){ int bb=i&1,s=i%WST; uint64_t vd0=make_desc(smem_u32(sV[s]),16384,1024,2);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[bb]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+OCOL),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); }
            mma_commit(OR[bb]); };
        if(NBk>0){
        mbar_wait(KVF[0],pf[0]); pf[0]^=1; mbar_wait(SE[0],pse[0]); pse[0]^=1; qk(0);
        for(int i=0;i<NBk;++i){ int s=i%WST;
            if(i+1<NBk){ int ns=(i+1)%WST,nb=(i+1)&1; mbar_wait(KVF[ns],pf[ns]); pf[ns]^=1; mbar_wait(SE[nb],pse[nb]); pse[nb]^=1; qk(i+1); }
            mbar_wait(PF[i&1],ppf[i&1]); ppf[i&1]^=1; pv(i); mma_commit(KVE[s]);
        } }
    } else if (warp<4) {                 // softmax WG (192 regs)
        int psf[2]={0,0}, por[2]={0,0};
        float m_i=-INFINITY, l_i=0.f;
        if(tid==0){ mbar_arrive(SE[0]); mbar_arrive(SE[1]); for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); }
        const int qp=seqlen_k-seqlen_q+row0+tid;
        int key0off=0;
        auto logit=[&](float sv,int c)->float{ float lg=scale*sv; if(softcap>0.f) lg=softcap*tanhf(lg*inv_sc);
            int kk=key0off+c; bool ok=(kk<seqlen_k);
            if(is_causal&&kk>qp) ok=false; if(win_l>=0&&kk<qp-win_l) ok=false; if(win_r>=0&&kk>qp+win_r) ok=false; return ok?lg:-INFINITY; };
        for(int i=0;i<NBk;++i){ int bb=i&1; key0off=(n_lo+i)*WTN;
            mbar_wait(SF[bb],psf[bb]); psf[bb]^=1; tmem_fence();
            // 2-pass .x32 softmax (P has its own tmem region -> no aliasing). SE after pass 2 (pass 2 re-reads S).
            float rmax=-INFINITY;
            for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[bb]+((uint32_t)tid<<16)+g*32, v);
                #pragma unroll
                for(int t=0;t<32;++t){ float sv; memcpy(&sv,&v[t],4); rmax=fmaxf(rmax,logit(sv,g*32+t)); } }
            tmem_wait_ld();
            float m_new=fmaxf(m_i,rmax); float corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E); float ps=0.f;
            for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[bb]+((uint32_t)tid<<16)+g*32, v);
                #pragma unroll
                for(int t=0;t<16;++t){ float s0,s1; memcpy(&s0,&v[2*t],4); memcpy(&s1,&v[2*t+1],4);
                    float l0=logit(s0,g*32+2*t), l1=logit(s1,g*32+2*t+1);
                    float p0=(l0==-INFINITY)?0.f:ex2f((l0-m_new)*LOG2E), p1=(l1==-INFINITY)?0.f:ex2f((l1-m_new)*LOG2E); ps+=p0+p1;
                    tmem_st(base+PCOL[bb]+((uint32_t)tid<<16)+g*16+t, pack16<T>(p0,p1)); } }
            tmem_wait_ld();                 // pass-2 S reads done before SE (QK i+1 may overwrite S)
            asm volatile("bar.sync 1, 128;");
            if((tid&127)==0) mbar_arrive(SE[bb]);
            l_i=l_i*corr+ps; m_i=m_new;
            if(i>0){ int pb=(i-1)&1; mbar_wait(OR[pb],por[pb]); por[pb]^=1;
                if(!__all_sync(0xffffffffu,corr==1.0f)) for(int d=0;d<HDV;++d){ uint32_t a=base+OCOL+((uint32_t)tid<<16)+d; uint32_t w=tmem_ld(a); float o; memcpy(&o,&w,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob);} }
            tmem_wait_st(); mbar_arrive(PF[bb]);   // all 128 arrive (no 2nd bar.sync)
        }
        { int pb=(NBk-1)&1; mbar_wait(OR[pb],por[pb]); por[pb]^=1; } tmem_fence();
        const int qr=row0+tid;
        // sink: extra key logit s_aux[h], value 0
        float mfin=m_i, lfin=l_i, osc=1.f;
        if(s_aux!=nullptr){ float sk=s_aux[h]; float mn=fmaxf(m_i,sk); float c2=(m_i==-INFINITY)?0.f:ex2f((m_i-mn)*LOG2E); lfin=l_i*c2+ex2f((sk-mn)*LOG2E); osc=c2; mfin=mn; }
        float inv=(lfin>0.f)?osc/lfin:0.f;
        for(int d=0;d<HDV;++d){ uint32_t w=tmem_ld(base+OCOL+((uint32_t)tid<<16)+d); tmem_wait_ld(); float o; memcpy(&o,&w,4);
            if(qr<seqlen_q && d<rhdv) out[(size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv+d]=from_f<T>(o*inv); }
        if(qr<seqlen_q) lse[(size_t)h*(cu_q[gridDim.z])+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin):-INFINITY;
    }
    __syncthreads();
    if(warp==4) tmem_dealloc(base,512);
}

// ---- 2-SM (cta_group::2) WARP-SPECIALIZED pipeline: the perf variant of fwd_kernel_2sm. ----
// A 2-CTA cluster jointly computes a 256-row Q-tile via cta_group::2 QK/PV (distributed accumulator:
// rank0 rows[0:128], rank1 rows[128:256]). Each CTA TMA-loads only its HALF of K (NH=64 keys, full HD)
// and V (all keys, HDVH=64 dv-cols) into a WST-deep ring -> half the K/V DRAM per CTA. Roles per CTA:
// w5 LOAD (2-SM TMA of own half; completion routed to LEADER's kv_full), w4 MMA (LEADER ONLY issues the
// cta_group::2 QK/PV + multicast-commits s/o/kv_empty to BOTH CTAs; the peer's w4 is idle in the loop),
// w0-3 softmax (local 128 rows, 2-pass, separate P region).
// KV producer->MMA handshake = LEADER-ANCHORED FULL BARRIER (CUTLASS copy_sm100_tma.hpp): both CTAs' TMAs
// complete_tx into the leader's kv_full (peer-bit-masked mbar), leader-only arrive_and_expect_tx w/ DOUBLED
// bytes. This (not a per-CTA barrier + generic remote-arrive) is what gives the UMMA the async->tensor proxy
// ordering to read the PEER's TMA-loaded K/V. Cross-CTA remote-arrive still used for tmem handoffs:
// s_empty, p_fwd (softmax->PV). Leader multicast-commit -> both: s_full, o_ready, kv_empty. No cluster
// barrier in steady state (one-time setup only). Serial core validated in fwd_kernel_2sm.
//
// STATUS (2026): CORRECT across 13 prefill configs (causal/noncausal/gqa/mqa/local/softcap/sink/oddseq/
// varlen/seq256..seq16k). PERF is still BELOW the 1-SM baseline (long-ctx ~0.17x vs 0.635x; short-ctx up
// to ~0.62x): the mainloop is LATENCY-bound (ncu: 25% compute, 0.32 eligible warps/sched, dominated by
// barrier + cross-CTA remote-arrive stalls). Root cause is structural: a 256-row tile consumes the full
// 512-col distributed tmem (S[128]x2 + O[128] + P[64]x2), so there is NO headroom to double-buffer a 2nd
// independent Q-tile the way the 1-SM q_stage=2 path hides softmax latency; each CTA runs a single softmax
// WG (only 6 working warps/CTA) whose SF->read->P->PV chain is serialized by two cross-CTA hops per block.
// PATH TO WIN (large): alias P into S's cols to free tmem for a 2nd staggered Q-tile per CTA, and/or
// software-pipeline the softmax S-read; deepen the K/V ring past WST=2. Gated (FA4_2SM=2); default path
// (1-SM launch_ws) is unaffected.
struct Bars2sm { uint64_t kv_full[WSTMAX], kv_empty[WSTMAX],
                          s_full[2], s_empty[2], p_full[2], p_fwd[2], o_ready[2]; };

template<class T, int HD, int HDV>
__global__ void __cluster_dims__(2,1,1) __launch_bounds__(512,1) fwd_kernel_ws_2sm(
    const __grid_constant__ CUtensorMap mapKh, const __grid_constant__ CUtensorMap mapVh,
    const T* __restrict__ q, T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv) {
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64;   // V is one 64-col block per rank (HDVH=64)
    constexpr int NH=WTN/2, HDVH=HDV/2;
    constexpr int SCOL[2]={0,128}, OCOL=256, PCOL[2]={384,448};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[WST];
    { T* p=sQ+WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=(HD/64)*8192; } for(int s=0;s<WST;s++){ sV[s]=p; p+=8192; } }
    __shared__ __align__(16) Bars2sm bars; __shared__ __align__(8) uint32_t s_base;
    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y, b=blockIdx.z, hk=h/rep;
    const int rank=mb&1; const bool leader=(rank==0);
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
    const int cl0=mb&~1;
    if (cl0*WTM>=seqlen_q) return;                         // cluster-uniform padding guard (both ranks together)
    const int row0=mb*WTM;
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    // cluster-uniform KV block range (both ranks iterate identical blocks for the joint MMA)
    const int cqp_max=seqlen_k-seqlen_q+cl0*WTM+(2*WTM-1), cqp_min=seqlen_k-seqlen_q+cl0*WTM;
    int n_lo=0, n_hi=(seqlen_k+WTN-1)/WTN;
    if(is_causal) n_hi=min(n_hi,cqp_max/WTN+1);
    if(win_r>=0)  n_hi=min(n_hi,(cqp_max+win_r)/WTN+1);
    if(win_l>=0){ int lo=cqp_min-win_l; n_lo=(lo>0)?lo/WTN:0; }
    n_hi=max(n_hi,n_lo);
    const int NBk=n_hi-n_lo;

    if (warp<4) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n" ::: "memory");
    else        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;\n" ::: "memory");
    for(int i=tid;i<WTM*HD;i+=512){ int r=i/HD,c=i%HD; int qr=row0+r;
        sQ[koff(r,c)] = (qr<seqlen_q && c<rhd) ? q[(size_t)(q0+qr)*hq*rhd+(size_t)h*rhd+c] : from_f<T>(0.f); }
    if(warp==4){ tmem_alloc2(smem_u32(&s_base),512); tmem_relinquish2(); }
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1); }
                for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.s_empty[j]),2);
                                      mbar_init(smem_u32(&bars.p_full[j]),128); mbar_init(smem_u32(&bars.p_fwd[j]),1); mbar_init(smem_u32(&bars.o_ready[j]),1); } }
    asm volatile("fence.proxy.async.shared::cta;");
    cluster_sync();                                        // one-time: publish Q + tmem base + mbars across the pair
    const uint32_t base=s_base;
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])}, SE[2]={smem_u32(&bars.s_empty[0]),smem_u32(&bars.s_empty[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])}, PFW[2]={smem_u32(&bars.p_fwd[0]),smem_u32(&bars.p_fwd[1])};
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    const uint32_t QKI2=(qk_id<T,WTN>()&~(0x1Fu<<24))|(16u<<24), PVI2=(pv_id<T,HDV>()&~(0x1Fu<<24))|(16u<<24);

    if (warp==5) {                       // LOAD warp (both CTAs): 2-SM TMA of own K/V half -> leader's kv_full
        int pe[WST]={0}; bool ld=(tid&31)==0;
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(KVE[s],pe[s]); pe[s]^=1;
            if(ld){ int ky=k0+(n_lo+i)*WTN + rank*NH, vy=k0+(n_lo+i)*WTN; uint32_t bar=KVF[s];
                if(leader) mbar_expect(bar, 2*(NH*HD*2 + WTN*HDVH*2));   // leader-only: DOUBLED (both CTAs complete here)
                for(int cb=0;cb<KCB;cb++) tma_box_2sm(sK[s]+cb*8192, &mapKh, hk*rhd+cb*64, ky, bar);
                tma_box_2sm(sV[s], &mapVh, hk*rhdv+rank*HDVH, vy, bar);
            } }
    } else if (warp==4 && leader) {      // MMA warp (LEADER ONLY): cta_group::2 QK/PV, multicast s/o/kv_empty
        int pf[WST]={0}, pse[2]={0,0}, ppf[2]={0,0}, ppfw[2]={0,0};
        auto qk=[&](int i){ int bb=i&1,s=i%WST; uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2),kd0=make_desc(smem_u32(sK[s]),16,1024,2);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb));
                mma_qk2(base+SCOL[bb],qd,kd,QKI2,(kb>0)?1u:0u); }
            mma_commit2(SF[bb],0x3); };
        auto pv=[&](int i){ int bb=i&1,s=i%WST; uint64_t vd0=make_desc(smem_u32(sV[s]),16384,1024,2);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pa=base+PCOL[bb]+(uint32_t)(kb*8);
                mma_pv2(base+OCOL,pa,vd,PVI2,(i>0||kb>0)?1u:0u); }
            mma_commit2(OR[bb],0x3); };
        if(NBk>0){
        mbar_wait(KVF[0],pf[0]); pf[0]^=1; mbar_wait(SE[0],pse[0]); pse[0]^=1; qk(0);
        for(int i=0;i<NBk;++i){ int s=i%WST;
            if(i+1<NBk){ int ns=(i+1)%WST,nb=(i+1)&1; mbar_wait(KVF[ns],pf[ns]); pf[ns]^=1; mbar_wait(SE[nb],pse[nb]); pse[nb]^=1; qk(i+1); }
            mbar_wait(PF[i&1],ppf[i&1]); ppf[i&1]^=1; mbar_wait(PFW[i&1],ppfw[i&1]); ppfw[i&1]^=1; pv(i); mma_commit2(KVE[s],0x3);
        } }
    } else if (warp<4) {                 // softmax WG (both ranks): local 128 rows, 2-pass, separate P
        int psf[2]={0,0}, por[2]={0,0};
        float m_i=-INFINITY, l_i=0.f;
        if(tid==0){ if(leader){ mbar_arrive(SE[0]); mbar_arrive(SE[1]); } else { mbar_arrive_remote(SE[0],0); mbar_arrive_remote(SE[1],0); }
                    for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); }
        const int qp=seqlen_k-seqlen_q+row0+tid; int key0off=0;
        auto logit=[&](float sv,int c)->float{ float lg=scale*sv; if(softcap>0.f) lg=softcap*tanhf(lg*inv_sc);
            int kk=key0off+c; bool ok=(kk<seqlen_k);
            if(is_causal&&kk>qp) ok=false; if(win_l>=0&&kk<qp-win_l) ok=false; if(win_r>=0&&kk>qp+win_r) ok=false; return ok?lg:-INFINITY; };
        for(int i=0;i<NBk;++i){ int bb=i&1; key0off=(n_lo+i)*WTN;
            mbar_wait(SF[bb],psf[bb]); psf[bb]^=1; tmem_fence();
            float rmax=-INFINITY;
            for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[bb]+((uint32_t)tid<<16)+g*32, v);
                #pragma unroll
                for(int t=0;t<32;++t){ float sv; memcpy(&sv,&v[t],4); rmax=fmaxf(rmax,logit(sv,g*32+t)); } }
            tmem_wait_ld();
            float m_new=fmaxf(m_i,rmax); float corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E); float ps=0.f;
            for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[bb]+((uint32_t)tid<<16)+g*32, v);
                #pragma unroll
                for(int t=0;t<16;++t){ float s0,s1; memcpy(&s0,&v[2*t],4); memcpy(&s1,&v[2*t+1],4);
                    float l0=logit(s0,g*32+2*t), l1=logit(s1,g*32+2*t+1);
                    float p0=(l0==-INFINITY)?0.f:ex2f((l0-m_new)*LOG2E), p1=(l1==-INFINITY)?0.f:ex2f((l1-m_new)*LOG2E); ps+=p0+p1;
                    tmem_st(base+PCOL[bb]+((uint32_t)tid<<16)+g*16+t, pack16<T>(p0,p1)); } }
            tmem_wait_ld();
            asm volatile("bar.sync 1, 128;");
            if((tid&127)==0){ if(leader) mbar_arrive(SE[bb]); else mbar_arrive_remote(SE[bb],0); }   // S read done -> free for QK i+2
            l_i=l_i*corr+ps; m_i=m_new;
            if(i>0){ int pb=(i-1)&1; mbar_wait(OR[pb],por[pb]); por[pb]^=1;
                if(!__all_sync(0xffffffffu,corr==1.0f)) for(int d=0;d<HDV;++d){ uint32_t a=base+OCOL+((uint32_t)tid<<16)+d; uint32_t w=tmem_ld(a); float o; memcpy(&o,&w,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob);} }
            tmem_wait_st();
            if(leader) mbar_arrive(PF[bb]);                  // rank0: 128 arrivals -> p_full
            else { asm volatile("bar.sync 2, 128;"); if((tid&127)==0) mbar_arrive_remote(PFW[bb],0); }  // rank1: one forward after all P written
        }
        { int pb=(NBk-1)&1; if(NBk>0){ mbar_wait(OR[pb],por[pb]); por[pb]^=1; } } tmem_fence();
        const int qr=row0+tid;
        float mfin=m_i, lfin=l_i, osc=1.f;
        if(s_aux!=nullptr){ float sk=s_aux[h]; float mn=fmaxf(m_i,sk); float c2=(m_i==-INFINITY)?0.f:ex2f((m_i-mn)*LOG2E); lfin=l_i*c2+ex2f((sk-mn)*LOG2E); osc=c2; mfin=mn; }
        float inv=(lfin>0.f)?osc/lfin:0.f;
        for(int d=0;d<HDV;++d){ float o=0.f; if(NBk>0){ uint32_t w=tmem_ld(base+OCOL+((uint32_t)tid<<16)+d); tmem_wait_ld(); memcpy(&o,&w,4); }
            if(qr<seqlen_q && d<rhdv) out[(size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv+d]=from_f<T>(o*inv); }
        if(qr<seqlen_q) lse[(size_t)h*(cu_q[gridDim.z])+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin):-INFINITY;
    }
    __syncthreads();
    cluster_sync();                                        // both ranks done reading distributed tmem before collective dealloc
    if(warp==4) tmem_dealloc2(base,512);
}

// Build a 2D 128B-swizzle CUtensorMap over a contiguous [total, heads, hd] tensor viewed as
// [total, heads*hd]; box [64, ROWS]. ROWS=WTN loads a full K/V tile; ROWS=WTN/2 loads a 2-SM key half.
template<class T>
static CUtensorMap ws_map_rows(const at::Tensor& t, uint32_t rows){
    CUtensorMap map{}; int64_t total=t.size(0), heads=t.size(1), hd=t.size(2);
    uint64_t gdim[2]={(uint64_t)(heads*hd),(uint64_t)total};
    uint64_t gstride[1]={(uint64_t)(t.stride(0)*(int64_t)sizeof(T))};
    uint32_t bdim[2]={64u,rows}; uint32_t estride[2]={1u,1u};
    CUresult cr=cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 2, t.data_ptr(), gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr==CUDA_SUCCESS, "ws cuTensorMapEncodeTiled(rows) failed: ", (int)cr);
    return map;
}

// Build a 2D 128B-swizzle CUtensorMap over a contiguous [total, heads, hd] tensor viewed as
// [total, heads*hd]; box [64, WTN]. Matches the koff/voff swizzle the MMA descriptors read.
template<class T>
static CUtensorMap ws_map(const at::Tensor& t){
    CUtensorMap map{}; int64_t total=t.size(0), heads=t.size(1), hd=t.size(2);
    uint64_t gdim[2]={(uint64_t)(heads*hd),(uint64_t)total};
    uint64_t gstride[1]={(uint64_t)(t.stride(0)*(int64_t)sizeof(T))};
    uint32_t bdim[2]={64u,(uint32_t)WTN}; uint32_t estride[2]={1u,1u};
    CUresult cr=cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 2, t.data_ptr(), gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(cr==CUDA_SUCCESS, "ws cuTensorMapEncodeTiled failed: ", (int)cr);
    return map;
}

// Gather4 tensormap over a physical paged K/V pool viewed 2D as [num_rows, row_width] (row_width=hkv*hd,
// innermost/contiguous). Box [64,1] (gather4 requires the row dim be 1), 128B swizzle. Driven per-tile by
// tma_gather4 with x=hk*hd+cb*64 and 4 physical-row y-coords. cr!=SUCCESS -> caller falls back to cp.async.
template<class T>
static CUtensorMap ws_map_gather(void* ptr, int64_t num_rows, int64_t row_width, CUresult& cr){
    CUtensorMap map{};
    uint64_t gdim[2]={(uint64_t)row_width,(uint64_t)num_rows};
    uint64_t gstride[1]={(uint64_t)(row_width*(int64_t)sizeof(T))};
    uint32_t bdim[2]={64u,1u}; uint32_t estride[2]={1u,1u};
    cr=cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 2, ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

// PAGE-ALIGNED tensormap over the same physical paged pool viewed 2D as [num_rows, row_width], but with a
// box of `segrows` ROWS instead of 1. A KV tile of WTN keys spanning contiguous physical pages is then one
// `cp.async.bulk.tensor.2d` per (page-segment, 64-col block) -- 8 TMA issues per hd128 tile at page_size=64,
// versus 128 gather4 issues or ~4096 cp.async issues. Requires segrows % 8 == 0 so each box lands on a
// 128B-swizzle atom boundary (koff/voff XOR (r&7)<<3 is then box-relative == absolute).
template<class T>
static CUtensorMap ws_map_pages(void* ptr, int64_t num_rows, int64_t row_width, uint32_t segrows, CUresult& cr){
    CUtensorMap map{};
    uint64_t gdim[2]={(uint64_t)row_width,(uint64_t)num_rows};
    uint64_t gstride[1]={(uint64_t)(row_width*(int64_t)sizeof(T))};
    uint32_t bdim[2]={64u,segrows}; uint32_t estride[2]={1u,1u};
    cr=cuTensorMapEncodeTiled(&map, tmap_dtype<T>(), 2, ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

// cuTensorMapEncodeTiled is a driver call on the launch critical path and a paged decode launch is only
// ~100us of GPU work, so memoize the page descriptors per (pool ptr, shape, segrows). Single-threaded per
// process (the harness runs one scenario per process/GPU at a time), so a plain direct cache is fine.
template<class T>
static CUtensorMap ws_page_map_cached(void* ptr, int64_t num_rows, int64_t row_width,
                                      uint32_t segrows, CUresult& cr){
    struct E { void* p; int64_t r, w; uint32_t s; CUtensorMap m; bool ok; };
    static E cache[8]{}; static unsigned nxt=0;
    for(int i=0;i<8;++i) if(cache[i].ok && cache[i].p==ptr && cache[i].r==num_rows
                            && cache[i].w==row_width && cache[i].s==segrows){ cr=CUDA_SUCCESS; return cache[i].m; }
    CUtensorMap m=ws_map_pages<T>(ptr,num_rows,row_width,segrows,cr);
    if(cr==CUDA_SUCCESS){ E& e=cache[nxt&7]; ++nxt; e.p=ptr; e.r=num_rows; e.w=row_width; e.s=segrows; e.m=m; e.ok=true; }
    return m;
}

// softmax warpgroups give the scheduler independent work to hide tmem scoreboard stalls (NCU: the
// q_stage=1 path is barrier-bound, tensor idle). Roles: w0-3 softmax tile0, w4-7 softmax tile1,
// w8 MMA, w9 load. tmem(512 cols): S0@0 S1@128 (P aliased in-place at +64), O0@256 O1@384. Both
// tiles iterate the UNION KV block range; a branchless per-row [clo,chi] valid-column range masks
// per-tile causal/local divergence (a tile harmlessly processes extra blocks -> p=0). Grid halved (CTA = 2 Q-tiles).
//
// ================= ROOT CAUSE (2026, apples-to-apples ncu, dense hd128 causal/prefill) =================
// The prefill kernel is LATENCY-bound, NOT instruction- or occupancy-bound. Proof (SpeedOfLight, seq16k/causal):
//   DRAM 3.6% (not mem-bound), Eligible-warps/sched 0.18, No-Eligible 86%, tensor-pipe ~6-15% active. The SM
//   simply has no eligible warp ~86% of the time -- it is stalled on the QK->softmax->PV producer/consumer chain.
//   Occupancy is NOT the lever: we run 25% (1 CTA/SM, reg-bound: 512thr x up-to-192-reg softmax = full RF), the
//   CUTLASS oracle runs at 12% yet gets 2x the eligible warps -- its edge is per-warp STALL REDUCTION (CuTe
//   software-pipelines loads ahead of use), which single-buffered tmem (S0,S1,O0,O1 = full 512 cols) blocks us
//   from fully matching (can't hold 2x f[128] to prefetch the S read: 256 regs > 192 budget).
// --- WHAT MOVED THE NEEDLE (this pass, wall-clock; instruction cuts alone did ~nothing since latency-bound): ---
//   (1) PACKED bf16 conversion: pack16 now = ONE cvt.rn.bf16x2.f32 (was 2 scalar cvt + PRMT). The P fp32->bf16
//       for the PV MMA was ~15% of steady-state time. BIGGEST win: seq16k 0.37->0.54, large4k 0.27->0.34.
//   (2) SPILL KILL: moved the online O-rescale to AFTER the exp loop (f[128] dead -> og[32] no longer coexists
//       -> no 192-reg spill) + rolled tmem_o_rescale (no 4-chunk pipeline = 128 live regs). causal 0.26->0.29.
//   (3) VECTORIZED Q load: 128-bit uint4 (8 elems/iter) vs scalar -- the manual Q load was ~27% of dynamic inst
//       (koff swizzle + 64b addr x64 iters x16 warps, once/CTA, dominates short-context). causal 0.24->0.26.
// --- STEADY-STATE (seq16k, fully amortized) top stalls now: mbar_wait spin 22% (real producer/consumer wait,
//     mostly MMA waiting on the slow softmax) + tmem_wait::ld 16% (S read latency) + MMA qk/pv long_sb 17%
//     (single-buffer WAR: qk(i+1) into S waits for softmax to free it). All 3 are the shallow-pipeline latency.
//   * The "barrier" stall is a CONFIRMED HARMLESS ARTIFACT (idle MMA lanes 1-31 at if(lead8) reconvergence): a
//     converged @lead8-predicated mma removes the metric but is NEUTRAL-to-WORSE (seq16k 0.37->0.345). Not it.
//   * order_s stagger and the correction-WG (CORR) are both NEUTRAL for causal: the 2 softmax tiles already
//     overlap and the CS/CO handshake costs as much as the O-rescale it offloads.
// PATH TO FULL PARITY (remaining, LARGE): break the single-buffer tmem coupling -- either a 2-CTA cluster
//   (cta_group::2) MMA like FA4/CUTLASS (distributed tmem across the cluster frees room to double-buffer S and
//   software-pipeline the softmax S read), or a 1-tile double-buffered-S kernel at 2 CTA/SM. CUTLASS
//   77_blackwell_fmha builds here (CUDA13/sm100a, /tmp/fmha) as the distill oracle.
// KEPT: elect-hoist in the MMA warp (elect.sync once via lane0, not per sub-MMA) -33 elect/block -> -1M inst.
// ---- (superseded) PROFILED prefill gap (ncu 2025.3, ours vs FA4 FlashAttentionForwardSm100, dense hd128 causal) ----
// BOTH run 512thr/128reg/1-CTA-per-SM @ ~25% occ (occupancy is NOT the lever). tensor-pipe-active ours 7% vs
// FA4 40%. Eligible-warps/sched 0.29, No-Eligible 78% => LATENCY-bound: not enough warps to hide the
// QK->softmax->PV chain at 25% occ. *** CORRECTED per-line stall attribution (smsp__pcsamp, source page): ***
//  - "barrier" 7.3 avg is 98.7% just the 6 IDLE warps (w10-15) parked at EXIT/__syncthreads. Pure averaging
//    artifact over 16 warps -- NOT on the critical path of the ~10 working warps. Do not chase it.
//  - "long_scoreboard" 6.9 (the real #1) attributes to the ISETP.GE/PRMT/FMAX that CONSUME the softmax S read
//    => it is TMEM-READ LATENCY of S (tcgen05 tmem loads sit on the LONG scoreboard, not short). The softmax
//    reads 128 fp32/row from tmem and the max/pack/exp stall on it. FA4's 6.9->2.7 edge = it keeps ALL 16
//    warps busy (correction WG rescales O EVERY block) so the scheduler always has an eligible warp to cover
//    that latency, whereas we left 6 warps idle and the old CORR skipped work when corr==1.
// KEPT WINS this pass: (1) MMA loop reorder -- issues qk(0,i+1) right after pv(0,i), matching FA4's exact MMA
//   issue schedule [qk0,qk1,pv0,qk0',pv1,qk1',pv0',...] (large4k 0.233->0.269, seq16k 0.302->0.376).
//   (2) correction WG (CORR): rescales O every block off the softmax path (tmem_o_rescale) to keep w12-15
//   busy. HISTORICAL WIN, NOW REVERTED TO OFF-BY-DEFAULT: once the softmax became spill-free (softmax_full_ptx,
//   192-reg PTX fragment) the CS/CO cross-WG handshake + split_P choreography costs MORE than the inline
//   O-rescale it offloads. Re-measured 2026 (CORR ON vs OFF): large4k 0.548 vs 0.595, seq8k 0.626 vs 0.702,
//   seq16k 0.654 vs 0.733 -- CORR is now a NET LOSS on every dense hd128 config. Gated OFF in launch_ws
//   (opt-in via FA4_CORR / FA4_CORR_TH). ncu (non-CORR large4k): long_scoreboard 3.69 (tmem S-read, #1),
//   barrier 5.09 (idle-warp artifact), wait 2.13; latency-bound at 25% occ (reg+smem capped, 1 CTA/SM).
// TRIED+REVERTED: correction on <4096 (too few blocks to overlap -> neutral/slightly worse); routing hd128 to
//   q_stage=1 (fewer active warps); 1-pass softmax in the q_stage=1 path (192reg single WG spills);
//   FA4 ORDERED SOFTMAX STAGGER (OrderBarrierSoftmax 2-group ping-pong of the exp so one group's S read
//   overlaps the other's exp) -- neutral-to-negative at every arrive point (seq16k 0.376->0.366, rest +-0.003):
//   our two softmax groups are INDEPENDENT tiles the scheduler already interleaves, so the explicit barrier
//   only adds sync. FA4 needs it because its 2 groups are stacked halves of ONE Q-tile over shared K blocks.
// P is aliased into the LOWER half of S (tmem full at 512 cols: S0,S1,O0,O1) => WAR: qk(i+1) can't overwrite
// S until pv(i) reads P; note FA4 has the SAME aliasing, it just fills the gap with the other group's MMA.
// REMAINING 3-4x gap: HOLISTIC-PORT INVESTIGATION CONCLUSION. Read the FA4 collective end-to-end and traced
// gemm_reset_zero_acc (fmha_common.hpp): the accumulate_ flag is Zero only on the FIRST pv of each group then
// One -> O0/O1 ACCUMULATE across blocks in tmem, and the correction WG rescales O *= corr in place. Tmem
// layout (S0/S1/O0/O1, P=S+kSizeP, V=stats@S) matches ours EXACTLY. => FA4's algorithm + pipeline structure
// are IDENTICAL to this kernel (accumulate-PV + correction-rescale + 2-group stacked + P aliased in S). The
// gap is therefore NOT architecture -- it is CuTe/ptxas CODEGEN at our 25% occ. Every distillable mechanism
// was tested here: MMA-schedule reorder (kept, +), correction WG (kept, + on large-ctx), ordered softmax
// stagger (neutral -> reverted: scheduler already interleaves our independent tiles), 4-way ILP max/sum
// accumulators (REVERTED: regressed -- extra regs blow the 192 budget; ptxas already schedules the unrolled
// single chain fine). Our softmax is register-bound at 192 (128 for the f[128] fragment) and tmem-read-
// latency bound; adding FA4 structure only adds pressure. The last untested lever is 2-CTA/SM (more eligible
// warps to hide the S read) but tmem=512cols/SM forces q_stage=1 (known slower) -> unlikely net win. True
// 1.0x prefill parity effectively requires the CUTLASS/CuTe collective itself (its codegen), not more hand
// -PTX bolted onto this already-equivalent kernel.
template<class T, int HD, int HDV, bool CORR=false, int VST=WST>
__global__ void __launch_bounds__(512,1) fwd_kernel_ws2(
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
    const T* __restrict__ q, T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv) {
    // q_stage=2 warp-specialized forward: one CTA owns an m_block PAIR (2 Q-tiles of WTM rows), sharing the
    // K/V ring so each gathered block feeds both tiles. Static grid ((max_mq+1)/2, hq, batch).
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64, VCB=HDV/64;
    const int SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};   // P aliased into LOWER half of S (spill-free 2-pass softmax)
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[VST];
    { T* p=sQ+2*WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<VST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) Bars bars; __shared__ __align__(8) uint32_t s_base;
    __shared__ int prow_c[WST][WTN];   // page-row per key, precomputed once/block (breaks the page_table->K dep chain)
    __shared__ float sScale[2][2][WTM];   // corr per (tile, block-parity, row): softmax -> correction WG (w12-15)
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    // Pre-scale Q so S=Q*K lands directly in the softmax domain: *scale*log2e when no softcap (softmax
    // becomes p=ex2(S-m), zero scale/LOG2E FMAs per element); *scale when softcap (tanh needs the raw
    // scaled logit, log2e folded later). Removes ~3 FMA/element from the hot loop.
    const float qsc=(softcap>0.f)?scale:scale*LOG2E;
    // ---- ONE-TIME setup (tmem alloc + producer reg dec + constant barrier handles) ----
    // 1-pass FULL-block softmax holds the 128-fp32 S row in an SROA-split register fragment -> ~150 regs.
    // Producers dec to 64 at the top (gated warp>=8, aligned per WG). Softmax warps inc to 192 IN-BRANCH
    // (first stmt inside the warp<8 branch) — top-level inc is ineffective (ptxas keeps 128 -> spill).
    // MEASURED (2026-07-07): bumping softmax past 192 (208/216, freeing idle WG3 to 24) REGRESSES ~0.62->0.51
    // on all 3 short configs — the S-fragment spill is CHEAP (overlapped L1), and >192 regs makes ptxas
    // schedule the softmax worse. 192/64 is the sweet spot; the pool is not the lever.
    // Register split. Non-CORR: 2-tier (softmax 192 / producers 64). CORR: 3-tier AVO v33 rebalance
    // 184/88/56 (softmax DOWN 192->184, correction UP 64->88 to stop its O-read spill, MMA/load 56).
    // Accounting (baseline 128/thr, __launch_bounds__(512,1)): softmax claim 56*256=14336 == release
    // WG2 72*128=9216 + WG3 40*128=5120. Exact balance; decs at top precede the softmax inc -> no deadlock.
    if constexpr(CORR){
        if      (warp>=8 && warp<12) asm volatile("setmaxnreg.dec.sync.aligned.u32 " AVO_STR(AVO_PROD_REG) ";\n" ::: "memory"); // MMA/load
        else if (warp>=12)           asm volatile("setmaxnreg.dec.sync.aligned.u32 " AVO_STR(AVO_CORR_REG) ";\n" ::: "memory"); // correction WG
    } else {
        if (warp>=8) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");
    }
    if(warp==8){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base=s_base;
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();
    // LPT launch order (causal): heavy tiles (high mb = many KV blocks) get the LOW blockIdx.x so the HW
    // launches them FIRST -> they finish alongside the many light tiles instead of straggling at the end
    // (the ~25% partial-wave tail is dominated by the last-launched heavy CTAs). Non-causal: order-neutral.
    const int mb=is_causal ? (gridDim.x-1-blockIdx.x) : blockIdx.x, h=blockIdx.y, b=blockIdx.z, hk=h/rep;
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
    // q_stage=2 m-block PAIRING. Causal: pair LIGHT tile m0=mb with HEAVY tile m1=mbc-1-mb so EVERY CTA does
    // ~balanced work (heavy tile keeps the pipeline warm >= mbc/2 blocks). Adjacent pairing (2mb,2mb+1) gave
    // early CTAs ~1 block -> pipeline never warmed -> 17% tensor on short causal. Non-causal: adjacent (no
    // triangle; the shared K/V range is the full seqlen for both tiles anyway).
    const int mbc=(seqlen_q+WTM-1)/WTM;
    const int m0 = 2*mb;   // adjacent pairing (distant light+heavy pairing was measured WORSE: tile0 idles for
    const int m1 = 2*mb+1; // the heavy tile's tail, losing q_stage=2 K/V sharing; short-config gap is per-CTA
    (void)mbc;             // pipeline efficiency, not load imbalance — persistence(cont) & balancing both fail.
    const int row0_0=m0*WTM, row0_1=m1*WTM;
    if (m0>m1 || row0_0>=seqlen_q){ __syncthreads(); if(warp==8) tmem_dealloc(base,512); return; }   // dup/idle CTA
    const bool a1=(m1>m0) && (row0_1<seqlen_q);
    int nlo0,nhi0,nlo1,nhi1;
    { int qx=seqlen_k-seqlen_q+row0_0+(WTM-1), qn=seqlen_k-seqlen_q+row0_0; nlo0=0; nhi0=(seqlen_k+WTN-1)/WTN;
      if(is_causal) nhi0=min(nhi0,qx/WTN+1); if(win_r>=0) nhi0=min(nhi0,(qx+win_r)/WTN+1);
      if(win_l>=0){ int l=qn-win_l; nlo0=(l>0)?l/WTN:0; } nhi0=max(nhi0,nlo0); }
    if(a1){ int qx=seqlen_k-seqlen_q+row0_1+(WTM-1), qn=seqlen_k-seqlen_q+row0_1; nlo1=0; nhi1=(seqlen_k+WTN-1)/WTN;
      if(is_causal) nhi1=min(nhi1,qx/WTN+1); if(win_r>=0) nhi1=min(nhi1,(qx+win_r)/WTN+1);
      if(win_l>=0){ int l=qn-win_l; nlo1=(l>0)?l/WTN:0; } nhi1=max(nhi1,nlo1); }
    else { nlo1=nlo0; nhi1=nlo0; }   // tile1 empty (skipped): hi1 = nhi1 - n_lo = 0
    const int n_lo=min(nlo0,nlo1), n_hi=max(nhi0,nhi1);
    const int NBk=n_hi-n_lo;

    // VECTORIZED Q load (128-bit / 8 elems per iter). 8 consecutive head-dim cols map to 8 consecutive swizzled
    // smem slots (koff XORs by a multiple of 8 and [c,c+8) never crosses a 64-col group boundary for 8-aligned c).
    if((rhd&7)==0){
        // SOFTWARE-PIPELINED Q load: issue ALL NIT global loads first (independent -> overlap in flight), THEN
        // convert+store. The old per-iter load->convert->store serialized each LDG.128's latency.
        constexpr int NIT = 32*HD/512;
        uint4 vi[NIT]; int dst[NIT]; bool vld[NIT];
        #pragma unroll
        for(int it=0; it<NIT; it++){ int iv=tid+it*512; int e=iv<<3; int r=e/HD,c=e%HD; int tt=r>>7, lr=r&127; int qr=(tt?m1:m0)*WTM+lr;
            vld[it]=(qr<seqlen_q && c<rhd); dst[it]=tt*WTM*HD + koff(lr,c);
            if(vld[it]) vi[it]=*(const uint4*)&q[(size_t)(q0+qr)*hq*rhd+(size_t)h*rhd+c]; }
        #pragma unroll
        for(int it=0; it<NIT; it++){ uint4 vo;
            if(vld[it]){ T* pi=(T*)&vi[it]; T* po=(T*)&vo;
                #pragma unroll
                for(int k=0;k<8;k++) po[k]=from_f<T>(to_f<T>(pi[k])*qsc); }
            else vo=make_uint4(0u,0u,0u,0u);
            *(uint4*)&sQ[dst[it]] = vo; }
    } else {
        for(int i=tid;i<2*WTM*HD;i+=512){ int r=i/HD,c=i%HD; int tt=r>>7, lr=r&127; int qr=(tt?m1:m0)*WTM+lr;
            sQ[tt*WTM*HD + koff(lr,c)] = (qr<seqlen_q && c<rhd) ? from_f<T>(to_f<T>(q[(size_t)(q0+qr)*hq*rhd+(size_t)h*rhd+c])*qsc) : from_f<T>(0.f); }
    }
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int s=0;s<VST;s++){ mbar_init(smem_u32(&bars.v_full[s]),1); mbar_init(smem_u32(&bars.v_empty[s]),1);}
                for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.p_full[j]),128); mbar_init(smem_u32(&bars.p_last[j]),128); mbar_init(smem_u32(&bars.o_ready[j]),1);
                                      mbar_init(smem_u32(&bars.corr_stats[j]),128); mbar_init(smem_u32(&bars.o_resc[j]),128);} }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    // Barrier handles recomputed per tile (constant smem addrs, ~free) so they don't stay live across the loop
    // back-edge and steal registers from the softmax 192-reg fragment (that liveness cost ~0.09x).
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t VVF[VST],VVE[VST]; for(int s=0;s<VST;s++){ VVF[s]=smem_u32(&bars.v_full[s]); VVE[s]=smem_u32(&bars.v_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
    uint32_t PL[2]={smem_u32(&bars.p_last[0]),smem_u32(&bars.p_last[1])};   // split_P: last-1/4 P ready (CORR path)
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    uint32_t CS[2]={smem_u32(&bars.corr_stats[0]),smem_u32(&bars.corr_stats[1])};
    uint32_t CO[2]={smem_u32(&bars.o_resc[0]),smem_u32(&bars.o_resc[1])};

    if (warp==9) {                       // LOAD warp: TMA K ring (shared by both Q-tiles); V is loaded by w10
        int pe[WST]={0}; bool lead=(tid&31)==0;
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(KVE[s],pe[s]); pe[s]^=1;
            if(lead){ int y=k0+(n_lo+i)*WTN; uint32_t bar=KVF[s];
                mbar_expect(bar, WTN*HD*2);
                for(int cb=0;cb<KCB;cb++) tma_box(sK[s]+cb*8192, &mapK, hk*rhd+cb*64, y, bar);
            } }
    } else if (warp==10) {               // V-LOAD warp: TMA V ring (depth VST, decoupled from K so hd192 fits smem)
        int pe[VST]={0}; bool lead=(tid&31)==0;
        for(int i=0;i<NBk;++i){ int s=i%VST; mbar_wait(VVE[s],pe[s]); pe[s]^=1;
            if(lead){ int y=k0+(n_lo+i)*WTN; uint32_t bar=VVF[s];
                mbar_expect(bar, WTN*HDV*2);
                for(int cb=0;cb<VCB;cb++) tma_box(sV[s]+cb*8192, &mapV, hk*rhdv+cb*64, y, bar);
            } }
    } else if (warp==8) {                // MMA warp: QK0,QK1 then PV0,PV1 per block
        int pf[WST]={0}, ppf[2]={0,0}, pco[2]={0,0}, ppl[2]={0,0};
        // The MMA warp is fully converged before every qk/pv, so lane 0 is always the elect.sync leader.
        // Hoist the election to ONE uniform lane check instead of re-electing per sub-MMA (NKT_QK+NKT_PV=16
        // elect.sync/block/tile). elect.sync generates ELECT/VOTEU/R2UR/PLOP3 uniform setup each time; this
        // was ~half the MMA-warp instruction stream (ncu). tcgen05.mma issued from lane 0 is standard; the
        // per-tile mma_commit still elect.syncs, providing the warp convergence barrier.
        const bool lead8=(tid&31)==0;
        // HOIST descriptors (lever a): the QK/PV matrix descriptors are pure functions of the FIXED shared
        // addresses (sQ per tile, sK[s]/sV[s] per ring stage), so precompute the base descriptors ONCE here
        // instead of make_desc'ing them inside qk/pv every block. Removes 6 make_desc/block off the MMA warp,
        // which is on the QK->S critical path that stalls the softmax. WST=2 -> 6 uint64 (~12 regs), cheap.
        // NB: the mma loop stays branched under if(lead8) (only lane0 runs desc_add + issues the async
        // tcgen05.mma). ncu attributes ~98.7% of the kernel's "barrier" stall to desc_add:97 -- but that is a
        // HARMLESS METRIC ARTIFACT: it is lanes 1-31 idle-waiting at the if(lead8) reconvergence for lane0,
        // NOT on lane0's critical path (confirmed: the real bottleneck is softmax long_scoreboard = tmem-read
        // latency + mbar_wait, per source-level pcsamp). A converged @l-predicated variant only removes the
        // (fake) metric while adding per-lane desc_add work, so it is not worth it.
        uint64_t qd0a[2], kd0a[WST], vd0a[VST];
        #pragma unroll
        for(int j=0;j<2;j++) qd0a[j]=make_desc(smem_u32(sQ+j*WTM*HD),16,1024,2);
        #pragma unroll
        for(int s=0;s<WST;s++){ kd0a[s]=make_desc(smem_u32(sK[s]),16,1024,2); }
        #pragma unroll
        for(int s=0;s<VST;s++){ vd0a[s]=make_desc(smem_u32(sV[s]),16384,1024,2); }
        auto qk=[&](int j,int i){ int s=i%WST;
            if(lead8){ uint64_t qd0=qd0a[j],kd0=kd0a[s];
                for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                        ::"r"(base+SCOL[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); } }
            mma_commit(SF[j]); };
        // split_P (FA4 choreography, CORR only): softmax releases the first 3/4 of P (PF) mid-store so PV can
        // start its first PSPLIT k-subtiles while softmax finishes the last 1/4 (PL). Requires O-rescale OFF the
        // softmax path (correction WG), else PV would accumulate into un-rescaled O. Non-CORR keeps the single PF.
        constexpr int PSPLIT = (WTN*3/4)/16;   // 6 subtiles = keys 0..95 (WTN=128); FA4 split_P_arrive=96
        constexpr int PV_A   = CORR ? PSPLIT : NKT_PV;
        auto pv=[&](int j,int i){ int s=i%VST;
            if(lead8){ uint64_t vd0=vd0a[s];
                #pragma unroll
                for(int kb=0;kb<PV_A;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                        ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); } }
            if constexpr(CORR){ mbar_wait(PL[j],ppl[j]); ppl[j]^=1;   // last-1/4 P now ready -> finish PV
                if(lead8){ uint64_t vd0=vd0a[s];
                    #pragma unroll
                    for(int kb=PSPLIT;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                        asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                            ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(1u),"r"(PVI)); } } }
            mma_commit(OR[j]); };
        // Per-tile causal block counts: tile j only touches blocks [0, hij). For causal the union K-range makes
        // tile0's LAST block fully masked (only tile1 needs it) -> skip tile0's qk/pv/softmax there (~10% of the
        // main-loop tile-blocks are this waste). Barriers stay matched because softmax j also loops [0,hij).
        const int hi0=nhi0-n_lo, hi1=nhi1-n_lo;
        int pvf[VST]={0};
        mbar_wait(KVF[0],pf[0]); pf[0]^=1; if(hi0>0) qk(0,0); if(hi1>0) qk(1,0);
        for(int i=0;i<NBk;++i){ int s=i%WST; bool hn=(i+1<NBk); int ns=(i+1)%WST; int vs=i%VST;
            mbar_wait(VVF[vs],pvf[vs]); pvf[vs]^=1;   // V[i] loaded by w10 (decoupled ring) before either tile's PV
            if(i<hi0){ mbar_wait(PF[0],ppf[0]); ppf[0]^=1; if constexpr(CORR){ mbar_wait(CO[0],pco[0]); pco[0]^=1; } pv(0,i); }
            // Issue tile0's NEXT qk right after pv(0,i) (P0 read done -> S0 WAR ok) so it overlaps the PF[1]
            // (softmax tile1) wait below, keeping the tensor pipe fed instead of idling for full softmax latency.
            // The KVF wait is per-block (both tiles share the K ring stage) so it stays outside the tile guard.
            // MEASURED DEAD (2026-07-07b): CUTLASS's one-block MMA stagger (pv(1,i-1) lags, so MMA waits only
            // tile0's current softmax) REGRESSES here (causal 0.629->0.574) — our 2 softmax tiles are INDEPENDENT
            // and the HW scheduler interleaves all 8 warps more freely than a forced stagger. CUTLASS needs the
            // stagger only because its 2 halves share ONE tile's K. Kept lockstep. Also NaN-traps: P aliases S
            // (PCOL==SCOL) so a lagging pv must read P BEFORE the same tile's next qk overwrites S.
            if(hn){ mbar_wait(KVF[ns],pf[ns]); pf[ns]^=1; if(i+1<hi0) qk(0,i+1); }
            if(i<hi1){ mbar_wait(PF[1],ppf[1]); ppf[1]^=1; if constexpr(CORR){ mbar_wait(CO[1],pco[1]); pco[1]^=1; } pv(1,i); }
            if(hn && i+1<hi1){ qk(1,i+1); }
            mma_commit(KVE[s]); mma_commit(VVE[vs]);
        }
    } else if (warp<8) {                 // softmax WG (tile j = warp>>2)
        if constexpr(CORR) asm volatile("setmaxnreg.inc.sync.aligned.u32 " AVO_STR(AVO_SM_REG) ";\n" ::: "memory");   // AVO v33: softmax reg (swept)
        else               asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n" ::: "memory");   // non-CORR: 192 (sweet spot; >192 regresses/deadlocks)
        const int j=warp>>2, lr=tid&127;
        // Hoist loop-invariant tmem row addresses (base, SCOL[j]/PCOL[j]/OCOL[j], lr are all block-invariant) out
        // of the softmax hot loop so ptxas doesn't recompute add.s32 on the S-read-gated critical path (AVO win).
        const uint32_t sAddrJ = base+SCOL[j]+((uint32_t)lr<<16);
        const uint32_t pAddrJ = base+PCOL[j]+((uint32_t)lr<<16);
        const uint32_t oAddrJ = base+OCOL[j]+((uint32_t)lr<<16);
        int psf=0, por=0;
        float m_i=-INFINITY, l_i=0.f;
        // TRIED+REVERTED (ILP port): FA4 OrderBarrierSoftmax ping-pong staggering the two groups' exp so one
        // group's S TMEM read overlaps the other's exp. Neutral-to-negative for us (seq16k 0.376->0.366, others
        // +-0.003) at any arrive point: our two softmax groups are INDEPENDENT tiles the scheduler already
        // interleaves, so the explicit barrier only adds sync. FA4 needs it because its 2 groups are stacked
        // halves of ONE tile sharing K blocks (tighter coupling); that structure doesn't transfer here.
        if(tid==0){ for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); for(int s=0;s<VST;s++) mbar_arrive(VVE[s]); }
        const int row0j=(j?m1:m0)*WTM;
        const int qp=seqlen_k-seqlen_q+row0j+lr;
        int key0off=0;
        // sv is Q-prescaled (log2/ex2 domain): no-softcap -> sv already = scale*log2e*raw; softcap ->
        // sv = scale*raw, apply softcap*tanh then fold log2e (done inline per-element in the BOUNDARY path).
        // Masking is BRANCHLESS via a precomputed [clo,chi] valid-column range (see BOUNDARY block below).
        // Per-tile block count: tile j only processes its own [0, hij) (matches the MMA warp's tile guards) so
        // tile0 skips its fully-masked last causal block instead of running a wasted no-op softmax on it.
        const int hij=((j==0)?nhi0:nhi1)-n_lo;
        for(int i=0;i<hij;++i){ key0off=(n_lo+i)*WTN;
            mbar_wait(SF[j],psf); psf^=1; tmem_fence();
            // FULL vs BOUNDARY split (FA4 design): interior blocks need no per-key mask -> branch-free
            // scale-only path, saving ~6 ALU ops/element (kk/compare) on the vast majority of blocks.
            const int qbase=seqlen_k-seqlen_q+row0j, kend=key0off+WTN-1;
            bool full=(kend<seqlen_k) && !(softcap>0.f);
            if(is_causal && kend>qbase) full=false;
            if(win_r>=0 && kend>qbase+win_r) full=false;
            if(win_l>=0 && key0off < qbase+(WTM-1)-win_l) full=false;
            float m_new, corr, ps=0.f;
            if(full){
              if constexpr(!CORR){
                // SPILL-FREE 1-PASS via INLINE PTX (FA4-faithful): f<128> lives in PTX .reg so ptxas allocates it
                // inside the true 192-reg softmax region (setmaxnreg.inc 192) instead of the nvcc uniform-128 cap.
                // The C f[128] array here spilled 48 dwords -> 347K local-mem requests (62% L1TEX overhead, our #1
                // stall). 512thr*128reg=65536 = the whole regfile, so uneven 192/48 (=FA4) is UNREACHABLE from nvcc
                // C++ (uniform alloc); only PTX-emitted .reg gets the uneven split ptxas honors. Load,max,exp,pack,
                // store P all in one asm block (peak ~140 live regs < 192). Q is prescaled (log2 domain).
                softmax_full_ptx<T>(sAddrJ, pAddrJ, m_i, m_new, ps);
                corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
              } else {
                // CORR path: PACKED split-P inline PTX (FA4-faithful). Releases PF (first 3/4 of P, keys 0..95)
                // mid-store so the MMA can start PV early. FA4 EARLY-CORRECTION: the PTX now signals the
                // correction WG (writes sScale + arrives CS) right after the ROW-MAX, BEFORE the exp2/P tail,
                // so the O-rescale overlaps that tail on w12-15 instead of being additive (the prior late-CS
                // arrive was the reason CORR lost). corr recomputed here (same formula) only for the l_i update.
                softmax_split_ptx<T>(base+SCOL[j]+((uint32_t)lr<<16), base+PCOL[j]+((uint32_t)lr<<16), m_i, m_new, ps, PF[j],
                                     smem_u32(&sScale[j][i&1][lr]), CS[j]);
                corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
              }
            } else {
                // BOUNDARY: 1-pass + BRANCHLESS range mask. Valid key cols are a single contiguous [clo,chi]
                // range computed once/block, holding f[128] like FULL (no 2nd tmem read).
                uint32_t f[128];
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+0,  f, 0);
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+32, f, 32);
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+64, f, 64);
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+96, f, 96);
                tmem_wait_ld();
                int chi=seqlen_k-1-key0off;                                 // seqlen_k upper bound
                if(is_causal && qp-key0off<chi) chi=qp-key0off;             // causal (diagonal)
                if(win_r>=0 && qp+win_r-key0off<chi) chi=qp+win_r-key0off;  // right window
                if(chi>WTN-1) chi=WTN-1;
                const int clo=(win_l>=0)?max(0,qp-win_l-key0off):0;         // left window (0 for plain causal)
                float rmax=-INFINITY;
                #pragma unroll
                for(int c=0;c<128;++c){ float sv=__uint_as_float(f[c]);
                    float lg=(softcap>0.f)?(softcap*tanhf(sv*inv_sc)*LOG2E):sv;
                    if(c<=chi && c>=clo) rmax=fmaxf(rmax,lg); }
                m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                if constexpr(CORR){ sScale[j][i&1][lr]=corr; mbar_arrive(CS[j]); }
                #pragma unroll
                for(int g=0;g<4;++g){ uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ int c0=g*32+2*t, c1=c0+1;
                        float s0=__uint_as_float(f[c0]), s1=__uint_as_float(f[c1]);
                        float lg0=(softcap>0.f)?(softcap*tanhf(s0*inv_sc)*LOG2E):s0;
                        float lg1=(softcap>0.f)?(softcap*tanhf(s1*inv_sc)*LOG2E):s1;
                        float p0=(c0<=chi&&c0>=clo)?ex2f(lg0-m_new):0.f, p1=(c1<=chi&&c1>=clo)?ex2f(lg1-m_new):0.f; ps+=p0+p1;
                        pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(base+PCOL[j]+((uint32_t)lr<<16)+g*16, pk);
                    if constexpr(CORR){ if(g==2){ tmem_wait_st(); mbar_arrive(PF[j]); } } }   // split_P: release first 3/4
            }
            // O-rescale AFTER the exp loop (f[128] is dead here): the vectorized rescale's og[32] fragment no
            // longer coexists with f[128], so this no longer blows the 192-reg softmax budget -> kills the
            // f+og spill (the hot line-68 LDL/STL). Still correct: O is rescaled BEFORE PF is signaled below,
            // so the MMA's PV (which waits PF) always accumulates into already-rescaled O.
            if constexpr(!CORR){ if(i>0){ mbar_wait(OR[j],por); por^=1;
                // NB: AVO branchless always-rescale (CUTLASS mainloop_fork.hpp:1002 "+8.1% non-causal") REGRESSES
                // here (noncausal 0.729->0.715, measured 2026-07-07c). That win needs a SEPARATE correction warp
                // whose per-block vote throttled the MMA via PipelineO; our rescale is INLINE on the bottleneck
                // softmax warp, so the vote-SKIP is a genuine save (no O tmem ld/st when the row-max stabilized).
                // Keep the vote. (Opposite structures -> opposite optima; the CUTLASS opt does not transfer.)
                if(!__all_sync(0xffffffffu,corr==1.0f)) tmem_o_rescale<HDV>(oAddrJ, corr); } }
            l_i=l_i*corr+ps; m_i=m_new;
            // split_P (CORR): PF already released at 3/4 above -> here signal the last 1/4 (PL). Non-CORR: single PF.
            tmem_wait_st(); if constexpr(CORR){ mbar_arrive(PL[j]); } else { mbar_arrive(PF[j]); }   // all 128 arrive (no bar.sync)
        }
        if constexpr(CORR){ if(hij>0) mbar_wait(OR[j],(hij-1)&1); } else { if(hij>0){ mbar_wait(OR[j],por); por^=1; } } tmem_fence();
        const int qr=row0j+lr;
        // m_i is in log2 domain (Q prescaled by scale*log2e); sink logit s_aux is natural -> *log2e.
        const float INV_LOG2E=0.6931471805599453f;   // ln2: convert log2-domain max back to natural for LSE
        float mfin=m_i, lfin=l_i, osc=1.f;
        if(s_aux!=nullptr){ float sk2=s_aux[h]*LOG2E; float mn=fmaxf(m_i,sk2); float c2=(m_i==-INFINITY)?0.f:ex2f(m_i-mn); lfin=l_i*c2+ex2f(sk2-mn); osc=c2; mfin=mn; }
        float inv=(lfin>0.f)?osc/lfin:0.f;
        // vectorized final O read (4x ld32/chunk) + VECTORIZED 128b store: 16x uint4 STG/row.
        // tmem read + tcgen05.wait::ld.sync.aligned MUST run on the FULL warp (convergent): guarding them
        // behind `qr<seqlen_q` deadlocks the boundary warp when seqlen_q%32!=0 (partial-warp divergence on
        // the aligned wait). So read/scale unconditionally; predicate only the global stores (mirrors ws1).
        const bool ow=(qr<seqlen_q) && (hij>0);
        const size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv):0;
        const unsigned long long inv2=bcast2(inv);
        #pragma unroll
        for(int g=0;g<HDV;g+=32){ uint32_t ov[32]; tmem_ld32(base+OCOL[j]+((uint32_t)lr<<16)+(uint32_t)g, ov); tmem_wait_ld();
            #pragma unroll
            for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);   // packed O*inv (FMUL2) then pack to bf16
            if(ow){
                #pragma unroll
                for(int t=0;t<32;t+=8){ int d=g+t;
                    if(d+8<=rhdv){   // one 16-byte store (8 bf16) instead of 8 scalar 2-byte STG -> 8x fewer store txns
                        uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                        vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3]));
                        vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                        vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7]));
                        *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                    } else {
                        #pragma unroll
                        for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); }
                    } } }
        }
        if(qr<seqlen_q && hij>0) lse[(size_t)h*(cu_q[gridDim.z])+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin*INV_LOG2E):-INFINITY;
    } else if constexpr(CORR) { if(warp>=12) {   // correction WG (idle w12-15): ALWAYS-ON vectorized O-rescale.
        // FA4 principle: keep these warps busy every block (not gated on corr!=1) so the scheduler always has
        // eligible warps to hide the softmax S TMEM-read latency (our #1 real stall: long_scoreboard). Rescale is
        // fully OFF the softmax critical path (softmax just hands corr via sScale+CS); MMA waits CO before PV.
        const int r=tid-384;
        int pcs[2]={0,0}, por2[2]={0,0};
        // Per-tile block counts must match the softmax/MMA tile guards (tile j only runs [0,hij)) or the
        // correction WG would wait a CS[j]/OR[j] that softmax never arrives -> hang. Keep the j-inside-i
        // interleave (MMA consumes CO[0],CO[1] per block) but skip tile j once i>=hij.
        const int ch0=nhi0-n_lo, ch1=nhi1-n_lo, chm=max(ch0,ch1);
        for(int i=0;i<chm;++i){
            #pragma unroll
            for(int j=0;j<2;++j){ const int hij=(j==0)?ch0:ch1;
                if(i<hij){
                    mbar_wait(CS[j],pcs[j]); pcs[j]^=1;
                    float corr=sScale[j][i&1][r];
                    if(i>0){ mbar_wait(OR[j],por2[j]); por2[j]^=1; tmem_fence();
                        // AVO v20/v33: depth-batched rescale (AVO_RESCALE_DEPTH overlaps O-read latencies, needs the
                        // rebalanced correction-warp regs); AVO_BRANCHLESS=1 drops the __all_sync(corr==1) skip
                        // (always-rescale, no warp divergence). AVO_BRANCHLESS=0 keeps the skip (baseline behavior).
#if AVO_BRANCHLESS
                        tmem_o_rescale_d<HDV,AVO_RESCALE_DEPTH>(base+OCOL[j]+((uint32_t)r<<16), corr); tmem_wait_st();
#else
                        if(!__all_sync(0xffffffffu,corr==1.0f)){ tmem_o_rescale_d<HDV,AVO_RESCALE_DEPTH>(base+OCOL[j]+((uint32_t)r<<16), corr); tmem_wait_st(); }
#endif
                    }
                    mbar_arrive(CO[j]);
                }
            }
        }
    } }
    __syncthreads();
    if(warp==8) tmem_dealloc(base,512);
}

// ================= FA4-FAITHFUL warp-specialized kernel (fwd_kernel_fa4) =====================
// Reimplements the CUTLASS FA4 structure: ONE 256-row Q tile = 2 STACKED 128-row halves sharing K/V.
// 16 warps: w0-3 Softmax0(rows0-127), w4-7 Softmax1(rows128-255), w8-11 Correction, w12 MMA,
// w13 Load(TMA K/V), w14 Epilogue, w15 Empty. tmem: S0@0 S1@128 P0@0 P1@128(alias S) O0@256 O1@384.
// Pipelines (all depth-1 except KV ring; Q preloaded so no Q pipe; order_s01 omitted, perf-only):
//   KV (Load->MMA), S0/S1 (MMA->Softmax, P-ready via p_full), C0/C1 (Softmax->Corr, stats via smem),
//   O0/O1 (MMA->Corr), E (Corr->Epilogue, sO smem).  Stats (corr factor + final sum/max) travel via
//   smem (sScale/sFin*) gated by the C barriers (like ws2), not tmem V0/V1.  Q is PRESCALED by
//   scale*log2e at load, so softmax is p=ex2(S-m) and the corr factor is ex2(m_old-m_new).
// Milestone target: CORRECT on non-causal (also handles causal / gqa). KVS=1 (serial K/V; correctness
// first). O-side split: MMA acquires o_empty (correction rescaled O) before each PV.
// Max K/V ring depth any instantiation uses. fwd_kernel_fa4_paged runs a DECOUPLED ring (K deeper than
// V) because PC sampling showed its producer starved for ring slots, not for issue slots.
constexpr int FA4KVSMAX = 3;
struct BarsFA4 { uint64_t kf_full[FA4KVS], vf_full[FA4KVS], kv_empty[FA4KVS], s_full[2], p_full[2], o_full[2], o_empty[2], c_full[4], c_empty[4], q_full[1], q_empty[1], q_tma[1], q_sched[1], ord[2]; }; // D4: c_full/c_empty are [half*2+slot] depth-2
// WINDOW/SINK/PADQ: template-gated feature grafts (ported line-for-line from fwd_kernel_ws2). The default
// instantiation <T,HD,HDV,false,false,false> compiles to byte-identical SASS (all new code behind if constexpr).
//   WINDOW: sliding-window (win_l/win_r) tile-range prune (n_lo..n_hi) + per-element band mask in boundary tiles.
//   SINK:   attention sink (s_aux) folded into final (m,l) + O-scale (osc) at the softmax/correction handoff.
//   PADQ:   hd<HD (e.g. hd96->HD=128) support: zero sQ pad cols [rhd,HD) after TMA-Q (Q=0 nullifies QK garbage).
//   PAGED:  paged K/V cache. `mapK`/`mapV` become PAGE descriptors (box [64, segrows] over the pool
//           viewed 2D as [num_rows, hkv*hd]), so a WTN-key tile is WTN/segrows contiguous physical page
//           segments = nseg*(HD/64+HDV/64) TMA boxes instead of one. seqlen_k comes from seqused_k[b] and
//           the base row from the page table. Everything else -- persistent LPT scheduling, the dual
//           stacked-half softmax ping-pong, the correction/epilogue warpgroups, the cross-tile K/V ring --
//           is shared with the dense path, which is the point: the separate non-persistent
//           fwd_kernel_fa4_paged tops out near 0.70x where this kernel reaches 0.91-1.19x on dense.
template<class T, int HD, int HDV, bool WINDOW=false, bool SINK=false, bool PADQ=false, bool PAGED=false>
__global__ void __launch_bounds__(512,1) fwd_kernel_fa4(
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
    const __grid_constant__ CUtensorMap mapQ,
    const T* __restrict__ q, T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int* __restrict__ tile_ctr, int total_tiles, int gx, int nB, int ragged,
    const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, int pt_seg, int grp,
    const T* __restrict__ kptr, const T* __restrict__ vptr) {   // raw pool pointers: PAGED software gather only
    // PERSISTENT + CONTINUOUS-OVERLAP fa4 (persist-lpt). Grid = NSM CTAs; each CTA grabs 256-row tiles via
    // atomicAdd(tile_ctr) in LPT (heaviest-mb-first) order. NO per-tile __syncthreads drain: tile N's
    // correction/epilogue (O-drain + direct global write) overlaps tile N+1's mainloop. Roles: w0-3 Softmax0,
    // w4-7 Softmax1, w8-11 Correction (O-rescale + final normalize + DIRECT global write + lse; sO removed),
    // w12 MMA, w13 K/V-load, w14-15 Q-producer (scheduler + TMA-Q + in-place prescale). Cross-tile mbars:
    // q_full (producer->all consumers: Q loaded+prescaled+s_tile ready), q_empty (both softmax WG leads ->
    // producer: SF fully consumed => QKs retired => sQ free), q_tma (Q TMA completion, producer-internal).
    // Ring KVE/KVF + SF/PF/OF/OE/CF/CE/ord phases CARRY across tiles (never reset). LPT balances causal wave.
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64, VCB=HDV/64;
    const uint32_t SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[FA4KVS]; T* sV[FA4KVS];
    { T* p=sQ+2*WTM*HD; for(int s=0;s<FA4KVS;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<FA4KVS;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) BarsFA4 bars; __shared__ __align__(8) uint32_t s_base; __shared__ int s_tile;
    __shared__ float sScale[2][2][WTM];              // D4 depth-2: [half][slot=prod&1][row] corr factor (softmax->correction)
    __shared__ float sFinSum[2][2][WTM], sFinMax[2][2][WTM]; // D4 depth-2 double-buffered final row_sum/row_max by slot
    __shared__ float sFinOsc[2][2][WTM];             // SINK: per (half,slot,row) O-scale osc (only referenced under if constexpr(SINK) -> DCE'd when SINK=false)
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f, INV_LOG2E=0.6931471805599453f;
    const float qsc=scale*LOG2E;                     // Q prescale -> log2 softmax domain (no softcap in this path)
    const int lse_stride=cu_q[nB];
    // A software paged gather needs registers in the PRODUCER tier: FA4 sizes exactly the non-TMA paged
    // case at softmax 184 / correction 64 / other 80 (flash_fwd_sm100.py num_regs_softmax +
    // num_regs_other), where its TMA path uses 192/64/64. At 64 the 32-iteration cp.async gather spills
    // and the whole point is lost (attn_real_llama8b_pf_s16k measured 0.33x).
    if constexpr(PAGED){ if(warp>=8 && warp<12) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");
                         else if(warp>=12)      asm volatile("setmaxnreg.dec.sync.aligned.u32 80;\n" ::: "memory"); }
    else if(warp>=8) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");
    if(warp==12){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base=s_base;
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();
    if(tid==0){ for(int s=0;s<FA4KVS;s++){ mbar_init(smem_u32(&bars.kf_full[s]),1); mbar_init(smem_u32(&bars.vf_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1); }
        for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.p_full[j]),128);
            mbar_init(smem_u32(&bars.o_full[j]),1); mbar_init(smem_u32(&bars.o_empty[j]),128);
            mbar_init(smem_u32(&bars.c_full[2*j]),128); mbar_init(smem_u32(&bars.c_full[2*j+1]),128); mbar_init(smem_u32(&bars.c_empty[2*j]),128); mbar_init(smem_u32(&bars.c_empty[2*j+1]),128); }
        mbar_init(smem_u32(&bars.q_full[0]),1); mbar_init(smem_u32(&bars.q_empty[0]),2); mbar_init(smem_u32(&bars.q_tma[0]),1);
        mbar_init(smem_u32(&bars.q_sched[0]),1);       // D4: scheduler->load "s_tile ready" (before Q prescale) so K/V TMA overlaps prescale
        mbar_init(smem_u32(&bars.ord[0]),128); mbar_init(smem_u32(&bars.ord[1]),128); }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t QFULL=smem_u32(&bars.q_full[0]), QEMPTY=smem_u32(&bars.q_empty[0]);
    const uint32_t QSCHED=smem_u32(&bars.q_sched[0]);

    // WARP ROLES. Dense: w13 load, w14-15 Q producer. PAGED-non-TMA needs a software gather, and FA4
    // sizes exactly that case with TWO load warps (flash_fwd_sm100.py: `elif not use_tma_KV:
    // load_warp_ids = (14, 15)`), so the paged instantiations move w14 to the load role and run the
    // scheduler + TMA-Q on w15 alone (only one thread of that warp ever schedules or issues TMA-Q).
    constexpr int QPW  = PAGED ? 15 : 14;   // first Q-producer warp
    constexpr int QPN  = PAGED ? 32 : 64;   // Q-producer threads
    constexpr int LDW1 = PAGED ? 14 : 13;   // last load warp
    constexpr int LDN  = (LDW1-13+1)*32;    // load threads
    if (warp>=QPW) {                     // -------- Q PRODUCER: scheduler + TMA-Q (RAW, no prescale) --------
        const bool sched=(tid==QPW*32);
        const uint32_t QTMA=smem_u32(&bars.q_tma[0]);
        int i=0, tile=blockIdx.x, qe=0, qt=0;
        for(;;){
            if(sched){ if(i>0){ mbar_wait(QEMPTY,qe); qe^=1; }
                       // RAGGED-BATCH EMPTY-TILE SKIP. total_tiles is the RECTANGLE gx*hq*nB sized by the
                       // batch's LONGEST sequence, so on a ragged batch most tiles of the short sequences
                       // have every Q row past seqlen_q. They used to run the full mainloop (n blocks of
                       // QK+PV) and be dropped only at the epilogue `ow` predicate: varlen_c_d128_realmix
                       // (seqlens 2048/1024/512/4096/256/768, hq32) launched 3072 tiles for 1088 real ones,
                       // 2.8x wasted MMA -- which is the whole 0.55x. Consume such tiles here instead;
                       // every counter value is still handed out exactly once, so the terminal
                       // self-reset condition below is unchanged.
                       while((ragged||grp) && tile<total_tiles){   // uniform-dense batches have no empty tiles
                           const int per=hq*nB, lvl=tile/per, rem=tile%per;
                           const int mb=(gx-1)-lvl, b=rem%nB;
                           const int sq=cu_q[b+1]-cu_q[b];
                           // Keep the tile iff it has >=1 real Q row AND this launch owns the sequence.
                           // grp==2 is the prefill half of a split mixed batch: without this test the
                           // persistent kernel would redo every decode sequence that the packed grp==1
                           // launch already handled -- same output, but at the unpacked 4x KV cost, which
                           // took mixed_2pf2k_8dec16k back from 1.02x to 0.44x.
                           if(((2*mb)*WTM < sq) && !(grp==2 && rep*sq <= WTM)) break;
                           int nx=atomicAdd(tile_ctr,1);
                           if(nx==total_tiles+(int)gridDim.x-1) atomicExch(tile_ctr,(int)gridDim.x);
                           tile=nx;
                       }
                       s_tile=(tile<total_tiles)?tile:-1;
                       mbar_arrive(QSCHED); }               // D4: publish s_tile to LOAD warp before TMA-Q
            asm volatile("bar.sync 3, %0;"::"r"(QPN));   // observe s_tile (+ prev sQ freed via sched's QEMPTY wait)
            int t=s_tile;
            if(sched){
                if(t>=0){
                    const int per=hq*nB, lvl=t/per, rem=t%per;
                    const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
                    const int q0=cu_q[b];
                    const int r00=q0+(2*mb)*WTM, r01=q0+(2*mb+1)*WTM;   // global token rows for the 2 Q-halves
                    mbar_expect(QTMA, 2*WTM*HD*(int)sizeof(T));
                    for(int cb=0;cb<KCB;cb++){ tma_box(sQ+0*WTM*HD+cb*8192, &mapQ, h*rhd+cb*64, r00, QTMA);
                                               tma_box(sQ+1*WTM*HD+cb*8192, &mapQ, h*rhd+cb*64, r01, QTMA); }
                    mbar_wait(QTMA,qt); qt^=1;               // D3: async Q TMA done -> sQ RAW ready. Prescale folded
                                                             // into softmax exp2 (qsc*S), so QFULL arrives NOW with
                                                             // zero prescale on the tile-boundary critical path.
                }
                if constexpr(!PADQ) mbar_arrive(QFULL);
            }
            if constexpr(PADQ){          // hd<HD (e.g. hd96->HD=128): zero sQ pad cols [rhd,HD) after TMA-Q so the QK
                if(t>=0){                //  contraction sees Q=0 there (nullifies the next-head garbage TMA reads into [rhd,HD))
                    asm volatile("bar.sync 3, %0;"::"r"(QPN));   // all producer threads observe Q landed (sched past QTMA wait)
                    const int PADW=HD-rhd;                   // ws2 does the equivalent via `c<rhd ? q : 0` at manual Q load
                    for(int idx=(tid-QPW*32); idx<2*WTM*PADW; idx+=QPN){ int hh=idx/(WTM*PADW); int t2=idx-hh*(WTM*PADW); int rr=t2/PADW; int cc=rhd+(t2-rr*PADW);
                        sQ[hh*WTM*HD + koff(rr,cc)]=from_f<T>(0.f); }
                    asm volatile("fence.proxy.async.shared::cta;");   // generic-proxy zeros -> visible to async-proxy QK MMA
                    asm volatile("bar.sync 3, %0;"::"r"(QPN));   // all zeros written+fenced before QFULL published
                }
                if(sched) mbar_arrive(QFULL);                // t>=0: Q ready; t<0: drain -> consumers break
            }
            if(t<0) break;
            if(sched){ tile=atomicAdd(tile_ctr,1);
                // D3 device-side self-reset: the atomicAdd that returns the terminal MAX value
                // (total_tiles+gridDim.x-1) is provably the temporally-LAST increment (monotonic
                // counter, exactly total_tiles increments total). After it no CTA touches tile_ctr,
                // so this CTA restores it to gridDim.x (== launch ctas) for the next same-stream call.
                if(tile==total_tiles+(int)gridDim.x-1) atomicExch(tile_ctr,(int)gridDim.x); }
            i++;
        }
    } else if (warp<8) {                 // -------- SOFTMAX half j = warp>>2 --------
        if constexpr(PAGED) asm volatile("setmaxnreg.inc.sync.aligned.u32 184;\n" ::: "memory");
        else                asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n" ::: "memory");
        const int j=warp>>2, lr=tid&127; const bool wglead=((tid&127)==0);
        const uint32_t sf=smem_u32(&bars.s_full[j]), pf=smem_u32(&bars.p_full[j]);
        const uint32_t cf0=smem_u32(&bars.c_full[2*j]), cf1=smem_u32(&bars.c_full[2*j+1]);
        const uint32_t ce0=smem_u32(&bars.c_empty[2*j]), ce1=smem_u32(&bars.c_empty[2*j+1]);
        const uint32_t sAddr=base+SCOL[j]+((uint32_t)lr<<16);
        const uint32_t pAddr=base+PCOL[j]+((uint32_t)lr<<16);
        const uint32_t ordw=smem_u32(&bars.ord[j]), orda=smem_u32(&bars.ord[j^1]);
        int qf=0, psf=0, ordph=(j==0)?1:0;             // ALL carry across tiles
        int pce0=1, pce1=1;                             // D4 depth-2: free-pass BOTH C slots => softmax leads correction by exactly 2 blocks (never more: 2-slot buffer + ce backpressure caps the lead)
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
            const int k0=PAGED?0:cu_k[b], seqlen_k=PAGED?seqused_k[b]:(cu_k[b+1]-k0);
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            int n_lo=0;   // WINDOW: sliding-window tile-range prune (mirrors ws2 nlo/nhi @L6543). n=NB=n_hi-n_lo.
            if constexpr(WINDOW){ int qxw=seqlen_k-seqlen_q+row0+255, qnw=seqlen_k-seqlen_q+row0; int nhi=nk_all;
                if(is_causal){ int hc=(qxw>=0)?(qxw/WTN+1):0; nhi=min(nhi,hc); }
                if(win_r>=0){ int hr=(qxw+win_r>=0)?((qxw+win_r)/WTN+1):0; nhi=min(nhi,hr); }
                if(win_l>=0){ int lw=qnw-win_l; n_lo=(lw>0)?(lw/WTN):0; }
                nhi=max(nhi,n_lo); n=nhi-n_lo; if(n<1)n=1; }
            const int qp=seqlen_k-seqlen_q+row0+j*WTM+lr;
            float m_i=-INFINITY, l_i=0.f;
            for(int i=0;i<n;++i){
                int sl=i&1;                                // D4 depth-2: C-channel slot by block parity
                if(sl==0){ mbar_wait(ce0, pce0); pce0^=1; } else { mbar_wait(ce1, pce1); pce1^=1; } // acquire empty slot (from 2 blocks ago)
                const uint32_t cf=(sl==0)?cf0:cf1;
                mbar_wait(sf, psf); psf^=1; tmem_fence();
                // D4: sQ half j is read ONLY by the QK MMAs. When this wait returns for the last
                // real block, block n-1's QK has completed reading sQ, so half j is free NOW --
                // earlier than end-of-mainloop. Releasing here lets the Q producer issue tile N+1's
                // TMA-Q + prescale during tile N's tail (block n-1 exp + terminal SF wait + stats +
                // correction + epilogue), software-pipelining the tile-boundary Q handshake.
                if(i==n-1 && wglead) mbar_arrive(QEMPTY);
                mbar_wait(ordw, ordph); ordph^=1;
                int key0=i*WTN; if constexpr(WINDOW) key0=(n_lo+i)*WTN;
                const int qbase=seqlen_k-seqlen_q+row0+j*WTM;
                const int kend=key0+WTN-1;
                bool full=(kend<seqlen_k) && !(is_causal && kend>qbase);
                if constexpr(WINDOW){ if(win_r>=0 && kend>qbase+win_r) full=false; if(win_l>=0 && key0 < qbase+(WTM-1)-win_l) full=false; }
                float m_new, ps=0.f;
                if(full){
                    softmax_full_ptx_ordcs<T>(sAddr, pAddr, m_i, m_new, ps, orda, smem_u32(&sScale[j][sl][lr]), cf, qsc);
                } else {
                    int chi=seqlen_k-1-key0; if(chi>WTN-1) chi=WTN-1;
                    if(is_causal){ int cc=qp-key0; if(cc<chi) chi=cc; }
                    int clo=0;
                    if constexpr(WINDOW){ if(win_r>=0){ int cr=qp+win_r-key0; if(cr<chi) chi=cr; } clo=(win_l>=0)?max(0,qp-win_l-key0):0; }
                    float rmax=-INFINITY;
                    #pragma unroll 1
                    for(int g=0; g<128; g+=32){ uint32_t ch[32]; tmem_ld32(sAddr+(uint32_t)g, ch); tmem_wait_ld();
                        #pragma unroll
                        for(int c=0;c<32;++c){ bool okc=(g+c<=chi); if constexpr(WINDOW) okc=okc&&(g+c>=clo); if(okc) rmax=fmaxf(rmax,__uint_as_float(ch[c])); } }
                    rmax*=qsc;                                 // D3: fold Q-prescale into softmax (raw S -> log2 domain)
                    mbar_arrive(orda);
                    m_new=fmaxf(m_i,rmax);
                    float cb=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                    sScale[j][sl][lr]=cb; mbar_arrive(cf);
                    bool ok=(m_new!=-INFINITY);
                    #pragma unroll 1
                    for(int g=0; g<128; g+=32){ uint32_t ch[32]; tmem_ld32(sAddr+(uint32_t)g, ch); tmem_wait_ld();
                        uint32_t pk[16];
                        #pragma unroll
                        for(int t=0;t<16;++t){ int c0=g+2*t,c1=c0+1; float s0=__uint_as_float(ch[2*t])*qsc, s1=__uint_as_float(ch[2*t+1])*qsc;
                            bool k0ok=(c0<=chi), k1ok=(c1<=chi); if constexpr(WINDOW){ k0ok=k0ok&&(c0>=clo); k1ok=k1ok&&(c1>=clo); }
                            float p0=(ok&&k0ok)?ex2f(s0-m_new):0.f, p1=(ok&&k1ok)?ex2f(s1-m_new):0.f; ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                        tmem_st16(pAddr+(uint32_t)(g/2), pk); }
                }
                float corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                tmem_wait_st(); mbar_arrive(pf);
                l_i=l_i*corr+ps; m_i=m_new;
            }
            mbar_wait(sf, psf); psf^=1;                    // terminal SF wait (pairs MMA final dummy S commit)
            int fsl=n&1;                                    // D4 depth-2: finsum production p=n uses slot n&1
            if(fsl==0){ mbar_wait(ce0, pce0); pce0^=1; } else { mbar_wait(ce1, pce1); pce1^=1; } // acquire finsum slot
            if constexpr(SINK){
                // attention sink: virtual always-attended col with logit s_aux[h] (natural -> *log2e), value 0.
                // Fold into final (m,l)+O-scale (ws2 L6785). Publish (lfin,mfin,osc); correction reads osc for inv=osc/l.
                float sk2=s_aux[h]*LOG2E; float mn=fmaxf(m_i,sk2); float c2=(m_i==-INFINITY)?0.f:ex2f(m_i-mn);
                float lfin=l_i*c2+ex2f(sk2-mn);
                sFinSum[j][fsl][lr]=lfin; sFinMax[j][fsl][lr]=mn; sFinOsc[j][fsl][lr]=c2;
            } else {
                sFinSum[j][fsl][lr]=l_i; sFinMax[j][fsl][lr]=m_i;
            }
            mbar_arrive((fsl==0)?cf0:cf1);                  // publish final stats (production p=n)
        }
    } else if (warp>=13 && warp<=LDW1) { // -------- LOAD warp(s): K/V ring (split K/V full barriers) --------
        uint32_t KF[FA4KVS],VF[FA4KVS],KVE[FA4KVS]; for(int s=0;s<FA4KVS;s++){ KF[s]=smem_u32(&bars.kf_full[s]); VF[s]=smem_u32(&bars.vf_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
        const int lid=tid-416; bool lead=(lid==0); int pe[FA4KVS]; for(int s=0;s<FA4KVS;s++) pe[s]=1;   // carry
        int qf=0;
        for(;;){
            mbar_wait(QSCHED,qf); qf^=1;                    // D4: s_tile ready (before Q prescale) => K/V TMA overlaps prescale
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB, hk=h/rep;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
            const int k0=PAGED?0:cu_k[b], seqlen_k=PAGED?seqused_k[b]:(cu_k[b+1]-k0);
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            int n_lo=0;   // WINDOW: sliding-window tile-range prune (mirrors ws2 nlo/nhi @L6543). n=NB=n_hi-n_lo.
            if constexpr(WINDOW){ int qxw=seqlen_k-seqlen_q+row0+255, qnw=seqlen_k-seqlen_q+row0; int nhi=nk_all;
                if(is_causal){ int hc=(qxw>=0)?(qxw/WTN+1):0; nhi=min(nhi,hc); }
                if(win_r>=0){ int hr=(qxw+win_r>=0)?((qxw+win_r)/WTN+1):0; nhi=min(nhi,hr); }
                if(win_l>=0){ int lw=qnw-win_l; n_lo=(lw>0)?(lw/WTN):0; }
                nhi=max(nhi,n_lo); n=nhi-n_lo; if(n<1)n=1; }
            const int segrows = PAGED ? pt_seg : WTN;   // 0 => tile::gather4 (page_size not a multiple of 8)
            const int nseg = segrows>0 ? WTN/segrows : 0;
            const int lane = lid&31, w2 = lid>>5, rg = lane>>4, sl = lane&15;
            for(int i=0;i<n;++i){ int s=i%FA4KVS; mbar_wait(rsel(KVE,s), rflip(pe,s));
              const uint32_t kfb=rsel(KF,s), vfb=rsel(VF,s);
              T* const kbs=rsel(sK,s); T* const vbs=rsel(sV,s);
              if constexpr(PAGED){
                // One box per (page segment, 64-col block) per tensor. Lanes 0..nseg-1 resolve their own
                // page row and issue their own boxes so the page-table loads are parallel, but the two
                // expect_tx must precede every box -> lane 0 arms both barriers, then a warp sync.
                int key0=(n_lo+i)*WTN;
                if(lid==0){ mbar_expect(kfb, WTN*HD*2); mbar_expect(vfb, WTN*HDV*2); }
                asm volatile("bar.sync 8, %0;"::"r"(LDN));   // expect_tx ordered ahead of every gather/box
                if(segrows>0){
                  if(lid<nseg){ int kk=key0+lid*segrows;
                    // Tail segments past seqlen_k have no valid page-table entry: pull pool row 0 and let
                    // the boundary softmax mask those keys (p=0), so V garbage cannot reach O.
                    int y = (kk<seqlen_k) ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : 0;
                    int dof = lid*segrows*64;
                    for(int cb=0;cb<KCB;cb++) tma_box(kbs+cb*8192+dof, &mapK, hk*rhd +cb*64, y, kfb);
                    for(int cb=0;cb<VCB;cb++) tma_box(vbs+cb*8192+dof, &mapV, hk*rhdv+cb*64, y, vfb); }
                } else {
                  // page_size < 8: no multi-row box is expressible, so gather 4 rows per issue with
                  // tile::gather4 -- NGT = (WTN/4)*(KCB+VCB) = 128 issues at hd128, two per lane across the
                  // two load warps. Each lane resolves the 4 page rows its issue needs directly (512 L1/L2-hot
                  // page_table reads per tile against 128 distinct ones -- redundant but independent, which is
                  // what matters: this producer is latency-bound, not issue-bound).
                  // The software cp.async gather FA4 uses here was tried and lost badly (0.30-0.33x): 4096
                  // 16B copies from 64 threads is 64 issues per lane per block, and this kernel's producer
                  // tier cannot hold the address state. Kept behind FA4_PFSW for the record.
                  constexpr int NG=WTN/4, NGK=NG*KCB, NGT=NGK+NG*VCB;
                  #pragma unroll 1
                  for(int op=lid; op<NGT; op+=LDN){
                      bool isK=op<NGK; int o=isK?op:op-NGK; int cb=o/NG, g=o-cb*NG;
                      int kk=key0+4*g, y0,y1,y2,y3;
                      y0=(kk+0<seqlen_k)?(page_table[b*pt_stride+(kk+0)/block_size]*block_size+(kk+0)%block_size):0;
                      y1=(kk+1<seqlen_k)?(page_table[b*pt_stride+(kk+1)/block_size]*block_size+(kk+1)%block_size):0;
                      y2=(kk+2<seqlen_k)?(page_table[b*pt_stride+(kk+2)/block_size]*block_size+(kk+2)%block_size):0;
                      y3=(kk+3<seqlen_k)?(page_table[b*pt_stride+(kk+3)/block_size]*block_size+(kk+3)%block_size):0;
                      if(isK) tma_gather4(kbs+cb*8192+4*g*64, &mapK, hk*rhd +cb*64, y0,y1,y2,y3, kfb);
                      else    tma_gather4(vbs+cb*8192+4*g*64, &mapV, hk*rhdv+cb*64, y0,y1,y2,y3, vfb);
                  }
                }
              } else {
                if(lead){ int y=k0+i*WTN; if constexpr(WINDOW) y=k0+(n_lo+i)*WTN;
                    // K first (needed by QK), signal on its own barrier so QK runs ahead of the V load.
                    mbar_expect(kfb, WTN*HD*2);
                    for(int cb=0;cb<KCB;cb++) tma_box(kbs+cb*8192, &mapK, hk*rhd+cb*64, y, kfb);
                    mbar_expect(vfb, WTN*HDV*2);
                    for(int cb=0;cb<VCB;cb++) tma_box(vbs+cb*8192, &mapV, hk*rhdv+cb*64, y, vfb); } }
              }
        }
    } else if (warp==12) {               // -------- MMA warp --------
        // Every one of these is indexed by the runtime ring slot `s = i % FA4KVS` inside the per-block
        // loop, so a plain a[s] puts the whole table in LOCAL memory -- the pathology documented at the top
        // of this file and fixed for the correction WG, but never for the MMA/Load warps. ncu on
        // attn_dense_hd128_seq16k: 865,568 local load REQUESTS against FA4's 16,384, with an `LDL.64` of the
        // MMA descriptor immediately ahead of every UTCHMMA and an LDL of the barrier address+phase ahead of
        // every mbar_wait. rsel/rflip keep them in registers (a single ISETP+SEL at depth 2).
        uint32_t KF[FA4KVS],VF[FA4KVS],KVE[FA4KVS]; for(int s=0;s<FA4KVS;s++){ KF[s]=smem_u32(&bars.kf_full[s]); VF[s]=smem_u32(&bars.vf_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
        const uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
        const uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
        const uint32_t OF[2]={smem_u32(&bars.o_full[0]),smem_u32(&bars.o_full[1])};
        const uint32_t OE[2]={smem_u32(&bars.o_empty[0]),smem_u32(&bars.o_empty[1])};
        const bool lead=(tid&31)==0;
        uint64_t qd0a[2], kd0a[FA4KVS], vd0a[FA4KVS];
        #pragma unroll
        for(int jj=0;jj<2;jj++) qd0a[jj]=make_desc(smem_u32(sQ+jj*WTM*HD),16,1024,2);
        #pragma unroll
        for(int s=0;s<FA4KVS;s++){ kd0a[s]=make_desc(smem_u32(sK[s]),16,1024,2); vd0a[s]=make_desc(smem_u32(sV[s]),16384,1024,2); }
        auto qk=[&](int j,int s){ if(lead){ uint64_t qd0=qd0a[j],kd0=rsel(kd0a,s);
                for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                        ::"r"(base+SCOL[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); } }
            mma_commit(SF[j]); };
        auto pv=[&](int j,int i,int s){ if(lead){ uint64_t vd0=rsel(vd0a,s);
                for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                        ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); } }
            mma_commit(OF[j]); };
        int qf=0, qfp=0, pkf[FA4KVS]={0}, pvf[FA4KVS]={0}, poe0=1, poe1=1, pf0=0, pf1=0;   // ALL carry across tiles
        for(;;){
            mbar_wait(QSCHED,qf); qf^=1;                    // D4: learn tile before Q prescale
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, b=rem%nB;
            const int seqlen_q=cu_q[b+1]-cu_q[b], seqlen_k=PAGED?seqused_k[b]:(cu_k[b+1]-cu_k[b]);
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            int n_lo=0;   // WINDOW: sliding-window tile-range prune (mirrors ws2 nlo/nhi @L6543). n=NB=n_hi-n_lo.
            if constexpr(WINDOW){ int qxw=seqlen_k-seqlen_q+row0+255, qnw=seqlen_k-seqlen_q+row0; int nhi=nk_all;
                if(is_causal){ int hc=(qxw>=0)?(qxw/WTN+1):0; nhi=min(nhi,hc); }
                if(win_r>=0){ int hr=(qxw+win_r>=0)?((qxw+win_r)/WTN+1):0; nhi=min(nhi,hr); }
                if(win_l>=0){ int lw=qnw-win_l; n_lo=(lw>0)?(lw/WTN):0; }
                nhi=max(nhi,n_lo); n=nhi-n_lo; if(n<1)n=1; }
            // prologue: block 0 QK0,QK1 + PV0(0). D4: pre-wait K[0] (overlaps prescale), then QFULL, then QK.
            // K/V full split: QK waits K only; PV waits V (may already have landed).
            mbar_wait(KF[0], pkf[0]); pkf[0]^=1;
            mbar_wait(QFULL,qfp); qfp^=1;                   // Q prescaled => safe to issue QK
            qk(0,0); qk(1,0);
            mbar_wait(OE[0], poe0); poe0^=1;
            mbar_wait(PF[0], pf0); pf0^=1;
            mbar_wait(VF[0], pvf[0]); pvf[0]^=1;
            pv(0,0,0);
            // steady loop i=1..n-1
            for(int i=1;i<n;++i){ int s=i%FA4KVS, sp=(i-1)%FA4KVS;
                mbar_wait(rsel(KF,s), rflip(pkf,s));         // K[i] ready -> QK runs ahead of V[i] load
                qk(0,s);
                mbar_wait(OE[1], poe1); poe1^=1;
                mbar_wait(PF[1], pf1); pf1^=1;
                pv(1,i-1,sp);
                mma_commit(rsel(KVE,sp));
                qk(1,s);
                mbar_wait(rsel(VF,s), rflip(pvf,s));         // V[i] ready (issued after K, lands slightly later)
                mbar_wait(OE[0], poe0); poe0^=1;
                mbar_wait(PF[0], pf0); pf0^=1;
                pv(0,i,s);
            }
            // epilogue: PV1(n-1)
            { int sl=(n-1)%FA4KVS;
              mbar_wait(OE[1], poe1); poe1^=1;
              mbar_wait(PF[1], pf1); pf1^=1;
              pv(1,n-1,sl);
              mma_commit(rsel(KVE,sl)); }
            mma_commit(SF[0]); mma_commit(SF[1]);          // final dummy S commits (pair softmax terminal SF wait)
        }
    } else if (warp>=8 && warp<12) {     // -------- CORRECTION WG (128 threads) --------
        const uint32_t OF[2]={smem_u32(&bars.o_full[0]),smem_u32(&bars.o_full[1])};
        const uint32_t OE[2]={smem_u32(&bars.o_empty[0]),smem_u32(&bars.o_empty[1])};
        // Per-half tables + rsel/rflip: as [2][2] arrays indexed by the runtime slot these lived in LOCAL
        // memory and were the fa4 family's dominant local-memory traffic (see rsel above).
        const uint32_t CF0[2]={smem_u32(&bars.c_full[0]),smem_u32(&bars.c_full[1])};
        const uint32_t CF1[2]={smem_u32(&bars.c_full[2]),smem_u32(&bars.c_full[3])};
        const uint32_t CE0[2]={smem_u32(&bars.c_empty[0]),smem_u32(&bars.c_empty[1])};
        const uint32_t CE1[2]={smem_u32(&bars.c_empty[2]),smem_u32(&bars.c_empty[3])};
        const int r=tid&127;
        int qf=0, po0=0, po1=0;                            // carry across tiles
        int pcf0[2]={0,0}, pcf1[2]={0,0};                  // D4 depth-2: per (half,slot) CF phase, carries across tiles
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
            const int k0=PAGED?0:cu_k[b], seqlen_k=PAGED?seqused_k[b]:(cu_k[b+1]-k0);
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            int n_lo=0;   // WINDOW: sliding-window tile-range prune (mirrors ws2 nlo/nhi @L6543). n=NB=n_hi-n_lo.
            if constexpr(WINDOW){ int qxw=seqlen_k-seqlen_q+row0+255, qnw=seqlen_k-seqlen_q+row0; int nhi=nk_all;
                if(is_causal){ int hc=(qxw>=0)?(qxw/WTN+1):0; nhi=min(nhi,hc); }
                if(win_r>=0){ int hr=(qxw+win_r>=0)?((qxw+win_r)/WTN+1):0; nhi=min(nhi,hr); }
                if(win_l>=0){ int lw=qnw-win_l; n_lo=(lw>0)?(lw/WTN):0; }
                nhi=max(nhi,n_lo); n=nhi-n_lo; if(n<1)n=1; }
            // prologue: swallow p=0 (block0 needs no rescale) both halves, release slot0
            mbar_wait(CF0[0], rflip(pcf0,0)); mbar_arrive(CE0[0]);
            mbar_wait(CF1[0], rflip(pcf1,0)); mbar_arrive(CE1[0]);
            for(int it=0; it<n-1; ++it){
                int sl=(it+1)&1;                            // production p=it+1
                mbar_wait(rsel(CF0,sl), rflip(pcf0,sl)); float sc0=sScale[0][sl][r];
                mbar_wait(OF[0], po0); po0^=1; tmem_fence();
                if(!__all_sync(0xffffffffu,sc0==1.0f)) tmem_o_rescale<HDV>(base+OCOL[0]+((uint32_t)r<<16), sc0);
                mbar_arrive(rsel(CE0,sl));
                tmem_wait_st(); mbar_arrive(OE[0]);
                mbar_wait(rsel(CF1,sl), rflip(pcf1,sl)); float sc1=sScale[1][sl][r];
                mbar_wait(OF[1], po1); po1^=1; tmem_fence();
                if(!__all_sync(0xffffffffu,sc1==1.0f)) tmem_o_rescale<HDV>(base+OCOL[1]+((uint32_t)r<<16), sc1);
                mbar_arrive(rsel(CE1,sl));
                tmem_wait_st(); mbar_arrive(OE[1]);
            }
            int fsl=n&1;                                    // finsum production p=n slot
            // ---- final stage 0: normalize O0 -> DIRECT global write + lse ----
            mbar_wait(rsel(CF0,fsl), rflip(pcf0,fsl)); float fs0=sFinSum[0][fsl][r], fm0=sFinMax[0][fsl][r];
            mbar_arrive(rsel(CE0,fsl));
            mbar_wait(OF[0], po0); po0^=1; tmem_fence();
            { float inv; if constexpr(SINK){ float osc0=sFinOsc[0][fsl][r]; inv=(fs0>0.f)?osc0/fs0:0.f; } else inv=(fs0>0.f)?1.f/fs0:0.f;
              unsigned long long inv2=bcast2(inv);
              uint32_t oA=base+OCOL[0]+((uint32_t)r<<16); int qr=row0+r; bool ow=(qr<seqlen_q);
              size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv):0;
              #pragma unroll 1
              for(int g=0; g<HDV; g+=32){ uint32_t ov[32]; tmem_ld32(oA+(uint32_t)g, ov); tmem_wait_ld();
                #pragma unroll
                for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);
                if(ow){
                    #pragma unroll
                    for(int t=0;t<32;t+=8){ int d=g+t;
                        if(d+8<=rhdv){ uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                            vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3])); vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                            vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7])); *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                        } else { for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); } } } }
              }
              if(ow) lse[(size_t)h*lse_stride+(q0+qr)] = (fs0>0.f)?(logf(fs0)+fm0*INV_LOG2E):-INFINITY;
            }
            tmem_wait_ld(); mbar_arrive(OE[0]);
            // ---- final stage 1: normalize O1 -> DIRECT global write + lse ----
            mbar_wait(rsel(CF1,fsl), rflip(pcf1,fsl)); float fs1=sFinSum[1][fsl][r], fm1=sFinMax[1][fsl][r];
            mbar_arrive(rsel(CE1,fsl));
            mbar_wait(OF[1], po1); po1^=1; tmem_fence();
            { float inv; if constexpr(SINK){ float osc1=sFinOsc[1][fsl][r]; inv=(fs1>0.f)?osc1/fs1:0.f; } else inv=(fs1>0.f)?1.f/fs1:0.f;
              unsigned long long inv2=bcast2(inv);
              uint32_t oA=base+OCOL[1]+((uint32_t)r<<16); int qr=row0+WTM+r; bool ow=(qr<seqlen_q);
              size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv):0;
              #pragma unroll 1
              for(int g=0; g<HDV; g+=32){ uint32_t ov[32]; tmem_ld32(oA+(uint32_t)g, ov); tmem_wait_ld();
                #pragma unroll
                for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);
                if(ow){
                    #pragma unroll
                    for(int t=0;t<32;t+=8){ int d=g+t;
                        if(d+8<=rhdv){ uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                            vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3])); vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                            vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7])); *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                        } else { for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); } } } }
              }
              if(ow) lse[(size_t)h*lse_stride+(q0+qr)] = (fs1>0.f)?(logf(fs1)+fm1*INV_LOG2E):-INFINITY;
            }
            tmem_wait_ld(); mbar_arrive(OE[1]);
            // D4 depth-2: per (half,slot) CE arrivals == CF arrivals == productions into that slot (p=0..n
            // mapped by p&1), so softmax ce-waits == correction ce-arrives exactly per slot per tile. The
            // ONE-time free pass (pce0=pce1=1 at kernel start) supplies the initial 2-block lead; no per-tile
            // rebalancing needed (unlike the depth-1 pre-loop-acquire+drain path).
        }
    }
    __syncthreads();
    if(warp==12) tmem_dealloc(base,512);
}

// ================= FA4-FAITHFUL warp-specialized kernel, DECOUPLED K/V RING (fwd_kernel_fa4dk) ===========
// Identical structure/choreography to fwd_kernel_fa4, but with the K and V smem rings DECOUPLED to distinct
// depths (K depth KVS, V depth VST) so head-dim 192 fits the 227KB opt-in smem cap: sQ(2 halves@HD=192,98KB)
// + K depth-2@192 (98KB) + V depth-1@128 (32KB) = 230400B < 232448B (K2/V2 would be 263KB -> overflow). The
// tmem S/O layout (S0@0 S1@128 O0@256 O1@384, 512 cols) is IDENTICAL to hd128 -- only the QK contracts over
// HD (NKT_QK=HD/16=12 k-tiles vs 8) and Q/K smem is HD-wide. Warp roles differ from fwd_kernel_fa4 ONLY in the
// load path: w13 = K-LOAD (ring KVS), w15 = V-LOAD (ring VST), w14 = Q-producer alone (bar.sync 3,32 vs ,64);
// the shared kv_empty barrier is split into k_empty[KVS] (freed after QK last-reads K) and v_empty[VST] (freed
// after PV1 last-reads V). Everything else (softmax w0-7, correction w8-11, MMA w12, the 9-pipeline barrier
// choreography incl. the atomic tile-counter self-reset) is byte-for-byte the same as fwd_kernel_fa4.
struct BarsFA4dk { uint64_t kf_full[FA4KVS], k_empty[FA4KVS], vf_full[FA4KVS], v_empty[FA4KVS], s_full[2], p_full[2], o_full[2], o_empty[2], c_full[4], c_empty[4], q_full[1], q_empty[1], q_tma[1], q_sched[1], ord[2]; };
template<class T, int HD, int HDV, int KVS, int VST>
__global__ void __launch_bounds__(512,1) fwd_kernel_fa4dk(
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
    const __grid_constant__ CUtensorMap mapQ,
    const T* __restrict__ q, T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int* __restrict__ tile_ctr, int total_tiles, int gx, int nB, int ragged) {
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64, VCB=HDV/64;
    const uint32_t SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[KVS]; T* sV[VST];
    { T* p=sQ+2*WTM*HD; for(int s=0;s<KVS;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<VST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) BarsFA4dk bars; __shared__ __align__(8) uint32_t s_base; __shared__ int s_tile;
    __shared__ float sScale[2][2][WTM];   // [half][slot=prod&1][row]: per-block corr factor AND (at the final
    // production p=n, slot n&1) the tile's final row-SUM l_i handed to the correction WG. sFinSum/sFinMax are
    // REMOVED to fit hd192's 230KB smem: the final MAX only feeds LSE, which the SOFTMAX warp now writes directly.
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f, INV_LOG2E=0.6931471805599453f;
    const float qsc=scale*LOG2E;
    const int lse_stride=cu_q[nB];
    if(warp>=8) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");
    if(warp==12){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base=s_base;
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();
    if(tid==0){ for(int s=0;s<KVS;s++){ mbar_init(smem_u32(&bars.kf_full[s]),1); mbar_init(smem_u32(&bars.k_empty[s]),1); }
        for(int s=0;s<VST;s++){ mbar_init(smem_u32(&bars.vf_full[s]),1); mbar_init(smem_u32(&bars.v_empty[s]),1); }
        for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.p_full[j]),128);
            mbar_init(smem_u32(&bars.o_full[j]),1); mbar_init(smem_u32(&bars.o_empty[j]),128);
            mbar_init(smem_u32(&bars.c_full[2*j]),128); mbar_init(smem_u32(&bars.c_full[2*j+1]),128); mbar_init(smem_u32(&bars.c_empty[2*j]),128); mbar_init(smem_u32(&bars.c_empty[2*j+1]),128); }
        mbar_init(smem_u32(&bars.q_full[0]),1); mbar_init(smem_u32(&bars.q_empty[0]),2); mbar_init(smem_u32(&bars.q_tma[0]),1);
        mbar_init(smem_u32(&bars.q_sched[0]),1);
        mbar_init(smem_u32(&bars.ord[0]),128); mbar_init(smem_u32(&bars.ord[1]),128); }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t QFULL=smem_u32(&bars.q_full[0]), QEMPTY=smem_u32(&bars.q_empty[0]);
    const uint32_t QSCHED=smem_u32(&bars.q_sched[0]);

    if (warp==14) {                      // -------- Q PRODUCER (w14 ONLY): scheduler + TMA-Q (RAW, no prescale) --------
        const bool sched=(tid==448);
        const uint32_t QTMA=smem_u32(&bars.q_tma[0]);
        int i=0, tile=blockIdx.x, qe=0, qt=0;
        for(;;){
            if(sched){ if(i>0){ mbar_wait(QEMPTY,qe); qe^=1; }
                       while(ragged && tile<total_tiles){   // ragged-batch empty-tile skip (see fwd_kernel_fa4)
                           const int per=hq*nB, lvl=tile/per, rem=tile%per;
                           const int mb=(gx-1)-lvl, b=rem%nB;
                           if((2*mb)*WTM < cu_q[b+1]-cu_q[b]) break;
                           int nx=atomicAdd(tile_ctr,1);
                           if(nx==total_tiles+(int)gridDim.x-1) atomicExch(tile_ctr,(int)gridDim.x);
                           tile=nx;
                       }
                       s_tile=(tile<total_tiles)?tile:-1;
                       mbar_arrive(QSCHED); }
            asm volatile("bar.sync 3, 32;");            // w14 only (w15 is V-load) -> 32-thread named barrier
            int t=s_tile;
            if(sched){
                if(t>=0){
                    const int per=hq*nB, lvl=t/per, rem=t%per;
                    const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
                    const int q0=cu_q[b];
                    const int r00=q0+(2*mb)*WTM, r01=q0+(2*mb+1)*WTM;
                    mbar_expect(QTMA, 2*WTM*HD*(int)sizeof(T));
                    for(int cb=0;cb<KCB;cb++){ tma_box(sQ+0*WTM*HD+cb*8192, &mapQ, h*rhd+cb*64, r00, QTMA);
                                               tma_box(sQ+1*WTM*HD+cb*8192, &mapQ, h*rhd+cb*64, r01, QTMA); }
                    mbar_wait(QTMA,qt); qt^=1;
                }
                mbar_arrive(QFULL);
            }
            if(t<0) break;
            if(sched){ tile=atomicAdd(tile_ctr,1);
                if(tile==total_tiles+(int)gridDim.x-1) atomicExch(tile_ctr,(int)gridDim.x); }
            i++;
        }
    } else if (warp==15) {               // -------- V-LOAD warp: TMA V ring (depth VST, decoupled from K) --------
        uint32_t VF[VST],VE[VST]; for(int s=0;s<VST;s++){ VF[s]=smem_u32(&bars.vf_full[s]); VE[s]=smem_u32(&bars.v_empty[s]); }
        bool lead=(tid&31)==0; int pe[VST]; for(int s=0;s<VST;s++) pe[s]=1;   // carry
        int qf=0;
        for(;;){
            mbar_wait(QSCHED,qf); qf^=1;                    // s_tile ready (before Q prescale) => V TMA overlaps
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB, hk=h/rep;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            for(int i=0;i<n;++i){ int s=i%VST; mbar_wait(rsel(VE,s), rflip(pe,s));   // rsel/rflip: ring state indexed by a runtime slot lives in LOCAL memory otherwise (see file header)
                if(lead){ int y=k0+i*WTN;
                    const uint32_t vfb=rsel(VF,s); T* const vbs=rsel(sV,s);
                    mbar_expect(vfb, WTN*HDV*2);
                    for(int cb=0;cb<VCB;cb++) tma_box(vbs+cb*8192, &mapV, hk*rhdv+cb*64, y, vfb); } }
        }
    } else if (warp<8) {                 // -------- SOFTMAX half j = warp>>2 --------
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n" ::: "memory");
        const int j=warp>>2, lr=tid&127; const bool wglead=((tid&127)==0);
        const uint32_t sf=smem_u32(&bars.s_full[j]), pf=smem_u32(&bars.p_full[j]);
        const uint32_t cf0=smem_u32(&bars.c_full[2*j]), cf1=smem_u32(&bars.c_full[2*j+1]);
        const uint32_t ce0=smem_u32(&bars.c_empty[2*j]), ce1=smem_u32(&bars.c_empty[2*j+1]);
        const uint32_t sAddr=base+SCOL[j]+((uint32_t)lr<<16);
        const uint32_t pAddr=base+PCOL[j]+((uint32_t)lr<<16);
        const uint32_t ordw=smem_u32(&bars.ord[j]), orda=smem_u32(&bars.ord[j^1]);
        int qf=0, psf=0, ordph=(j==0)?1:0;
        int pce0=1, pce1=1;
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            const int qp=seqlen_k-seqlen_q+row0+j*WTM+lr;
            float m_i=-INFINITY, l_i=0.f;
            for(int i=0;i<n;++i){
                int sl=i&1;
                if(sl==0){ mbar_wait(ce0, pce0); pce0^=1; } else { mbar_wait(ce1, pce1); pce1^=1; }
                const uint32_t cf=(sl==0)?cf0:cf1;
                mbar_wait(sf, psf); psf^=1; tmem_fence();
                if(i==n-1 && wglead) mbar_arrive(QEMPTY);
                mbar_wait(ordw, ordph); ordph^=1;
                int key0=i*WTN;
                const int qbase=seqlen_k-seqlen_q+row0+j*WTM;
                const int kend=key0+WTN-1;
                bool full=(kend<seqlen_k) && !(is_causal && kend>qbase);
                float m_new, ps=0.f;
                if(full){
                    softmax_full_ptx_ordcs<T>(sAddr, pAddr, m_i, m_new, ps, orda, smem_u32(&sScale[j][sl][lr]), cf, qsc);
                } else {
                    int chi=seqlen_k-1-key0; if(chi>WTN-1) chi=WTN-1;
                    if(is_causal){ int cc=qp-key0; if(cc<chi) chi=cc; }
                    float rmax=-INFINITY;
                    #pragma unroll 1
                    for(int g=0; g<128; g+=32){ uint32_t ch[32]; tmem_ld32(sAddr+(uint32_t)g, ch); tmem_wait_ld();
                        #pragma unroll
                        for(int c=0;c<32;++c){ if(g+c<=chi) rmax=fmaxf(rmax,__uint_as_float(ch[c])); } }
                    rmax*=qsc;
                    mbar_arrive(orda);
                    m_new=fmaxf(m_i,rmax);
                    float cb=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                    sScale[j][sl][lr]=cb; mbar_arrive(cf);
                    bool ok=(m_new!=-INFINITY);
                    #pragma unroll 1
                    for(int g=0; g<128; g+=32){ uint32_t ch[32]; tmem_ld32(sAddr+(uint32_t)g, ch); tmem_wait_ld();
                        uint32_t pk[16];
                        #pragma unroll
                        for(int t=0;t<16;++t){ int c0=g+2*t,c1=c0+1; float s0=__uint_as_float(ch[2*t])*qsc, s1=__uint_as_float(ch[2*t+1])*qsc;
                            float p0=(ok&&c0<=chi)?ex2f(s0-m_new):0.f, p1=(ok&&c1<=chi)?ex2f(s1-m_new):0.f; ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                        tmem_st16(pAddr+(uint32_t)(g/2), pk); }
                }
                float corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                tmem_wait_st(); mbar_arrive(pf);
                l_i=l_i*corr+ps; m_i=m_new;
            }
            mbar_wait(sf, psf); psf^=1;
            int fsl=n&1;
            if(fsl==0){ mbar_wait(ce0, pce0); pce0^=1; } else { mbar_wait(ce1, pce1); pce1^=1; }
            sScale[j][fsl][lr]=l_i;                         // final row-SUM -> correction (reuses sScale slot fsl)
            { int qr=row0+j*WTM+lr;                         // SOFTMAX writes LSE directly (correction no longer holds m_i)
              if(qr<seqlen_q) lse[(size_t)h*lse_stride+(q0+qr)] = (l_i>0.f)?(logf(l_i)+m_i*INV_LOG2E):-INFINITY; }
            mbar_arrive((fsl==0)?cf0:cf1);
        }
    } else if (warp==13) {               // -------- K-LOAD warp: TMA K ring (depth KVS, decoupled from V) --------
        uint32_t KF[KVS],KE[KVS]; for(int s=0;s<KVS;s++){ KF[s]=smem_u32(&bars.kf_full[s]); KE[s]=smem_u32(&bars.k_empty[s]); }
        bool lead=(tid&31)==0; int pe[KVS]; for(int s=0;s<KVS;s++) pe[s]=1;   // carry
        int qf=0;
        for(;;){
            mbar_wait(QSCHED,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB, hk=h/rep;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            for(int i=0;i<n;++i){ int s=i%KVS; mbar_wait(rsel(KE,s), rflip(pe,s));   // rsel/rflip: ring state indexed by a runtime slot lives in LOCAL memory otherwise (see file header)
                if(lead){ int y=k0+i*WTN;
                    const uint32_t kfb=rsel(KF,s); T* const kbs=rsel(sK,s);
                    mbar_expect(kfb, WTN*HD*2);
                    for(int cb=0;cb<KCB;cb++) tma_box(kbs+cb*8192, &mapK, hk*rhd+cb*64, y, kfb); } }
        }
    } else if (warp==12) {               // -------- MMA warp --------
        uint32_t KF[KVS],KE[KVS]; for(int s=0;s<KVS;s++){ KF[s]=smem_u32(&bars.kf_full[s]); KE[s]=smem_u32(&bars.k_empty[s]); }
        uint32_t VF[VST],VE[VST]; for(int s=0;s<VST;s++){ VF[s]=smem_u32(&bars.vf_full[s]); VE[s]=smem_u32(&bars.v_empty[s]); }
        const uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
        const uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
        const uint32_t OF[2]={smem_u32(&bars.o_full[0]),smem_u32(&bars.o_full[1])};
        const uint32_t OE[2]={smem_u32(&bars.o_empty[0]),smem_u32(&bars.o_empty[1])};
        const bool lead=(tid&31)==0;
        uint64_t qd0a[2], kd0a[KVS], vd0a[VST];
        #pragma unroll
        for(int jj=0;jj<2;jj++) qd0a[jj]=make_desc(smem_u32(sQ+jj*WTM*HD),16,1024,2);
        #pragma unroll
        for(int s=0;s<KVS;s++){ kd0a[s]=make_desc(smem_u32(sK[s]),16,1024,2); }
        #pragma unroll
        for(int s=0;s<VST;s++){ vd0a[s]=make_desc(smem_u32(sV[s]),16384,1024,2); }
        auto qk=[&](int j,int s){ if(lead){ uint64_t qd0=qd0a[j],kd0=rsel(kd0a,s);
                for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                        ::"r"(base+SCOL[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); } }
            mma_commit(SF[j]); };
        auto pv=[&](int j,int i,int s){ if(lead){ uint64_t vd0=rsel(vd0a,s);
                for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                        ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); } }
            mma_commit(OF[j]); };
        int qf=0, qfp=0, pkf[KVS]={0}, pvf[VST]={0}, poe0=1, poe1=1, pf0=0, pf1=0;   // ALL carry across tiles
        for(;;){
            mbar_wait(QSCHED,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, b=rem%nB;
            const int seqlen_q=cu_q[b+1]-cu_q[b], seqlen_k=cu_k[b+1]-cu_k[b];
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            // prologue: block 0 QK0,QK1 + PV0(0).
            mbar_wait(KF[0], pkf[0]); pkf[0]^=1;
            mbar_wait(QFULL,qfp); qfp^=1;
            qk(0,0); qk(1,0);
            mbar_wait(OE[0], poe0); poe0^=1;
            mbar_wait(PF[0], pf0); pf0^=1;
            mbar_wait(VF[0], pvf[0]); pvf[0]^=1;
            pv(0,0,0);
            // steady loop i=1..n-1: K slot sk=i%KVS, V slot sv=i%VST; empties freed after PV1 last-reads V[i-1]
            for(int i=1;i<n;++i){ int sk=i%KVS, spk=(i-1)%KVS, sv=i%VST, spv=(i-1)%VST;
                mbar_wait(KF[sk], pkf[sk]); pkf[sk]^=1;      // K[i] ready -> QK runs ahead of V[i] load
                qk(0,sk);
                mbar_wait(OE[1], poe1); poe1^=1;
                mbar_wait(PF[1], pf1); pf1^=1;
                pv(1,i-1,spv);                               // last read of V[i-1]
                mma_commit(VE[spv]);                         // free V[i-1] (V ring)
                mma_commit(KE[spk]);                         // free K[i-1] (K ring; K[i-1] done since qk(1,i-1))
                qk(1,sk);
                mbar_wait(VF[sv], pvf[sv]); pvf[sv]^=1;       // V[i] ready
                mbar_wait(OE[0], poe0); poe0^=1;
                mbar_wait(PF[0], pf0); pf0^=1;
                pv(0,i,sv);
            }
            // epilogue: PV1(n-1)
            { int slk=(n-1)%KVS, slv=(n-1)%VST;
              mbar_wait(OE[1], poe1); poe1^=1;
              mbar_wait(PF[1], pf1); pf1^=1;
              pv(1,n-1,slv);
              mma_commit(VE[slv]); mma_commit(KE[slk]); }
            mma_commit(SF[0]); mma_commit(SF[1]);          // final dummy S commits (pair softmax terminal SF wait)
        }
    } else if (warp>=8 && warp<12) {     // -------- CORRECTION WG (128 threads) --------
        const uint32_t OF[2]={smem_u32(&bars.o_full[0]),smem_u32(&bars.o_full[1])};
        const uint32_t OE[2]={smem_u32(&bars.o_empty[0]),smem_u32(&bars.o_empty[1])};
        // Per-half tables + rsel/rflip: as [2][2] arrays indexed by the runtime slot these lived in LOCAL
        // memory and were the fa4 family's dominant local-memory traffic (see rsel above).
        const uint32_t CF0[2]={smem_u32(&bars.c_full[0]),smem_u32(&bars.c_full[1])};
        const uint32_t CF1[2]={smem_u32(&bars.c_full[2]),smem_u32(&bars.c_full[3])};
        const uint32_t CE0[2]={smem_u32(&bars.c_empty[0]),smem_u32(&bars.c_empty[1])};
        const uint32_t CE1[2]={smem_u32(&bars.c_empty[2]),smem_u32(&bars.c_empty[3])};
        const int r=tid&127;
        int qf=0, po0=0, po1=0;
        int pcf0[2]={0,0}, pcf1[2]={0,0};
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
            const int row0=(2*mb)*WTM;
            const int nk_all=(seqlen_k+WTN-1)/WTN; int n=nk_all;
            if(is_causal){ int qx=seqlen_k-seqlen_q+(row0+255); int hi=(qx>=0)?(qx/WTN+1):0; n=min(nk_all,hi); if(n<1)n=1; }
            mbar_wait(CF0[0], rflip(pcf0,0)); mbar_arrive(CE0[0]);
            mbar_wait(CF1[0], rflip(pcf1,0)); mbar_arrive(CE1[0]);
            for(int it=0; it<n-1; ++it){
                int sl=(it+1)&1;
                mbar_wait(rsel(CF0,sl), rflip(pcf0,sl)); float sc0=sScale[0][sl][r];
                mbar_wait(OF[0], po0); po0^=1; tmem_fence();
                if(!__all_sync(0xffffffffu,sc0==1.0f)) tmem_o_rescale<HDV>(base+OCOL[0]+((uint32_t)r<<16), sc0);
                mbar_arrive(rsel(CE0,sl));
                tmem_wait_st(); mbar_arrive(OE[0]);
                mbar_wait(rsel(CF1,sl), rflip(pcf1,sl)); float sc1=sScale[1][sl][r];
                mbar_wait(OF[1], po1); po1^=1; tmem_fence();
                if(!__all_sync(0xffffffffu,sc1==1.0f)) tmem_o_rescale<HDV>(base+OCOL[1]+((uint32_t)r<<16), sc1);
                mbar_arrive(rsel(CE1,sl));
                tmem_wait_st(); mbar_arrive(OE[1]);
            }
            int fsl=n&1;
            mbar_wait(rsel(CF0,fsl), rflip(pcf0,fsl)); float fs0=sScale[0][fsl][r];   // final SUM (softmax wrote LSE)
            mbar_arrive(rsel(CE0,fsl));
            mbar_wait(OF[0], po0); po0^=1; tmem_fence();
            { float inv=(fs0>0.f)?1.f/fs0:0.f; unsigned long long inv2=bcast2(inv);
              uint32_t oA=base+OCOL[0]+((uint32_t)r<<16); int qr=row0+r; bool ow=(qr<seqlen_q);
              size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv):0;
              #pragma unroll 1
              for(int g=0; g<HDV; g+=32){ uint32_t ov[32]; tmem_ld32(oA+(uint32_t)g, ov); tmem_wait_ld();
                #pragma unroll
                for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);
                if(ow){
                    #pragma unroll
                    for(int t=0;t<32;t+=8){ int d=g+t;
                        if(d+8<=rhdv){ uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                            vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3])); vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                            vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7])); *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                        } else { for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); } } } }
              }
            }
            tmem_wait_ld(); mbar_arrive(OE[0]);
            mbar_wait(rsel(CF1,fsl), rflip(pcf1,fsl)); float fs1=sScale[1][fsl][r];   // final SUM (softmax wrote LSE)
            mbar_arrive(rsel(CE1,fsl));
            mbar_wait(OF[1], po1); po1^=1; tmem_fence();
            { float inv=(fs1>0.f)?1.f/fs1:0.f; unsigned long long inv2=bcast2(inv);
              uint32_t oA=base+OCOL[1]+((uint32_t)r<<16); int qr=row0+WTM+r; bool ow=(qr<seqlen_q);
              size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv):0;
              #pragma unroll 1
              for(int g=0; g<HDV; g+=32){ uint32_t ov[32]; tmem_ld32(oA+(uint32_t)g, ov); tmem_wait_ld();
                #pragma unroll
                for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);
                if(ow){
                    #pragma unroll
                    for(int t=0;t<32;t+=8){ int d=g+t;
                        if(d+8<=rhdv){ uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                            vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3])); vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                            vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7])); *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                        } else { for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); } } } }
              }
            }
            tmem_wait_ld(); mbar_arrive(OE[1]);
        }
    }
    __syncthreads();
    if(warp==12) tmem_dealloc(base,512);
}

// ============ CONTINUOUS-OVERLAP PERSISTENT prefill (dense non-paged hd128/hd64, bf16/fp16) ============
// Fixes the persistent kernel's tile-boundary BUBBLE: no CTA barrier in the steady-state loop, so tile N's
// softmax epilogue overlaps tile N+1's mainloop. Roles: w0-3 sm0, w4-7 sm1, w8 MMA, w9 K/V-load,
// w10 Q-load+scheduler (grabs tiles via atomic, LPT decode, loads Q into a SINGLE sQ), w11-15 idle(return).
// Cross-tile handshakes (all mbar, no __syncthreads): q_full (w10->consumers: Q + s_tile ready),
// q_empty (both softmax WGs -> w10: SF fully consumed => QKs retired => sQ free — no async race),
// o_free[j] (softmax WG j -> MMA: O[j] tmem read, safe to overwrite). Ring KVE/KVF + SF/PF/OR CARRY.
// Gated dense-only (uniform seqlen => no idle tiles). LPT (heaviest-mb first) balances the causal wave.
template<class T, int HD, int HDV>
__global__ void __launch_bounds__(512,1) fwd_kernel_ws2_cont(
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
    const __grid_constant__ CUtensorMap mapQ,
    const T* __restrict__ q, T* __restrict__ out, float* __restrict__ lse,
    const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int* __restrict__ tile_ctr, int total_tiles, int gx, int nB, int ragged) {
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64, VCB=HDV/64;
    const int SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[WST];
    { T* p=sQ+2*WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<WST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) Bars bars; __shared__ __align__(8) uint32_t s_base; __shared__ int s_tile;
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    const float qsc=(softcap>0.f)?scale:scale*LOG2E;
    const int lse_stride=cu_q[nB];
    if (warp>=8) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");
    if(warp==8){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    __syncthreads();
    const uint32_t base=s_base;
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    const uint32_t QFULL=smem_u32(&bars.s_empty[0]), QEMPTY=smem_u32(&bars.s_empty[1]);
    const uint32_t OFREE[2]={smem_u32(&bars.corr_stats[0]),smem_u32(&bars.corr_stats[1])};
    const uint32_t QTMA=smem_u32(&bars.p_last[0]);   // Q TMA-load completion (producer-only)
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(KVF[s],1); mbar_init(KVE[s],1);}
                for(int j=0;j<2;j++){ mbar_init(SF[j],1); mbar_init(PF[j],128); mbar_init(OR[j],1); mbar_init(OFREE[j],128);}
                mbar_init(QFULL,1); mbar_init(QEMPTY,2); mbar_init(QTMA,1); }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    if(tid==0){ for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); }   // prime K/V ring ONCE

    if (warp>=10) {                       // ---- Q-LOAD via TMA (w10 leader issues) + in-place scale (w10-15) ----
        const int ptid=tid-320;           // 0..191 within the producer group
        const bool sched=(tid==320);      // w10 lane0 = scheduler + TMA issuer
        int i=0, tile=blockIdx.x, qe=0, qt=0;
        for(;;){
            if(sched){ if(i>0){ mbar_wait(QEMPTY,qe); qe^=1; } s_tile = (tile<total_tiles)?tile:-1; }
            asm volatile("bar.sync 3, 192;");            // observe s_tile (+ prev sQ freed via sched's QEMPTY wait)
            int t=s_tile;
            if(t>=0){
                const int per=hq*nB, lvl=t/per, rem=t%per;
                const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
                const int q0=cu_q[b];
                const int r00=q0+(2*mb)*WTM, r01=q0+(2*mb+1)*WTM;   // global token rows for the 2 Q-tiles
                if(sched){ mbar_expect(QTMA, 2*WTM*HD*(int)sizeof(T));
                    for(int cb=0;cb<KCB;cb++){ tma_box(sQ+0*WTM*HD+cb*8192, &mapQ, h*rhd+cb*64, r00, QTMA);
                                               tma_box(sQ+1*WTM*HD+cb*8192, &mapQ, h*rhd+cb*64, r01, QTMA); } }
                mbar_wait(QTMA,qt); qt^=1;               // async Q TMA done (copy engine); now prescale in smem
                for(int e=ptid*8; e<2*WTM*HD; e+=192*8){ uint4* p=(uint4*)&sQ[e]; uint4 v=*p; T* pp=(T*)&v;
                    #pragma unroll
                    for(int k=0;k<8;k++) pp[k]=from_f<T>(to_f<T>(pp[k])*qsc); *p=v; }
            }
            asm volatile("fence.proxy.async.shared::cta;");
            asm volatile("bar.sync 3, 192;");            // all producer sQ scale-writes + fences done
            if(sched) mbar_arrive(QFULL);
            if(t<0) break;
            if(sched) tile=atomicAdd(tile_ctr,1);
            i++;
        }
    } else if (warp==9) {                 // ---- K/V LOAD ----
        bool lead=(tid&31)==0; int qf=0; int pe[WST]={0};
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB, hk=h/rep;
            const int seqlen_q=cu_q[b+1]-cu_q[b], k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
            const int row0_1=(2*mb+1)*WTM;
            int cqmax=seqlen_k-seqlen_q+row0_1+(WTM-1), cqmin=seqlen_k-seqlen_q+(2*mb)*WTM;
            int n_lo=0, n_hi=(seqlen_k+WTN-1)/WTN;
            if(is_causal) n_hi=min(n_hi,cqmax/WTN+1); if(win_r>=0) n_hi=min(n_hi,(cqmax+win_r)/WTN+1);
            if(win_l>=0){ int l=cqmin-win_l; n_lo=(l>0)?l/WTN:0; } n_hi=max(n_hi,n_lo);
            const int NBk=n_hi-n_lo;
            for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(KVE[s],pe[s]); pe[s]^=1;
                if(lead){ int y=k0+(n_lo+i)*WTN; uint32_t bar=KVF[s];
                    mbar_expect(bar, WTN*HD*2 + WTN*HDV*2);
                    for(int cb=0;cb<KCB;cb++) tma_box(sK[s]+cb*8192, &mapK, hk*rhd+cb*64, y, bar);
                    for(int cb=0;cb<VCB;cb++) tma_box(sV[s]+cb*8192, &mapV, hk*rhdv+cb*64, y, bar);
                } }
        }
    } else if (warp==8) {                 // ---- MMA ----
        const bool lead8=(tid&31)==0; int qf=0, pf[WST]={0}, ppf[2]={0,0}, ofph[2]={0,0}; int mtile=0;
        uint64_t vd0a[WST]; for(int s=0;s<WST;s++) vd0a[s]=make_desc(smem_u32(sV[s]),16384,1024,2);
        uint64_t kd0a[WST]; for(int s=0;s<WST;s++) kd0a[s]=make_desc(smem_u32(sK[s]),16,1024,2);
        uint64_t qd0a[2]; for(int j=0;j<2;j++) qd0a[j]=make_desc(smem_u32(sQ+j*WTM*HD),16,1024,2);
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, b=rem%nB;
            const int seqlen_q=cu_q[b+1]-cu_q[b], seqlen_k=cu_k[b+1]-cu_k[b];
            const int row0_0=(2*mb)*WTM, row0_1=(2*mb+1)*WTM; const bool a1=(row0_1<seqlen_q);
            int nlo0,nhi0,nlo1,nhi1;
            { int qx=seqlen_k-seqlen_q+row0_0+(WTM-1),qn=seqlen_k-seqlen_q+row0_0; nlo0=0; nhi0=(seqlen_k+WTN-1)/WTN;
              if(is_causal) nhi0=min(nhi0,qx/WTN+1); if(win_r>=0) nhi0=min(nhi0,(qx+win_r)/WTN+1);
              if(win_l>=0){int l=qn-win_l; nlo0=(l>0)?l/WTN:0;} nhi0=max(nhi0,nlo0); }
            if(a1){ int qx=seqlen_k-seqlen_q+row0_1+(WTM-1),qn=seqlen_k-seqlen_q+row0_1; nlo1=0; nhi1=(seqlen_k+WTN-1)/WTN;
              if(is_causal) nhi1=min(nhi1,qx/WTN+1); if(win_r>=0) nhi1=min(nhi1,(qx+win_r)/WTN+1);
              if(win_l>=0){int l=qn-win_l; nlo1=(l>0)?l/WTN:0;} nhi1=max(nhi1,nlo1); }
            else { nlo1=nlo0; nhi1=nhi0; }
            const int n_lo=min(nlo0,nlo1), n_hi=max(nhi0,nhi1), NBk=n_hi-n_lo;
            const int hi0=nhi0-n_lo, hi1=nhi1-n_lo;
            auto qk=[&](int j,int i){ int s=i%WST;
                if(lead8){ uint64_t qd0=qd0a[j],kd0=kd0a[s];
                    for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                        asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                            ::"r"(base+SCOL[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); } }
                mma_commit(SF[j]); };
            auto pv=[&](int j,int i){ int s=i%WST;
                if(lead8){ uint64_t vd0=vd0a[s];
                    #pragma unroll
                    for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                        asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                            ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); } }
                mma_commit(OR[j]); };
            // cross-tile O reuse: wait tile N-1's softmax done reading O[j] before overwriting (skip CTA's 1st tile)
            if(mtile>0){ mbar_wait(OFREE[0],ofph[0]); ofph[0]^=1; if(a1){ mbar_wait(OFREE[1],ofph[1]); ofph[1]^=1; } }
            mbar_wait(KVF[0],pf[0]); pf[0]^=1; if(hi0>0) qk(0,0); if(hi1>0) qk(1,0);
            for(int i=0;i<NBk;++i){ int s=i%WST; bool hn=(i+1<NBk); int ns=(i+1)%WST;
                if(i<hi0){ mbar_wait(PF[0],ppf[0]); ppf[0]^=1; pv(0,i); }
                if(hn){ mbar_wait(KVF[ns],pf[ns]); pf[ns]^=1; if(i+1<hi0) qk(0,i+1); }
                if(i<hi1){ mbar_wait(PF[1],ppf[1]); ppf[1]^=1; pv(1,i); }
                if(hn && i+1<hi1){ qk(1,i+1); }
                mma_commit(KVE[s]);
            }
            mtile++;
        }
    } else if (warp<8) {                  // ---- SOFTMAX (WG j) ----
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n" ::: "memory");
        const int j=warp>>2, lr=tid&127; const bool wglead=((tid&127)==0);
        int qf=0, psf=0, por=0;
        for(;;){
            mbar_wait(QFULL,qf); qf^=1;
            int tile=s_tile; if(tile<0) break;
            const int per=hq*nB, lvl=tile/per, rem=tile%per;
            const int mb=(gx-1)-lvl, h=rem/nB, b=rem%nB;
            const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0, k0=cu_k[b], seqlen_k=cu_k[b+1]-k0;
            const int row0_0=(2*mb)*WTM, row0_1=(2*mb+1)*WTM; const bool a1=(row0_1<seqlen_q);
            int nlo0,nhi0,nlo1,nhi1;
            { int qx=seqlen_k-seqlen_q+row0_0+(WTM-1),qn=seqlen_k-seqlen_q+row0_0; nlo0=0; nhi0=(seqlen_k+WTN-1)/WTN;
              if(is_causal) nhi0=min(nhi0,qx/WTN+1); if(win_r>=0) nhi0=min(nhi0,(qx+win_r)/WTN+1);
              if(win_l>=0){int l=qn-win_l; nlo0=(l>0)?l/WTN:0;} nhi0=max(nhi0,nlo0); }
            if(a1){ int qx=seqlen_k-seqlen_q+row0_1+(WTM-1),qn=seqlen_k-seqlen_q+row0_1; nlo1=0; nhi1=(seqlen_k+WTN-1)/WTN;
              if(is_causal) nhi1=min(nhi1,qx/WTN+1); if(win_r>=0) nhi1=min(nhi1,(qx+win_r)/WTN+1);
              if(win_l>=0){int l=qn-win_l; nlo1=(l>0)?l/WTN:0;} nhi1=max(nhi1,nlo1); }
            else { nlo1=nlo0; nhi1=nhi0; }
            const int n_lo=min(nlo0,nlo1);
            // WG j==1 with !a1 has no tile -> still must advance QEMPTY/OFREE counts consistently.
            const bool active=(j==0)||a1;
            float m_i=-INFINITY, l_i=0.f;
            const int row0j=(j==0?row0_0:row0_1);
            const int qp=seqlen_k-seqlen_q+row0j+lr;
            const int hij=active?(((j==0)?nhi0:nhi1)-n_lo):0;
            int key0off=0;
            for(int i=0;i<hij;++i){ key0off=(n_lo+i)*WTN;
                mbar_wait(SF[j],psf); psf^=1; tmem_fence();
                const int qbase=seqlen_k-seqlen_q+row0j, kend=key0off+WTN-1;
                bool full=(kend<seqlen_k) && !(softcap>0.f);
                if(is_causal && kend>qbase) full=false;
                if(win_r>=0 && kend>qbase+win_r) full=false;
                if(win_l>=0 && key0off < qbase+(WTM-1)-win_l) full=false;
                float m_new, corr, ps=0.f;
                if(full){
                    softmax_full_ptx<T>(base+SCOL[j]+((uint32_t)lr<<16), base+PCOL[j]+((uint32_t)lr<<16), m_i, m_new, ps);
                    corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                } else {
                    uint32_t f[128];
                    LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+0,  f, 0);
                    LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+32, f, 32);
                    LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+64, f, 64);
                    LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+96, f, 96);
                    tmem_wait_ld();
                    int chi=seqlen_k-1-key0off;
                    if(is_causal && qp-key0off<chi) chi=qp-key0off;
                    if(win_r>=0 && qp+win_r-key0off<chi) chi=qp+win_r-key0off;
                    if(chi>WTN-1) chi=WTN-1;
                    const int clo=(win_l>=0)?max(0,qp-win_l-key0off):0;
                    float rmax=-INFINITY;
                    #pragma unroll
                    for(int c=0;c<128;++c){ float sv=__uint_as_float(f[c]);
                        float lg=(softcap>0.f)?(softcap*tanhf(sv*inv_sc)*LOG2E):sv;
                        if(c<=chi && c>=clo) rmax=fmaxf(rmax,lg); }
                    m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                    #pragma unroll
                    for(int g=0;g<4;++g){ uint32_t pk[16];
                        #pragma unroll
                        for(int t=0;t<16;++t){ int c0=g*32+2*t, c1=c0+1;
                            float s0=__uint_as_float(f[c0]), s1=__uint_as_float(f[c1]);
                            float lg0=(softcap>0.f)?(softcap*tanhf(s0*inv_sc)*LOG2E):s0;
                            float lg1=(softcap>0.f)?(softcap*tanhf(s1*inv_sc)*LOG2E):s1;
                            float p0=(c0<=chi&&c0>=clo)?ex2f(lg0-m_new):0.f, p1=(c1<=chi&&c1>=clo)?ex2f(lg1-m_new):0.f; ps+=p0+p1;
                            pk[t]=pack16<T>(p0,p1); }
                        tmem_st16(base+PCOL[j]+((uint32_t)lr<<16)+g*16, pk); }
                }
                if(i>0){ mbar_wait(OR[j],por); por^=1;
                    if(!__all_sync(0xffffffffu,corr==1.0f)) tmem_o_rescale<HDV>(base+OCOL[j]+((uint32_t)lr<<16), corr); }
                l_i=l_i*corr+ps; m_i=m_new;
                tmem_wait_st(); mbar_arrive(PF[j]);
            }
            if(hij>0){ mbar_wait(OR[j],por); por^=1; } tmem_fence();
            // SF fully consumed => tile's QKs retired => sQ free. Signal q_empty (WG lead, count 2).
            if(wglead) mbar_arrive(QEMPTY);
            const int qr=row0j+lr;
            const float INV_LOG2E=0.6931471805599453f;
            float mfin=m_i, lfin=l_i, osc=1.f;
            if(s_aux!=nullptr){ float sk2=s_aux[h]*LOG2E; float mn=fmaxf(m_i,sk2); float c2=(m_i==-INFINITY)?0.f:ex2f(m_i-mn); lfin=l_i*c2+ex2f(sk2-mn); osc=c2; mfin=mn; }
            float inv=(lfin>0.f)?osc/lfin:0.f;
            const bool ow=active && (qr<seqlen_q);
            const size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)h*rhdv):0;
            const unsigned long long inv2=bcast2(inv);
            if(active){
                #pragma unroll
                for(int g=0;g<HDV;g+=32){ uint32_t ov[32]; tmem_ld32(base+OCOL[j]+((uint32_t)lr<<16)+(uint32_t)g, ov); tmem_wait_ld();
                    #pragma unroll
                    for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);
                    if(ow){
                        #pragma unroll
                        for(int t=0;t<32;t+=8){ int d=g+t;
                            if(d+8<=rhdv){ uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                                vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3]));
                                vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                                vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7]));
                                *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                            } else { for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); } } } }
                }
            }
            // O[j] tmem read done -> free for next tile's PV (count 128; all 128 WG threads arrive)
            if(active) mbar_arrive(OFREE[j]);
            if(active && qr<seqlen_q) lse[(size_t)h*lse_stride+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin*INV_LOG2E):-INFINITY;
        }
    }
    __syncthreads();   // all warps done (softmax finished reading O) before tmem free
    if(warp==8) tmem_dealloc(base,512);
}

// HW-async paged K/V gather via TMA tile::gather4 into the koff/voff swizzled smem the MMA reads.
// Replaces the manual cp.async.cg gather (~4096 16B issues/block across the load warps) with ~(WTN/4)*
// (HD/64 + HDV/64) gather4 issues/block (128 for hd128), each moving 4 rows x 128B HW-async -> frees the
// load warps' issue slots (ncu: -34% total instr on ws_paged). One mbarrier (bar); the caller sets
// expect_tx (WTN*HD*2+WTN*HDV*2) via lid0 BEFORE the prow barrier so it is ordered ahead of every gather.
// prowS[r] = physical row for key r (-1 = invalid -> clamped to row 0; softmax masks it / PV mult by p=0).
// Requires rhd==HD && rhdv==HDV (fixed 64-col gather blocks can't zero-pad a partial-hd column-block).
template<class T,int HD,int HDV>
__device__ __forceinline__ void tma_gather4_kv(const CUtensorMap* mK,const CUtensorMap* mV,
        T* sKs, T* sVs, const int* prowS, int hk, int rhd, int rhdv, uint32_t bar, int lid, int LN){
    constexpr int KCB=HD/64, VCB=HDV/64, NG=WTN/4;   // 32 groups of 4 keys per column-block
    constexpr int NGK=NG*KCB, NGT=NGK+NG*VCB;
    for(int op=lid; op<NGT; op+=LN){
        bool isK=op<NGK; int o=isK?op:op-NGK; int cb=o/NG, g=o%NG;
        int y0=prowS[4*g],y1=prowS[4*g+1],y2=prowS[4*g+2],y3=prowS[4*g+3];
        y0=y0>=0?y0:0; y1=y1>=0?y1:0; y2=y2>=0?y2:0; y3=y3>=0?y3:0;
        if(isK) tma_gather4(sKs+cb*8192+4*g*64, mK, hk*rhd +cb*64, y0,y1,y2,y3, bar);
        else    tma_gather4(sVs+cb*8192+4*g*64, mV, hk*rhdv+cb*64, y0,y1,y2,y3, bar);
    }
}

// WARP-UNIFORM gather4 producer. `cp.async.bulk.tensor` takes its operands in UNIFORM registers, so a
// gather4 issued from divergent lanes makes ptxas emit an ELECT / R2UR.BROADCAST / `BRA.U.ANY` replay loop
// that runs ONCE PER ACTIVE LANE. With `op = lid` above, lids 0..127 map to 4 warps that each replay 32
// times while the other 7 load warps do nothing (ncu on attn_real_llama8b_dec_s16k: UTMALDG inst_executed
// 1048576 = 32x the warp-parallel count, 165 cycles per issue, the gather4 issue region alone is 29.5% of
// every instruction the kernel executes, 3.12 TB/s against a measured 6.14 TB/s DRAM roof).
//
// Instead give each load WARP a contiguous slice of the WTN/4 key-groups and let its LANE 0 issue that
// slice's K and V boxes: ~12 issues per warp, no replay, all warps working. The warp also stages the page
// rows for its OWN keys with its own lanes + __syncwarp(), so the per-tile CTA-wide `bar.sync 3` over 352
// threads (the single hottest PC, 32.6% of stall samples, `barrier` 8.42 cycles/issue vs FA4's 2.61)
// disappears. Exactness of the transaction accounting is kept by giving `bar` an arrive count of NW and
// having every warp expect_tx only its own bytes: the phase can only flip once the last warp has arrived,
// and that arrive carries its expect atomically, so tx cannot transiently reach 0 early.
template<class T,int HD,int HDV>
__device__ __forceinline__ void gather4_kv_warp(const CUtensorMap* mK,const CUtensorMap* mV,
        T* sKs, T* sVs, int* prowS, int hk, int rhd, int rhdv, uint32_t bar,
        int lw, int NW, int lane, int key0, int seqlen_k,
        const int* ptrow, int block_size, int paged, int k0){
    constexpr int KCB=HD/64, VCB=HDV/64, NG=WTN/4;
    const int g0=(NG*lw)/NW, g1=(NG*(lw+1))/NW, ng=g1-g0;
    if(ng<=0){ if(lane==0) mbar_arrive(bar); return; }        // no keys: still owes its arrival
    for(int j=lane; j<4*ng; j+=32){ int r=4*g0+j, kk=key0+r;
        prowS[r] = (kk<seqlen_k) ? (paged ? (ptrow[kk/block_size]*block_size + kk%block_size) : (k0+kk)) : -1; }
    __syncwarp();
    if(lane==0){
        mbar_expect(bar, ng*(KCB+VCB)*512);                   // 4 rows x 64 cols x 2B per box
        for(int g=g0; g<g1; ++g){
            int y0=prowS[4*g],y1=prowS[4*g+1],y2=prowS[4*g+2],y3=prowS[4*g+3];
            y0=y0>=0?y0:0; y1=y1>=0?y1:0; y2=y2>=0?y2:0; y3=y3>=0?y3:0;
            for(int cb=0;cb<KCB;++cb) tma_gather4(sKs+cb*8192+4*g*64, mK, hk*rhd +cb*64, y0,y1,y2,y3, bar);
            for(int cb=0;cb<VCB;++cb) tma_gather4(sVs+cb*8192+4*g*64, mV, hk*rhdv+cb*64, y0,y1,y2,y3, bar);
        }
    }
}

// PAGE-ALIGNED TMA paged K/V load. The WTN-key tile is covered by `nseg` physical-page segments of
// `segrows` contiguous rows each (segrows = min(block_size, WTN), a multiple of 8); `pseg[s]` is the
// physical pool row where segment s starts. One cp.async.bulk.tensor.2d per (segment, 64-col block) per
// tensor => nseg*(HD/64 + HDV/64) issues per tile (8 for hd128 @ page_size 64) against ~4096 cp.async or
// 128 gather4. Byte count is identical to the other loaders (WTN*HD*2 + WTN*HDV*2), so the caller's
// expect_tx is unchanged. Destination offsets reproduce koff/voff exactly because segrows%8==0 keeps each
// box on a 1024B swizzle atom.
template<class T,int HD,int HDV>
__device__ __forceinline__ void tma_pages_kv(const CUtensorMap* mK,const CUtensorMap* mV,
        T* sKs, T* sVs, const int* pseg, int nseg, int segrows, int hk, int rhd, int rhdv,
        uint32_t bar, int lid, int LN){
    constexpr int KCB=HD/64, VCB=HDV/64;
    const int nk=nseg*KCB, ntot=nk+nseg*VCB;
    for(int op=lid; op<ntot; op+=LN){
        bool isK=op<nk; int o=isK?op:op-nk; int cb=o/nseg, s=o-cb*nseg;
        int y=pseg[s], dof=cb*8192 + s*segrows*64;
        if(isK) tma_box(sKs+dof, mK, hk*rhd +cb*64, y, bar);
        else    tma_box(sVs+dof, mV, hk*rhdv+cb*64, y, bar);
    }
}


// FULL-WIDTH TMA for a page_size<8 tile whose WTN physical rows happen to be CONSECUTIVE. A gather4 box
// is [64,1], so it moves 4 rows = 512 B per issue and a 128-key hd128 tile costs 128 issues; a [64,WTN] box
// moves 16 KB per issue and the same tile costs KCB+VCB = 4. That transaction size is the entire decode gap:
// on identical shapes, llama8b_dec_b16_s16k (page_size 64 -> 8 multi-row boxes/tile) runs 1.03x of FA4 while
// attn_real_llama8b_dec_s16k (page_size 1 -> 128 gather4/tile) runs 0.64x, same bytes, same everything else
// (ncu: DRAM 50.8% of the measured 6.14 TB/s roof, zero over-fetch, ~165 cycles per gather4 issue -> the TMA
// engine's in-flight descriptor budget, not bandwidth, is the limit).
//
// The run is a property of the INPUT, so it is detected at run time per CTA (see `s_contig`) and only used
// for tiles fully inside seqlen_k; a fragmented page table keeps the gather4 producer unchanged. One box per
// warp keeps every issue warp-uniform, and each load warp arrives on `bar` so the NW-count barrier is exact.
template<class T,int HD,int HDV>
__device__ __forceinline__ void tma_full_kv_warp(const CUtensorMap* mK,const CUtensorMap* mV,
        T* sKs, T* sVs, int prow0, int hk, int rhd, int rhdv, uint32_t bar, int lw, int NW, int lane){
    constexpr int KCB=HD/64, VCB=HDV/64, NOP=KCB+VCB;
    if(lane!=0) return;
    int n=0; for(int op=lw; op<NOP; op+=NW) ++n;               // NW may be < NOP (narrow CTA)
    if(n==0){ mbar_arrive(bar); return; }
    mbar_expect(bar, n*(WTN*64*2));                           // n [64,WTN] boxes
    for(int op=lw; op<NOP; op+=NW){
        if(op<KCB) tma_box(sKs+op*8192,       mK, hk*rhd +op*64,       prow0, bar);
        else       tma_box(sVs+(op-KCB)*8192, mV, hk*rhdv+(op-KCB)*64, prow0, bar);
    }
}


// large-q paged (prefill3577) the 2 Q-tiles share each gathered K/V block, so the gather amortizes
// across 2x compute (unlike the correction-WG which just stole gather warps). Roles: softmax0 w0-3,
// softmax1 w4-7, MMA w8, gather load w9-15.
// GQA=true: the 2 Q-tiles are the rep=2 query-heads of ONE kv-head (blockIdx.y=kv head), each tile
// holding the full q_len (<=WTM) rows. This shares the K/V gather across the 2 heads AND runs 2
// softmax WGs -> hides the tmem-read latency that pins multi-row GQA decode at ~1 CTA/SM (qlen120).
template<class T, int HD, int HDV, bool GQA=false, bool UG4=false>
__global__ void __launch_bounds__(512,1) fwd_kernel_ws2_paged(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float* __restrict__ lse, const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int paged, const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, int leftpad, int nB, int nsplits,
    float* __restrict__ o_part, float* __restrict__ lse_part,
    const __grid_constant__ CUtensorMap mKg, const __grid_constant__ CUtensorMap mVg, int use_g4, int grp) {
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16;
    const int SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};   // P aliased into LOWER half of S (spill-free 2-pass softmax)
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[WST];
    { T* p=sQ+2*WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<WST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) Bars bars; __shared__ __align__(8) uint32_t s_base;
    __shared__ int prow_c[WST][WTN];   // page-row per key, precomputed once/block (breaks the page_table->K dep chain)
    __shared__ float sScale[2][2][WTM];   // corr per (tile, block-parity, row): softmax -> correction WG (w12-15)
    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y;
    const int b=blockIdx.z%nB, split_idx=blockIdx.z/nB, hk=GQA?h:h/rep;   // GQA: blockIdx.y = kv head
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
    const int k0 = paged ? 0 : cu_k[b];
    const int seqlen_k = paged ? seqused_k[b] : (cu_k[b+1]-cu_k[b]);
    // GQA: both tiles = query-tile mb of qheads (h*rep+0, h*rep+1) -> same rows, different head.
    // non-GQA: the 2 tiles are row-ranges (2mb, 2mb+1) of the same qhead h.
    int qh[2], r0[2];
    if constexpr(GQA){ qh[0]=h*rep; qh[1]=h*rep+1; r0[0]=mb*WTM; r0[1]=mb*WTM; }
    else            { qh[0]=h;      qh[1]=h;       r0[0]=(2*mb)*WTM; r0[1]=(2*mb+1)*WTM; }
    const int row0_0=r0[0], row0_1=r0[1];
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    if (row0_0>=seqlen_q) return;
    if (grp==2 && rep*seqlen_q <= WTM) return;         // mixed batch: decode-like sequences go to the packed route
    const bool a1 = GQA ? (qh[1]<hq && row0_1<seqlen_q) : (row0_1<seqlen_q);
    int nlo0,nhi0,nlo1,nhi1;
    { int qx=seqlen_k-seqlen_q+row0_0+(WTM-1), qn=seqlen_k-seqlen_q+row0_0; nlo0=0; nhi0=(seqlen_k+WTN-1)/WTN;
      if(is_causal) nhi0=min(nhi0,qx/WTN+1); if(win_r>=0) nhi0=min(nhi0,(qx+win_r)/WTN+1);
      if(win_l>=0){ int l=qn-win_l; nlo0=(l>0)?l/WTN:0; } nhi0=max(nhi0,nlo0); }
    if(a1){ int qx=seqlen_k-seqlen_q+row0_1+(WTM-1), qn=seqlen_k-seqlen_q+row0_1; nlo1=0; nhi1=(seqlen_k+WTN-1)/WTN;
      if(is_causal) nhi1=min(nhi1,qx/WTN+1); if(win_r>=0) nhi1=min(nhi1,(qx+win_r)/WTN+1);
      if(win_l>=0){ int l=qn-win_l; nlo1=(l>0)?l/WTN:0; } nhi1=max(nhi1,nlo1); }
    else { nlo1=nlo0; nhi1=nhi0; }
    int n_lo=min(nlo0,nlo1), n_hi=max(nhi0,nhi1);
    if (nsplits>1){ int on_lo=n_lo,on_hi=n_hi,tot=on_hi-on_lo,per=(tot+nsplits-1)/nsplits;
        n_lo=min(on_hi,on_lo+split_idx*per); n_hi=min(on_hi,on_lo+(split_idx+1)*per); }
    const int NBk=n_hi-n_lo;

    if (warp<8) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n" ::: "memory");
    else        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;\n" ::: "memory");
    const float qsc=(softcap>0.f)?scale:scale*LOG2E;   // GQA: pre-scale Q into the softmax (log2) domain -> 1-pass
    for(int i=tid;i<2*WTM*HD;i+=512){ int r=i/HD,c=i%HD; int t=r>>7, lr=r&127; int qr=r0[t]+lr;
        bool okq=(qr<seqlen_q && c<rhd && (!GQA || qh[t]<hq));
        T qv = okq ? q[(size_t)(q0+qr)*hq*rhd+(size_t)qh[t]*rhd+c] : from_f<T>(0.f);
        if constexpr(GQA) if(okq) qv = from_f<T>(to_f<T>(qv)*qsc);
        sQ[t*WTM*HD + koff(lr,c)] = qv; }
    if(warp==8){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.p_full[j]),128); mbar_init(smem_u32(&bars.o_ready[j]),1);
                                      mbar_init(smem_u32(&bars.corr_stats[j]),128); mbar_init(smem_u32(&bars.o_resc[j]),128);} }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
    uint32_t OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();

    if (warp>=9) {                       // LOAD: 7 warps (w9-15) manual paged gather (shared by both tiles)
        const int lid=tid-288, LN=512-288; int pe[WST]={0};
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(KVE[s],pe[s]); pe[s]^=1;
            if constexpr(UG4){ if(lid==0) mbar_expect(KVF[s], WTN*HD*2 + WTN*HDV*2); }   // arrive+set tx before the barrier -> ordered ahead of every gather4
            int key0=(n_lo+i)*WTN;
            for(int r=lid;r<WTN;r+=LN){ int kk=key0+r;
                prow_c[s][r] = (kk<seqlen_k) ? (paged ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : (k0+kk)) : -1; }
            asm volatile("bar.sync 3, %0;"::"r"(LN));   // prow_c[s] ready (+ g4 expect_tx visible)
          if constexpr(UG4){
            // HW-async TMA tile::gather4 paged load (shared by both Q-tiles). Frees the 7 load warps'
            // issue slots -> hides the long_scoreboard gather latency that pins prefill3577 at 1 CTA/SM.
            tma_gather4_kv<T,HD,HDV>(&mKg,&mVg, sK[s], sV[s], prow_c[s], hk, rhd, rhdv, KVF[s], lid, LN);
          } else {
            // cp.async gather: koff(r,c..c+7)/voff contiguous & 16B-aligned -> fire all 16B (uint4=8xbf16)
            // global->smem async copies (pipelined, no per-load L1TEX scoreboard stall) then ONE wait_group.
            // Cuts the ~53% long-scoreboard stall the serial load->scatter incurred on the paged gather.
            for(int idx=lid;idx<WTN*(HD/8);idx+=LN){ int r=idx/(HD/8), c=(idx%(HD/8))*8; int prow=prow_c[s][r];
                if(prow>=0 && c<rhd) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sK[s][koff(r,c)])),"l"(&k[(size_t)prow*hkv*rhd+(size_t)hk*rhd+c]));
                else *(uint4*)&sK[s][koff(r,c)]=make_uint4(0,0,0,0); }
            for(int idx=lid;idx<WTN*(HDV/8);idx+=LN){ int key=idx/(HDV/8), dv=(idx%(HDV/8))*8; int prow=prow_c[s][key];
                if(prow>=0 && dv<rhdv) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sV[s][voff(key,dv)])),"l"(&v[(size_t)prow*hkv*rhdv+(size_t)hk*rhdv+dv]));
                else *(uint4*)&sV[s][voff(key,dv)]=make_uint4(0,0,0,0); }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n");   // one wait for the whole block's copies (they pipelined)
            asm volatile("fence.proxy.async.shared::cta;");
            asm volatile("bar.sync 3, %0;"::"r"(LN));
            if(lid==0) mbar_arrive(KVF[s]);
          }
        }
    } else if (warp==8) {                // MMA
        int pf[WST]={0}, ppf[2]={0,0};
        auto qk=[&](int j,int i){ int s=i%WST; uint64_t qd0=make_desc(smem_u32(sQ+j*WTM*HD),16,1024,2),kd0=make_desc(smem_u32(sK[s]),16,1024,2);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SCOL[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); }
            mma_commit(SF[j]); };
        auto pv=[&](int j,int i){ int s=i%WST; uint64_t vd0=make_desc(smem_u32(sV[s]),16384,1024,2);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[j]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+OCOL[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); }
            mma_commit(OR[j]); };
        if(NBk>0){
        mbar_wait(KVF[0],pf[0]); pf[0]^=1; qk(0,0); qk(1,0);
        for(int i=0;i<NBk;++i){ int s=i%WST;
            mbar_wait(PF[0],ppf[0]); ppf[0]^=1; pv(0,i);
            mbar_wait(PF[1],ppf[1]); ppf[1]^=1; pv(1,i);
            mma_commit(KVE[s]);
            if(i+1<NBk){ int ns=(i+1)%WST; mbar_wait(KVF[ns],pf[ns]); pf[ns]^=1; qk(0,i+1); qk(1,i+1); }
        } }
    } else if (warp<8) {                 // softmax WG (tile j)
        const int j=warp>>2, lr=tid&127;
        const int qhj=qh[j];             // output qhead for this tile (GQA: h*rep+j, else h)
        int psf=0, por=0;
        float m_i=-INFINITY, l_i=0.f;
        if(tid==0){ for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); }
        const int row0j=r0[j];
        const int qp=seqlen_k-seqlen_q+row0j+lr;
        int key0off=0;
        // GQA: sv is Q-prescaled (log2 domain); non-GQA: sv raw (apply scale here).
        auto logit=[&](float sv,int c)->float{ float lg;
            if constexpr(GQA) lg=(softcap>0.f)?(softcap*tanhf(sv*inv_sc)*LOG2E):sv;
            else { lg=scale*sv; if(softcap>0.f) lg=softcap*tanhf(lg*inv_sc); }
            int kk=key0off+c; bool ok=(kk<seqlen_k)&&(kk>=leftpad);
            if(is_causal&&kk>qp) ok=false; if(win_l>=0&&kk<qp-win_l) ok=false; if(win_r>=0&&kk>qp+win_r) ok=false; return ok?lg:-INFINITY; };
        for(int i=0;i<NBk;++i){ key0off=(n_lo+i)*WTN;
            mbar_wait(SF[j],psf); psf^=1; tmem_fence();
            // FULL vs BOUNDARY split (FA4 design): interior blocks need no per-key mask -> branch-free
            // scale-only path, saving ~6 ALU ops/element on the vast majority of blocks.
            const int qbase=seqlen_k-seqlen_q+row0j, kend=key0off+WTN-1;
            bool full=(kend<seqlen_k) && !(softcap>0.f) && key0off>=leftpad;
            if(is_causal && kend>qbase) full=false;
            if(win_r>=0 && kend>qbase+win_r) full=false;
            if(win_l>=0 && key0off < qbase+(WTM-1)-win_l) full=false;
            float m_new, corr, ps=0.f;
          if constexpr(GQA){   // 1-PASS fragment (Q prescaled -> log2 domain); P packed into SCOL[j] alias
            if(full){
                uint32_t f[128];
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+0,  f, 0);
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+32, f, 32);
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+64, f, 64);
                LD32_INTO(base+SCOL[j]+((uint32_t)lr<<16)+96, f, 96);
                tmem_wait_ld();
                float rmax=-INFINITY;
                #pragma unroll
                for(int t=0;t<128;++t) rmax=fmaxf(rmax,__uint_as_float(f[t]));
                m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                #pragma unroll
                for(int g=0;g<4;++g){ uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ float p0=ex2f(__uint_as_float(f[g*32+2*t])-m_new), p1=ex2f(__uint_as_float(f[g*32+2*t+1])-m_new); ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(base+PCOL[j]+((uint32_t)lr<<16)+g*16, pk); }
            } else {
                float rmax=-INFINITY;
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[j]+((uint32_t)lr<<16)+g*32, v);
                    #pragma unroll
                    for(int t=0;t<32;++t) rmax=fmaxf(rmax,logit(__uint_as_float(v[t]),g*32+t)); }
                tmem_wait_ld();
                m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[j]+((uint32_t)lr<<16)+g*32, v); uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ float l0=logit(__uint_as_float(v[2*t]),g*32+2*t), l1=logit(__uint_as_float(v[2*t+1]),g*32+2*t+1);
                        float p0=(l0==-INFINITY)?0.f:ex2f(l0-m_new), p1=(l1==-INFINITY)?0.f:ex2f(l1-m_new); ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(base+PCOL[j]+((uint32_t)lr<<16)+g*16, pk); }
            }
            if(i>0){ mbar_wait(OR[j],por); por^=1;
                if(!__all_sync(0xffffffffu,corr==1.0f)) tmem_o_rescale<HDV>(base+OCOL[j]+((uint32_t)lr<<16), corr); }
            l_i=l_i*corr+ps; m_i=m_new;
            tmem_wait_st(); mbar_arrive(PF[j]);
          } else {
            // pass 1: block max via 4x .x32 reads (32 cols/instr, consumed immediately -> no spill)
            float rmax=-INFINITY;
            if(full){
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[j]+((uint32_t)lr<<16)+g*32, v);
                    #pragma unroll
                    for(int t=0;t<32;++t){ float sv; memcpy(&sv,&v[t],4); rmax=fmaxf(rmax,scale*sv); } }
            } else {
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[j]+((uint32_t)lr<<16)+g*32, v);
                    #pragma unroll
                    for(int t=0;t<32;++t){ float sv; memcpy(&sv,&v[t],4); rmax=fmaxf(rmax,logit(sv,g*32+t)); } }
            }
            tmem_wait_ld();
            m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f((m_i-m_new)*LOG2E);
            // O-rescale (OCOL, independent of S) before P overwrites S's lower half
            if(i>0){ mbar_wait(OR[j],por); por^=1;
                if(!__all_sync(0xffffffffu,corr==1.0f)) for(int d=0;d<HDV;++d){ uint32_t a=base+OCOL[j]+((uint32_t)lr<<16)+d; uint32_t w=tmem_ld(a); float o; memcpy(&o,&w,4); o*=corr; uint32_t ob; memcpy(&ob,&o,4); tmem_st(a,ob);} }
            // pass 2: re-read 32-col blocks, exp, write 16 P-cols/block at SCOL+16g.. (block-of-32 keeps alias-safety)
            if(full){
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[j]+((uint32_t)lr<<16)+g*32, v);
                    #pragma unroll
                    for(int t=0;t<16;++t){ float s0,s1; memcpy(&s0,&v[2*t],4); memcpy(&s1,&v[2*t+1],4);
                        float p0=ex2f((scale*s0-m_new)*LOG2E), p1=ex2f((scale*s1-m_new)*LOG2E); ps+=p0+p1;
                        tmem_st(base+PCOL[j]+((uint32_t)lr<<16)+g*16+t, pack16<T>(p0,p1)); } }
            } else {
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[j]+((uint32_t)lr<<16)+g*32, v);
                    #pragma unroll
                    for(int t=0;t<16;++t){ float s0,s1; memcpy(&s0,&v[2*t],4); memcpy(&s1,&v[2*t+1],4);
                        float l0=logit(s0,g*32+2*t), l1=logit(s1,g*32+2*t+1);
                        float p0=(l0==-INFINITY)?0.f:ex2f((l0-m_new)*LOG2E), p1=(l1==-INFINITY)?0.f:ex2f((l1-m_new)*LOG2E); ps+=p0+p1;
                        tmem_st(base+PCOL[j]+((uint32_t)lr<<16)+g*16+t, pack16<T>(p0,p1)); } }
            }
            l_i=l_i*corr+ps; m_i=m_new;
            tmem_wait_st(); mbar_arrive(PF[j]);   // all 128 arrive (no bar.sync -> warps proceed independently, cuts barrier stall)
          }
        }
        if(NBk>0){ mbar_wait(OR[j],por); por^=1; } tmem_fence();
        const int qr=row0j+lr;
        const bool ow=(qr<seqlen_q) && (!GQA || qhj<hq);
        // GQA: m_i is log2-domain (Q prescaled) -> LSE needs *ln2; non-GQA: m_i natural.
        const float M2N = GQA ? 0.6931471805599453f : 1.0f;
        if (nsplits>1) {
            const int TQ=cu_q[nB]; float invp=(l_i>0.f)?1.f/l_i:0.f;
            for(int d=0;d<HDV;++d){ float o=0.f;
                if(NBk>0){ uint32_t w=tmem_ld(base+OCOL[j]+((uint32_t)lr<<16)+d); tmem_wait_ld(); memcpy(&o,&w,4); }
                if(ow && d<rhdv) o_part[(((size_t)split_idx*TQ+(q0+qr))*hq+qhj)*rhdv+d]=o*invp; }
            if(ow) lse_part[((size_t)split_idx*hq+qhj)*TQ+(q0+qr)] = (l_i>0.f)?(logf(l_i)+m_i*M2N):-INFINITY;
        } else {
            float mfin=m_i, lfin=l_i, osc=1.f;
            if(s_aux!=nullptr){ float sk=GQA?(s_aux[qhj]*LOG2E):s_aux[qhj]; float mn=fmaxf(m_i,sk);
                float c2=(m_i==-INFINITY)?0.f:(GQA?ex2f(m_i-mn):ex2f((m_i-mn)*LOG2E));
                lfin=l_i*c2+(GQA?ex2f(sk-mn):ex2f((sk-mn)*LOG2E)); osc=c2; mfin=mn; }
            float inv=(lfin>0.f)?osc/lfin:0.f;
            for(int d=0;d<HDV;++d){ float o=0.f;
                if(NBk>0){ uint32_t w=tmem_ld(base+OCOL[j]+((uint32_t)lr<<16)+d); tmem_wait_ld(); memcpy(&o,&w,4); }
                if(ow && d<rhdv) out[(size_t)(q0+qr)*hq*rhdv+(size_t)qhj*rhdv+d]=from_f<T>(o*inv); }
            if(ow) lse[(size_t)qhj*cu_q[nB]+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin*M2N):-INFINITY;
        }
    }
    __syncthreads();
    if(warp==8) tmem_dealloc(base,512);
}

// ================= FA4-CHOREOGRAPHY PAGED kernel (fwd_kernel_fa4_paged) ==========================
// The fwd_kernel_fa4 intra-CTA choreography (S0/S1/O0/O1 512-col tmem, softmax warp-spec w0-7,
// dedicated CORRECTION WG w8-11, MMA w12, depth-2 C-channel, ord ping-pong, FA4KVS K/V ring) but:
//   (a) NON-PERSISTENT: one work-item per CTA from blockIdx (like fwd_kernel_ws2_paged); no scheduler
//       warp, no cross-tile phase carry, no tile_ctr.
//   (b) K/V PRODUCER = HW-async TMA tile::gather4 paged gather (w13-15), freeing the load warps' issue
//       slots so the deep fa4 consumer pipeline can spend them (the thesis: fa4 consumer + gather4 producer).
//   (c) Q preloaded RAW by all 512 threads (no TMA-Q producer); qsc folded into the softmax exp2.
//   (d) SplitKV (n_lo/n_hi range-split -> o_part/lse_part -> combine_kernel), page_table indexing,
//       seqused_k/leftpad, causal tile pruning, invalid-key masking -- all ported from ws2_paged.
// GQA=false: the 2 tiles are row-ranges (2mb, 2mb+1) of qhead h (large-q prefill: prefill3577).
// GQA=true : the 2 tiles are the rep=2 qheads (h*rep, h*rep+1) of kv-head h, same rows (decode qlen120).
// Both softmax halves iterate the SAME [n_lo,n_hi) block range (fa4 shares n); per-row causal/seqlen
// masking zeroes P for invalid keys, an inactive half's O is computed then discarded (ow=false).
// DECOUPLED K/V RING (KD, VD). A K slot frees once BOTH QKs of its block are issued; a V slot only after
// PV1. One shared `kv_empty` therefore held K hostage to PV and capped the pipeline at KD=VD=2 pairs =
// 192KB of the 227KB cap. K-depth 3 / V-depth 2 is 160KB of K/V + 64KB of Q = 224KB and gives the
// producer one more slot to run ahead on exactly the tensor the consumer needs first. PC sampling
// (attn_real_llama8b_pf_s16k, page_size 1): 49% of samples in the two consumer mbarrier waits, 5.2% in
// the load warp's own kv_empty wait, 1.2% on the gather4 issues -- out of slots, not out of issue slots.
struct BarsFA4pg { uint64_t kf_full[FA4KVSMAX], k_empty[FA4KVSMAX], vf_full[FA4KVSMAX], v_empty[FA4KVSMAX],
                   s_full[2], p_full[2], o_full[2], o_empty[2], c_full[4], c_empty[4],
                   q_full[1], q_empty[1], q_tma[1], q_sched[1], ord[2]; };
template<class T, int HD, int HDV, bool GQA=false, int KD=FA4KVS, int VD=FA4KVS>
__global__ void __launch_bounds__(512,1) fwd_kernel_fa4_paged(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float* __restrict__ lse, const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int paged, const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, int leftpad, int nB, int nsplits,
    float* __restrict__ o_part, float* __restrict__ lse_part,
    const __grid_constant__ CUtensorMap mKg, const __grid_constant__ CUtensorMap mVg, int prow_cap, int use_g4, int grp) {
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16, KCB=HD/64, VCB=HDV/64;
    const uint32_t SCOL[2]={0,128}, PCOL[2]={0,128}, OCOL[2]={256,384};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[KD]; T* sV[VD]; int* prowAll;
    { T* p=sQ+2*WTM*HD; for(int s=0;s<KD;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<VD;s++){ sV[s]=p; p+=WTN*HDV; } prowAll=(int*)p; }
    __shared__ __align__(16) BarsFA4pg bars; __shared__ __align__(8) uint32_t s_base;
    __shared__ float sScale[2][2][WTM];
    __shared__ float sFinSum[2][2][WTM], sFinMax[2][2][WTM];
    __shared__ int prow_c[FA4KVSMAX][WTN];            // physical page-row per key (breaks page_table dep chain)
    const int tid=threadIdx.x, warp=tid>>5;
    const float LOG2E=1.4426950408889634f, INV_LOG2E=0.6931471805599453f;
    const float qsc=scale*LOG2E;                      // Q prescale -> log2 softmax domain (no softcap here)

    // ---- non-persistent work-item geometry from blockIdx ----
    const int mb=blockIdx.x, h=blockIdx.y;
    const int b=blockIdx.z%nB, split_idx=blockIdx.z/nB, hk=GQA?h:h/rep;
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
    const int k0 = paged ? 0 : cu_k[b];
    const int seqlen_k = paged ? seqused_k[b] : (cu_k[b+1]-cu_k[b]);
    const int TQ=cu_q[nB];
    int qh[2], r0[2];
    if constexpr(GQA){ qh[0]=h*rep; qh[1]=h*rep+1; r0[0]=mb*WTM; r0[1]=mb*WTM; }
    else            { qh[0]=h;      qh[1]=h;       r0[0]=(2*mb)*WTM; r0[1]=(2*mb+1)*WTM; }
    if (r0[0]>=seqlen_q) return;                       // whole CTA idle
    if (grp==2 && rep*seqlen_q <= WTM) return;         // mixed batch: decode-like sequences go to the packed route
    const bool a1 = GQA ? (qh[1]<hq && r0[1]<seqlen_q) : (r0[1]<seqlen_q);
    // causal/window tile-range pruning + SplitKV (n shared across both halves)
    int nlo0,nhi0,nlo1,nhi1;
    { int qx=seqlen_k-seqlen_q+r0[0]+(WTM-1), qn=seqlen_k-seqlen_q+r0[0]; nlo0=0; nhi0=(seqlen_k+WTN-1)/WTN;
      if(is_causal) nhi0=min(nhi0,qx/WTN+1); if(win_r>=0) nhi0=min(nhi0,(qx+win_r)/WTN+1);
      if(win_l>=0){ int l=qn-win_l; nlo0=(l>0)?l/WTN:0; } nhi0=max(nhi0,nlo0); }
    if(a1){ int qx=seqlen_k-seqlen_q+r0[1]+(WTM-1), qn=seqlen_k-seqlen_q+r0[1]; nlo1=0; nhi1=(seqlen_k+WTN-1)/WTN;
      if(is_causal) nhi1=min(nhi1,qx/WTN+1); if(win_r>=0) nhi1=min(nhi1,(qx+win_r)/WTN+1);
      if(win_l>=0){ int l=qn-win_l; nlo1=(l>0)?l/WTN:0; } nhi1=max(nhi1,nlo1); }
    else { nlo1=nlo0; nhi1=nhi0; }
    int n_lo=min(nlo0,nlo1), n_hi=max(nhi0,nhi1);
    if (nsplits>1){ int on_lo=n_lo,on_hi=n_hi,tot=on_hi-on_lo,per=(tot+nsplits-1)/nsplits;
        n_lo=min(on_hi,on_lo+split_idx*per); n_hi=min(on_hi,on_lo+(split_idx+1)*per); }
    const int NBk=n_hi-n_lo;
    const bool useAll=(prow_cap>0 && NBk<=prow_cap);   // Lever 2b: full-range page rows resolved once (fits smem)
    // Empty split (or no blocks): write -INF partial LSE so combine_kernel skips this split, then leave.
    // NBk is CTA-uniform -> this early return is divergence-free (no tmem alloc / __syncthreads reached).
    if (NBk<=0){
        if(nsplits>1 && warp<8){ int j=warp>>2, r=tid&127;
            int qr=r0[j]+r, head=GQA?(h*rep+j):h;
            bool ow=(qr<seqlen_q)&&(!GQA||head<hq)&&((j==0)||a1);
            if(ow){ for(int d=0;d<rhdv;++d) o_part[(((size_t)split_idx*TQ+(q0+qr))*hq+head)*rhdv+d]=0.f;
                    lse_part[((size_t)split_idx*hq+head)*TQ+(q0+qr)]=-INFINITY; } }
        return;
    }

    // ---- Q preload via cp.async.cg (128-bit, overlapped with tmem alloc / barrier init); qsc folded into softmax exp2 ----
    for(int i=tid;i<2*WTM*(HD/8);i+=512){ int r=i/(HD/8),c=(i%(HD/8))*8; int t=r>>7, lr=r&127; int qr=r0[t]+lr;
        bool okq=(qr<seqlen_q && c<rhd && (!GQA || qh[t]<hq));
        if(okq) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sQ[t*WTM*HD + koff(lr,c)])),"l"(&q[(size_t)(q0+qr)*hq*rhd+(size_t)qh[t]*rhd+c]));
        else *(uint4*)&sQ[t*WTM*HD + koff(lr,c)]=make_uint4(0,0,0,0); }
    asm volatile("cp.async.commit_group;\n");
    // ---- Lever 2b: resolve the WHOLE [n_lo,n_hi) block range's page rows ONCE (off the per-block critical path) ----
    if(useAll && use_g4!=2){ for(int idx=tid; idx<NBk*WTN; idx+=512){ int i=idx/WTN, r=idx%WTN; int kk=(n_lo+i)*WTN+r;
        prowAll[idx] = (kk<seqlen_k && kk>=leftpad) ? (paged ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : (k0+kk)) : -1; } }

    if(warp>=8) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");  // producers dec FIRST (fa4 order) so softmax's inc 192 finds a filled reg pool -> no deadlock
    if(warp==12){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int s=0;s<KD;s++){ mbar_init(smem_u32(&bars.kf_full[s]),1); mbar_init(smem_u32(&bars.k_empty[s]),1); }
                for(int s=0;s<VD;s++){ mbar_init(smem_u32(&bars.vf_full[s]),1); mbar_init(smem_u32(&bars.v_empty[s]),1); }
        for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.p_full[j]),128);
            mbar_init(smem_u32(&bars.o_full[j]),1); mbar_init(smem_u32(&bars.o_empty[j]),128);
            mbar_init(smem_u32(&bars.c_full[2*j]),128); mbar_init(smem_u32(&bars.c_full[2*j+1]),128); mbar_init(smem_u32(&bars.c_empty[2*j]),128); mbar_init(smem_u32(&bars.c_empty[2*j+1]),128); }
        mbar_init(smem_u32(&bars.ord[0]),128); mbar_init(smem_u32(&bars.ord[1]),128); }
    asm volatile("cp.async.wait_group 0;\n");    // Q cp.async landed before the MMA reads sQ
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t base=s_base;
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();

    if (warp>=13) {                      // -------- LOAD (w13-15, 96 threads): cp.async gather (default) or TMA gather4 (FA4_NOG4=1) --------
        const int lid=tid-416, LN=96;
        uint32_t KF[KD],KE[KD],VF[VD],VE[VD];
        for(int s=0;s<KD;s++){ KF[s]=smem_u32(&bars.kf_full[s]); KE[s]=smem_u32(&bars.k_empty[s]); }
        for(int s=0;s<VD;s++){ VF[s]=smem_u32(&bars.vf_full[s]); VE[s]=smem_u32(&bars.v_empty[s]); }
        int pek[KD]; for(int s=0;s<KD;s++) pek[s]=1;         // primed: the first KD/VD waits pass immediately
        int pev[VD]; for(int s=0;s<VD;s++) pev[s]=1;
      if(use_g4==2){
        // ---- PAGE-TMA producer: the WTN-key tile is nseg contiguous physical-page segments, so K+V is
        // nseg*(KCB+VCB) cp.async.bulk.tensor.2d boxes (8 for hd128 @ page 64) instead of ~4096 cp.async
        // issues or 128 gather4 issues. Keeps the K-then-V barrier split (QK waits K only).
        const int segrows = block_size<WTN ? block_size : WTN, nseg = WTN/segrows;
        for(int i=0;i<NBk;++i){ int sk=i%KD, sv=i%VD;
            const int key0=(n_lo+i)*WTN;
            const uint32_t kf=rsel(KF,sk), vf=rsel(VF,sv); T* pK=rsel(sK,sk); T* pV=rsel(sV,sv);
            mbar_wait(rsel(KE,sk), rflip(pek,sk));          // K first: QK is the consumer that waits on it
            for(int t=lid;t<nseg;t+=LN){ int kk=key0+t*segrows;
                prow_c[sk][t] = (kk<seqlen_k) ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : 0; }
            if(lid==0) mbar_expect(kf, WTN*HD*2);
            asm volatile("bar.sync 8, %0;"::"r"(LN));       // pseg ready + expect_tx ordered ahead of the boxes
            for(int op=lid; op<nseg*KCB; op+=LN){ int cb=op/nseg, t=op-cb*nseg;
                tma_box(pK+cb*8192+t*segrows*64, &mKg, hk*rhd +cb*64, prow_c[sk][t], kf); }
            mbar_wait(rsel(VE,sv), rflip(pev,sv));
            if(lid==0) mbar_expect(vf, WTN*HDV*2);
            asm volatile("bar.sync 8, %0;"::"r"(LN));
            for(int op=lid; op<nseg*VCB; op+=LN){ int cb=op/nseg, t=op-cb*nseg;
                tma_box(pV+cb*8192+t*segrows*64, &mVg, hk*rhdv+cb*64, prow_c[sk][t], vf); }
        }
      } else if(use_g4){
        // ---- HW-async TMA tile::gather4 into the koff/voff swizzled smem, with the page-row resolve
        // SOFTWARE-PIPELINED ONE BLOCK AHEAD. The resolve is WTN dependent global page_table loads plus a
        // barrier every load thread waits on, and it used to sit immediately before the gather it feeds --
        // fully exposed once per K block. NCU on attn_real_llama8b_pf_s16k (page_size 1): long_scoreboard
        // 9.21/issue vs FA4's 1.95 and tensor pipe 31.7% vs 71.6%, with DRAM under 2% on BOTH -- pure
        // latency, not bandwidth. Resolving block i+1 AFTER issuing block i's gathers hides it behind the
        // next kv_empty wait. (FA4's PagedKVManager gets this for free by holding the page rows in
        // registers and broadcasting them with a warp shuffle instead of staging them through smem.)
        constexpr int NG=WTN/4, NGK=NG*KCB, NGT=NGK+NG*VCB;
        auto resolve=[&](int ii){ const int key0=(n_lo+ii)*WTN; const int st=ii%KD;
            for(int r=lid;r<WTN;r+=LN){ int kk=key0+r;
                prow_c[st][r] = (kk<seqlen_k && kk>=leftpad) ? (paged ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : (k0+kk)) : -1; } };
        if(!useAll && NBk>0){ resolve(0); asm volatile("bar.sync 8, %0;"::"r"(LN)); }
        for(int i=0;i<NBk;++i){ int sk=i%KD, sv=i%VD;
            const int* prow = useAll ? (prowAll+i*WTN) : prow_c[sk];
            const uint32_t kf=rsel(KF,sk), vf=rsel(VF,sv); T* pK=rsel(sK,sk); T* pV=rsel(sV,sv);
            mbar_wait(rsel(KE,sk), rflip(pek,sk));          // K first: QK is the only consumer that waits on it
            if(lid==0) mbar_expect(kf, WTN*HD*2);
            asm volatile("bar.sync 8, %0;"::"r"(LN));       // expect_tx ordered before every gather4
            for(int op=lid; op<NGK; op+=LN){ int cb=op/NG, g=op-cb*NG;
                int y0=prow[4*g],y1=prow[4*g+1],y2=prow[4*g+2],y3=prow[4*g+3];
                y0=y0>=0?y0:0; y1=y1>=0?y1:0; y2=y2>=0?y2:0; y3=y3>=0?y3:0;
                tma_gather4(pK+cb*8192+4*g*64, &mKg, hk*rhd +cb*64, y0,y1,y2,y3, kf); }
            mbar_wait(rsel(VE,sv), rflip(pev,sv));
            if(lid==0) mbar_expect(vf, WTN*HDV*2);
            asm volatile("bar.sync 8, %0;"::"r"(LN));
            for(int op=lid; op<NGT-NGK; op+=LN){ int cb=op/NG, g=op-cb*NG;
                int y0=prow[4*g],y1=prow[4*g+1],y2=prow[4*g+2],y3=prow[4*g+3];
                y0=y0>=0?y0:0; y1=y1>=0?y1:0; y2=y2>=0?y2:0; y3=y3>=0?y3:0;
                tma_gather4(pV+cb*8192+4*g*64, &mVg, hk*rhdv+cb*64, y0,y1,y2,y3, vf); }
            // Block i+1's page rows: issued AFTER i's gathers so the dependent page_table loads overlap
            // them, landing in the OTHER ring slot (FA4KVS>=2) -- never the one i is still gathering from.
            if(!useAll && i+1<NBk){ resolve(i+1); asm volatile("bar.sync 8, %0;"::"r"(LN)); }
        }
      } else {
        // ---- NEW producer (default): FA4-style coalesced 128-bit cp.async.cg gather into the SAME koff/voff
        // swizzled smem the MMA descriptors read. Higher memory-level parallelism than serialized gather4
        // transactions. K signaled on kf_full[s] FIRST (QK waits K only), V on vf_full[s] -> K-then-V split kept.
        for(int i=0;i<NBk;++i){ int sk=i%KD, sv=i%VD, s=sk;
            mbar_wait(rsel(KE,sk), rflip(pek,sk));
            const uint32_t kf=rsel(KF,sk), vf=rsel(VF,sv); T* pK=rsel(sK,sk); T* pV=rsel(sV,sv);
            const int* prow;
            if(useAll){ prow=prowAll+i*WTN; }
            else { const int key0=(n_lo+i)*WTN;
                for(int r=lid;r<WTN;r+=LN){ int kk=key0+r;
                    prow_c[s][r] = (kk<seqlen_k && kk>=leftpad) ? (paged ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : (k0+kk)) : -1; }
                asm volatile("bar.sync 8, %0;"::"r"(LN)); prow=prow_c[s]; }
            // K rows -> sK[s] koff swizzle (koff(r,c..c+7) are 8 contiguous 16B-aligned elems)
            for(int idx=lid; idx<WTN*(HD/8); idx+=LN){ int r=idx/(HD/8), c=(idx%(HD/8))*8; int prw=prow[r];
                if(prw>=0 && c<rhd) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&pK[koff(r,c)])),"l"(&k[(size_t)prw*hkv*rhd+(size_t)hk*rhd+c]));
                else *(uint4*)&pK[koff(r,c)]=make_uint4(0,0,0,0); }
            asm volatile("cp.async.commit_group;\n");
            mbar_wait(rsel(VE,sv), rflip(pev,sv));
            // V rows -> the V-ring slot's voff swizzle
            for(int idx=lid; idx<WTN*(HDV/8); idx+=LN){ int key=idx/(HDV/8), dv=(idx%(HDV/8))*8; int prw=prow[key];
                if(prw>=0 && dv<rhdv) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&pV[voff(key,dv)])),"l"(&v[(size_t)prw*hkv*rhdv+(size_t)hk*rhdv+dv]));
                else *(uint4*)&pV[voff(key,dv)]=make_uint4(0,0,0,0); }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");           // K group complete (V group still pending)
            asm volatile("fence.proxy.async.shared::cta;");
            asm volatile("bar.sync 8, %0;"::"r"(LN));           // all load threads' K copies globally done
            if(lid==0) mbar_arrive(kf);                      // -> QK MMA may start on K
            asm volatile("cp.async.wait_group 0;\n");           // V group complete
            asm volatile("fence.proxy.async.shared::cta;");
            asm volatile("bar.sync 8, %0;"::"r"(LN));           // all load threads' V copies globally done
            if(lid==0) mbar_arrive(vf);                      // -> PV MMA may proceed on V
        }
      }
    } else if (warp<8) {                 // -------- SOFTMAX half j = warp>>2 --------
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n" ::: "memory");
        const int j=warp>>2, lr=tid&127;
        const uint32_t sf=smem_u32(&bars.s_full[j]), pf=smem_u32(&bars.p_full[j]);
        const uint32_t cf0=smem_u32(&bars.c_full[2*j]), cf1=smem_u32(&bars.c_full[2*j+1]);
        const uint32_t ce0=smem_u32(&bars.c_empty[2*j]), ce1=smem_u32(&bars.c_empty[2*j+1]);
        const uint32_t sAddr=base+SCOL[j]+((uint32_t)lr<<16);
        const uint32_t pAddr=base+PCOL[j]+((uint32_t)lr<<16);
        const uint32_t ordw=smem_u32(&bars.ord[j]), orda=smem_u32(&bars.ord[j^1]);
        int psf=0, ordph=(j==0)?1:0;
        int pce0=1, pce1=1;                             // depth-2 free-pass BOTH C slots -> lead correction by 2
        const int row0j=r0[j];
        const int qp=seqlen_k-seqlen_q+row0j+lr;
        const int qbase=seqlen_k-seqlen_q+row0j;
        float m_i=-INFINITY, l_i=0.f;
        for(int i=0;i<NBk;++i){
            int sl=i&1;
            if(sl==0){ mbar_wait(ce0, pce0); pce0^=1; } else { mbar_wait(ce1, pce1); pce1^=1; }
            const uint32_t cf=(sl==0)?cf0:cf1;
            mbar_wait(sf, psf); psf^=1; tmem_fence();
            mbar_wait(ordw, ordph); ordph^=1;
            const int key0=(n_lo+i)*WTN, kend=key0+WTN-1;
            bool full=(kend<seqlen_k) && key0>=leftpad && !(is_causal && kend>qbase);
            float m_new, ps=0.f;
            if(full){
                softmax_full_ptx_ordcs<T>(sAddr, pAddr, m_i, m_new, ps, orda, smem_u32(&sScale[j][sl][lr]), cf, qsc);
            } else {
                int chi=seqlen_k-1-key0; if(chi>WTN-1) chi=WTN-1;
                if(is_causal){ int cc=qp-key0; if(cc<chi) chi=cc; }
                int clo=leftpad-key0; if(clo<0) clo=0;
                float rmax=-INFINITY;
                #pragma unroll 1
                for(int g=0; g<128; g+=32){ uint32_t ch[32]; tmem_ld32(sAddr+(uint32_t)g, ch); tmem_wait_ld();
                    #pragma unroll
                    for(int c=0;c<32;++c){ int gc=g+c; if(gc>=clo && gc<=chi) rmax=fmaxf(rmax,__uint_as_float(ch[c])); } }
                rmax*=qsc;
                mbar_arrive(orda);
                m_new=fmaxf(m_i,rmax);
                float cb=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                sScale[j][sl][lr]=cb; mbar_arrive(cf);
                bool ok=(m_new!=-INFINITY);
                #pragma unroll 1
                for(int g=0; g<128; g+=32){ uint32_t ch[32]; tmem_ld32(sAddr+(uint32_t)g, ch); tmem_wait_ld();
                    uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ int c0=g+2*t,c1=c0+1; float s0=__uint_as_float(ch[2*t])*qsc, s1=__uint_as_float(ch[2*t+1])*qsc;
                        float p0=(ok&&c0>=clo&&c0<=chi)?ex2f(s0-m_new):0.f, p1=(ok&&c1>=clo&&c1<=chi)?ex2f(s1-m_new):0.f; ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(pAddr+(uint32_t)(g/2), pk); }
            }
            float corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
            tmem_wait_st(); mbar_arrive(pf);
            l_i=l_i*corr+ps; m_i=m_new;
        }
        mbar_wait(sf, psf); psf^=1;                     // terminal SF wait (pairs MMA final dummy S commit)
        int fsl=NBk&1;
        if(fsl==0){ mbar_wait(ce0, pce0); pce0^=1; } else { mbar_wait(ce1, pce1); pce1^=1; }
        sFinSum[j][fsl][lr]=l_i; sFinMax[j][fsl][lr]=m_i;
        mbar_arrive((fsl==0)?cf0:cf1);
    } else if (warp==12) {               // -------- MMA warp --------
        uint32_t KF[KD],KE[KD],VF[VD],VE[VD];
        for(int s=0;s<KD;s++){ KF[s]=smem_u32(&bars.kf_full[s]); KE[s]=smem_u32(&bars.k_empty[s]); }
        for(int s=0;s<VD;s++){ VF[s]=smem_u32(&bars.vf_full[s]); VE[s]=smem_u32(&bars.v_empty[s]); }
        const uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])};
        const uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])};
        const uint32_t OF[2]={smem_u32(&bars.o_full[0]),smem_u32(&bars.o_full[1])};
        const uint32_t OE[2]={smem_u32(&bars.o_empty[0]),smem_u32(&bars.o_empty[1])};
        const bool lead=(tid&31)==0;
        uint64_t qd0a[2], kd0a[KD], vd0a[VD];
        #pragma unroll
        for(int jj=0;jj<2;jj++) qd0a[jj]=make_desc(smem_u32(sQ+jj*WTM*HD),16,1024,2);
        #pragma unroll
        for(int s=0;s<KD;s++) kd0a[s]=make_desc(smem_u32(sK[s]),16,1024,2);
        #pragma unroll
        for(int s=0;s<VD;s++) vd0a[s]=make_desc(smem_u32(sV[s]),16384,1024,2);
        auto qk=[&](int jj,int s){ if(lead){ uint64_t qd0=qd0a[jj],kd0=rsel(kd0a,s);   // rsel/rflip: ring state indexed by a runtime slot lives in LOCAL memory otherwise (see file header)
                for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                        ::"r"(base+SCOL[jj]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); } }
            mma_commit(SF[jj]); };
        auto pv=[&](int jj,int i,int s){ if(lead){ uint64_t vd0=rsel(vd0a,s);
                for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[jj]+(uint32_t)(kb*8);
                    asm volatile("{ .reg .pred p; setp.ne.b32 p,%3,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                        ::"r"(base+OCOL[jj]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); } }
            mma_commit(OF[jj]); };
        int pkf[KD]={0}, pvf[VD]={0}, poe0=1, poe1=1, pf0=0, pf1=0;
        // prologue: block0 QK0,QK1 + PV0(0). QK waits K only; PV waits V. A K slot is released right
        // after its SECOND QK is issued (both QKs read sK, nothing else does) -- decoupling it from PV is
        // the whole point of KD > VD.
        mbar_wait(KF[0], pkf[0]); pkf[0]^=1;
        qk(0,0); qk(1,0); mma_commit(KE[0]);
        mbar_wait(OE[0], poe0); poe0^=1;
        mbar_wait(PF[0], pf0); pf0^=1;
        mbar_wait(VF[0], pvf[0]); pvf[0]^=1;
        pv(0,0,0);
        for(int i=1;i<NBk;++i){ int sk=i%KD, sv=i%VD, svp=(i-1)%VD;
            mbar_wait(rsel(KF,sk), rflip(pkf,sk));
            qk(0,sk);
            mbar_wait(OE[1], poe1); poe1^=1;
            mbar_wait(PF[1], pf1); pf1^=1;
            pv(1,i-1,svp);
            mma_commit(rsel(VE,svp));                       // V slot i-1 free: PV0 and PV1 both consumed it
            qk(1,sk);
            mma_commit(rsel(KE,sk));                        // K slot i free: both QKs issued
            mbar_wait(rsel(VF,sv), rflip(pvf,sv));
            mbar_wait(OE[0], poe0); poe0^=1;
            mbar_wait(PF[0], pf0); pf0^=1;
            pv(0,i,sv);
        }
        { int svl=(NBk-1)%VD;
          mbar_wait(OE[1], poe1); poe1^=1;
          mbar_wait(PF[1], pf1); pf1^=1;
          pv(1,NBk-1,svl);
          mma_commit(rsel(VE,svl)); }
        mma_commit(SF[0]); mma_commit(SF[1]);           // final dummy S commits (pair softmax terminal SF wait)
    } else {                             // -------- CORRECTION WG (w8-11, 128 threads) --------
        const uint32_t OF[2]={smem_u32(&bars.o_full[0]),smem_u32(&bars.o_full[1])};
        const uint32_t OE[2]={smem_u32(&bars.o_empty[0]),smem_u32(&bars.o_empty[1])};
        // Per-half tables + rsel/rflip: as [2][2] arrays indexed by the runtime slot these lived in LOCAL
        // memory and were the fa4 family's dominant local-memory traffic (see rsel above).
        const uint32_t CF0[2]={smem_u32(&bars.c_full[0]),smem_u32(&bars.c_full[1])};
        const uint32_t CF1[2]={smem_u32(&bars.c_full[2]),smem_u32(&bars.c_full[3])};
        const uint32_t CE0[2]={smem_u32(&bars.c_empty[0]),smem_u32(&bars.c_empty[1])};
        const uint32_t CE1[2]={smem_u32(&bars.c_empty[2]),smem_u32(&bars.c_empty[3])};
        const int r=tid&127;
        int po0=0, po1=0; int pcf0[2]={0,0}, pcf1[2]={0,0};
        // final normalize + store for half jj (split -> o_part/lse_part, else direct out+lse+sink)
        auto store_half=[&](int jj, float fs, float fm){
            int qr=r0[jj]+r, head=GQA?(h*rep+jj):h;
            bool ow=(qr<seqlen_q) && (!GQA || head<hq);
            uint32_t oA=base+OCOL[jj]+((uint32_t)r<<16);
            if(nsplits>1){
                float invp=(fs>0.f)?1.f/fs:0.f;
                #pragma unroll 1
                for(int g=0; g<HDV; g+=32){ uint32_t ov[32]; tmem_ld32(oA+(uint32_t)g, ov); tmem_wait_ld();
                    if(ow){ for(int t=0;t<32;++t){ int d=g+t; if(d<rhdv) o_part[(((size_t)split_idx*TQ+(q0+qr))*hq+head)*rhdv+d]=__uint_as_float(ov[t])*invp; } } }
                if(ow) lse_part[((size_t)split_idx*hq+head)*TQ+(q0+qr)] = (fs>0.f)?(logf(fs)+fm*INV_LOG2E):-INFINITY;
            } else {
                float mfin=fm, lfin=fs, osc=1.f;
                if(s_aux!=nullptr){ float sk=s_aux[head]*LOG2E; float mn=fmaxf(fm,sk);
                    float c2=(fm==-INFINITY)?0.f:ex2f(fm-mn); lfin=fs*c2+ex2f(sk-mn); osc=c2; mfin=mn; }
                float inv=(lfin>0.f)?osc/lfin:0.f; unsigned long long inv2=bcast2(inv);
                size_t obase=ow?((size_t)(q0+qr)*hq*rhdv+(size_t)head*rhdv):0;
                #pragma unroll 1
                for(int g=0; g<HDV; g+=32){ uint32_t ov[32]; tmem_ld32(oA+(uint32_t)g, ov); tmem_wait_ld();
                    #pragma unroll
                    for(int t=0;t<32;t+=2) pmul2(ov[t],ov[t+1],inv2);
                    if(ow){
                        #pragma unroll
                        for(int t=0;t<32;t+=8){ int d=g+t;
                            if(d+8<=rhdv){ uint4 vec; vec.x=pack16<T>(__uint_as_float(ov[t+0]),__uint_as_float(ov[t+1]));
                                vec.y=pack16<T>(__uint_as_float(ov[t+2]),__uint_as_float(ov[t+3])); vec.z=pack16<T>(__uint_as_float(ov[t+4]),__uint_as_float(ov[t+5]));
                                vec.w=pack16<T>(__uint_as_float(ov[t+6]),__uint_as_float(ov[t+7])); *reinterpret_cast<uint4*>(&out[obase+d])=vec;
                            } else { for(int u=0;u<8;++u){ int dd=d+u; if(dd<rhdv) out[obase+dd]=from_f<T>(__uint_as_float(ov[t+u])); } } } }
                }
                if(ow) lse[(size_t)head*TQ+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin*INV_LOG2E):-INFINITY;
            }
        };
        // prologue: swallow p=0 (block0 needs no rescale) both halves, release slot0
        mbar_wait(CF0[0], rflip(pcf0,0)); mbar_arrive(CE0[0]);
        mbar_wait(CF1[0], rflip(pcf1,0)); mbar_arrive(CE1[0]);
        for(int it=0; it<NBk-1; ++it){
            int sl=(it+1)&1;
            mbar_wait(rsel(CF0,sl), rflip(pcf0,sl)); float sc0=sScale[0][sl][r];
            mbar_wait(OF[0], po0); po0^=1; tmem_fence();
            if(!__all_sync(0xffffffffu,sc0==1.0f)) tmem_o_rescale<HDV>(base+OCOL[0]+((uint32_t)r<<16), sc0);
            mbar_arrive(rsel(CE0,sl));
            tmem_wait_st(); mbar_arrive(OE[0]);
            mbar_wait(rsel(CF1,sl), rflip(pcf1,sl)); float sc1=sScale[1][sl][r];
            mbar_wait(OF[1], po1); po1^=1; tmem_fence();
            if(!__all_sync(0xffffffffu,sc1==1.0f)) tmem_o_rescale<HDV>(base+OCOL[1]+((uint32_t)r<<16), sc1);
            mbar_arrive(rsel(CE1,sl));
            tmem_wait_st(); mbar_arrive(OE[1]);
        }
        int fsl=NBk&1;
        // ---- final stage 0: normalize O0 -> store ----
        mbar_wait(rsel(CF0,fsl), rflip(pcf0,fsl)); float fs0=sFinSum[0][fsl][r], fm0=sFinMax[0][fsl][r];
        mbar_arrive(rsel(CE0,fsl));
        mbar_wait(OF[0], po0); po0^=1; tmem_fence();
        store_half(0, fs0, fm0);
        tmem_wait_ld(); mbar_arrive(OE[0]);
        // ---- final stage 1: normalize O1 -> store ----
        mbar_wait(rsel(CF1,fsl), rflip(pcf1,fsl)); float fs1=sFinSum[1][fsl][r], fm1=sFinMax[1][fsl][r];
        mbar_arrive(rsel(CE1,fsl));
        mbar_wait(OF[1], po1); po1^=1; tmem_fence();
        store_half(1, fs1, fm1);
        tmem_wait_ld(); mbar_arrive(OE[1]);
    }
    __syncthreads();
    if(warp==12) tmem_dealloc(base,512);
}

// Paged warp-specialized pipeline (q_stage=1) + SplitKV. Like fwd_kernel_ws but the load is a MANUAL
// multi-warp gather from paged K/V (page_table) into the koff/voff smem layout the MMA descriptors
// read — TMA can't gather scattered pages cheaply at block_size<128. Roles: w0-3 softmax, w4 MMA,
// w5-15 LOAD (11 warps gather+swizzle). SplitKV: this split's K sub-range -> partial (O,LSE); the
// combine_kernel merges. This unlocks the WS pipeline (fast per-block) for the paged/large-K configs
// that were stuck on the slow serial kernel (ctx20k, prefill3577, paged decode).
// PACK=true: pack the `rep` GQA q-heads' queries densely into the tile rows (dense-M decode) so
// warp 0 does useful work instead of 1-of-32 rows. h=blockIdx.y is then the KV head. row_global r ->
// qhead = h*rep + r/seqlen_q, qrow = r%seqlen_q. Shared K/V load across the rep heads.
// SD = K/V smem stage depth. SD=WST(2): 512 threads, 1 CTA/SM, pipelined (prefetch QK[i+1]) — prefill/large-K.
// SD=1: 256 threads + single K/V buffer -> ~96KB smem => 2 CTAs/SM. Serial per-block (no intra-CTA overlap) but
// two independent CTAs interleave to hide the exposed tmem-read latency that pins low-rep decode (0.26 eligible warps).
// NTHR: CTA width. The page-TMA producer needs only nseg*(HD/64+HDV/64) issues per tile (8 at hd128 /
// page 64), so a 256-thread CTA (w0-3 softmax, w4 MMA, w5-7 load) is enough -- and at
// __launch_bounds__(256,1) ptxas gets 256 regs/thread, which makes the 1-pass softmax's f[128] and the
// epilogue's of[HDV] fragment SPILL-FREE (the 512-thread form is capped at 128 regs and spills both).
// No setmaxnreg redistribution is needed at NTHR==256 for the same reason.
template<class T, int HD, int HDV, bool PACK=false, int SD=WST, bool CORR=false, bool UG4=false,
         bool UPT=false, int NTHR=512>
__global__ void __launch_bounds__(SD==1?256:NTHR, SD==1?2:1) fwd_kernel_ws_paged(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float* __restrict__ lse, const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int paged, const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, int leftpad, int nB, int nsplits,
    float* __restrict__ o_part, float* __restrict__ lse_part,
    const __grid_constant__ CUtensorMap mKg, const __grid_constant__ CUtensorMap mVg, int use_g4, int grp,
    const __grid_constant__ CUtensorMap mKf, const __grid_constant__ CUtensorMap mVf, int have_full) {
    constexpr int WST=SD;   // shadow namespace WST -> per-instantiation K/V stage depth (buffer sizing + ring)
    // CORR: FA4 correction-warpgroup choreography. w0-3 softmax, w4-7 correction (O-rescale), w8 MMA,
    // w9-15 gather. Softmax writes corr->sScale + signals CS; correction rescales O off the softmax
    // critical path + signals CO; MMA waits CO before PV. Decouples the per-block O-rescale (the
    // multi-row decode softmax bottleneck) from softmax so softmax[i+1] overlaps correction[i].
    constexpr int MMA_W  = CORR ? 8 : 4;    // MMA warp
    constexpr int GW0    = CORR ? 9 : 5;    // first gather warp
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16;
    constexpr int SCOL[2]={0,128}, OCOL=256, PCOL[2]={384,448};
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[WST];
    { T* p=sQ+WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<WST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) Bars bars; __shared__ __align__(8) uint32_t s_base;
    __shared__ int prow_c[WST][WTN];   // page-row per key, precomputed once/block (breaks the page_table->K dep chain)
    __shared__ float sScale[2][2][WTM];   // corr per (tile, block-parity, row): softmax -> correction WG (w12-15)
    __shared__ int s_contig;              // page-table run detection: are this CTA's WTN-key tiles contiguous?
    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y;
    const int b=blockIdx.z%nB, split_idx=blockIdx.z/nB, hk=PACK?h:h/rep;   // PACK: h is the KV head
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
    const int k0 = paged ? 0 : cu_k[b];
    const int seqlen_k = paged ? seqused_k[b] : (cu_k[b+1]-cu_k[b]);
    const int MROWS = PACK ? rep*seqlen_q : seqlen_q;   // tile rows this CTA covers (packed heads*queries)
    const int row0=mb*WTM;
    const float LOG2E=1.4426950408889634f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    if (row0>=MROWS) return;
    // MIXED-BATCH GROUP FILTER (grp: 0=all, 1=decode-like only, 2=prefill-like only). A ragged paged batch
    // holding one long prefill plus many q=1 decodes used to go down ONE route sized by the longest
    // sequence: pack=false, so every decode sequence ran hq separate 128-row Q tiles with a single valid
    // row, reading K/V once per Q-HEAD instead of once per KV-head (rep=4 -> 4x the KV traffic) across
    // gx*hq CTAs of which one in gx did anything. Splitting the batch by q length lets the decode
    // sequences use the packed-GQA route. The filter is a CTA-uniform predicate, so no b_map/D2H is
    // needed -- each launch simply covers the whole batch and drops the sequences it does not own.
    if (grp==1 && rep*seqlen_q >  WTM) return;
    if (grp==2 && rep*seqlen_q <= WTM) return;
    int qp_max, qp_min;
    if(PACK){ qp_max=seqlen_k-1; qp_min=seqlen_k-seqlen_q; }   // packed tile spans all queries 0..seqlen_q-1
    else { qp_max=seqlen_k-seqlen_q+row0+(WTM-1); qp_min=seqlen_k-seqlen_q+row0; }
    int n_lo=0, n_hi=(seqlen_k+WTN-1)/WTN;
    if(is_causal) n_hi=min(n_hi,(qp_max)/WTN+1);
    if(win_r>=0)  n_hi=min(n_hi,(qp_max+win_r)/WTN+1);
    if(win_l>=0){ int lo=qp_min-win_l; n_lo=(lo>0)?lo/WTN:0; }
    n_hi=max(n_hi,n_lo);
    if (nsplits>1){ int on_lo=n_lo,on_hi=n_hi,tot=on_hi-on_lo,per=(tot+nsplits-1)/nsplits;
        n_lo=min(on_hi,on_lo+split_idx*per); n_hi=min(on_hi,on_lo+(split_idx+1)*per); }
    const int NBk=n_hi-n_lo;

    if (warp>=4 && NTHR>256){ if constexpr(SD==1) asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory");   // 2 CTAs/SM: tighter producer budget
                  else                 asm volatile("setmaxnreg.dec.sync.aligned.u32 104;\n" ::: "memory"); }   // producers release; softmax incs to 192 in-branch for the 1-pass fragment
    // Q is loaded RAW and `qsc` (*scale*log2e no-softcap, *scale w/ softcap) is folded into the softmax's
    // exp2 argument instead. Pre-scaling Q rounded q*qsc back to bf16, adding ~2^-9 relative error to every
    // Q element BEFORE the QK dot product -- enough to push attn_kv_hd128_decode_qlen120 (err ratio 2.23)
    // and llama8b_spec_q8_b16_s4k (2.10) past the FA "2x the baseline's error" gate. Folding is free: the
    // exp2 argument was already an add (f - m_new) and becomes an FFMA (f*qsc - m_new).
    const float qsc=(softcap>0.f)?scale:scale*LOG2E;
    // Split the Q stage: gather only the rows this tile OWNS, then bulk-zero the rest. A GQA-packed decode
    // tile holds rep rows (4 or 8) of WTM=128, so the old single loop ran 32x its useful work through the
    // per-element (div, mod, koff, predicate, store) path -- once per CTA, but a b64/ctx4k decode CTA only
    // runs 32 K-blocks, so the prologue was a real share of it. The unowned rows must still be ZEROED, not
    // left as garbage: a garbage row's row-max makes its `corr` != 1, which breaks the correction WG's
    // warp-ballot skip of the 128-f32 tmem O rescale and costs far more than the fill saves (measured:
    // leaving them uninitialised took dec_s16k 0.955 -> 0.874). koff keeps each row inside two contiguous
    // 64-element blocks per 64-column group, so the fill is a straight uint4 vector store.
    const int nrowQ=min((PACK?MROWS:seqlen_q)-row0, WTM);
    for(int i=tid;i<nrowQ*HD;i+=blockDim.x){ int r=i/HD,c=i%HD; int rg=row0+r;
        int qh=PACK?(hk*rep+rg/seqlen_q):h; int qr=PACK?(rg%seqlen_q):rg;
        bool ok=(PACK?(rg<MROWS):(qr<seqlen_q))&&c<rhd;
        sQ[koff(r,c)] = ok ? q[(size_t)(q0+qr)*hq*rhd+(size_t)qh*rhd+c] : from_f<T>(0.f); }
    { constexpr int NCB=HD/64; const int nvec=(WTM-nrowQ)*64/8;
      for(int j=0;j<NCB;++j)
        for(int i=tid;i<nvec;i+=blockDim.x) ((uint4*)(sQ+j*8192+nrowQ*64))[i]=make_uint4(0,0,0,0); }
    if(warp==MMA_W){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    // UG4's warp-uniform producer has each load WARP arrive+expect_tx for its own key-group slice, so
    // kv_full's arrive count is the load-warp count; the cp.async producer defers ONE arrival per load
    // THREAD via cp.async.mbarrier.arrive.noinc, so its count is the load-thread count. Either way the
    // per-tile CTA rendezvous on the completion side is gone.
    const int NW_LOAD=((int)blockDim.x-GW0*32)>>5, NT_LOAD=(int)blockDim.x-GW0*32;
    // PAGE-TABLE RUN DETECTION (UG4 only, i.e. page_size % 8 != 0 where no multi-row box was buildable at
    // launch): if the physical rows this CTA will touch are consecutive, the whole tile is 4 full-width
    // [64,WTN] boxes instead of 128 gather4 (see tma_full_kv_warp). Scan only the CTA's OWN block range so
    // the cost tracks the work; the page table is tiny and shared by every head of the batch, so all but the
    // first CTA hits L2. CTA-uniform result -> no per-tile decision and no extra rendezvous in the mainloop.
    if(tid==0) s_contig=0;
    __syncthreads();
    if constexpr(UG4){
      if(have_full && paged && block_size>0 && NBk>0){
        const int* pt=page_table+(size_t)b*pt_stride;
        int pg_lo=(n_lo*WTN)/block_size, pg_hi=((n_hi*WTN<seqlen_k?n_hi*WTN:seqlen_k)+block_size-1)/block_size;
        int base_pg=pt[pg_lo], bad=0;
        for(int p=pg_lo+tid; p<pg_hi; p+=blockDim.x) if(pt[p]!=base_pg+(p-pg_lo)){ bad=1; break; }
        if(bad) atomicOr(&s_contig,2);              // bit1 = a break was found
      } else if(tid==0) atomicOr(&s_contig,2);      // not eligible (no full-width descriptor) -> stay on gather4
      if(tid==0) atomicOr(&s_contig,1);             // bit0 = the scan ran
    }
    __syncthreads();
    // Participating load warps. A contiguous tile is only KCB+VCB full-width boxes, so parking the surplus
    // load warps stops them from spinning on kv_empty/kv_full and competing for issue slots with the softmax
    // (11 warps spinning to issue 4 boxes cost ~19% on dec_s16k). kv_full's arrive count follows.
    constexpr int NOP_FULL = HD/64 + HDV/64;
    const bool CONTIG = UG4 && (s_contig==1);
    const int NWU = (CONTIG && NW_LOAD>NOP_FULL) ? NOP_FULL : NW_LOAD;
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),UG4?NWU:(UPT?1:NT_LOAD)); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.s_empty[j]),1);
                                      mbar_init(smem_u32(&bars.p_full[j]),128); mbar_init(smem_u32(&bars.o_ready[j]),1);}
                if constexpr(CORR){ mbar_init(smem_u32(&bars.corr_stats[0]),128); mbar_init(smem_u32(&bars.o_resc[0]),128); } }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])}, SE[2]={smem_u32(&bars.s_empty[0]),smem_u32(&bars.s_empty[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])}, OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    const uint32_t CS0=smem_u32(&bars.corr_stats[0]), CO0=smem_u32(&bars.o_resc[0]);   // CORR: softmax->correction / correction->MMA
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();

    if (warp>=GW0) {                     // LOAD: gather warps (w5.. or w9.. under CORR) manual paged gather -> koff/voff smem
        const int lid=tid-GW0*32, LN=(int)blockDim.x-GW0*32; int pe[WST]={0};
      if constexpr(UPT){
        // PAGE-ALIGNED TMA producer: the WTN-key tile is nseg contiguous physical-page segments, so the
        // whole K+V tile is nseg*(HD/64+HDV/64) HW-async TMA boxes (8 for hd128 @ page 64) instead of
        // ~4096 cp.async issues. Only nseg page-table lookups per tile, not WTN.
        const int segrows = block_size<WTN ? block_size : WTN, nseg = WTN/segrows;
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(rsel(KVE,s),rflip(pe,s));
            const uint32_t kvfb=rsel(KVF,s); T* const kbs=rsel(sK,s); T* const vbs=rsel(sV,s);
            if(lid==0) mbar_expect(kvfb, WTN*HD*2 + WTN*HDV*2);   // ordered ahead of every box below
            int key0=(n_lo+i)*WTN;
            for(int t=lid;t<nseg;t+=LN){ int kk=key0+t*segrows;
                // OOB tail segments (kk>=seqlen_k) have no valid page-table entry: pull pool row 0 and let
                // the boundary softmax mask those keys (p=0, so V garbage cannot reach O).
                prow_c[s][t] = (kk<seqlen_k) ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : 0; }
            asm volatile("bar.sync 3, %0;"::"r"(LN));   // pseg ready + expect_tx visible
            tma_pages_kv<T,HD,HDV>(&mKg,&mVg, kbs, vbs, prow_c[s], nseg, segrows, hk, rhd, rhdv, kvfb, lid, LN);
        }
      } else if constexpr(UG4){
        // WARP-UNIFORM tile::gather4: one warp per key-group slice, lane 0 issues, page rows staged by the
        // warp's own lanes. No CTA-wide bar.sync and no per-lane UTMALDG replay (see gather4_kv_warp).
        // `contig` upgrades FULL tiles to 4 full-width [64,WTN] boxes (32x fewer TMA descriptors).
        const int lw=lid>>5, lane=lid&31;
        const int pt0 = CONTIG ? (page_table[(size_t)b*pt_stride+(n_lo*WTN)/block_size]*block_size
                                  + (n_lo*WTN)%block_size - n_lo*WTN) : 0;   // key -> physical row bias
        if(lw<NWU)
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(rsel(KVE,s),rflip(pe,s));
            int key0=(n_lo+i)*WTN; const uint32_t kvfb=rsel(KVF,s); T* const kbs=rsel(sK,s); T* const vbs=rsel(sV,s);
            if(CONTIG && key0+WTN<=seqlen_k)
                tma_full_kv_warp<T,HD,HDV>(&mKf,&mVf, kbs, vbs, pt0+key0, hk, rhd, rhdv, kvfb, lw, NWU, lane);
            else
                gather4_kv_warp<T,HD,HDV>(&mKg,&mVg, kbs, vbs, prow_c[s], hk, rhd, rhdv, kvfb,
                    lw, NWU, lane, key0, seqlen_k, page_table+(size_t)b*pt_stride, block_size, paged, k0);
        }
      } else {
        // FA4's cp.async paged gather, uncrippled. 16B (uint4) LDGSTS per lane straight into the swizzled
        // MMA layout, completion signalled by `cp.async.mbarrier.arrive.noinc` (kv_full's arrive count is
        // the load-thread count) so the producer NEVER blocks. The previous form ran
        // `cp.async.commit_group; cp.async.wait_group 0` INSIDE this loop, which pinned the ring at ONE
        // tile in flight no matter how deep it was -- that, not the instruction count, is why this path
        // used to lose to gather4. Both move 512B per warp-instruction and issue the same 128 of them per
        // tile, but gather4 goes through the TMA engine, whose in-flight descriptor budget caps a
        // page_size=1 decode at ~3.1 TB/s against a measured 6.14 TB/s roof (ncu: 165 cycles per gather4
        // issue, DRAM 50.8% of roof, zero over-fetch); LDGSTS misses use the SM's own far deeper queues.
        // Columns >= rhd are zeroed ONCE (the copies never touch them) and OOB keys clamp to pool row 0
        // (the softmax masks them, PV multiplies by p=0), so the hot loop has no synchronous smem store to
        // order against the deferred arrival.
        if(rhd<HD||rhdv<HDV){
            for(int s=0;s<WST;s++){
                for(int i=lid;i<WTN*HD /8;i+=LN) ((uint4*)sK[s])[i]=make_uint4(0,0,0,0);
                for(int i=lid;i<WTN*HDV/8;i+=LN) ((uint4*)sV[s])[i]=make_uint4(0,0,0,0);
            }
        }
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(rsel(KVE,s),rflip(pe,s));
            int key0=(n_lo+i)*WTN;
            for(int r=lid;r<WTN;r+=LN){ int kk=key0+r;
                prow_c[s][r] = (kk<seqlen_k) ? (paged ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : (k0+kk)) : 0; }
            asm volatile("bar.sync 3, %0;"::"r"(LN));   // prow_c[s] ready
          {
            for(int idx=lid;idx<WTN*(HD/8);idx+=LN){ int r=idx/(HD/8), c=(idx%(HD/8))*8;
                if(c<rhd) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sK[s][koff(r,c)])),"l"(&k[(size_t)prow_c[s][r]*hkv*rhd+(size_t)hk*rhd+c])); }
            for(int idx=lid;idx<WTN*(HDV/8);idx+=LN){ int key=idx/(HDV/8), dv=(idx%(HDV/8))*8;
                if(dv<rhdv) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sV[s][voff(key,dv)])),"l"(&v[(size_t)prow_c[s][key]*hkv*rhdv+(size_t)hk*rhdv+dv])); }
            cp_async_mbar_arrive(rsel(KVF,s));
          }
        }
      }
    } else if (warp==MMA_W) {            // MMA warp
        int pf[WST]={0}, pse[2]={0,0}, ppf[2]={0,0}, pco=0;
        // // rsel/rflip: ring state indexed by a runtime slot lives in LOCAL memory otherwise (see file header). The descriptors were also
        // rebuilt from sK[s]/sV[s] on EVERY block; build them once and select in registers.
        const uint64_t qd0c=make_desc(smem_u32(sQ),16,1024,2);
        uint64_t kd0a[WST],vd0a[WST];
        #pragma unroll
        for(int s=0;s<WST;s++){ kd0a[s]=make_desc(smem_u32(sK[s]),16,1024,2); vd0a[s]=make_desc(smem_u32(sV[s]),16384,1024,2); }
        auto qk=[&](int i){ int bb=i&1,s=i%WST; uint64_t qd0=qd0c,kd0=rsel(kd0a,s);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SCOL[bb]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); }
            mma_commit(SF[bb]); };
        auto pv=[&](int i){ int bb=i&1,s=i%WST; uint64_t vd0=rsel(vd0a,s);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=(i>0||kb>0)?1u:0u; uint32_t pa=base+PCOL[bb]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+OCOL),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); }
            mma_commit(OR[bb]); };
        if constexpr(SD==1){             // serial: single K/V buffer (2 CTAs/SM interleave to hide latency)
            for(int i=0;i<NBk;++i){ int bb=i&1;
                mbar_wait(KVF[0],pf[0]); pf[0]^=1;
                mbar_wait(SE[bb],pse[bb]); pse[bb]^=1; qk(i);
                mbar_wait(PF[bb],ppf[bb]); ppf[bb]^=1; pv(i); mma_commit(KVE[0]);
            }
        } else if(NBk>0){                // pipelined: prefetch QK[i+1] across double-buffered K/V
        mbar_wait(KVF[0],pf[0]); pf[0]^=1; mbar_wait(SE[0],pse[0]); pse[0]^=1; qk(0);
        for(int i=0;i<NBk;++i){ int s=i%WST;
            if(i+1<NBk){ int ns=(i+1)%WST,nb=(i+1)&1; mbar_wait(rsel(KVF,ns),rflip(pf,ns)); mbar_wait(SE[nb],pse[nb]); pse[nb]^=1; qk(i+1); }
            mbar_wait(PF[i&1],ppf[i&1]); ppf[i&1]^=1;
            if constexpr(CORR){ mbar_wait(CO0,pco); pco^=1; }   // wait correction: O rescaled by corr[i] before PV[i] accumulates
            pv(i); mma_commit(rsel(KVE,s));
        } }
    } else if (warp<4) {                 // softmax WG (w0-3): 1-pass full / 2-pass boundary (Q prescaled -> log2 domain)
        if constexpr(NTHR>256) asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n" ::: "memory");   // IN-BRANCH: 192 regs for the held S fragment (NTHR==256 already has 256/thread)
        const float INV_LOG2E=0.6931471805599453f;   // ln2: log2-domain max -> natural for LSE
        int psf[2]={0,0}, por[2]={0,0};
        float m_i=-INFINITY, l_i=0.f;
        if(tid==0){ mbar_arrive(SE[0]); mbar_arrive(SE[1]); for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); }
        const int rg0=row0+tid, qrow=PACK?(rg0%seqlen_q):rg0, qhd=PACK?(hk*rep+rg0/seqlen_q):h;
        const int qp=seqlen_k-seqlen_q+qrow;
        int key0off=0;
        // sv is the RAW dot product: no-softcap -> scale by qsc(=scale*log2e); softcap -> qsc(=scale) then tanh.
        auto logit=[&](float sv,int c)->float{ float lg=(softcap>0.f)?(softcap*tanhf(sv*qsc*inv_sc)*LOG2E):(sv*qsc);
            int kk=key0off+c; bool ok=(kk<seqlen_k)&&(kk>=leftpad);
            if(is_causal&&kk>qp) ok=false; if(win_l>=0&&kk<qp-win_l) ok=false; if(win_r>=0&&kk>qp+win_r) ok=false; return ok?lg:-INFINITY; };
        const bool wv=((tid&~31)<MROWS);   // warp has >=1 valid (packed) row; else skip its softmax work
        for(int i=0;i<NBk;++i){ int bb=i&1; key0off=(n_lo+i)*WTN;
            mbar_wait(SF[bb],psf[bb]); psf[bb]^=1; tmem_fence();
            const int kend=key0off+WTN-1;    // uniform full-block test (no per-key mask needed for any row of the tile)
            bool full=(kend<seqlen_k)&&(softcap<=0.f)&&(key0off>=leftpad);
            if(is_causal && kend>qp_min) full=false;
            if(win_r>=0 && kend>qp_min+win_r) full=false;
            if(win_l>=0 && key0off<qp_max-win_l) full=false;
            float m_new, corr, ps=0.f;
            if(wv && full){
                uint32_t f[128];
                LD32_INTO(base+SCOL[bb]+((uint32_t)tid<<16)+0,  f, 0);
                LD32_INTO(base+SCOL[bb]+((uint32_t)tid<<16)+32, f, 32);
                LD32_INTO(base+SCOL[bb]+((uint32_t)tid<<16)+64, f, 64);
                LD32_INTO(base+SCOL[bb]+((uint32_t)tid<<16)+96, f, 96);
                tmem_wait_ld();
                float rmax=-INFINITY;
                #pragma unroll
                for(int t=0;t<128;++t) rmax=fmaxf(rmax,__uint_as_float(f[t]));
                m_new=fmaxf(m_i,rmax*qsc);   // qsc>0 -> scaling preserves the argmax, so scale the max once here
                corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                #pragma unroll
                for(int g=0;g<4;++g){ uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ float p0=ex2f(fmaf(__uint_as_float(f[g*32+2*t]),qsc,-m_new)), p1=ex2f(fmaf(__uint_as_float(f[g*32+2*t+1]),qsc,-m_new)); ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(base+PCOL[bb]+((uint32_t)tid<<16)+g*16, pk); }
            } else if(wv){                   // BOUNDARY: 2-pass (per-key masking via logit)
                float rmax=-INFINITY;
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[bb]+((uint32_t)tid<<16)+g*32, v);
                    #pragma unroll
                    for(int t=0;t<32;++t) rmax=fmaxf(rmax,logit(__uint_as_float(v[t]),g*32+t)); }
                tmem_wait_ld();
                m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                for(int g=0;g<4;++g){ uint32_t v[32]; tmem_ld32(base+SCOL[bb]+((uint32_t)tid<<16)+g*32, v); uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ float l0=logit(__uint_as_float(v[2*t]),g*32+2*t), l1=logit(__uint_as_float(v[2*t+1]),g*32+2*t+1);
                        float p0=(l0==-INFINITY)?0.f:ex2f(l0-m_new), p1=(l1==-INFINITY)?0.f:ex2f(l1-m_new); ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(base+PCOL[bb]+((uint32_t)tid<<16)+g*16, pk); }
            } else { m_new=m_i; corr=1.f; }
            if(wv){ l_i=l_i*corr+ps; m_i=m_new; }
            asm volatile("bar.sync 1, 128;");
            if(tid==0) mbar_arrive(SE[bb]);
            if constexpr(CORR){                    // publish corr to correction WG; O-rescale is off the softmax path
                sScale[0][bb][tid]=corr; tmem_wait_st(); mbar_arrive(PF[bb]); mbar_arrive(CS0);
            } else {
                if(i>0){ int pb=(i-1)&1; mbar_wait(OR[pb],por[pb]); por[pb]^=1;   // wait unguarded (barrier-phase consistency)
                    if(wv && !__all_sync(0xffffffffu,corr==1.0f)) tmem_o_rescale<HDV>(base+OCOL+((uint32_t)tid<<16), corr); }
                tmem_wait_st(); mbar_arrive(PF[bb]);   // all 128 arrive (no 2nd bar.sync)
            }
        }
        // final O: wait last PV. CORR -> correction consumed OR[0..NBk-2]; softmax polls OR[last] for its
        // (cnt)-th flip (cnt = # blocks of that parity). non-CORR -> por tracked through the loop.
        if constexpr(CORR){ if(NBk>0){ int fb=(NBk-1)&1; int cnt=(NBk-1-fb)/2+1; mbar_wait(OR[fb],(cnt-1)&1); } }
        else { if(NBk>0){ int pb=(NBk-1)&1; mbar_wait(OR[pb],por[pb]); por[pb]^=1; } }
        tmem_fence();
        const int qr=qrow; const bool ow=(PACK?(rg0<MROWS):(qr<seqlen_q));
        uint32_t of[HDV];   // vectorized O read: HDV/32 x LD32 + ONE wait, not HDV serial tmem_ld/tmem_wait_ld
        if(NBk>0 && wv){ LD32_INTO(base+OCOL+((uint32_t)tid<<16)+0, of, 0);
            if constexpr(HDV>32) LD32_INTO(base+OCOL+((uint32_t)tid<<16)+32, of, 32);
            if constexpr(HDV>64) LD32_INTO(base+OCOL+((uint32_t)tid<<16)+64, of, 64);
            if constexpr(HDV>96) LD32_INTO(base+OCOL+((uint32_t)tid<<16)+96, of, 96);
            tmem_wait_ld(); }
        if (nsplits>1) {
            const int TQ=cu_q[nB]; float invp=(l_i>0.f)?1.f/l_i:0.f;
            for(int d=0;d<HDV;++d){ float o=(NBk>0&&wv)?__uint_as_float(of[d]):0.f;
                if(ow && d<rhdv) o_part[(((size_t)split_idx*TQ+(q0+qr))*hq+qhd)*rhdv+d]=o*invp; }
            if(ow) lse_part[((size_t)split_idx*hq+qhd)*TQ+(q0+qr)] = (l_i>0.f)?(logf(l_i)+m_i*INV_LOG2E):-INFINITY;
        } else {
            float mfin=m_i, lfin=l_i, osc=1.f;
            if(s_aux!=nullptr){ float sk2=s_aux[qhd]*LOG2E; float mn=fmaxf(m_i,sk2); float c2=(m_i==-INFINITY)?0.f:ex2f(m_i-mn); lfin=l_i*c2+ex2f(sk2-mn); osc=c2; mfin=mn; }
            float inv=(lfin>0.f)?osc/lfin:0.f;
            for(int d=0;d<HDV;++d){ float o=(NBk>0&&wv)?__uint_as_float(of[d]):0.f;
                if(ow && d<rhdv) out[(size_t)(q0+qr)*hq*rhdv+(size_t)qhd*rhdv+d]=from_f<T>(o*inv); }
            if(ow) lse[(size_t)qhd*cu_q[nB]+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin*INV_LOG2E):-INFINITY;
        }
    } else if constexpr(CORR){            // correction WG (w4-7): O-rescale off the softmax critical path
        const int r=tid-128;             // 128 correction threads -> one O row each (matches softmax tid->row)
        int pcs=0, por2[2]={0,0};
        for(int i=0;i<NBk;++i){
            mbar_wait(CS0,pcs); pcs^=1;                          // corr[i] published in sScale
            float corr=sScale[0][i&1][r];
            if(i>0){ int pb=(i-1)&1; mbar_wait(OR[pb],por2[pb]); por2[pb]^=1; tmem_fence();   // O_{i-1}=PV[i-1] done
                if(!__all_sync(0xffffffffu,corr==1.0f)) tmem_o_rescale<HDV>(base+OCOL+((uint32_t)r<<16), corr);
                tmem_wait_st(); }
            mbar_arrive(CO0);            // signal MMA: O rescaled -> PV[i] may accumulate
        }
    }
    __syncthreads();
    if(warp==MMA_W) tmem_dealloc(base,512);
}

// FA4 decode choreography: intra-CTA 2-way split-K (distilled from cutlass 77_blackwell_fmha gen kernel,
// ThreadShape<1,2,1> "side-by-side" K split). Two softmax warpgroups run the online softmax over HALF the
// K blocks each (w0-3 = even blocks/stream0, w4-7 = odd blocks/stream1) of the SAME query rows -> while
// one WG stalls on its tmem-S read, the other issues, hiding the read latency that pins the single-WG
// kernel at ~1 CTA/SM. MMA (w8) feeds both streams: 1 qk + 1 pv per K block (NOT doubled, unlike q_stage=2
// which splits by qhead). Gather w9-15. tmem(512 cols): S0@0 S1@128 O0@256 O1@256+HDV; P aliased into S
// (P0@0 P1@128) since S is dead once read into regs -- the MMA's in-order pv(i)-before-qk(i+2) protects the
// WAR. Final out = combine(O0,m0,l0 ; O1,m1,l1) done by w0-3. Non-paged/PACK/softcap-free multi-row decode.
template<class T, int HD, int HDV>
__global__ void __launch_bounds__(512,1) fwd_kernel_ws_paged_spk(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ out, float* __restrict__ lse, const int* __restrict__ cu_q, const int* __restrict__ cu_k,
    int hq, int hkv, int rep, float scale, int is_causal, int win_l, int win_r,
    float softcap, const float* __restrict__ s_aux, int rhd, int rhdv,
    int paged, const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, int leftpad, int nB, int nsplits,
    float* __restrict__ o_part, float* __restrict__ lse_part) {
    constexpr int WST=2, MMA_W=8, GW0=9;
    constexpr int NKT_QK=HD/16, NKT_PV=WTN/16;
    constexpr int SC[2]={0,128}, OC[2]={256,256+HDV}, PC[2]={0,128};   // S0/S1, O0/O1, P aliased in S
    extern __shared__ __nv_bfloat16 draw[];
    uint32_t rp=smem_u32(draw); uint32_t pad=((rp+1023u)&~1023u)-rp;
    T* dyn=(T*)((char*)draw+pad);
    T* sQ=dyn; T* sK[WST]; T* sV[WST];
    { T* p=sQ+WTM*HD; for(int s=0;s<WST;s++){ sK[s]=p; p+=WTN*HD; } for(int s=0;s<WST;s++){ sV[s]=p; p+=WTN*HDV; } }
    __shared__ __align__(16) Bars bars; __shared__ __align__(8) uint32_t s_base;
    __shared__ int prow_c[WST][WTN];
    __shared__ float sM1[WTM], sL1[WTM];   // stream1 (w4-7) row stats -> stream0 (w0-3) for the final combine
    const int tid=threadIdx.x, warp=tid>>5;
    const int mb=blockIdx.x, h=blockIdx.y;
    const int b=blockIdx.z%nB, split_idx=blockIdx.z/nB, hk=h/rep;
    const int q0=cu_q[b], seqlen_q=cu_q[b+1]-q0;
    const int seqlen_k = paged ? seqused_k[b] : (cu_k[b+1]-cu_k[b]);
    const int k0 = paged ? 0 : cu_k[b];
    const int row0=mb*WTM;
    const float LOG2E=1.4426950408889634f, INV_LOG2E=0.6931471805599453f, inv_sc=(softcap>0.f)?1.f/softcap:0.f;
    if (row0>=seqlen_q) return;
    int qp_max=seqlen_k-seqlen_q+row0+(WTM-1), qp_min=seqlen_k-seqlen_q+row0;
    int n_lo=0, n_hi=(seqlen_k+WTN-1)/WTN;
    if(is_causal) n_hi=min(n_hi,(qp_max)/WTN+1);
    if(win_r>=0)  n_hi=min(n_hi,(qp_max+win_r)/WTN+1);
    if(win_l>=0){ int lo=qp_min-win_l; n_lo=(lo>0)?lo/WTN:0; }
    n_hi=max(n_hi,n_lo);
    if (nsplits>1){ int on_lo=n_lo,on_hi=n_hi,tot=on_hi-on_lo,per=(tot+nsplits-1)/nsplits;
        n_lo=min(on_hi,on_lo+split_idx*per); n_hi=min(on_hi,on_lo+(split_idx+1)*per); }
    const int NBk=n_hi-n_lo;

    if(warp<8) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n" ::: "memory");   // 2 softmax WGs (2-pass, no held f[128])
    else       asm volatile("setmaxnreg.dec.sync.aligned.u32 128;\n" ::: "memory");   // MMA + gather producers (no spill @128)
    const float qsc=(softcap>0.f)?scale:scale*LOG2E;
    for(int i=tid;i<WTM*HD;i+=blockDim.x){ int r=i/HD,c=i%HD; int qr=row0+r;
        sQ[koff(r,c)] = (qr<seqlen_q && c<rhd) ? from_f<T>(to_f<T>(q[(size_t)(q0+qr)*hq*rhd+(size_t)h*rhd+c])*qsc) : from_f<T>(0.f); }
    if(warp==MMA_W){ tmem_alloc(smem_u32(&s_base),512); tmem_relinquish(); }
    if(tid==0){ for(int s=0;s<WST;s++){ mbar_init(smem_u32(&bars.kv_full[s]),1); mbar_init(smem_u32(&bars.kv_empty[s]),1);}
                for(int j=0;j<2;j++){ mbar_init(smem_u32(&bars.s_full[j]),1); mbar_init(smem_u32(&bars.s_empty[j]),1);
                                      mbar_init(smem_u32(&bars.p_full[j]),128); mbar_init(smem_u32(&bars.o_ready[j]),1);} }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    const uint32_t base=s_base;
    uint32_t KVF[WST],KVE[WST]; for(int s=0;s<WST;s++){ KVF[s]=smem_u32(&bars.kv_full[s]); KVE[s]=smem_u32(&bars.kv_empty[s]); }
    uint32_t SF[2]={smem_u32(&bars.s_full[0]),smem_u32(&bars.s_full[1])}, SE[2]={smem_u32(&bars.s_empty[0]),smem_u32(&bars.s_empty[1])};
    uint32_t PF[2]={smem_u32(&bars.p_full[0]),smem_u32(&bars.p_full[1])}, OR[2]={smem_u32(&bars.o_ready[0]),smem_u32(&bars.o_ready[1])};
    const uint32_t QKI=qk_id<T,WTN>(), PVI=pv_id<T,HDV>();

    if (warp>=GW0) {                         // gather (w9-15): manual paged gather via cp.async (verbatim from ws_paged)
        const int lid=tid-GW0*32, LN=(int)blockDim.x-GW0*32; int pe[WST]={0};
        for(int i=0;i<NBk;++i){ int s=i%WST; mbar_wait(KVE[s],pe[s]); pe[s]^=1;
            int key0=(n_lo+i)*WTN;
            for(int r=lid;r<WTN;r+=LN){ int kk=key0+r;
                prow_c[s][r] = (kk<seqlen_k) ? (paged ? (page_table[b*pt_stride+kk/block_size]*block_size + kk%block_size) : (k0+kk)) : -1; }
            asm volatile("bar.sync 3, %0;"::"r"(LN));
            for(int idx=lid;idx<WTN*(HD/8);idx+=LN){ int r=idx/(HD/8), c=(idx%(HD/8))*8; int prow=prow_c[s][r];
                if(prow>=0 && c<rhd) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sK[s][koff(r,c)])),"l"(&k[(size_t)prow*hkv*rhd+(size_t)hk*rhd+c]));
                else *(uint4*)&sK[s][koff(r,c)]=make_uint4(0,0,0,0); }
            for(int idx=lid;idx<WTN*(HDV/8);idx+=LN){ int key=idx/(HDV/8), dv=(idx%(HDV/8))*8; int prow=prow_c[s][key];
                if(prow>=0 && dv<rhdv) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"(smem_u32(&sV[s][voff(key,dv)])),"l"(&v[(size_t)prow*hkv*rhdv+(size_t)hk*rhdv+dv]));
                else *(uint4*)&sV[s][voff(key,dv)]=make_uint4(0,0,0,0); }
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n");
            asm volatile("fence.proxy.async.shared::cta;");
            asm volatile("bar.sync 3, %0;"::"r"(LN));
            if(lid==0) mbar_arrive(KVF[s]);
        }
    } else if (warp==MMA_W) {                // MMA: qk(i)->S[i&1], pv(i)->O[i&1] (per-stream accumulate)
        int pf[WST]={0}, pse[2]={0,0}, ppf[2]={0,0};
        auto qk=[&](int i){ int j=i&1,s=i%WST; uint64_t qd0=make_desc(smem_u32(sQ),16,1024,2),kd0=make_desc(smem_u32(sK[s]),16,1024,2);
            for(int kb=0;kb<NKT_QK;++kb){ uint64_t qd=desc_add(qd0,qk_doff(kb)),kd=desc_add(kd0,qk_doff(kb)); uint32_t pr=(kb>0)?1u:0u;
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p; }"
                    ::"r"(base+SC[j]),"l"(qd),"l"(kd),"r"(pr),"r"(QKI)); }
            mma_commit(SF[j]); };
        auto pv=[&](int i){ int j=i&1,s=i%WST; uint64_t vd0=make_desc(smem_u32(sV[s]),16384,1024,2);
            for(int kb=0;kb<NKT_PV;++kb){ uint64_t vd=desc_add(vd0,v_doff(kb)); uint32_t pr=((i>=2)||kb>0)?1u:0u; uint32_t pa=base+PC[j]+(uint32_t)(kb*8);
                asm volatile("{ .reg .pred l,p; elect.sync _|l,-1; setp.ne.b32 p,%3,0;\n @l tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p; }"
                    ::"r"(base+OC[j]),"r"(pa),"l"(vd),"r"(pr),"r"(PVI)); }
            mma_commit(OR[j]); };
        if(NBk>0){
        mbar_wait(KVF[0],pf[0]); pf[0]^=1; mbar_wait(SE[0],pse[0]); pse[0]^=1; qk(0);
        for(int i=0;i<NBk;++i){ int s=i%WST;
            if(i+1<NBk){ int ns=(i+1)%WST,nj=(i+1)&1; mbar_wait(KVF[ns],pf[ns]); pf[ns]^=1; mbar_wait(SE[nj],pse[nj]); pse[nj]^=1; qk(i+1); }
            mbar_wait(PF[i&1],ppf[i&1]); ppf[i&1]^=1; pv(i); mma_commit(KVE[s]);
        } }
    } else {                                 // softmax: w0-3 = stream0 (even blocks), w4-7 = stream1 (odd blocks)
        const int j = (warp<4)?0:1;          // stream index
        const int lane = tid&127;            // 0..127 -> query row
        const int barid = (warp<4)?1:2;      // per-WG named barrier
        int psf=0, por=0;
        float m_i=-INFINITY, l_i=0.f;
        if(warp<4){ if(lane==0){ mbar_arrive(SE[0]); for(int s=0;s<WST;s++) mbar_arrive(KVE[s]); } }
        else      { if(lane==0)  mbar_arrive(SE[1]); }
        const int qrow=row0+lane, qhd=h, qp=seqlen_k-seqlen_q+qrow;
        int key0off=0;
        auto logit=[&](float sv,int c)->float{ float lg=(softcap>0.f)?(softcap*tanhf(sv*inv_sc)*LOG2E):sv;
            int kk=key0off+c; bool ok=(kk<seqlen_k)&&(kk>=leftpad);
            if(is_causal&&kk>qp) ok=false; if(win_l>=0&&kk<qp-win_l) ok=false; if(win_r>=0&&kk>qp+win_r) ok=false; return ok?lg:-INFINITY; };
        int done=0;                          // # blocks this stream has processed (for O-rescale gating)
        for(int i=j;i<NBk;i+=2){ key0off=(n_lo+i)*WTN;
            mbar_wait(SF[j],psf); psf^=1; tmem_fence();
            const int kend=key0off+WTN-1;
            bool full=(kend<seqlen_k)&&(softcap<=0.f)&&(key0off>=leftpad);
            if(is_causal && kend>qp_min) full=false;
            if(win_r>=0 && kend>qp_min+win_r) full=false;
            if(win_l>=0 && key0off<qp_max-win_l) full=false;
            float m_new, corr, ps=0.f;
            {   // 2-pass softmax (no held f[128] -> 128 regs -> producers keep 128 regs, MMA spill-free).
                // full blocks skip per-key masking (branchless logit -> lg=sv). Two WGs hide the 2x tmem reads.
                float rmax=-INFINITY;
                for(int g=0;g<4;++g){ uint32_t vv[32]; tmem_ld32(base+SC[j]+((uint32_t)lane<<16)+g*32, vv);
                    #pragma unroll
                    for(int t=0;t<32;++t){ float sv=__uint_as_float(vv[t]); rmax=fmaxf(rmax, full?sv:logit(sv,g*32+t)); } }
                tmem_wait_ld();
                m_new=fmaxf(m_i,rmax); corr=(m_i==-INFINITY||m_new==-INFINITY)?1.f:ex2f(m_i-m_new);
                for(int g=0;g<4;++g){ uint32_t vv[32]; tmem_ld32(base+SC[j]+((uint32_t)lane<<16)+g*32, vv); uint32_t pk[16];
                    #pragma unroll
                    for(int t=0;t<16;++t){ float a=__uint_as_float(vv[2*t]), c=__uint_as_float(vv[2*t+1]);
                        float l0=full?a:logit(a,g*32+2*t), l1=full?c:logit(c,g*32+2*t+1);
                        float p0=(l0==-INFINITY)?0.f:ex2f(l0-m_new), p1=(l1==-INFINITY)?0.f:ex2f(l1-m_new); ps+=p0+p1; pk[t]=pack16<T>(p0,p1); }
                    tmem_st16(base+PC[j]+((uint32_t)lane<<16)+g*16, pk); }
            }
            l_i=l_i*corr+ps; m_i=m_new;
            asm volatile("bar.sync %0, 128;"::"r"(barid));
            if(lane==0) mbar_arrive(SE[j]);
            if(done>0){ mbar_wait(OR[j],por); por^=1;   // rescale O[j] (prev PV of this stream) by corr before PV[i]
                if(!__all_sync(0xffffffffu,corr==1.0f)) tmem_o_rescale<HDV>(base+OC[j]+((uint32_t)lane<<16), corr); }
            tmem_wait_st(); mbar_arrive(PF[j]);
            done++;
        }
        if(done>0){ mbar_wait(OR[j],por); por^=1; } tmem_fence();   // final PV of this stream done
        float mS=(done>0)?m_i:-INFINITY, lS=(done>0)?l_i:0.f;
        if(j==1){ sM1[lane]=mS; sL1[lane]=lS; }
        asm volatile("bar.sync 4, 256;");    // sync the two softmax WGs: stream1 O1/stats visible to stream0
        if(j==0){                            // only stream0 (w0-3) combines O0+O1 + writes output (w4-7 -> __syncthreads)
        float m0=mS, l0=lS, m1=sM1[lane], l1=sL1[lane];
        float m=fmaxf(m0,m1);
        float a0=(m0==-INFINITY)?0.f:ex2f(m0-m), a1=(m1==-INFINITY)?0.f:ex2f(m1-m);
        float l=l0*a0 + l1*a1;
        const bool h0=(l0>0.f), h1=(l1>0.f);
        const int qr=qrow; const bool ow=(qr<seqlen_q);
        float mfin=m, lfin=l, osc=1.f;
        if(nsplits<=1 && s_aux!=nullptr){ float sk2=s_aux[qhd]*LOG2E; float mn=fmaxf(m,sk2); float c2=(m==-INFINITY)?0.f:ex2f(m-mn); lfin=l*c2+ex2f(sk2-mn); osc=c2; mfin=mn; }
        const float sc_out = (nsplits>1) ? ((l>0.f)?1.f/l:0.f) : ((lfin>0.f)?osc/lfin:0.f);   // per-elem O scale
        const int TQ=cu_q[nB];
        // O = O0*a0 + O1*a1, read in 32-col chunks (of[128]+o1[128] would blow the 176-reg budget)
        for(int g=0; g<HDV; g+=32){ uint32_t o0g[32], o1g[32];
            if(h0){ LD32_INTO(base+OC[0]+((uint32_t)lane<<16)+g, o0g, 0); }
            if(h1){ LD32_INTO(base+OC[1]+((uint32_t)lane<<16)+g, o1g, 0); }
            tmem_wait_ld();
            #pragma unroll
            for(int t=0;t<32;++t){ int d=g+t; if(!ow || d>=rhdv) continue;
                float oo=(h0?__uint_as_float(o0g[t])*a0:0.f)+(h1?__uint_as_float(o1g[t])*a1:0.f);
                if(nsplits>1) o_part[(((size_t)split_idx*TQ+(q0+qr))*hq+qhd)*rhdv+d]=oo*sc_out;
                else          out[(size_t)(q0+qr)*hq*rhdv+(size_t)qhd*rhdv+d]=from_f<T>(oo*sc_out); }
        }
        if(ow){ if(nsplits>1) lse_part[((size_t)split_idx*hq+qhd)*TQ+(q0+qr)] = (l>0.f)?(logf(l)+m*INV_LOG2E):-INFINITY;
                else          lse[(size_t)qhd*cu_q[nB]+(q0+qr)] = (lfin>0.f)?(logf(lfin)+mfin*INV_LOG2E):-INFINITY; }
        }
    }
    __syncthreads();
    if(warp==MMA_W) tmem_dealloc(base,512);
}
}  // namespace ws

// Warp-specialized fast-path launcher (non-paged contiguous K/V). Returns true if it handled the config.
// 2-SM (cta_group::2) cluster prefill launcher. Gated behind FA4_2SM. A 2-CTA cluster jointly computes
// one 256-row Q-tile (rank0 rows[0:128], rank1 rows[128:256]); grid.x is rounded up to a multiple of 2
// so consecutive blocks pair into clusters. Correctness-first SERIAL variant (no WS pipeline yet).
// Returns true if it handled the config, false to fall through to the existing launchers.
static bool launch_2sm(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
                       at::Tensor cu_q, at::Tensor cu_k, int hq, int hkv, float scale,
                       bool is_causal, int win_l, int win_r, float softcap, const float* s_aux, int hd, int hdv) {
    const char* e2=getenv("FA4_2SM"); if(e2==nullptr) return false;
    int mode=atoi(e2);                                                // 1 = serial (correctness), 2 = WS pipeline
    int hdp=(hd+63)/64*64, hdvp=(hdv+63)/64*64;
    int rep=hq/hkv, b=(int)cu_q.size(0)-1;
    auto cu_q_cpu=cu_q.to(at::kCPU); const int* cq=cu_q_cpu.data_ptr<int>();
    constexpr int TM2=128;
    int max_mq=0; for(int i=0;i<b;++i){ int s=cq[i+1]-cq[i]; max_mq=std::max(max_mq,(s+TM2-1)/TM2); }
    if (max_mq==0) return false;
    int gx=((max_mq+1)/2)*2;                                          // grid.x multiple of cluster dim (=2)
    auto stream=at::cuda::getCurrentCUDAStream();
    bool bf16=(q.scalar_type()==at::kBFloat16);

    if (mode>=2) {   // WS pipeline: TMA ring + overlapped QK/softmax/PV. hd128 only (V 64-col split needs 64-wide box).
        if (!(hdp==128&&hdvp==128)) return false;
        dim3 grid(gx,hq,b), block(512);
        size_t smem=(size_t)(ws::WTM*hdp + ws::WST*(hdp/64)*8192 + ws::WST*8192)*2 + 1024;
#define WS2SM_LAUNCH(T,HDP,HDVP) do { \
            CUtensorMap mKh=ws::ws_map_rows<T>(k, ws::WTN/2), mVh=ws::ws_map<T>(v); \
            cudaFuncSetAttribute(ws::fwd_kernel_ws_2sm<T,HDP,HDVP>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem); \
            ws::fwd_kernel_ws_2sm<T,HDP,HDVP><<<grid,block,smem,stream>>>(mKh,mVh,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
                cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv); } while(0)
        if(bf16) WS2SM_LAUNCH(__nv_bfloat16,128,128); else WS2SM_LAUNCH(__half,128,128);
#undef WS2SM_LAUNCH
        return true;
    }

    if (!((hdp==128&&hdvp==128)||(hdp==64&&hdvp==64))) return false;  // serial probe-validated envelope
    dim3 grid(gx,hq,b), block(128);
    int hdvh=hdvp/2;
    size_t smem=(size_t)(TM2*hdp + (hdp/64)*8192 + ((hdvh+63)/64)*8192)*2 + 1024;  // sQ + sK(koff blocks) + sV(voff blocks)
#define C2_LAUNCH(T,HDP,HDVP) do { \
        cudaFuncSetAttribute(fwd_kernel_2sm<T,HDP,HDVP>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem); \
        fwd_kernel_2sm<T,HDP,HDVP><<<grid,block,smem,stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
            (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),cu_k.data_ptr<int>(), \
            hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv); } while(0)
    if      (hdp==128&&hdvp==128){ if(bf16) C2_LAUNCH(__nv_bfloat16,128,128); else C2_LAUNCH(__half,128,128); }
    else if (hdp==64 &&hdvp==64 ){ if(bf16) C2_LAUNCH(__nv_bfloat16,64,64);   else C2_LAUNCH(__half,64,64); }
    else return false;
#undef C2_LAUNCH
    return true;
}

// ======================== DE-PAGE: paged K/V -> contiguous varlen K/V ========================
// A LARGE-Q (prefill) attention call reads every KV row once per query tile that can see it:
// ~rep * seqlen_q / (2*WTM) times under causal masking, i.e. 256x for the llama8b 16k prefill.
// The per-tile page walk (tile::gather4 / cp.async gather) therefore gets paid ~256 times, and it
// is the measured limiter of fwd_kernel_fa4_paged (PC sampling: ~61% of samples are mbarrier spins,
// producer out of ring slots, and the smem cap will not hold a 3rd K/V pair -- see
// docs/blackwell_gap_analysis.md). Materializing the cache ONCE into the contiguous [total,hkv,hd]
// layout costs 2 passes over the *unique* KV bytes (1 read + 1 write) and then lets the shape run on
// the fully TMA-fed persistent dense kernel (fwd_kernel_fa4), which is 0.84-1.19x of FA4 on the same
// math instead of 0.57-0.71x. At 256x re-read the copy is <1% of the attention's own KV traffic.
//
// Fully general over page_size / page order / ragged seqlens: it is a real gather, not a contiguity
// assumption. Only the two `at::empty` scratch buffers are new state.

// Exclusive scan of seqused_k -> cu_k[0..B]. B is O(batch) (<=few thousand), and this runs once per
// call on the attention stream, so a single-thread running sum is cheaper than a real scan launch.
__global__ void depage_cuk(const int* __restrict__ seqused_k, int* __restrict__ cu_k, int B) {
  if (threadIdx.x != 0) return;
  int acc = 0; cu_k[0] = 0;
  for (int i = 0; i < B; ++i) { acc += seqused_k[i]; cu_k[i + 1] = acc; }
}

// One uint4 (16B) per thread. A token's whole [hkv, hd] slab is contiguous in BOTH layouts (hd is
// innermost and hkv is the middle dim of the paged pool too), so this is a pure strided block copy.
template<typename T>
__global__ void __launch_bounds__(256) depage_kv(
    const T* __restrict__ kp, const T* __restrict__ vp, T* __restrict__ kc, T* __restrict__ vc,
    const int* __restrict__ page_table, int pt_stride, int block_size,
    const int* __restrict__ seqused_k, const int* __restrict__ cu_k,
    int nvk, int nvv, int max_sk) {
  const int b = blockIdx.y;
  const int slen = seqused_k ? seqused_k[b] : max_sk;
  const int nv = nvk > nvv ? nvk : nvv;
  const long idx = (long)blockIdx.x * 256 + threadIdx.x;
  if (idx >= (long)max_sk * nv) return;
  const int t = (int)(idx / nv), j = (int)(idx - (long)t * nv);
  if (t >= slen) return;
  const int pb = t / block_size;
  const long src = (long)page_table[(long)b * pt_stride + pb] * block_size + (t - pb * block_size);
  const long dst = (long)cu_k[b] + t;
  if (j < nvk) reinterpret_cast<uint4*>(kc)[dst * nvk + j] = reinterpret_cast<const uint4*>(kp)[src * nvk + j];
  if (j < nvv) reinterpret_cast<uint4*>(vc)[dst * nvv + j] = reinterpret_cast<const uint4*>(vp)[src * nvv + j];
}

// Cached de-page scratch. A per-call `at::empty` is a caching-allocator round trip, and a real
// cudaMalloc whenever the surrounding process has fragmented the pool -- ~100us on a 125us kernel,
// which is what made pf_s2k look like a regression in the multi-config sweep while a clean
// single-config run showed the win. Grow-only, heap-leaked like persistent_ctr (dodges static
// destructor ordering against the CUDA context).
static at::Tensor depage_pool(int slot, int64_t nelem, const at::TensorOptions& opts) {
  static at::Tensor* cache = new at::Tensor[3];
  at::Tensor& t = cache[slot];
  if (!t.defined() || t.device() != opts.device() || t.dtype() != opts.dtype() || t.numel() < nelem)
    t = at::empty({t.defined() ? std::max<int64_t>(nelem, t.numel()) : nelem}, opts);
  return t.narrow(0, 0, nelem);
}

// Persistent-scheduler tile counter. at::full({1},ctas) per call costs ~7us net (caching-allocator
// alloc + FillFunc). Cache the [1]-int tensor across calls (heap-leaked to dodge shutdown-order issues)
// and only re-fill it -> removes the per-call allocation. fill_ runs on the current stream, so it stays
// ordered before the kernel that atomic-increments it. (eval is single-stream sequential; safe.)
static at::Tensor& persistent_ctr(int ctas, const at::Tensor& q, bool self_reset){
    static at::Tensor* cached = new at::Tensor();
    static int  last_ctas  = -1;      // ctas value the counter is currently known to hold
    static bool last_reset = false;   // did the previous call's kernel device-side reset it?
    bool realloc = (!cached->defined() || cached->device()!=q.device());
    if(realloc){ *cached = at::empty({1}, q.options().dtype(at::kInt)); last_ctas=-1; last_reset=false; }
    // Skip the fill_ launch iff the previous fa4 call self-reset the counter to exactly this ctas
    // (same shape). Same-stream program order makes that device store visible before our next atomic
    // read, so no host re-fill is needed. Any shape change, first use, realloc, or a non-self-reset
    // predecessor (e.g. the cont path) forces the fill_ to re-establish the invariant.
    bool can_skip = self_reset && last_reset && !realloc && (last_ctas==ctas);
    if(!can_skip) cached->fill_(ctas);
    last_ctas = ctas; last_reset = self_reset;
    return *cached;
}
static bool launch_ws(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
                      at::Tensor cu_q, at::Tensor cu_k, int hq, int hkv, float scale,
                      bool is_causal, int win_l, int win_r, float softcap, const float* s_aux, int hd, int hdv,
                      int msq, int msk) {
    int rep=hq/hkv, b=(int)cu_q.size(0)-1;
    int hdp=(hd+63)/64*64, hdvp=(hdv+63)/64*64;
    // HOST FAST-PATH: max_seqlen_q/k (already op args) give max_mq/max_sk directly, avoiding the two
    // Device->Pageable cu_q/cu_k.to(kCPU) copies. Pageable DtoH is SYNCHRONOUS -- it drains+syncs the
    // stream and blocks the CPU before it can launch, and prevents consecutive calls from pipelining
    // (~35us/call measured). Because ceil() is monotonic, ceil(max_seqlen_q/WTM) == max_i ceil(s_i/WTM),
    // so this is exact for varlen too. Fall back to the CPU reduction only when the args are absent.
    const int* cq=nullptr; const int* ck=nullptr;
    at::Tensor cu_q_cpu, cu_k_cpu;
    auto ensure_cpu=[&](){ if(!cq){ cu_q_cpu=cu_q.to(at::kCPU); cq=cu_q_cpu.data_ptr<int>();
                                    cu_k_cpu=cu_k.to(at::kCPU); ck=cu_k_cpu.data_ptr<int>(); } };
    int max_mq, max_sk;
    if(msq>0 && msk>0){ max_mq=(msq+ws::WTM-1)/ws::WTM; max_sk=msk; }
    else { ensure_cpu();
           max_mq=0; for(int i=0;i<b;++i){ int s=cq[i+1]-cq[i]; max_mq=std::max(max_mq,(s+ws::WTM-1)/ws::WTM); }
           max_sk=0; for(int i=0;i<b;++i) max_sk=std::max(max_sk, ck[i+1]-ck[i]); }
    // Correction-WG (offload O-rescale to w12-15) is now OFF BY DEFAULT: it became a NET LOSS on every
    // dense hd128 config once the softmax went spill-free (measured 2026, CORR ON vs OFF: large4k
    // 0.548 vs 0.595, seq8k 0.626 vs 0.702, seq16k 0.654 vs 0.733). The faster softmax makes the CS/CO
    // cross-WG handshake + split_P choreography cost more than the O-rescale it offloads; the non-CORR
    // softmax rescales O inline (dead-f[128] window, no extra regs) more cheaply. Kept behind env for
    // experimentation: FA4_CORR=1 forces on; FA4_CORR_TH=N enables when max_sk>=N.
    bool corr=false;
    if(const char* e=getenv("FA4_CORR_TH")) corr=(max_sk>=atoi(e));
    else if(getenv("FA4_CORR")) corr=true;
    dim3 grid2((max_mq+1)/2,hq,b), grid1(max_mq,hq,b), block(512);  // grid2: q_stage=2 (2 Q-tiles/CTA)
    size_t smem2=(size_t)(2*ws::WTM*hdp + ws::WST*ws::WTN*hdp + ws::WST*ws::WTN*hdvp)*2 + 1024;
    size_t smem1=(size_t)(  ws::WTM*hdp + ws::WST*ws::WTN*hdp + ws::WST*ws::WTN*hdvp)*2 + 1024;
    // hd192 q_stage=2 with V-ring depth 1 (K depth WST): 2*Q(96K)+2*K(96K)+1*V(32K)=224K fits 227K opt-in,
    // whereas full WST=2 V (256K) overflows. Q_stage=2 gives the 2-tile softmax (8 warps) the q_stage=1
    // WS1 path lacks — the 4.7x hd192 gap is the single-WG latency stall (ncu: barrier 58%, compute 26%).
    size_t smem2v1=(size_t)(2*ws::WTM*hdp + ws::WST*ws::WTN*hdp + 1*ws::WTN*hdvp)*2 + 1024;
    auto stream=at::cuda::getCurrentCUDAStream();
    bool bf16=(q.scalar_type()==at::kBFloat16);
    // TIER0 gate NSM (device SM count) for the grid-fill amortization test below. Cheap driver query.
    int NSM_gate=148; { int dev=0; if(cudaGetDevice(&dev)==cudaSuccess){ int n=0; if(cudaDeviceGetAttribute(&n,cudaDevAttrMultiProcessorCount,dev)==cudaSuccess&&n>0) NSM_gate=n; } }
    // TIER0 grid-fill / amortization gate (replaces the STALE max_mq>=4). The persistent fa4 kernel needs
    // ~>=1 wave of Q-tile work to amortize its per-tile handshake; below that it loses to the non-persistent
    // ws2 inline kernel. b*hq*max_mq counts 128-row Q-tiles: small REGULAR dense (seq256: 9*16*2=288) clears
    // one wave (NSM~148) -> fa4; sub-wave ragged VARLEN (24-72 tiles, ~0.24 waves) does NOT -> stays on ws2
    // (correct: persistent fa4 has no amortization headroom on varlen; ws2+the tier0 cont-guard beats FA4 there).
    bool grid_fill = ((long)b * hq * max_mq >= (long)NSM_gate);
    // A UNIFORM-seqlen batch is dense (not sub-wave ragged varlen): route it to fa4 regardless of grid
    // size. total_q == b*max_seqlen_q iff every seq == max (uniform); ragged varlen has total_q < b*msq
    // and STAYS on ws2 (its tier0 cont-guard win). No D2H: total_q=q.size(0), msq is the op arg. This
    // pulls the degenerate small-grid dense shapes (q>=k / q>k causal) onto fa4 -- correct on them
    // (fixes the ws2 q512k256 wrong-output + q799k3/q640k128 1-K-block/multi-q-tile HANGs); the fa4
    // path handles the class ws2 deadlocks on (cf. deg_q239k1 routes to fa4 and passes).
    bool dense_uniform = (msq>0) && ((long)q.size(0) == (long)b * msq);
    grid_fill = grid_fill || dense_uniform;
    // FA4-faithful rebuild (fwd_kernel_fa4): 256-row Q tile = 2 stacked halves, correction+epilogue WGs.
    // AUTO-ROUTE the configs it implements correctly — hd128 dense bf16/fp16, causal/noncausal/gqa/mqa, NO
    // window/softcap/sink — to fwd_kernel_fa4, the fastest cuda-only path (evolutionary search: ~0.76 short /
    // ~0.81 long geomean vs FA4, beats the ws2 inline kernel's ~0.66). Windowed/softcap/sink/fp8/hd!=128 fall
    // through to the ws2 path below (which handles them). FA4_NOFA4=1 disables; FA4_REBUILD=1 forces (also hd64).
    // TIER0: hd64 added — fwd_kernel_fa4<T,64,64> is already instantiated, spill-free, err_ratio 1.000, cycle-parity.
    bool fa4_auto = ((hd==128&&hdv==128)||(hd==64&&hdv==64)) && (bf16 || q.scalar_type()==at::kHalf)
                    && (win_l<0) && (win_r<0) && (softcap<=0.f) && (s_aux==nullptr)
                    && grid_fill;   // see grid_fill above: >=1 wave of Q-tiles amortizes the persistent kernel; small/ragged/varlen/decode stay on ws2. hd must be TRUE 128 or 64 (hd96 pads to hdp=128 but fa4's 128-dim template needs Q-col zeroing — kept on ws2 here).
    // hd192_dv128 dense: route to the DECOUPLED-ring FA4 kernel (fwd_kernel_fa4dk, K depth-2/V depth-1) which
    // brings the persistent+LPT+cross-tile-overlap FA4 structure to hd192 (ws2-hd192 was 0.584). Same gates as
    // fa4_auto but on hd==192&&hdv==128. Kept SEPARATE from fa4_auto so hd128 stays byte-identical.
    bool fa4_192 = (hd==192&&hdv==128) && (bf16 || q.scalar_type()==at::kHalf)
                   && (win_l<0) && (win_r<0) && (softcap<=0.f) && (s_aux==nullptr) && (max_mq>=4);
    // GRAFT ROUTES onto fwd_kernel_fa4 (dense hd128 window/sink + hd96 via PADQ). All gated by grid_fill so
    // sub-wave VARLEN window/sink stay on ws2 (they have no persistent-kernel amortization headroom). softcap
    // stays excluded (not ported). Mutually exclusive with fa4_auto's (win<0 && win>0(sic) && s_aux==null).
    bool _fa4h = (bf16 || q.scalar_type()==at::kHalf);
    bool fa4_win  = (hd==128&&hdv==128) && _fa4h && (softcap<=0.f) && (s_aux==nullptr) && (win_l>=0 || win_r>=0) && grid_fill;
    bool fa4_sink = (hd==128&&hdv==128) && _fa4h && (softcap<=0.f) && (s_aux!=nullptr) && (win_l<0) && (win_r<0) && grid_fill;
    bool fa4_padq = (hd==96 &&hdv==96 ) && _fa4h && (softcap<=0.f) && (s_aux==nullptr) && (win_l<0) && (win_r<0) && grid_fill;
    if(getenv("FA4_NOFA4")) { fa4_auto=false; fa4_192=false; fa4_win=false; fa4_sink=false; fa4_padq=false; }
    if(getenv("FA4_NO192")) fa4_192=false;   // force hd192 back to the ws2 q_stage=2 path (0.584 baseline)
    if(getenv("FA4_FORCE192")) fa4_192=(hd==192&&hdv==128)&&(bf16||q.scalar_type()==at::kHalf);  // test: route hd192 to fa4dk regardless of max_mq
    if(getenv("FA4_REBUILD")) fa4_auto=true;
    if(fa4_192){
        // hd192 uses essentially the ENTIRE 232448B opt-in smem: data = 2Q@192 + 2K@192 + 1V@128 = 229376B
        // (K depth-2 for QK prefetch, V depth-1 to fit) + the in-kernel 1024-align pad + ~2.3KB static (Bars +
        // sScale). The dynamic-smem base is placed after the static smem at a NON-1024-aligned offset, so the
        // kernel's align pad can be up to 768B. To fit exactly, size the dynamic request as (CAP - staticSmem):
        // that leaves the kernel's pad+data == CAP-static (since ceil(static/1024)*1024 + data == CAP here), so
        // static+dynamic == CAP and the pad'd V-ring writes land inside the allocation. Robust to nvcc static drift.
        int gxp=(max_mq+1)/2; int total_tiles=gxp*hq*b;
        int NSM=148; { int dev=0; if(cudaGetDevice(&dev)==cudaSuccess){ int n=0; if(cudaDeviceGetAttribute(&n,cudaDevAttrMultiProcessorCount,dev)==cudaSuccess&&n>0) NSM=n; } }
        int ctas=std::min(total_tiles,NSM); if(ctas<1) ctas=1;
        auto& ctr=persistent_ctr(ctas,q,true);
        dim3 gridf(ctas);
        const int SMEM_CAP=232448;
#define FA4DK_LAUNCH(T) do { \
        cudaFuncAttributes _fa; cudaFuncGetAttributes(&_fa,(const void*)(ws::fwd_kernel_fa4dk<T,192,128,2,1>)); \
        size_t smemf=(size_t)SMEM_CAP - _fa.sharedSizeBytes; \
        CUtensorMap mK=ws::ws_map<T>(k), mV=ws::ws_map<T>(v), mQ=ws::ws_map<T>(q); \
        cudaFuncSetAttribute((ws::fwd_kernel_fa4dk<T,192,128,2,1>), cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemf); \
        (ws::fwd_kernel_fa4dk<T,192,128,2,1>)<<<gridf,block,smemf,stream>>>(mK,mV,mQ,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
            cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
            ctr.data_ptr<int>(),total_tiles,gxp,b,dense_uniform?0:1); } while(0)
        if(bf16) FA4DK_LAUNCH(__nv_bfloat16); else FA4DK_LAUNCH(__half);
#undef FA4DK_LAUNCH
        return true;
    }
    if(fa4_auto||fa4_win||fa4_sink||fa4_padq){
        size_t smemf=(size_t)(2*ws::WTM*hdp + ws::FA4KVS*ws::WTN*hdp + ws::FA4KVS*ws::WTN*hdvp)*2 + 1024;  // sO aliased into sQ (no WTM*hdvp term)
        // PERSISTENT + CONTINUOUS-OVERLAP: NSM CTAs, atomic LPT tile grab, TMA-Q, cross-tile mbar overlap.
        int gxp=(max_mq+1)/2; int total_tiles=gxp*hq*b;
        int NSM=148; { int dev=0; if(cudaGetDevice(&dev)==cudaSuccess){ int n=0; if(cudaDeviceGetAttribute(&n,cudaDevAttrMultiProcessorCount,dev)==cudaSuccess&&n>0) NSM=n; } }
        int ctas=std::min(total_tiles,NSM); if(ctas<1) ctas=1;
        auto& ctr=persistent_ctr(ctas,q,true);
        dim3 gridf(ctas);
#define FA4R_LAUNCH(T,HDP,HDVP,WIN,SNK,PDQ) do { \
        CUtensorMap mK=ws::ws_map<T>(k), mV=ws::ws_map<T>(v), mQ=ws::ws_map<T>(q); \
        cudaFuncSetAttribute(ws::fwd_kernel_fa4<T,HDP,HDVP,WIN,SNK,PDQ>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemf); \
        ws::fwd_kernel_fa4<T,HDP,HDVP,WIN,SNK,PDQ><<<gridf,block,smemf,stream>>>(mK,mV,mQ,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
            cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
            ctr.data_ptr<int>(),total_tiles,gxp,b,dense_uniform?0:1,nullptr,0,0,nullptr,0,0,nullptr,nullptr); } while(0)
        if(fa4_win){ if(bf16) FA4R_LAUNCH(__nv_bfloat16,128,128,true,false,false); else FA4R_LAUNCH(__half,128,128,true,false,false); }
        else if(fa4_sink){ if(bf16) FA4R_LAUNCH(__nv_bfloat16,128,128,false,true,false); else FA4R_LAUNCH(__half,128,128,false,true,false); }
        else if(fa4_padq){ if(bf16) FA4R_LAUNCH(__nv_bfloat16,128,128,false,false,true); else FA4R_LAUNCH(__half,128,128,false,false,true); }
        else if(hdp==128&&hdvp==128){ if(bf16) FA4R_LAUNCH(__nv_bfloat16,128,128,false,false,false); else FA4R_LAUNCH(__half,128,128,false,false,false); }
        else if(hdp==64&&hdvp==64){ if(bf16) FA4R_LAUNCH(__nv_bfloat16,64,64,false,false,false); else FA4R_LAUNCH(__half,64,64,false,false,false); }
        else return false;
#undef FA4R_LAUNCH
        return true;
    }
    // CONTINUOUS-OVERLAP persistent: dedicated Q-producer warp + TMA-Q + mbar-only cross-tile handshakes
    // (no CTA barrier in the loop) so tile N epilogue overlaps tile N+1 mainloop; LPT-balanced. WINS on
    // long-context causal (measured: seq8k .734 vs .704, seq16k .760 vs .729, large4k .625 vs .599); on
    // short/noncausal the handshake overhead loses, so DEFAULT ON only for dense-uniform causal seqlen>=4096
    // (multiple of the tile, for TMA-Q correctness). FA4_CONT=1 forces on / =0 forces off. hd128/hd64.
    { const char* ce=getenv("FA4_CONT");
      // TIER0 HOST-FIX (fix #4): only pay the 2x synchronous cu_q/cu_k.to(kCPU) D2H (drains+syncs the stream,
      // blocks pipelining ~15-32us/call) when cont could actually engage. cont_auto needs max_sq>=4096; when the
      // max_seqlen_q arg (msq) is present and <4096 (all varlen/short/window/sink/decode configs) it is provably
      // false, so skip the copies entirely. Env FA4_CONT forces the slow-but-general path (msq<=0 also falls back).
      bool maybe_cont = (ce!=nullptr) ? (atoi(ce)!=0) : (is_causal && (msq<=0 || msq>=4096));
      if(maybe_cont){
      ensure_cpu(); int max_sq=0; bool uniform=(b>0); for(int i=0;i<b;++i){ int s=cq[i+1]-cq[i]; max_sq=std::max(max_sq,s); if(s!=cq[1]-cq[0]) uniform=false; }
      bool cont_auto = is_causal && uniform && (max_sq>=4096) && (max_sq%ws::WTM==0);
      bool use_cont = (ce!=nullptr)?(atoi(ce)!=0):cont_auto;
      if(use_cont && !corr && ((hdp==128&&hdvp==128)||(hdp==64&&hdvp==64))){
        int gxp=(max_mq+1)/2; int total_tiles=gxp*hq*b;
        int NSM=148; { int dev=0; if(cudaGetDevice(&dev)==cudaSuccess){ int n=0; if(cudaDeviceGetAttribute(&n,cudaDevAttrMultiProcessorCount,dev)==cudaSuccess&&n>0) NSM=n; } }
        int ctas=std::min(total_tiles,NSM); if(ctas<1) ctas=1;
        auto& ctr=persistent_ctr(ctas,q,false);
        dim3 gridc(ctas);
#define WSC_LAUNCH(T,HDP,HDVP) do { \
        CUtensorMap mK=ws::ws_map<T>(k), mV=ws::ws_map<T>(v), mQ=ws::ws_map<T>(q); \
        cudaFuncSetAttribute(ws::fwd_kernel_ws2_cont<T,HDP,HDVP>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem2); \
        ws::fwd_kernel_ws2_cont<T,HDP,HDVP><<<gridc,block,smem2,stream>>>(mK,mV,mQ,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
            cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
            ctr.data_ptr<int>(),total_tiles,gxp,b,dense_uniform?0:1); } while(0)
        if(hdp==128&&hdvp==128){ if(bf16) WSC_LAUNCH(__nv_bfloat16,128,128); else WSC_LAUNCH(__half,128,128); }
        else { if(bf16) WSC_LAUNCH(__nv_bfloat16,64,64); else WSC_LAUNCH(__half,64,64); }
#undef WSC_LAUNCH
        return true;
      } } }
    // q_stage=2 fast path (~1.35x); hd192 keeps q_stage=1 (2x Q would overflow 227KB smem).
#define WS2_LAUNCH(T,HDP,HDVP,CORR) do { \
        CUtensorMap mK=ws::ws_map<T>(k), mV=ws::ws_map<T>(v); \
        cudaFuncSetAttribute(ws::fwd_kernel_ws2<T,HDP,HDVP,CORR>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem2); \
        ws::fwd_kernel_ws2<T,HDP,HDVP,CORR><<<grid2,block,smem2,stream>>>(mK,mV,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
            cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv); } while(0)
#define WS1_LAUNCH(T,HDP,HDVP) do { \
        CUtensorMap mK=ws::ws_map<T>(k), mV=ws::ws_map<T>(v); \
        cudaFuncSetAttribute(ws::fwd_kernel_ws<T,HDP,HDVP>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem1); \
        ws::fwd_kernel_ws<T,HDP,HDVP><<<grid1,block,smem1,stream>>>(mK,mV,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
            cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv); } while(0)
#define WS2V1_LAUNCH(T,HDP,HDVP) do { \
        CUtensorMap mK=ws::ws_map<T>(k), mV=ws::ws_map<T>(v); \
        cudaFuncSetAttribute((ws::fwd_kernel_ws2<T,HDP,HDVP,false,1>), cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem2v1); \
        (ws::fwd_kernel_ws2<T,HDP,HDVP,false,1>)<<<grid2,block,smem2v1,stream>>>(mK,mV,(const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(), \
            cu_q.data_ptr<int>(),cu_k.data_ptr<int>(),hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv); } while(0)
#define WS2_DISP(HDP,HDVP) do { if(corr){ if(bf16) WS2_LAUNCH(__nv_bfloat16,HDP,HDVP,true); else WS2_LAUNCH(__half,HDP,HDVP,true); } \
                                 else    { if(bf16) WS2_LAUNCH(__nv_bfloat16,HDP,HDVP,false); else WS2_LAUNCH(__half,HDP,HDVP,false); } } while(0)
#define WS1_DISP(HDP,HDVP) do { if(bf16) WS1_LAUNCH(__nv_bfloat16,HDP,HDVP); else WS1_LAUNCH(__half,HDP,HDVP); } while(0)
    if      (hdp==128&&hdvp==128){ if(getenv("FA4_WS1")) WS1_DISP(128,128); else WS2_DISP(128,128); }
    else if (hdp==192&&hdvp==128){ if(getenv("FA4_WS1")) WS1_DISP(192,128);   // hd192: q_stage=2 K-ring2/V-ring1 (fits smem); FA4_WS1=1 forces old q_stage=1
                                   else if(bf16) WS2V1_LAUNCH(__nv_bfloat16,192,128); else WS2V1_LAUNCH(__half,192,128); }
    else if (hdp==64 &&hdvp==64 ) WS2_DISP(64,64);
    else return false;
#undef WS2_DISP
#undef WS1_DISP
#undef WS2_LAUNCH
#undef WS2V1_LAUNCH
#undef WS1_LAUNCH
    return true;
}

// Paged warp-specialized launcher + SplitKV combine. Routes paged K/V (bf16/fp16, hd in the WS set)
// to the fast WS pipeline instead of the slow serial kernel. Returns true if it handled the config.
static bool launch_ws_paged(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
    at::Tensor cu_q, int hq, int hkv, float scale, bool is_causal, int win_l, int win_r,
    float softcap, const float* s_aux, int hd, int hdv, const int* page_table,
    int pt_stride, int block_size, const int* seqused_k, int leftpad, int req_splits, int max_sk, int msq,
    int grp = 0) {
  int rep=hq/hkv, B=(int)cu_q.size(0)-1, total_q=(int)q.size(0);
  int hdp=(hd+63)/64*64, hdvp=(hdv+63)/64*64;
  if(!((hdp==128&&hdvp==128)||(hdp==192&&hdvp==128)||(hdp==64&&hdvp==64))) return false;
  // TIER0 HOST FAST-PATH (fix #5): max_seqlen_q op arg (msq) gives max_sq directly, avoiding the synchronous
  // pageable cu_q.to(kCPU) D2H (~22us/call). pack + max_mq are both monotonic in seqlen_q, so ceil()/threshold
  // over the max is exact for varlen too. Fall back to the CPU reduction only when the arg is absent (msq<=0).
  // NB: q2 (fwd_kernel_fa4_paged / prefill3577 routing) keys off max_mq>=2, which the ceil-derived max_mq
  // reproduces exactly, so this must NOT change the fa4_q2 routing decision.
  int max_sq;
  if(msq>0){ max_sq=msq; }
  else { auto cu_q_cpu=cu_q.to(at::kCPU); const int* cq=cu_q_cpu.data_ptr<int>();
         max_sq=0; for(int i=0;i<B;++i) max_sq=std::max(max_sq,cq[i+1]-cq[i]); }
  // grp==1 covers ONLY the sequences with rep*seqlen_q <= WTM, so clamp max_sq to that group's bound:
  // the packed single-tile geometry is then correct for every sequence the kernel will not skip.
  if(grp==1 && rep>0) max_sq = std::min(max_sq, ws::WTM/rep);
  bool pack = (rep>1) && (rep*max_sq <= ws::WTM);   // pack GQA heads only when they fit one tile (decode/short-q); large-q GQA gains nothing
  int max_mq = pack ? (rep*max_sq+ws::WTM-1)/ws::WTM : (max_sq+ws::WTM-1)/ws::WTM;   // == max_i ceil((pack?rep*s_i:s_i)/WTM) (ceil monotonic)
  bool q2 = (!pack) && (hdp==128||hdp==64) && (max_mq>=2);   // large-q paged -> q_stage=2 (2 tiles/CTA, 2 softmax WGs)
  // ---- DE-PAGE + DENSE route (large-q paged prefill). See depage_kv above for the rationale. ----
  // Restricted to caches the page-aligned TMA box CANNOT express (page_size % 8 != 0, i.e. 1/2/4):
  // where `pt_ok` holds, fwd_kernel_fa4_paged already streams K/V as nseg*(HD/64+HDV/64) multi-row
  // `cp.async.bulk.tensor.2d` boxes and is 0.92-1.03x -- de-paging there only adds the copy
  // (measured: mistral_pf_q4k 1.03->0.17, llama8b_pf_q8k 0.92->0.64, ext_q1k_k64k 0.93->0.52).
  // Engages only where the one-time gather amortizes: `rep*max_sq/(2*WTM)` re-reads per KV row must be
  // comfortably >1 (threshold 16*WTM == 2048 -> >=8 re-reads), and only for the exact envelope the dense
  // persistent kernel handles natively so launch_ws is guaranteed to take fa4_auto (no
  // window/sink/softcap/leftpad, true hd 128 or 64, bf16/fp16, single KV pass, ungrouped).
  // FA4_NODEPAGE=1 disables (A/B); FA4_DEPAGE=1 forces past the amortization threshold.
  const int _dp_es = (int)q.element_size();
  const int _dp_seg = (block_size>0 && block_size<ws::WTN) ? block_size : ws::WTN;
  const bool _dp_pageTMA = (getenv("FA4_NOPT")==nullptr) && (block_size>0) && (_dp_seg%8==0)
                           && (hd==hdp) && (hdv==hdvp) && k.is_contiguous() && v.is_contiguous();
  bool dp_ok = (getenv("FA4_NODEPAGE")==nullptr) && !_dp_pageTMA && !pack && q2 && (grp==0) && (req_splits<=1)
               && (block_size>0) && page_table && seqused_k && (leftpad==0) && (max_sk>0) && (B>0)
               && ((hdp==128&&hdvp==128)||(hdp==64&&hdvp==64)) && (hd==hdp) && (hdv==hdvp)
               && (q.scalar_type()==at::kBFloat16 || q.scalar_type()==at::kHalf)
               && (win_l<0) && (win_r<0) && (softcap<=0.f) && (s_aux==nullptr)
               && k.is_contiguous() && v.is_contiguous()
               && ((hkv*hd)%8==0) && ((hkv*hdv)%8==0)
               && ((getenv("FA4_DEPAGE")!=nullptr) || ((long)rep*(long)max_sq >= 16L*(long)ws::WTM));
  if (dp_ok) {
    // Upper bound on the packed row count: sum(seqused_k) <= B*max_sk. The scratch is pooled across
    // calls, so the size guard only has to reject absurd requests (a huge-batch long-context prefill);
    // an actual OOM on the first touch also falls through to the paged kernels.
    const long rows = (long)B * (long)max_sk;
    const long bytes = rows * (long)hkv * (long)(hd + hdv) * (long)_dp_es;
    size_t mfree = 0, mtot = 0; cudaMemGetInfo(&mfree, &mtot);
    bool room = bytes > 0 && (double)bytes < 0.25 * (double)mtot;
    if (room) try {
      auto kc = depage_pool(0, rows * hkv * hd,  q.options()).view({(int64_t)rows, (int64_t)hkv, (int64_t)hd });
      auto vc = depage_pool(1, rows * hkv * hdv, q.options()).view({(int64_t)rows, (int64_t)hkv, (int64_t)hdv});
      auto cuk = depage_pool(2, (int64_t)B + 1, q.options().dtype(at::kInt));
      const int nvk = hkv*hd/8, nvv = hkv*hdv/8, nv = std::max(nvk, nvv);
      auto st = at::cuda::getCurrentCUDAStream();
      depage_cuk<<<1, 32, 0, st>>>(seqused_k, cuk.data_ptr<int>(), B);
      dim3 dg((unsigned)(((long)max_sk*nv + 255)/256), (unsigned)B);
      if (q.scalar_type()==at::kBFloat16)
        depage_kv<__nv_bfloat16><<<dg, 256, 0, st>>>((const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
            (__nv_bfloat16*)kc.data_ptr(), (__nv_bfloat16*)vc.data_ptr(), page_table, pt_stride, block_size,
            seqused_k, cuk.data_ptr<int>(), nvk, nvv, max_sk);
      else
        depage_kv<__half><<<dg, 256, 0, st>>>((const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
            (__half*)kc.data_ptr(), (__half*)vc.data_ptr(), page_table, pt_stride, block_size,
            seqused_k, cuk.data_ptr<int>(), nvk, nvv, max_sk);
      if (launch_ws(q, kc, vc, out, lse, cu_q, cuk, hq, hkv, scale, is_causal, win_l, win_r,
                    softcap, s_aux, hd, hdv, msq, max_sk))
        return true;
      // launch_ws declined (should not happen inside the gate above): fall through to the paged path.
    } catch (const std::exception&) { /* OOM on the scratch: fall through to the paged kernels. */ }
  }
  // GQA q_stage=2 (2 qheads -> 2 Q-tiles, shared gather, 2 softmax WGs w/ 1-pass softmax). Intended to
  // hide the tmem-read latency that pins qlen120 at ~1 CTA/SM. DEAD-END: measured 0.48 vs 0.60 for the
  // q_stage=1 1-pass kernel — doubling the (serialized) MMA per block + halving gather warps outweighs
  // the 2-WG latency hiding. qlen120 is latency-bound in a way the FA4 2-WG/corr structure doesn't fix
  // here. Kept opt-in for future experiments (would need q_stage=2 + correction-WG combined).
  bool gqa2=false;
  if(const char* e=getenv("FA4_GQA2")) gqa2 = atoi(e) && (rep==2) && !pack && !q2 && (max_sq<=ws::WTM) && (max_sq>=16) && (hdp==128||hdp==64);
  // ---- FA4-choreography paged routing (fwd_kernel_fa4_paged): fa4 consumer pipeline + gather4 async
  // producer. Requires the gather4 loader (hd==hdp, contiguous pool), hd128, bf16/fp16, causal, no
  // window/softcap/sink/leftpad. FA4_NOFA4PAGED=1 forces the old ws_paged/ws2_paged path (A/B).
  // FA4_NOG4=1 still disables fwd_kernel_fa4_paged entirely (-> ws2_paged), as before. ----
  bool g4ok = (hd==hdp)&&(hdv==hdvp)&&k.is_contiguous()&&v.is_contiguous()&&(getenv("FA4_NOG4")==nullptr);
  bool fa4_ok = (getenv("FA4_NOFA4PAGED")==nullptr) && (hdp==128&&hdvp==128) && (hd==128&&hdv==128)
                && (q.scalar_type()==at::kBFloat16 || q.scalar_type()==at::kHalf) && is_causal
                && (win_l<0) && (win_r<0) && (softcap<=0.f) && (s_aux==nullptr) && (leftpad==0) && g4ok;
  bool fa4_q2  = fa4_ok && q2 && !gqa2;                              // large-q 2-row-tile (prefill3577): WINS (0.32->0.51)
  // GQA decode (qlen120) via fa4 2-qhead tiles: measured 0.56 vs 0.62 for the lean single-WG default -- the
  // 2-WG+correction pipeline doesn't amortize over the short per-split block count (though it beats the old
  // ws2 GQA 0.48). Kept OPT-IN (FA4_FA4GQA=1) for A/B; default OFF so qlen120 keeps its faster old path.
  bool fa4_gqa = fa4_ok && !q2 && !gqa2 && !pack && (rep==2) && (max_sq<=ws::WTM) && (max_sq>=16)
                 && (getenv("FA4_FA4GQA")!=nullptr);
  int gy = (pack||gqa2||fa4_gqa) ? hkv : hq;
  int gx = (q2||fa4_q2) ? (max_mq+1)/2 : max_mq;
  int nCTA=gx*gy*B;
  int NSM=132; { int dev=0; if(cudaGetDevice(&dev)==cudaSuccess){ int n=0; if(cudaDeviceGetAttribute(&n,cudaDevAttrMultiProcessorCount,dev)==cudaSuccess && n>0) NSM=n; } }
  int num_nb=(max_sk>0)?(max_sk+ws::WTN-1)/ws::WTN:1;               // KV blocks (max over batch)
  // fwd_kernel_fa4_paged: Lever 2b full-range page-row precompute buffer (int[num_nb][WTN] in dyn smem)
  // when it fits under the smem cap; else per-block. Producer A/B (Lever 1): the HW-async TMA tile::gather4
  // producer is the DEFAULT and the measured WINNER (it hides KV-feed latency far better than a hand-rolled
  // cp.async gather at this 96-load-thread / FA4KVS=2 / K-then-V-split choreography). FA4_CPASYNC=1 selects
  // the coalesced 128-bit cp.async.cg producer instead (kept for A/B; measured SLOWER here — see report).
  int fa4_prow_cap = std::min(num_nb, 48);                          // 48*WTN*4 = 24KB extra dyn smem (193KB+24KB<227KB)
  int fa4_use_g4   = (getenv("FA4_CPASYNC")!=nullptr) ? 0 : 1;
  // Paged K-ring depth. The decoupled 3/2 ring is wired up (FA4_PGKD=3) but DOES NOT LAUNCH at hd128:
  // data is 64KB Q + 96KB K + 64KB V = 229376B, and once the 1024B align pad and ~8.2KB of static smem
  // (sScale + sFinSum + sFinMax at [2][2][128] floats each = 6144B, prow_c, the barrier struct) are added
  // it is 238600B against the 232448B opt-in cap -- ~6KB over. Shrinking sFinSum/sFinMax to [2][WTM]
  // (they are written once at the last block, so the slot dimension is dead) and prow_c to [2][WTN]
  // recovers only 2560B. Default stays at the coupled 2/2 ring; the decoupled plumbing is kept because
  // it is the enabling refactor and is verified byte-for-byte equivalent at depth 2.
  int pg_kd = 2; if(const char* e=getenv("FA4_PGKD")) pg_kd=atoi(e);
  // FA4 num_splits_heuristic: floor(num_SMs / total_mblocks) -> pack CTAs into ~1 wave WITHOUT a partial-wave
  // tail. ceil over-splits into a 2nd near-empty wave (~2x wall time at 1 CTA/SM). No split for tiny K (<=4 blk).
  int waves=0; if(const char* e=getenv("FA4_SPLIT_WAVES")) waves=atoi(e);   // sweep override: target = NSM*waves
  int auto_splits;
  if(num_nb<=4 || nCTA<=0) auto_splits=1;
  else if(waves>0) auto_splits=std::max(1,std::min((NSM*waves)/nCTA, std::min(num_nb,128)));
  else auto_splits=std::max(1,std::min(NSM/nCTA, std::min(num_nb,128)));
  int splits=(req_splits==1)?1:std::max(req_splits,auto_splits);
  // combine_kernel reduces over EVERY (row, head) of the call, but a grouped launch only owns a subset of
  // the rows -- it would read the other group's uninitialized lse_part and clobber its output. Grouped
  // launches therefore stay single-pass. (Costless in practice: a mixed batch has enough sequences that
  // the floor(NSM/nCTA) heuristic already picks 1 split.)
  if(grp!=0) splits=1;
  if(const char* e=getenv("FA4_FORCE_SPLITS")){ int fs=atoi(e); if(fs>0) splits=fs; }   // diag: force split count
  splits=std::max(1,std::min(splits,std::min(num_nb,64)));
  float* o_part=nullptr; float* lse_part=nullptr; at::Tensor opt,lpt;
  if(splits>1){ opt=at::empty({splits,total_q,hq,hdv},q.options().dtype(at::kFloat)); lpt=at::empty({splits,hq,total_q},q.options().dtype(at::kFloat));
                o_part=opt.data_ptr<float>(); lse_part=lpt.data_ptr<float>(); }
  dim3 grid(gx,gy,B*splits), block(512), blockD(256);
  size_t smem =(size_t)(  ws::WTM*hdp + ws::WST*ws::WTN*hdp + ws::WST*ws::WTN*hdvp)*2 + 1024;
  size_t smem2=(size_t)(2*ws::WTM*hdp + ws::WST*ws::WTN*hdp + ws::WST*ws::WTN*hdvp)*2 + 1024;
  size_t smemD=(size_t)(  ws::WTM*hdp +        ws::WTN*hdp +        ws::WTN*hdvp)*2 + 1024;   // SD=1 single K/V buffer (~97KB @hd128 -> 2 CTAs/SM)
  size_t smem3=(size_t)(  ws::WTM*hdp + 3*ws::WTN*hdp + 3*ws::WTN*hdvp)*2 + 1024;              // SD=3 deeper K/V prefetch (~225KB @hd128)
  // Decode/paged is paged-gather latency-bound (long_scoreboard/L1TEX ~53%, DRAM ~5%). SD=3 (deeper K/V
  // prefetch) was NEUTRAL-to-worse in clean A/B (0.643 vs 0.661): the stall is serialized loads WITHIN a
  // gather warp (load->scatter->load), not prefetch depth. Kept opt-in via env. Fix is cp.async (below).
  int pt_segrows = (block_size>0 && block_size<ws::WTN) ? block_size : ws::WTN;
  bool pt_ok = (getenv("FA4_NOPT")==nullptr) && (block_size>0) && (pt_segrows%8==0)
               && (hd==hdp) && (hdv==hdvp) && k.is_contiguous() && v.is_contiguous();
  // Ring depth. The page-TMA producer made the paged mainloop memory-latency-bound rather than
  // issue-bound, so a 3-deep K/V ring (2 tiles in flight, 224KB of the 227KB smem cap) is now a clear
  // win where it used to be neutral: decode b64_s16k 0.69->0.84, b16_s16k 0.66->0.77, b16_s32k
  // 0.63->0.71, mistral_dec_b16_s4k 0.93->1.01. The cp.async producer keeps depth 2 (measured worse
  // at 3 there -- its stall is serialized issue inside a load warp, not prefetch depth).
  // ...but only once a CTA has enough K blocks to reach steady state. Below ~24 blocks/CTA the extra
  // ring-fill latency and the 224KB smem request cost more than the prefetch buys (measured at
  // nt=256: dec1024 2.26->2.03, page128 2.27->2.14, dec339 1.46->1.39, dec_b16_s512 1.58->1.47),
  // while long-context decode wants the depth (b16_s16k 128 blocks/CTA, b16_s4k 32).
  int nb_per_cta = (num_nb + splits - 1) / splits;
  int kvst = (pt_ok && nb_per_cta >= 24) ? 3 : 2; if(const char* e=getenv("FA4_KVST")) kvst=atoi(e);
  // Page-TMA CTA width. 256 is the DEFAULT and a large win: the producer only needs ~8 TMA issues per
  // tile, and __launch_bounds__(256,1) buys 256 regs/thread, so the 1-pass softmax f[128] and the
  // epilogue of[HDV] stop spilling to local memory (measured, page-TMA + ring 3, ours vs FA4:
  // dec_b16_s16k 0.79->1.05, b16_s32k 0.71->0.97, b64_s16k 0.84->0.99, b16_s4k 1.00->1.20,
  // decode_128k_splitkv 0.68->0.86). FA4_PTNT=512 restores the wide CTA for A/B.
  int pt_nt = 256; if(const char* e=getenv("FA4_PTNT")) pt_nt=atoi(e);
  // The gather4 route (page_size < 8) gets the same two levers the page-TMA route won on: a 256-thread
  // spill-free CTA and a 3-deep ring once there are enough K blocks per CTA to reach steady state.
  // NB gather4 wants the OPPOSITE CTA width to page-TMA: 32*(KCB+VCB) issues per tile need many load
  // threads, so 512 (11 load warps) beats the 256-thread spill-free form by a wide margin
  // (attn_real_llama8b_dec_s16k 0.61 vs 0.45, dec_b64_s4k 0.57 vs 0.44). Ring depth 3 still helps.
  int g4_nt = 512; if(const char* e=getenv("FA4_G4NT")) g4_nt=atoi(e);
  int g4_kv = (nb_per_cta>=24) ? 3 : 2; if(const char* e=getenv("FA4_G4KV")) g4_kv=atoi(e);
  // ---- PROFILED qlen120 gap (ncu 2025.3, ours vs FA4 FlashAttentionForwardSm100, SAME shape) ----
  // BOTH run 512thr / 128reg / 1-CTA-per-SM @ ~25% occ -> occupancy is NOT the lever. Main-kernel
  // duration 152us(ours) vs 54us(FA4). Per-issue stall cycles ours->FA4: long_scoreboard(global load)
  // 4.1->3.0, barrier 3.0->2.1, wait 1.9->2.0, short_scoreboard(tmem/O-rescale) 0.33->0.89; DRAM 8%->20%.
  // => gap = FA4 (a) hides global-load latency far better (TMA paged loads: 2.5x DRAM, less long_scoreboard)
  // and (b) has less barrier/handshake overhead. It is NOT tmem-read/O-rescale bound (short_scoreboard is
  // TINY) -- so CORR (O-rescale offload) and SPK (2 softmax streams to hide tmem reads) attack the WRONG
  // stall, which is why both regress. splits>2 and KVST=3 both regress too (verified). Closing qlen120 needs
  // FA4's actual design: 1-warp-per-softmax (frees warps/regs) + TMA-based paged K/V load. From-scratch kernel.
  // SD=1 (256 thr, single-buffer, 2 CTAs/SM) DEAD-END: serial exposed-gather latency >> occupancy gain
  // (mha 0.46->0.17x). FA4 also runs ~1 CTA/SM, so the gap is choreography, not occupancy. Opt-in only.
  bool decode1=false;
  if(getenv("FA4_SD1")) decode1=(max_sq<=8)&&!q2&&(hdp==128||hdp==64);
  // CORR (correction-WG): O-rescale offloaded to w4-7, overlapped with softmax[i+1] (FA4 choreography).
  // Helps multi-row (non-pack) decode where the per-block O-rescale dominates the softmax critical path.
  // Needs SD=2 (512 thr). Off for pack (few rows -> tiny rescale) / q2 (its own 2-WG choreography).
  bool corr=false; if(const char* e=getenv("FA4_DEC_CORR")) corr=atoi(e);
  corr = corr && !pack && !q2 && (kvst==2) && (hdp==128||hdp==64) && (max_sq>=16);
  // SPK (intra-CTA 2-way split-K decode, distilled from cutlass 77_blackwell_fmha gen kernel): two softmax
  // warpgroups run the online softmax over even/odd K blocks of the SAME rows to hide the tmem-S read
  // latency that pins multi-row decode (qlen120) at ~1 CTA/SM. VERIFIED CORRECT but DEAD-END on perf --
  // ALL variants < the 0.60 single-WG 1-pass: 1-pass@176regs 0.28 (spills), 1-pass@192/producers-64 0.33
  // (MMA spills), 2-pass@128/128 0.43 (no spill, but 2x tmem reads + 2-WG coord + single-buffer/stream).
  // Root cause is the REGISTER WALL: 2 full 4-warp softmax WGs can't hold a spill-free f[128] 1-pass AND
  // leave the 2 producer WGs enough regs. FA4 sidesteps it with ONE WARP per softmax (32 threads / 128 rows
  // via a 32dp tmem-partition map) + a dedicated correction warp; that mapping doesn't distill cleanly into
  // this 512-thread/1-row-per-thread layout. Kept opt-in via FA4_SPK for future work (would need the rewrite).
  bool spk=false;
  if(const char* e=getenv("FA4_SPK")) spk=atoi(e) && !pack && !q2 && (hdp==128||hdp==64) && (kvst==2) && !corr && !decode1;
  // NOTE: 256-thread SD=2 (fewer idle gather warps) DEAD-ENDS too: launch_bounds(512,1) sizes the reg
  // pool at 128/thread*256=32KB but setmaxnreg.inc 192 needs 37KB -> reallocation fails -> deadlock.
  // Would need a separate launch_bounds(256,*) instantiation. Same as FA4: 512 thr / 1 CTA / SD=2.
  auto stream=at::cuda::getCurrentCUDAStream();
  bool bf16=(q.scalar_type()==at::kBFloat16);
  // gather4 (tile::gather4 HW-async loader) is used ONLY by fwd_kernel_fa4_paged (prefill3577), which
  // needs the descriptors below. It is compiled OUT of ws_paged/ws2_paged (UG4=false) — measured a net
  // loss on SHORT paged-decode (dec1024/mqa32/gqa5/noncausal/local/gqa3/page128 -2.4..4.6%: the 128
  // gather4 issues + mbarrier setup per tile don't amortize over few K-blocks, and building the two TMA
  // descriptors per launch is host overhead on tiny decode kernels). It only wins where the fa4 consumer
  // can spend the freed issue slots (the long chunked-prefill). Descriptors built per-launch in WSP_FA4.
  int64_t g4_krw=(int64_t)hkv*hd, g4_vrw=(int64_t)hkv*hdv;
  int64_t g4_krows=(g4_krw>0)?k.numel()/g4_krw:0, g4_vrows=(g4_vrw>0)?v.numel()/g4_vrw:0;
  // PAGE-TMA producer (UPT): a WTN-key tile spans WTN/segrows CONTIGUOUS physical page segments, so the
  // whole K+V tile is nseg*(HD/64+HDV/64) `cp.async.bulk.tensor.2d` boxes -- 8 for hd128 @ page_size 64,
  // versus 128 gather4 issues or ~4096 cp.async issues. Needs segrows%8==0 (128B-swizzle atom = 8 rows) and
  // hd==hdp (a fixed 64-col box can't zero-pad a partial column block). FA4_NOPT=1 disables (A/B).
  // Sub-8 page sizes (page_size 1/2/4) cannot use a multi-row box, but tile::gather4 still moves 4 rows per
  // issue (128 issues/tile vs ~4096 cp.async). FA4_DECG4=0 forces the cp.async gather back on (A/B).
  bool g4_ok = !pt_ok && (hd==hdp) && (hdv==hdvp) && k.is_contiguous() && v.is_contiguous()
               && (getenv("FA4_DECG4")==nullptr || atoi(getenv("FA4_DECG4"))!=0);
  // ---- PAGED PREFILL on the PERSISTENT fa4 kernel (fwd_kernel_fa4<...,PAGED=true>) ----
  // fwd_kernel_fa4_paged is a non-persistent graft of the fa4 choreography onto the paged gather and tops
  // out near 0.70x; fwd_kernel_fa4 itself reaches 0.91-1.19x on the same math dense. With the page-aligned
  // TMA producer the only structural difference left is the K/V descriptor, so route large-q paged prefill
  // straight at the persistent kernel: same LPT tile scheduler, ragged empty-tile skip, dual stacked-half
  // softmax ping-pong, correction/epilogue warpgroups and cross-tile ring as the dense path.
  // Requires a page-TMA-expressible cache (pt_ok), a single KV pass (fa4 has no SplitKV), a Q tile big
  // enough that the persistent kernel amortizes, and none of the window/sink/softcap/leftpad grafts.
  // pt_ok caches use page boxes. page_size<8 CAN be expressed with tile::gather4 and a [64,1] descriptor
  // instead, but it LOSES here (FA4_PFG4=1 to A/B): the persistent kernel has a single 32-lane load warp,
  // so 32*(KCB+VCB) gather4 issues plus 4 dependent page-table loads per lane do not keep the ring fed --
  // attn_real_llama8b_pf_s16k 0.53->0.38, qwen3moe_pf_s4k 0.57->0.45. Those page_size-1 shapes stay on
  // fwd_kernel_fa4_paged, whose 3 load warps (96 threads) absorb the gather.
  const bool fa4p_g4 = !pt_ok && (hd==hdp) && (hdv==hdvp) && (block_size>0)
                       && k.is_contiguous() && v.is_contiguous() && (getenv("FA4_PFG4")!=nullptr);
  // ^ page_size<8 on the PERSISTENT kernel is OPT-IN. It has the better schedule (persistent + LPT vs
  // fwd_kernel_fa4_paged's plain 2048-CTA grid, which on a causal 16k prefill dispatches the light tiles
  // first and leaves the heavy ones for the last wave) but only 2 load warps against fa4_paged's 3, and
  // for page_size 1 the gather is the binding constraint: measured attn_real_llama8b_pf_s2k 0.695->0.716
  // but attn_real_llama8b_pf_s16k 0.558->0.497. Closing this properly needs a load warpgroup that is both
  // wide and persistent, i.e. taking warps from the correction WG.
  const int fa4p_seg = pt_ok ? pt_segrows : 0;
  bool fa4p = (getenv("FA4_NOPFPAGED")==nullptr) && (pt_ok||fa4p_g4) && !pack && (max_mq>=2)
              && ((hdp==128&&hdvp==128)||(hdp==64&&hdvp==64)) && (hd==hdp) && (hdv==hdvp)
              && (q.scalar_type()==at::kBFloat16 || q.scalar_type()==at::kHalf)
              && (win_l<0) && (win_r<0) && (softcap<=0.f) && (s_aux==nullptr) && (leftpad==0)
              && (splits<=1) && (grp!=1);
  if(fa4p){
      int gxp=(max_mq+1)/2; int total_tiles=gxp*hq*B;
      int ctas=std::min(total_tiles,NSM); if(ctas<1) ctas=1;
      auto& ctr=persistent_ctr(ctas,q,true);
      dim3 gridf(ctas), blockf(512);
      size_t smemf=(size_t)(2*ws::WTM*hdp + ws::FA4KVS*ws::WTN*hdp + ws::FA4KVS*ws::WTN*hdvp)*2 + 1024;
      bool bf16p=(q.scalar_type()==at::kBFloat16);
      int ragged=((long)total_q==(long)B*(long)max_sq)?0:1;
#define FA4PF(T,HDP,HDVP) do { \
      CUresult _c1,_c2; \
      CUtensorMap mKg,mVg; \
      if(pt_ok){ mKg=ws::ws_page_map_cached<T>(k.data_ptr(),g4_krows,g4_krw,(uint32_t)pt_segrows,_c1); \
                 mVg=ws::ws_page_map_cached<T>(v.data_ptr(),g4_vrows,g4_vrw,(uint32_t)pt_segrows,_c2); } \
      else     { mKg=ws::ws_map_gather<T>(k.data_ptr(),g4_krows,g4_krw,_c1); \
                 mVg=ws::ws_map_gather<T>(v.data_ptr(),g4_vrows,g4_vrw,_c2); } \
      CUtensorMap mQ=ws::ws_map<T>(q); \
      if(_c1!=CUDA_SUCCESS||_c2!=CUDA_SUCCESS) return false; \
      cudaFuncSetAttribute((ws::fwd_kernel_fa4<T,HDP,HDVP,false,false,false,true>), cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemf); \
      (ws::fwd_kernel_fa4<T,HDP,HDVP,false,false,false,true>)<<<gridf,blockf,smemf,stream>>>(mKg,mVg,mQ, \
          (const T*)q.data_ptr(),(T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr, \
          hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
          ctr.data_ptr<int>(),total_tiles,gxp,B,ragged,page_table,pt_stride,block_size,seqused_k,fa4p_seg,grp, \
          (const T*)k.data_ptr(),(const T*)v.data_ptr()); } while(0)
      if(hdp==128){ if(bf16p) FA4PF(__nv_bfloat16,128,128); else FA4PF(__half,128,128); }
      else        { if(bf16p) FA4PF(__nv_bfloat16,64,64);   else FA4PF(__half,64,64); }
#undef FA4PF
      return true;
  }
#define WSP_ONE(T,HDP,HDVP,PK,SDV,CR,BLK,SM) do { \
    CUtensorMap _mKg{},_mVg{}; \
    cudaFuncSetAttribute(ws::fwd_kernel_ws_paged<T,HDP,HDVP,PK,SDV,CR>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(SM)); \
    ws::fwd_kernel_ws_paged<T,HDP,HDVP,PK,SDV,CR><<<grid,(BLK),(SM),stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part,_mKg,_mVg,0,grp,_mKg,_mVg,0); } while(0)
/* Page-aligned-TMA variant of WSP_ONE (UPT=true). Falls back to WSP_ONE if either descriptor fails. */
#define WSP_PT(T,HDP,HDVP,PK,SDV,SM,NT) do { \
    CUresult _c1,_c2; \
    CUtensorMap _mKg=ws::ws_page_map_cached<T>(k.data_ptr(),g4_krows,g4_krw,(uint32_t)pt_segrows,_c1); \
    CUtensorMap _mVg=ws::ws_page_map_cached<T>(v.data_ptr(),g4_vrows,g4_vrw,(uint32_t)pt_segrows,_c2); \
    if(_c1!=CUDA_SUCCESS||_c2!=CUDA_SUCCESS){ WSP_ONE(T,HDP,HDVP,PK,ws::WST,false,block,smem); } \
    else { \
      cudaFuncSetAttribute((ws::fwd_kernel_ws_paged<T,HDP,HDVP,PK,SDV,false,false,true,NT>), cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(SM)); \
      (ws::fwd_kernel_ws_paged<T,HDP,HDVP,PK,SDV,false,false,true,NT>)<<<grid,(NT),(SM),stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part,_mKg,_mVg,0,grp,_mKg,_mVg,0); } } while(0)
/* tile::gather4 variant of WSP_ONE (UG4=true), for page sizes a multi-row TMA box cannot express. */
/* tile::gather4 + a SECOND, full-width [64,WTN] page descriptor. page_size<8 cannot be expressed as a
   multi-row box at LAUNCH time, but a tile whose WTN physical rows happen to be consecutive can use one at
   RUN time (32x fewer TMA descriptors -- the measured decode limiter). The kernel scans its own page-table
   range once and picks; a fragmented table keeps gather4. FA4_NOPTRUN=1 disables the upgrade (A/B). */
#define WSP_G4(T,HDP,HDVP,PK,SDV,SM,NT) do { \
    CUresult _c1,_c2,_c3=CUDA_ERROR_INVALID_VALUE,_c4=CUDA_ERROR_INVALID_VALUE; \
    CUtensorMap _mKg=ws::ws_map_gather<T>(k.data_ptr(),g4_krows,g4_krw,_c1); \
    CUtensorMap _mVg=ws::ws_map_gather<T>(v.data_ptr(),g4_vrows,g4_vrw,_c2); \
    CUtensorMap _mKf=_mKg,_mVf=_mVg; \
    if(getenv("FA4_NOPTRUN")==nullptr){ \
      _mKf=ws::ws_page_map_cached<T>(k.data_ptr(),g4_krows,g4_krw,(uint32_t)ws::WTN,_c3); \
      _mVf=ws::ws_page_map_cached<T>(v.data_ptr(),g4_vrows,g4_vrw,(uint32_t)ws::WTN,_c4); } \
    int _hf=(_c3==CUDA_SUCCESS&&_c4==CUDA_SUCCESS)?1:0; \
    if(_c1!=CUDA_SUCCESS||_c2!=CUDA_SUCCESS){ WSP_ONE(T,HDP,HDVP,PK,ws::WST,false,block,smem); } \
    else { \
      cudaFuncSetAttribute((ws::fwd_kernel_ws_paged<T,HDP,HDVP,PK,SDV,false,true,false,NT>), cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(SM)); \
      (ws::fwd_kernel_ws_paged<T,HDP,HDVP,PK,SDV,false,true,false,NT>)<<<grid,(NT),(SM),stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part,_mKg,_mVg,1,grp,_mKf,_mVf,_hf); } } while(0)
#define WSP2(T,HDP,HDVP,G) do { \
    CUtensorMap _mKg{},_mVg{}; \
    cudaFuncSetAttribute(ws::fwd_kernel_ws2_paged<T,HDP,HDVP,G>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem2); \
    ws::fwd_kernel_ws2_paged<T,HDP,HDVP,G><<<grid,block,smem2,stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part,_mKg,_mVg,0,grp); } while(0)
#define WSP_SPK(T,HDP,HDVP) do { \
    cudaFuncSetAttribute(ws::fwd_kernel_ws_paged_spk<T,HDP,HDVP>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem); \
    ws::fwd_kernel_ws_paged_spk<T,HDP,HDVP><<<grid,block,smem,stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part); } while(0)
/* FA4-choreography paged kernel: fa4 consumer pipeline (correction WG + depth-2 C-channel + K/V ring) +
   HW-async gather4 producer. smem == fwd_kernel_fa4 (sO aliased into sQ). G=GQA (true: 2 qheads/CTA). */
/* Decoupled K-depth-3 / V-depth-1 ring. K3/V2 is ~6KB over the smem cap once static smem is counted
   (see the FA4_PGKD comment below), but K3/V1 costs 197632B against the coupled K2/V2's 196608B -- the
   same budget -- so it is a free A/B of "does a deeper K ring help?". PC sampling said the paged producer
   starves for ring slots, and QK is the consumer that waits first, so the slot goes on K. The 24KB
   `prowAll` full-range page-row buffer is given up either way (it only engaged at NBk<=48).
   FA4_PGKD=2 restores the coupled 2/2 ring. */
#define WSP_FA4KD(T,HDP,HDVP,G) do { \
    CUresult _crk=CUDA_ERROR_INVALID_VALUE,_crv=CUDA_ERROR_INVALID_VALUE; CUtensorMap _mKg{},_mVg{}; int _ldr=fa4_use_g4; \
    if(pt_ok){ _mKg=ws::ws_page_map_cached<T>(k.data_ptr(),g4_krows,g4_krw,(uint32_t)pt_segrows,_crk); \
               _mVg=ws::ws_page_map_cached<T>(v.data_ptr(),g4_vrows,g4_vrw,(uint32_t)pt_segrows,_crv); _ldr=2; } \
    if(_crk!=CUDA_SUCCESS||_crv!=CUDA_SUCCESS){ _mKg=ws::ws_map_gather<T>(k.data_ptr(),g4_krows,g4_krw,_crk); \
               _mVg=ws::ws_map_gather<T>(v.data_ptr(),g4_vrows,g4_vrw,_crv); _ldr=fa4_use_g4; } \
    size_t _smf=(size_t)(2*ws::WTM*HDP + 3*ws::WTN*HDP + 1*ws::WTN*HDVP)*2 + 1024; \
    cudaFuncSetAttribute((ws::fwd_kernel_fa4_paged<T,HDP,HDVP,G,3,1>), cudaFuncAttributeMaxDynamicSharedMemorySize,(int)_smf); \
    (ws::fwd_kernel_fa4_paged<T,HDP,HDVP,G,3,1>)<<<grid,block,_smf,stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part,_mKg,_mVg,0,_ldr,grp); } while(0)
#define WSP_FA4(T,HDP,HDVP,G) do { \
    CUresult _crk=CUDA_ERROR_INVALID_VALUE,_crv=CUDA_ERROR_INVALID_VALUE; CUtensorMap _mKg{},_mVg{}; int _ldr=fa4_use_g4; \
    if(pt_ok){ _mKg=ws::ws_page_map_cached<T>(k.data_ptr(),g4_krows,g4_krw,(uint32_t)pt_segrows,_crk); \
               _mVg=ws::ws_page_map_cached<T>(v.data_ptr(),g4_vrows,g4_vrw,(uint32_t)pt_segrows,_crv); _ldr=2; } \
    if(_crk!=CUDA_SUCCESS||_crv!=CUDA_SUCCESS){ _mKg=ws::ws_map_gather<T>(k.data_ptr(),g4_krows,g4_krw,_crk); \
               _mVg=ws::ws_map_gather<T>(v.data_ptr(),g4_vrows,g4_vrw,_crv); _ldr=fa4_use_g4; } \
    size_t _smf=(size_t)(2*ws::WTM*HDP + ws::FA4KVS*ws::WTN*HDP + ws::FA4KVS*ws::WTN*HDVP)*2 + 1024 \
                + (size_t)fa4_prow_cap*ws::WTN*sizeof(int); \
    cudaFuncSetAttribute(ws::fwd_kernel_fa4_paged<T,HDP,HDVP,G>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)_smf); \
    ws::fwd_kernel_fa4_paged<T,HDP,HDVP,G><<<grid,block,_smf,stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
        (T*)out.data_ptr(),lse.data_ptr<float>(),cu_q.data_ptr<int>(),nullptr,hq,hkv,rep,scale,is_causal?1:0,win_l,win_r,softcap,s_aux,hd,hdv, \
        1,page_table,pt_stride,block_size,seqused_k,leftpad,B,splits,o_part,lse_part,_mKg,_mVg,fa4_prow_cap,_ldr,grp); } while(0)
#define WSP_LAUNCH(T,HDP,HDVP) do { \
    if(fa4_q2){ if(pg_kd==3) WSP_FA4KD(T,HDP,HDVP,false); else WSP_FA4(T,HDP,HDVP,false); } \
    else if(fa4_gqa){ if(pg_kd==3) WSP_FA4KD(T,HDP,HDVP,true); else WSP_FA4(T,HDP,HDVP,true); } \
    else if(spk) WSP_SPK(T,HDP,HDVP); \
    else if(gqa2) WSP2(T,HDP,HDVP,true); \
    else if(q2) WSP2(T,HDP,HDVP,false); \
    else if(decode1){ if(pack) WSP_ONE(T,HDP,HDVP,true,1,false,blockD,smemD); else WSP_ONE(T,HDP,HDVP,false,1,false,blockD,smemD); } \
    else if(corr){ WSP_ONE(T,HDP,HDVP,false,ws::WST,true,block,smem); } \
    else if(pt_ok&&kvst==3){ if(pt_nt==256){ if(pack) WSP_PT(T,HDP,HDVP,true,3,smem3,256); else WSP_PT(T,HDP,HDVP,false,3,smem3,256); } \
                            else          { if(pack) WSP_PT(T,HDP,HDVP,true,3,smem3,512); else WSP_PT(T,HDP,HDVP,false,3,smem3,512); } } \
    else if(pt_ok){ if(pt_nt==256){ if(pack) WSP_PT(T,HDP,HDVP,true,ws::WST,smem,256); else WSP_PT(T,HDP,HDVP,false,ws::WST,smem,256); } \
                    else          { if(pack) WSP_PT(T,HDP,HDVP,true,ws::WST,smem,512); else WSP_PT(T,HDP,HDVP,false,ws::WST,smem,512); } } \
    else if(kvst==3){ if(pack) WSP_ONE(T,HDP,HDVP,true,3,false,block,smem3); else WSP_ONE(T,HDP,HDVP,false,3,false,block,smem3); } \
    else if(g4_ok&&g4_kv==3){ if(g4_nt==256){ if(pack) WSP_G4(T,HDP,HDVP,true,3,smem3,256); else WSP_G4(T,HDP,HDVP,false,3,smem3,256); } \
                              else          { if(pack) WSP_G4(T,HDP,HDVP,true,3,smem3,512); else WSP_G4(T,HDP,HDVP,false,3,smem3,512); } } \
    else if(g4_ok){ if(g4_nt==256){ if(pack) WSP_G4(T,HDP,HDVP,true,ws::WST,smem,256); else WSP_G4(T,HDP,HDVP,false,ws::WST,smem,256); } \
                    else          { if(pack) WSP_G4(T,HDP,HDVP,true,ws::WST,smem,512); else WSP_G4(T,HDP,HDVP,false,ws::WST,smem,512); } } \
    else { if(g4_kv==3){ if(pack) WSP_ONE(T,HDP,HDVP,true,3,false,block,smem3); else WSP_ONE(T,HDP,HDVP,false,3,false,block,smem3); } \
           else        { if(pack) WSP_ONE(T,HDP,HDVP,true,ws::WST,false,block,smem); else WSP_ONE(T,HDP,HDVP,false,ws::WST,false,block,smem); } } \
    if(splits>1){ int NN=total_q*hq*hdv,TPB=128; combine_kernel<T><<<(NN+TPB-1)/TPB,TPB,0,stream>>>(o_part,lse_part,(T*)out.data_ptr(),lse.data_ptr<float>(),total_q,hq,hdv,hdv,splits,s_aux); } \
  } while(0)
#define WSP_DISP(HDP,HDVP) do { if(bf16) WSP_LAUNCH(__nv_bfloat16,HDP,HDVP); else WSP_LAUNCH(__half,HDP,HDVP); } while(0)
  if(hdp==128&&hdvp==128) WSP_DISP(128,128);
  else if(hdp==192&&hdvp==128) WSP_DISP(192,128);
  else if(hdp==64&&hdvp==64) WSP_DISP(64,64);
  else return false;
#undef WSP_DISP
#undef WSP_LAUNCH
#undef WSP_FA4
#undef WSP_SPK
#undef WSP2
#undef WSP_PT
#undef WSP_G4
#undef WSP_ONE
  return true;
}

// MIXED (ragged) PAGED BATCH: one call holding a long chunked-prefill sequence plus many q=1 decode
// sequences (vLLM-style continuous batching). A single launch has to be sized by the LONGEST sequence,
// which forces pack=false -- so every decode sequence ran `hq` separate 128-row Q tiles with one valid
// row each, reading K/V once per QUERY head instead of once per KV head (rep=4 -> 4x the KV traffic),
// spread over gx*hq CTAs of which one in gx did any work. Splitting by query length fixes both: the
// decode-like sequences (rep*seqlen_q <= WTM) get the packed-GQA route, the rest keep the prefill route.
//
// `total_q != batch * max_seqlen_q` proves the batch is ragged from op arguments alone, and each launch
// filters the sequences it does not own with a CTA-uniform predicate on `cu_q`, so this needs no
// scheduler metadata, no batch-index map, and no synchronous cu_seqlens D2H.
static bool launch_paged_maybe_split(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
    at::Tensor cu_q, int hq, int hkv, float scale, bool is_causal, int win_l, int win_r,
    float softcap, const float* s_aux, int hd, int hdv, const int* page_table,
    int pt_stride, int block_size, const int* seqused_k, int leftpad, int req_splits, int max_sk, int msq) {
    const int rep=hq/hkv, B=(int)cu_q.size(0)-1, total_q=(int)q.size(0);
    const bool split_batch = (getenv("FA4_NOMIXSPLIT")==nullptr) && (msq>0) && (rep>1) && (B>1)
                             && ((long)total_q != (long)B*(long)msq)   // ragged
                             && (msq > ws::WTM/rep);                   // else one launch already packs everything
    if(!split_batch)
        return launch_ws_paged(q,k,v,out,lse,cu_q,hq,hkv,scale,is_causal,win_l,win_r,softcap,s_aux,hd,hdv,
                               page_table,pt_stride,block_size,seqused_k,leftpad,req_splits,max_sk,msq,0);
    bool ok = launch_ws_paged(q,k,v,out,lse,cu_q,hq,hkv,scale,is_causal,win_l,win_r,softcap,s_aux,hd,hdv,
                              page_table,pt_stride,block_size,seqused_k,leftpad,req_splits,max_sk,msq,1);
    ok = ok && launch_ws_paged(q,k,v,out,lse,cu_q,hq,hkv,scale,is_causal,win_l,win_r,softcap,s_aux,hd,hdv,
                               page_table,pt_stride,block_size,seqused_k,leftpad,req_splits,max_sk,msq,2);
    return ok;
}

// ------------------------------- serial fallback launch --------------------------------
static void launch(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, at::Tensor lse,
                   at::Tensor cu_q, const int* cu_k_ptr, int hq, int hkv, float scale,
                   bool is_causal, int win_l, int win_r, float softcap, const float* s_aux,
                   int hd, int hdv, int paged, const int* page_table, int pt_stride, int block_size,
                   const int* seqused_k, int leftpad, int req_splits, int max_sk, int msq) {
    int rep = hq/hkv;
    int B = (int)cu_q.size(0)-1;
    int total_q = (int)q.size(0);
    // TIER0 HOST FAST-PATH (fix #5): max_seqlen_q op arg (msq) -> max_mq via ceil (monotonic, exact for varlen),
    // avoiding the synchronous pageable cu_q.to(kCPU) D2H. Fall back to the CPU reduction only when absent.
    int max_mq;
    if (msq>0) { max_mq = (msq+TM-1)/TM; }
    else { auto cu_q_cpu = cu_q.to(at::kCPU); const int* cq = cu_q_cpu.data_ptr<int>();
           max_mq = 0; for (int i=0;i<B;++i){ int s=cq[i+1]-cq[i]; max_mq = std::max(max_mq, (s+TM-1)/TM); } }
    int hdp = (hd + 63)/64*64, hdvp = (hdv + 63)/64*64;   // pad to multiple of 64 (128B swizzle col-block)
    size_t smem = (size_t)(TM*hdp + TN*hdp + TN*hdvp)*2 + 1024;
    auto stream = at::cuda::getCurrentCUDAStream();
    bool bf16 = (q.scalar_type()==at::kBFloat16);

    // SplitKV: when the base grid underfills the GPU (decode: small q -> few CTAs), split the KV
    // reduction across CTAs and combine. Output is identical for any split count (reduction reorg),
    // so we auto-fill ~132 SMs; req_splits==1 means explicit no-split, else at least the auto count.
    int nCTA = max_mq * hq * B;
    const int NSM=132, TARGET=NSM*3;                     // decode CTAs are light (q small) -> oversubscribe
    int auto_splits = (nCTA>0 && nCTA<TARGET) ? (TARGET + nCTA - 1)/nCTA : 1;
    int splits = (req_splits==1) ? 1 : std::max(req_splits, auto_splits);
    int max_nb = (max_sk>0) ? (max_sk + TN - 1)/TN : splits;      // don't split finer than the K blocks
    splits = std::max(1, std::min(splits, std::min(max_nb, 64)));

    float* o_part=nullptr; float* lse_part=nullptr;
    at::Tensor o_part_t, lse_part_t;
    if (splits>1) {
        o_part_t = at::empty({splits, total_q, hq, hdv}, q.options().dtype(at::kFloat));
        lse_part_t = at::empty({splits, hq, total_q}, q.options().dtype(at::kFloat));
        o_part = o_part_t.data_ptr<float>(); lse_part = lse_part_t.data_ptr<float>();
    }
    dim3 grid(max_mq, hq, B*splits), block(128);

#define FA4_LAUNCH(T, HDP, HDVP) do { \
        cudaFuncSetAttribute(fwd_kernel<T,HDP,HDVP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        fwd_kernel<T,HDP,HDVP><<<grid,block,smem,stream>>>((const T*)q.data_ptr(),(const T*)k.data_ptr(),(const T*)v.data_ptr(), \
            (T*)out.data_ptr(), lse.data_ptr<float>(), cu_q.data_ptr<int>(), cu_k_ptr, \
            hq, hkv, rep, scale, is_causal?1:0, win_l, win_r, softcap, s_aux, hd, hdv, \
            paged, page_table, pt_stride, block_size, seqused_k, leftpad, B, splits, o_part, lse_part); \
        if (splits>1) { int NN=total_q*hq*hdv, TPB=128; \
            combine_kernel<T><<<(NN+TPB-1)/TPB,TPB,0,stream>>>(o_part, lse_part, (T*)out.data_ptr(), lse.data_ptr<float>(), total_q, hq, hdv, hdv, splits, s_aux); } \
    } while(0)
#define FA4_DISPATCH(HDP, HDVP) do { if (bf16) FA4_LAUNCH(__nv_bfloat16, HDP, HDVP); else FA4_LAUNCH(__half, HDP, HDVP); } while(0)

    if      (hdp==128 && hdvp==128) FA4_DISPATCH(128,128);
    else if (hdp==192 && hdvp==128) FA4_DISPATCH(192,128);
    else if (hdp==64  && hdvp==64 ) FA4_DISPATCH(64,64);
    else TORCH_CHECK(false, "fa4_blackwell: unsupported (hd_pad,hdv_pad)=(", hdp, ",", hdvp, ")");
#undef FA4_DISPATCH
#undef FA4_LAUNCH
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
  int total_q = (int)q.size(0), hq = (int)q.size(1), hd = (int)q.size(2);
  bool paged = page_table.has_value();
  int hkv = paged ? (int)k.size(2) : (int)k.size(1);
  int hdv = paged ? (int)v.size(3) : (int)v.size(2);

  auto out_dtype = q.scalar_type();
  at::Tensor out = out_.has_value() ? *out_ : at::empty({total_q, hq, hdv}, q.options().dtype(out_dtype));
  at::Tensor lse = at::empty({hq, total_q}, q.options().dtype(at::kFloat));

  // Correctness-first envelope: bf16/fp16, hq%hkv==0, no fp8/splitkv. Paged OR non-paged. Head dims
  // padded to (hd_pad,hdv_pad) in {(128,128),(192,128),(64,64)}. softcap + attention-sink supported.
  bool dt_ok = (q.scalar_type()==at::kBFloat16 || q.scalar_type()==at::kHalf);
  int hdp = (hd+63)/64*64, hdvp = (hdv+63)/64*64;
  bool shape_ok = (hq%hkv==0) && ((hdp==128&&hdvp==128)||(hdp==192&&hdvp==128)||(hdp==64&&hdvp==64));
  bool leftpad = leftpad_k.has_value();
  // num_splits is a parallelization of the same KV reduction — computing the full range in one pass
  // yields identical output, so we accept any num_splits and ignore it (correct; forgoes the extra
  // parallelism until the SplitKV+combine perf path exists). cache_leftpad = mask keys < leftpad.
  bool feat_ok = (cu_seqlens_q.has_value() && !q_descale.has_value()
                  && (paged ? seqused_k.has_value() : cu_seqlens_k.has_value()));
  TORCH_CHECK(dt_ok && shape_ok && feat_ok,
      "fa4_blackwell: unsupported config (this build: bf16/fp16, hq%hkv==0, "
      "(hd,hdv) in {(<=128,==hd),(192,128),(64,64)}, no fp8). hd=", hd, " hdv=", hdv,
      " paged=", paged, " dtype=", q.scalar_type());

  // s_aux (attention sink): per-q-head logit, convert to fp32 for the kernel.
  at::Tensor s_aux_f;
  const float* s_aux_ptr = nullptr;
  if (s_aux.has_value()) { s_aux_f = s_aux->to(at::kFloat).contiguous(); s_aux_ptr = s_aux_f.data_ptr<float>(); }

  int leftpad_val = 0;
  if (leftpad) { auto lp = leftpad_k->to(at::kCPU); leftpad_val = lp.numel() ? lp.data_ptr<int>()[0] : 0; }

  const int* cu_k_ptr = cu_seqlens_k.has_value() ? cu_seqlens_k->data_ptr<int>() : nullptr;
  const int* pt_ptr = nullptr; int pt_stride = 0, block_size = 0;
  const int* seqused_k_ptr = nullptr;
  if (paged) {
    pt_ptr = page_table->data_ptr<int>(); pt_stride = (int)page_table->size(1);
    block_size = (int)k.size(1);   // paged K/V: (num_blocks, block_size, hkv, hd)
    seqused_k_ptr = seqused_k->data_ptr<int>();
  }

  // Fast path: non-paged contiguous -> WS pipeline; paged -> paged WS pipeline (+SplitKV); else serial.
  bool ws_ok = (!paged && !leftpad && cu_seqlens_k.has_value());
  int max_sk = max_seqlen_k.has_value() ? (int)*max_seqlen_k : 0;
  int max_sq_arg = max_seqlen_q.has_value() ? (int)*max_seqlen_q : 0;
  // NORMALIZE a degenerate sliding window to "no window" before routing. Callers (vLLM, Mistral-style
  // configs) commonly pass window_left == max context and window_right == 0 alongside causal=true, which
  // is *exactly* plain causal -- but every fast route (fwd_kernel_fa4, fwd_kernel_fa4_paged) gates on
  // win_l<0 && win_r<0, so the un-normalized form silently demotes to the slow generic kernel.
  //   * causal already implies key <= qpos, so any win_r >= 0 is redundant under causal.
  //   * qpos <= seqlen_k-1 <= max_sk-1 for bottom-right alignment, so win_l >= max_sk-1 never masks.
  {
    int wl=(int)window_size_left, wr=(int)window_size_right;
    if (is_causal && wr>=0) wr=-1;
    if (wl>=0 && max_sk>0 && wl>=max_sk-1) wl=-1;
    window_size_left=wl; window_size_right=wr;
  }
  if (ws_ok && launch_2sm(q, k, v, out, lse, *cu_seqlens_q, *cu_seqlens_k, hq, hkv, (float)softmax_scale,
                          is_causal, (int)window_size_left, (int)window_size_right, (float)softcap, s_aux_ptr, hd, hdv)) {
    // handled by opt-in 2-SM cluster prefill (FA4_2SM)
  } else if (ws_ok && launch_ws(q, k, v, out, lse, *cu_seqlens_q, *cu_seqlens_k, hq, hkv, (float)softmax_scale,
                         is_causal, (int)window_size_left, (int)window_size_right, (float)softcap, s_aux_ptr, hd, hdv,
                         max_sq_arg, max_sk)) {
    // handled by non-paged fast path
  } else if (paged && launch_paged_maybe_split(q, k, v, out, lse, *cu_seqlens_q, hq, hkv, (float)softmax_scale,
                         is_causal, (int)window_size_left, (int)window_size_right, (float)softcap, s_aux_ptr, hd, hdv,
                         pt_ptr, pt_stride, block_size, seqused_k_ptr, leftpad_val, (int)num_splits, max_sk, max_sq_arg)) {
    // handled by paged fast path
  } else {
    launch(q, k, v, out, lse, *cu_seqlens_q, cu_k_ptr, hq, hkv, (float)softmax_scale,
           is_causal, (int)window_size_left, (int)window_size_right, (float)softcap, s_aux_ptr, hd, hdv,
           paged?1:0, pt_ptr, pt_stride, block_size, seqused_k_ptr, leftpad_val, (int)num_splits, max_sk, max_sq_arg);
  }

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
  return at::zeros({1}, seqused_k.options().dtype(at::kInt));
}

}  // namespace fa4

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
  m.impl("fwd", torch::kCUDA, &fa4::mha_fwd);
  m.def("get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype, "
        "Tensor seqused_k, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, "
        "Tensor? seqused_q, Tensor? leftpad_k, int? page_size, int max_seqlen_k_new, "
        "bool is_causal, int window_size_left, int window_size_right, bool has_softcap, "
        "int num_splits, bool? pack_gqa, int sm_margin) -> Tensor");
  m.impl("get_scheduler_metadata", torch::kCUDA, &fa4::mha_fwd_get_scheduler_metadata);
}
