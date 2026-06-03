---
title: Half-Precision Math APIs
status: draft
source:
- path: spec
  anchor: Reference
apis:
- func_name: __hadd
  namespace: cuda-runtime
  kind: fp16-arith-scalar
  signature: __half __hadd(__half a, __half b);
  notes: Scalar fp16 add. Emits add.f16. Same throughput as fp32 on sm_9.0a scalar
    path (45 TFLOPS measured).
- func_name: __hadd2
  namespace: cuda-runtime
  kind: fp16-arith-packed
  signature: __half2 __hadd2(__half2 a, __half2 b);
  notes: Packed fp16 add (2 lanes). Emits add.f16x2. Measured 1.16x scalar __hadd
    throughput on H200, not 2x (see probe).
- func_name: __hmul
  namespace: cuda-runtime
  kind: fp16-arith-scalar
  signature: __half __hmul(__half a, __half b);
  notes: Scalar fp16 multiply. Emits mul.f16.
- func_name: __hmul2
  namespace: cuda-runtime
  kind: fp16-arith-packed
  signature: __half2 __hmul2(__half2 a, __half2 b);
  notes: Packed fp16 multiply. Emits mul.f16x2.
- func_name: __hfma
  namespace: cuda-runtime
  kind: fp16-arith-scalar
  signature: __half __hfma(__half a, __half b, __half c);
  notes: 'Scalar fp16 fused multiply-add (a*b + c). Emits fma.rn.f16. Measured on
    H200: 45351 GFLOPS (scalar op count), 1.58x scalar fp32 FMA.'
- func_name: __hfma2
  namespace: cuda-runtime
  kind: fp16-arith-packed
  signature: __half2 __hfma2(__half2 a, __half2 b, __half2 c);
  notes: 'Packed fp16 FMA (2 lanes). Emits fma.rn.f16x2. Measured on H200: 52608 GFLOPS
    (scalar op count), 1.16x scalar __hfma — the residual after nvcc''s auto-packing
    of scalar chains into HFMA2.MMA.'
- func_name: __hfma_relu
  namespace: cuda-runtime
  kind: fp16-fused-activation
  signature: __half __hfma_relu(__half a, __half b, __half c);
  notes: Scalar fp16 max(a*b + c, 0). Emits fma.rn.relu.f16. ~4% instruction reduction
    over explicit max; wall-clock benefit only in compute-bound kernels (see pitfall
    P11).
- func_name: __hfma2_relu
  namespace: cuda-runtime
  kind: fp16-fused-activation-packed
  signature: __half2 __hfma2_relu(__half2 a, __half2 b, __half2 c);
  notes: Packed fp16 max(a*b + c, 0). Emits fma.rn.relu.f16x2.
- func_name: __halves2half2
  namespace: cuda-runtime
  kind: fp16-pack
  signature: __half2 __halves2half2(__half lo, __half hi);
  notes: Build a half2 from two existing halves. Zero-cost on most paths (just register
    alias).
- func_name: __floats2half2_rn
  namespace: cuda-runtime
  kind: fp16-pack-from-float
  signature: __half2 __floats2half2_rn(float lo, float hi);
  notes: Build a half2 from two fp32 values with round-to-nearest-even. One cvt.rn.f16.f32
    per lane, then packed.
- func_name: __float2half_rn
  namespace: cuda-runtime
  kind: fp16-cast
  signature: __half __float2half_rn(float x);
  notes: fp32 -> fp16 round-to-nearest-even. Emits cvt.rn.f16.f32.
- func_name: __half2float
  namespace: cuda-runtime
  kind: fp16-cast
  signature: float __half2float(__half x);
  notes: fp16 -> fp32 widening. Emits cvt.f32.f16. No rounding needed (exact).
- func_name: hexp
  namespace: cuda-runtime
  kind: fp16-transcendental
  signature: __half hexp(__half x);
  notes: Scalar fp16 exp. May decompose into fp32 ops on some arches (see pitfall
    P10); sm_9.0a coverage not re-measured by this probe.
- func_name: h2exp
  namespace: cuda-runtime
  kind: fp16-transcendental-packed
  signature: __half2 h2exp(__half2 x);
  notes: Packed fp16 exp. Expected to map to dedicated ex2.approx.f16x2 but not re-measured.
- func_name: hlog
  namespace: cuda-runtime
  kind: fp16-transcendental
  signature: __half hlog(__half x);
  notes: Scalar fp16 log. Emits lg2.approx + scale.
- func_name: h2log
  namespace: cuda-runtime
  kind: fp16-transcendental-packed
  signature: __half2 h2log(__half2 x);
- func_name: hrsqrt
  namespace: cuda-runtime
  kind: fp16-transcendental
  signature: __half hrsqrt(__half x);
  notes: Scalar fp16 reciprocal sqrt. Emits rsqrt.approx.f16.
- func_name: h2rsqrt
  namespace: cuda-runtime
  kind: fp16-transcendental-packed
  signature: __half2 h2rsqrt(__half2 x);
- func_name: __hadd
  namespace: cuda-runtime
  kind: bf16-arith-scalar
  signature: __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b);
  notes: Overloaded for bf16 on sm_80+. Emits add.bf16. Same scalar throughput as
    fp16 (45 TFLOPS measured on H200).
- func_name: __hfma
  namespace: cuda-runtime
  kind: bf16-arith-scalar
  signature: __nv_bfloat16 __hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16
    c);
  notes: Overloaded for bf16. Emits fma.rn.bf16.
- func_name: __hfma2
  namespace: cuda-runtime
  kind: bf16-arith-packed
  signature: __nv_bfloat162 __hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162
    c);
  notes: 'Overloaded packed bf16 FMA. Emits fma.rn.bf16x2. Measured on H200: 46478
    GFLOPS — **12% slower** than fp16 packed (52608 GFLOPS); only **1.02x** scalar
    bf16 (the compiler already auto-packs scalar bf16 into HFMA2.MMA.BF16_V2, so explicit
    packing adds essentially nothing for bf16).'
- func_name: __float2bfloat16_rn
  namespace: cuda-runtime
  kind: bf16-cast
  signature: __nv_bfloat16 __float2bfloat16_rn(float x);
  notes: fp32 -> bf16 round-to-nearest-even. Emits cvt.rn.bf16.f32.
- func_name: __bfloat162float
  namespace: cuda-runtime
  kind: bf16-cast
  signature: float __bfloat162float(__nv_bfloat16 x);
  notes: bf16 -> fp32 widening. Emits cvt.f32.bf16.
- func_name: __floats2bfloat162_rn
  namespace: cuda-runtime
  kind: bf16-pack-from-float
  signature: __nv_bfloat162 __floats2bfloat162_rn(float lo, float hi);
- func_name: add.f16
  namespace: ptx
  kind: ptx-fp16-add-scalar
- func_name: add.f16x2
  namespace: ptx
  kind: ptx-fp16-add-packed
  notes: 2-way SIMD fp16 add. Throughput on sm_9.0a measured at 1.16x scalar add.f16.
- func_name: fma.rn.f16
  namespace: ptx
  kind: ptx-fp16-fma-scalar
- func_name: fma.rn.f16x2
  namespace: ptx
  kind: ptx-fp16-fma-packed
  notes: Emitted by __hfma2. Measured 52.6 TFLOPS on H200 (scalar op count).
- func_name: fma.rn.relu.f16
  namespace: ptx
  kind: ptx-fp16-fma-relu
  notes: Fused FMA + ReLU. Single instruction; max(a*b + c, 0).
- func_name: fma.rn.relu.f16x2
  namespace: ptx
  kind: ptx-fp16-fma-relu-packed
- func_name: fma.rn.bf16
  namespace: ptx
  kind: ptx-bf16-fma-scalar
- func_name: fma.rn.bf16x2
  namespace: ptx
  kind: ptx-bf16-fma-packed
  notes: Emitted by __hfma2 on __nv_bfloat162. Lowers to SASS HFMA2.MMA.BF16_V2 (audit
    2026-04-23) — distinct opcode from fp16's HFMA2.MMA, stays within the half-precision
    family (NOT a FFMA fallback). Measured 46.5 TFLOPS on H200 — 12% below fma.rn.f16x2.
- func_name: tanh.approx.f16
  namespace: ptx
  kind: ptx-fp16-tanh
  notes: Single-instruction fp16 tanh approximation.
- func_name: tanh.approx.bf16
  namespace: ptx
  kind: ptx-bf16-tanh
- func_name: cvt.rn.f16.f32
  namespace: ptx
  kind: ptx-cvt-fp32-to-fp16
- func_name: cvt.f32.f16
  namespace: ptx
  kind: ptx-cvt-fp16-to-fp32
  notes: fp16 -> fp32 widening. Exact; no rounding mode needed.
- func_name: cvt.rn.bf16.f32
  namespace: ptx
  kind: ptx-cvt-fp32-to-bf16
- func_name: cvt.f32.bf16
  namespace: ptx
  kind: ptx-cvt-bf16-to-fp32
- func_name: atom.add.noftz.f16
  namespace: ptx
  kind: ptx-fp16-atomic-add
  notes: Native fp16 atomicAdd. `.noftz` preserves denormals. Contention throughput
    is worse than fp32 atomicAdd (see pitfall P12); prefer hierarchical fp32 fan-in
    via the atomic-reduction skill.
- func_name: atom.add.noftz.bf16
  namespace: ptx
  kind: ptx-bf16-atomic-add
  notes: Native bf16 atomicAdd (sm_80+).
id: api-half-precision-math-ref
type: api-definition
vendor: nvidia
func_name: Half-Precision Math APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25350-L25420
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L11000-L11200
---
## Scalar FMA throughput table (H200 sm_9.0a, measured)

| intrinsic   | PTX emit          | SASS emit             | scalar GFLOPS | vs fp32 |
|-------------|-------------------|-----------------------|--------------:|--------:|
| `fmaf`      | `fma.rn.f32`      | `FFMA`                |         28633 |   1.00× |
| `__hfma`    | `fma.rn.f16`      | `HFMA2.MMA` (auto-packed by nvcc) |  45351 | **1.58×** |
| `__hfma2`   | `fma.rn.f16x2`    | `HFMA2.MMA`           |         52608 | **1.84×** |
| `__hfma` (bf16) | `fma.rn.bf16` | `HFMA2.MMA.BF16_V2`   |         45381 |   1.59× |
| `__hfma2` (bf16)| `fma.rn.bf16x2` | `HFMA2.MMA.BF16_V2` | 46478 |   1.62× |

Compute SOL (fp32 variant) = 78% at 4-chain ILP × 2048 FMAs/chain.

## Precision quick table (for caller annotation)

| format | exp | mantissa | max finite | min normal | eps at 1.0 | accum safe? |
|--------|----:|---------:|-----------:|-----------:|-----------:|:-----------:|
| fp32   |   8 |       23 |    3.4e38 |   1.18e-38 |   1.19e-7  |     yes     |
| fp16   |   5 |       10 |    6.55e4 |   6.10e-5  |   4.88e-4  |     **no**  |
| bf16   |   8 |        7 |    3.4e38 |   1.18e-38 |   7.81e-3  |     **no**  |

## Cross-references

- **`fast-math` skill**: `-use_fast_math` only affects fp32 paths; it does NOT change fp16/bf16 code generation.
- **`vectorized-access` skill**: `__half2` packs naturally to 4-byte aligned loads; use vectorized-access to widen LDS/STS instructions.
- **`ilp` skill**: the half2-throughput probe 4-chain FMA harness is the same pattern as the ILP skill's throughput methodology.
- **`register-pressure` skill**: `__half2` doubles accumulator width; a compute-bound kernel adopting §S2 must verify regs-per-thread.
- **`atomic-reduction` skill**: hierarchical fp32 fan-in + cast-to-fp16 at the final store is preferred over native fp16 `atomicAdd` under contention (skill §S6 + pitfall P12).

## Related Probes

- sources/experience/hw-probes/half2-throughput.md — 5-variant FMA throughput sweep. Load-bearing results: __hfma2 = 1.16× __hfma (not 2×); bf16-packed = 0.88× fp16-packed (not equal).
