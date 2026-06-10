---
title: Half-Precision Math — Pitfalls
status: verified
related_skill: wiki/nvidia/foundations/compute/half-precision-math/skill.md
source:
- path: spec
  anchor: Reference
experience_refs:
- sources/experience/hw-probes/half2-throughput.md
id: pitfall-half-precision-math
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25350-L25420
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L11000-L11200
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
techniques:
- pipeline-stages
- vectorized-loads
- kernel-fusion
- shared-memory-optimization
- software-exp
kernel_types:
- gemm
- fused-kernel
confidence: inferred
tags:
- pipeline-stages
- vectorized-loads
- kernel-fusion
- shared-memory-optimization
- software-exp
- gemm
- fused-kernel
- ptx
- cuda-cpp
---
Pitfalls group: P1–P6 are numerical-format and availability facts grounded in PG §5.5.2 / PTX ISA; P13 is a build-system fact; P14–P16 are measured on H200 sm_9.0a by the half2-throughput probe 2026-04-23. P10/P11/P12 are open risks not re-measured by the probe — each records what to look for in SASS / wall-clock and flags a follow-up probe.

## P1. FP16 overflow in accumulation

**Symptom**: NaN or Inf in reduction / dot-product outputs using fp16 accumulation.
**Detection**: Output contains NaN/Inf even when inputs are bounded. fp16 max is ~**65504**; any sum of ~1000 terms of magnitude ~100 exceeds it.
**Fix**: Always accumulate in FP32. Convert fp16 inputs via `__half2float` before summation, cast back with `__float2half_rn` at the final store. Skill §S3 is the canonical pattern.
**Numerical grounding**: fp16 mantissa ULP at 1.0 is 4.88e-4; summing 1000 terms of magnitude 1.0 accumulates rounding error ~0.5 per unit of sum, hitting 1 ULP at sum ≈ 2000. **Do not use fp16 for reductions.**
**Source**: PG §5.5.2; skill §Precision.

## P2. BF16 precision loss in computation

**Symptom**: BF16 computations show larger relative errors than fp16 for the same operation despite equivalent dynamic range.
**Detection**: Numerical comparison shows ~**7-bit mantissa** precision for bf16 vs ~**10-bit** for fp16 (16× coarser eps).
**Fix**: Understand bf16's mantissa limit. Use it for storage (weights, activations) where range matters more than precision; **not** for full-precision arithmetic chains. Promote to fp32 for accumulation (skill §S3).
**Measured on H200 sm_9.0a**: bf16 scalar FMA throughput equals fp16 scalar (both ~45 TFLOPS), so the choice of bf16 vs fp16 is precision/range-driven, not throughput-driven — see skill §Precision table.
**Source**: PG §5.5.2; half2-throughput probe 2026-04-23.

## P3. Mixing `__half` and `__half2` alignment

**Symptom**: Misaligned memory-access errors or performance degradation when switching between scalar `__half` and packed `__half2` access patterns.
**Detection**: NCU reports uncoalesced access or unaligned load warnings; wall-clock regresses after packing.
**Fix**: When using `__half2`, ensure the base pointer is **4-byte aligned** (natural alignment for 2× 16-bit). Use `reinterpret_cast<__half2*>` only on properly aligned pointers. Pad arrays to even lengths so tail elements don't require scalar fixup.
**Source**: PTX `ld.global` alignment requirements (PTX ISA §8.5); storage-format property not H200-specific.

## P4. FP16 denormal handling differs from FP32

**Symptom**: Very small fp16 values (below ~6.1e-5, the min-normal threshold) behave differently than expected — especially with `-ftz=true` (flush-to-zero) compiler option.
**Detection**: Results differ between GPU and CPU reference for near-zero values; normalisation kernels with eps < 6e-5 produce unexpected zeros.
**Fix**: Know that fp16 has a **larger denormal range relative to its precision** than fp32. With `-ftz=true`, denormals flush to zero. PTX `.noftz` variants (e.g. `atom.add.noftz.f16`) preserve denormals through that specific instruction. Verify your epsilons: for fp16 use eps ≥ 1e-4; for bf16 the min-normal is fp32-level so not a concern.
**Source**: PTX ISA `atom.add.noftz.f16`; skill §Precision.

## P5. Implicit type-conversion overhead

**Symptom**: Kernel is slower than expected because the compiler inserts many `cvt` instructions between fp16 and fp32.
**Detection**: SASS dump shows excessive `F2F` (float-to-float conversion) instructions; NCU reports high instruction count relative to arithmetic intensity.
**Fix**: Minimise conversions by keeping data in fp16/bf16 through the computation pipeline. Use packed `half2` / `bfloat162` operations where helpful. Convert to fp32 **only** at the accumulation boundary. Consider mixed-precision FMA (`fma.rn.f32.f16` available via some intrinsics) when the hardware supports it.
**Measured on H200 sm_9.0a (indirect)**: the half2-throughput probe scalar fp16 variant runs at 45.4 TFLOPS — if implicit conversions dominated, we would see scalar fp16 closer to scalar fp32 (28.6 TFLOPS). Absence of that collapse is weak evidence that the compiler is not conversion-bound in a pure FMA chain. In mixed workloads the risk remains.
**Source**: PG §5.5.2; half2-throughput probe 2026-04-23.

## P6. BF16 is sm_80+ only

**Symptom**: Compilation error or runtime failure when using `__nv_bfloat16` operations on sm_70 / sm_75.
**Detection**: Compile error or `CUDA_ERROR_INVALID_SOURCE` at runtime.
**Fix**: Guard bf16 code with `#if __CUDA_ARCH__ >= 800`. Fall back to fp16 for older architectures. BF16 native arithmetic requires Compute Capability 8.0+.
**Measured on H200 sm_9.0a**: all bf16 intrinsics tested (`__hadd`, `__hfma`, `__hfma2`, `__float2bfloat16_rn`, `__floats2bfloat162_rn`) compile and execute correctly — consistent with sm_9.0a ⊇ sm_8.0. The "not available on sm_70/75" side is a documented constraint inherited from the PTX ISA.
**Source**: PG §5.5.2; half2-throughput probe 2026-04-23.

## P10. `h2exp` may decompose into fp32 on some architectures

**Symptom**: `h2exp` does not map to a single native instruction; it decomposes into a multi-step fp32 approximation that increases instruction count rather than reducing it.
**Detection**: Inspect SASS; expect a single `ex2.approx.f16x2` but may see `ex2.approx.f32` + conversions.
**Fix**: Verify per architecture. If the decomposition happens on your target, either do the exp in fp32 explicitly (saves the wrapper overhead) or use `__expf` + cast at the boundary.
**Status**: Not re-measured on H200 sm_9.0a. Follow-up probe `sources/experience/hw-probes/half-transcendental/` (open).
**Source**: PG §5.4.11.2 (documents the packed intrinsics but does not guarantee single-instruction lowering on all SMs).

## P11. `__hfma2_relu` benefit invisible in memory-bound kernels

**Symptom**: Switching to `__hfma2_relu` reduces instruction count by ~4% but produces no measurable wall-clock speedup.
**Detection**: Compare wall-clock of (FMA then max) vs `__hfma2_relu` variants on your actual kernel, not a microbench.
**Fix**: Use `__hfma2_relu` only when the kernel is compute-bound (skill §S5). In the memory-bound regime the activation epilogue is a tiny fraction of wall-clock; code complexity not worth the 4% instruction saving.
**Status**: Not re-measured on H200 sm_9.0a. Follow-up probe `sources/experience/hw-probes/half-fma-relu/` (open).
**General principle**: instruction-count reduction ≠ wall-clock speedup; always measure end-to-end before committing to a fused intrinsic.

## P12. Native fp16 atomicAdd has worse contention than fp32

**Symptom**: `atomicAdd(__half*, __half)` on a contended destination has lower effective throughput than `atomicAdd(float*, float)` on the same address.
**Detection**: Compare the contended-atomic throughput of fp16 vs fp32 destinations under the same launch shape.
**Fix**: Use the hierarchical fan-in pattern from `atomic-reduction` skill — accumulate per-block in shared memory in fp32, then atomicAdd a **single fp32 partial per block** into the global reducer. Cast to fp16/bf16 only at the final store.
**Status**: Not re-measured on H200 sm_9.0a. Follow-up probe (open).
**Source**: Documented architectural constraint that native half-precision atomics share a lower-throughput atomic pipe vs fp32; exact H200 ratio unmeasured by the half2-throughput probe.

## P13. the external framework extension build disables half operators

**Symptom**: Compile errors like `no operator ">" matches these operands` or `no suitable constructor from "float" to "__half"` when using `__half` in CUDA kernels built via `framework extension build`.
**Root cause**: Framework extension build adds
- `-D__CUDA_NO_HALF_OPERATORS__` — disables `__half + __half`, `__half > __half`, etc.
- `-D__CUDA_NO_HALF_CONVERSIONS__` — disables implicit `float` → `__half`.
- `-D__CUDA_NO_HALF2_OPERATORS__` — disables `__half2` arithmetic operators.
**Fix**: Use explicit function calls instead of operators:

```cpp
// Not: a > b, a + b, __half h = 3.14f;
// Use:
bool cond = __hgt(a, b);
__half s = __hadd(a, b);
__half h = __float2half_rn(3.14f);
```

For `__half2` use `__hadd2`, `__hmul2`, `__hfma2`. Always `#include <cuda_fp16.h>`. Do **not** use `__half` types in `.cpp` files — keep them strictly in `.cu` sources compiled by nvcc.
**Source**: Framework extension build build flags (external framework constraint, not H200-specific).

## P14. `__hfma2` is 1.16× scalar `__hfma` on H200, not 2× (measured)

**Symptom (measured on H200 sm_9.0a, 2026-04-23)**: Packed `__hfma2` runs at **52608 GFLOPS** (scalar op count), only **1.16×** the scalar `__hfma` rate of 45351 GFLOPS. Documentation and folklore often imply a 2× advantage from packing.
**Cause (SASS-confirmed 2026-04-23)**: `cuobjdump --dump-sass` on the probe binary shows BOTH `fma_bench_fp16_scalar` AND `fma_bench_fp16_packed` emit the SAME `HFMA2.MMA` SASS opcode. nvcc auto-packs the 4 independent scalar `__hfma` chains into `HFMA2.MMA` instructions at codegen time, so "scalar vs packed" at C-source is largely erased at the hardware opcode level. The residual 1.16× is register-layout / alignment overhead on the explicit-packed path, not the packing gain itself. **The scalar fp16's 1.58× vs fp32 is itself the packing gain** — just made invisible by the compiler.
**Fix**: Do not promise 2× throughput from packing to downstream callers. The real multiplier over scalar fp32 is **1.84×**, and most of that (1.58×) comes from just switching fp32 → fp16, with packing as a secondary 16% gain. In memory-bound kernels, skip packing entirely — the win is not measurable.
**Source**: half2-throughput probe 2026-04-23.

## P15. BF16 packed is 12% slower than FP16 packed on H200 (measured)

**Symptom (measured on H200 sm_9.0a, 2026-04-23)**: `__hfma2` on `__nv_bfloat162` runs at **46478 GFLOPS**, **12% slower** than the same intrinsic on `__half2` at **52608 GFLOPS**. Scalar bf16 and scalar fp16 match at 45 TFLOPS; the divergence is only on the packed path.
**Cause (SASS-confirmed 2026-04-23)**: `fma.rn.bf16x2` lowers to `HFMA2.MMA.BF16_V2` SASS, a distinct opcode from fp16 packed's `HFMA2.MMA`. Both stay within the half-precision HFMA2 family — not a fall-through to `FFMA`. The BF16_V2 variant simply has lower throughput than the base HFMA2.MMA on sm_9.0a hardware; root cause (pipeline width, multiplier latency, register port allocation) requires an isolated-instruction microbench to resolve and is out of scope here.
**Fix**: Prefer fp16 packed when the workload's dynamic range fits fp16 (values in [6e-5, 6.5e4]). If range forces bf16 — training, un-normalised activations, deep transformers — accept the 12% cost as the price of the wider dynamic range.
**Source**: half2-throughput probe 2026-04-23.

## P16. `__hfma` / `__hfma2` overloads must come from `<cuda_bf16.h>` for bf16 (methodology)

**Symptom**: Using `__hfma(a, b, c)` where `a, b, c` are `__nv_bfloat16` without `#include <cuda_bf16.h>` triggers ambiguous overload or cast-to-fp16 at compile time. Runtime result is either zero or garbage (values silently truncated through fp16).
**Detection**: Compile warning "implicit conversion from __nv_bfloat16 to __half"; measured kernel produces wrong numerics at ~1e-4 scale.
**Fix**: Always `#include <cuda_bf16.h>` alongside `<cuda_fp16.h>` when the kernel uses both types. The bf16 header defines overloads of `__hfma`, `__hadd`, `__hmul` for the bf16 argument types.
**Discovered while writing the half2-throughput probe harness**: the probe's bf16 variants would have silently converted to fp16 without the header guard — the bf16-packed 46 TFLOPS result depends on the correct overload resolution.
**Source**: half2-throughput probe 2026-04-23 methodology note.
