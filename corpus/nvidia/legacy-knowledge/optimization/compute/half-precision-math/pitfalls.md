# Half Precision Math -- Pitfalls

## P1: FP16 Overflow in Accumulation
**Symptom**: NaN or Inf values appearing in reduction or dot-product outputs when using fp16 accumulation.
**Detection**: Output contains NaN/Inf even though input values are reasonable. FP16 max value is ~65504; sums easily exceed this.
**Fix**: Always accumulate in FP32. Convert fp16 inputs to fp32 before summation, then convert the final result back. Use `__half2float()` / `__float2half_rn()` for conversion.
**Source**: Programming Guide, Section 5.5.2 (Floating-Point Data Types) -- fp16 largest value ~65504

## P2: BF16 Precision Loss in Computation
**Symptom**: BF16 computations show larger relative errors than fp16 for the same operation, despite having the same dynamic range as fp32.
**Detection**: Numerical comparison shows ~7-bit mantissa precision for bf16 vs ~10-bit for fp16.
**Fix**: Understand that BF16 has only 7 bits of mantissa (vs 10 for fp16). For operations requiring higher precision, use fp32 accumulation. BF16 is best suited for storing activations and weights, not for full-precision arithmetic chains.
**Source**: Programming Guide, Section 5.5.2 (Floating-Point Data Types)

## P3: Mixing half and half2 Causing Alignment Issues
**Symptom**: Misaligned memory access errors or performance degradation when switching between scalar `__half` and packed `__half2` access patterns.
**Detection**: Ncu shows uncoalesced memory access or misaligned load warnings.
**Fix**: When using `half2`, ensure the base pointer is 4-byte aligned (natural alignment for 2x 16-bit). Use `reinterpret_cast<half2*>` only on properly aligned pointers. Pad arrays to even lengths.
**Source**: PTX ISA, ld.global alignment requirements

## P4: FP16 Denormal Handling Differs from FP32
**Symptom**: Very small fp16 values (below ~6.1e-5) behave differently than expected, especially with `-ftz=true`.
**Detection**: Results differ between GPU and CPU for near-zero values.
**Fix**: Be aware that fp16 has a larger denormal range relative to its precision. If using `-ftz=true`, denormals are flushed to zero. For `atom.add.noftz.f16`, denormals are preserved. Test edge cases near the fp16 minimum normal value.
**Source**: PTX ISA, atom.add.noftz.f16 instruction

## P5: Implicit Type Conversion Overhead
**Symptom**: Kernel is slower than expected because the compiler inserts many `cvt` (conversion) instructions between fp16 and fp32.
**Detection**: Inspect SASS for excessive `F2F` (float-to-float conversion) instructions. Ncu shows high instruction count relative to arithmetic intensity.
**Fix**: Minimize conversions by keeping data in fp16/bf16 throughout the computation pipeline. Use packed `half2`/`bfloat162` operations. Convert to fp32 only for accumulation or operations that need wider precision. Use mixed-precision FMA (`fma.rn.f32.f16`) where available.
**Source**: Programming Guide, Section 5.5.2 (Floating-Point Data Types)

## P6: BF16 Not Available on Pre-Ampere Architectures
**Symptom**: Compilation error or runtime failure when using `__nv_bfloat16` operations on sm_70 or sm_75.
**Detection**: Compile error or `CUDA_ERROR_INVALID_SOURCE` at runtime.
**Fix**: Guard bf16 code with `#if __CUDA_ARCH__ >= 800`. Use fp16 as a fallback for older architectures. BF16 native arithmetic requires sm_80+.
**Source**: Programming Guide, Section 5.5.2 (Floating-Point Data Types) -- BF16 requires Compute Capability 8.0+

## P7: L1 cache hit rate dropped from 22% to 0% with half2 packing — the coarser access granularity may change cache behavior, though it was a net win here because instruction reduction dominates (discovered in verification)

**Symptom**: L1 cache hit rate dropped from 22% to 0% with half2 packing — the coarser access granularity may change cache behavior, though it was a net win here because instruction reduction dominates.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: L1 cache hit rate dropped from 21 (discovered in verification)

**Symptom**: L1 cache hit rate dropped from 21.9% to 0.0% in the optimized kernel — packed bf16x2 loads may change cache line reuse patterns; worth monitoring if combined with other optimizations that depend on L1 residency.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: Verification can pass even when the kernel is effectively a no-op under profiling; always cross-check that DRAM bytes read and instruction counts are in the same ballpark as baseline to confirm the kernel is doing equivalent work (discovered in verification)

**Symptom**: Verification can pass even when the kernel is effectively a no-op under profiling; always cross-check that DRAM bytes read and instruction counts are in the same ballpark as baseline to confirm the kernel is doing equivalent work.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: h2exp() may not map to a single native instruction on all architectures; on some GPUs it decomposes into multiple fp32 ops or multi-step approximation, increasing instruction count rather than reducing it (discovered in verification)

**Symptom**: h2exp() may not map to a single native instruction on all architectures; on some GPUs it decomposes into multiple fp32 ops or multi-step approximation, increasing instruction count rather than reducing it.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: On small, memory-bound workloads __hfma2_relu shows no measurable speedup despite reducing instruction count by ~4%; the benefit is masked by memory latency (discovered in verification)

**Symptom**: On small, memory-bound workloads __hfma2_relu shows no measurable speedup despite reducing instruction count by ~4%; the benefit is masked by memory latency. The skill's claim of a meaningful single-instruction advantage only holds for compute-bound kernels.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: Native fp16 atomicAdd has far worse contention throughput than fp32 atomicAdd on the same address; the smaller data type does not help when every warp in the grid collides on one location (discovered in verification)

**Symptom**: Native fp16 atomicAdd has far worse contention throughput than fp32 atomicAdd on the same address; the smaller data type does not help when every warp in the grid collides on one location.
**Source**: Level 3 sandbox verification (2026-04-05)

## P13: PyTorch Extension Build Disables Half-Precision Operators

**Symptom**: Compile errors like `no operator ">" matches these operands` or `no suitable constructor from "float" to "__half"` when using `__half` in CUDA kernels built via `torch.utils.cpp_extension`.
**Root cause**: PyTorch's extension build system adds these compiler flags:
- `-D__CUDA_NO_HALF_OPERATORS__` — disables `__half + __half`, `__half > __half`, etc.
- `-D__CUDA_NO_HALF_CONVERSIONS__` — disables implicit `float` → `__half` conversion
- `-D__CUDA_NO_HALF2_OPERATORS__` — disables `__half2` arithmetic operators

**Fix**: Use explicit function calls instead of operators:
- Comparison: `__hgt(a, b)` instead of `a > b`
- Arithmetic: `__hadd(a, b)` instead of `a + b`
- Conversion: `__float2half_rn(x)` instead of `__half h = x`
- Always `#include <cuda_fp16.h>` at the top of the .cu file

**Fix**: For `__half2`, use: `__hadd2()`, `__hmul2()`, `__hfma2()` etc.

**Important**: Do NOT use `__half` types in `.cpp` binding files — `__half` requires CUDA headers and is not available in plain C++ compiled with g++. Keep `__half` usage strictly in `.cu` files.
**Source**: KernelBench experiment analysis — 14 compile failures in Group A traced to these flags
