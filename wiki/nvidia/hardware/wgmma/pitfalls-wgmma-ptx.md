---
id: pitfall-wgmma-ptx
type: pitfall
vendor: nvidia
title: Pitfalls
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
techniques:
- shared-memory-optimization
- swizzling
confidence: inferred
tags:
- wgmma
- tma
- shared-memory-optimization
- swizzling
- ptx
- cuda-cpp
- cute-dsl
---
# Hopper wgmma via raw PTX — pitfalls

## 1. `-arch=sm_90a` alone is insufficient for wgmma inline PTX

nvcc 12.9 with `-arch=sm_90a` generates an intermediate `compute_90` (NO `a` suffix) PTX file that ptxas rejects with `Instruction 'wgmma.fence' not supported on .target 'sm_90'`. The fix is to use the explicit `-gencode=arch=compute_90a,code=sm_90a` form, which forces both the intermediate compute target AND the SM target to the `a` (architecture-specific) variant. See `build.sh` in the artifacts/ directory for the exact flags.

## 2. wgmma is a warpgroup instruction — issue from exactly 128 threads

`wgmma.mma_async.sync.aligned.*` is documented as a warpgroup instruction: it must be issued by all 128 threads (4 warps) of a warpgroup simultaneously. Issuing it from a single warp, or from 64 / 192 / 256 threads, is undefined behavior. The hello-world kernel uses `<<<1, 128, smem_bytes>>>` exactly for this reason.

## 3. Smem descriptor must be 16-byte aligned

The descriptor's address bits store `smem_addr >> 4`, so the smem allocation start must be 16-byte aligned. `extern __shared__ uint8_t smem[]` is naturally 16-byte aligned by CUDA's smem allocator, but if you carve up the buffer with non-16-byte-aligned offsets (e.g. an arbitrary uint8_t offset), the descriptor will silently encode the wrong start address.

## 4. wgmma.fence + wgmma.wait_group are not optional

Without the `wgmma.fence.sync.aligned` before the issue, the wgmma may race with previous shared-memory stores; without `wgmma.wait_group.sync.aligned 0` after, the per-thread fragment registers are not yet committed when downstream code reads them. Both fences MUST appear in the issue sequence, in the order shown in `wgmma_hello.cu`.

## 5. The atom shape `m64nNkK` constrains the per-thread fragment count

For `m64n8k16` bf16: 64×8 = 512 outputs, distributed across 128 threads = 4 floats per thread, packed as 2 register pairs. For `m64n128k8` tf32 (the cutlass-API reference): 64×128 = 8192 outputs, 64 floats per thread, 32 register pairs. The accumulator clobber list in the inline asm must enumerate all of them. Get the count wrong and ptxas will silently truncate or the kernel will hang at the wgmma issue.

## 6. The swizzle mode interacts with leading-dim / stride-dim encoding

For non-swizzled smem layouts, `swizzle = 0` and the leading-dim / stride-dim fields encode the actual byte stride. For 128B-swizzle layouts (`swizzle = 1`), the leading-dim field's interpretation changes — see PTX ISA spec section "Hopper Tensor Core Operations" for the table. This skill's hello-world uses no swizzle; production kernels typically use 128B swizzle for higher TMA bandwidth.

## 7. Numerically equivalent to cutlass's wgmma at the same atom

The cutlass `cute::SM90::GMMA::MMA_64x128x8_F32TF32TF32_SS_TN` atom compiles to the same `wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32` instruction this skill issues by hand. The expected output is bit-identical for the same A/B inputs. If they diverge, the descriptor encoding (or per-thread fragment layout) is wrong; verify by setting up an identical input pair and comparing element-by-element.

## 8. PTX immediate count varies by dtype family

The `wgmma.mma_async` instruction takes a different number of trailing immediates depending on the dtype suffix. Get this wrong and ptxas rejects with `Arguments mismatch for instruction 'wgmma.mma_async'`:

| Dtype suffix | Immediates after `descA, descB` | Form |
|---|---|---|
| `.f32.bf16.bf16`, `.f32.f16.f16`, `.f16.f16.f16` | 5 | `scaleD, scaleA, scaleB, transA, transB` |
| `.f32.tf32.tf32` | 3 | `scaleD, scaleA, scaleB` (A and B are fixed K-major; no trans) |
| `.s32.s8.s8`, `.s32.u8.u8`, mixed-sign | 1 | `scaleD` only (no scaleA / scaleB / trans) |

This is easy to miss because the ptxas error message points only at the line number. The codegen at `artifacts/experience/hw-probes/wgmma-ptx/gen_wgmma_zoo.py` enumerates the cases.

## 9. TF32 wgmma supports only the TN format

For `.f32.tf32.tf32`, A and B are *spec-fixed* to K-major; there is no `transA`/`transB` immediate (and consequently no SS_NT / SS_NN / SS_TT variant). cute reflects this — only `MMA_64xNx8_F32TF32TF32_SS_TN` exists in `mma_traits_sm90_gmma.hpp`, with no NT/NN/TT siblings. By contrast, bf16/fp16 atoms exist in all four layouts.

## 10. RS variant: A is always K-major

For `wgmma_RS_<...>` (A in registers, B in smem), only `transB` is a free immediate; A's layout is fixed by the in-register fragment format. The PTX form is:
```
wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 {acc...}, {a0,a1,a2,a3}, descB, scaleD, scaleA, scaleB, transB;
```
For bf16 / fp16 m64xK16, each thread holds 4 × 32-bit registers (= 8 fp16/bf16 elements per K-row slice). Where the bytes for those registers come from is *not* dictated by the wgmma instruction — it is the kernel's job to arrange them per the per-thread fragment layout in PTX ISA section "Asynchronous Warpgroup-Level Matrix Instructions / Matrix Fragments". Get the load wrong and the result is *layout-permuted but not catastrophic*; an all-ones-input correctness gate cannot detect this. Use non-uniform inputs to expose the bug.

## 11. Single-warpgroup serialized wgmma plateaus at ~5.6 TFLOPS on H200

A naive one-warpgroup-per-CTA kernel with a single accumulator chain caps near 5.6 TFLOPS regardless of N — the wgmma engine is throughput-bound at the issue port. The path to higher throughput on H200 is *not* a larger atom; it is **multi-warpgroup-per-CTA** (cutlass cooperative scheme uses 2 producer + 2 consumer warpgroups), **multi-accumulator pipelining** (typically 2-4 chains so consecutive wgmmas don't depend on each other), and **multi-CTA scaling** (132 SMs). These three multipliers compose to lift the floor by ~1000× and reach the ~990 TFLOPS device-peak figure. Treat the 5.6 TFLOPS plateau as a baseline / single-instance characterization, not a target.

## 12. Dtype is a precision/range knob, not a throughput knob

At fixed atom shape (N=64, SS_TN), bf16 / fp16 / tf32 / s8 all issue at the same per-instance rate. The TFLOPS-difference reported in benchmarks is purely from the per-instance MAC count (= K's worth of scalar MACs). Choose dtype on the precision × dynamic-range trade alone — there is no "faster" dtype on Hopper wgmma at the same K.
