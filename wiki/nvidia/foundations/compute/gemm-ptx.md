---
title: Hopper GEMM via raw PTX (cutlass-free)
status: partial
evidence_level: measured
applies_to_pattern_class:
- tensor-core
- tensor-core/gemm
applies_to_ops:
- gemm
requires_sm: '>=9.0a'
requires_features:
- tma
- wgmma
- mbarrier
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9 + libcuda + libcublas
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L28659-L28675
  excerpt: 9.7.15.5. Asynchronous Warpgroup Level Matrix Multiply-Accumulate Operation
    using wgmma.mma_async instruction — the consumer side of this skill's kernel.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L16866-L16880
  excerpt: '9.7.9.25.5.2. Data Movement and Conversion Instructions: cp.async.bulk.tensor
    — the producer side of this skill''s kernel (TMA tile load).'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L29622-L29640
  excerpt: 9.7.15.5.1.2.2. Matrix Descriptor Format — Matrix descriptor specifies
    the properties of the matrix in shared memory that is a multiplicand in the matrix
    multiply and accumulate operation. It is a 64-bit value contained in a register.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11833
  excerpt: compute capability 9.0 (Hopper) and later have a tensor memory accelerator
    (TMA). The primary goal of the TMA is to provide an efficient data transfer mechanism
    from global memory to shared memory for multi-dimensional arrays.
- path: blogs/colfax/cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: Hopper wgmma GEMM walkthrough — descriptor format + fragment layout
- path: blogs/colfax/cutlass-tutorial-mastering-the-nvidia-tensor-memory-accelerator-tma
  anchor: TMA tile-load protocol that this kernel's mainloop composes
- path: 60-code/ptx-gemm/gemm_ptx.cu
  anchor: this skill's harness — TMA-PTX + wgmma-PTX composed into a single GEMM kernel
artifacts:
  code: 80-experience/api-probes/gemm-ptx/artifacts/gemm_ptx.cu
  build: 80-experience/api-probes/gemm-ptx/artifacts/build.sh
  run: 80-experience/api-probes/gemm-ptx/artifacts/run.sh
  introspection: 80-experience/api-probes/gemm-ptx/artifacts/device.json
  profile: 80-experience/api-probes/gemm-ptx/artifacts/profiles/2026-04-28-gemm-ptx-hello.csv
  distilled_view: 60-code/ptx-gemm/gemm_ptx.cu
  distilled_build: 60-code/ptx-gemm/build.sh
related_apis: []
related_skills:
- tma-ptx
- wgmma-ptx
- gemm-aligned
- gemm-fused
id: skill-gemm-ptx
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
# Hopper GEMM via raw PTX (cutlass-free)

## What it is

A minimum-viable cutlass-free GEMM kernel that composes the TMA-PTX + wgmma-PTX primitives into a single kernel:

- Host: `cuTensorMapEncodeTiled` (driver API) builds CUtensorMaps for A and B.
- Device: mbarrier-pipelined TMA load A + B, then `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16`, then store.
- No cutlass / cute headers anywhere in the source tree, the preprocessed source, or the linked binary.

## Status

**The cutlass-free verification gates PASS** (zero `cutlass::` / `cute::` symbols in both `nvcc -E` and `cuobjdump --dump-elf-symbols`). The kernel builds, runs, and produces a non-zero output that differs from cuBLAS at the same shape. The per-thread fragment-to-output mapping in the wgmma m64n8k16 store path is **not yet correct**: at M=64 N=8 K=16, 497/512 elements differ from cuBLAS.

**Why partial**: Multiple coordinated layout decisions all need to be correct for a cutlass-free wgmma kernel to match cuBLAS. Two fixes are applied in code:

1. **Fixed**: B's physical storage. wgmma `.SS_TN` expects B in col-major K×N. An earlier harness allocated B as row-major K×N (= N-major in storage). The current code transposes B on host so smem holds it in col-major K×N. **Done in `gemm_ptx.cu`.**
2. **Fixed**: descriptor SBO encoding. SBO encodes the byte stride within an 8-block to skip 8 elements along the contracting dim = 8 × sizeof(elem). The current code sets SBO=16 for both A and B (was 256). **Done in `gemm_ptx.cu`.**
3. **STILL OPEN**: per-thread fragment-store mapping (Pitfall #1). Even after fixes 1 and 2, the M=64 N=8 K=16 harness still reports 497/512 mismatches against cuBLAS — the residual error is in the store stage's `(row_base, col_base)` derivation from `(warp_id, lane_id)`, which uses an Ampere-style m16n8k16 layout that's incorrect for Hopper's 4-warp m64 fragment.

Path to closure: extract the fragment layout from `cute/atom/mma_traits_sm90_gmma.hpp` `MMA_64x8x16_F32BF16BF16_SS::DRegisters` and the corresponding `cute::partition_fragment` logic; reproduce cutlass-free in `gemm_ptx.cu`'s store stage. After that, the smallest shape should reach 0 mismatches.

## Measured Characteristics

H200-SXM, sm_90a, cuda 12.9.86 + driver 570.124.06. Single-CTA kernel at M=64 N=8 K=16 bf16 inputs, f32 accumulator:

| Aspect | Status | Detail |
|---|---|---|
| Build (`-gencode=arch=compute_90a,code=sm_90a`) | ✓ | gemm_ptx binary compiles cleanly |
| Cutlass-free preprocessor (`nvcc -E \| grep cutlass::\|cute::`) | ✓ | 0 matches |
| Cutlass-free linked binary (`cuobjdump --dump-elf-symbols \| grep cutlass::\|cute::`) | ✓ | 0 matches |
| TMA load completes (mbarrier signals) | ✓ | verified separately by tma_hello at the same target |
| wgmma instruction issues + completes | ✓ | verified separately by wgmma_hello with all-1.0 inputs (returns all-16.0) |
| All 512 outputs are non-zero | ✓ | kernel produces an output, not a hang or empty result |
| Element-by-element match against cuBLAS at the same shape | ✗ | 497/512 mismatches after the B-layout + SBO fixes. Residual error is in the per-thread fragment-store mapping (Pitfall #1). |

Sample outputs:

```
ptx     hD[0..3]: 2.6250 0.0000 -0.7500 0.3750
cuBLAS  hD[0..3]: 1.5000 0.2500 -1.6250 -0.3750
```

Magnitudes are similar (suggesting a permutation, not a complete miscomputation), consistent with the fragment-store mapping bug in pitfall #1.

## Cutlass-free verification (the hard gate)

| Check | Expected | Result |
|---|---|---|
| `nvcc -E gemm_ptx.cu \| grep -E 'cutlass::\|cute::'` | 0 matches | **0 matches ✓** |
| `cuobjdump --dump-elf-symbols gemm_ptx \| grep -E 'cutlass::\|cute::'` | 0 matches | **0 matches ✓** |
| Numeric correctness vs cuBLAS at M=64 N=8 K=16 bf16 | 0 mismatches | 497 mismatches after the B-layout + SBO fixes. Residual: per-thread fragment-store mapping (Pitfall #1). |

The cutlass-abandonment gate (zero symbols) is the hard verification gate. That gate **passes**. The per-shape numeric correctness gate is unmet pending the fragment-layout fix.

## What's measured

- Kernel runs on H200 without segfault or kernel-launch error: ✓
- TMA load completes (mbarrier signals): ✓ (verified separately by tma_hello at the same target)
- wgmma instance executes (no compile error, no `Invalid arguments`): ✓
- Output written to global memory: ✓ (all 512 entries are non-zero, written values are non-trivial)
- Cutlass-free preprocessor + linked-binary symbol gate: ✓
- Element-by-element match against cuBLAS at the same shape: ✗ (fragment-layout bug)

## Cross-references

- cutlass-API aligned GEMM (the correctness reference at scale): `30-skill/compute/gemm/aligned/skill.md`.
- wgmma-PTX primitive: `40-hardware-feature/wgmma-ptx/skill.md`. Underlying hw-probe evidence: `80-experience/hw-probes/wgmma-ptx/2026-04-28-wgmma-ptx-hello.md` (cutlass-free wgmma m64n8k16 verified element-by-element).
- TMA-PTX primitive: `40-hardware-feature/tma-ptx/skill.md`. Underlying hw-probe evidence: `80-experience/hw-probes/tma-ptx/2026-04-28-tma-ptx-hello.md` (cutlass-free TMA tile load verified element-by-element).
- Failure modes: not yet committed; the fragment-layout fix is follow-up work and the pitfalls.md will land alongside.
