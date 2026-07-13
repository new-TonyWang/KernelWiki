---
id: code-cutlass-cute-wgmma_skeleton
type: code-walkthrough
vendor: nvidia
title: Wgmma_Skeleton
upstream_repo: NVIDIA/cutlass-cute
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- cluster
- fp8
techniques:
- warp-specialization
- shared-memory-optimization
- swizzling
kernel_types:
- gemm
- quantization
confidence: inferred
tags:
- wgmma
- cluster
- fp8
- warp-specialization
- shared-memory-optimization
- swizzling
- gemm
- quantization
- ptx
- cuda-cpp
- cute-dsl
---
# CUTLASS WGMMA Atom Skeleton (example 48, commit `f74fea9c`)

This is a reading guide for the CUTLASS / CuTe WGMMA atom path in `48_hopper_warp_specialized_gemm.cu`. The same source file is represented by the runnable repro under `artifacts/experience/api-probes/gemm/gemm_aligned.cu`; here we annotate only the CUTLASS symbols needed to understand which WGMMA instruction is being described.

The warp-specialized producer / consumer mainloop that surrounds these calls lives in `../example48-hopper-warp-specialized-gemm/mainloop_skeleton.md`.

## Mental Model

Read "atom" as "one hardware WGMMA instruction described as a C++ type."

The path is:

```text
MMA_64x128x8_F32TF32TF32_SS_TN   // atom type: one instruction shape
  -> MMA_Atom<...>               // CuTe wrapper around that atom
  -> TiledMMA                    // how this atom is repeated across a CTA tile
  -> cute::gemm(...)             // emits repeated wgmma.mma_async instructions
```

For example 48:

```text
CTA tile:       128 x 128 x 32
selected atom:   64 x 128 x  8
repeat pattern:   2 x   1 x  4 atom issues
```

So the atom is smaller than the CTA tile. It is the instruction-level building block that CuTe repeats. This file stops at that code-reading boundary; it does not try to recommend an atom for a new handwritten kernel.

## What cute emits

```
[device, consumer wg]   wgmma.fence.sync.aligned                    // make smem operands visible to the wgmma issue group
[device, consumer wg]   wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32.f32   // the actual MMA
[device, consumer wg]   wgmma.commit_group.sync.aligned             // mark issue boundary
[device, consumer wg]   wgmma.wait_group.sync.aligned 0             // block until N-1 groups have completed
```

cutlass-cute hides every step behind `cute::gemm(tiled_mma, sA, sB, accumulator)`. The `tiled_mma` object contains the atom (`MMA_64xNxK_F32TF32TF32_SS_TN<...>`) plus the repeat layout.

## Critical wgmma fragments

### A. Atom selection (compile-time)

Lines 110-134 of the vendored example 48 variant:

```cpp
using TileShape    = Shape<_128, _128, _32>;     // Threadblock-level tile (M, N, K)
using ClusterShape = Shape<_4,   _2,  _1>;       // Cluster of CTAs

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    ...,
    cutlass::gemm::collective::KernelScheduleAuto         // picks Cooperative / Pingpong / Multistage
  >::CollectiveOp;
```

`CollectiveBuilder` auto-selects the wgmma atom from `TileShape`, dtype, layouts, and operand source. The N dimension of `TileShape` controls the wgmma N, and therefore which atom in the `MMA_64xNxK` family is issued. For `TileShape = <_128,_128,_32>` and TF32 input, cute picks `MMA_64x128x8_F32TF32TF32_SS_TN<1, 1>`. Smaller TileShape N -> smaller atom; larger TileShape N -> larger atom. The atom-shape sweep exercises this knob.

`KernelScheduleAuto` is shown here only because it appears in the same template block. Its producer / consumer schedule choice belongs to the warp-specialization track.

### B. Issue + wait (runtime)

Inside the consumer warpgroup, the wgmma part of the loop is:

```cpp
warpgroup_fence_operand(accum);                       // wgmma.fence
cute::gemm(tiled_mma, sA, sB, accum);                 // emits N wgmma.mma_async issues
warpgroup_commit_batch();                             // wgmma.commit_group
warpgroup_wait<0>();                                  // wgmma.wait_group 0 — block until previous batch lands
```

`warpgroup_fence_operand` / `warpgroup_commit_batch` / `warpgroup_wait` are inline helpers in `include/cute/arch/mma_sm90.hpp` that wrap `__nvvm_intrinsic`-style PTX `wgmma.fence`/`wgmma.commit_group`/`wgmma.wait_group`.

### C. Pre-conditions (from the cute traits headers)

Decoded from the vendored example's compiled kernel signature:

- `MMA_Atom<MMA_64x128x8_F32TF32TF32_SS_TN<1,1>>` — atom is 64M × 128N × 8K, TF32 inputs, F32 accumulator, SS-form (both A and B from smem), TN layout (A row-major, B column-major).
- `Layout<tuple<C<2>, C<1>, C<1>>, ...>` — the atom is repeated 2× along M, so the consumer warpgroup covers 128 = 2 × 64 rows for this CTA tile.
- Smem layout uses `Swizzle<3,4,3>` over a 32-bit unit (canonical bank-conflict-free pattern for SS wgmma).

## Atom Family Names Seen In CUTLASS

| Family | M | N | K | Notes |
|---|---|---|---|---|
| F32 ← BF16 × BF16 | 64 | 8/16/32/64/96/128/192/256 | 16 | |
| F32 ← FP16 × FP16 | 64 | 8/16/32/64/96/128/192/256 | 16 | |
| F32 ← TF32 × TF32 | 64 | 8/16/32/64/96/128/192/256 | 8 | Example 48 uses this family. |
| F32 ← FP8 × FP8 | 64 | 8/16/32/64/96/128/192/256 | 32 | Need `e4m3` or `e5m2` dtype. |

## Cross-references

- Skill: `wiki/nvidia/hardware/wgmma/skill.md`
- Pitfalls: `wiki/nvidia/hardware/wgmma/pitfalls.md`
- Single-atom probe: `sources/experience/api-probes/gemm.md`
- Atom-shape sweep: `sources/experience/api-probes/gemm.md`
- Reproducible bundle: `artifacts/experience/api-probes/gemm/`
- Surrounding WS mainloop: `../example48-hopper-warp-specialized-gemm/mainloop_skeleton.md`
- Upstream (pinned): `cutlass@f74fea9c` (`{{CUTLASS_REPO_REF}}`); see `examples/48_hopper_warp_specialized_gemm/` and `include/cute/atom/mma_traits_sm90_gmma.hpp`.
- Authoritative blog: `corpus/nvidia/blogs/colfax/cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus`.
