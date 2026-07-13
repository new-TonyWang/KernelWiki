---
title: Reduction Pattern -- Decision Tree
pattern_class: cuda-core
op: reduction
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: spec
  anchor: Reference
id: routing-reduction-INDEX
type: operator-routing
vendor: nvidia
operator: reduction
source_refs:
- source_id: source-code/cuda-samples
  path: Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
  anchor: L75-L91
- source_id: source-code/cuda-samples
  path: Samples/2_Concepts_and_Techniques/reduction/reduction.cpp
  anchor: L28-L62
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L26607-L26630
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
kernel_types:
- fused-kernel
confidence: inferred
tags:
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- fused-kernel
- cuda-cpp
---
# Reduction Pattern -- Decision Tree

This document guides the kernel-writing agent through a reduction task from a custom-kernel requirement to a working, optimized kernel.

## Scope

This decision tree covers custom-kernel implementation choices only. It starts after the task has been classified as requiring a dedicated kernel implementation.

## Step 1 -- Choose the custom reduction strategy

```
Q4. How many elements per reduction instance?
    <=32 (single warp)
        --> Warp-level butterfly reduction using __shfl_xor_sync.
            Each warp independently reduces its 32 (or fewer) elements
            in log2(32) = 5 shuffle steps. No shared memory needed.
            See wiki/nvidia/foundations/compute/warp-primitives/skill.md.

    33..1024 (single block)
        --> Block-level reduction:
            Phase 1: each warp reduces its 32 elements via __shfl_xor_sync.
            Phase 2: warp leaders write partial sums to shared memory.
            Phase 3: a single warp reduces the partial sums (<=32 values).
            Requires shared memory (see bank-conflict avoidance below).

    >1024 (multi-block / grid-level)
        --> Grid-level two-pass reduction:
            Pass 1: launch many blocks, each reduces a tile of the input
            to a single partial sum (using block-level reduction above).
            Write partial sums to a global buffer.
            Pass 2: reduce the partial sums (a second kernel launch, or
            ONE atomicAdd per block if combining into a scalar).
            Alternative: threadFenceReduction pattern (single-kernel with
            __threadfence() and atomicInc to detect the last block).
            See cuda-samples/threadFenceReduction/ for the canonical example.

            CRITICAL: the atomic tail must issue ONE atomic per block,
            NOT one per thread. A naive "every-thread atomicAdd to one
            scalar" pattern is both 248x slower AND numerically wrong
            for FP32 sums with N > ~1e5 (the running sum saturates the
            24-bit mantissa). Use skill wiki/nvidia/foundations/sync/atomic-reduction/
            (S1 hierarchical) for the grid-level combining step. On H200,
            S1 + grid-stride loop achieves 1498x speedup over naive and
            75% of HBM peak bandwidth for sum-reduction.
```

## Step 2 -- Optimization via ROUTING.md skills

After the basic custom kernel is working and correct, apply optimization skills from ROUTING.md in priority order:

1. **Coalescing** (wiki/nvidia/foundations/memory/coalescing/) -- ensure the input load phase uses coalesced (stride-1) global memory access. This is the single most impactful optimization for memory-bound reduction kernels.

2. **Warp primitives** (wiki/nvidia/foundations/compute/warp-primitives/) -- replace shared-memory tree reduction within a warp with register-based `__shfl_xor_sync` / `__shfl_down_sync` butterfly reduction. Eliminates shared-memory round-trips for the final 32-element reduce stage.

3. **Bank-conflict avoidance** (wiki/nvidia/foundations/memory/bank-conflict/) -- if the kernel uses shared memory for the block-level reduction tree, ensure the layout avoids bank conflicts. For a standard tree reduction, sequential addressing (reduce2 pattern from cuda-samples) is conflict-free; but interleaved addressing (reduce1 pattern) causes conflicts.

4. **Atomic reduction contention control** (wiki/nvidia/foundations/sync/atomic-reduction/) -- for the grid-level combining step in multi-block reductions, the atomic tail must issue **one atomic per block**, not one per thread. The S1 hierarchical pattern (warp shuffle -> shmem -> block-level atomicAdd) is required when the kernel's final stage is a global atomicAdd. Pair with skill 2 (warp-primitives, for the inner shuffle) and skill 9 (memory-ordering, for scope / ordering correctness). Measured on H200: 247.8x speedup over the naive per-thread atomic pattern; 1498x with grid-stride loop on top of S1.

After each skill application, re-benchmark against the task-provided baseline and follow the bottleneck-triage procedure in reasoning/bottleneck-triage.md.

## Step 3 -- Advanced techniques (Brent's theorem)

The cuda-samples reduce6 kernel demonstrates **Brent's theorem optimization**: each thread serially accumulates multiple elements from global memory before participating in the parallel reduction tree.  This increases arithmetic intensity and reduces the total number of threads (and blocks) needed, which can improve occupancy and reduce launch overhead for very large arrays (>1M elements).

```cuda
// Each thread accumulates multiple elements (grid-stride loop)
float mySum = 0;
unsigned int i = blockIdx.x * blockSize + threadIdx.x;
unsigned int gridSize = blockSize * gridDim.x;
while (i < n) {
    mySum += g_idata[i];
    i += gridSize;
}
// Then proceed with block-level reduction of mySum
```

## Cross-references

- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
