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
---
# Reduction Pattern -- Decision Tree

This document guides the kernel-writing agent through a reduction task from initial problem statement to a working, optimized kernel. The decision tree enforces a **library-first** policy: only proceed to a custom kernel when the library path has been proven insufficient.

## Step 0 -- Try the library first

Before writing any custom CUDA code, check whether a production-quality library already handles the reduction.

```
Q0. Is the caller's environment PyTorch-based?
    YES --> Can torch.sum / torch.mean / torch.max / torch.min /
            torch.argmax / torch.argmin handle the shape + dtype + axis?
            YES --> Use the PyTorch op. DONE.
            NO  --> Continue to Q1.
    NO  --> Continue to Q1.

Q1. Is the reduction a flat (1-D) or axis-aligned reduction over a
    contiguous buffer with a standard operator (sum, min, max, argmin, argmax)?
    YES --> Use cub::DeviceReduce::Sum / Min / Max / ArgMin / ArgMax.
            See library-fallback.md for API details and usage examples.
            DONE.
    NO  --> Continue to Q2.

Q2. Is the reduction a simple fold with a custom binary operator over a
    flat, contiguous range?
    YES --> Use cub::DeviceReduce::Reduce with a user-defined functor,
            or thrust::reduce with a custom op.
            See library-fallback.md. DONE.
    NO  --> Continue to Q3.

Q3. Does the library path fail to meet performance requirements after
    benchmarking (e.g., fused reduction inside a larger kernel, non-standard
    axis layout, or measured >10% overhead vs. theoretical peak)?
    YES --> Proceed to Step 1 (custom kernel).
    NO  --> Re-examine the library path. Most reductions are well-served
            by cub::DeviceReduce. Only proceed to custom if benchmark
            evidence shows the library is insufficient.
```

**When to skip the library**: the library path is insufficient when:
- The reduction must be fused with preceding or following elementwise operations to avoid an extra global-memory round-trip.
- The reduction axis is non-contiguous and cannot be made contiguous via a reshape / permute without excessive overhead.
- A partial (per-row, per-column) reduction of a 2-D tensor is needed and the library's 1-D API would require repeated calls.
- The measured library latency exceeds the theoretical bandwidth-bound limit by more than 10% for the given shape.

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

After each skill application, re-benchmark against the baseline (torch.<op> or cub::DeviceReduce) and follow the bottleneck-triage procedure in reasoning/bottleneck-triage.md.

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

- **Library fallback details**: `library-fallback.md`
- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
