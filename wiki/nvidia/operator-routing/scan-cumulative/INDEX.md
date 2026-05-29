---
title: Scan (Cumulative) Pattern -- Decision Tree
pattern_class: cuda-core
op: scan-cumulative
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: spec
  anchor: Reference
id: routing-scan-cumulative-INDEX
type: operator-routing
vendor: nvidia
operator: scan-cumulative
source_refs:
- source_id: source-code/cuda-samples
  path: Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
  anchor: L55-L131
- source_id: source-code/cuda-samples
  path: Samples/2_Concepts_and_Techniques/scan/scan.cu
  anchor: L43-L62
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L28474-L28508
---
# Scan (Cumulative) Pattern -- Decision Tree

This document guides the kernel-writing agent through a scan (prefix sum / cumulative sum) task from initial problem statement to a working, optimized kernel. The decision tree enforces a **library-first** policy: only proceed to a custom kernel when the library path has been proven insufficient.

Scan operators covered: inclusive scan, exclusive scan, cumulative sum (cumsum), prefix sum, prefix max/min, and segmented variants.

Key distinction from reduction: a scan produces an output element for every input element (output length == input length), whereas a reduction collapses the input to a single value (or one value per reduction axis).

## Step 0 -- Try the library first

Before writing any custom CUDA code, check whether a production-quality library already handles the scan.

```
Q0. Is the caller's environment PyTorch-based?
    YES --> Can torch.cumsum (or torch.cumprod / torch.cummax / torch.cummin)
            handle the shape + dtype + axis?
            YES --> Use the PyTorch op. DONE.
            NO  --> Continue to Q1.
    NO  --> Continue to Q1.

Q1. Is the scan a flat (1-D) or axis-aligned scan over a contiguous
    buffer with a standard operator (sum, max, min)?
    YES --> Use cub::DeviceScan::InclusiveSum / ExclusiveSum (for sum)
            or cub::DeviceScan::InclusiveScan / ExclusiveScan (for
            custom ops such as max, min).
            See library-fallback.md for API details and usage examples.
            DONE.
    NO  --> Continue to Q2.

Q2. Is the scan a simple prefix fold with a custom binary operator
    over a flat, contiguous range?
    YES --> Use cub::DeviceScan::InclusiveScan / ExclusiveScan with a
            user-defined functor, or thrust::inclusive_scan /
            thrust::exclusive_scan with a custom op.
            See library-fallback.md. DONE.
    NO  --> Continue to Q3.

Q3. Does the library path fail to meet performance requirements after
    benchmarking (e.g., fused scan inside a larger kernel, segmented
    scan with non-standard segment boundaries, or measured >10%
    overhead vs. theoretical peak)?
    YES --> Proceed to Step 1 (custom kernel).
    NO  --> Re-examine the library path. CUB's decoupled look-back
            scan runs at near-memcpy speed for large N. Only proceed
            to custom if benchmark evidence shows the library is
            insufficient.
```

**When to skip the library**: the library path is insufficient when:
- The scan must be **fused** with preceding or following elementwise operations to avoid an extra global-memory round-trip (e.g., a cumulative softmax where the scan feeds directly into an elementwise divide).
- A **segmented scan** is needed and the segment boundaries do not map to CUB's DeviceSegmentedScan API (e.g., variable-length segments stored as a flag array rather than an offset array).
- The scan is part of a larger **multi-operator pipeline** (e.g., compact/stream compaction where the scan generates scatter indices) and fusing avoids a kernel-launch boundary.
- The measured library latency exceeds the theoretical bandwidth-bound limit by more than 10% for the given shape.

## Step 1 -- Choose the custom scan strategy

```
Q4. How many elements per scan instance?
    <=32 (single warp)
        --> Warp-level Hillis-Steele scan using __shfl_up_sync.
            Each warp independently computes an inclusive prefix sum
            in log2(32) = 5 shuffle steps. No shared memory needed.
            See wiki/nvidia/foundations/compute/warp-primitives/skill.md.

            For exclusive scan: shift the inclusive result right by 1
            lane and insert the identity element at lane 0.

    33..1024 (single block)
        --> Block-level scan:
            Phase 1: each warp computes an inclusive warp scan via
            __shfl_up_sync.
            Phase 2: the last lane of each warp writes its total to
            shared memory.
            Phase 3: warp 0 scans the warp totals (<=32 values) via
            __shfl_up_sync.
            Phase 4: every thread (except those in warp 0) adds the
            scanned total of the preceding warp to its own value.
            Requires shared memory (see bank-conflict avoidance).
            See cuda-samples/shfl_scan/shfl_scan.cu for the canonical
            example.

    >1024 (multi-block / grid-level)
        --> Grid-level multi-pass scan:
            Option A -- Three-kernel approach (Blelloch-style):
              Pass 1: each block scans its tile, writes block total.
              Pass 2: scan the block totals (a second kernel launch).
              Pass 3: uniform_add -- each block adds its prefix to
              all of its elements.
              Work-efficient (O(n)), but requires 3 kernel launches.
              See cuda-samples/shfl_scan/ for a 2-kernel variant
              (the third pass is uniform_add).

            Option B -- Single-pass decoupled lookback:
              Each block scans its tile locally, then propagates its
              partial aggregate to subsequent blocks via global memory
              using atomics and a status flag (X/P/A protocol).
              Single kernel launch, O(2n) data movement.
              This is the algorithm used by CUB internally.
              Reference: Merrill & Garland, "Single-pass Parallel
              Prefix Scan with Decoupled Look-back", NVR-2016-002.
              Custom implementation is complex; prefer CUB unless
              fusion requires it.
```

## Step 2 -- Optimization via ROUTING.md skills

After the basic custom kernel is working and correct, apply optimization skills from ROUTING.md in priority order:

1. **Warp primitives** (wiki/nvidia/foundations/compute/warp-primitives/) -- replace shared-memory tree scan within a warp with register-based `__shfl_up_sync` Hillis-Steele scan. This is the single most impactful optimization for scan kernels since the intra-warp scan is on the critical path of every element.

2. **Coalescing** (wiki/nvidia/foundations/memory/coalescing/) -- ensure the input load and output store phases use coalesced (stride-1) global memory access. Scan kernels read and write every element, so both load and store phases matter (unlike reduction, which only loads).

3. **Bank-conflict avoidance** (wiki/nvidia/foundations/memory/bank-conflict/) -- if the kernel uses shared memory for inter-warp communication of warp totals, ensure the access pattern does not cause bank conflicts. The standard pattern of writing to `smem[warp_id]` is conflict-free (one thread per bank) but the subsequent read by warp 0 of all entries must also be checked.

After each skill application, re-benchmark against the baseline (torch.cumsum or cub::DeviceScan) and follow the bottleneck-triage procedure in reasoning/bottleneck-triage.md.

## Step 3 -- Advanced techniques

### Vectorized loads for scan input

For large scans where each thread processes multiple elements sequentially before participating in the parallel scan tree, use `float4` / `int4` vectorized loads to increase memory throughput. Each thread serially scans its 4 (or more) elements, then the intra-warp and inter-warp parallel scan operates on those partial sums. After the parallel phase, each thread applies the prefix offset back to its individual elements.

### Decoupled lookback for single-pass

If the library cannot be used but single-pass performance is needed, implement the decoupled lookback protocol:
1. Each block atomically publishes its status: `X` (not started), `P` (partial aggregate available), `A` (full inclusive prefix available).
2. After computing its local scan, a block lookbacks through preceding blocks, accumulating their aggregates until it finds a block with status `A`.
3. The block then updates its own status to `A` and writes its inclusive prefix for subsequent blocks.

This is non-trivial to implement correctly. Memory ordering (`__threadfence()`, `cuda::atomic`) is critical. See wiki/nvidia/foundations/sync/memory-ordering/ for the correctness constraints.

## Cross-references

- **Library fallback details**: `library-fallback.md`
- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
