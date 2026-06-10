---
title: Indexing Pattern -- Decision Tree
pattern_class: cuda-core
op: indexing
covers:
- gather
- scatter
- index_select
- topk
- one_hot
- embedding_lookup
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: spec
  anchor: Reference
id: routing-indexing-INDEX
type: operator-routing
vendor: nvidia
operator: indexing
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L101875-L101920
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L112950-L113004
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L22478-L22555
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L538-L542
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- vectorized-loads
- cache-policy
- kernel-fusion
kernel_types:
- fused-kernel
confidence: inferred
tags:
- vectorized-loads
- cache-policy
- kernel-fusion
- fused-kernel
- cuda-cpp
---
# Indexing Pattern -- Decision Tree

This document guides the kernel-writing agent through an indexing task from initial problem statement to a working, optimized kernel. Indexing operators move data between positions according to an index array. The family includes: gather, scatter, index_select, topk, one_hot, and embedding lookup.

**Key characteristic**: indexing operators produce inherently **non-coalesced** global memory accesses because the index array determines which addresses each thread touches. The indices are typically irregular, causing random-access patterns that waste memory bandwidth. Understanding and mitigating non-coalesced access is the central optimization challenge.

---

## Scope

This decision tree covers custom-kernel implementation choices only. It starts after the task has been classified as requiring a dedicated kernel implementation.

## Step 1 -- Choose the custom kernel strategy

### Gather (indirect load)

Thread `i` reads `src[idx[i]]`.  The index array determines the load addresses. Each thread produces exactly one output element.

```
Q4a. Are the indices sorted or locally clustered?
     YES --> Partition the index range into tiles; each block processes
             a contiguous chunk of the output.  Adjacent threads within
             a tile may still access non-adjacent source addresses, but
             locality in idx[] increases L2 cache hit rate.
     NO  --> Use a simple 1-thread-per-element kernel with a grid-stride
             loop.  Expect low effective bandwidth due to random access.
```

```cuda
// Minimal gather kernel
__global__ void gather_kernel(const float* __restrict__ src,
                              const int*   __restrict__ idx,
                              float*       __restrict__ dst,
                              int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        dst[tid] = src[idx[tid]];
}
```

The output write (`dst[tid]`) is coalesced (stride-1). The source read (`src[idx[tid]]`) is non-coalesced when indices are random. This is the fundamental bottleneck.

### Scatter (indirect store)

Thread `i` writes `dst[idx[i]] = val[i]`.  The index array determines the store addresses.

```
Q4b. Can multiple threads write to the same destination index?
     YES (scatter with conflicts) -->
         Use atomicAdd / atomicMax / atomicMin to resolve conflicts.
         This is "scatter-reduce" or "scatter-add".
         See wiki/nvidia/foundations/sync/memory-ordering/ for correctness, and
         wiki/nvidia/foundations/sync/atomic-reduction/ for contention control:
           - small destination set (histogram/bincount) -> S4 shmem atomics
           - large destination with skewed index hot-spots -> S1 hierarchical
             fan-in after sorting/grouping the indices
           - uniform random index distribution -> naive atomic is fine
     NO  (scatter without conflicts) -->
         Simple indirect store: dst[idx[tid]] = val[tid].
         The input read is coalesced; the output write is non-coalesced.
```

```cuda
// Minimal scatter kernel (no conflicts)
__global__ void scatter_kernel(const float* __restrict__ val,
                               const int*   __restrict__ idx,
                               float*       __restrict__ dst,
                               int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        dst[idx[tid]] = val[tid];
}

// Scatter-add kernel (with conflicts)
__global__ void scatter_add_kernel(const float* __restrict__ val,
                                   const int*   __restrict__ idx,
                                   float*       __restrict__ dst,
                                   int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        atomicAdd(&dst[idx[tid]], val[tid]);
}
```

### Index select

Index select is a structured gather: select entire rows (or slices along a given dimension) from a source tensor.  When the selected dimension is the outermost (dim=0), each selected row is a contiguous block that can be copied with coalesced loads and stores.

```
Q4c. Is the selection along the outermost (contiguous) dimension?
     YES --> Each selected slice is contiguous in memory.
             Use a block-per-slice approach: each block copies one
             slice using coalesced vectorized loads.
             This is effectively a batched memcpy.  DONE.
     NO  --> The slices are non-contiguous (strided).
             Fall back to the general gather pattern with stride
             adjustments, or transpose the tensor first.
```

### Top-k (partial sort / selection)

```
Q4d. What is the relationship between k and N?
     k <= 32 (fits in a single warp) -->
         Warp-level partial sort using warp shuffle comparisons.
         Each thread holds one candidate; perform a bitonic-like
         tournament to extract the top-k.
         See wiki/nvidia/foundations/compute/warp-primitives/.

     k <= 1024 (fits in a single block) -->
         Block-level radix select: use shared-memory histogram
         to identify the k-th largest value (the "pivot"), then
         do a second pass to collect all elements >= pivot.

     k > 1024 or full sort needed -->
         Prefer a task-approved reference baseline for correctness, then
         implement a bounded partial-select or radix-select kernel if the
         workload still requires a custom path.
```

### One-hot encoding

One-hot encoding is a specialized scatter: for each input index `idx[i]` in range `[0, C)`, write a `1` at position `(i, idx[i])` in an `(N, C)` output matrix that is otherwise `0`.

```cuda
__global__ void one_hot_kernel(const int* __restrict__ indices,
                               float*     __restrict__ out,
                               int N, int C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        out[tid * C + indices[tid]] = 1.0f;
}
```

The write `out[tid * C + indices[tid]]` is non-coalesced when `indices` values vary across adjacent threads. Pre-zeroing the output with `cudaMemset` is required.

---

## Step 2 -- Optimization via ROUTING.md skills

After the basic custom kernel is working and correct, apply optimization skills from ROUTING.md in priority order:

1. **Coalescing** (wiki/nvidia/foundations/memory/coalescing/) -- this is the single most critical concern for indexing kernels. Understand that one side (load or store) is inherently non-coalesced and focus optimization on the coalesced side. Sorting indices to improve spatial locality can help the non-coalesced side benefit from L2 caching.

2. **Vectorized access** (wiki/nvidia/foundations/memory/vectorized-access/) -- when the indices select contiguous ranges (index_select on dim=0), the copy of each selected slice can use float4 loads/stores. For random gather, vectorized access is generally not applicable because each thread accesses a different random location.

3. **Warp primitives** (wiki/nvidia/foundations/compute/warp-primitives/) -- useful for warp-level topk (bitonic sort / tournament) and for warp-level coordination when doing scatter with conflict resolution.

4. **Atomic reduction** (wiki/nvidia/foundations/sync/memory-ordering/) -- required for scatter-add / scatter-max where multiple threads write to the same destination index.

After each skill application, re-benchmark against the task-provided baseline and follow the bottleneck-triage procedure in reasoning/bottleneck-triage.md.

---

## Cross-references

- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
