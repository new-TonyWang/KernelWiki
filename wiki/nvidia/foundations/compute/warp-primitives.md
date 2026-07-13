---
title: Warp Primitives
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- scan
- broadcast
requires_sm: '>=3.0'
requires_features:
- warp-shuffle
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: spec
  anchor: Reference
artifacts:
  code: artifacts/experience/hw-probes/shfl-sync-bfly/warp_reduce_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o warp_reduce_probe warp_reduce_probe.cu
  introspection: ''
  profile: ''
related_apis:
- __shfl_sync
- __shfl_xor_sync
- __shfl_up_sync
- __shfl_down_sync
- __ballot_sync
- __all_sync
- __any_sync
related_skills:
- coalescing
- vectorized-access
- bank-conflict
id: skill-warp-primitives
type: skill
vendor: nvidia
tags:
- cuda-cpp
- pipeline-stages
- vectorized-loads
- shared-memory-optimization
- fused-kernel
- ptx
applies_to:
- general
source_refs:
- source_id: source-code/cuda-samples
  path: Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
  anchor: L75-L91
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23945-L24034
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23868-L23890
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13435-L13520
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L20298-L20361
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1289-L1297
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
techniques:
- pipeline-stages
- vectorized-loads
- shared-memory-optimization
kernel_types:
- fused-kernel
confidence: experimental
artifact_dir: artifacts/experience/hw-probes/shfl-sync-bfly
---
## What

Warp primitives are hardware-level operations that allow threads within a single warp (32 threads) to communicate directly through registers, bypassing shared memory entirely. They fall into two families:

**Warp shuffle family** (`shfl.sync`): exchanges register values between lanes.

- `__shfl_sync(mask, val, srcLane)` -- direct copy from a named lane (`.idx` mode).
- `__shfl_up_sync(mask, val, delta)` -- copy from `laneId - delta` (`.up` mode).
- `__shfl_down_sync(mask, val, delta)` -- copy from `laneId + delta` (`.down` mode).
- `__shfl_xor_sync(mask, val, laneMask)` -- copy from `laneId ^ laneMask` (`.bfly` mode).

**Warp vote family** (`vote.sync`): reduces a per-thread predicate across the warp.

- `__all_sync(mask, pred)` -- returns non-zero if predicate is true for all threads in mask.
- `__any_sync(mask, pred)` -- returns non-zero if predicate is true for any thread in mask.
- `__ballot_sync(mask, pred)` -- returns a bitmask with bit N set if thread N's predicate is true.

All warp primitives require a `mask` parameter specifying participating threads. The mask must satisfy the constraints documented in the programming guide section "Warp __sync Intrinsic Constraints" (L24203-L24226): each calling thread must have its bit set, non-calling threads must have their bit cleared, and all named threads must execute the intrinsic with the same mask.

The `width` parameter in shuffle functions allows logical sub-warp partitioning: when `width < warpSize`, each group of `width` consecutive lanes operates as an independent entity. `width` must be a power of two in `{1, 2, 4, 8, 16, 32}`.

Supported types for shuffle: `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float`, `double`, `__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`.

## Why

Warp shuffles eliminate shared-memory round-trips for intra-warp communication. Shared memory requires a write, a `__syncthreads()` (or `__syncwarp()`), and a read -- three operations plus a barrier. A warp shuffle replaces all three with a single instruction that moves data directly between register files.

The throughput of `shfl.sync` on sm_9.0 (H200) is 32 operations per clock per SM (from the best-practices guide Table 5). Since a warp contains 32 threads, this means one shuffle instruction per clock per warp at peak throughput. The dependent-chain latency is higher (see Measured Characteristics below) because each shuffle must wait for its source lane's result.

Warp vote operations are even cheaper in throughput terms: 64 ops/clock/SM on sm_9.0 (Table 5), meaning they can retire two vote instructions per clock per warp at peak.

## When to use

- **Warp-level reduction**: summing 32 values using butterfly (`__shfl_xor_sync`) or tree (`__shfl_down_sync`) pattern. A full warp reduction requires `log2(32) = 5` shuffle steps. This is the classical use case.

- **Block-level reduction bottom stage**: after reducing each warp to a single value, a final warp of "warp leaders" shuffles across their partial sums. For a block of 1024 threads (32 warps), the first stage reduces each warp with 5 shuffles, writes 32 partial sums to shared memory, then a single warp reads those 32 values and does another 5-step shuffle reduction.

- **Warp-level prefix scan**: `__shfl_up_sync` with doubling `delta` implements an inclusive scan across a warp in `log2(warpSize)` steps, as shown in the programming guide Example 2 (L24086-L24144).

- **Broadcast**: `__shfl_sync(mask, val, 0)` broadcasts lane 0's value to all lanes -- a single instruction replaces a shared-memory broadcast pattern.

- **Predicate-driven control**: `__ballot_sync` packs per-thread predicates into a bitmask for efficient population count (`__popc(__ballot_sync(...))` counts how many threads satisfy a condition), conditional processing, or compaction.

- **Early exit checks**: `__any_sync` / `__all_sync` let a warp collectively test a termination condition without shared memory.

## When NOT to use

- **Cross-warp communication**: shuffles only work within a single warp. For cross-warp data exchange, shared memory or cooperative groups are required.

- **Large data types without decomposition**: shuffles operate on 32-bit values (`b32` at PTX level). For 64-bit values (`double`, `long long`), the compiler automatically decomposes into two 32-bit shuffles; for 128-bit or larger structs, manual decomposition is needed and the benefit over shared memory diminishes.

- **Non-power-of-two subgroup sizes**: the `width` parameter must be a power of two. Attempting `width=31` produces undefined behavior.

- **Divergent control flow**: if threads in the mask have different code paths that reach the shuffle at different program points, behavior is undefined. The mask must match the set of threads that actually reach the call site.

## Classical example: block-level sum of 1024 fp32 elements

This pattern combines warp-level butterfly reduction with a shared-memory exchange to reduce 1024 values (32 warps x 32 threads) to a single sum.

```cuda
__global__ void block_reduce_sum(const float* __restrict__ in,
                                 float*       __restrict__ out,
                                 int n) {
    // Phase 1: each thread loads one element
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    // Phase 2: warp-level butterfly reduction (5 steps)
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

    // Phase 3: warp leaders write partial sums to shared memory
    __shared__ float warp_sums[32];  // 1024 / 32 = 32 warps
    int lane   = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (lane == 0)
        warp_sums[warpId] = val;
    __syncthreads();

    // Phase 4: first warp reduces the 32 partial sums
    if (warpId == 0) {
        val = (lane < 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (lane == 0)
            atomicAdd(out, val);
    }
}
```

This two-phase approach uses exactly 10 shuffle instructions and one `__syncthreads()`, compared to a pure shared-memory reduction that would require `log2(1024) = 10` store-barrier-load rounds.

## Measured Characteristics

- shfl-sync-bfly warp-reduce probe: On H200 (sm_90a, CUDA 12.9), a dependent chain of butterfly reductions (`__shfl_xor_sync` with delta 16/8/4/2/1, each followed by `fadd`) measured **29.00 cycles per shfl_xor_sync** and **144.98 cycles per full 5-step butterfly reduction**. The canonical baseline for a pure dependent `shfl.sync.bfly` chain (no interleaved fadd) is 23.83 cycles. The ~5-cycle gap is attributable to the dependent `fadd` between each shuffle step.
