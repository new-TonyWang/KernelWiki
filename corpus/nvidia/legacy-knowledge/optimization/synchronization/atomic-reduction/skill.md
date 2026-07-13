# Atomic Reduction -- Skills

```yaml
status: draft
source:
  - "Programming Guide 2.2.5 (Atomics)"
  - "Programming Guide 3.2.4.1 (Scoped Atomics)"
  - "Programming Guide 3.2.4.1.2 (Performance Considerations)"
  - "Programming Guide 5.4.5 (Atomic Functions)"
cross_ref:
  - optimization/synchronization/thread-scopes
  - optimization/compute/warp-primitives
  - optimization/synchronization/barrier-optimization
related_apis:
  - atomicAdd
  - atomicCAS
  - "cuda::atomic"
  - "atom.op.type (PTX)"
  - "red.op.type (PTX)"
  - "cp.reduce.async.bulk"
  - "atom.add.noftz.f16/bf16"
  - "atom.add.vec.f32"
unlocks:
  - Correct inter-block communication
  - High-throughput reductions
conflicts_with:
  - None
```

---

## S1: Hierarchical Reduction (Warp -> Block -> Grid)

**When to Use:** For any reduction operation (sum, min, max) across a large number of threads. Direct atomics to a single global location cause massive contention.

**How to Apply:**
1. Warp-level: use `__shfl_down_sync` or `__reduce_add_sync` for intra-warp reduction.
2. Block-level: combine warp results in shared memory, then reduce across warps.
3. Grid-level: use `atomicAdd` on the final per-block result to a global accumulator.

**Code Template:**
```cpp
__global__ void hierarchicalReduce(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? input[tid] : 0.0f;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Block-level: lane 0 of each warp writes to shared memory
    __shared__ float warpSums[32];  // max 32 warps per block
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    // First warp reduces warp sums
    if (warpId == 0) {
        val = (lane < blockDim.x / 32) ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0) {
            atomicAdd(output, val);  // Grid-level: one atomic per block
        }
    }
}
```

> Source: PG 2.2.5 -- "Atomic functions should be used sparingly as they enforce thread synchronization that can impact performance."

---

## S2: Use Scoped Atomics with Narrowest Scope

**When to Use:** Always -- narrower scopes are faster because they resolve at a closer point in the memory hierarchy.

**How to Apply:**
1. Use `cuda::thread_scope_block` for intra-block atomics (resolves at L1).
2. Use `cuda::thread_scope_device` for inter-block, same-GPU atomics (resolves at L2).
3. Use `cuda::thread_scope_system` only for GPU-CPU or GPU-GPU communication.

**Code Template:**
```cpp
#include <cuda/atomic>

// Block-scoped: fast, resolves at L1
__shared__ cuda::atomic<int, cuda::thread_scope_block> blockCounter;

// Device-scoped: resolves at L2
__device__ cuda::atomic<int, cuda::thread_scope_device> deviceCounter;

// System-scoped: slowest, visible to CPU and other GPUs
__managed__ cuda::atomic<int, cuda::thread_scope_system> systemCounter;
```

> Source: PG 3.2.4.1.2 -- "Use the narrowest scope possible: block-scoped atomics are much faster than system-scoped atomics."

---

## S3: Use Relaxed Memory Ordering for Simple Counters

**When to Use:** When you only need atomicity (indivisible read-modify-write) and do not need ordering guarantees relative to other memory locations.

**How to Apply:**
1. Use `cuda::memory_order_relaxed` for counters, histograms, and statistics.
2. Use `acquire`/`release` only for producer-consumer patterns where ordering matters.

**Code Template:**
```cpp
// Relaxed: just need atomicity, no ordering
int old = counter.fetch_add(1, cuda::memory_order_relaxed);

// Acquire-release: need ordering for flag-based signaling
flag.store(true, cuda::memory_order_release);   // producer
while (!flag.load(cuda::memory_order_acquire));  // consumer
```

> Source: PG 3.2.4.1.2 -- "Prefer weaker orderings: use stronger orderings only when necessary for correctness."

---

## S4: Use Shared Memory Atomics Instead of Global

**When to Use:** When multiple threads in the same block need to atomically update the same location. Shared memory atomics are much faster than global memory atomics.

**How to Apply:**
1. Perform the block-local reduction using shared memory atomics.
2. Write the final block result to global memory with a single global atomic.

**Code Template:**
```cpp
__shared__ int sharedHistogram[256];

// Initialize
if (threadIdx.x < 256) sharedHistogram[threadIdx.x] = 0;
__syncthreads();

// Block-local histogram in shared memory (fast)
atomicAdd(&sharedHistogram[myBin], 1);
__syncthreads();

// Single atomic write per bin to global memory
if (threadIdx.x < 256) {
    atomicAdd(&globalHistogram[threadIdx.x], sharedHistogram[threadIdx.x]);
}
```

> Source: PG 3.2.4.1.2 -- "Consider memory location: shared memory atomics are faster than global memory atomics."

---

## S5: Use Vectorized Atomics for FP32 (CC 9.0+)

**When to Use:** On Hopper+ when performing atomic additions on multiple consecutive float values. `atom.add.vec.f32` processes 2 or 4 floats in one atomic operation.

**How to Apply:**
1. Ensure data is aligned to 8 bytes (.v2) or 16 bytes (.v4).
2. Use the vectorized atomic PTX instruction or intrinsic.

**Code Template:**
```cpp
// Vectorized atomic add (2 floats at once) via inline PTX
float2 vals = make_float2(a, b);
asm volatile(
    "atom.add.v2.f32 {%0, %1}, [%2], {%3, %4};"
    : "=f"(vals.x), "=f"(vals.y)
    : "l"(ptr), "f"(a), "f"(b)
);
```

> Source: PTX ISA -- atom.add.vec.f32 (.v2/.v4) for vectorized atomic FP32 addition.

---

## S6: Use Async Bulk Reduction (TMA)

**When to Use:** On Hopper+ for large-scale reductions that can be expressed as bulk copy-with-reduce operations (e.g., gradient accumulation).

**How to Apply:**
1. Use `cp.reduce.async.bulk` to copy data with a simultaneous reduction operation.
2. Track completion with mbarrier.

**Code Template:**
```cpp
// Async bulk reduce: atomically add src to dst during copy
// cp.reduce.async.bulk.shared::cluster.shared::cta.bulk_group.add.f32
// This is typically used via CUTLASS or libcu++ abstractions
```

> Source: PTX ISA -- cp.reduce.async.bulk for fused copy + reduction.

---

## Cascading Opportunities

- Hierarchical reduction feeds `warp-primitives` shuffle reductions into `barrier-optimization` for multi-stage patterns.
- Scoped atomics interact with `thread-scopes` -- use the narrowest scope that is correct.
- FP16/BF16 atomics enable efficient `half-precision-math` gradient accumulation.

## Conflicts

- None inherent. Optimization direction is always to reduce atomic contention.

## Principles

1. **Minimize contention:** One atomic per block is much better than one per thread (PG 2.2.5).
2. **Narrowest scope, weakest ordering:** Block scope + relaxed ordering is fastest when sufficient (PG 3.2.4.1.2).
3. **Shared memory before global:** Resolve as much contention as possible in shared memory (PG 3.2.4.1.2).

## Open Questions

- Q1: What is the throughput of `atom.add.vec.f32.v4` vs. 4 individual `atomicAdd` on Hopper?
- Q2: When should `red` (no-return-value reduction) be preferred over `atom` (returns old value)?
