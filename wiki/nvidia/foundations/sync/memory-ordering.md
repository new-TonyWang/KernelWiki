---
title: Memory Ordering and Thread Scopes
status: draft
evidence_level: spec
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- scan
- producer-consumer
- synchronization
requires_sm: '>=7.0'
requires_features:
- scoped-atomics
single_kernel_useful: true
source:
- path: spec
  anchor: part-2--the-cuda-memory-model-allart
  excerpt: A memory model determines the values a load can legally return from memory - it is a contract between the user, the compiler, the hardware and the programming language.
artifacts:
  code: ''
  build: ''
  introspection: ''
  profile: ''
related_apis:
- cuda::atomic
- cuda::atomic_ref
- cuda::atomic_thread_fence
- __threadfence
- __threadfence_block
- __threadfence_system
related_skills:
- async-copy
- warp-primitives
id: skill-memory-ordering
type: skill
vendor: nvidia
tags:
- cuda-cpp
- cluster
- pipeline-stages
- cache-policy
- shared-memory-optimization
- fused-kernel
- ptx
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3471-L3515
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3524-L3610
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23069-L3190
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L28467-L28663
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L4680-L4780
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
hardware_features:
- cluster
techniques:
- pipeline-stages
- cache-policy
- shared-memory-optimization
kernel_types:
- fused-kernel
confidence: source-reported
---
## What

The CUDA C++ memory model defines the rules governing which values a load can legally return when multiple threads read and write shared memory. It is a contract between the programmer, the compiler, the hardware, and the language runtime. Understanding it is essential for writing correct multi-threaded CUDA kernels that communicate data between threads, blocks, or devices.

The model has three core concepts:

1. **Same-address ordering** -- within a single thread, loads and stores to the same address cannot overtake each other in the memory subsystem. This is the baseline guarantee.

2. **Memory orders** -- C++ memory ordering semantics (adopted by CUDA since Volta / sm_70) control how operations to *different* addresses may be reordered relative to each other.

3. **Thread scopes** -- CUDA extends the C++ model with a scope hierarchy that maps to the GPU's physical memory hierarchy. A narrower scope is cheaper because it requires coherence at a closer cache level.

### Same-address ordering (single thread)

For a single thread, stores are visible to subsequent loads of the same address. The following always holds:

```cuda
__device__ int val;
__global__ void kernel() {
    val = 42;
    assert(val == 42);   // always holds
}
```

The programming guide states: "Loads and stores to the same address cannot overtake each other in the memory subsystem."

**Exception -- the constant cache**: the constant cache sits in the SM with a direct link to L2 and is *not kept coherent* with the L1 cache. A store through a mutable pointer goes through L1/global, but a subsequent load through a `__constant__` variable may hit the constant cache and return a stale value. This violates same-address ordering and is undefined behavior. The GTC25-S72683 transcript illustrates this with:

```cuda
__constant__ int val = 1;
__global__ void kernel() {
    int* mut_val = const_cast<int*>(&val);
    asm volatile("": "+l"(mut_val));   // tricks the compiler
    *mut_val = 42;
    assert(val == 42);   // UB: may fail -- constant cache returns stale value
}
```

### Memory orders

CUDA adopts the four C++ memory orders (programming guide section 5.4.5, PTX ISA section 8):

| Memory Order | Semantics | Use Case |
|---|---|---|
| `memory_order_seq_cst` | Strongest. All loads/stores before this operation stay before; all after stay after. Establishes a single total order across all seq_cst operations. | Default when no order is specified. Expensive; avoid unless a total order is required. |
| `memory_order_acquire` | Consumer side. Any load or store *after* the acquire stays after. Prior loads/stores may move past it. | Spin-waiting on a flag to see data published by a producer. |
| `memory_order_release` | Producer side. Any load or store *before* the release stays before. Subsequent operations may move earlier. | Publishing data to memory before signaling readiness. |
| `memory_order_relaxed` | No ordering constraint between different addresses. Same-address ordering within a single thread still holds. | Simple counters where only atomicity (not ordering) is needed. |

The `acquire` and `release` orders form a pair: a producer does a release-store to a flag, and a consumer does an acquire-load of that flag. When the consumer sees the stored value, all memory operations that preceded the release in the producer are guaranteed to be visible to the consumer.

### Thread scopes

Each scope has an associated **point of coherence** in the GPU memory hierarchy. Using the narrowest scope that covers the communicating threads yields the cheapest synchronization.

| Scope | PTX | Coherence Point | Cost Implication |
|---|---|---|---|
| `cuda::thread_scope_thread` | -- | -- (single thread) | Trivial; used internally by `cuda::pipeline`. |
| `cuda::thread_scope_block` | `.cta` | L1 | Cheapest multi-thread scope. All threads in a block run on the same SM, so release/acquire only touch L1. |
| `cuda::thread_scope_cluster` | `.cluster` | L2 | Hopper+ (sm_90). Multiple SMs in a cluster. Release must flush L1 to L2; acquire must invalidate L1. |
| `cuda::thread_scope_device` | `.gpu` | L2 | All SMs on the GPU. Same cost profile as cluster scope (L1 flush/invalidate via L2). |
| `cuda::thread_scope_system` | `.sys` | L2 + connected caches | All GPUs and CPUs in the system. Release must ensure stores reach remote caches. Most expensive scope. |

The programming guide (section 3.2.3) defines these scopes and their coherence points. The GTC25-S72683 transcript provides the following cost summary:

- **Block-scoped** atomics are essentially free (L1 only, single SM).
- **Device-scoped** operations require L1 invalidate on acquire and L1-to-L2 flush on release.
- **System-scoped** operations require the full cache hierarchy flush so that stores reach other GPUs and CPUs.

### Relationship between legacy fences and modern scoped atomics

The legacy intrinsics map directly to scoped atomic thread fences at `seq_cst` order:

| Legacy Intrinsic | Modern Equivalent |
|---|---|
| `__threadfence_block()` | `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)` |
| `__threadfence()` | `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)` |
| `__threadfence_system()` | `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)` |

Similarly, legacy atomic functions map to relaxed-order device-scoped operations:

| Legacy Atomic | Modern Equivalent |
|---|---|
| `atomicAdd(addr, val)` | `cuda::atomic_ref<T, cuda::thread_scope_device>(addr).fetch_add(val, cuda::memory_order_relaxed)` |
| `atomicAdd_block(addr, val)` | `cuda::atomic_ref<T, cuda::thread_scope_block>(addr).fetch_add(val, cuda::memory_order_relaxed)` |
| `atomicAdd_system(addr, val)` | `cuda::atomic_ref<T, cuda::thread_scope_system>(addr).fetch_add(val, cuda::memory_order_relaxed)` |

The programming guide (section 5.4.5) recommends using the `cuda::` atomic types from libcu++ for safety and portability.

## Why

Correct use of memory ordering and scopes is required to avoid data races and undefined behavior when threads communicate through shared memory. Every producer-consumer pattern, every inter-block synchronization, and every device-host data exchange depends on getting these right.

Beyond correctness, choosing the narrowest sufficient scope directly affects performance. A block-scoped atomic on a flag between threads in the same block only touches L1. Using `thread_scope_device` for the same communication forces an unnecessary L1-to-L2 round trip. Using `thread_scope_system` forces a full hierarchy flush that is orders of magnitude more expensive. Conversely, using a scope that is too narrow (e.g., block scope for cross-block communication) is a data race and produces undefined behavior.

## When to use

- **Intra-block producer-consumer**: one thread (or warp) writes data into shared or global memory and another thread in the same block consumes it. Use `cuda::atomic<T, cuda::thread_scope_block>` with `memory_order_release` / `memory_order_acquire`.

- **Inter-block coordination**: a global counter or flag is shared across thread blocks (e.g., the "last block" pattern for single-pass reductions). Use `cuda::thread_scope_device`.

- **GPU-CPU communication**: pinned host memory shared between a running kernel and a host thread. Use `cuda::thread_scope_system`.

- **Simple atomic counters**: when only atomicity is needed (e.g., counting how many blocks have finished), `memory_order_relaxed` is sufficient and avoids unnecessary fence cost.

## When NOT to use

- **When `__syncthreads()` suffices**: if all threads in a block participate in a barrier, `__syncthreads()` already provides full memory ordering for the block (it strongly happens-before any subsequent operation by any participating thread). Adding scoped atomics on top is redundant.

- **For single-warp communication**: `__syncwarp(mask)` provides memory ordering among the named threads. Warp shuffles do *not* provide memory ordering (they are register-to-register transfers), so if memory ordering is needed after a shuffle result is used to index memory, an explicit fence or sync is required.

- **When the scope is wrong**: using a too-narrow scope for cross-scope communication is worse than not using atomics at all, because it compiles and runs but silently produces undefined behavior (see pitfalls.md P1).

## Classical example: device-wide single-pass reduction (last-block pattern)

This pattern from the programming guide (section 5.4.4) uses a device-scoped fence and atomic counter so that the last block to finish can safely read partial sums written by all other blocks.

```cuda
#include <cuda/atomic>

__device__ int count = 0;

__global__ void sum(const float*    array,
                    int             N,
                    volatile float* result) {
    __shared__ bool isLastBlockDone;
    float partialSum = calculatePartialSum(array, N);

    if (threadIdx.x == 0) {
        result[blockIdx.x] = partialSum;

        // Ensure the partial sum store is visible before the counter increment.
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst,
                                  cuda::thread_scope_device);

        int count_old = atomicInc(&count, gridDim.x);
        isLastBlockDone = (count_old == (gridDim.x - 1));
    }
    __syncthreads();

    if (isLastBlockDone) {
        float totalSum = calculateTotalSum(result);
        if (threadIdx.x == 0) {
            result[0] = totalSum;
            count     = 0;
        }
    }
}
```

Without the fence between the `result[blockIdx.x]` store and the `atomicInc`, the counter could increment before the partial sum is visible in L2, causing the last block to read stale data.

## Classical example: intra-block producer-consumer with acquire/release

```cuda
#include <cuda/atomic>

__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;

    if (threadIdx.x == 0) {
        data = 42;
        ready.store(true, cuda::memory_order_release);
    } else {
        while (!ready.load(cuda::memory_order_acquire)) { /* spin */ }
        int value = data;   // guaranteed to see 42
    }
}
```

The release-store on `ready` ensures that the write to `data` is ordered before the flag. The acquire-load ensures that the read of `data` is ordered after observing the flag. Block scope is sufficient because both threads are in the same block.
