---
title: Memory Ordering - Pitfalls
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
single_kernel_useful: true
source:
- path: <path-removed>
  anchor: non-coherence-the-constant-cache
  excerpt: The constant cache sits in the SM, has a direct link to L2, and is not
    kept coherent with the L1 cache.
id: pitfall-memory-ordering
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L28617-L28663
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23272-L23289
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23195-L23200
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L4692-L4695
---
## P1: Scope mismatch -- block scope for cross-block communication

**Symptom**: Data race; consumer thread reads stale or uninitialized data despite the producer having completed its store and the atomic flag showing the expected value.

**Cause**: The producer uses `cuda::thread_scope_block` for its release-store, but the consumer is in a different block. Block scope only ensures coherence among threads in the same block. Threads in other blocks are not in that scope and thus the store-load pair is not atomic from their perspective, resulting in a data race (undefined behavior).

**Example (from programming guide section 5.7.5)**:

```cuda
// Thread 0 Block 0 -- WRONG scope
x = 42;
cuda::atomic_ref<int, cuda::thread_scope_block> flag(f);
flag.store(1, memory_order_release);   // UB: block scope does not include Block 1

// Thread 0 Block 1
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
while (flag.load(memory_order_acquire) != 1);
assert(x == 42);   // UB: x may not be 42
```

**Fix**: Both the producer and consumer must use a scope that includes both threads. For cross-block communication within the same device, use `cuda::thread_scope_device`.

**Detection**: This bug is silent at runtime on many configurations. Compute-sanitizer (`--tool racecheck`) may detect some instances. Review all atomic scope annotations during code review.

## P2: Constant cache non-coherence

**Symptom**: A load from a `__constant__` variable returns a stale value even though the same address was just written by the same thread via a mutable pointer.

**Cause**: The constant cache has a direct path to L2 and is not kept coherent with the L1 cache. Stores go through the L1/global path; constant-qualified loads go through the constant cache. There is no communication between the two paths, so the constant cache may hold a stale copy.

**Example (from GTC25-S72683)**:

```cuda
__constant__ int val = 1;
__global__ void kernel() {
    int* mut_val = const_cast<int*>(&val);
    asm volatile("": "+l"(mut_val));
    *mut_val = 42;
    assert(val == 42);   // UB: constant cache may return 1
}
```

**Fix**: Do not write to `__constant__` memory from device code. `__constant__` memory should only be set from the host via `cudaMemcpyToSymbol` before kernel launch.

**Detection**: Grep for `const_cast` applied to `__constant__` pointers. The PTX ISA explicitly states (section 8.1) that the memory consistency model does not apply to `ld.global.nc` (non-coherent loads).

## P3: Relaxed order does not order different addresses

**Symptom**: Consumer thread sees the flag value but reads stale data from the payload address.

**Cause**: `memory_order_relaxed` only guarantees atomicity and same-address ordering within a single thread. It does not order operations to different addresses. If a producer writes data to address A and then does a relaxed store of a flag to address B, the consumer may see the flag but not the data.

**Example**:

```cuda
// Producer (WRONG)
data = 42;
flag.store(1, cuda::memory_order_relaxed);

// Consumer
while (flag.load(cuda::memory_order_relaxed) != 1);
int value = data;   // may NOT see 42
```

**Fix**: Use `memory_order_release` on the producer's flag store and `memory_order_acquire` on the consumer's flag load:

```cuda
// Producer
data = 42;
flag.store(1, cuda::memory_order_release);

// Consumer
while (flag.load(cuda::memory_order_acquire) != 1);
int value = data;   // guaranteed to see 42
```

## P4: Defaulting to device scope when block scope suffices

**Symptom**: Kernel runs correctly but is slower than necessary.

**Cause**: Using `cuda::thread_scope_device` (or the unsuffixed legacy `atomicAdd`) for intra-block communication forces an L1-to-L2 round trip on every release/acquire, when `cuda::thread_scope_block` would keep coherence entirely within L1.

**Example**:

```cuda
// Slow: device-scoped for intra-block counter
__shared__ cuda::atomic<int, cuda::thread_scope_device> counter;  // too wide
counter.fetch_add(1, cuda::memory_order_relaxed);

// Better: block-scoped
__shared__ cuda::atomic<int, cuda::thread_scope_block> counter;
counter.fetch_add(1, cuda::memory_order_relaxed);
```

**Fix**: Audit every scoped atomic. If both producer and consumer are guaranteed to be in the same block, use `thread_scope_block`. If in the same cluster, use `.cluster` scope (Hopper+). Only widen to `thread_scope_device` or `thread_scope_system` when required.

## P5: Legacy __threadfence vs. modern cuda::atomic -- mixing old and new

**Symptom**: Confusing code that may subtly violate ordering rules when the legacy fence scope does not match the atomic operation scope.

**Cause**: Legacy `__threadfence()` is equivalent to `cuda::atomic_thread_fence(memory_order_seq_cst, thread_scope_device)`. Pairing it with a `_block`-suffixed legacy atomic creates an inconsistency: the fence orders at device scope but the atomic only provides atomicity at block scope.

**Fix**: Prefer the `cuda::atomic<T, Scope>` or `cuda::atomic_ref<T, Scope>` types from libcu++. They combine scope and ordering in a single type, eliminating scope mismatch. The programming guide (section 5.4.4) recommends libcu++ atomics for safety and portability.

## P6: Fence does not guarantee visibility -- only ordering

**Symptom**: Data appears in the wrong order despite a fence being present, because the data itself is cached in a register or optimized away by the compiler.

**Cause**: The programming guide (section 5.4.4) states: "The memory fence only affects the *order* in which memory operations are executed; it does not guarantee visibility of these operations to other threads." Without `volatile` or atomic loads, the compiler may cache a value in a register and never re-read from memory, so the fence has nothing to order.

**Fix**: For the data variable that must be visible to other threads, either:
- Declare it `volatile` (forces every load/store to go through the memory subsystem), or
- Access it through a `cuda::atomic_ref` (which implicitly prevents register caching), or
- Use `cuda::atomic<T, Scope>` for the data itself.

The last-block reduction example in the programming guide uses `volatile float* result` for exactly this reason.

## P7: Texture and surface loads bypass the memory model

**Symptom**: Texture fetch returns stale data after a global-memory store to the same address.

**Cause**: The PTX ISA (section 8.1) states: "The memory consistency model does not apply to texture (including `ld.global.nc`) and surface accesses." Texture reads go through a separate cache hierarchy (the texture cache / L1 read-only data cache) that is not coherent with the generic store path.

**Fix**: Do not use texture fetches to read memory that has been recently written by a store in the same kernel launch. Use normal global loads instead. If a texture is populated by a prior kernel, ensure a `cudaDeviceSynchronize()` or stream dependency before the consumer kernel launches.

## P8: System scope on non-system-accessible memory

**Symptom**: Atomic operation at system scope is not actually atomic with respect to the CPU, leading to torn reads or lost updates.

**Cause**: System-scope atomicity has preconditions (programming guide section 5.7.3). The memory must be either: (a) managed memory with `concurrentManagedAccess == 1`, (b) mapped (pinned) memory with `hostNativeAtomicSupported == 1`, or (c) system-allocated memory with `pageableMemoryAccess == 1`. Using `thread_scope_system` on device-only (`cudaMalloc`) memory does not give CPU-visible atomicity because the CPU cannot access that memory at all.

**Fix**: For GPU-CPU shared atomics, allocate with `cudaMallocManaged` or `cudaHostAlloc` (with `cudaHostAllocMapped`). Verify the device property flags before relying on system-scope atomicity.
