---
func_name: cuda::atomic
namespace: runtime
header: <cuda/atomic>
signature: template <class T, cuda::thread_scope Scope = cuda::thread_scope_system>
  class cuda::atomic
status: documented
has_end_to_end_example: false
source:
- path: spec
  anchor: Reference
id: api-memory-ordering-ref
type: api-definition
vendor: nvidia
title: Apis
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3534-L3551
---
# Memory Ordering API Reference

This file lists the APIs touched by the memory-ordering skill. Each entry records the function or type name, namespace, and signature as found in upstream documentation.

## Scoped Atomic Types (libcu++)

### `cuda::atomic<T, Scope>`

- **Namespace**: `cuda`
- **Header**: `<cuda/atomic>`
- **Signature**: `template <class T, cuda::thread_scope Scope = cuda::thread_scope_system> class cuda::atomic`
- **Semantics**: An atomic type that provides atomic operations on `T` with C++ standard memory ordering semantics, scoped to the specified thread scope. Supports `store`, `load`, `exchange`, `compare_exchange_weak`, `compare_exchange_strong`, `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`. Each operation accepts a `cuda::memory_order` argument.
- **Usable in**: Host and device code.
- **Source**: Programming guide L3534-L3551, L23257-L23259

### `cuda::atomic_ref<T, Scope>`

- **Namespace**: `cuda`
- **Header**: `<cuda/atomic>`
- **Signature**: `template <class T, cuda::thread_scope Scope = cuda::thread_scope_system> class cuda::atomic_ref`
- **Semantics**: A non-owning reference that provides atomic operations on an existing object of type `T`. Same operations and ordering semantics as `cuda::atomic`, but operates on externally managed storage. Useful for atomically accessing elements in existing arrays or `__shared__` variables without changing their type.
- **Usable in**: Host and device code.
- **Source**: Programming guide L1660-L1667, L28598-L28610

## Thread Fences (libcu++)

### `cuda::atomic_thread_fence`

- **Namespace**: `cuda`
- **Header**: `<cuda/atomic>`
- **Signature**: `void cuda::atomic_thread_fence(cuda::memory_order order, cuda::thread_scope scope)`
- **Semantics**: Establishes an ordering of memory accesses by the calling thread. The `order` parameter specifies the strength of the ordering (relaxed, acquire, release, acq_rel, seq_cst). The `scope` parameter specifies which threads observe the ordering effect. Equivalent to the legacy `__threadfence*` intrinsics when used with `memory_order_seq_cst`.
- **Usable in**: Device code.
- **Source**: Programming guide L23078-L23131

## Legacy Fence Intrinsics

### `__threadfence_block()`

- **Namespace**: runtime (CUDA C++ intrinsic)
- **Header**: `device_functions.h` (included via `cuda_runtime.h`)
- **Signature**: `void __threadfence_block()`
- **Semantics**: All writes by the calling thread before the call are observed by all threads in the same block as occurring before all writes after the call. All reads before the call are ordered before all reads after the call. Equivalent to `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`.
- **Source**: Programming guide L23091-L23097

### `__threadfence()`

- **Namespace**: runtime
- **Header**: `device_functions.h`
- **Signature**: `void __threadfence()`
- **Semantics**: No write by the calling thread after the call is observed by any thread in the device as occurring before any write before the call. Equivalent to `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)`.
- **Source**: Programming guide L23114-L23119

### `__threadfence_system()`

- **Namespace**: runtime
- **Header**: `device_functions.h`
- **Signature**: `void __threadfence_system()`
- **Semantics**: All writes by the calling thread before the call are observed by all threads in the device, host threads, and peer device threads as occurring before all writes after the call. Equivalent to `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)`.
- **Source**: Programming guide L23136-L23141

## Memory Orders

### `cuda::memory_order_seq_cst`

- **Value**: Sequentially consistent. Strongest ordering.
- **Source**: Programming guide L23547 (`__NV_ATOMIC_SEQ_CST`)

### `cuda::memory_order_acquire`

- **Value**: Acquire ordering. Subsequent operations stay after.
- **Source**: Programming guide L23547 (`__NV_ATOMIC_ACQUIRE`)

### `cuda::memory_order_release`

- **Value**: Release ordering. Prior operations stay before.
- **Source**: Programming guide L23547 (`__NV_ATOMIC_RELEASE`)

### `cuda::memory_order_relaxed`

- **Value**: Relaxed ordering. Only atomicity guaranteed.
- **Source**: Programming guide L23547 (`__NV_ATOMIC_RELAXED`)

## Thread Scopes

### `cuda::thread_scope_thread`

- **PTX**: (none)
- **Coherence point**: N/A (single thread)
- **Source**: Programming guide L3486-L3489

### `cuda::thread_scope_block`

- **PTX**: `.cta`
- **Coherence point**: L1
- **Source**: Programming guide L3491-L3495

### `cuda::thread_scope_cluster`

- **PTX**: `.cluster`
- **Coherence point**: L2
- **Note**: Hopper+ (sm_90) only. Available at PTX level; exposed via built-in atomics as `__NV_THREAD_SCOPE_CLUSTER`.
- **Source**: Programming guide L3497-L3499, L23564

### `cuda::thread_scope_device`

- **PTX**: `.gpu`
- **Coherence point**: L2
- **Source**: Programming guide L3501-L3505

### `cuda::thread_scope_system`

- **PTX**: `.sys`
- **Coherence point**: L2 + connected caches
- **Source**: Programming guide L3506-L3510
