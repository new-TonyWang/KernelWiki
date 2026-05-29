---
func_name: __pipeline_memcpy_async
namespace: runtime
header: cuda_pipeline.h
signature: void __pipeline_memcpy_async(void* __restrict__ dst_shared, const void*
  __restrict__ src_global, size_t size_and_align)
status: documented
has_end_to_end_example: false
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3921
  excerpt: __pipeline_memcpy_async
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L971-L973
  excerpt: __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x], &global[blockDim.x
    * i + threadIdx.x], sizeof(T));
id: api-async-copy-ref
type: api-definition
vendor: nvidia
title: Apis
---
# Asynchronous Data Copies API Reference

This file lists the APIs touched by the async-copy skill. Each entry records the function name, namespace, and signature as found in upstream documentation.

## Primitives API (`<cuda_pipeline.h>`)

### `__pipeline_memcpy_async`

- **Namespace**: runtime (CUDA C++ intrinsic)
- **Header**: `cuda_pipeline.h`
- **Signature**: `void __pipeline_memcpy_async(void* __restrict__ dst_shared, const void* __restrict__ src_global, size_t size_and_align)`
- **Semantics**: Initiates an asynchronous copy of `size_and_align` bytes from global memory (`src_global`) to shared memory (`dst_shared`). The `size_and_align` parameter must be 4, 8, or 16 bytes and the pointers must be aligned to that same value. The copy proceeds asynchronously; completion is tracked through subsequent `__pipeline_commit()` and `__pipeline_wait_prior()` calls.
- **PTX**: `cp.async.ca` (L1 ACCESS mode for 4/8 bytes) or `cp.async.cg` (L1 BYPASS mode for 16 bytes, 16-byte aligned).
- **Compute capability**: 8.0+ (Ampere)
- **Source**: Programming guide L3921, Best practices guide L971-L973

### `__pipeline_commit`

- **Namespace**: runtime (CUDA C++ intrinsic)
- **Header**: `cuda_pipeline.h`
- **Signature**: `void __pipeline_commit()`
- **Semantics**: Associates a dependency barrier with all preceding `__pipeline_memcpy_async` calls that have not yet been committed. Increments the pipeline's stage counter. Must be called from **converged code** (all threads in the warp must execute it).
- **PTX**: `LDGDEPBAR` (load dependency barrier)
- **Source**: Programming guide L3924

### `__pipeline_wait_prior`

- **Namespace**: runtime (CUDA C++ intrinsic)
- **Header**: `cuda_pipeline.h`
- **Signature**: `void __pipeline_wait_prior(unsigned N)`
- **Semantics**: Blocks the calling thread until at most `N` committed stages remain pending. `__pipeline_wait_prior(0)` waits for all pending copies to complete. `__pipeline_wait_prior(1)` waits until at most 1 stage is still in-flight (used with 2-stage prefetching).
- **PTX**: `DEPBAR.LE SB, N`
- **Source**: Programming guide L3927

## libcudacxx API (`<cuda/pipeline>`, `<cuda/barrier>`)

### `cuda::memcpy_async`

- **Namespace**: libcupp
- **Header**: `<cuda/barrier>` or `<cuda/pipeline>`
- **Signatures**:
  - `void cuda::memcpy_async(T* dst, const T* src, size_t size, cuda::pipeline<scope>& pipe)`
  - `void cuda::memcpy_async(T* dst, const T* src, cuda::aligned_size_t<Align> size, cuda::barrier<scope>& bar)`
- **Semantics**: Initiates an asynchronous copy from global to shared memory. When used with a `cuda::pipeline`, completion is tracked per stage. When used with a `cuda::barrier`, completion is signaled when the barrier arrives. The `cuda::aligned_size_t<N>` wrapper tells the compiler the alignment, enabling L1 BYPASS mode when N=16. If source/destination are 16-byte aligned and size is a multiple of 16, this function may automatically use TMA on Hopper+.
- **Source**: Programming guide L11352, L11376-L11380

### `cuda::pipeline<cuda::thread_scope_thread>`

- **Namespace**: libcupp
- **Header**: `<cuda/pipeline>`
- **Factory**: `cuda::make_pipeline()`
- **Semantics**: A thread-local pipeline object for staging multi-buffer async copies. Provides `producer_acquire()`, `producer_commit()`, `consumer_release()` methods. The friend function `cuda::pipeline_consumer_wait_prior<N>(pipe)` waits until at most N stages are pending.
- **Source**: Programming guide L10964-L11042

### `cuda::barrier<cuda::thread_scope_block>`

- **Namespace**: libcupp
- **Header**: `<cuda/barrier>`
- **Semantics**: A block-scoped barrier for synchronizing async copies across all threads in a block. Required when using TMA or when a subset of threads performs copies for the whole block. Initialized from a single thread under `__syncthreads()`.
- **Source**: Programming guide L3649

### `cuda::aligned_size_t<N>`

- **Namespace**: libcupp
- **Header**: `<cuda/barrier>` or `<cuda/pipeline>`
- **Semantics**: A size wrapper that communicates alignment to the compiler. `cuda::aligned_size_t<16>(num_bytes)` tells the compiler that both source and destination are 16-byte aligned, enabling L1 BYPASS mode for LDGSTS and TMA eligibility.
- **Source**: Programming guide L11377

## Cooperative Groups API

### `cooperative_groups::memcpy_async`

- **Namespace**: cooperative_groups
- **Header**: `<cooperative_groups/memcpy_async.h>`
- **Signature**: `void cooperative_groups::memcpy_async(Group& group, void* dst, const void* src, size_t size)`
- **Semantics**: Block-wide cooperative async copy. The group collectively copies `size` bytes from `src` (global) to `dst` (shared). Completion is signaled via `cooperative_groups::wait(group)`. Only asynchronous if source is global memory and destination is shared memory and both are at least 4-byte aligned (programming guide L8855).
- **Source**: Programming guide L8824-L8855

## Thrust Integration

### `cuda::proclaim_copyable_arguments`

- **Namespace**: libcupp / CCCL
- **Header**: included via `<thrust/transform.h>` with CCCL
- **Signature**: wraps a functor: `auto f = cuda::proclaim_copyable_arguments(lambda)`
- **Semantics**: Tells `thrust::transform` that the functor's arguments can be safely copied to shared memory, which enables TMA under the hood. Thrust will internally auto-tune to maximize bytes-in-flight based on the lambda's register usage (GTC25-S72683, slide interval_0201).
- **Source**: GTC25-S72683, slide interval_0201-L0203

## Related Probes

- [`__pipeline_memcpy_async`](../../../80-experience/api-probes/2026-04-17-runtime-pipeline-memcpy-async.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__pipeline_commit`](../../../80-experience/api-probes/2026-04-17-runtime-pipeline-commit.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__pipeline_wait_prior`](../../../80-experience/api-probes/2026-04-17-runtime-pipeline-wait-prior.md) — end-to-end example, build command, and H200 measurement. See probe record.
