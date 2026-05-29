---
title: Atomic Reduction APIs
status: draft
source:
- path: spec
  anchor: Reference
apis:
- func_name: atomicAdd
  namespace: cuda-runtime
  kind: legacy-atomic
- func_name: atomicAdd_block
  namespace: cuda-runtime
  kind: legacy-atomic
- func_name: atomicAdd_system
  namespace: cuda-runtime
  kind: legacy-atomic
- func_name: atomicCAS
  namespace: cuda-runtime
  kind: legacy-atomic
- func_name: cuda::atomic
  namespace: cuda
  kind: libcu++-atomic
- func_name: cuda::atomic_ref
  namespace: cuda
  kind: libcu++-atomic
- func_name: __shfl_down_sync
  namespace: cuda-runtime
  kind: warp-primitive
- func_name: atom.{.sem}{.scope}.op.type
  namespace: ptx
  kind: ptx-atom
- func_name: red.{.sem}{.scope}.op.type
  namespace: ptx
  kind: ptx-red
id: api-atomic-reduction-ref
type: api-definition
vendor: nvidia
func_name: Atomic Reduction APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23252-L23295
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3524-L3610
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L19647-L19680
---
# Atomic Reduction API Surface

This file enumerates the APIs touched by the atomic-reduction skill. Entries marked "ptx-atom" / "ptx-red" are only cited for completeness; typical CUDA C++ code should reach for the `cuda::atomic` wrappers first. Promotion of any entry to `wiki/nvidia/api-definitions/` is gated on a full end-to-end example under this skill's probe task.

## Legacy atomic functions (`cuda_runtime.h`)

### `atomicAdd(T* addr, T val)`

- **Scope**: `cuda::thread_scope_device` by default.
- **Memory ordering**: `cuda::std::memory_order_relaxed` (legacy semantics).
- **Supported T** on sm_90a: `int`, `unsigned int`, `unsigned long long`, `float`, `double`, `__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`; vector types are RMW per element, not atomic as a whole.
- **Source**: Programming Guide L23252-L23295.

### `atomicAdd_block` / `atomicAdd_system`

- Scope-suffixed variants of `atomicAdd`. `_block` resolves at the CTA scope (L1/shared); `_system` resolves at the system scope (CPU + peer GPUs). The bare `atomicAdd` sits between them at device scope.
- **Source**: Programming Guide L23252-L23295.

### `atomicCAS(T* addr, T compare, T val)`

- Compare-and-swap primitive. The building block for all atomic reductions whose operator is not directly supported (e.g., user-defined `atomicMax` on `float`).
- **Source**: Programming Guide L23252-L23295.

## libcu++ scoped atomics (`<cuda/atomic>`)

### `cuda::atomic<T, Scope>`

- **Header**: `<cuda/atomic>`.
- **Signature**: `template <class T, cuda::thread_scope Scope = cuda::thread_scope_system> class cuda::atomic`.
- **Scope values**: `cuda::thread_scope_block`, `cuda::thread_scope_cluster`, `cuda::thread_scope_device`, `cuda::thread_scope_system`.
- **Members**: `store`, `load`, `exchange`, `compare_exchange_*`, `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor` -- each accepts a `cuda::memory_order` argument.
- **Usable in**: host and device code.
- **Source**: Programming Guide L3524-L3610.

### `cuda::atomic_ref<T, Scope>`

- Wraps a preexisting non-atomic location (e.g., a slot inside a `__shared__` array) and exposes atomic operations on it without requiring the storage itself to be declared atomic.
- **Source**: Programming Guide L3524-L3610.

## Warp primitive used in S1

### `__shfl_down_sync(unsigned mask, T val, int delta)`

- Used inside S1 to reduce across the 32 lanes of a warp without any memory traffic. See the `warp-primitives` skill for the full shuffle family.

## PTX atomic / reduction instructions

### `atom{.sem}{.scope}.op.type`

```
atom{.sem}{.scope}{.space}.op{.level::cache_hint}.type d, [a], b;
atom{.sem}{.scope}{.space}.cas.b16 d, [a], b, c;
atom{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.f16     d, [a], b;
atom{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.f16x2   d, [a], b;
atom{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.bf16    d, [a], b;
atom{.sem}{.scope}{.space}.add.noftz{.level::cache_hint}.bf16x2  d, [a], b;

.space = { .global, .shared{::cta, ::cluster} }
.sem   = { .relaxed, .acquire, .release, .acq_rel }
.scope = { .cta, .cluster, .gpu, .sys }
.op    = { .and, .or, .xor, .cas, .exch, .add, .inc, .dec, .min, .max }
.type  = { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 }
```

- Returns the old value in `d`. Use only when the old value is needed; otherwise prefer the `red` form below.
- `.noftz` variants preserve FP16/BF16 denormals; see P5 in `pitfalls.md`.
- **Source**: PTX ISA L19647-L19680.

### `red{.sem}{.scope}.op.type`

- Same semantics as `atom` but does not return the old value. Lower latency than `atom` when the old value is unused (the common case for reductions).
- **Source**: PTX ISA L19647-L19680.
