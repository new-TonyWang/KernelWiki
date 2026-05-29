---
title: Shared Memory Cache APIs
status: draft
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1130-L1170
  excerpt: Shared memory is allocated via __shared__ (static) or extern __shared__
    (dynamic, size at launch).
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4094-L4130
  excerpt: cudaFuncSetAttribute with cudaFuncAttributePreferredSharedMemoryCarveout
    / cudaFuncAttributeMaxDynamicSharedMemorySize configures the L1/shared split and
    >48 KB opt-in.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L728-L732
  excerpt: Shared memory staging reduces redundant global reads and converts non-coalesced
    access to coalesced.
apis:
- func_name: __shared__
  namespace: cuda-language
  kind: storage-class
  signature: __shared__ T arr[N];
  notes: Static (compile-time-known size) shared-memory declaration. Lives for block
    lifetime.
- func_name: extern __shared__
  namespace: cuda-language
  kind: storage-class
  signature: extern __shared__ T smem[];
  notes: Dynamic shared-memory declaration; size passed as 3rd launch parameter.
- func_name: __syncthreads
  namespace: cuda-runtime
  kind: barrier
  signature: void __syncthreads();
  notes: Block-level barrier; required after smem writes and before cross-warp smem
    reads.
- func_name: __syncwarp
  namespace: cuda-runtime
  kind: barrier
  signature: void __syncwarp(unsigned mask = 0xFFFFFFFF);
  notes: Warp-level barrier; sufficient for intra-warp smem sharing.
- func_name: cudaFuncSetAttribute
  namespace: cuda-runtime
  kind: host-config
  signature: cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute
    attr, int value);
  notes: Sets per-kernel attributes including MaxDynamicSharedMemorySize and PreferredSharedMemoryCarveout.
- func_name: cudaFuncAttributeMaxDynamicSharedMemorySize
  namespace: cuda-runtime
  kind: attr-enum
  notes: Required opt-in for dynamic smem >48 KB per block.
- func_name: cudaFuncAttributePreferredSharedMemoryCarveout
  namespace: cuda-runtime
  kind: attr-enum
  notes: Integer percentage (0..100) or cudaSharedmemCarveoutMaxShared / MaxL1 / Default.
- func_name: cudaFuncSetCacheConfig
  namespace: cuda-runtime
  kind: host-config
  signature: cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig);
  notes: Legacy API; prefer cudaFuncSetAttribute because it does not force serialization
    between differently-configured launches (PG §3.2.6 note).
- func_name: cudaOccupancyAvailableDynamicSMemPerBlock
  namespace: cuda-runtime
  kind: host-occupancy
  signature: cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynSmemSize,
    const void* func, int numBlocks, int blockSize);
  notes: Query max dynamic smem allowed if launching numBlocks per SM at blockSize.
- func_name: cudaDeviceGetAttribute
  namespace: cuda-runtime
  kind: host-query
  signature: cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int
    device);
  notes: Query cudaDevAttrMaxSharedMemoryPerBlockOptin for the architecture's hard
    ceiling.
- func_name: ld.shared.{vec}.{type}
  namespace: ptx
  kind: ptx-load
  notes: Shared-memory load. Generated automatically from C++ smem reads; inspect
    via -Xptxas=-v or SASS.
- func_name: st.shared.{vec}.{type}
  namespace: ptx
  kind: ptx-store
  notes: Shared-memory store.
- func_name: cp.async.ca.shared.global
  namespace: ptx
  kind: ptx-async-copy
  notes: Async global->shared copy (sm_80+). See the async-copy skill for orchestration.
id: api-shared-memory-cache-ref
type: api-definition
vendor: nvidia
func_name: Shared Memory Cache APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
---
## Core APIs

| API / Instruction | Layer | Purpose |
|-------------------|-------|---------|
| `__shared__ T arr[N]` | C++/CUDA | Static shared-memory declaration |
| `extern __shared__ T smem[]` | C++/CUDA | Dynamic shared-memory declaration |
| `__syncthreads()` | Runtime | Block-scope barrier (required before cross-warp smem reads) |
| `__syncwarp(mask)` | Runtime | Warp-scope barrier (sufficient intra-warp) |
| `cudaFuncSetAttribute(func, attr, value)` | Host | Per-kernel attribute setter |
| `cudaFuncAttributeMaxDynamicSharedMemorySize` | Enum | >48 KB dynamic-smem opt-in |
| `cudaFuncAttributePreferredSharedMemoryCarveout` | Enum | L1/smem split per kernel |
| `cudaSharedmemCarveoutMaxShared` / `MaxL1` / `Default` | Enum | Carveout convenience values |
| `cudaOccupancyAvailableDynamicSMemPerBlock` | Host | Budget solver for dynamic smem vs blocks-per-SM |
| `cudaDeviceGetAttribute(..., cudaDevAttrMaxSharedMemoryPerBlockOptin, ...)` | Host | Architecture hard ceiling for opt-in smem |

## PTX-level

| Instruction | Purpose |
|-------------|---------|
| `ld.shared[.vec][.type]` | Shared-memory load |
| `st.shared[.vec][.type]` | Shared-memory store |
| `cp.async.ca.shared.global[.vec]` | Async global->shared (sm_80+); overlaps with compute |
| `cp.async.bulk` | TMA bulk async copy (sm_90+); see `40-hardware-feature/tma/` |

## Cross-references

- **Bank conflicts**: `30-skill/memory/bank-conflict/` — covers the `[TILE][TILE+1]` padding rule and 4 B vs 8 B stride rules that make the staged smem access itself fast.
- **Coalescing**: `30-skill/memory/coalescing/` — the prerequisite for the global-to-smem load to be efficient.
- **Async copy**: `30-skill/memory/async-copy/` — covers `cp.async` / `cuda::memcpy_async` that replace a blocking `load + __syncthreads` pair.
- **Distributed shared memory (>one block)**: `40-hardware-feature/thread-block-cluster/` (pending bucket F bootstrap) — covers `cluster.map_shared_rank` and `ld.shared::cluster`.
