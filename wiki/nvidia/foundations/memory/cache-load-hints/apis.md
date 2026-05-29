---
title: Cache Load Hints APIs
status: draft
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25083-L25230
  excerpt: Read-Only Data Cache Load Function (__ldg) + Low-Level Load and Store Functions
    (__ldca/__ldcg/__ldcs/__ldlu/__ldcv and store counterparts __stcg/__stcs/__stwb/__stwt).
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L10400-L10490
  excerpt: 'PTX ld cache operators: .ca / .cg / .cs / .lu / .cv. PTX st cache operators:
    .wb / .cg / .cs / .wt.'
apis:
- func_name: __ldg
  namespace: cuda-runtime
  kind: load-intrinsic-readonly
  signature: T __ldg(const T* ptr);
  notes: 'Non-coherent read-only cache load (PTX ld.global.nc). On sm_70+ the compiler
    auto-emits this for `const __restrict__` pointers; explicit __ldg is redundant
    on H200. Measured on sm_9.0a: 0.1-0.4% difference vs default (within noise).'
  ten_api_raw: wiki/nvidia/api-definitions/runtime/__ldg.md
- func_name: __ldca
  namespace: cuda-runtime
  kind: load-intrinsic-cache-all
  signature: T __ldca(const T* ptr);
  notes: Cache-all load (PTX ld.global.ca, SASS LDG.E.STRONG.SM on H200). Distinct
    from the compiler default on `const __restrict__` pointers (which lower to ld.global.nc
    / LDG.E.CONSTANT); __ldca forces the L1 + L2 cache-all path. Use for any load
    where data is reused within the SM but the pointer is not const-qualified (otherwise
    default does the same thing via the .nc path).
  ten_api_raw: wiki/nvidia/api-definitions/runtime/__ldca.md
- func_name: __ldcg
  namespace: cuda-runtime
  kind: load-intrinsic-cache-global
  signature: T __ldcg(const T* ptr);
  notes: 'Cache-global load (PTX ld.global.cg). Caches in L2 only, bypasses L1. Measured
    on H200: 2.26x slower than default at L2-resident 16-pass regime because L1 staging
    is bypassed. Use ONLY when data is not reused within the SM.'
  ten_api_raw: wiki/nvidia/api-definitions/runtime/__ldcg.md
- func_name: __ldcs
  namespace: cuda-runtime
  kind: load-intrinsic-cache-streaming
  signature: T __ldcs(const T* ptr);
  notes: 'Cache-streaming load (PTX ld.global.cs). L1 + L2 but tagged evict-first.
    Measured on H200: identical to default in un-contended L2 (8 MiB buffer in 60
    MiB L2); the evict-first tag only matters when another workload is fighting for
    L2. Legacy P4.'
  ten_api_raw: wiki/nvidia/api-definitions/runtime/__ldcs.md
- func_name: __ldlu
  namespace: cuda-runtime
  kind: load-intrinsic-last-use
  signature: T __ldlu(const T* ptr);
  notes: Last-use load (PTX ld.global.lu). Marks the cache line for eviction after
    this load retires. Effect is on subsequent kernels' L2 state; not observable in
    single-kernel microbenches. Retained as inferred pending multi-kernel probe.
  ten_api_raw: wiki/nvidia/api-definitions/runtime/__ldlu.md
- func_name: __ldcv
  namespace: cuda-runtime
  kind: load-intrinsic-volatile
  signature: T __ldcv(const T* ptr);
  notes: 'Cache-volatile load (PTX ld.global.cv). Always re-fetches (bypasses cache
    tag check). Use ONLY for flags written by other threads/kernels (correctness).
    Measured on H200: 2.26x slower than default at L2 regime — same penalty as __ldcg
    — with no compensating benefit for non-volatile data.'
  ten_api_raw: wiki/nvidia/api-definitions/runtime/__ldcv.md
- func_name: __stwb
  namespace: cuda-runtime
  kind: store-intrinsic-write-back
  signature: void __stwb(T* ptr, T value);
  notes: Write-back store (PTX st.global.wb, the default). Line stays in L2 and may
    be re-read by same SM. Not re-measured on H200; retained from legacy Skill 4.
- func_name: __stcg
  namespace: cuda-runtime
  kind: store-intrinsic-cache-global
  signature: void __stcg(T* ptr, T value);
  notes: Cache-global store (PTX st.global.cg). L2 only, bypass L1. Symmetric with
    __ldcg. Not re-measured on H200.
- func_name: __stcs
  namespace: cuda-runtime
  kind: store-intrinsic-cache-streaming
  signature: void __stcs(T* ptr, T value);
  notes: Cache-streaming store (PTX st.global.cs). L1 + L2 evict-first. Symmetric
    with __ldcs. Not re-measured on H200.
- func_name: __stwt
  namespace: cuda-runtime
  kind: store-intrinsic-write-through
  signature: void __stwt(T* ptr, T value);
  notes: Write-through store (PTX st.global.wt). Writes visible to system memory immediately;
    useful for cross-device / host-observable writes. Not re-measured on H200.
- func_name: ld.global.nc
  namespace: ptx
  kind: ptx-load-readonly
  signature: ld.global.nc.type d, [a];
  notes: Non-coherent read-only load. Emitted by __ldg and by compiler from const
    __restrict__ parameters. On H200 sm_9.0a, the L1/TEX are unified so this is the
    same cache as .ca.
- func_name: ld.global.ca
  namespace: ptx
  kind: ptx-load-cache-all
  signature: ld.global.ca.type d, [a];
  notes: Cache-all (default). Caches in L1 + L2. Emitted by __ldca and by default
    C loads.
- func_name: ld.global.cg
  namespace: ptx
  kind: ptx-load-cache-global
  signature: ld.global.cg.type d, [a];
  notes: Cache-global. Caches in L2 only; bypasses L1. Emitted by __ldcg.
- func_name: ld.global.cs
  namespace: ptx
  kind: ptx-load-cache-streaming
  signature: ld.global.cs.type d, [a];
  notes: Cache-streaming. Caches in L1 + L2 with evict-first priority. Emitted by
    __ldcs.
- func_name: ld.global.lu
  namespace: ptx
  kind: ptx-load-last-use
  signature: ld.global.lu.type d, [a];
  notes: Last-use. Caches normally but tags the line for eviction after this load.
    Emitted by __ldlu.
- func_name: ld.global.cv
  namespace: ptx
  kind: ptx-load-cache-volatile
  signature: ld.global.cv.type d, [a];
  notes: Cache-volatile. Always re-fetches, bypassing tag check. Emitted by __ldcv.
- func_name: st.global.wb
  namespace: ptx
  kind: ptx-store-write-back
  notes: Write-back (default). L1 + L2. Emitted by default C stores and __stwb.
- func_name: st.global.cg
  namespace: ptx
  kind: ptx-store-cache-global
  notes: L2 only, bypass L1. Emitted by __stcg.
- func_name: st.global.cs
  namespace: ptx
  kind: ptx-store-cache-streaming
  notes: Cache-streaming evict-first. Emitted by __stcs.
- func_name: st.global.wt
  namespace: ptx
  kind: ptx-store-write-through
  notes: Write-through to system memory. Emitted by __stwt.
- func_name: cudaFuncSetCacheConfig
  namespace: cuda-runtime
  kind: cache-config-setter
  signature: cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig);
  notes: 'Sets the preferred L1 / shared memory balance for a kernel. Legacy pitfall
    P10: on Ampere+ the split is hardware-managed; this call has no observable effect.
    Not re-measured on H200 in this probe.'
- func_name: cudaFuncAttributePreferredSharedMemoryCarveout
  namespace: cuda-runtime
  kind: func-attribute
  notes: Newer alternative to cudaFuncSetCacheConfig; expresses desired shared-memory
    carveout as a percentage. Also may be a no-op on sm_9.0a.
id: api-cache-load-hints-ref
type: api-definition
vendor: nvidia
func_name: Cache Load Hints APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
---
## Core load intrinsics (measured on H200 sm_9.0a)

| Intrinsic   | PTX operator       | H200 measured behavior |
|-------------|--------------------|------------------------|
| *default* (plain pointer) | `ld.global.ca` | Baseline; L1 + L2. Rare in modern code — usually auto-promoted to `.nc` by `const __restrict__`. |
| *default* (`const __restrict__`) | `ld.global.nc` | SASS `LDG.E.CONSTANT`. Auto-emitted by nvcc for read-only pointers; identical opcode to `__ldg`. Audited 2026-04-23. |
| `__ldg`     | `ld.global.nc`     | SASS `LDG.E.CONSTANT`. **0.1-0.4% diff vs default** — noise. Same opcode as default when the pointer is `const __restrict__`. |
| `__ldca`    | `ld.global.ca`     | SASS `LDG.E.STRONG.SM`. **Distinct from default under `const __restrict__`** — forces the L1+L2 cache-all path instead of the read-only `.nc` path. |
| `__ldcg`    | `ld.global.cg`     | **2.26x slower** at L2-resident reuse regime (L1 bypass). |
| `__ldcs`    | `ld.global.cs`     | Identical to default in un-contended L2; evict-first tag inactive. |
| `__ldlu`    | `ld.global.lu`     | Not re-measured (cross-kernel effect). |
| `__ldcv`    | `ld.global.cv`     | **2.26x slower** at L2-resident reuse. Use only for correctness (flag polling). |

## Store intrinsics (not re-measured on H200)

| Intrinsic   | PTX operator       | Purpose |
|-------------|--------------------|---------|
| *default*   | `st.global.wb`     | Write-back; line stays in L2. |
| `__stwb`    | `st.global.wb`     | Explicit write-back. |
| `__stcg`    | `st.global.cg`     | L2 only, bypass L1. Symmetric with `__ldcg`. |
| `__stcs`    | `st.global.cs`     | Evict-first. Symmetric with `__ldcs`. |
| `__stwt`    | `st.global.wt`     | Write-through to system memory. |

## Cross-references

- **10-api-raw stubs** (already exist): `__ldg.md`, `__ldca.md`,
  `__ldcg.md`, `__ldcs.md`, `__ldcv.md`, `__ldlu.md`. This skill's
  measured characteristics feed directly into the `probed_by:` field
  of those stubs after `probe_loop crosslink`.
- **`l2-access-policy` skill** (l2-access-policy): the stream-level window picks
  which L2 lines survive under pressure; this skill picks which
  cache levels the *load instruction* visits. Complementary axes —
  can and should be used together for hot buffers.
- **`coalescing` skill**: coalescing fixes the access *pattern*;
  cache hints steer *where cached bytes live*. Fix coalescing first;
  hints are a downstream concern.
- **`vectorized-access` skill**: wider loads amortize the per-load
  hint decision over more data. The probe's DRAM regime result
  (all variants tie at 29% SOL) suggests single-float loads are
  not the bottleneck — `float4` would shift DRAM throughput up
  without changing the hint-choice verdict.

## Related Probes

- sources/experience/hw-probes/cache-hint/2026-04-23-cache-hint.md —
  6 variants × 2 regimes sweep. Load-bearing results: `__ldg` ≡
  default on H200; `__ldcg` and `__ldcv` 2.26× slower on L2-resident
  reuse; `__ldcs` is a null in un-contended L2.
