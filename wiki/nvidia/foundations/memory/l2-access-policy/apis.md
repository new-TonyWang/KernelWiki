---
title: L2 Access Policy APIs
status: draft
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L4060-L4170
  excerpt: L2 Cache Set-Aside for Persisting Accesses — cudaDeviceSetLimit / cudaLimitPersistingL2CacheSize;
    accessPolicyWindow fields (base_ptr, num_bytes, hitRatio, hitProp, missProp);
    cudaCtxResetPersistingL2Cache; per-stream, per-graph-node, and per-launch attribute
    entry points.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1561-L1610
  excerpt: Best Practices §10.2.2 — L2 cache set-aside + access window use cases,
    MIG caveat, hitRatio tuning formula set_aside / WS.
apis:
- func_name: cudaStreamSetAttribute
  namespace: cuda-runtime
  kind: stream-attribute-setter
  signature: cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaLaunchAttributeID
    id, const cudaLaunchAttributeValue* value);
  notes: Attach an attribute (e.g. accessPolicyWindow) to a stream; applies to every
    kernel launched on that stream until overridden. id = cudaLaunchAttributeAccessPolicyWindow
    selects the L2 window. value points to a cudaStreamAttrValue whose accessPolicyWindow
    member carries the policy.
- func_name: cudaStreamGetAttribute
  namespace: cuda-runtime
  kind: stream-attribute-getter
  signature: cudaError_t cudaStreamGetAttribute(cudaStream_t stream, cudaLaunchAttributeID
    id, cudaLaunchAttributeValue* value_out);
  notes: Read back the current accessPolicyWindow attached to a stream. Useful when
    threading multiple policies through the same stream.
- func_name: cudaAccessPolicyWindow
  namespace: cuda-runtime
  kind: struct
  signature: struct cudaAccessPolicyWindow { void* base_ptr; size_t num_bytes; float
    hitRatio; cudaAccessProperty hitProp; cudaAccessProperty missProp; };
  notes: Descriptor consumed by stream/graph/launch attribute setters. num_bytes must
    be <= cudaDeviceProp::accessPolicyMaxWindowSize. hitRatio ∈ [0,1]. Set num_bytes
    = 0 to disable.
- func_name: cudaAccessProperty
  namespace: cuda-runtime
  kind: enum
  notes: 'cudaAccessPropertyNormal (default), cudaAccessPropertyStreaming (evict-first),
    cudaAccessPropertyPersisting (evict-last / sticky). Typical pairing: hitProp=Persisting
    + missProp=Streaming.'
- func_name: cudaDeviceSetLimit
  namespace: cuda-runtime
  kind: device-limit-setter
  signature: cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
  notes: Call with limit=cudaLimitPersistingL2CacheSize to reserve the set-aside portion
    of L2. Silent no-op under MIG (legacy P3). Value is clamped to cudaDeviceProp::persistingL2CacheMaxSize
    (37.5 MiB on H200).
- func_name: cudaDeviceGetLimit
  namespace: cuda-runtime
  kind: device-limit-getter
  signature: cudaError_t cudaDeviceGetLimit(size_t* value_out, cudaLimit limit);
  notes: Read the currently reserved set-aside size. Use to verify that cudaDeviceSetLimit
    actually took effect (MIG silent-failure detection).
- func_name: cudaLimitPersistingL2CacheSize
  namespace: cuda-runtime
  kind: cudaLimit-enum-value
  notes: Enum selector for cudaDeviceSetLimit / cudaDeviceGetLimit to address the
    persisting-L2 set-aside. Also accessible as cudaLimit::cudaLimitPersistingL2CacheSize.
- func_name: cudaCtxResetPersistingL2Cache
  namespace: cuda-runtime
  kind: persisting-cache-reset
  signature: cudaError_t cudaCtxResetPersistingL2Cache(void);
  notes: 'Evict all persisting-tagged lines. Call between distinct hot regions to
    avoid stale pins (legacy P2). Also: set the window num_bytes=0 first, then call
    this.'
- func_name: cudaLaunchKernelEx
  namespace: cuda-runtime
  kind: extended-launch
  signature: cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, F kernel,
    Args... args);
  notes: Per-launch attribute variant. config->attrs[] can include cudaLaunchAttributeAccessPolicyWindow
    for per-launch L2 policy, independent of stream defaults.
- func_name: cudaLaunchConfig_t
  namespace: cuda-runtime
  kind: struct
  notes: 'Launch config consumed by cudaLaunchKernelEx. Fields: gridDim, blockDim,
    dynamicSmemBytes, stream, attrs, numAttrs. attrs is an array of cudaLaunchAttribute
    whose id can be cudaLaunchAttributeAccessPolicyWindow.'
- func_name: cudaLaunchAttributeAccessPolicyWindow
  namespace: cuda-runtime
  kind: launch-attribute-id
  notes: ID constant selecting the access-policy-window attribute. Unified across
    stream (cudaStreamSetAttribute), graph node (cudaGraphKernelNodeSetAttribute),
    and launch (cudaLaunchAttribute) entry points.
- func_name: cudaStreamAttributeAccessPolicyWindow
  namespace: cuda-runtime
  kind: stream-attribute-id
  notes: Alias for cudaLaunchAttributeAccessPolicyWindow used in cudaStreamSetAttribute
    signatures. Same underlying ID.
- func_name: cudaKernelNodeAttributeAccessPolicyWindow
  namespace: cuda-runtime
  kind: graph-node-attribute-id
  notes: Alias for cudaLaunchAttributeAccessPolicyWindow used in cudaGraphKernelNodeSetAttribute
    signatures. Per-kernel-node L2 policy inside a CUDA Graph.
- func_name: cudaGraphKernelNodeSetAttribute
  namespace: cuda-runtime
  kind: graph-node-attribute-setter
  signature: cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t node, cudaKernelNodeAttrID
    id, const cudaKernelNodeAttrValue* value);
  notes: Per-graph-node L2 policy variant. Semantically equivalent to the stream variant;
    picks whichever submission model the caller uses (legacy S4).
- func_name: cudaGraphKernelNodeGetAttribute
  namespace: cuda-runtime
  kind: graph-node-attribute-getter
  signature: cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t node, cudaKernelNodeAttrID
    id, cudaKernelNodeAttrValue* value_out);
  notes: Read-back companion. Useful when composing reusable graph templates.
- func_name: cudaStreamCopyAttributes
  namespace: cuda-runtime
  kind: stream-attribute-copy
  signature: cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);
  notes: Copies all attributes (including accessPolicyWindow) from src to dst. Convenient
    when cloning a stream's policy without re-specifying the struct.
- func_name: cudaDeviceProp::l2CacheSize
  namespace: cuda-runtime
  kind: device-property
  notes: 'Total L2 in bytes. H200: 62914560 (60 MiB).'
- func_name: cudaDeviceProp::persistingL2CacheMaxSize
  namespace: cuda-runtime
  kind: device-property
  notes: 'Maximum set-aside for persisting accesses. H200: 39321600 (37.5 MiB). Hardware-fixed
    upper bound for cudaLimitPersistingL2CacheSize.'
- func_name: cudaDeviceProp::accessPolicyMaxWindowSize
  namespace: cuda-runtime
  kind: device-property
  notes: 'Maximum num_bytes accepted in cudaAccessPolicyWindow. H200: 134217728 (128
    MiB). Exceeding this value is silently clamped (legacy P5).'
- func_name: createpolicy.fractional.L2::evict_last.L2::evict_first.b64
  namespace: ptx
  kind: ptx-l2-policy-create
  signature: createpolicy.fractional.L2::evict_last.L2::evict_first.b64 cache_policy,
    fraction;
  notes: PTX-level creation of a 64-bit L2 cache policy descriptor. Mirrors the runtime
    hitProp=Persisting + missProp=Streaming + hitRatio=fraction combination, but applies
    per-instruction via .L2::cache_hint load/store modifiers instead of per-stream.
- func_name: ld.global.L2::cache_hint
  namespace: ptx
  kind: ptx-load-with-policy
  signature: ld.global.L2::cache_hint.type d, [a], cache_policy;
  notes: Issue a load using the policy descriptor from createpolicy. Per-instruction
    form; complementary to the per-stream runtime window. See the `cache-load-hints`
    skill (pending).
- func_name: st.global.L2::cache_hint
  namespace: ptx
  kind: ptx-store-with-policy
  signature: st.global.L2::cache_hint.type [a], b, cache_policy;
  notes: Store variant of the above.
- func_name: applypriority.global.L2
  namespace: ptx
  kind: ptx-l2-priority
  notes: Apply eviction priority (evict_normal / evict_first / evict_last / no_allocate)
    to a span of L2 lines. Orthogonal to the runtime window — a single-shot hint,
    not a persistent tag.
- func_name: discard.global.L2
  namespace: ptx
  kind: ptx-l2-discard
  notes: Drop L2 lines (no write-back). Dangerous on dirty lines; intended for scratch
    regions the kernel has finished with.
- func_name: cp.async.bulk.prefetch
  namespace: ptx
  kind: ptx-l2-prefetch
  notes: Bulk async prefetch into L2. Pairs with the persisting window when you want
    to prime the cache before the hot phase.
- func_name: cp.async.bulk.prefetch.tensor
  namespace: ptx
  kind: ptx-l2-prefetch-tensor
  notes: TMA variant of the above; prefetches tensor tiles into L2 ahead of a cp.async.bulk.tensor
    load into smem.
id: api-l2-access-policy-ref
type: api-definition
vendor: nvidia
func_name: L2 Access Policy APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
---
## Core APIs

| API / Instruction | Layer | Purpose |
|-------------------|-------|---------|
| `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, ...)` | Runtime | Attach L2 policy to a stream |
| `cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, ...)` | Runtime | Attach L2 policy to a graph kernel node |
| `cudaLaunchKernelEx(config, ...)` + `cudaLaunchAttributeAccessPolicyWindow` | Runtime | Attach L2 policy per launch |
| `cudaAccessPolicyWindow` | Runtime | Struct: base_ptr / num_bytes / hitRatio / hitProp / missProp |
| `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, n)` | Runtime | Reserve the set-aside portion of L2 |
| `cudaCtxResetPersistingL2Cache()` | Runtime | Evict all persisting-tagged lines |
| `cudaDeviceProp::{l2CacheSize, persistingL2CacheMaxSize, accessPolicyMaxWindowSize}` | Runtime | Device-specific limits |

## Policy-value enum

| `cudaAccessProperty` | Meaning |
|---|---|
| `Normal` | Default L2 LRU; no tagging |
| `Streaming` | Evict-first; this line should not stay in L2 |
| `Persisting` | Evict-last; prefer to keep this line under pressure |

## PTX-level complement (per-instruction hints)

| Instruction | Purpose |
|-------------|---------|
| `createpolicy.fractional.L2::evict_last.L2::evict_first.b64` | Build a 64-bit policy descriptor |
| `ld.global.L2::cache_hint.type` | Load using the descriptor |
| `st.global.L2::cache_hint.type` | Store using the descriptor |
| `applypriority.global.L2` | Per-span eviction priority (one-shot) |
| `discard.global.L2` | Drop L2 lines (scratch release) |
| `cp.async.bulk.prefetch[.tensor]` | Prefetch into L2 ahead of consume |

## Cross-references

- **`cache-load-hints` skill** (pending): `__ldcs` / `__ldca` / `.cg` / `.cs` are the per-instruction cousins of the runtime-level window. The two compose — a stream with `hitProp=Persisting` together with `__ldcs` loads inside the kernel signals both "keep this region hot at L2" and "do not pollute L1 with this access".
- **`async-copy` skill**: `cp.async.bulk.prefetch` + persisting window is the canonical "prime L2, then stream through L1/smem" recipe.
- **`coalescing` skill**: neither the window nor the PTX hints change memory-access *coalescing*. Fix coalescing first; L2 policy is downstream.
- **`wiki/nvidia/hardware/thread-block-cluster/`** (pending bucket F): cluster-scope kernels share an L2 across blocks; the window semantics generalize but are not re-measured.

## Related Probes

- [sources/experience/hw-probes/l2-residency/2026-04-23-l2-residency.md](../../../sources/experience/hw-probes/l2-residency/2026-04-23-l2-residency.md) — 3 WS × 3 policy sweep on H200. Load-bearing result: `hitRatio = set_aside / WS` at WS = 80 MiB gives +17.7% effective BW.
