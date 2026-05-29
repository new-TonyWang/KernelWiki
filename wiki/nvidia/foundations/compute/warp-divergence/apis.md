---
title: Warp Divergence APIs
status: draft
source:
- path: spec
  anchor: Reference
apis:
- func_name: __ballot_sync
  namespace: cuda-runtime
  kind: warp-vote
  signature: unsigned __ballot_sync(unsigned mask, int predicate);
  notes: Returns a 32-bit mask whose bit i is set iff lane i in mask evaluated predicate
    non-zero. Full-warp masks should be 0xFFFFFFFFu.
- func_name: __all_sync
  namespace: cuda-runtime
  kind: warp-vote
  signature: int __all_sync(unsigned mask, int predicate);
  notes: Returns non-zero iff predicate is non-zero on every lane in mask.
- func_name: __any_sync
  namespace: cuda-runtime
  kind: warp-vote
  signature: int __any_sync(unsigned mask, int predicate);
  notes: Returns non-zero iff predicate is non-zero on at least one lane in mask.
- func_name: __uni_sync
  namespace: cuda-runtime
  kind: warp-vote
  notes: Returns non-zero iff the predicate is uniform (same value) across all lanes
    in mask.
- func_name: __syncwarp
  namespace: cuda-runtime
  kind: warp-barrier
  signature: void __syncwarp(unsigned mask = 0xFFFFFFFFu);
  notes: Forces reconvergence of lanes in mask. Required before warp-synchronous code
    on CC 7.0+ (ITS).
- func_name: __activemask
  namespace: cuda-runtime
  kind: warp-introspect
  signature: unsigned __activemask();
  notes: Returns mask of currently active lanes. Does NOT force convergence; do not
    use as a synchronization primitive (pitfall P4).
- func_name: __popc
  namespace: cuda-runtime
  kind: intrinsic
  signature: int __popc(unsigned x);
  notes: 'Population count. Paired with __ballot_sync for stream compaction: __popc(mask
    & ((1u << laneId) - 1)) is the per-lane compact offset.'
- func_name: __ffs
  namespace: cuda-runtime
  kind: intrinsic
  signature: int __ffs(unsigned x);
  notes: Find first set bit (1-indexed). Paired with ballot to locate the first active
    lane.
- func_name: setp.CmpOp.type
  namespace: ptx
  kind: ptx-predicate-set
  signature: setp.{eq,ne,lt,le,gt,ge}.{s32,u32,f32,...} p, a, b;
  notes: Set predicate register p from comparison. Emitted by nvcc for any simple
    conditional.
- func_name: selp.type
  namespace: ptx
  kind: ptx-predicate-select
  signature: selp.type d, a, b, p;
  notes: 'Select-on-predicate: d = p ? a : b. Branchless conditional move.'
- func_name: '@{!}p instruction'
  namespace: ptx
  kind: ptx-predication
  notes: Any PTX instruction can be prefixed with a predicate guard. Predicated false
    instructions are dispatched but produce no side effects (BP §13.2).
- func_name: bra.uni
  namespace: ptx
  kind: ptx-branch
  notes: Uniform branch — cheap, no divergence. Emitted for conditions the compiler
    proves are warp-uniform.
- func_name: vote.sync.{all,any,uni,ballot}.b32
  namespace: ptx
  kind: ptx-warp-vote
  notes: PTX form of the warp-vote intrinsics above.
id: api-warp-divergence-ref
type: api-definition
vendor: nvidia
func_name: Warp Divergence APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1602-L1628
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3427-L3436
---
## Core APIs

| API / Instruction | Layer | Purpose |
|-------------------|-------|---------|
| `__ballot_sync(mask, predicate)` | Runtime | Warp-wide predicate bitmask |
| `__all_sync(mask, predicate)` | Runtime | All-lanes-true check |
| `__any_sync(mask, predicate)` | Runtime | Any-lane-true check |
| `__syncwarp(mask)` | Runtime | Force lane reconvergence (ITS) |
| `__popc(x)` | Intrinsic | Population count (pairs with ballot) |
| `__ffs(x)` | Intrinsic | First-set-bit (pairs with ballot) |

## PTX-level

| Instruction | Purpose |
|-------------|---------|
| `setp.{CmpOp}.{type}` | Set predicate register from comparison |
| `selp.{type}` | Branchless conditional move (d = p ? a : b) |
| `@{!}p INSTR` | Predicate guard on any PTX instruction |
| `bra.uni LABEL` | Uniform (non-divergent) branch |
| `vote.sync.{all,any,uni,ballot}` | Warp vote |

## Cross-references

- **`branch-elimination` skill** (pending migration): catalog of C++ rewrites (ternary, `fmax`, masked conditional write) that encourage predication.
- **`warp-primitives` skill**: warp shuffle + vote used together for divergence-free warp-level algorithms.
- **`coalescing` skill**: the pitfall P6 caveat — warp-aligning a branch can break stride-1 access.
- **`memory-ordering` skill**: `__syncwarp` is the warp-scope analog of `__syncthreads`; memory ordering semantics documented there.
