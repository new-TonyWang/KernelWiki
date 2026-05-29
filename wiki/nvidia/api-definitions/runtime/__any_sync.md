---
func_name: __any_sync
namespace: runtime
header: cuda_runtime.h
signature: int __any_sync(unsigned mask, int predicate)
since_cuda: '9.0'
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23872-L23882
  excerpt: '__any_sync(unsigned mask, predicate): Evaluates predicate for all non-exited
    threads in mask and returns non-zero if predicate evaluates to non-zero for any
    of them.'
probed_by: sources/experience/api-probes/2026-04-17-runtime-any-sync.md
id: api-__any_sync
type: api-definition
vendor: nvidia
title: __Any_Sync
---
# __any_sync

Warp vote intrinsic. Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if and only if `predicate` evaluates to non-zero for **any** participating thread. All participating threads receive the same return value (broadcast).

## Parameters

- `mask` — 32-bit unsigned bitmask specifying which threads participate.
- `predicate` — integer expression; non-zero → true, zero → false.

## Return

`int` — non-zero if `predicate` is non-zero for **any** non-exited thread in `mask`; zero if all predicates are zero. Every participating thread receives the same value.

## Relation to sibling intrinsics

- `__all_sync` — returns non-zero only if **all** predicates are non-zero.
- `__ballot_sync` — returns a 32-bit bitmask of per-lane predicates.

## End-to-End Example

See the probe record at [sources/experience/api-probes/2026-04-17-runtime-any-sync.md](../../sources/experience/api-probes/2026-04-17-runtime-any-sync.md) for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
