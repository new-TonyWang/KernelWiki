---
func_name: __all_sync
namespace: runtime
header: cuda_runtime.h
signature: int __all_sync(unsigned mask, int predicate)
since_cuda: '9.0'
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23872-L23882
  excerpt: '__all_sync(unsigned mask, predicate): Evaluates predicate for all non-exited
    threads in mask and returns non-zero if predicate evaluates to non-zero for all
    of them.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24220-L24295
  excerpt: The intrinsics achieve the best efficiency when all warp threads participate
    in the call, namely when the mask is set to 0xFFFFFFFF.
- path: source-code/cuda-samples/Samples/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh
  anchor: L58-L58
  excerpt: result[tx] = __all_sync(mask, input[tx]);
probed_by: 80-experience/api-probes/2026-04-17-runtime-all-sync.md
id: api-__all_sync
type: api-definition
vendor: nvidia
title: __All_Sync
---
# __all_sync

Warp vote intrinsic. Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if and only if `predicate` evaluates to non-zero for **every** participating thread. All participating threads receive the same return value (broadcast).

## Parameters

- `mask` — 32-bit unsigned bitmask specifying which threads participate. Must include all non-exited threads that reach this call. Best efficiency when `mask == 0xFFFFFFFF` (full warp).
- `predicate` — integer expression; non-zero → true, zero → false.

## Return

`int` — non-zero if `predicate` is non-zero for **all** non-exited threads in `mask`; zero otherwise. Every participating thread receives the same value.

## Preconditions

- All non-exited threads specified in `mask` must reach the same call site with the same `mask` value (Warp __sync Intrinsic Constraints).
- Threads not specified in `mask` must not reach this call (or must have already exited).
- The `mask` must be identical across all participating threads at the same program point.

## Error Modes

- **Deadlock / undefined behavior**: if a non-exited thread in `mask` fails to call `__all_sync` at the same program point.
- **Undefined behavior**: if `mask` includes threads that are not active (have exited or diverged before reaching the call).
- **No memory ordering**: `__all_sync` does **not** provide any memory fence or ordering guarantee. Use `__syncwarp()` or `__threadfence_block()` if memory visibility is needed.

## Typical Use Cases

- Checking whether all lanes in a warp satisfy a condition (e.g., all threads have converged, all elements are zero).
- Early-exit optimization: skip a code path when `__all_sync` indicates universal agreement.
- Warp-level consensus before proceeding to a collective operation.

## Related APIs

- `__any_sync(mask, predicate)` — returns non-zero if **any** thread's predicate is non-zero.
- `__ballot_sync(mask, predicate)` — returns a 32-bit bitmask of per-lane predicates.
- `__syncwarp(mask)` — warp barrier (no predicate evaluation).

## End-to-End Example

See the probe record at [80-experience/api-probes/2026-04-17-runtime-all-sync.md](../../80-experience/api-probes/2026-04-17-runtime-all-sync.md) for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
