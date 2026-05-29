---
func_name: __shfl_sync
namespace: runtime
header: sm_30_intrinsics.h
signature: T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize)
status: documented
has_end_to_end_example: false
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23952
  excerpt: T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize)
id: api-warp-primitives-ref
type: api-definition
vendor: nvidia
title: Apis
---
# Warp Primitives API Reference

This file lists the APIs touched by the warp-primitives skill. Each entry records the function name, namespace, and signature as found in upstream documentation.

## Warp Shuffle Family

### `__shfl_sync`

- **Namespace**: runtime (CUDA C++ intrinsic)
- **Header**: `sm_30_intrinsics.h` (included via `cuda_runtime.h`)
- **Signature**: `T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize)`
- **PTX**: `shfl.sync.idx.b32`
- **Semantics**: Direct copy from indexed lane. Returns the value of `value` held by the thread whose ID is given by `srcLane`. If `srcLane` is outside `[0, width-1]`, the result comes from `srcLane % width` within the same sub-warp segment.
- **Throughput (sm_9.0)**: 32 ops/clock/SM (best-practices guide Table 5)
- **Source**: Programming guide L23952, L23961-L23963

### `__shfl_up_sync`

- **Namespace**: runtime
- **Header**: `sm_30_intrinsics.h`
- **Signature**: `T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width=warpSize)`
- **PTX**: `shfl.sync.up.b32`
- **Semantics**: Copy from a lane with lower ID. Source lane = `laneId - delta`. The lower `delta` lanes within each sub-warp segment remain unchanged (source index does not wrap).
- **Source**: Programming guide L23953, L23967-L23969

### `__shfl_down_sync`

- **Namespace**: runtime
- **Header**: `sm_30_intrinsics.h`
- **Signature**: `T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width=warpSize)`
- **PTX**: `shfl.sync.down.b32`
- **Semantics**: Copy from a lane with higher ID. Source lane = `laneId + delta`. The upper `delta` lanes within each sub-warp segment remain unchanged.
- **Source**: Programming guide L23954, L23973-L23975

### `__shfl_xor_sync`

- **Namespace**: runtime
- **Header**: `sm_30_intrinsics.h`
- **Signature**: `T __shfl_xor_sync(unsigned mask, T value, int laneMask, int width=warpSize)`
- **PTX**: `shfl.sync.bfly.b32`
- **Semantics**: Copy from a lane computed by XOR of own lane ID with `laneMask`. Implements a butterfly addressing pattern used for tree reduction and broadcast.
- **Source**: Programming guide L23955, L23979-L23980

## Warp Vote Family

### `__all_sync`

- **Namespace**: runtime
- **Header**: `sm_30_intrinsics.h`
- **Signature**: `int __all_sync(unsigned mask, int predicate)`
- **PTX**: `vote.sync.all.pred`
- **Semantics**: Returns non-zero if `predicate` evaluates to non-zero for all non-exited threads in `mask`.
- **Throughput (sm_9.0)**: 64 ops/clock/SM (best-practices guide Table 5)
- **Source**: Programming guide L23872, L23880

### `__any_sync`

- **Namespace**: runtime
- **Header**: `sm_30_intrinsics.h`
- **Signature**: `int __any_sync(unsigned mask, int predicate)`
- **PTX**: `vote.sync.any.pred`
- **Semantics**: Returns non-zero if `predicate` evaluates to non-zero for any thread in `mask`.
- **Source**: Programming guide L23873, L23882

### `__ballot_sync`

- **Namespace**: runtime
- **Header**: `sm_30_intrinsics.h`
- **Signature**: `unsigned __ballot_sync(unsigned mask, int predicate)`
- **PTX**: `vote.sync.ballot.b32`
- **Semantics**: Returns an integer whose Nth bit is set if the Nth thread of the warp evaluates `predicate` to non-zero and is active. A thread not in `mask` contributes 0 for its bit position.
- **Source**: Programming guide L23874, L23884

## Related Probes

- [`__any_sync`](../../../sources/experience/api-probes/2026-04-17-runtime-any-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__shfl_xor_sync`](../../../sources/experience/api-probes/2026-04-16-runtime-shfl-xor-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__all_sync`](../../../sources/experience/api-probes/2026-04-17-runtime-all-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__ballot_sync`](../../../sources/experience/api-probes/2026-04-16-runtime-ballot-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__shfl_down_sync`](../../../sources/experience/api-probes/2026-04-17-runtime-shfl-down-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__shfl_sync`](../../../sources/experience/api-probes/2026-04-17-runtime-shfl-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__shfl_up_sync`](../../../sources/experience/api-probes/2026-04-17-runtime-shfl-up-sync.md) — end-to-end example, build command, and H200 measurement. See probe record.
