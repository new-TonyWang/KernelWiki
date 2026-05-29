---
func_name: __shfl_xor_sync
namespace: runtime
header: cuda_runtime.h
signature: T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize)
since_cuda: '9.0'
status: draft
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
probed_by: sources/experience/api-probes/2026-04-16-runtime-shfl-xor-sync.md
id: api-__shfl_xor_sync
type: api-definition
vendor: nvidia
title: __Shfl_Xor_Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23945-L24034
---
# __shfl_xor_sync

Exchanges `var` between threads in a warp using XOR of the caller's lane ID with `laneMask`. All threads named in `mask` must call the function; behavior is undefined for threads not in `mask`.

## Parameters

- `mask` — 32-bit bitmask of participating threads (must include the caller)
- `var` — value to exchange (supports int, float, and vector types via overloads)
- `laneMask` — XOR delta applied to caller's lane ID to compute source lane
- `width` — logical warp width (power of 2, ≤ 32; default `warpSize`)

## Return

Returns `var` from the source lane (caller's laneId XOR laneMask). If the source lane is inactive or outside the logical sub-warp, the result is undefined.

## Typical use case

Butterfly reduction: `val += __shfl_xor_sync(0xffffffff, val, delta)` with delta = 16, 8, 4, 2, 1 sums 32 elements in 5 steps.

## End-to-End Example

See probe record for a complete end-to-end example: host allocation, H2D copy, warp butterfly reduction kernel, D2H copy, and correctness verification against a CPU sequential sum. Tested on H200 (sm_90a, CUDA 12.9).
