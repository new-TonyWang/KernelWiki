---
func_name: __ballot_sync
namespace: runtime
header: cuda_runtime.h
signature: unsigned int __ballot_sync(unsigned mask, int predicate)
since_cuda: '9.0'
status: draft
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
probed_by: sources/experience/api-probes/2026-04-16-runtime-ballot-sync.md
id: api-__ballot_sync
type: api-definition
vendor: nvidia
title: __Ballot_Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23868-L23890
---
# __ballot_sync

Warp vote intrinsic. Each thread in `mask` contributes a 1-bit predicate; the function returns a 32-bit bitmask where bit `i` is set iff thread `i`'s predicate is non-zero.

## Parameters

- `mask` — 32-bit bitmask of participating threads
- `predicate` — integer expression; non-zero → contributes 1, zero → contributes 0

## Return

32-bit unsigned: bit `i` is the predicate of thread `i` (for threads in `mask`). Bits for threads not in `mask` are zero.

## Typical use case

Population count of a warp-wide condition: `int count = __popc(__ballot_sync(0xffffffff, cond))`.

## End-to-End Example

See probe record for a complete end-to-end example: host allocation of random floats, H2D copy, warp ballot + popcount kernel (counting lanes above a threshold), D2H copy, and exact correctness verification. Tested on H200 (sm_90a, CUDA 12.9).
