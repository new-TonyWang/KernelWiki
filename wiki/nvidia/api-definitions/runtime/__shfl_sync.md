---
func_name: __shfl_sync
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
probed_by: sources/experience/api-probes/2026-04-17-runtime-shfl-sync.md
id: api-__shfl_sync
type: api-definition
vendor: nvidia
title: __Shfl_Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cccl/cccl.md
  anchor: L22825-L22825
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11966-L11966
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L12144-L12144
- source_id: source-code/cutlass
  path: docs/gemm__pipelined_8h_source.html
  anchor: L101-L101
- source_id: source-code/cutlass
  path: examples/111_hopper_ssd/collective/common.hpp
  anchor: L148-L148
- source_id: source-code/cutlass
  path: examples/111_hopper_ssd/collective/common.hpp
  anchor: L162-L162
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
confidence: inferred
tags:
- cuda-cpp
---
# __shfl_sync

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at sources/experience/api-probes/2026-04-17-runtime-shfl-sync.md for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
