---
func_name: cudaOccupancyMaxActiveBlocksPerMultiprocessor
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
probed_by: sources/experience/api-probes/2026-04-17-runtime-cuda-occupancy-max-active-blocks-per-multiprocessor.md
id: api-cudaOccupancyMaxActiveBlocksPerMultiprocessor
type: api-definition
vendor: nvidia
title: Cudaoccupancymaxactiveblockspermultiprocessor
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3862-L3862
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3865-L3865
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3890-L3890
- source_id: source-code/cuda-samples
  path: source-code/cuda-samples/Samples/0_Introduction/simpleAWBarrier/README.md
  anchor: L26-L26
- source_id: source-code/cuda-samples
  path: source-code/cuda-samples/Samples/0_Introduction/simpleAWBarrier/simpleAWBarrier.cu
  anchor: L204-L204
- source_id: source-code/cuda-samples
  path: source-code/cuda-samples/Samples/0_Introduction/simpleIPC/README.md
  anchor: L26-L26
---
# cudaOccupancyMaxActiveBlocksPerMultiprocessor

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at sources/experience/api-probes/2026-04-17-runtime-cuda-occupancy-max-active-blocks-per-multiprocessor.md for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
