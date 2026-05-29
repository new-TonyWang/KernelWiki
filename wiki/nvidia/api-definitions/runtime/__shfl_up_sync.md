---
func_name: __shfl_up_sync
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23953-L23953
  excerpt: 5.4.6.5. Warp Shuffle Functions
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23967-L23967
  excerpt: 5.4.6.5. Warp Shuffle Functions
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23975-L23975
  excerpt: 5.4.6.5. Warp Shuffle Functions
- path: source-code/cuda-samples/Samples/2_Concepts_and_Techniques/README.md
  anchor: L80-L80
  excerpt: '[shfl_scan](./shfl_scan)'
- path: source-code/cuda-samples/Samples/2_Concepts_and_Techniques/shfl_scan/README.md
  anchor: L5-L5
  excerpt: Description
- path: source-code/cuda-samples/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh
  anchor: L301-L301
  excerpt: pragma unroll
probed_by: 80-experience/api-probes/2026-04-17-runtime-shfl-up-sync.md
id: api-__shfl_up_sync
type: api-definition
vendor: nvidia
title: __Shfl_Up_Sync
---
# __shfl_up_sync

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `70-reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at [80-experience/api-probes/2026-04-17-runtime-shfl-up-sync.md](../../80-experience/api-probes/2026-04-17-runtime-shfl-up-sync.md) for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
