---
func_name: __shfl_down_sync
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23954-L23954
  excerpt: 5.4.6.5. Warp Shuffle Functions
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23973-L23973
  excerpt: 5.4.6.5. Warp Shuffle Functions
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24011-L24011
  excerpt: 5.4.6.5. Warp Shuffle Functions
- path: source-code/cuda-samples/Samples/2_Concepts_and_Techniques/README.md
  anchor: L65-L65
  excerpt: '[reduction](.)'
- path: source-code/cuda-samples/Samples/2_Concepts_and_Techniques/reduction/README.md
  anchor: L5-L5
  excerpt: Description
- path: source-code/cuda-samples/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
  anchor: L78-L78
  excerpt: include <stdio.h>
probed_by: sources/experience/api-probes/2026-04-17-runtime-shfl-down-sync.md
id: api-__shfl_down_sync
type: api-definition
vendor: nvidia
title: __Shfl_Down_Sync
---
# __shfl_down_sync

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at sources/experience/api-probes/2026-04-17-runtime-shfl-down-sync.md for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
