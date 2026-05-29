---
func_name: __pipeline_wait_prior
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L975-L975
  excerpt: 10.2.3.4. Asynchronous Copy from Global Memory to Shared Memory
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L985-L985
  excerpt: 10.2.3.4. Asynchronous Copy from Global Memory to Shared Memory
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3927-L3927
  excerpt: 3.2.4.3. Pipelines
probed_by: sources/experience/api-probes/2026-04-17-runtime-pipeline-wait-prior.md
id: api-__pipeline_wait_prior
type: api-definition
vendor: nvidia
title: __Pipeline_Wait_Prior
---
# __pipeline_wait_prior

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at sources/experience/api-probes/2026-04-17-runtime-pipeline-wait-prior.md for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
