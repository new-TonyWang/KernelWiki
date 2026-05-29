---
func_name: __pipeline_commit
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L974-L974
  excerpt: 10.2.3.4. Asynchronous Copy from Global Memory to Shared Memory
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3924-L3924
  excerpt: 3.2.4.3. Pipelines
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11126-L11126
  excerpt: 4.10.6. Tracking Asynchronous Memory Operations
probed_by: 80-experience/api-probes/2026-04-17-runtime-pipeline-commit.md
id: api-__pipeline_commit
type: api-definition
vendor: nvidia
title: __Pipeline_Commit
---
# __pipeline_commit

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `70-reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at [80-experience/api-probes/2026-04-17-runtime-pipeline-commit.md](../../80-experience/api-probes/2026-04-17-runtime-pipeline-commit.md) for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
