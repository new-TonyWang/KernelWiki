---
func_name: __ldcs
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24561-L24561
  excerpt: 5.4.8.3. Low-Level Load and Store Functions
probed_by: sources/experience/api-probes/2026-04-17-runtime-ldcs.md
id: api-__ldcs
type: api-definition
vendor: nvidia
title: __Ldcs
---
# __ldcs

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at [sources/experience/api-probes/2026-04-17-runtime-ldcs.md](../../sources/experience/api-probes/2026-04-17-runtime-ldcs.md) for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
