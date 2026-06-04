---
func_name: __ldcg
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
probed_by: sources/experience/api-probes/2026-04-17-runtime-ldcg.md
id: api-__ldcg
type: api-definition
vendor: nvidia
title: __Ldcg
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24559
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- cache-policy
confidence: inferred
tags:
- cache-policy
- cuda-cpp
---
# __ldcg

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at sources/experience/api-probes/2026-04-17-runtime-ldcg.md for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
