---
func_name: cudaFuncSetAttribute
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
probed_by: sources/experience/api-probes/2026-04-17-runtime-cuda-func-set-attribute.md
id: api-cudaFuncSetAttribute
type: api-definition
vendor: nvidia
title: Cudafuncsetattribute
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3372-L3372
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3375-L3375
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L15838-L15838
- source_id: source-code/cuda-samples
  path: source-code/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/README.md
  anchor: L26-L26
- source_id: source-code/cuda-samples
  path: source-code/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu
  anchor: L774-L774
- source_id: source-code/cuda-samples
  path: source-code/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu
  anchor: L782-L782
---
# cudaFuncSetAttribute

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.

## End-to-End Example

See the probe record at sources/experience/api-probes/2026-04-17-runtime-cuda-func-set-attribute.md for a complete end-to-end example, kernel source, build command, and measurement on H200 (sm_90a, CUDA 12.9).
