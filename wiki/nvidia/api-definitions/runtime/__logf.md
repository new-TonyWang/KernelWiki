---
func_name: __logf
namespace: runtime
header: cuda_runtime.h
signature: TBD — agent will fill during probe (see api-probing.md Step 2)
since_cuda: ''
status: draft
has_end_to_end_example: false
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-math-api/cuda_cuda-math-api_index.html.md
  anchor: L2725-L2725
  excerpt: 7. Single Precision Intrinsics
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-math-api/cuda_cuda-math-api_index.html.md
  anchor: L3675-L3675
  excerpt: 7.1. Functions
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L27005-L27005
  excerpt: 5.5.9.2. Single-Precision-Only Intrinsic Functions
- path: source-code/cuda-samples/Samples/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh
  anchor: L66-L66
  excerpt: BlackScholes_kernel.cuh
- path: source-code/cuda-samples/Samples/5_Domain_Specific/BlackScholes_nvrtc/BlackScholes_kernel.cuh
  anchor: L67-L67
  excerpt: BlackScholes_kernel.cuh
- path: source-code/cuda-samples/Samples/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_kernel.cu
  anchor: L133-L133
  excerpt: define MUL(a, b) __umul24(a, b)
probed_by: ''
id: api-__logf
type: api-definition
vendor: nvidia
title: __Logf
---
# __logf

<!-- Auto-seeded stub from `probe_loop seed-stubs`. Agent must probe this API following `reasoning/api-probing.md` and fill the `signature`, body sections, and artifacts. -->

## Semantics (TBD)

Agent: replace this section with parameter list + return semantics + typical use case grounded in source corpus hits listed in frontmatter.
