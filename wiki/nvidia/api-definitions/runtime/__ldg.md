---
func_name: __ldg
namespace: runtime
header: cuda_runtime.h
signature: T __ldg(const T* ptr)
since_cuda: '5.0'
status: draft
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L25102-L25130
probed_by: 80-experience/api-probes/2026-04-16-runtime-ldg.md
id: api-__ldg
type: api-definition
vendor: nvidia
title: __Ldg
---
# __ldg

Read-only data cache load intrinsic. Loads the value at `ptr` through the read-only (texture) data cache rather than the default L1 path. On architectures ≥ sm_35, this can improve bandwidth when the data is known to be immutable during the kernel execution.

## Parameters

- `ptr` — pointer to the value to load (must be in global memory)

## Return

The value at `*ptr`, loaded through the read-only cache.

## Notes

On sm_70+ with `const __restrict__` pointers, the compiler may automatically route loads through the read-only path, making explicit `__ldg` unnecessary. However, explicit `__ldg` is still useful for non-const pointers or when the compiler cannot prove immutability.

## End-to-End Example

See [probe record](../../../80-experience/api-probes/2026-04-16-runtime-ldg.md) for a complete end-to-end example: host allocation of 1M floats, H2D copy, `__ldg`-based load + add kernel, D2H copy, correctness verification, and a side-by-side comparison with a plain global load baseline. Tested on H200 (sm_90a, CUDA 12.9). Result: `__ldg` and plain loads show identical latency (ratio ~1.00), confirming the compiler auto-optimizes `const __restrict__` loads on this architecture.
