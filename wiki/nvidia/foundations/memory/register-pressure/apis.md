---
func_name: cudaFuncGetAttributes
namespace: runtime
header: cuda_runtime_api.h
signature: cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const
  void *func)
status: documented
has_end_to_end_example: false
source:
- path: spec
  anchor: Reference
id: api-register-pressure-ref
type: api-definition
vendor: nvidia
title: Apis
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L28314
---
# Register Pressure API Reference

This file lists the APIs touched by the register-pressure skill. Each entry records the function name, namespace, and signature as found in upstream documentation.

## Detection APIs

### cudaFuncGetAttributes

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **Signature**: `cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)`
- Returns `cudaFuncAttributes` containing `numRegs` (registers per thread) and `localSizeBytes` (local memory per thread; non-zero implies spilling).

### cudaOccupancyMaxActiveBlocksPerMultiprocessor

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **Signature**: `cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize)`
- Computes the maximum number of active blocks per SM for a given kernel and block size. When registers are the binding constraint, the result will be lower than the thread/smem limits would allow.

## Control APIs

### __launch_bounds__

- **Namespace**: runtime (compiler directive)
- **Signature**: `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`
- Annotates a `__global__` function with occupancy targets. The compiler derives a register ceiling L = regsPerSM / (maxTPB * minBlocks).
- Cross-reference: [compiler-hints skill](../../compute/compiler-hints.md)

### __maxnreg__

- **Namespace**: runtime (compiler directive)
- **Signature**: `__maxnreg__(maxNumberRegistersPerThread)`
- Directly caps per-thread register count. Mutually exclusive with `__launch_bounds__` on the same kernel.
- Cross-reference: [compiler-hints skill](../../compute/compiler-hints.md)

## Diagnostic Flags

### --maxrregcount=N

- **Type**: nvcc compiler flag
- Caps register count for all `__global__` functions in the compilation unit. Ignored for kernels with `__maxnreg__`.

### -Xptxas=-v

- **Type**: nvcc compiler flag
- Reports per-kernel register count, shared memory, and spill stores/loads at compile time. Primary tool for detecting register pressure.

## Related Probes

- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` — end-to-end example, build command, and H200 measurement. See probe record.
