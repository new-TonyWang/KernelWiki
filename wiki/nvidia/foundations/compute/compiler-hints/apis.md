---
func_name: __launch_bounds__
namespace: runtime
header: cuda_runtime.h
signature: __launch_bounds__(maxThreadsPerBlock [, minBlocksPerMultiprocessor [, maxBlocksPerCluster]])
status: documented
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
id: api-compiler-hints-ref
type: api-definition
vendor: nvidia
title: Apis
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22816-L22905
---
# Compiler Hints API Reference

This file lists the APIs touched by the compiler-hints skill. Each entry records the function name, namespace, and signature as found in upstream documentation.

## Launch Configuration Directives

### `__launch_bounds__`

- **Namespace**: runtime (CUDA C++ function qualifier)
- **Header**: `cuda_runtime.h` (language built-in, no explicit include needed)
- **Signature**: `__launch_bounds__(maxThreadsPerBlock [, minBlocksPerMultiprocessor [, maxBlocksPerCluster]])`
- **PTX**: `.maxntid` (from maxThreadsPerBlock), `.minnctapersm` (from minBlocksPerMultiprocessor), `.maxclusterrank` (from maxBlocksPerCluster)
- **Semantics**: Annotates a `__global__` function with launch configuration hints. The compiler derives a register ceiling L = registersPerSM / (maxThreadsPerBlock * minBlocksPerMultiprocessor) and adjusts code generation to fit within it. If only `maxThreadsPerBlock` is specified, the compiler uses it to determine register usage thresholds for occupancy transitions. If both parameters are specified, the compiler may increase register usage up to L to reduce instruction count.
- **Runtime enforcement**: A kernel will fail to launch if invoked with more threads per block than `maxThreadsPerBlock` or more blocks per cluster than `maxBlocksPerCluster`.
- **Source**: Programming guide L22816-L22905

### `__maxnreg__`

- **Namespace**: runtime (CUDA C++ function qualifier)
- **Header**: `cuda_runtime.h` (language built-in)
- **Signature**: `__maxnreg__(maxNumberRegistersPerThread)`
- **PTX**: `.maxnreg`
- **Semantics**: Directly caps the per-thread register count for a kernel. Mutually exclusive with `__launch_bounds__` on the same kernel.
- **Source**: Programming guide L22890-L22905

### `--maxrregcount=N`

- **Namespace**: nvcc compiler flag
- **Signature**: `nvcc --maxrregcount=N` or `nvcc -maxrregcount=N`
- **Semantics**: Sets a per-file maximum register count for all `__global__` functions. Ignored for kernels that have `__maxnreg__`. Does not interact with `__launch_bounds__`.
- **Source**: Best practices guide L2320

## Pointer and Memory Hints

### `__restrict__`

- **Namespace**: runtime (CUDA C++ type qualifier)
- **Header**: `cuda_runtime.h` (language built-in)
- **Signature**: applied to pointer parameters, e.g., `const float* __restrict__ ptr`
- **PTX effect**: Enables `ld.global.nc` (non-coherent / read-only cache) loads for `const __restrict__` pointers in `__global__` functions.
- **Semantics**: Promises the compiler that the pointer does not alias any other pointer in scope. Enables code reordering, common sub-expression elimination, and register caching of loaded values. All pointer parameters should be annotated for the optimizer to be effective. Increases register pressure as a tradeoff.
- **Source**: Programming guide L22292-L22365

## Runtime Configuration

### `cudaFuncSetAttribute`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **Signature**: `cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value)`
- **Semantics**: Sets a per-kernel attribute at runtime. Key attributes:
  - `cudaFuncAttributeMaxDynamicSharedMemorySize`: opt-in for dynamic shared memory exceeding 48KB. Must be called before the kernel launch. Required for kernels using large shared memory buffers.
  - `cudaFuncAttributePreferredSharedMemoryCarveout`: hint for L1/shared memory partitioning. Accepted values include `cudaSharedmemCarveoutDefault`, `cudaSharedmemCarveoutMaxL1`, `cudaSharedmemCarveoutMaxShared`, or an integer percentage. The driver rounds up to the next supported capacity.
  - `cudaFuncAttributeNonPortableClusterSizeAllowed`: enables non-portable cluster sizes.
- **Source**: Programming guide L4094-L4130

## Loop Control

### `#pragma unroll`

- **Namespace**: compiler pragma (not a function)
- **Signature**: `#pragma unroll [N]`
- **Semantics**: Controls loop unrolling for the immediately following loop.
  - No argument: fully unrolls if trip count is known at compile time.
  - N > 1: unrolls by factor N.
  - N = 0 or 1: disables unrolling.
  - Negative N: pragma is ignored (warning issued).
  The expression can be a constexpr, e.g., `#pragma unroll (MyStruct::value)`.
- **Source**: Programming guide L24765-L24810

## Diagnostic Flags

### `-Xptxas=-v`

- **Namespace**: nvcc compiler flag
- **Signature**: `nvcc -Xptxas=-v` or `nvcc --ptxas-options=-v`
- **Semantics**: Passes `-v` (verbose) to the ptxas assembler, which reports per-kernel: register count, shared memory usage (static + dynamic), constant memory usage, stack frame size, and spill store/load byte counts. Essential for diagnosing register pressure.
- **Source**: Best practices guide L2321

## Related Probes

- `cudaFuncSetAttribute` — end-to-end example, build command, and H200 measurement. See probe record.
