---
func_name: cudaOccupancyMaxActiveBlocksPerMultiprocessor
namespace: runtime
header: cuda_runtime_api.h
signature: cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
  const void* func, int blockSize, size_t dynamicSMemSize)
status: documented
has_end_to_end_example: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3862-L3890
  excerpt: Returns in *numBlocks the maximum number of active blocks per streaming
    multiprocessor for the device function.
id: api-occupancy-tuning-ref
type: api-definition
vendor: nvidia
title: Apis
---
# Occupancy Tuning API Reference

This file lists the APIs touched by the occupancy-tuning skill. Each entry records the function name, namespace, and signature as found in upstream documentation.

## Occupancy Query Functions

### `cudaOccupancyMaxActiveBlocksPerMultiprocessor`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C signature**: `__host__ __device__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize)`
- **C++ template**: `template <class T> __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize)`
- **Semantics**: Returns in `*numBlocks` the maximum number of active blocks per SM for the given kernel, block size, and dynamic shared memory size. This is the primary API for computing theoretical occupancy.
- **Occupancy derivation**: `occupancy = (numBlocks * blockSize / warpSize) / (maxThreadsPerSM / warpSize)`
- **Host and device callable**: Can be called from both host and device code (C version only; C++ template is host-only).
- **Source**: CUDA Runtime API Reference L3862-L3890

### `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C signature**: `__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags)`
- **C++ template**: `template <class T> __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize, unsigned int flags)`
- **Semantics**: Same as `cudaOccupancyMaxActiveBlocksPerMultiprocessor` but accepts a flags parameter. Supported flags:
  - `cudaOccupancyDefault`: default behavior (same as the non-flags version)
  - `cudaOccupancyDisableCachingOverride`: suppresses the default behavior where the occupancy calculator pretends caching is disabled when per-block SM resource usage would result in zero occupancy with caching enabled
- **Source**: CUDA Runtime API Reference L3894-L3929

## Launch Configuration Functions

### `cudaOccupancyMaxPotentialBlockSize`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C++ template**: `template <class T> __host__ cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0)`
- **Semantics**: Returns in `*minGridSize` and `*blockSize` a grid/block size pair that achieves the maximum potential occupancy (i.e., the maximum number of active warps with the smallest number of blocks). The `blockSizeLimit` parameter caps the block size (0 means no limit).
- **Important**: This optimizes for occupancy, not latency. The suggested block size may not be the fastest in practice — always benchmark alternatives.
- **Source**: CUDA Runtime API Reference L17092-L17121

### `cudaOccupancyMaxPotentialBlockSizeVariableSMem`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C++ template**: `template <typename UnaryFunction, class T> __host__ cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0)`
- **Semantics**: Same as `cudaOccupancyMaxPotentialBlockSize` but accepts a unary function that maps block size to dynamic shared memory size. Use this when the amount of dynamic shared memory depends on the block size (e.g., `smem = blockSize * sizeof(float)`).
- **Source**: CUDA Runtime API Reference L17141-L17181

### `cudaOccupancyMaxPotentialBlockSizeWithFlags`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C++ template**: `template <class T> __host__ cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0, unsigned int flags = 0)`
- **Semantics**: Same as `cudaOccupancyMaxPotentialBlockSize` with an additional flags parameter (same flags as `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`).
- **Source**: CUDA Runtime API Reference (see also section)

### `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C++ template**: `template <typename UnaryFunction, class T> __host__ cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int* minGridSize, int* blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0, unsigned int flags = 0)`
- **Semantics**: Combines variable shared memory and flags support.
- **Source**: CUDA Runtime API Reference L17185-L17228

## Shared Memory Query

### `cudaOccupancyAvailableDynamicSMemPerBlock`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C signature**: `__host__ cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize)`
- **C++ template**: `template <class T> __host__ cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, T* func, int numBlocks, int blockSize)`
- **Semantics**: Returns in `*dynamicSmemSize` the maximum size of dynamic shared memory that allows `numBlocks` blocks per SM. This is the inverse of the occupancy query: instead of "given SMem, how many blocks?", it answers "given N blocks, how much SMem?"
- **Use case**: Determine how much shared memory a kernel can use while maintaining a target occupancy level.
- **Source**: CUDA Runtime API Reference L3828-L3858

## Cluster Occupancy (Hopper+)

### `cudaOccupancyMaxActiveClusters`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C signature**: `__host__ cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, const void* func, const cudaLaunchConfig_t* launchConfig)`
- **Semantics**: Returns the maximum number of active clusters on the device. For applications using Thread Block Clusters, the Hopper tuning guide recommends using this API instead of the per-SM occupancy APIs.
- **Source**: CUDA Runtime API Reference L3933-L3968

### `cudaOccupancyMaxPotentialClusterSize`

- **Namespace**: runtime
- **Header**: `cuda_runtime_api.h`
- **C signature**: `__host__ cudaError_t cudaOccupancyMaxPotentialClusterSize(int* clusterSize, const void* func, const cudaLaunchConfig_t* launchConfig)`
- **Semantics**: Returns the maximum potential cluster size for a kernel. Used to determine the optimal cluster configuration for Thread Block Clusters.
- **Source**: CUDA Runtime API Reference L3972-L4009

## Related Probes

- [`cudaOccupancyMaxActiveBlocksPerMultiprocessor`](../../../sources/experience/api-probes/2026-04-17-runtime-cuda-occupancy-max-active-blocks-per-multiprocessor.md) — end-to-end example, build command, and H200 measurement. See probe record.
