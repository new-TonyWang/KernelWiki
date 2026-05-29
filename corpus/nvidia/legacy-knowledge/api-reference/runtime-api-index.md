# CUDA Runtime API Index

Source: CUDA Toolkit 13.2 Runtime API Reference
Generated from: `cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/`

Knowledge tree prefix: `optimization/` (abbreviated as `b/` below for readability)

---

## 6.1. Device Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaChooseDevice` | Select compute-device which best matches criteria | -- | low-relevance (device selection) |
| `cudaDeviceFlushGPUDirectRDMAWrites` | Blocks until remote writes are visible to the specified scope | b/synchronization-bound/memory-sync-domains | related |
| `cudaDeviceGetAttribute` | Returns information about the device (SM count, shared mem size, warp size, etc.) | b/latency-bound/occupancy-tuning | related |
| `cudaDeviceGetByPCIBusId` | Returns a handle to a compute device by PCI bus ID | -- | low-relevance (device enumeration) |
| `cudaDeviceGetCacheConfig` | Returns the preferred cache configuration (L1 vs shared memory split) | b/memory-bound/shared-memory-cache, b/memory-bound/cache-load-hints | core |
| `cudaDeviceGetDefaultMemPool` | Returns the default mempool of a device | b/memory-bound/host-device-transfer | related |
| `cudaDeviceGetHostAtomicCapabilities` | Queries atomic operations supported between device and host | b/synchronization-bound/atomic-reduction, b/synchronization-bound/thread-scopes | related |
| `cudaDeviceGetLimit` | Return resource limits (stack size, L2 cache, malloc heap, etc.) | b/memory-bound/l2-cache-control, b/memory-bound/register-pressure | core |
| `cudaDeviceGetMemPool` | Gets the current mempool for a device | b/memory-bound/host-device-transfer | related |
| `cudaDeviceGetNvSciSyncAttributes` | Return NvSciSync attributes that the device can support | -- | low-relevance (interop) |
| `cudaDeviceGetP2PAtomicCapabilities` | Queries atomic operations supported between two devices | b/synchronization-bound/atomic-reduction | related |
| `cudaDeviceGetP2PAttribute` | Queries attributes of the link between two devices | b/memory-bound/host-device-transfer, b/memory-bound/numa-binding | related |
| `cudaDeviceGetPCIBusId` | Returns a PCI Bus Id string for the device | -- | low-relevance (device enumeration) |
| `cudaDeviceGetStreamPriorityRange` | Returns least and greatest stream priorities | b/latency-bound/stream-concurrency | related |
| `cudaDeviceGetTexture1DLinearMaxWidth` | Returns max elements in a 1D linear texture | -- | low-relevance (texture management) |
| `cudaDeviceRegisterAsyncNotification` | Registers a callback to receive async notifications | -- | low-relevance (notification/debug) |
| `cudaDeviceReset` | Destroy all allocations and reset all state | -- | low-relevance (lifecycle management) |
| `cudaDeviceSetCacheConfig` | Sets the preferred cache configuration (L1 vs shared memory) | b/memory-bound/shared-memory-cache, b/memory-bound/cache-load-hints | core |
| `cudaDeviceSetLimit` | Set resource limits (L2 persistent cache size, stack size, L2 fetch granularity) | b/memory-bound/l2-cache-control, b/latency-bound/dynamic-parallelism | core |
| `cudaDeviceSetMemPool` | Sets the current memory pool of a device | b/memory-bound/host-device-transfer | related |
| `cudaDeviceSynchronize` | Wait for compute device to finish | b/synchronization-bound/barrier-optimization | related |
| `cudaDeviceUnregisterAsyncNotification` | Unregisters an async notification callback | -- | low-relevance (notification/debug) |
| `cudaGetDevice` | Returns the device on which the active host thread executes | -- | low-relevance (device query) |
| `cudaGetDeviceCount` | Returns the number of compute-capable devices | -- | low-relevance (device enumeration) |
| `cudaGetDeviceFlags` | Gets the flags for the current device | -- | low-relevance (device query) |
| `cudaGetDeviceProperties` | Returns properties for the selected device (SM count, clock, memory, etc.) | b/latency-bound/occupancy-tuning | related |
| `cudaInitDevice` | Initialize device to be used for GPU executions | b/latency-bound/context-management | related |
| `cudaIpcCloseMemHandle` | Closes memory mapped with cudaIpcOpenMemHandle | -- | low-relevance (IPC) |
| `cudaIpcGetEventHandle` | Gets an interprocess handle for an event | -- | low-relevance (IPC) |
| `cudaIpcGetMemHandle` | Gets an interprocess memory handle for a device allocation | -- | low-relevance (IPC) |
| `cudaIpcOpenEventHandle` | Opens an interprocess event handle | -- | low-relevance (IPC) |
| `cudaIpcOpenMemHandle` | Opens an interprocess memory handle | -- | low-relevance (IPC) |
| `cudaSetDevice` | Set device to be used for GPU executions | b/latency-bound/context-management | related |
| `cudaSetDeviceFlags` | Sets flags for the current device (scheduling mode) | b/latency-bound/context-management | related |
| `cudaSetValidDevices` | Set a list of devices that can be used for CUDA | -- | low-relevance (device selection) |

## 6.2. Device Management [DEPRECATED]

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaDeviceGetSharedMemConfig` | Returns the shared memory configuration for the current device | b/memory-bound/shared-memory-cache, b/memory-bound/bank-conflict-avoidance | core |
| `cudaDeviceSetSharedMemConfig` | Sets the shared memory bank size for the current device (4-byte or 8-byte) | b/memory-bound/bank-conflict-avoidance | core |

## 6.3. Error Handling

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaGetErrorName` | Returns the string name for an error code | -- | low-relevance (error handling) |
| `cudaGetErrorString` | Returns the description string for an error code | -- | low-relevance (error handling) |
| `cudaGetLastError` | Returns the last error and resets it to cudaSuccess | -- | low-relevance (error handling) |
| `cudaPeekAtLastError` | Returns the last error without resetting it | -- | low-relevance (error handling) |

## 6.4. Stream Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaCtxResetPersistingL2Cache` | Resets all persisting lines in L2 cache to normal status | b/memory-bound/l2-cache-control | core |
| `cudaStreamAddCallback` | Add a callback to a compute stream (deprecated in favor of cudaLaunchHostFunc) | b/latency-bound/stream-concurrency | related |
| `cudaStreamAttachMemAsync` | Attach memory to a stream asynchronously (for unified memory visibility) | b/memory-bound/unified-memory | core |
| `cudaStreamBeginCapture` | Begins graph capture on a stream | b/latency-bound/cuda-graphs | core |
| `cudaStreamBeginCaptureToGraph` | Begins graph capture on a stream to an existing graph | b/latency-bound/cuda-graphs | core |
| `cudaStreamCopyAttributes` | Copies attributes (including L2 access policy) from source to destination stream | b/memory-bound/l2-cache-control | related |
| `cudaStreamCreate` | Create an asynchronous stream | b/latency-bound/stream-concurrency | core |
| `cudaStreamCreateWithFlags` | Create an asynchronous stream with flags (e.g., cudaStreamNonBlocking) | b/latency-bound/stream-concurrency | core |
| `cudaStreamCreateWithPriority` | Create an asynchronous stream with specified priority | b/latency-bound/stream-concurrency | core |
| `cudaStreamDestroy` | Destroys and cleans up an asynchronous stream | b/latency-bound/stream-concurrency | related |
| `cudaStreamEndCapture` | Ends capture on a stream, returning the captured graph | b/latency-bound/cuda-graphs | core |
| `cudaStreamGetAttribute` | Queries stream attribute (including cudaAccessPolicyWindow for L2 cache) | b/memory-bound/l2-cache-control, b/latency-bound/stream-concurrency | core |
| `cudaStreamGetCaptureInfo` | Query a stream's capture state (id, graph, dependencies) | b/latency-bound/cuda-graphs | related |
| `cudaStreamGetDevice` | Query the device associated with a stream | -- | low-relevance (stream query) |
| `cudaStreamGetFlags` | Query the flags of a stream | b/latency-bound/stream-concurrency | related |
| `cudaStreamGetId` | Query the unique Id of a stream | -- | low-relevance (stream query) |
| `cudaStreamGetPriority` | Query the priority of a stream | b/latency-bound/stream-concurrency | related |
| `cudaStreamIsCapturing` | Returns a stream's capture status | b/latency-bound/cuda-graphs | related |
| `cudaStreamQuery` | Queries an asynchronous stream for completion status | b/latency-bound/stream-concurrency | related |
| `cudaStreamSetAttribute` | Sets stream attribute (including cudaAccessPolicyWindow for L2 persistence) | b/memory-bound/l2-cache-control | core |
| `cudaStreamSynchronize` | Waits for stream tasks to complete | b/synchronization-bound/barrier-optimization | related |
| `cudaStreamUpdateCaptureDependencies` | Update capture dependencies of a stream | b/latency-bound/cuda-graphs | related |
| `cudaStreamWaitEvent` | Make a compute stream wait on an event | b/latency-bound/stream-concurrency, b/synchronization-bound/barrier-optimization | core |
| `cudaThreadExchangeStreamCaptureMode` | Swaps the stream capture mode of the calling thread | b/latency-bound/cuda-graphs | related |

## 6.5. Event Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaEventCreate` | Creates an event object | b/latency-bound/stream-concurrency | related |
| `cudaEventCreateWithFlags` | Creates an event with the specified flags (timing disabled, blocking sync, interprocess) | b/latency-bound/stream-concurrency, b/latency-bound/kernel-launch-overhead | core |
| `cudaEventDestroy` | Destroys an event object | b/latency-bound/stream-concurrency | related |
| `cudaEventElapsedTime` | Computes elapsed time between events | b/latency-bound/kernel-launch-overhead | related |
| `cudaEventQuery` | Queries an event's status | b/latency-bound/stream-concurrency | related |
| `cudaEventRecord` | Records an event | b/latency-bound/stream-concurrency | related |
| `cudaEventRecordWithFlags` | Records an event with flags (default or external) | b/latency-bound/stream-concurrency, b/synchronization-bound/memory-sync-domains | related |
| `cudaEventSynchronize` | Waits for an event to complete | b/synchronization-bound/barrier-optimization | related |

## 6.6. External Resource Interoperability

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaDestroyExternalMemory` | Destroys an external memory object | -- | low-relevance (external interop) |
| `cudaDestroyExternalSemaphore` | Destroys an external semaphore | -- | low-relevance (external interop) |
| `cudaExternalMemoryGetMappedBuffer` | Maps a buffer onto an imported external memory object | -- | low-relevance (external interop) |
| `cudaExternalMemoryGetMappedMipmappedArray` | Maps a mipmap onto an imported external memory object | -- | low-relevance (external interop) |
| `cudaImportExternalMemory` | Imports an external memory object | -- | low-relevance (external interop) |
| `cudaImportExternalSemaphore` | Imports an external semaphore | -- | low-relevance (external interop) |
| `cudaSignalExternalSemaphoresAsync` | Signals a set of external semaphore objects | -- | low-relevance (external interop) |
| `cudaWaitExternalSemaphoresAsync` | Waits on a set of external semaphore objects | -- | low-relevance (external interop) |

## 6.7. Execution Control

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaFuncGetAttributes` | Find out attributes for a given function (maxThreadsPerBlock, numRegs, sharedSizeBytes, etc.) | b/latency-bound/occupancy-tuning, b/memory-bound/register-pressure, b/memory-bound/shared-memory-cache | core |
| `cudaFuncGetName` | Returns the function name for a device entry function pointer | -- | low-relevance (debug/introspection) |
| `cudaFuncGetParamCount` | Returns the number of parameters used by the function | -- | low-relevance (introspection) |
| `cudaFuncGetParamInfo` | Returns the offset and size of a kernel parameter in device-side parameter layout | b/latency-bound/cuda-graphs | related |
| `cudaFuncSetAttribute` | Set attributes for a function (max dynamic shared mem, cluster dims, shared mem carveout) | b/memory-bound/shared-memory-cache, b/latency-bound/occupancy-tuning, b/compute-bound/compiler-hints | core |
| `cudaFuncSetCacheConfig` | Sets the preferred cache configuration (L1 vs shared memory) for a device function | b/memory-bound/shared-memory-cache, b/memory-bound/cache-load-hints | core |
| `cudaGetParameterBuffer` | (__device__) Obtains a parameter buffer for dynamic parallelism kernel launch | b/latency-bound/dynamic-parallelism | core |
| `cudaGridDependencySynchronize` | (__device__) Programmatic grid dependency synchronization; blocks until direct grid dependencies complete | b/latency-bound/programmatic-dependent-launch, b/synchronization-bound/barrier-optimization | core |
| `cudaLaunchCooperativeKernel` | Launches a kernel where thread blocks can cooperate and synchronize across grid | b/synchronization-bound/cooperative-groups, b/latency-bound/occupancy-tuning | core |
| `cudaLaunchDevice` | (__device__) Launches a kernel from device code (dynamic parallelism) | b/latency-bound/dynamic-parallelism | core |
| `cudaLaunchHostFunc` | Enqueues a host function call in a stream | b/latency-bound/stream-concurrency | related |
| `cudaLaunchHostFunc_v2` | Enqueues a host function call with sync mode control | b/latency-bound/stream-concurrency | related |
| `cudaLaunchKernel` | Launches a device function (standard kernel launch) | b/latency-bound/kernel-launch-overhead, b/latency-bound/stream-concurrency | core |
| `cudaLaunchKernelExC` | Launches a kernel with launch-time configuration (cudaLaunchConfig_t: cluster dims, access policy, mem sync domain, programmatic events) | b/latency-bound/kernel-launch-overhead, b/memory-bound/l2-cache-control, b/synchronization-bound/memory-sync-domains, b/latency-bound/programmatic-dependent-launch | core |
| `cudaTriggerProgrammaticLaunchCompletion` | (__device__) Triggers programmatic launch completion for dependent launch | b/latency-bound/programmatic-dependent-launch | core |

## 6.8. Execution Control [DEPRECATED]

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaFuncSetSharedMemConfig` | Sets the shared memory bank size for a function (4-byte or 8-byte banks) | b/memory-bound/bank-conflict-avoidance | core |

## 6.9. Occupancy

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaOccupancyAvailableDynamicSMemPerBlock` | Returns max dynamic shared memory per block when launching N blocks per SM | b/latency-bound/occupancy-tuning, b/memory-bound/shared-memory-cache | core |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Returns max active blocks per SM for a function | b/latency-bound/occupancy-tuning | core |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | Returns max active blocks per SM with flags (caching override control) | b/latency-bound/occupancy-tuning | core |
| `cudaOccupancyMaxActiveClusters` | Returns max clusters that could co-exist on the device | b/latency-bound/occupancy-tuning | core |
| `cudaOccupancyMaxPotentialClusterSize` | Returns max cluster size for a kernel function | b/latency-bound/occupancy-tuning | core |

### C++ API Occupancy Functions (Section 6.34)

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaOccupancyMaxPotentialBlockSize` | (C++ template) Suggest block size to achieve maximum occupancy | b/latency-bound/occupancy-tuning | core |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` | (C++ template) Suggest block size with variable shared memory per block | b/latency-bound/occupancy-tuning, b/memory-bound/shared-memory-cache | core |
| `cudaOccupancyMaxPotentialBlockSizeWithFlags` | (C++ template) Suggest block size with flags for caching behavior | b/latency-bound/occupancy-tuning | core |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` | (C++ template) Suggest block size with variable shared memory and flags | b/latency-bound/occupancy-tuning, b/memory-bound/shared-memory-cache | core |

## 6.10. Memory Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaArrayGetInfo` | Gets info about the specified cudaArray | -- | low-relevance (array query) |
| `cudaArrayGetMemoryRequirements` | Returns memory requirements of a CUDA array | -- | low-relevance (array query) |
| `cudaArrayGetPlane` | Gets a CUDA array plane from a CUDA array | -- | low-relevance (array management) |
| `cudaArrayGetSparseProperties` | Returns the layout properties of a sparse CUDA array | -- | low-relevance (sparse array) |
| `cudaFree` | Frees device memory | b/memory-bound/host-device-transfer | related |
| `cudaFreeArray` | Frees an array on the device | -- | low-relevance (array management) |
| `cudaFreeHost` | Frees page-locked host memory | b/memory-bound/host-device-transfer | related |
| `cudaFreeMipmappedArray` | Frees a mipmapped array on the device | -- | low-relevance (texture management) |
| `cudaGetMipmappedArrayLevel` | Gets a mipmap level of a CUDA mipmapped array | -- | low-relevance (texture management) |
| `cudaGetSymbolAddress` | Finds the address associated with a CUDA symbol | -- | low-relevance (symbol management) |
| `cudaGetSymbolSize` | Finds the size of the object associated with a CUDA symbol | -- | low-relevance (symbol management) |
| `cudaHostAlloc` | Allocates page-locked (pinned) host memory with flags (mapped, portable, write-combined) | b/memory-bound/host-device-transfer, b/memory-bound/coalescing | core |
| `cudaHostGetDevicePointer` | Passes back device pointer of mapped host memory | b/memory-bound/host-device-transfer | related |
| `cudaHostGetFlags` | Passes back flags used to allocate pinned host memory | -- | low-relevance (query) |
| `cudaHostRegister` | Registers an existing host memory range for use by CUDA (page-locked, mapped, read-only) | b/memory-bound/host-device-transfer | core |
| `cudaHostUnregister` | Unregisters a memory range that was registered with cudaHostRegister | b/memory-bound/host-device-transfer | related |
| `cudaMalloc` | Allocate device memory | b/memory-bound/host-device-transfer | related |
| `cudaMalloc3D` | Allocates logical 3D device memory with pitch alignment | b/memory-bound/coalescing, b/memory-bound/layout-transform | related |
| `cudaMalloc3DArray` | Allocate a 3D CUDA array | -- | low-relevance (array management) |
| `cudaMallocArray` | Allocate a CUDA array | -- | low-relevance (array management) |
| `cudaMallocHost` | Allocates page-locked (pinned) host memory | b/memory-bound/host-device-transfer | core |
| `cudaMallocManaged` | Allocates memory accessible from host and device (unified memory) | b/memory-bound/unified-memory | core |
| `cudaMallocMipmappedArray` | Allocates a mipmapped CUDA array | -- | low-relevance (texture management) |
| `cudaMallocPitch` | Allocates pitched device memory (ensures coalesced access for 2D arrays) | b/memory-bound/coalescing, b/memory-bound/layout-transform | core |
| `cudaMemAdvise` | Advise about usage of a memory range (read-mostly, preferred location, accessed-by) | b/memory-bound/unified-memory, b/memory-bound/data-prefetch, b/memory-bound/numa-binding | core |
| `cudaMemDiscardAndPrefetchBatchAsync` | Batch memory discards followed by prefetches | b/memory-bound/unified-memory, b/memory-bound/data-prefetch | core |
| `cudaMemDiscardBatchAsync` | Batch memory discards (informs driver contents are no longer useful) | b/memory-bound/unified-memory, b/memory-bound/data-prefetch | related |
| `cudaMemGetInfo` | Gets free and total device memory | -- | low-relevance (memory query) |
| `cudaMemPrefetchAsync` | Prefetches memory to a specified destination location | b/memory-bound/data-prefetch, b/memory-bound/unified-memory, b/memory-bound/numa-binding | core |
| `cudaMemPrefetchBatchAsync` | Batch prefetch of multiple memory ranges | b/memory-bound/data-prefetch, b/memory-bound/unified-memory | core |
| `cudaMemRangeGetAttribute` | Query an attribute of a given memory range | b/memory-bound/unified-memory | related |
| `cudaMemRangeGetAttributes` | Query attributes of a given memory range | b/memory-bound/unified-memory | related |
| `cudaMemcpy` | Copies data between host and device | b/memory-bound/host-device-transfer | core |
| `cudaMemcpy2D` | Copies data between host and device (2D, with pitch) | b/memory-bound/host-device-transfer, b/memory-bound/coalescing | related |
| `cudaMemcpy2DArrayToArray` | Copies data between 2D CUDA arrays | -- | low-relevance (array copy) |
| `cudaMemcpy2DAsync` | Copies data between host and device (2D, async) | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | core |
| `cudaMemcpy2DFromArray` | Copies data from a CUDA array to host/device memory (2D) | -- | low-relevance (array copy) |
| `cudaMemcpy2DFromArrayAsync` | Async copy from CUDA array (2D) | -- | low-relevance (array copy) |
| `cudaMemcpy2DToArray` | Copies data to a CUDA array (2D) | -- | low-relevance (array copy) |
| `cudaMemcpy2DToArrayAsync` | Async copy to CUDA array (2D) | -- | low-relevance (array copy) |
| `cudaMemcpy3D` | Copies data between 3D objects | b/memory-bound/host-device-transfer | related |
| `cudaMemcpy3DAsync` | Copies data between 3D objects (async) | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | related |
| `cudaMemcpy3DBatchAsync` | Batch async 3D memory copies | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | related |
| `cudaMemcpy3DPeer` | Copies memory between devices | b/memory-bound/host-device-transfer | related |
| `cudaMemcpy3DPeerAsync` | Copies memory between devices asynchronously | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | related |
| `cudaMemcpy3DWithAttributesAsync` | 3D async copy with memory copy attributes (e.g., src/dst access hints) | b/memory-bound/host-device-transfer, b/memory-bound/cache-load-hints | related |
| `cudaMemcpyAsync` | Copies data between host and device asynchronously | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | core |
| `cudaMemcpyBatchAsync` | Batch async memory copies | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | core |
| `cudaMemcpyFromSymbol` | Copies data from the given symbol on the device | -- | low-relevance (symbol copy) |
| `cudaMemcpyFromSymbolAsync` | Copies data from a symbol on the device asynchronously | -- | low-relevance (symbol copy) |
| `cudaMemcpyPeer` | Copies memory between two devices | b/memory-bound/host-device-transfer | related |
| `cudaMemcpyPeerAsync` | Copies memory between two devices asynchronously | b/memory-bound/host-device-transfer, b/latency-bound/stream-concurrency | related |
| `cudaMemcpyToSymbol` | Copies data to the given symbol on the device | -- | low-relevance (symbol copy) |
| `cudaMemcpyToSymbolAsync` | Copies data to a symbol on the device asynchronously | -- | low-relevance (symbol copy) |
| `cudaMemcpyWithAttributesAsync` | Async copy with memory copy attributes (src/dst access policy hints) | b/memory-bound/cache-load-hints, b/memory-bound/host-device-transfer | related |
| `cudaMemset` | Initializes or sets device memory to a value | -- | low-relevance (memory initialization) |
| `cudaMemset2D` | Initializes or sets device memory to a value (2D) | -- | low-relevance (memory initialization) |
| `cudaMemset2DAsync` | Initializes or sets device memory asynchronously (2D) | -- | low-relevance (memory initialization) |
| `cudaMemset3D` | Initializes or sets device memory to a value (3D) | -- | low-relevance (memory initialization) |
| `cudaMemset3DAsync` | Initializes or sets device memory asynchronously (3D) | -- | low-relevance (memory initialization) |
| `cudaMemsetAsync` | Initializes or sets device memory asynchronously | b/latency-bound/stream-concurrency | related |
| `cudaMipmappedArrayGetMemoryRequirements` | Returns memory requirements of a mipmapped array | -- | low-relevance (texture management) |
| `cudaMipmappedArrayGetSparseProperties` | Returns the sparse properties of a mipmapped array | -- | low-relevance (sparse array) |

## 6.11. Memory Management [DEPRECATED]

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaMemcpyArrayToArray` | Copies data between host and device (deprecated) | -- | low-relevance (deprecated array copy) |
| `cudaMemcpyFromArray` | Copies data from a CUDA array (deprecated) | -- | low-relevance (deprecated array copy) |
| `cudaMemcpyFromArrayAsync` | Async copy from CUDA array (deprecated) | -- | low-relevance (deprecated array copy) |
| `cudaMemcpyToArray` | Copies data to a CUDA array (deprecated) | -- | low-relevance (deprecated array copy) |
| `cudaMemcpyToArrayAsync` | Async copy to CUDA array (deprecated) | -- | low-relevance (deprecated array copy) |

## 6.12. Stream Ordered Memory Allocator

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaFreeAsync` | Frees memory asynchronously (stream-ordered) | b/latency-bound/stream-concurrency, b/memory-bound/host-device-transfer | core |
| `cudaMallocAsync` | Allocates memory asynchronously (stream-ordered) | b/latency-bound/stream-concurrency, b/memory-bound/host-device-transfer | core |
| `cudaMallocFromPoolAsync` | Allocates memory from a specified pool asynchronously | b/latency-bound/stream-concurrency, b/memory-bound/host-device-transfer | core |
| `cudaMemGetDefaultMemPool` | Returns the default memory pool of a device | b/memory-bound/host-device-transfer | related |
| `cudaMemGetMemPool` | Gets the memory pool for a device | b/memory-bound/host-device-transfer | related |
| `cudaMemPoolCreate` | Creates a memory pool | b/latency-bound/kernel-launch-overhead, b/memory-bound/host-device-transfer | core |
| `cudaMemPoolDestroy` | Destroys a memory pool | b/memory-bound/host-device-transfer | related |
| `cudaMemPoolExportPointer` | Export data to share a memory pool allocation between processes | -- | low-relevance (IPC) |
| `cudaMemPoolExportToShareableHandle` | Exports a memory pool to a requested type | -- | low-relevance (IPC) |
| `cudaMemPoolGetAccess` | Returns the accessibility of a pool from a device | b/memory-bound/host-device-transfer | related |
| `cudaMemPoolGetAttribute` | Gets attributes of a memory pool (reuse policy, release threshold) | b/memory-bound/host-device-transfer | related |
| `cudaMemPoolImportFromShareableHandle` | Imports a memory pool from a shared handle | -- | low-relevance (IPC) |
| `cudaMemPoolImportPointer` | Import a memory pool allocation from another process | -- | low-relevance (IPC) |
| `cudaMemPoolSetAccess` | Controls the pool's accessibility on the specified device | b/memory-bound/host-device-transfer, b/memory-bound/numa-binding | related |
| `cudaMemPoolSetAttribute` | Sets attributes of a memory pool (reuse policy, release threshold) | b/memory-bound/host-device-transfer | related |
| `cudaMemPoolTrimTo` | Releases memory back to the OS up to specified threshold | b/memory-bound/host-device-transfer | related |
| `cudaMemSetMemPool` | Sets the current memory pool for a device | b/memory-bound/host-device-transfer | related |

## 6.13. Unified Addressing

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaPointerGetAttributes` | Returns attributes about a specified pointer (type, device, hostPointer, devicePointer) | b/memory-bound/unified-memory, b/memory-bound/host-device-transfer | related |

## 6.14. Peer Device Memory Access

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaDeviceCanAccessPeer` | Queries if a device may directly access a peer device's memory | b/memory-bound/host-device-transfer, b/memory-bound/numa-binding | related |
| `cudaDeviceDisablePeerAccess` | Disables direct access to peer device memory | b/memory-bound/host-device-transfer | related |
| `cudaDeviceEnablePeerAccess` | Enables direct access to peer device memory allocations | b/memory-bound/host-device-transfer, b/memory-bound/numa-binding | core |

## 6.15-6.16. OpenGL Interoperability

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaGLGetDevices` | Gets CUDA devices associated with the current OpenGL context | -- | low-relevance (graphics interop) |
| `cudaGraphicsGLRegisterBuffer` | Registers an OpenGL buffer object for access by CUDA | -- | low-relevance (graphics interop) |
| `cudaGraphicsGLRegisterImage` | Register an OpenGL texture/renderbuffer for access by CUDA | -- | low-relevance (graphics interop) |
| `cudaWGLGetDevice` | Gets the CUDA device associated with an hGpu (Windows) | -- | low-relevance (graphics interop) |
| *(deprecated GL APIs)* | cudaGLMapBufferObject, cudaGLUnmapBufferObject, etc. | -- | low-relevance (deprecated graphics interop) |

## 6.17-6.22. Direct3D Interoperability (D3D9, D3D10, D3D11)

All Direct3D interop APIs (both current and deprecated) are classified as **low-relevance** (graphics interop). These include:

`cudaD3D9GetDevice`, `cudaD3D9GetDevices`, `cudaGraphicsD3D9RegisterResource`, `cudaD3D10GetDevice`, `cudaD3D10GetDevices`, `cudaGraphicsD3D10RegisterResource`, `cudaD3D11GetDevice`, `cudaD3D11GetDevices`, `cudaGraphicsD3D11RegisterResource`, and their deprecated counterparts.

## 6.23. VDPAU Interoperability

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaGraphicsVDPAURegisterOutputSurface` | Register a VDPAU output surface | -- | low-relevance (graphics interop) |
| `cudaGraphicsVDPAURegisterVideoSurface` | Register a VDPAU video surface | -- | low-relevance (graphics interop) |
| `cudaVDPAUGetDevice` | Gets the CUDA device associated with a VDPAU device | -- | low-relevance (graphics interop) |
| `cudaVDPAUSetVDPAUDevice` | Sets a CUDA device to use VDPAU interoperability | -- | low-relevance (graphics interop) |

## 6.24. EGL Interoperability

All EGL interop APIs are classified as **low-relevance** (graphics interop). These include:

`cudaEGLStreamConsumerAcquireFrame`, `cudaEGLStreamConsumerConnect`, `cudaEGLStreamConsumerConnectWithFlags`, `cudaEGLStreamConsumerDisconnect`, `cudaEGLStreamConsumerReleaseFrame`, `cudaEGLStreamProducerConnect`, `cudaEGLStreamProducerDisconnect`, `cudaEGLStreamProducerPresentFrame`, `cudaEGLStreamProducerReturnFrame`, `cudaEventCreateFromEGLSync`, `cudaGraphicsEGLRegisterImage`, `cudaGraphicsResourceGetMappedEglFrame`.

## 6.25. Graphics Interoperability

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaGraphicsMapResources` | Map graphics resources for access by CUDA | -- | low-relevance (graphics interop) |
| `cudaGraphicsResourceGetMappedMipmappedArray` | Get mipmapped array through which to access a mapped graphics resource | -- | low-relevance (graphics interop) |
| `cudaGraphicsResourceGetMappedPointer` | Get device pointer through which to access a mapped graphics resource | -- | low-relevance (graphics interop) |
| `cudaGraphicsResourceSetMapFlags` | Set usage flags for mapping a graphics resource | -- | low-relevance (graphics interop) |
| `cudaGraphicsSubResourceGetMappedArray` | Get array through which to access a subresource | -- | low-relevance (graphics interop) |
| `cudaGraphicsUnmapResources` | Unmap graphics resources | -- | low-relevance (graphics interop) |
| `cudaGraphicsUnregisterResource` | Unregisters a graphics resource for access by CUDA | -- | low-relevance (graphics interop) |

## 6.26. Texture Object Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaCreateChannelDesc` | Returns a channel descriptor using the specified format | b/memory-bound/cache-load-hints | related |
| `cudaCreateTextureObject` | Creates a texture object with specified resource, texture, and resource view descriptors | b/memory-bound/cache-load-hints | related |
| `cudaDestroyTextureObject` | Destroys a texture object | -- | low-relevance (texture cleanup) |
| `cudaGetChannelDesc` | Get the channel descriptor of a CUDA array | -- | low-relevance (texture query) |
| `cudaGetTextureObjectResourceDesc` | Returns a texture object's resource descriptor | -- | low-relevance (texture query) |
| `cudaGetTextureObjectResourceViewDesc` | Returns a texture object's resource view descriptor | -- | low-relevance (texture query) |
| `cudaGetTextureObjectTextureDesc` | Returns a texture object's texture descriptor | -- | low-relevance (texture query) |

## 6.27. Surface Object Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaCreateSurfaceObject` | Creates a surface object | -- | low-relevance (surface management) |
| `cudaDestroySurfaceObject` | Destroys a surface object | -- | low-relevance (surface management) |
| `cudaGetSurfaceObjectResourceDesc` | Returns a surface object's resource descriptor | -- | low-relevance (surface management) |

## 6.28. Version Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaDriverGetVersion` | Returns the latest version of the CUDA driver API | -- | low-relevance (version query) |
| `cudaRuntimeGetVersion` | Returns the version number of the CUDA runtime | -- | low-relevance (version query) |

## 6.29. Error Log Management Functions

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaLogsCurrent` | Returns the current error log handle | -- | low-relevance (error logging) |
| `cudaLogsDumpToFile` | Dump error logs to a file | -- | low-relevance (error logging) |
| `cudaLogsDumpToMemory` | Dump error logs to memory | -- | low-relevance (error logging) |
| `cudaLogsRegisterCallback` | Registers a callback for error logs | -- | low-relevance (error logging) |
| `cudaLogsUnregisterCallback` | Unregisters an error log callback | -- | low-relevance (error logging) |

## 6.30. Graph Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaDeviceGetGraphMemAttribute` | Query async allocation attributes related to graphs | b/latency-bound/cuda-graphs | related |
| `cudaDeviceGraphMemTrim` | Free unused cached graph memory back to OS | b/latency-bound/cuda-graphs | related |
| `cudaDeviceSetGraphMemAttribute` | Set async allocation attributes related to graphs | b/latency-bound/cuda-graphs | related |
| `cudaGetCurrentGraphExec` | (__device__) Get the currently running device graph id | b/latency-bound/cuda-graphs | related |
| `cudaGraphCreate` | Creates a graph | b/latency-bound/cuda-graphs | core |
| `cudaGraphDestroy` | Destroys a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphInstantiate` | Creates an executable graph from a graph | b/latency-bound/cuda-graphs | core |
| `cudaGraphInstantiateWithFlags` | Creates an executable graph from a graph with flags | b/latency-bound/cuda-graphs | core |
| `cudaGraphInstantiateWithParams` | Creates an executable graph from a graph with params (upload stream, error info) | b/latency-bound/cuda-graphs | core |
| `cudaGraphLaunch` | Launches an executable graph in a stream | b/latency-bound/cuda-graphs, b/latency-bound/kernel-launch-overhead | core |
| `cudaGraphUpload` | Uploads an executable graph to a device | b/latency-bound/cuda-graphs | core |
| `cudaGraphExecUpdate` | Updates an instantiated graph with changes from the source graph | b/latency-bound/cuda-graphs | core |
| `cudaGraphExecDestroy` | Destroys an executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddKernelNode` | Creates a kernel execution node and adds it to a graph | b/latency-bound/cuda-graphs | core |
| `cudaGraphAddMemAllocNode` | Creates a memory allocation node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddMemFreeNode` | Creates a memory free node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddMemcpyNode` | Creates a memcpy node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddMemcpyNode1D` | Creates a 1D memcpy node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddMemsetNode` | Creates a memset node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddHostNode` | Creates a host execution node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddChildGraphNode` | Creates a child graph node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddEmptyNode` | Creates an empty node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddEventRecordNode` | Creates an event record node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddEventWaitNode` | Creates an event wait node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddExternalSemaphoresSignalNode` | Creates an external semaphore signal node | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddExternalSemaphoresWaitNode` | Creates an external semaphore wait node | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddDependencies` | Adds dependency edges to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphRemoveDependencies` | Removes dependency edges from a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphAddNode` | Creates a generic typed node and adds it to a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphClone` | Clones a graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphConditionalHandleCreate` | Create a conditional handle for graph conditional nodes | b/latency-bound/cuda-graphs | related |
| `cudaGraphConditionalHandleCreate_v2` | Create a conditional handle with extended params | b/latency-bound/cuda-graphs | related |
| `cudaGraphSetConditional` | Set conditional handle value | b/latency-bound/cuda-graphs | related |
| `cudaGraphDebugDotPrint` | Write graph to DOT file for debugging | -- | low-relevance (debug) |
| `cudaGraphDestroyNode` | Remove a node from the graph and destroy it | b/latency-bound/cuda-graphs | related |
| `cudaGraphGetEdges` | Returns a graph's dependency edges | b/latency-bound/cuda-graphs | related |
| `cudaGraphGetNodes` | Returns a graph's nodes | b/latency-bound/cuda-graphs | related |
| `cudaGraphGetRootNodes` | Returns a graph's root nodes | b/latency-bound/cuda-graphs | related |
| `cudaGraphGetId` | Returns a unique id for the graph | -- | low-relevance (query) |
| `cudaGraphExecGetFlags` | Returns the flags that were passed to instantiation | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecGetId` | Returns a unique id for the graph exec | -- | low-relevance (query) |
| `cudaGraphKernelNodeCopyAttributes` | Copy attributes from one kernel node to another | b/latency-bound/cuda-graphs | related |
| `cudaGraphKernelNodeGetAttribute` | Queries node attribute (e.g., access policy window) | b/latency-bound/cuda-graphs, b/memory-bound/l2-cache-control | related |
| `cudaGraphKernelNodeGetParams` | Gets a kernel node's parameters | b/latency-bound/cuda-graphs | related |
| `cudaGraphKernelNodeSetAttribute` | Sets node attribute (e.g., access policy window) | b/latency-bound/cuda-graphs, b/memory-bound/l2-cache-control | related |
| `cudaGraphKernelNodeSetEnabled` | Enables or disables a kernel node | b/latency-bound/cuda-graphs | related |
| `cudaGraphKernelNodeSetGridDim` | Updates the grid dimensions of a kernel node | b/latency-bound/cuda-graphs | related |
| `cudaGraphKernelNodeSetParam` | (__device__) Sets a single parameter of a kernel node from device code | b/latency-bound/cuda-graphs | related |
| `cudaGraphKernelNodeSetParams` | Sets a kernel node's parameters | b/latency-bound/cuda-graphs | related |
| `cudaGraphKernelNodeUpdatesApply` | (__device__) Applies kernel node parameter updates from device code | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecKernelNodeSetParams` | Sets the params of a kernel node in an executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecMemcpyNodeSetParams` | Sets memcpy node params in an executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecMemsetNodeSetParams` | Sets memset node params in an executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecHostNodeSetParams` | Sets host node params in an executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecChildGraphNodeSetParams` | Updates child graph node params in executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecEventRecordNodeSetEvent` | Sets event for event record node in executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecEventWaitNodeSetEvent` | Sets event for event wait node in executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphExecNodeSetParams` | Update params of any node type in an executable graph | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeGetType` | Returns a node's type | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeGetParams` | Returns params for a node of any type | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeSetParams` | Update params for any node type | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeSetEnabled` | Enables/disables a node | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeGetEnabled` | Queries whether a node is enabled | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeFindInClone` | Finds a node in a cloned graph corresponding to original | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeGetDependencies` | Returns a node's dependencies | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeGetDependentNodes` | Returns a node's dependent nodes | b/latency-bound/cuda-graphs | related |
| `cudaGraphNodeGetContainingGraph` | Returns the graph that contains the node | -- | low-relevance (query) |
| `cudaGraphNodeGetLocalId` | Returns the local ID of a node within its graph | -- | low-relevance (query) |
| `cudaGraphNodeGetToolsId` | Returns the tools ID of a node | -- | low-relevance (debug) |
| `cudaGraphReleaseUserObject` | Release a user object reference from a graph | -- | low-relevance (lifecycle) |
| `cudaGraphRetainUserObject` | Retains a reference to a user object from a graph | -- | low-relevance (lifecycle) |
| `cudaUserObjectCreate` | Create a user object | -- | low-relevance (lifecycle) |
| `cudaUserObjectRelease` | Release a reference to a user object | -- | low-relevance (lifecycle) |
| `cudaUserObjectRetain` | Retain a reference to a user object | -- | low-relevance (lifecycle) |

Remaining graph node get/set APIs (`cudaGraphHostNodeGetParams`, `cudaGraphHostNodeSetParams`, `cudaGraphMemAllocNodeGetParams`, `cudaGraphMemFreeNodeGetParams`, `cudaGraphMemcpyNodeGetParams`, `cudaGraphMemcpyNodeSetParams`, `cudaGraphMemcpyNodeSetParams1D`, `cudaGraphMemsetNodeGetParams`, `cudaGraphMemsetNodeSetParams`, `cudaGraphEventRecordNodeGetEvent`, `cudaGraphEventRecordNodeSetEvent`, `cudaGraphEventWaitNodeGetEvent`, `cudaGraphEventWaitNodeSetEvent`, `cudaGraphChildGraphNodeGetGraph`, `cudaGraphExternalSemaphoresSignalNodeGetParams`, `cudaGraphExternalSemaphoresSignalNodeSetParams`, `cudaGraphExternalSemaphoresWaitNodeGetParams`, `cudaGraphExternalSemaphoresWaitNodeSetParams`, and their `GraphExec` variants) are all **related** to `b/latency-bound/cuda-graphs`.

## 6.31. Driver Entry Point Access

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaGetDriverEntryPoint` | Returns the requested driver API function pointer | -- | low-relevance (driver interop) |
| `cudaGetDriverEntryPointByVersion` | Returns driver API function pointer by version | -- | low-relevance (driver interop) |

## 6.32. Library Management

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaKernelSetAttributeForDevice` | Sets attribute for a kernel on a specified device | b/latency-bound/occupancy-tuning, b/memory-bound/shared-memory-cache | related |
| `cudaLibraryEnumerateKernels` | Enumerates kernels in a library | -- | low-relevance (library management) |
| `cudaLibraryGetGlobal` | Returns a global device pointer from a library | -- | low-relevance (library management) |
| `cudaLibraryGetKernel` | Returns a kernel handle from a library | b/latency-bound/kernel-launch-overhead | related |
| `cudaLibraryGetKernelCount` | Returns number of kernels in a library | -- | low-relevance (library management) |
| `cudaLibraryGetManaged` | Returns a managed pointer from a library | -- | low-relevance (library management) |
| `cudaLibraryGetUnifiedFunction` | Returns a unified function pointer from a library | -- | low-relevance (library management) |
| `cudaLibraryLoadData` | Loads a library from data in memory | -- | low-relevance (library management) |
| `cudaLibraryLoadFromFile` | Loads a library from a file | -- | low-relevance (library management) |
| `cudaLibraryUnload` | Unloads a library | -- | low-relevance (library management) |

## 6.33. Execution Context Management (Green Contexts)

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaDevResourceGenerateDesc` | Generates a resource descriptor from specified resources (SM, workqueue) for green context creation | b/latency-bound/green-contexts | core |
| `cudaDevSmResourceSplit` | Splits SM resources into structured groups (with co-scheduling control) | b/latency-bound/green-contexts, b/latency-bound/work-stealing | core |
| `cudaDevSmResourceSplitByCount` | Splits SM resources by count with default granularity | b/latency-bound/green-contexts | core |
| `cudaDeviceGetDevResource` | Gets device resources (SM count, alignment, workqueue config) | b/latency-bound/green-contexts | core |
| `cudaDeviceGetExecutionCtx` | Returns the execution context for a device's primary context | b/latency-bound/green-contexts, b/latency-bound/context-management | core |
| `cudaExecutionCtxDestroy` | Destroys an execution context | b/latency-bound/green-contexts | related |
| `cudaExecutionCtxGetDevResource` | Gets device resource from an execution context | b/latency-bound/green-contexts | related |
| `cudaExecutionCtxGetDevice` | Gets the device associated with an execution context | b/latency-bound/green-contexts | related |
| `cudaExecutionCtxGetId` | Gets the unique ID of an execution context | -- | low-relevance (query) |
| `cudaExecutionCtxRecordEvent` | Records an event on the execution context | b/latency-bound/green-contexts, b/latency-bound/stream-concurrency | related |
| `cudaExecutionCtxStreamCreate` | Creates a stream on the execution context | b/latency-bound/green-contexts, b/latency-bound/stream-concurrency | core |
| `cudaExecutionCtxSynchronize` | Synchronizes the execution context | b/latency-bound/green-contexts | related |
| `cudaExecutionCtxWaitEvent` | Makes the execution context wait on an event | b/latency-bound/green-contexts | related |
| `cudaGreenCtxCreate` | Creates a green context with specified resources | b/latency-bound/green-contexts | core |
| `cudaStreamGetDevResource` | Gets device resource from a stream | b/latency-bound/green-contexts | related |

## 6.34. C++ API Routines (Selected Optimization-Relevant)

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaCreateChannelDesc` | (C++ template) Create a channel descriptor | -- | low-relevance (texture helper) |
| `cudaEventCreate` | (C++ overload) Create an event | b/latency-bound/stream-concurrency | related |
| `cudaFuncGetAttributes` | (C++ template) Get function attributes | b/latency-bound/occupancy-tuning | core |
| `cudaFuncSetAttribute` | (C++ template) Set function attributes | b/memory-bound/shared-memory-cache, b/latency-bound/occupancy-tuning | core |
| `cudaFuncSetCacheConfig` | (C++ template) Set cache config for a function | b/memory-bound/shared-memory-cache | core |
| `cudaGetKernel` | Returns a cudaKernel_t handle for a __global__ function | b/latency-bound/kernel-launch-overhead | related |
| `cudaLaunchCooperativeKernel` | (C++ template) Launch cooperative kernel | b/synchronization-bound/cooperative-groups | core |
| `cudaLaunchKernel` | (C++ template) Launch kernel | b/latency-bound/kernel-launch-overhead | core |
| `cudaLaunchKernelEx` | (C++ template) Launch kernel with extended config (cluster, L2 policy, mem sync domain) | b/latency-bound/kernel-launch-overhead, b/memory-bound/l2-cache-control, b/synchronization-bound/memory-sync-domains | core |
| `cudaMallocHost` | (C++ template) Allocate pinned host memory | b/memory-bound/host-device-transfer | core |
| `cudaMallocManaged` | (C++ template) Allocate managed memory | b/memory-bound/unified-memory | core |
| `cudaMallocAsync` | (C++ template) Stream-ordered async allocation | b/latency-bound/stream-concurrency | core |
| `cudaStreamAttachMemAsync` | (C++ template) Attach memory to stream | b/memory-bound/unified-memory | core |
| `cudaGraphInstantiate` | (C++ template) Instantiate a graph | b/latency-bound/cuda-graphs | core |
| `cudaGetFuncBySymbol` | Gets the entry function address for a __global__ function | -- | low-relevance (introspection) |

## 6.35. Interactions with the CUDA Driver API

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| -- | (Section describes interop rules, not new APIs) | -- | low-relevance (driver interop) |

## 6.36. Profiler Control

| API | Description | Knowledge Node | Relevance |
|-----|-------------|----------------|-----------|
| `cudaProfilerStart` | Enable profiling | -- | low-relevance (profiling) |
| `cudaProfilerStop` | Disable profiling | -- | low-relevance (profiling) |

## Device-Side Built-in Functions and Intrinsics

These are device-side built-ins referenced throughout the documentation but not part of the runtime API proper. They are critical for kernel optimization:

| API / Intrinsic | Description | Knowledge Node | Relevance |
|-----------------|-------------|----------------|-----------|
| `__syncthreads()` | Block-level barrier synchronization | b/synchronization-bound/barrier-optimization, b/memory-bound/shared-memory-cache | core |
| `__syncwarp(mask)` | Warp-level synchronization | b/synchronization-bound/barrier-optimization, b/compute-bound/warp-primitives | core |
| `__threadfence()` | Memory fence (device scope) | b/synchronization-bound/thread-scopes, b/synchronization-bound/memory-sync-domains | core |
| `__threadfence_block()` | Memory fence (block scope) | b/synchronization-bound/thread-scopes | core |
| `__threadfence_system()` | Memory fence (system scope, including host) | b/synchronization-bound/thread-scopes, b/synchronization-bound/memory-sync-domains | core |
| `atomicAdd()` | Atomic addition | b/synchronization-bound/atomic-reduction | core |
| `atomicSub()` | Atomic subtraction | b/synchronization-bound/atomic-reduction | related |
| `atomicExch()` | Atomic exchange | b/synchronization-bound/atomic-reduction | related |
| `atomicMin()` / `atomicMax()` | Atomic min/max | b/synchronization-bound/atomic-reduction | related |
| `atomicCAS()` | Atomic compare-and-swap | b/synchronization-bound/atomic-reduction | core |
| `atomicAnd()` / `atomicOr()` / `atomicXor()` | Atomic bitwise operations | b/synchronization-bound/atomic-reduction | related |
| `__shfl_sync()` | Warp shuffle (direct indexed) | b/compute-bound/warp-primitives, b/memory-bound/shared-memory-cache | core |
| `__shfl_up_sync()` | Warp shuffle up | b/compute-bound/warp-primitives | core |
| `__shfl_down_sync()` | Warp shuffle down (used in reductions) | b/compute-bound/warp-primitives, pattern/reduction | core |
| `__shfl_xor_sync()` | Warp shuffle XOR (butterfly reduction) | b/compute-bound/warp-primitives, pattern/reduction | core |
| `__ballot_sync()` | Warp vote (returns bitmask of predicate across warp) | b/compute-bound/warp-primitives, b/latency-bound/warp-divergence | core |
| `__any_sync()` | Warp vote (any predicate true) | b/compute-bound/warp-primitives, b/latency-bound/warp-divergence | related |
| `__all_sync()` | Warp vote (all predicates true) | b/compute-bound/warp-primitives, b/latency-bound/warp-divergence | related |
| `__match_any_sync()` | Warp match (finds threads with matching value) | b/compute-bound/warp-primitives | related |
| `__match_all_sync()` | Warp match (checks if all threads have same value) | b/compute-bound/warp-primitives | related |
| `__activemask()` | Returns mask of active threads in the warp | b/compute-bound/warp-primitives, b/latency-bound/warp-divergence | related |
| `__ldg()` | Read-only data cache load (through texture cache path) | b/memory-bound/cache-load-hints | core |
| `__ldcg()` | Cache at global level, evict first | b/memory-bound/cache-load-hints | core |
| `__ldca()` | Cache at all levels | b/memory-bound/cache-load-hints | core |
| `__ldcs()` | Cache streaming (likely to be accessed once) | b/memory-bound/cache-load-hints | core |
| `__ldlu()` | Load last use (evict after use) | b/memory-bound/cache-load-hints | core |
| `__stcg()` | Store cache at global level | b/memory-bound/cache-load-hints | related |
| `__stcs()` | Store cache streaming | b/memory-bound/cache-load-hints | related |
| `__stwb()` | Store write-back | b/memory-bound/cache-load-hints | related |
| `__stwt()` | Store write-through | b/memory-bound/cache-load-hints | related |
| `__nanosleep()` | Put thread to sleep for approximately ns nanoseconds | b/latency-bound/warp-divergence | related |
| `cooperative_groups::*` | Cooperative Groups API (thread_block, grid_group, tiled_partition, etc.) | b/synchronization-bound/cooperative-groups | core |
| `__pipeline_memcpy_async()` | Async copy from global to shared memory (hardware-accelerated) | b/memory-bound/data-prefetch, b/memory-bound/shared-memory-cache | core |
| `__pipeline_commit()` | Commit outstanding async copies | b/memory-bound/data-prefetch | core |
| `__pipeline_wait_prior()` | Wait for prior pipeline stages to complete | b/memory-bound/data-prefetch | core |
| `cuda::memcpy_async()` | C++ async memory copy (global to shared) | b/memory-bound/data-prefetch, b/memory-bound/shared-memory-cache | core |
| `cuda::barrier` | C++ barrier with arrive/wait pattern | b/synchronization-bound/barrier-optimization | core |
| `cuda::atomic` | Scoped atomic operations (thread/block/device/system scope) | b/synchronization-bound/thread-scopes, b/synchronization-bound/atomic-reduction | core |
| `nvcuda::wmma::*` | Warp Matrix Multiply-Accumulate (WMMA) API for Tensor Cores | b/compute-bound/tensor-core, pattern/gemm | core |
| `mma_sync()` | PTX-level matrix multiply-accumulate for Tensor Cores | b/compute-bound/tensor-core, pattern/gemm | core |

## Key Data Structures for Optimization

| Structure | Description | Knowledge Node | Relevance |
|-----------|-------------|----------------|-----------|
| `cudaAccessPolicyWindow` | Specifies L2 access policy window (base_ptr, num_bytes, hit/miss ratio, persistent/streaming) | b/memory-bound/l2-cache-control | core |
| `cudaDeviceProp` | Device properties (sharedMemPerBlock, regsPerBlock, warpSize, maxThreadsPerBlock, etc.) | b/latency-bound/occupancy-tuning | core |
| `cudaFuncAttributes` | Function attributes (numRegs, sharedSizeBytes, maxThreadsPerBlock, etc.) | b/latency-bound/occupancy-tuning, b/memory-bound/register-pressure | core |
| `cudaLaunchConfig_t` | Launch configuration (gridDim, blockDim, dynamicSmemBytes, stream, attrs for L2 policy, cluster, mem sync domain) | b/latency-bound/kernel-launch-overhead, b/memory-bound/l2-cache-control | core |
| `cudaLaunchAttribute` | Launch attribute (id + value) for cudaLaunchKernelExC | b/memory-bound/l2-cache-control, b/synchronization-bound/memory-sync-domains | core |
| `cudaLaunchAttributeValue` | Union holding cluster dims, programmatic event, mem sync domain, access policy, shared memory mode | b/memory-bound/l2-cache-control, b/latency-bound/programmatic-dependent-launch, b/synchronization-bound/memory-sync-domains | core |
| `cudaLaunchMemSyncDomainMap` | Maps default/remote memory sync domains | b/synchronization-bound/memory-sync-domains | core |
| `cudaMemLocation` | Specifies memory location (type + id, supports device, host, host NUMA) | b/memory-bound/numa-binding, b/memory-bound/unified-memory | core |
| `cudaMemPoolProps` | Memory pool properties (allocation type, location, handle type) | b/memory-bound/host-device-transfer | related |
| `cudaMemcpyAttributes` | Memory copy attributes (src/dst access flags) | b/memory-bound/cache-load-hints | related |
| `cudaDevResource` | Device resource (SM or workqueue) for green context partitioning | b/latency-bound/green-contexts | core |
| `cudaDevSmResource` | SM resource with count and co-scheduling alignment | b/latency-bound/green-contexts | core |
| `cudaDevSmResourceGroupParams` | SM resource group params for splitting (smCount, coscheduledSmCount) | b/latency-bound/green-contexts | core |
| `cudaDevWorkqueueConfigResource` | Workqueue configuration (concurrency limit, sharing scope) | b/latency-bound/green-contexts | core |

## Key Enumerations for Optimization

| Enumeration | Description | Knowledge Node | Relevance |
|-------------|-------------|----------------|-----------|
| `cudaFuncCache` | L1 vs shared memory preference (PreferNone, PreferShared, PreferL1, PreferEqual) | b/memory-bound/shared-memory-cache, b/memory-bound/cache-load-hints | core |
| `cudaSharedMemConfig` | Shared memory bank size (Default, FourByte, EightByte) | b/memory-bound/bank-conflict-avoidance | core |
| `cudaSharedCarveout` | Shared memory carveout values (Default, MaxShared, MaxL1) | b/memory-bound/shared-memory-cache | core |
| `cudaSharedMemoryMode` | Shared memory mode (Default, 8ByteBankSize, 16ByteBankSize) | b/memory-bound/bank-conflict-avoidance | core |
| `cudaFuncAttribute` | Function attribute identifiers (MaxDynamicSharedMemorySize, PreferredSharedMemoryCarveout, RequiredCluster*) | b/memory-bound/shared-memory-cache, b/latency-bound/occupancy-tuning | core |
| `cudaLimit` | Resource limits (StackSize, PrintfFifoSize, MallocHeapSize, DevRuntimeSyncDepth, MaxL2FetchGranularity, PersistingL2CacheSize) | b/memory-bound/l2-cache-control, b/latency-bound/dynamic-parallelism | core |
| `cudaMemoryAdvise` | Memory advice (SetReadMostly, SetPreferredLocation, SetAccessedBy and Unset variants) | b/memory-bound/unified-memory, b/memory-bound/data-prefetch | core |
| `cudaLaunchAttributeID` | Launch attribute identifiers (AccessPolicyWindow, CooperativeLaunch, ProgrammaticStreamSerialization, ProgrammaticEvent, ClusterDim, MemSyncDomain, SharedMemoryMode, etc.) | b/memory-bound/l2-cache-control, b/latency-bound/programmatic-dependent-launch, b/synchronization-bound/memory-sync-domains | core |
| `cudaLaunchMemSyncDomain` | Memory sync domain identifiers (Default, Remote) | b/synchronization-bound/memory-sync-domains | core |
| `cudaAccessProperty` | Access property for L2 policy window (Normal, Streaming, Persisting) | b/memory-bound/l2-cache-control | core |
| `cudaClusterSchedulingPolicy` | Cluster scheduling (Default, Spread, LoadBalancing) | b/latency-bound/occupancy-tuning | related |
| `cudaMemcpyKind` | Memory copy direction (HostToHost, HostToDevice, DeviceToHost, DeviceToDevice, Default) | b/memory-bound/host-device-transfer | related |
| `cudaStreamCaptureMode` | Graph capture mode (Global, ThreadLocal, Relaxed) | b/latency-bound/cuda-graphs | related |
| `cudaMemPoolAttr` | Memory pool attributes (ReuseFollowEventDependencies, ReuseAllowOpportunistic, ReleaseThreshold, etc.) | b/memory-bound/host-device-transfer | related |
| `cudaGraphInstantiateFlags` | Graph instantiate flags (DeviceLaunch, UseNodePriority) | b/latency-bound/cuda-graphs | related |
| `cudaHostTaskSyncMode` | Host task sync mode control for cudaLaunchHostFunc_v2 | b/latency-bound/stream-concurrency | related |

---

## Summary

### Counts by Relevance

| Category | Count |
|----------|-------|
| **Total unique API functions documented** | ~300 (excluding data structures, enums, and type aliases) |
| **core** | ~95 |
| **related** | ~105 |
| **low-relevance** | ~100 |
| **Device-side intrinsics/built-ins** | ~40 |

### Counts by Knowledge Node (core + related APIs)

| Knowledge Node | core | related |
|----------------|------|---------|
| **optimization/memory/** | | |
| vectorized-access | 0 | 0 |
| shared-memory-cache | 14 | 2 |
| bank-conflict-avoidance | 4 | 0 |
| data-prefetch | 7 | 1 |
| coalescing | 2 | 2 |
| layout-transform | 1 | 1 |
| l2-cache-control | 11 | 5 |
| register-pressure | 2 | 1 |
| cache-load-hints | 12 | 4 |
| host-device-transfer | 11 | 28 |
| unified-memory | 6 | 4 |
| numa-binding | 2 | 4 |
| **optimization/compute/** | | |
| tensor-core | 2 | 0 |
| instruction-level-parallelism | 0 | 0 |
| fast-math | 0 | 0 |
| operator-fusion | 0 | 0 |
| compiler-hints | 1 | 0 |
| half-precision-math | 0 | 0 |
| warp-primitives | 10 | 3 |
| **optimization/latency/** | | |
| occupancy-tuning | 14 | 3 |
| warp-divergence | 1 | 3 |
| kernel-launch-overhead | 5 | 3 |
| stream-concurrency | 12 | 15 |
| cuda-graphs | 14 | ~55 |
| programmatic-dependent-launch | 4 | 0 |
| dynamic-parallelism | 3 | 0 |
| context-management | 1 | 3 |
| green-contexts | 10 | 6 |
| work-stealing | 1 | 0 |
| gpu-library-usage | 0 | 0 |
| **optimization/synchronization/** | | |
| barrier-optimization | 3 | 4 |
| atomic-reduction | 3 | 4 |
| cooperative-groups | 3 | 0 |
| thread-scopes | 4 | 1 |
| memory-sync-domains | 6 | 1 |
| **pattern/** | | |
| gemm | 2 | 0 |
| reduction | 2 | 0 |
| elementwise | 0 | 0 |
| attention | 0 | 0 |
| normalization | 0 | 0 |
| pooling | 0 | 0 |
| scan-cumulative | 0 | 0 |

### Notes

- **vectorized-access**, **instruction-level-parallelism**, **fast-math**, **operator-fusion**, **half-precision-math**: These are achieved through PTX/compiler-level techniques and math intrinsics (`__fadd_rn`, `__fmaf_rn`, `__half2float`, `__float2half`, etc.) rather than runtime API calls. See the separate `math-intrinsics-index.md` and `ptx-instruction-index.md` for coverage.

- **gpu-library-usage**: Covered by cuBLAS, cuDNN, cuFFT, and similar library APIs. See `cublas-api-index.md` for cuBLAS coverage.

- **pattern/** nodes (elementwise, attention, normalization, pooling, scan-cumulative): These are algorithm patterns that use combinations of the APIs above rather than having dedicated runtime APIs. The GEMM and reduction patterns are mapped because `wmma`/`mma_sync` and `__shfl_down_sync`/`__shfl_xor_sync` are directly targeted at those patterns.

- **Low-relevance categories breakdown**: ~35 graphics/display interop (GL, D3D9/10/11, VDPAU, EGL), ~15 IPC, ~15 error/version/profiling/logging, ~10 deprecated array copies, ~10 texture/surface management, ~15 device enumeration/query/lifecycle.
