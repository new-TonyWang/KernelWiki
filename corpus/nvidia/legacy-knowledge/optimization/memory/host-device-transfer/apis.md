# Host Device Transfer -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceEnablePeerAccess` | Runtime API | Enables direct access to peer device memory allocations |
| `cudaFreeAsync` | Runtime API | Frees memory asynchronously (stream-ordered) |
| `cudaHostAlloc` | Runtime API | Allocates page-locked (pinned) host memory with flags (mapped, portable, write-combined) |
| `cudaHostRegister` | Runtime API | Registers an existing host memory range for use by CUDA (page-locked, mapped, read-only) |
| `cudaMallocAsync` | Runtime API | Allocates memory asynchronously (stream-ordered) |
| `cudaMallocFromPoolAsync` | Runtime API | Allocates memory from a specified pool asynchronously |
| `cudaMallocHost` | Runtime API | Allocates page-locked (pinned) host memory |
| `cudaMemPoolCreate` | Runtime API | Creates a memory pool |
| `cudaMemcpy` | Runtime API | Copies data between host and device |
| `cudaMemcpy2DAsync` | Runtime API | Copies data between host and device (2D, async) |
| `cudaMemcpyAsync` | Runtime API | Copies data between host and device asynchronously |
| `cudaMemcpyBatchAsync` | Runtime API | Batch async memory copies |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceCanAccessPeer` | Runtime API | Queries if a device may directly access a peer device's memory |
| `cudaDeviceDisablePeerAccess` | Runtime API | Disables direct access to peer device memory |
| `cudaDeviceGetDefaultMemPool` | Runtime API | Returns the default mempool of a device |
| `cudaDeviceGetMemPool` | Runtime API | Gets the current mempool for a device |
| `cudaDeviceGetP2PAttribute` | Runtime API | Queries attributes of the link between two devices |
| `cudaDeviceSetMemPool` | Runtime API | Sets the current memory pool of a device |
| `cudaFree` | Runtime API | Frees device memory |
| `cudaFreeHost` | Runtime API | Frees page-locked host memory |
| `cudaHostGetDevicePointer` | Runtime API | Passes back device pointer of mapped host memory |
| `cudaHostUnregister` | Runtime API | Unregisters a memory range that was registered with cudaHostRegister |
| `cudaMalloc` | Runtime API | Allocate device memory |
| `cudaMemGetDefaultMemPool` | Runtime API | Returns the default memory pool of a device |
| `cudaMemGetMemPool` | Runtime API | Gets the memory pool for a device |
| `cudaMemPoolAttr` | Runtime API | Memory pool attributes (ReuseFollowEventDependencies, ReuseAllowOpportunistic, ReleaseThreshold, etc.) |
| `cudaMemPoolDestroy` | Runtime API | Destroys a memory pool |
| `cudaMemPoolGetAccess` | Runtime API | Returns the accessibility of a pool from a device |
| `cudaMemPoolGetAttribute` | Runtime API | Gets attributes of a memory pool (reuse policy, release threshold) |
| `cudaMemPoolProps` | Runtime API | Memory pool properties (allocation type, location, handle type) |
| `cudaMemPoolSetAccess` | Runtime API | Controls the pool's accessibility on the specified device |
| `cudaMemPoolSetAttribute` | Runtime API | Sets attributes of a memory pool (reuse policy, release threshold) |
| `cudaMemPoolTrimTo` | Runtime API | Releases memory back to the OS up to specified threshold |
| `cudaMemSetMemPool` | Runtime API | Sets the current memory pool for a device |
| `cudaMemcpy2D` | Runtime API | Copies data between host and device (2D, with pitch) |
| `cudaMemcpy3D` | Runtime API | Copies data between 3D objects |
| `cudaMemcpy3DAsync` | Runtime API | Copies data between 3D objects (async) |
| `cudaMemcpy3DBatchAsync` | Runtime API | Batch async 3D memory copies |
| `cudaMemcpy3DPeer` | Runtime API | Copies memory between devices |
| `cudaMemcpy3DPeerAsync` | Runtime API | Copies memory between devices asynchronously |
| `cudaMemcpy3DWithAttributesAsync` | Runtime API | 3D async copy with memory copy attributes (e.g., src/dst access hints) |
| `cudaMemcpyKind` | Runtime API | Memory copy direction (HostToHost, HostToDevice, DeviceToHost, DeviceToDevice, Default) |
| `cudaMemcpyPeer` | Runtime API | Copies memory between two devices |
| `cudaMemcpyPeerAsync` | Runtime API | Copies memory between two devices asynchronously |
| `cudaMemcpyWithAttributesAsync` | Runtime API | Async copy with memory copy attributes (src/dst access policy hints) |
| `cudaPointerGetAttributes` | Runtime API | Returns attributes about a specified pointer (type, device, hostPointer, devicePointer) |
