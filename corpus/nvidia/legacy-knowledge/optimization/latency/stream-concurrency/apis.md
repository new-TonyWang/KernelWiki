# Stream Concurrency -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaEventCreateWithFlags` | Runtime API | Creates an event with the specified flags (timing disabled, blocking sync, interprocess) |
| `cudaExecutionCtxStreamCreate` | Runtime API | Creates a stream on the execution context |
| `cudaFreeAsync` | Runtime API | Frees memory asynchronously (stream-ordered) |
| `cudaLaunchKernel` | Runtime API | Launches a device function (standard kernel launch) |
| `cudaMallocAsync` | Runtime API | Allocates memory asynchronously (stream-ordered) |
| `cudaMallocFromPoolAsync` | Runtime API | Allocates memory from a specified pool asynchronously |
| `cudaMemcpy2DAsync` | Runtime API | Copies data between host and device (2D, async) |
| `cudaMemcpyAsync` | Runtime API | Copies data between host and device asynchronously |
| `cudaMemcpyBatchAsync` | Runtime API | Batch async memory copies |
| `cudaStreamCreate` | Runtime API | Create an asynchronous stream |
| `cudaStreamCreateWithFlags` | Runtime API | Create an asynchronous stream with flags (e.g., cudaStreamNonBlocking) |
| `cudaStreamCreateWithPriority` | Runtime API | Create an asynchronous stream with specified priority |
| `cudaStreamGetAttribute` | Runtime API | Queries stream attribute (including cudaAccessPolicyWindow for L2 cache) |
| `cudaStreamWaitEvent` | Runtime API | Make a compute stream wait on an event |
| `cublas<t>gemmBatched()` | cuBLAS | Batch GEMM via array of pointers |
| `cublasSetStream()` | cuBLAS | Associate CUDA stream with handle |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceGetStreamPriorityRange` | Runtime API | Returns least and greatest stream priorities |
| `cudaEventCreate` | Runtime API | Creates an event object |
| `cudaEventDestroy` | Runtime API | Destroys an event object |
| `cudaEventQuery` | Runtime API | Queries an event's status |
| `cudaEventRecord` | Runtime API | Records an event |
| `cudaEventRecordWithFlags` | Runtime API | Records an event with flags (default or external) |
| `cudaExecutionCtxRecordEvent` | Runtime API | Records an event on the execution context |
| `cudaHostTaskSyncMode` | Runtime API | Host task sync mode control for cudaLaunchHostFunc_v2 |
| `cudaLaunchHostFunc` | Runtime API | Enqueues a host function call in a stream |
| `cudaLaunchHostFunc_v2` | Runtime API | Enqueues a host function call with sync mode control |
| `cudaMemcpy3DAsync` | Runtime API | Copies data between 3D objects (async) |
| `cudaMemcpy3DBatchAsync` | Runtime API | Batch async 3D memory copies |
| `cudaMemcpy3DPeerAsync` | Runtime API | Copies memory between devices asynchronously |
| `cudaMemcpyPeerAsync` | Runtime API | Copies memory between two devices asynchronously |
| `cudaMemsetAsync` | Runtime API | Initializes or sets device memory asynchronously |
| `cudaStreamAddCallback` | Runtime API | Add a callback to a compute stream (deprecated in favor of cudaLaunchHostFunc) |
| `cudaStreamDestroy` | Runtime API | Destroys and cleans up an asynchronous stream |
| `cudaStreamGetFlags` | Runtime API | Query the flags of a stream |
| `cudaStreamGetPriority` | Runtime API | Query the priority of a stream |
| `cudaStreamQuery` | Runtime API | Queries an asynchronous stream for completion status |
| `cublasGetSmCountTarget()` | cuBLAS | Query SM count target |
| `cublasGetStream()` | cuBLAS | Get associated CUDA stream |
| `cublasSetSmCountTarget()` | cuBLAS | Limit SM count for cuBLAS operations |
