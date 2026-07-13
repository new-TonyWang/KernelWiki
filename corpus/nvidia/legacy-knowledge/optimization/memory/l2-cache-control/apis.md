# L2 Cache Control -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `.L2::cache_hint` | PTX ISA | L2 cache policy hint (used with createpolicy) |
| `.cg (cache-global)` | PTX ISA | Cache in L2 only, bypass L1 |
| `.cs (cache-streaming)` | PTX ISA | Streaming access, evict-first policy |
| `cp.async.bulk.prefetch` | PTX ISA | Bulk async prefetch to L2 cache |
| `cp.async.bulk.prefetch.tensor` | PTX ISA | TMA tensor prefetch to L2 |
| `createpolicy` | PTX ISA | Create L2 cache access policy |
| `evict_normal / evict_first / evict_last / no_allocate` | PTX ISA | Cache eviction priority hints |
| `cudaAccessPolicyWindow` | Runtime API | Specifies L2 access policy window (base_ptr, num_bytes, hit/miss ratio, persistent/streaming) |
| `cudaAccessProperty` | Runtime API | Access property for L2 policy window (Normal, Streaming, Persisting) |
| `cudaCtxResetPersistingL2Cache` | Runtime API | Resets all persisting lines in L2 cache to normal status |
| `cudaDeviceGetLimit` | Runtime API | Return resource limits (stack size, L2 cache, malloc heap, etc.) |
| `cudaDeviceSetLimit` | Runtime API | Set resource limits (L2 persistent cache size, stack size, L2 fetch granularity) |
| `cudaLaunchAttribute` | Runtime API | Launch attribute (id + value) for cudaLaunchKernelExC |
| `cudaLaunchAttributeID` | Runtime API | Launch attribute identifiers (AccessPolicyWindow, CooperativeLaunch, ProgrammaticStreamSerialization, ProgrammaticEvent, ClusterDim, MemSyncDomain, SharedMemoryMode, etc.) |
| `cudaLaunchAttributeValue` | Runtime API | Union holding cluster dims, programmatic event, mem sync domain, access policy, shared memory mode |
| `cudaLaunchConfig_t` | Runtime API | Launch configuration (gridDim, blockDim, dynamicSmemBytes, stream, attrs for L2 policy, cluster, mem sync domain) |
| `cudaLaunchKernelEx` | Runtime API | (C++ template) Launch kernel with extended config (cluster, L2 policy, mem sync domain) |
| `cudaLaunchKernelExC` | Runtime API | Launches a kernel with launch-time configuration (cudaLaunchConfig_t: cluster dims, access policy, mem sync domain, programmatic events) |
| `cudaLimit` | Runtime API | Resource limits (StackSize, PrintfFifoSize, MallocHeapSize, DevRuntimeSyncDepth, MaxL2FetchGranularity, PersistingL2CacheSize) |
| `cudaStreamGetAttribute` | Runtime API | Queries stream attribute (including cudaAccessPolicyWindow for L2 cache) |
| `cudaStreamSetAttribute` | Runtime API | Sets stream attribute (including cudaAccessPolicyWindow for L2 persistence) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `applypriority.global.L2` | PTX ISA | Apply eviction priority to L2 cache range |
| `discard.global.L2` | PTX ISA | Discard L2 cache line (no write-back) |
| `cudaGraphKernelNodeGetAttribute` | Runtime API | Queries node attribute (e.g., access policy window) |
| `cudaGraphKernelNodeSetAttribute` | Runtime API | Sets node attribute (e.g., access policy window) |
| `cudaStreamCopyAttributes` | Runtime API | Copies attributes (including L2 access policy) from source to destination stream |
