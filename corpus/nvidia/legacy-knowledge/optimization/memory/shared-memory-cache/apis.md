# Shared Memory Cache -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cp.async.bulk[.dst][.src]` | PTX ISA | Bulk async copy (TMA-style, large transfers) |
| `cp.async.ca.shared.global[.vec]` | PTX ISA | Async copy global->shared (bypasses registers) |
| `ld.shared::cluster[.vec][.type]` | PTX ISA | Distributed shared memory load across cluster |
| `ld.shared::cta[.vec][.type]` | PTX ISA | Shared memory load (CTA scope, explicit) |
| `ld.shared[.vec][.type]` | PTX ISA | Shared memory load |
| `st.shared::cluster[.vec][.type]` | PTX ISA | Distributed shared memory store across cluster |
| `st.shared[.vec][.type]` | PTX ISA | Shared memory store |
| `__pipeline_memcpy_async()` | Runtime API | Async copy from global to shared memory (hardware-accelerated) |
| `__shfl_sync()` | Runtime API | Warp shuffle (direct indexed) |
| `__syncthreads()` | Runtime API | Block-level barrier synchronization |
| `cuda::memcpy_async()` | Runtime API | C++ async memory copy (global to shared) |
| `cudaDeviceGetCacheConfig` | Runtime API | Returns the preferred cache configuration (L1 vs shared memory split) |
| `cudaDeviceGetSharedMemConfig` | Runtime API | Returns the shared memory configuration for the current device |
| `cudaDeviceSetCacheConfig` | Runtime API | Sets the preferred cache configuration (L1 vs shared memory) |
| `cudaFuncAttribute` | Runtime API | Function attribute identifiers (MaxDynamicSharedMemorySize, PreferredSharedMemoryCarveout, RequiredCluster*) |
| `cudaFuncCache` | Runtime API | L1 vs shared memory preference (PreferNone, PreferShared, PreferL1, PreferEqual) |
| `cudaFuncGetAttributes` | Runtime API | Find out attributes for a given function (maxThreadsPerBlock, numRegs, sharedSizeBytes, etc.) |
| `cudaFuncSetAttribute` | Runtime API | Set attributes for a function (max dynamic shared mem, cluster dims, shared mem carveout) |
| `cudaFuncSetCacheConfig` | Runtime API | Sets the preferred cache configuration (L1 vs shared memory) for a device function |
| `cudaOccupancyAvailableDynamicSMemPerBlock` | Runtime API | Returns max dynamic shared memory per block when launching N blocks per SM |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` | Runtime API | (C++ template) Suggest block size with variable shared memory per block |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` | Runtime API | (C++ template) Suggest block size with variable shared memory and flags |
| `cudaSharedCarveout` | Runtime API | Shared memory carveout values (Default, MaxShared, MaxL1) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cvta.space.size` | PTX ISA | Convert between generic and space-specific addresses |
| `getctarank` | PTX ISA | Get CTA rank from shared memory address in cluster |
| `mapa.space` | PTX ISA | Map shared memory address to another CTA in cluster |
| `st.bulk[.type]` | PTX ISA | Bulk store to shared memory (zeros or initial values) |
| `cudaKernelSetAttributeForDevice` | Runtime API | Sets attribute for a kernel on a specified device |
