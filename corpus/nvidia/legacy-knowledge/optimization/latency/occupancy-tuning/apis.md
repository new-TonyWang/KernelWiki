# Occupancy Tuning -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `setmaxnreg.{inc/dec}.sync.aligned.u32` | PTX ISA | Dynamically adjust per-warp register count (warpgroup-level) |
| `cudaDeviceProp` | Runtime API | Device properties (sharedMemPerBlock, regsPerBlock, warpSize, maxThreadsPerBlock, etc.) |
| `cudaFuncAttribute` | Runtime API | Function attribute identifiers (MaxDynamicSharedMemorySize, PreferredSharedMemoryCarveout, RequiredCluster*) |
| `cudaFuncAttributes` | Runtime API | Function attributes (numRegs, sharedSizeBytes, maxThreadsPerBlock, etc.) |
| `cudaFuncGetAttributes` | Runtime API | Find out attributes for a given function (maxThreadsPerBlock, numRegs, sharedSizeBytes, etc.) |
| `cudaFuncSetAttribute` | Runtime API | Set attributes for a function (max dynamic shared mem, cluster dims, shared mem carveout) |
| `cudaLaunchCooperativeKernel` | Runtime API | Launches a kernel where thread blocks can cooperate and synchronize across grid |
| `cudaOccupancyAvailableDynamicSMemPerBlock` | Runtime API | Returns max dynamic shared memory per block when launching N blocks per SM |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Runtime API | Returns max active blocks per SM for a function |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | Runtime API | Returns max active blocks per SM with flags (caching override control) |
| `cudaOccupancyMaxActiveClusters` | Runtime API | Returns max clusters that could co-exist on the device |
| `cudaOccupancyMaxPotentialBlockSize` | Runtime API | (C++ template) Suggest block size to achieve maximum occupancy |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem` | Runtime API | (C++ template) Suggest block size with variable shared memory per block |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` | Runtime API | (C++ template) Suggest block size with variable shared memory and flags |
| `cudaOccupancyMaxPotentialBlockSizeWithFlags` | Runtime API | (C++ template) Suggest block size with flags for caching behavior |
| `cudaOccupancyMaxPotentialClusterSize` | Runtime API | Returns max cluster size for a kernel function |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `nanosleep.u32` | PTX ISA | Suspend thread for ~N nanoseconds (backoff/polling) |
| `cudaClusterSchedulingPolicy` | Runtime API | Cluster scheduling (Default, Spread, LoadBalancing) |
| `cudaDeviceGetAttribute` | Runtime API | Returns information about the device (SM count, shared mem size, warp size, etc.) |
| `cudaGetDeviceProperties` | Runtime API | Returns properties for the selected device (SM count, clock, memory, etc.) |
| `cudaKernelSetAttributeForDevice` | Runtime API | Sets attribute for a kernel on a specified device |
