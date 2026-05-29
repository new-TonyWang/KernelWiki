# Kernel Launch Overhead -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaEventCreateWithFlags` | Runtime API | Creates an event with the specified flags (timing disabled, blocking sync, interprocess) |
| `cudaGraphLaunch` | Runtime API | Launches an executable graph in a stream |
| `cudaLaunchConfig_t` | Runtime API | Launch configuration (gridDim, blockDim, dynamicSmemBytes, stream, attrs for L2 policy, cluster, mem sync domain) |
| `cudaLaunchKernel` | Runtime API | Launches a device function (standard kernel launch) |
| `cudaLaunchKernelEx` | Runtime API | (C++ template) Launch kernel with extended config (cluster, L2 policy, mem sync domain) |
| `cudaLaunchKernelExC` | Runtime API | Launches a kernel with launch-time configuration (cudaLaunchConfig_t: cluster dims, access policy, mem sync domain, programmatic events) |
| `cudaMemPoolCreate` | Runtime API | Creates a memory pool |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaEventElapsedTime` | Runtime API | Computes elapsed time between events |
| `cudaGetKernel` | Runtime API | Returns a cudaKernel_t handle for a __global__ function |
| `cudaLibraryGetKernel` | Runtime API | Returns a kernel handle from a library |
