# Programmatic Dependent Launch -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `griddepcontrol.launch_dependents` | PTX ISA | Signal dependent grids may launch (programmatic dependent launch) |
| `griddepcontrol.wait` | PTX ISA | Wait for prerequisite grid completion |
| `cudaGridDependencySynchronize` | Runtime API | (__device__) Programmatic grid dependency synchronization; blocks until direct grid dependencies complete |
| `cudaLaunchAttributeID` | Runtime API | Launch attribute identifiers (AccessPolicyWindow, CooperativeLaunch, ProgrammaticStreamSerialization, ProgrammaticEvent, ClusterDim, MemSyncDomain, SharedMemoryMode, etc.) |
| `cudaLaunchAttributeValue` | Runtime API | Union holding cluster dims, programmatic event, mem sync domain, access policy, shared memory mode |
| `cudaLaunchKernelExC` | Runtime API | Launches a kernel with launch-time configuration (cudaLaunchConfig_t: cluster dims, access policy, mem sync domain, programmatic events) |
| `cudaTriggerProgrammaticLaunchCompletion` | Runtime API | (__device__) Triggers programmatic launch completion for dependent launch |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `clusterlaunchcontrol.query_cancel` | PTX ISA | Query if a cluster launch was cancelled |
| `clusterlaunchcontrol.try_cancel` | PTX ISA | Try to cancel a pending cluster launch |
