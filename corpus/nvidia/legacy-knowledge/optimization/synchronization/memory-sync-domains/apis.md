# Memory Sync Domains -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `fence.proxy.async[.space]` | PTX ISA | Proxy fence for async operations (.global, .shared::cta, .shared::cluster) |
| `fence[.sem].scope` | PTX ISA | Thread fence (.sc/.acq_rel/.acquire/.release at .cta/.cluster/.gpu/.sys) |
| `membar.cta / .gl / .sys` | PTX ISA | Memory barrier at CTA/global/system level (old-style) |
| `__threadfence()` | Runtime API | Memory fence (device scope) |
| `__threadfence_system()` | Runtime API | Memory fence (system scope, including host) |
| `cudaLaunchAttribute` | Runtime API | Launch attribute (id + value) for cudaLaunchKernelExC |
| `cudaLaunchAttributeID` | Runtime API | Launch attribute identifiers (AccessPolicyWindow, CooperativeLaunch, ProgrammaticStreamSerialization, ProgrammaticEvent, ClusterDim, MemSyncDomain, SharedMemoryMode, etc.) |
| `cudaLaunchAttributeValue` | Runtime API | Union holding cluster dims, programmatic event, mem sync domain, access policy, shared memory mode |
| `cudaLaunchKernelEx` | Runtime API | (C++ template) Launch kernel with extended config (cluster, L2 policy, mem sync domain) |
| `cudaLaunchKernelExC` | Runtime API | Launches a kernel with launch-time configuration (cudaLaunchConfig_t: cluster dims, access policy, mem sync domain, programmatic events) |
| `cudaLaunchMemSyncDomain` | Runtime API | Memory sync domain identifiers (Default, Remote) |
| `cudaLaunchMemSyncDomainMap` | Runtime API | Maps default/remote memory sync domains |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `fence.proxy.alias` | PTX ISA | Proxy fence for aliased memory accesses |
| `fence.proxy.tensormap::generic` | PTX ISA | Fence for tensormap updates |
| `tensormap.cp_fenceproxy` | PTX ISA | Fence for tensormap copy operations between proxies |
| `cudaDeviceFlushGPUDirectRDMAWrites` | Runtime API | Blocks until remote writes are visible to the specified scope |
| `cudaEventRecordWithFlags` | Runtime API | Records an event with flags (default or external) |
