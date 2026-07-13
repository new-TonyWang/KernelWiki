# Dynamic Parallelism -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceSetLimit` | Runtime API | Set resource limits (L2 persistent cache size, stack size, L2 fetch granularity) |
| `cudaGetParameterBuffer` | Runtime API | (__device__) Obtains a parameter buffer for dynamic parallelism kernel launch |
| `cudaLaunchDevice` | Runtime API | (__device__) Launches a kernel from device code (dynamic parallelism) |
| `cudaLimit` | Runtime API | Resource limits (StackSize, PrintfFifoSize, MallocHeapSize, DevRuntimeSyncDepth, MaxL2FetchGranularity, PersistingL2CacheSize) |
