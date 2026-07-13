# Green Contexts -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDevResource` | Runtime API | Device resource (SM or workqueue) for green context partitioning |
| `cudaDevResourceGenerateDesc` | Runtime API | Generates a resource descriptor from specified resources (SM, workqueue) for green context creation |
| `cudaDevSmResource` | Runtime API | SM resource with count and co-scheduling alignment |
| `cudaDevSmResourceGroupParams` | Runtime API | SM resource group params for splitting (smCount, coscheduledSmCount) |
| `cudaDevSmResourceSplit` | Runtime API | Splits SM resources into structured groups (with co-scheduling control) |
| `cudaDevSmResourceSplitByCount` | Runtime API | Splits SM resources by count with default granularity |
| `cudaDevWorkqueueConfigResource` | Runtime API | Workqueue configuration (concurrency limit, sharing scope) |
| `cudaDeviceGetDevResource` | Runtime API | Gets device resources (SM count, alignment, workqueue config) |
| `cudaDeviceGetExecutionCtx` | Runtime API | Returns the execution context for a device's primary context |
| `cudaExecutionCtxStreamCreate` | Runtime API | Creates a stream on the execution context |
| `cudaGreenCtxCreate` | Runtime API | Creates a green context with specified resources |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaExecutionCtxDestroy` | Runtime API | Destroys an execution context |
| `cudaExecutionCtxGetDevResource` | Runtime API | Gets device resource from an execution context |
| `cudaExecutionCtxGetDevice` | Runtime API | Gets the device associated with an execution context |
| `cudaExecutionCtxRecordEvent` | Runtime API | Records an event on the execution context |
| `cudaExecutionCtxSynchronize` | Runtime API | Synchronizes the execution context |
| `cudaExecutionCtxWaitEvent` | Runtime API | Makes the execution context wait on an event |
| `cudaStreamGetDevResource` | Runtime API | Gets device resource from a stream |
