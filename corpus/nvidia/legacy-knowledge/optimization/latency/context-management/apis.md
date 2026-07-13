# Context Management -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceGetExecutionCtx` | Runtime API | Returns the execution context for a device's primary context |
| `cudaInitDevice` | Runtime API | Explicit device initialization (avoids lazy-init latency on first API call) |
| `cudaSetDevice` | Runtime API | Set device to be used for GPU executions (context configuration) |
| `cudaSetDeviceFlags` | Runtime API | Sets flags for the current device (scheduling mode, map host memory, etc.) |
