# Register Pressure -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `setmaxnreg.{inc/dec}.sync.aligned.u32` | PTX ISA | Dynamically adjust per-warp register count (warpgroup-level) |
| `cudaDeviceGetLimit` | Runtime API | Return resource limits (stack size, L2 cache, malloc heap, etc.) |
| `cudaFuncAttributes` | Runtime API | Function attributes (numRegs, sharedSizeBytes, maxThreadsPerBlock, etc.) |
| `cudaFuncGetAttributes` | Runtime API | Find out attributes for a given function (maxThreadsPerBlock, numRegs, sharedSizeBytes, etc.) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `ld.local[.vec][.type]` | PTX ISA | Local (thread-private) memory load |
| `st.local[.vec][.type]` | PTX ISA | Local memory store |
