# Pooling -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__shfl_down_sync()` | Runtime API | Warp shuffle down (used in reductions for max/avg pooling) |
| `fmaxf(x,y)` | Math Intrinsic | Maximum (NaN-safe); used for max pooling comparison |
| `fminf(x,y)` | Math Intrinsic | Minimum (NaN-safe); used for min pooling comparison |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `ld.global[.vec][.type]` | PTX ISA | Global memory load; .v2/.v4 vector variants for vectorized input tile loads |
| `__syncthreads()` | Runtime API | Block-level barrier synchronization for shared memory tiling |
