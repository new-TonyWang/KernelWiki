# Scan Cumulative -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__shfl_up_sync()` | Runtime API | Warp shuffle up (inclusive scan within warp) |
| `__shfl_xor_sync()` | Runtime API | Warp shuffle XOR (butterfly reduction pattern) |
| `redux.sync.op.type` | PTX ISA | Warp-level hardware reduction (.add/.min/.max/.and/.or/.xor) |
| `__syncthreads()` | Runtime API | Block-level barrier synchronization (inter-warp sync for block-level scan) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `atomicAdd()` | Runtime API | Atomic addition (inter-block communication for decoupled lookback) |
