# Reduction Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| Warp-Level Reduction with Shuffle | Tree or butterfly reduction in 5 steps | Zero shared memory; __shfl_down_sync / __shfl_xor_sync |
| Block-Level Reduction | Warp results combined via shared memory | Two-stage: warp reduce then inter-warp SMEM reduce |
| Grid-Level Reduction | Block results combined via atomics or second pass | atomicAdd for simple cases; two-kernel for large grids |
| Vectorized Loading | float4 loads to maximize bandwidth during reduction | 128-bit transactions feed partial accumulator per thread |
| Hardware Warp Reduction (SM80+) | PTX redux.sync.add.s32 single-instruction warp reduce | Faster than shuffle tree for 32-bit integer operations |
| Multi-Dimensional Reduction | Reduce one dimension while preserving others | Thread mapping aligned to reduction axis for coalescing |
| Segmented Reduction | Independent reductions over variable-length segments | Warp-per-segment or block-per-segment depending on size |
