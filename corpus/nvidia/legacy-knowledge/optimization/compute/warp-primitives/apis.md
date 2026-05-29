# Warp Primitives -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `bar.warp.sync membermask` | PTX ISA | Warp-level barrier sync (subset of threads via bitmask) |
| `redux.sync.op.type` | PTX ISA | Warp-level reduction (.add/.min/.max/.and/.or/.xor) |
| `shfl.sync[.mode].b32` | PTX ISA | Warp shuffle: exchange registers between lanes (.up/.down/.bfly/.idx) |
| `vote.sync.{all/any/uni/ballot}.b32` | PTX ISA | Warp vote with sync (all/any-true, uniform, ballot mask) |
| `__ballot_sync()` | Runtime API | Warp vote (returns bitmask of predicate across warp) |
| `__shfl_down_sync()` | Runtime API | Warp shuffle down (used in reductions) |
| `__shfl_sync()` | Runtime API | Warp shuffle (direct indexed) |
| `__shfl_up_sync()` | Runtime API | Warp shuffle up |
| `__shfl_xor_sync()` | Runtime API | Warp shuffle XOR (butterfly reduction) |
| `__syncwarp(mask)` | Runtime API | Warp-level synchronization |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `activemask.b32` | PTX ISA | Get bitmask of active threads in warp |
| `elect.sync` | PTX ISA | Elect a single leader thread from active mask |
| `match.sync.{any/all}.b32/b64` | PTX ISA | Warp match: find threads with matching values |
| `__activemask()` | Runtime API | Returns mask of active threads in the warp |
| `__all_sync()` | Runtime API | Warp vote (all predicates true) |
| `__any_sync()` | Runtime API | Warp vote (any predicate true) |
| `__match_all_sync()` | Runtime API | Warp match (checks if all threads have same value) |
| `__match_any_sync()` | Runtime API | Warp match (finds threads with matching value) |
