# Warp Divergence -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `@{!}p` | PTX ISA | Predicated execution (all instructions) |
| `setp.CmpOp[.BoolOp][.ftz].type` | PTX ISA | Set predicate from comparison; core tool for predication |
| `selp.type` | PTX ISA | Select based on predicate (branchless conditional move) |
| `vote.sync.{all/any/uni/ballot}.b32` | PTX ISA | Warp vote with sync (all/any-true, uniform, ballot mask) |
| `__ballot_sync()` | Runtime API | Warp vote (returns bitmask of predicate across warp) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `activemask.b32` | PTX ISA | Get bitmask of active threads in warp |
| `bra[.uni]` | PTX ISA | Branch (conditional/unconditional; .uni = uniform) |
| `exit` | PTX ISA | Terminate thread |
| `__activemask()` | Runtime API | Returns mask of active threads in the warp |
| `__all_sync()` | Runtime API | Warp vote (all predicates true) |
| `__any_sync()` | Runtime API | Warp vote (any predicate true) |
| `__nanosleep()` | Runtime API | Put thread to sleep for approximately ns nanoseconds |
