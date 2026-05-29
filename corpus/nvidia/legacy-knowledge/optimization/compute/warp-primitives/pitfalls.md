# Warp Primitives -- Pitfalls

## P1: Incorrect Mask Causing Hang or Undefined Behavior
**Symptom**: Kernel hangs indefinitely or produces undefined results when using warp `__sync` intrinsics.
**Detection**: Kernel timeout or infinite loop. Difficult to debug with standard tools.
**Fix**: Ensure every calling thread has its own bit set in the mask. Non-participating threads must have their bit cleared. All threads specified in the mask must reach the call with the same mask value. In divergent code, use disjoint masks for different branches.
**Source**: Programming Guide, Section 5.4.6.6 (Warp __sync Intrinsic Constraints)

## P2: Using __activemask() to Determine Branch Participation
**Symptom**: `__activemask()` returns a non-deterministic result that varies between runs because the compiler can reorder instructions around it.
**Detection**: Intermittent incorrect results when using `__activemask()` inside conditional code.
**Fix**: `__activemask()` provides only an instantaneous snapshot and cannot be used to determine which threads took a branch. For branch-dependent cooperation, use explicit masks based on the branch condition, or use `__ballot_sync(0xFFFFFFFF, predicate)` which synchronizes first.
**Source**: Programming Guide, Section 5.4.6.1 (Warp Active Mask)

## P3: Warp Shuffle on Inactive Source Lane
**Symptom**: Reading from a lane that is inactive (e.g., exited or not participating) returns undefined values.
**Detection**: Intermittent garbage values in shuffled results.
**Fix**: Ensure the source lane is active in the mask. For `__shfl_down_sync`, upper lanes (where source would be out-of-bounds) return their own value, which is defined. For `__shfl_sync` with explicit `srcLane`, ensure that lane is participating.
**Source**: Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions) -- "If the target thread is inactive, the retrieved value is undefined"

## P4: Width Parameter Not a Power of Two
**Symptom**: `__shfl_sync` with `width` not a power of two produces undefined results.
**Detection**: Silent incorrect results; no runtime error.
**Fix**: The `width` parameter must be 1, 2, 4, 8, 16, or 32. Any other value is undefined. If you need a non-power-of-two partition, implement it manually with explicit lane ID arithmetic and predication.
**Source**: Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions)

## P5: Shuffle Does Not Provide Memory Ordering
**Symptom**: Shuffled value from another thread is stale because the other thread's prior store has not yet been visible.
**Detection**: Intermittent incorrect results that depend on execution order.
**Fix**: Warp shuffle intrinsics do NOT imply a memory barrier. If the shuffled value was recently stored to shared or global memory by the source thread, insert a `__syncwarp()` or appropriate fence BEFORE the shuffle to ensure visibility.
**Source**: Programming Guide, Section 5.4.6.5 (Warp Shuffle Functions) -- "These intrinsics do not imply a memory barrier"

## P6: Hardware Reduce Only Supports Integer Types
**Symptom**: Compilation error when using `__reduce_add_sync` with `float` argument.
**Detection**: Compile-time error.
**Fix**: `__reduce_add_sync` / `__reduce_min_sync` / `__reduce_max_sync` accept only unsigned or signed integer types. For floating-point reduction, use the 5-step XOR shuffle tree pattern instead. There is no hardware float reduction instruction as of sm_90.
**Source**: Programming Guide, Section 5.4.6.4 (Warp Reduce Functions)

## P7: Warp shuffle reduction skills cannot be meaningfully verified at tiny input sizes; launch overhead and measurement noise dwarf any algorithmic improvement, making the skill appear to have no effect (discovered in verification)

**Symptom**: Warp shuffle reduction skills cannot be meaningfully verified at tiny input sizes; launch overhead and measurement noise dwarf any algorithmic improvement, making the skill appear to have no effect.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Roofline efficiency percentage dropped from 46 (discovered in verification)

**Symptom**: Roofline efficiency percentage dropped from 46.9% to 28.7% despite the kernel being faster — shorter kernels show lower sustained throughput percentages due to launch overhead dominating, so percentage-of-peak metrics can be misleading for very fast kernels. Long scoreboard stalls also rose from 29.7% to 43.5%, suggesting the shuffle-heavy code is more sensitive to register dependency latency.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: Roofline efficiency actually dropped (45 (discovered in verification)

**Symptom**: Roofline efficiency actually dropped (45.5%→24.5%) despite faster execution — the kernel became so short that launch overhead and measurement noise dominate, making NCU throughput percentages misleading for micro-kernels under ~4μs.
**Source**: Level 3 sandbox verification (2026-04-06)

## P10: Warp shuffle replacing shared memory can increase memory stall dominance — fewer instructions means fewer opportunities to hide DRAM latency, potentially worsening the long-scoreboard stall ratio even as total work decreases (discovered in verification)

**Symptom**: Warp shuffle replacing shared memory can increase memory stall dominance — fewer instructions means fewer opportunities to hide DRAM latency, potentially worsening the long-scoreboard stall ratio even as total work decreases.
**Source**: Level 3 sandbox verification (2026-04-06)

## P11: Both rounds show the kernel is severely underutilized (efficiency <1%, 0 DRAM writes, only 124 instructions executed) — the test problem is too small to stress the GPU, so the 4x speedup is mostly from eliminating atomic contention overhead rather than demonstrating throughput scaling (discovered in verification)

**Symptom**: Both rounds show the kernel is severely underutilized (efficiency <1%, 0 DRAM writes, only 124 instructions executed) — the test problem is too small to stress the GPU, so the 4x speedup is mostly from eliminating atomic contention overhead rather than demonstrating throughput scaling.
**Source**: Level 3 sandbox verification (2026-04-06)

## P12: Short scoreboard stalls dropped (17 (discovered in verification)

**Symptom**: Short scoreboard stalls dropped (17.8%→13.2%) confirming fewer dependent instruction chains, yet long scoreboard stalls slightly increased (45.6%→47.4%), suggesting the faster reduction exposes more time waiting on DRAM — the kernel finishes compute sooner but still waits on memory.
**Source**: Level 3 sandbox verification (2026-04-07)
