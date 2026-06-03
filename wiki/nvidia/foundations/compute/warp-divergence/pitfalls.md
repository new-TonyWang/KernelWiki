---
title: Warp Divergence — Pitfalls
status: draft
id: pitfall-warp-divergence
type: pitfall
vendor: nvidia
---
# Warp Divergence — Pitfalls

## P1. Implicit warp-synchronous execution on Volta+ (correctness)

**Symptom**: Warp shuffle or shared-memory exchange produces incorrect results on CC 7.0+ but works on CC 6.x and earlier.

**Root cause**: Independent Thread Scheduling (CC 7.0+) allows diverged lanes to stay diverged after the conditional block ends. Any subsequent code that assumes the warp is "back in lockstep" may operate on the wrong set of lanes.

**Fix**: Always use `_sync` variants of warp intrinsics with explicit masks (`0xFFFFFFFFu` for full warp, `__ballot_sync` for data-dependent masks). Insert `__syncwarp()` after any divergent region followed by warp-synchronous code.

```cuda
// Bad on Volta+: assumes implicit reconvergence
if (laneId < 16) { smem[laneId] = val; }
if (laneId >= 16) { result = smem[laneId - 16]; }       // may see stale data

// Good: explicit __syncwarp() + _sync intrinsics
if (laneId < 16) { smem[laneId] = val; }
__syncwarp();
if (laneId >= 16) { result = smem[laneId - 16]; }

// Also good: use warp shuffle with _sync
val = __shfl_xor_sync(0xFFFFFFFFu, val, 16);
```

**Source**: CUDA C++ Programming Guide §3.2.2.1.1 (L3427-L3436). This is a **correctness pitfall**, not a performance one — failure mode is silent data corruption.

## P2. Branching on `threadIdx.x` with sub-warp granularity

**Symptom**: Every warp in the block serializes its two branch bodies. NCU shows `smsp__thread_inst_executed_per_inst_executed.ratio` dropping below 32 (towards 16 for a 50/50 split).

**Root cause**: Condition depends on `threadIdx.x` directly (or `threadIdx.x % k` for some `k < 32`). Every warp straddles the boundary, so every warp diverges.

**Fix**: Restructure to align the condition on warp boundaries — typically by expressing it in terms of `threadIdx.x / 32` (or `>> 5`).

```cuda
// Bad: 50/50 split within each warp
if (threadIdx.x % 2 == 0) { /* A */ } else { /* B */ }

// Good: every warp picks one side
int warpId = threadIdx.x >> 5;
if ((warpId & 1) == 0) { /* A */ } else { /* B */ }
```

**Source**: BP Guide §12.1.2 (L1396-L1400).

## P3. Early return in only some lanes of a warp

**Symptom**: Adding an early-return guard (`if (i >= N) return;`) does not actually shrink the warp's runtime — the warp remains active until the last lane finishes.

**Root cause**: A per-lane return does not "release" the warp; the warp's remaining lanes continue executing with the returned lanes masked off. The warp occupies an SM slot until *all* its lanes have exited.

**Fix**: Structure the kernel so entire warps exit together (or the warp leader issues an early-exit vote). Use `__all_sync` to check if the whole warp has no work.

```cuda
// Ineffective: individual lanes return, warp stays resident
if (tid >= N) return;
process(tid);

// Better: warp-wide early exit
bool done = (tid >= N);
if (__all_sync(0xFFFFFFFFu, done)) return;   // whole warp exits
if (!done) process(tid);                      // surviving lanes work
```

**Source**: CUDA C++ Programming Guide §3.2.2.1 (L3427-L3436; "threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp").

## P4. Using `__activemask()` as a synchronization primitive

**Symptom**: Subtle correctness bugs on Volta+; warp intrinsics called with the `__activemask()` result sometimes include the wrong lanes.

**Root cause**: `__activemask()` reports which lanes are currently active, but makes **no guarantee** about what *will be* active when the next warp-synchronous operation executes. It is an **introspection** primitive, not a synchronization primitive.

**Fix**: Use explicit masks whenever possible. If the mask must be data-dependent, compute it with `__ballot_sync` (which includes a convergence point) rather than `__activemask`.

```cuda
// Bad: using activemask as a sync primitive
unsigned mask = __activemask();
val = __shfl_down_sync(mask, val, 1);     // may include wrong lanes

// Good: ballot has a sync, so the mask is valid at the next op
unsigned mask = __ballot_sync(0xFFFFFFFFu, isActive);
if (isActive) {
    val = __shfl_down_sync(mask, val, 1);  // mask holds all through this op
}
```

**Source**: CUDA C++ Programming Guide §5.4.6.1.

## P5. Nested divergent branches causing exponential serialization

**Symptom**: NCU reports warp efficiency well below 50 %; multiple divergence points stack, each roughly halving active lanes.

**Root cause**: Each nested divergent `if/else` multiplies the serialization factor. Three levels of 2-way divergence can serialize to 8× (25 % SIMT efficiency) even if each individual split is 50/50.

**Fix**: Flatten nested branches into a single multi-way dispatch (lookup table / switch compiled to jump table), or use predication for the innermost conditions.

```cuda
// Bad: 3 levels of nesting, up to 8x serialization
if (condA) {
    if (condB) {
        if (condC) { do_ABC(); } else { do_ABnC(); }
    }
}

// Flatten via lookup table (branchless)
int path = (condA << 2) | (condB << 1) | condC;
result = lookup_table[path];

// Or flatten via chained predication (works for short bodies)
float r = condA ? (condB ? (condC ? vABC : vABnC) : 0.f) : 0.f;
```

**Source**: BP Guide §12.1.2 (L1396-L1400).

## P6. Warp-boundary alignment destroying memory coalescing (measured on legacy)

**Symptom**: After rewriting a branch to align on warp boundaries (pitfall P2 fix), wall-clock regresses. NCU shows coalescing metrics dropping: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` per warp rises from 4 to 16.

**Root cause**: The two branch bodies access interleaved addresses. When the branch was per-lane, adjacent lanes within a warp hit adjacent memory addresses (coalesced). After warp-alignment, lanes in each warp hit *strided* addresses because they're all executing the same branch body against dispersed data indices.

**Fix**: Before committing to warp-alignment, check memory coalescing. If the two bodies access interleaved addresses, keep the per-lane branch and let predication handle it (S2), or restructure the data layout so the warp-aligned access pattern remains stride-1.

**Measured (legacy L3 sandbox, 2026-04-06)**: warp-aligning a gather-shaped kernel destroyed coalescing and regressed wall-clock by ~15 %. Scheduled for H200 re-measurement in a follow-up probe.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-06). Retained as measured-on-legacy; the principle holds but the exact number needs H200 re-measurement.

## P7. Warp-vote early-exit shifts bottleneck from compute to memory latency

**Symptom**: After adding `if (!__any_sync(...)) return;` the kernel's SM throughput drops from 90.5 % to 44.8 %, and `stall_long_scoreboard` rises from 0.11 % to 32.1 % — making it *look* worse in NCU section view.

**Root cause**: The early exit dramatically reduces total work, so the surviving warps are relatively more memory-latency-bound. Absolute wall-clock is faster, but the NCU metrics show the new bottleneck shape.

**Fix**: Read the absolute duration and memory throughput, not the SOL percentage alone. A drop in SOL accompanied by a larger drop in wall-clock is a win, not a regression.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-06). Retained as anecdotal; the metric-interpretation lesson is general.

## P8. Data compaction overhead exceeds the divergence it was meant to fix

**Symptom**: Sorting / binning / stream-compaction added upstream "to eliminate divergence" makes the pipeline slower overall.

**Root cause**: Compaction has its own memory traffic and launch overhead. For mild divergence (SIMT efficiency > 75 %) or small problem sizes (N < 1e5), the compaction step is pure overhead.

**Fix**: Profile divergence severity before applying S4 compaction. The NCU metric is `smsp__thread_inst_executed_per_inst_executed.ratio`: only pursue compaction when it is < 24 (i.e. average active lanes < 24/32 = 75 %).

**Source**: KernelPilot legacy L3 sandbox run (2026-04-06).

## P10. The SIMT-efficiency NCU metric can stay at 32/32 during actual serialization (measured on H200)

**Symptom**: A branching kernel is obviously paying a divergence cost (wall-clock 1.82× slower at p=0.25 than at p=0), but NCU's `smsp__thread_inst_executed_per_inst_executed.ratio` reports `32` (i.e. "perfect SIMT, no divergence") for every probability. The usual metric-based divergence hunt misses it entirely.

**Root cause**: NCU counts each serialized sub-warp execution as a *distinct* issued instruction. When a warp splits 20/12 into two bodies, NCU sees (issued=2, active-threads-per-issue=32) — not (issued=1, active-threads-per-issue=16). The ratio averages to 32. What actually grew is the **total count** of issued instructions, not their per-issue fill.

**Fix**: Do not rely on `smsp__thread_inst_executed_per_inst_executed.ratio` alone for divergence hunting. The authoritative signal is **wall-clock × Compute(SM) Throughput**:

- If Compute(SM) Throughput is high (≥ 80 %) AND wall-clock rises after adding a branch AND total output is unchanged → divergence serialized the body. Measured on H200: compute-bound branching kernel at 89 % SOL showed 1.82× wall-clock at p ∈ (0, 1), while SIMT efficiency reported 32/32 unchanged.
- If Compute(SM) Throughput is low → the kernel is memory-bound, divergence may be invisible in wall-clock, and the metric is honestly reporting "nothing to see here" (pitfall P7 / skill §"When NOT to use").

Look at `inst_issued.avg_per_cycle_active` instead to detect doubled-issue-rate from serialization; pair with wall-clock at constant occupancy to confirm.

**Source**: `sources/experience/hw-probes/warp-divergence-cost.md` (measured on H200).

## P9. Mis-categorizing `__syncwarp()` as a latency optimization

**Symptom**: `__syncwarp()` calls deleted from a kernel as "dead code" because they "don't do anything visible"; later, sporadic silent data corruption under heavy load.

**Root cause**: `__syncwarp()` prevents correctness bugs (pitfall P1) rather than accelerating anything. Removing it looks safe in short-run tests but opens a window where ITS permits a scheduling pattern that exposes stale data.

**Fix**: Treat `__syncwarp()` as a correctness primitive. Never remove one without confirming the surrounding code has no warp-synchronous dependency. The performance cost of `__syncwarp()` is one cycle on all lanes — negligible against the correctness it guarantees.

**Source**: KernelPilot legacy L3 sandbox run (2026-04-06).
