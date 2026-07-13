---
title: Barrier Optimization — Pitfalls
status: draft
id: pitfall-barrier-optimization
type: pitfall
vendor: nvidia
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
hardware_features:
- mbarrier
techniques:
- vectorized-loads
- shared-memory-optimization
- communication-overlap
kernel_types:
- quantization
confidence: inferred
tags:
- mbarrier
- vectorized-loads
- shared-memory-optimization
- communication-overlap
- quantization
- ptx
- cuda-cpp
---
# Barrier Optimization — Pitfalls

## P1. Divergent warp causes per-lane mbarrier arrivals

**Symptom**: An `mbarrier.arrive` or `cuda::barrier::arrive` completes the barrier too early. The barrier's arrival count is reached before all *logical* participants have produced their output, producing race conditions in the consumer phase.

**Root cause**: When a warp reaches `mbarrier.arrive` while diverged (some lanes active, others masked off), each active lane issues its own arrival. If the warp was expected to contribute `arrival_count = blockDim / 32` but the warp is split into e.g. 16 active + 16 masked lanes, only 16 arrivals happen from that warp — *or*, worse, if later the masked lanes continue and also arrive, you get double-counting. ITS on sm_70+ makes this non-deterministic across runs.

**Fix**: Always `__syncwarp(0xFFFFFFFFu)` to re-converge the warp before `mbarrier.arrive`. PG §4.9.2.1: "If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier." The warp-divergence skill's pitfall P1 also covers this under the correctness-on-Volta+ lens.

```cuda
// WRONG: diverged warp arrives multiple times
if (lane_local_ready) {
    auto token = bar.arrive();     // only active lanes count
    bar.wait(std::move(token));
}

// RIGHT: all 32 lanes arrive together
__syncwarp(0xFFFFFFFFu);            // re-converge
auto token = bar.arrive();          // always 1 arrival per thread
// ... later ...
bar.wait(std::move(token));
```

**Source**: CUDA C++ Programming Guide §4.9.2.1.

## P2. Using an arrive token from the wrong phase

**Symptom**: `bar.wait(std::move(token))` either hangs (future phase) or returns immediately (stale past phase), producing silent correctness bugs.

**Root cause**: The token returned by `bar.arrive()` is specific to the barrier's current phase. If the barrier advances phases between arrive and wait (because other threads have completed the phase), the token becomes invalid.

**Fix**: Keep arrive and wait in the same lexical scope and same phase. Never save a token across `bar.wait`. Per PG §4.9.2: "bar.wait() must only be called using a token object of the current phase or the immediately preceding phase." The "immediately preceding" part is the only legal cross-phase case.

**Source**: CUDA C++ Programming Guide §4.9.2.

## P3. `__syncthreads()` inside a divergent branch deadlocks the kernel

**Symptom**: Kernel hangs indefinitely. `nvidia-smi` shows the GPU busy with no progress. Timeouts.

**Root cause**: `__syncthreads` is a *collective* barrier — every thread in the block must execute it. If some threads skip the call because they took a different branch, the remaining threads wait forever.

**Fix**: Move `__syncthreads` outside the branch so every thread reaches it.

```cuda
// WRONG: threads with tid >= 128 never reach the barrier
if (threadIdx.x < 128) {
    smem[tid] = compute(tid);
    __syncthreads();            // DEADLOCK — only half the block is here
}

// RIGHT: all threads reach the barrier; predicate only the work
if (threadIdx.x < 128) {
    smem[tid] = compute(tid);
}
__syncthreads();                // all threads reach this point
```

**Source**: CUDA C++ Programming Guide §5.4.4.1.

## P4. Excessive `__syncthreads` calls dominate SM throughput

**Symptom**: NCU shows `smsp__warps_issue_stalled_barrier_per_issue_active.pct` above 20 %. Instruction throughput is low; kernel is mostly idle at barriers.

**Root cause**: Each `__syncthreads` stalls the SM until the slowest warp arrives. A kernel with many small phases separated by `__syncthreads` pays this "wait for slowest" cost once per phase. The problem compounds when occupancy is low (one block per SM) — there are no other warps to cover the idle.

**Fix (in order)**:
1. Merge adjacent barrier-separated phases when correctness allows.
2. Increase blocks-per-SM (smaller blocks or smaller smem footprint) so warps from other blocks run during a barrier wait.
3. Split the reduction into warp-local + one block-level merge (atomic-reduction S1 pattern).
4. For producer-consumer shapes, convert to async-barrier (S2 in this skill) so the consumer warp does real work between arrive and wait.

**Source**: Best Practices Guide §12.1.3 (L1402-L1404).

## P5. Forgetting to initialize `mbarrier` before first use

**Symptom**: Sporadic deadlocks, or barrier completes immediately on first phase with no threads having arrived. Non-reproducible bugs that only appear on some inputs or with some block sizes.

**Root cause**: `mbarrier`'s 64-bit state is memory-backed. Without `mbarrier.init`, the state is garbage and the first `arrive`/`wait` works against whatever values happen to be in that shared memory slot.

**Fix**: Always initialize from a single thread and `__syncthreads` before first use:

```cuda
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
if (threadIdx.x == 0) init(&bar, blockDim.x);
__syncthreads();                  // all threads see initialized bar
// ... now safe to call bar.arrive() / bar.wait() ...
```

**Source**: CUDA C++ Programming Guide §4.9.1.

## P6. `cuda::barrier` arrive+wait is slower than `__syncthreads` in bare cost (measured on H200)

**Symptom**: Replacing `__syncthreads` with `cuda::barrier` + arrive/wait split *increases* per-barrier wall-clock on kernels that don't have independent work to overlap.

**Root cause**: `cuda::barrier<thread_scope_block>` lives in shared memory as a 64-bit mbarrier object. Each `arrive` + `wait` pair emits multiple PTX instructions (`mbarrier.arrive.shared::cta.b64` returning a token, then `mbarrier.try_wait.shared::cta.b64` that blocks on the token), plus a read-modify-write to the shared-memory phase state. `__syncthreads` uses the SM's dedicated named-barrier hardware (16 slots per SM) without touching smem. The overlap benefit only materializes when there is real independent work to run between arrive and wait (~100+ cycles of compute).

**Fix**: Reach for `cuda::barrier` only when (a) the kernel has independent producer/consumer work that can overlap across arrive/wait, OR (b) you need `expect_tx` for async-copy tracking. For simple fence-between-phases, `__syncthreads` is still the right tool.

**Measured on H200 sm_90a** (`sources/experience/hw-probes/barrier-cost.md`): bare arrive+wait cost per call is 1.59× (at B=128) to **2.23× (at B=1024) slower** than `__syncthreads`. Scaling factor across 128→1024 is 2.72× for mbarrier vs 1.93× for `__syncthreads` — the mbarrier gap widens with block size, so the "no overlap" loss is worst at exactly the block sizes where barrier cost matters most.

**Source**: H200 probe (measured, 2026-04-22), supersedes KernelPilot legacy L3 sandbox (2026-04-04) which was directionally correct but un-quantified.

## P7. Removing barriers shifts the bottleneck to memory latency (legacy-measured)

**Symptom**: After removing `__syncthreads` stalls, `smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct` rises from 24.8 % to 30.5 %. The NCU "barrier stall" metric went down but a new stall metric went up.

**Root cause**: The barrier stall was *hiding* the memory-latency stall. Once the barrier no longer serializes warps, the warps discover the memory wait that had been happening in parallel with the barrier. The total wall-clock is still better, but the NCU stall profile looks different.

**Fix**: Read wall-clock, not just the stall %. If wall-clock improved, the optimization worked — the "rising long_scoreboard" is just the next bottleneck showing itself. Follow with memory-side optimizations (coalescing, vectorized access, async-copy).

**Source**: KernelPilot legacy L3 sandbox (2026-04-04). Consistent with the "metric shape shift" observation from the warp-divergence skill pitfall P7.

## P8. `__nanosleep` requires actual spin-wait contention to be useful

**Symptom**: Adding `__nanosleep(N)` into a spin-wait loop shows no wall-clock improvement — sometimes a small regression (`short_scoreboard` stall rose from 5.21 % to 6.17 %).

**Root cause**: `__nanosleep` is a thread-throttle that reduces SM resource pressure **when the spin is actually contending for a lock or flag**. In a synthetic microbench with no real contention, the nanosleep instruction itself adds latency without saving anything.

**Fix**: Only apply `__nanosleep` in spin loops that have real contention (lock-based producer-consumer, CAS-based free lists, spinlock-guarded data structures). For plain phase-waiting, `mbarrier.try_wait` with its implicit hardware throttling is the right answer.

**Source**: KernelPilot legacy L3 sandbox (2026-04-04).

## P9. `smsp__warps_issue_stalled_barrier` metric drops after a bottleneck shift

**Symptom**: Combined "stall barrier + stall scoreboard" rose from 36.25 % to 49.37 % after a barrier optimization — looks like a regression.

**Root cause**: Same as P7 — the optimization shifted the stall shape, not the underlying bottleneck intensity. Individual stall categories add up to total idle time; making one go down can make another appear to "rise" simply because the total is redistributed.

**Fix**: Use wall-clock as the final judge. Only use NCU stall percentages to identify **which category to address next**, never to grade the success of an already-applied optimization.

**Source**: KernelPilot legacy L3 sandbox (2026-04-04).
