# Cooperative Groups -- Pitfalls

## P1: Exceeding Maximum Resident Blocks with Cooperative Kernel Launch

**Symptom:** `cudaLaunchCooperativeKernel` fails because the grid size exceeds what can simultaneously reside on the GPU.

**Detection:** `cudaErrorCooperativeLaunchTooLarge` or similar error.

**Fix:** Query maximum blocks with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and limit grid size accordingly.

```cpp
int maxBlocksPerSM;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel, blockSize, 0);
int gridSize = maxBlocksPerSM * numSMs;  // max that can co-reside
```

> Source: PG 4.4 -- cooperative launch requires all blocks to be resident simultaneously.

---

## P2: Copy-Constructing Group Handles

**Symptom:** Performance degradation or unexpected behavior from copy-constructing cooperative group handles.

**Detection:** Multiple handle creations in profiled code paths.

**Fix:** Create handles once at kernel entry and pass by reference to helper functions.

> Source: PG 4.4.3.2 -- "Copy-constructing group handles is discouraged."

---

## P3: Using coalesced_threads Across Divergent Points

**Symptom:** The set of threads in the coalesced group changes unexpectedly between operations, leading to incorrect collective results.

**Detection:** Different threads participate in successive operations on the same coalesced group handle.

**Fix:** Understand that `coalesced_threads()` reflects a snapshot of active threads at the call site. Re-call it if the active set may have changed.

> Source: PG 4.4.3 -- "coalesced_threads makes no guarantee about which threads are returned or that they will stay coalesced throughout execution."

---

## P4: Grid Sync Deadlock from Insufficient Blocks

**Symptom:** `grid.sync()` deadlocks because not all blocks in the grid can be resident, and some have not yet started.

**Detection:** Kernel hangs at `grid.sync()`.

**Fix:** Only use `grid.sync()` with cooperative launch API, which ensures all blocks are resident. Never use `grid.sync()` with normal `<<<>>>` launch.

> Source: PG 4.4 -- grid-wide sync requires all blocks to be concurrently resident.

---

## P5: Assuming this_cluster() Works Without Cluster Launch

**Symptom:** `this_cluster()` returns a cluster with 1 block when the kernel was not launched with cluster dimensions.

**Detection:** Cluster group has `num_blocks() == 1` even though cluster behavior was expected.

**Fix:** Launch with cluster dimensions via `cudaLaunchKernelExC` and appropriate attributes.

> Source: PG 4.4.3 -- "this_cluster() assumes a 1x1x1 cluster when a non-cluster grid is launched."

## P6: After eliminating barrier stalls, long scoreboard (memory latency) stalls rose from 21 (discovered in verification)

**Symptom**: After eliminating barrier stalls, long scoreboard (memory latency) stalls rose from 21.4% to 38.9%, becoming the new dominant bottleneck — tiled_partition exposes latency-hiding as the next optimization target.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: grid (discovered in verification)

**Symptom**: grid.sync() introduced 39.4% barrier stalls (from 0%) and halved active warp occupancy (80.6% → 48.8%); cooperative launch also increased register pressure (16→18 regs/thread) and tightened the shared-memory occupancy limit (32→8 blocks), so the block count must be carefully tuned to the SM's concurrent capacity.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Both baseline and optimized kernels remain massively underutilized (<1% of roofline); the 59% speedup is real but measured at microsecond scale on a tiny workload — gains may narrow or widen significantly at larger problem sizes (discovered in verification)

**Symptom**: Both baseline and optimized kernels remain massively underutilized (<1% of roofline); the 59% speedup is real but measured at microsecond scale on a tiny workload — gains may narrow or widen significantly at larger problem sizes.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: This skill is a correctness/maintainability guideline masquerading as a performance tip; at 85% SM throughput the kernel is already well-optimized and group handle overhead is negligible compared to actual compute work (discovered in verification)

**Symptom**: This skill is a correctness/maintainability guideline masquerading as a performance tip; at 85% SM throughput the kernel is already well-optimized and group handle overhead is negligible compared to actual compute work.
**Source**: Level 3 sandbox verification (2026-04-06)
