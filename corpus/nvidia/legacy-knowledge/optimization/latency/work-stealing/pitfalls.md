# Work Stealing -- Pitfalls

## P1: Using Work Stealing on Pre-Blackwell Hardware

**Symptom:** Compilation or runtime error when using `clusterlaunchcontrol` instructions on CC < 10.0.

**Detection:** PTX assembly errors or unsupported instruction errors at runtime.

**Fix:** Check compute capability at compile time. Fall back to atomic-counter persistent kernel pattern for older hardware.

> Source: PG 4.12 -- Cluster Launch Control is introduced in NVIDIA Blackwell (CC 10.0).

---

## P2: Not Handling the try_cancel Failure Case

**Symptom:** Kernel hangs or runs indefinitely because the work-stealing loop does not exit when cancellation fails.

**Detection:** Kernel timeout. GPU appears hung.

**Fix:** Always check the return value of `clusterlaunchcontrol.try_cancel`. Exit the loop when cancellation fails.

```cpp
// BAD: infinite loop
while (true) {
    processBlock(data, idx);
    // Missing failure check!
    asm volatile("clusterlaunchcontrol.try_cancel %0, %1;" : "=r"(ok), "=r"(idx));
}

// GOOD: exit on failure
while (true) {
    processBlock(data, idx);
    bool ok; int newIdx;
    asm volatile("clusterlaunchcontrol.try_cancel %0, %1;" : "=r"(ok), "=r"(newIdx));
    if (!ok) break;
    idx = newIdx;
}
```

> Source: PG 4.12 -- "The cancellation will fail if there are no more thread block indices available or for other reasons."

---

## P3: Atomic Counter Bottleneck in Software Work Stealing

**Symptom:** On pre-Blackwell hardware using atomic counters for work distribution, the atomic becomes a serialization bottleneck.

**Detection:** Nsight Compute shows high atomic contention. Many warps stalled on `atom.add`.

**Fix:** Use hierarchical work distribution: each SM claims a batch of tiles atomically, then distributes within the batch locally. Or use Blackwell's hardware cluster launch control.

```cpp
// Better: claim batches of tiles to reduce atomic contention
__shared__ int localBatch;
if (threadIdx.x == 0) {
    localBatch = atomicAdd(&globalCounter, BATCH_SIZE);
}
__syncthreads();
for (int i = 0; i < BATCH_SIZE; i++) {
    int tileIdx = localBatch + i;
    if (tileIdx >= totalTiles) break;
    processTile(tileIdx);
}
```

> Source: inferred from PG 4.12 -- cluster launch control eliminates software atomic overhead.

---

## P4: Work Stealing Preventing Higher-Priority Kernel Preemption (Persistent Kernel)

**Symptom:** A persistent kernel (grid-stride loop) occupies all SMs indefinitely, preventing a higher-priority kernel from launching.

**Detection:** Higher-priority kernel launch blocks until the persistent kernel completes.

**Fix:** Use cluster launch control instead of persistent kernel. The `try_cancel` mechanism allows graceful yielding to higher-priority work.

> Source: PG 4.12 -- fixed-number-of-blocks approach has "Preemption: X" while cluster launch control has "Preemption: V".

## P5: Persistent work-stealing kernels on already memory-bound, uniform-cost tile workloads add atomic contention and reduce occupancy without any load-balancing benefit — the baseline's static tile assignment was already near-optimal at 93 (discovered in verification)

**Symptom**: Persistent work-stealing kernels on already memory-bound, uniform-cost tile workloads add atomic contention and reduce occupancy without any load-balancing benefit — the baseline's static tile assignment was already near-optimal at 93.6% memory SOL.
**Source**: Level 3 sandbox verification (2026-04-04)
