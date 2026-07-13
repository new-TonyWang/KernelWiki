# Barrier Optimization -- Pitfalls

## P1: Warp Divergence Causing Multiple Barrier Arrivals

**Symptom:** Barrier countdown reaches zero too early because diverged threads cause per-lane arrivals instead of per-warp.

**Detection:** Barrier completes before all logical arrivals, leading to race conditions.

**Fix:** Re-converge warps with `__syncwarp()` before calling `bar.arrive()`.

> Source: PG 4.9.2.1 -- "If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier."

---

## P2: Using arrive Token from Wrong Phase

**Symptom:** `bar.wait(std::move(token))` hangs or returns immediately because the token is from a different phase.

**Detection:** Deadlock or race condition in multi-phase barrier usage.

**Fix:** Ensure `token = bar.arrive()` and `bar.wait(std::move(token))` are called in the same barrier phase. Do not save tokens across multiple phases.

> Source: PG 4.9.2 -- "bar.wait() must only be called using a token object of the current phase or the immediately preceding phase."

---

## P3: __syncthreads in Conditional Code Causing Deadlock

**Symptom:** Kernel hangs because some threads in a block reach `__syncthreads()` while others do not.

**Detection:** GPU timeout. Kernel appears to hang at a barrier.

**Fix:** Ensure all threads in the block reach the same `__syncthreads()` call, or use conditional synchronization patterns (e.g., async barriers with subset participation).

```cpp
// BAD: deadlock if condition is not uniform across block
if (threadIdx.x < 128) {
    __syncthreads();  // only half the block reaches this!
}

// GOOD: all threads reach the barrier
__syncthreads();
if (threadIdx.x < 128) {
    // ... subset work ...
}
```

> Source: PG 5.4.4.1 -- "__syncthreads acts as a barrier at which all threads in the block must wait."

---

## P4: Excessive __syncthreads Calls Reducing SM Throughput

**Symptom:** SM spends significant time idle waiting at barriers. Low instruction throughput.

**Detection:** Nsight Compute shows high "stall barrier" percentage.

**Fix:**
1. Reduce the number of synchronization points.
2. Merge multiple barrier-separated phases when possible.
3. Use multiple smaller blocks per SM so warps from non-stalled blocks can run.

> Source: BP 12.1.3 -- "__syncthreads() can impact performance by forcing the multiprocessor to idle."

---

## P5: Forgetting to Initialize mbarrier Before Use

**Symptom:** Undefined behavior; barrier may never complete or complete immediately.

**Detection:** Sporadic deadlocks or race conditions involving mbarrier operations.

**Fix:** Always initialize mbarrier with `init(&bar, expectedCount)` from a single thread, then synchronize all participating threads before first use.

> Source: PG 4.9.1 -- "Initialization must happen before any thread begins participating in a barrier."

## P6: cuda::barrier objects consume more shared memory (2048→2176 bytes) and generate significantly more instructions than __syncthreads(); on small kernels the synchronization primitive overhead dominates any potential overlap benefit (discovered in verification)

**Symptom**: cuda::barrier objects consume more shared memory (2048→2176 bytes) and generate significantly more instructions than __syncthreads(); on small kernels the synchronization primitive overhead dominates any potential overlap benefit.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: Eliminating barrier stalls shifts the bottleneck—long scoreboard stalls rose from 24 (discovered in verification)

**Symptom**: Eliminating barrier stalls shifts the bottleneck—long scoreboard stalls rose from 24.8% to 30.5%, meaning memory latency hiding becomes the next optimization target once synchronization overhead is removed.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: Verifying __nanosleep requires a workload with genuine spin-wait contention (e (discovered in verification)

**Symptom**: Verifying __nanosleep requires a workload with genuine spin-wait contention (e.g., lock-based producer-consumer); without it the technique has nothing to optimize and short-scoreboard stalls actually increased slightly (5.21% → 6.17%), suggesting the nanosleep intrinsic itself adds minor overhead in the no-contention case.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: Scoreboard stalls (long + short) rose from 36 (discovered in verification)

**Symptom**: Scoreboard stalls (long + short) rose from 36.25% to 49.37% combined — not because they got worse in absolute terms, but because barrier stalls no longer dominate the stall breakdown; after eliminating barriers, memory latency becomes the next bottleneck to address.
**Source**: Level 3 sandbox verification (2026-04-04)
