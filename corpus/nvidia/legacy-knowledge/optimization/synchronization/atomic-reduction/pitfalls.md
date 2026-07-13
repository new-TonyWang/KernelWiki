# Atomic Reduction -- Pitfalls

## P1: All Threads Atomically Adding to a Single Global Location

**Symptom:** Massive contention at one memory address. Kernel throughput is orders of magnitude below peak.

**Detection:** Nsight Compute shows high "stall long scoreboard" or "stall MIO throttle" from atomic operations. Most warps stalled on atomics.

**Fix:** Use hierarchical reduction (warp shuffle -> shared memory -> single global atomic per block).

> Source: PG 2.2.5 -- "Atomic functions should be used sparingly as they enforce thread synchronization."

---

## P2: Using System Scope When Device Scope Suffices

**Symptom:** Atomic operations are 2-10x slower than necessary because they flush to the system scope (across PCIe/NVLink).

**Detection:** Profile shows long latency on atomic instructions despite no cross-device communication.

**Fix:** Use `cuda::thread_scope_device` (or `cuda::thread_scope_block` for intra-block) instead of `cuda::thread_scope_system`.

```cpp
// BAD: system scope when only device threads participate
cuda::atomic<int, cuda::thread_scope_system> counter;

// GOOD: device scope is sufficient
cuda::atomic<int, cuda::thread_scope_device> counter;
```

> Source: PG 3.2.4.1.2 -- "block-scoped atomics are much faster than system-scoped atomics."

---

## P3: Using seq_cst When Relaxed Ordering Suffices

**Symptom:** Unnecessary memory fences around every atomic operation, reducing throughput.

**Detection:** SASS shows fence instructions around atomics that only need atomicity, not ordering.

**Fix:** Use `cuda::memory_order_relaxed` for simple counters and accumulators. Only use `acquire`/`release` for producer-consumer patterns.

> Source: PG 3.2.4.1.2 -- "Prefer weaker orderings."

---

## P4: Race Condition from Non-Atomic Read-Modify-Write

**Symptom:** Incorrect results that vary between runs. "Lost updates" in counters or histograms.

**Detection:** Results are non-deterministic. Reducing to a single block or single thread makes the result correct.

**Fix:** Use atomic operations for all shared-state updates that can be concurrent.

```cpp
// BAD: non-atomic increment (race condition)
counter++;

// GOOD: atomic increment
atomicAdd(&counter, 1);
```

> Source: PG 2.2.5 -- atomics provide indivisible read-modify-write operations.

---

## P5: FP16/BF16 Atomics with Flush-to-Zero Behavior

**Symptom:** Small values in FP16 accumulation are lost because `atom.add.f16` flushes denormals to zero.

**Detection:** Numerical accuracy issues when accumulating many small FP16 values.

**Fix:** Use `atom.add.noftz.f16` or `atom.add.noftz.bf16` to preserve denormal values.

> Source: PTX ISA -- atom.add.noftz.f16/bf16 preserves denormals.

## P6: Both rounds show the kernel is deeply underutilized (memory SOL 0 (discovered in verification)

**Symptom**: Both rounds show the kernel is deeply underutilized (memory SOL 0.7%, compute SOL 0.0%) — the test input is too small (4KB read) to meaningfully stress the GPU, so the 174x speedup is entirely from fixing launch/algorithmic overhead, not from demonstrating memory-throughput gains of hierarchical reduction at scale.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: Scoped atomics only help when atomic contention is a significant fraction of execution time; in memory-bound kernels with few atomics relative to loads, the optimization is invisible (discovered in verification)

**Symptom**: Scoped atomics only help when atomic contention is a significant fraction of execution time; in memory-bound kernels with few atomics relative to loads, the optimization is invisible.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: Relaxed ordering visibly reduces instructions and memory stalls in NCU metrics, which can mislead you into thinking it helped — always check wall-clock time, not just micro-metrics (discovered in verification)

**Symptom**: Relaxed ordering visibly reduces instructions and memory stalls in NCU metrics, which can mislead you into thinking it helped — always check wall-clock time, not just micro-metrics.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: Both baseline and optimized kernels remain ~5x slower than PyTorch (~150us) and are massively underutilized (<1% efficiency on a 4KB problem); shared memory atomics help but cannot overcome launch-overhead dominance at this tiny problem size (discovered in verification)

**Symptom**: Both baseline and optimized kernels remain ~5x slower than PyTorch (~150us) and are massively underutilized (<1% efficiency on a 4KB problem); shared memory atomics help but cannot overcome launch-overhead dominance at this tiny problem size.
**Source**: Level 3 sandbox verification (2026-04-04)
