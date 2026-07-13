# Thread Scopes -- Pitfalls

## P1: Using Device Scope When System Scope Is Required

**Symptom:** CPU reads stale GPU data despite GPU kernel having completed. Multi-GPU communication sees incorrect values.

**Detection:** Intermittent correctness failures in GPU-CPU or GPU-GPU communication. Works sometimes due to timing.

**Fix:** Use `cuda::thread_scope_system` or `__threadfence_system()` for any communication crossing the GPU boundary.

```cpp
// BAD: device scope doesn't guarantee CPU visibility
__threadfence();  // device scope only
gpuToCpuFlag = 1;

// GOOD: system scope ensures CPU sees the writes
__threadfence_system();
gpuToCpuFlag = 1;
```

> Source: PG 3.2.3 -- device scope is visible only to threads on the same GPU.

---

## P2: Using System Scope Everywhere "To Be Safe"

**Symptom:** All atomics and fences use system scope, causing 2-10x performance degradation compared to device or block scope.

**Detection:** Profile shows high latency on fence/atomic instructions despite no cross-device communication.

**Fix:** Audit each atomic/fence and use the narrowest scope that is correct. Most GPU-internal operations need at most device scope; intra-block operations need only block scope.

> Source: PG 3.2.4.1.2 -- "Use the narrowest scope possible: block-scoped atomics are much faster than system-scoped atomics."

---

## P3: Missing Fence Between Data Write and Flag Write

**Symptom:** Consumer thread sees the flag but reads stale data because the data write was not fenced before the flag write.

**Detection:** Race condition where the flag is observed as set but associated data is incorrect.

**Fix:** Use release semantics on the flag store (which implies a fence) or an explicit fence before the flag write.

```cpp
// BAD: no ordering between data write and flag write
data[tid] = result;
flag = 1;  // consumer may see flag=1 but data[tid] is stale

// GOOD: release ensures data is visible before flag
data[tid] = result;
flag.store(1, cuda::memory_order_release);
```

> Source: PG 3.2.4.1.1 -- release semantics ensure prior writes are visible to the acquiring thread.

---

## P4: Confusing Legacy membar with Modern fence Semantics

**Symptom:** Code uses `membar.gl` (legacy) when modern `fence.acq_rel.gpu` would be more efficient and semantically clearer.

**Detection:** Legacy `membar` instructions in PTX/SASS output.

**Fix:** Use modern `fence` instructions or `cuda::atomic_thread_fence` which provide explicit scope and ordering control. Legacy `membar` provides stronger-than-necessary guarantees.

> Source: PTX ISA -- `fence` provides more precise control than legacy `membar`.

---

## P5: Block-Scoped Fence Used for Inter-Block Communication

**Symptom:** Data written by one block is not visible to another block, despite the fence appearing correct.

**Detection:** Inter-block communication fails while intra-block communication works.

**Fix:** Use device scope (`__threadfence()` or `fence.acq_rel.gpu`) for inter-block communication. Block scope only ensures visibility within the same block.

> Source: PG 3.2.3 -- block scope coherency is at L1, which is private per SM.

## P6: Barrier stall percentage rose from 14 (discovered in verification)

**Symptom**: Barrier stall percentage rose from 14.5% to 20.6% — the cheaper fence makes synchronization a larger fraction of remaining execution time, so further gains require reducing sync frequency itself.
**Source**: Level 3 sandbox verification (2026-04-06)

## P7: Applying device-scope fences with kernel splitting can move all computation out of the profiled kernel, making NCU metrics meaningless while adding kernel launch overhead that worsens end-to-end latency (discovered in verification)

**Symptom**: Applying device-scope fences with kernel splitting can move all computation out of the profiled kernel, making NCU metrics meaningless while adding kernel launch overhead that worsens end-to-end latency.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Long scoreboard stalls nearly doubled (16 (discovered in verification)

**Symptom**: Long scoreboard stalls nearly doubled (16.8% → 30.1%), suggesting the faster kernel exposes more memory latency pressure that was previously hidden by fence stall time — further optimization may need to address memory-level parallelism.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: Scoped atomics may slightly increase short-scoreboard stalls (10 (discovered in verification)

**Symptom**: Scoped atomics may slightly increase short-scoreboard stalls (10.07% → 12.67%) on trivially small kernels, possibly due to different instruction scheduling of the libcu++ intrinsics.
**Source**: Level 3 sandbox verification (2026-04-06)
