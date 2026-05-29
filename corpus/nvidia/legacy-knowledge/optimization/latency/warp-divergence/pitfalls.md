# Warp Divergence -- Pitfalls

## P1: Relying on Implicit Warp-Synchronous Execution on Volta+

**Symptom:** Warp shuffle or shared memory exchange produces incorrect results on CC 7.0+ but works on CC 6.x.

**Detection:** Race conditions that appear only on Volta/Turing/Ampere+. Results differ between runs.

**Fix:** Replace implicit warp-synchronous assumptions with explicit `__syncwarp()` and use `_sync` variants of all warp intrinsics.

```cpp
// Bad: assumes lockstep execution (broken on Volta+)
if (laneId < 16) { smem[laneId] = val; }
if (laneId >= 16) { result = smem[laneId - 16]; }

// Good: explicit synchronization
if (laneId < 16) { smem[laneId] = val; }
__syncwarp();
if (laneId >= 16) { result = smem[laneId - 16]; }
```

> Source: PG 3.2.2.1.1 -- "Warp-synchronous code assumes that threads in the same warp execute in lockstep at every instruction, but the ability for threads to diverge and reconverge at sub-warp granularity makes such assumptions invalid."

---

## P2: Branching on threadIdx.x with Sub-Warp Granularity

**Symptom:** All threads within a warp execute both sides of a branch, doubling execution time for that warp.

**Detection:** Nsight Compute shows high "divergent branch" count. Instruction throughput is low relative to expected.

**Fix:** Restructure the branch to align with warp boundaries (multiples of 32).

```cpp
// Bad: threads 0-15 and 16-31 diverge within each warp
if (threadIdx.x < blockDim.x / 2) { ... } else { ... }

// Good: align to warp boundary
int warpId = threadIdx.x / 32;
if (warpId < numWarps / 2) { ... } else { ... }
```

> Source: BP 12.1.2 -- "Any flow control instruction can significantly impact the effective instruction throughput by causing threads of the same warp to diverge."

---

## P3: Early Return in Only Some Threads of a Warp

**Symptom:** Threads that return early are still consuming SM resources (their warp remains active). Performance does not improve with early exit.

**Detection:** `activemask` shows partial warps with many inactive threads.

**Fix:** Structure the exit condition so that entire warps exit together, not individual threads. Use `__ballot_sync` to check if all threads are done.

```cpp
// Bad: individual threads exit, warp stays active
if (myElement >= N) return;

// Better: check if entire warp is done
bool done = (myElement >= N);
if (__all_sync(0xFFFFFFFF, done)) return;
// Only process if at least one thread has work
if (!done) { /* skip */ } else { process(); }
```

> Source: PG 3.2.2.1 -- "threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp."

---

## P4: Using activemask Instead of Explicit Masks

**Symptom:** Incorrect warp-level operations due to using `__activemask()` as a synchronization primitive.

**Detection:** Subtle correctness bugs on Volta+ where `__activemask()` may not return the expected set of threads.

**Fix:** Use explicit full-warp masks (`0xFFFFFFFF`) or computed masks from `__ballot_sync()` instead of `__activemask()`.

> Source: PG 5.4.6.1 -- activemask returns the currently active threads but makes no guarantee about convergence.

---

## P5: Nested Divergent Branches Causing Exponential Serialization

**Symptom:** Multiple levels of nested if/else each halve the active threads, leading to very low SIMT efficiency.

**Detection:** Nsight Compute shows warp efficiency well below 50%. Multiple levels of predicated or branched code in SASS.

**Fix:** Flatten nested branches. Use lookup tables, predication, or data-driven approaches. Pre-compute and store decisions rather than branching at runtime.

```cpp
// Bad: 3 levels of nesting, up to 8x serialization
if (condA) {
    if (condB) {
        if (condC) { ... }
    }
}

// Better: flatten with combined condition
int path = (condA << 2) | (condB << 1) | condC;
result = lookupTable[path];  // branchless
```

> Source: BP 12.1.2 -- "the different execution paths have to be serialized, increasing the total number of instructions executed for this warp."

## P6: Aligning branches to warp boundaries can silently destroy memory coalescing when the two branch paths access interleaved addresses — the warp that previously had threads 0-31 reading contiguous memory now has non-contiguous threads grouped together (discovered in verification)

**Symptom**: Aligning branches to warp boundaries can silently destroy memory coalescing when the two branch paths access interleaved addresses — the warp that previously had threads 0-31 reading contiguous memory now has non-contiguous threads grouped together
**Source**: Level 3 sandbox verification (2026-04-06)

## P7: The optimized kernel drops from 90 (discovered in verification)

**Symptom**: The optimized kernel drops from 90.5% to 44.8% SM throughput and shows higher long scoreboard stalls (0.11%→32.13%), indicating that while total work is massively reduced, the remaining active warps are more memory-latency bound — the warp vote early-exit shifts the bottleneck from compute to memory latency for surviving warps.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Data restructuring techniques like sorting/binning have non-trivial overhead that can dominate when the original divergence is minimal or the problem size is small; always profile divergence severity before applying compaction (discovered in verification)

**Symptom**: Data restructuring techniques like sorting/binning have non-trivial overhead that can dominate when the original divergence is minimal or the problem size is small; always profile divergence severity before applying compaction.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: Categorizing __syncwarp() as a latency optimization is misleading; it prevents subtle correctness bugs (silent data corruption) that only manifest under specific scheduling conditions, so verification-pass alone does not prove it is unnecessary (discovered in verification)

**Symptom**: Categorizing __syncwarp() as a latency optimization is misleading; it prevents subtle correctness bugs (silent data corruption) that only manifest under specific scheduling conditions, so verification-pass alone does not prove it is unnecessary.
**Source**: Level 3 sandbox verification (2026-04-06)
