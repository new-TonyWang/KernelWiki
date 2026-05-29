# Warp Divergence -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 12.1.2 (Control Flow Instructions)"
  - "Best Practices Guide 13.1 (Branching and Divergence)"
  - "Best Practices Guide 13.2 (Branch Predication)"
  - "Programming Guide 1.2.2.2 (Warps and SIMT)"
  - "Programming Guide 2.2.1 (Basics of SIMT)"
  - "Programming Guide 3.2.2.1 (SIMT Execution Model)"
  - "Programming Guide 3.2.2.1.1 (Independent Thread Scheduling)"
cross_ref:
  - optimization/compute/warp-primitives
  - optimization/compute/compiler-hints
  - optimization/latency/occupancy-tuning
related_apis:
  - "@{!}p (predicated execution)"
  - "setp.CmpOp"
  - "selp.type"
  - "vote.sync"
  - __ballot_sync
  - __all_sync
  - __any_sync
unlocks:
  - Full SIMT efficiency (all 32 threads executing useful work)
  - Reduced instruction count per warp
conflicts_with:
  - None (divergence elimination is always beneficial)
```

---

## S1: Align Branch Conditions to Warp Boundaries

**When to Use:** When control flow depends on thread ID and the branch condition can be structured so that entire warps take the same path.

**How to Apply:**
1. Express the condition in terms of `threadIdx / warpSize` rather than `threadIdx`.
2. Ensure data-dependent branches partition along warp boundaries where possible.

**Code Template:**
```cpp
int warpId = threadIdx.x / 32;

// Good: no divergence within a warp
if (warpId < numProducerWarps) {
    produce(data);
} else {
    consume(data);
}

// Bad: threads 0-15 vs 16-31 diverge within each warp
if (threadIdx.x % 32 < 16) {
    pathA();
} else {
    pathB();
}
```

> Source: BP 13.1 -- "the controlling condition should be written so as to minimize the number of divergent warps... A trivial example is when the controlling condition depends only on (threadIdx / WSIZE)."

---

## S2: Use Predication for Short Conditional Blocks

**When to Use:** When a branch body is small (a few instructions). The compiler can replace the branch with predicated execution, avoiding divergence overhead entirely.

**How to Apply:**
1. Keep conditional blocks short (compiler threshold is typically ~7 instructions).
2. Use ternary operators or `min`/`max` instead of if/else where possible.
3. The compiler will automatically use `selp` (select-on-predicate) or `@p` predicated instructions.

**Code Template:**
```cpp
// Encourages predication (no actual branch)
float result = (condition) ? valueA : valueB;

// Also good: branchless clamp
float clamped = fminf(fmaxf(value, lo), hi);

// Instead of:
float result;
if (condition) {
    result = complexA();  // too many instructions -> real branch
} else {
    result = complexB();
}
```

> Source: BP 13.2 -- "The compiler replaces a branch instruction with predicated instructions only if the number of instructions controlled by the branch condition is less than or equal to a certain threshold."

---

## S3: Use Warp Vote Functions to Eliminate Uniform Branches

**When to Use:** When you need to check a condition across all threads in a warp and skip work if the entire warp agrees.

**How to Apply:**
1. Use `__all_sync(mask, predicate)` to test if all threads satisfy a condition.
2. Use `__any_sync(mask, predicate)` to test if any thread satisfies a condition.
3. Use `__ballot_sync(mask, predicate)` to get a bitmask for more complex logic.

**Code Template:**
```cpp
// Early exit if entire warp has no work
bool hasWork = (myData[tid] > threshold);
if (!__any_sync(0xFFFFFFFF, hasWork)) {
    return;  // no divergence: all threads in warp exit together
}

// Process only if all threads are valid
if (__all_sync(0xFFFFFFFF, tid < N)) {
    // safe to proceed without bounds checking per-thread
    output[tid] = compute(input[tid]);
}
```

> Source: PG 5.4.6.2 -- warp vote functions allow collective warp-level decisions.

---

## S4: Restructure Data to Avoid Data-Dependent Divergence

**When to Use:** When input data causes threads within the same warp to take different paths (e.g., sparse data, irregular graphs).

**How to Apply:**
1. Sort or bin input data so that similar items are processed by the same warp.
2. For sparse computations, compact active elements with a prefix sum before processing.
3. Use `__ballot_sync` + `__popc` to count and redistribute work.

**Code Template:**
```cpp
// Compact active elements to avoid divergence
unsigned mask = __ballot_sync(0xFFFFFFFF, isActive);
int activeCount = __popc(mask);
int myOffset = __popc(mask & ((1u << laneId) - 1));

if (isActive) {
    compactBuffer[baseOffset + myOffset] = myData;
}
// Then launch a second pass on the compacted, dense buffer
```

> Source: BP 12.1.2 -- "the controlling condition should be written so as to minimize the number of divergent warps."

---

## S5: Use __syncwarp() After Divergent Regions (Volta+)

**When to Use:** On Volta+ (CC 7.0+) where Independent Thread Scheduling means diverged threads may not automatically reconverge.

**How to Apply:**
1. After any divergent branch that is followed by warp-level communication (shuffle, vote), insert `__syncwarp()`.
2. Always use `_sync` variants of warp intrinsics with explicit masks.

**Code Template:**
```cpp
if (threadIdx.x % 2 == 0) {
    val = computeEven();
} else {
    val = computeOdd();
}
__syncwarp();  // ensure reconvergence before shuffle
float neighbor = __shfl_xor_sync(0xFFFFFFFF, val, 1);
```

> Source: PG 3.2.2.1.1 -- "Independent thread scheduling can break code that relies on implicit warp-synchronous behavior from previous GPU architectures."

---

## Cascading Opportunities

- After reducing divergence, the kernel may become compute-bound -- check `fast-math` and `instruction-level-parallelism`.
- Warp-aligned branching enables efficient `warp-primitives` usage (shuffle, reduce).
- Data compaction to eliminate divergence feeds into `coalescing` improvements.

## Conflicts

- No inherent conflicts. Divergence elimination is universally beneficial.

## Principles

1. **Divergence is warp-local:** Different warps can take different paths with no penalty. Only threads within the same warp matter (PG 3.2.2.1).
2. **Predication hides small branches:** The compiler converts short branches to predicated instructions automatically (BP 13.2).
3. **Volta changed the rules:** Independent Thread Scheduling (CC 7.0+) requires explicit `__syncwarp()` for correctness after divergent code (PG 3.2.2.1.1).

## Open Questions

- Q1: What is the exact compiler threshold for predication vs. branching in CUDA 13.x?
- Q2: How does the cost of warp reconvergence compare across CC 7.0/8.0/9.0/10.0?
