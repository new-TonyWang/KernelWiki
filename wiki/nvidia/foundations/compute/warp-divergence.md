---
title: Warp Divergence Cost & Mitigation
status: verified
evidence_level: measured
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9.86 + ptxas 12.9
measured_on: H200-SXM
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- indexing
- pooling
- reduction
- scan
requires_sm: '>=7.0'
requires_features:
- independent-thread-scheduling
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1602-L1612
  excerpt: Avoid different execution paths within the same warp. Flow control instructions
    (if, switch, do, for, while) can significantly affect the instruction throughput
    by causing threads of the same warp to diverge; if this happens, the different
    execution paths must be executed separately, increasing the total number of instructions
    executed for this warp.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1396-L1400
  excerpt: To obtain best performance in cases where the control flow depends on the
    thread ID, the controlling condition should be written so as to minimize the number
    of divergent warps. A trivial example is when the controlling condition only depends
    on (threadIdx / warpSize), in which case no warp diverges since the controlling
    condition is perfectly aligned with the warps.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1614-L1628
  excerpt: When using branch predication, none of the instructions whose execution
    depends on the controlling condition is skipped. Instead, each such instruction
    is associated with a per-thread condition code or predicate. The compiler replaces
    a branch with predicated instructions only if the number of instructions controlled
    by the branch condition is less than or equal to a certain threshold.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3427-L3436
  excerpt: Independent thread scheduling (CC 7.0+) can break code that relies on implicit
    warp-synchronous behavior. Warp-synchronous code assumes that threads in the same
    warp execute in lockstep at every instruction, but the ability for threads to
    diverge and reconverge at sub-warp granularity makes such assumptions invalid.
    Developers should explicitly synchronize with __syncwarp() to ensure correct behavior.
artifacts:
  code: sources/experience/hw-probes/warp-divergence-cost/artifacts/divergence_cost_probe.cu
  build: sources/experience/hw-probes/warp-divergence-cost/artifacts/build.sh
  introspection: sources/experience/hw-probes/warp-divergence-cost/artifacts/device.json
  profile: ''
related_apis:
- __ballot_sync
- __all_sync
- __any_sync
- __syncwarp
- __activemask
related_skills:
- branch-elimination
- warp-primitives
- coalescing
id: skill-warp-divergence
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

**Warp divergence** is what happens when threads of the same 32-lane warp need to execute different instructions because a branch condition evaluated differently across those threads. The hardware handles this by executing **each path serially** — lanes that take path A run while the lanes that took path B are masked off, then the paths swap. Both paths count toward instruction cost, so a fully-divergent branch costs up to the **sum** of both paths.

BP §13.1 (L1602-L1612) states the rule bluntly: "Avoid different execution paths within the same warp." The quantifier that matters is **how much of the warp's work each path actually represents** — the serialization cost scales with the number of instructions in the longer path, multiplied by the number of distinct paths the warp must walk.

Divergence has three relevant sub-cases, each with a different mitigation:

1. **ThreadIdx-aligned divergence** — the condition depends on `threadIdx.x` and not `threadIdx.x / 32`; both paths live in the same warp. Fix: **restructure the branch to align on warp boundaries** (S1).
2. **Short-body divergence** — both paths are small (few instructions). Fix: **let the compiler predicate it** (S2). Often happens automatically, but hostile C++ patterns (function calls inside the branch, volatile writes, `printf`) block predication — rewrite for branch-free expressions.
3. **Data-dependent divergence** — the condition depends on input data that cannot be grouped at compile time (sparse data, graph traversal). Fix: **compact / sort / bin the data before the kernel** so each warp sees homogeneous work, or use **warp vote** for early-exit when entire warps have no work (S3 + S4).

Plus one correctness wrinkle since Volta (sm_70):

4. **Independent Thread Scheduling (ITS)** — diverged threads may stay diverged after the conditional block ends. Any subsequent warp-synchronous code (shuffle, smem exchange without a barrier) must use `_sync` intrinsics with explicit masks, and must call `__syncwarp()` explicitly after the divergent region (PG §3.2.2.1.1, L3427-L3436). This is a **correctness** issue, not a performance one — but failure mode is silent data corruption on CC 7.0+.

## Why

Divergence is cheap to detect and almost always cheap to fix; the reason it persists in real kernels is that the cost is **not visible from wall-clock alone** for kernels that are also memory-bound. The warp is often waiting on memory anyway, so a doubled instruction count on the compute side gets hidden.

The cost *becomes* visible when:

- The kernel is compute-bound (elementwise math, fused activations).
- Both branch paths together exceed the memory-wait window.
- Nested divergence multiplies: two 2-way divergent conditions can halve throughput twice, to 25 % SIMT efficiency (pitfall P5).

BP §13.2 (L1614-L1628) documents the compiler's own escape hatch: predication. The compiler converts branches to predicated instructions when both paths are short enough; the hardware then runs both paths **as predicated NOPs** instead of serializing them. Predicated instructions still occupy issue slots but don't split the warp, so the warp stays at full SIMT efficiency. Understanding predication is the core of this skill — most of the time, "fix the divergence" means "let the compiler predicate it."

## When to use

### S1. Align branches on warp boundaries when the condition depends on threadIdx

When a branch condition is `threadIdx.x < K` for some `K < blockDim.x`, and K is not a multiple of 32, every warp straddling the boundary diverges. BP §12.1.2 (L1396-L1400) gives the canonical pattern: **make the condition depend on `threadIdx.x / warpSize` instead**.

```cuda
// Bad: warp at threadIdx.x 0-31 splits 16/16
if (threadIdx.x < blockDim.x / 2) {
    produce(data);
} else {
    consume(data);
}

// Good: warp boundary aligned — every warp takes one branch cleanly
int warpId = threadIdx.x >> 5;           // threadIdx.x / 32
int numWarps = blockDim.x >> 5;
if (warpId < (numWarps >> 1)) {
    produce(data);
} else {
    consume(data);
}
```

**Caveat** (pitfall P6, measured in legacy sandbox): reorganizing threads into warps by role can unalign their memory accesses. If the two branch bodies access interleaved addresses, the warp-aligned version may kill coalescing. Measure before committing.

### S2. Keep short divergent bodies short so the compiler predicates them

When both bodies of an `if/else` are a few instructions, prefer branch-free idioms. The compiler then emits `selp` / `@p` predicated instructions with zero divergence overhead (BP §13.2, L1614-L1628).

```cuda
// Predication-friendly: no real branch generated
float y = (x > 0.0f) ? x : 0.0f;           // ReLU
float z = fmaxf(x, 0.0f);                   // same, even better
float c = condition ? a : b;                // scalar selp

// Hostile to predication: forces a real branch
if (condition) {
    do_expensive_thing_A();                 // function call blocks predication
} else {
    do_expensive_thing_B();
}
```

The threshold is documented as "a certain threshold" (BP §13.2) — empirically around 7 instructions in modern nvcc. Function calls, volatile memory ops, and side-effectful intrinsics block predication entirely. **When in doubt, inspect SASS**; `ptxas -v` reports predicated-vs-branched stats.

For the pattern-level rewrite library (ReLU / clamp / abs / conditional write / masked gather), see the sibling `branch-elimination` skill — this skill covers the *why and how much it costs*; that skill covers *the catalog of rewrites*.

### S3. Use warp vote functions for lane-uniform conditions

If every lane in a warp is likely to agree (a per-warp-uniform condition that *might* still diverge at some warps), `__any_sync` / `__all_sync` / `__ballot_sync` turn the check into a per-warp decision at one instruction:

```cuda
// Early-exit an entire warp if no lane has work
bool has_work = (work_indicator[tid] != 0);
if (!__any_sync(0xFFFFFFFFu, has_work)) {
    return;     // every lane returns together; no divergence
}

// Cheap bounds check for a multi-element-per-thread loop
if (__all_sync(0xFFFFFFFFu, tid + n_per_thread <= N)) {
    process_unchecked();   // full coalesced fast path
} else {
    process_with_bounds_checks();   // slow tail path
}
```

`__ballot_sync` returns the full 32-bit mask, enabling warp-local work counts (`__popc`) and lane-compaction (`__popc(mask & ((1u << laneId) - 1))`).

### S4. Data compaction / sorting for input-driven divergence

When divergence is caused by input heterogeneity that the kernel cannot control at launch time (sparse input, irregular graph nodes, variable-length work per thread), the mitigation lives **outside the kernel**:

- Sort the input so similar items land in the same warp.
- Bin by work-class and launch one kernel per bin.
- Run a compaction pass (`__ballot_sync` + `__popc` + stream compaction) to produce a dense homogeneous array, then launch the actual kernel on the dense array.

Pitfall P8 (legacy, measured): the compaction overhead itself can dominate if the original divergence is small or the problem size is small. Always **profile divergence severity first**: `smsp__thread_inst_executed_per_inst_executed.ratio` (average active lanes per warp execution) is the NCU metric — if it's near 32, no compaction is needed.

## When NOT to use

- **When the kernel is memory-bound and divergence is already hidden by memory latency.** Rewriting for predication is pure churn with no wall-clock benefit. Profile first — if `stall_long_scoreboard` dominates and SIMT efficiency is >50 %, the divergence is not the bottleneck.
- **When "fixing" divergence breaks coalescing.** Re-grouping lanes by branch role can unalign adjacent threads' memory addresses. Pitfall P6 (measured on legacy): aligning branches to warp boundaries destroyed coalescing in a gather-shaped kernel and regressed wall-clock by ~15 %.
- **When the branch is on a truly warp-uniform scalar.** If the condition is a kernel argument or a block-wide broadcast that never differs across lanes, there is no divergence — the branch emits a `bra.uni` and costs a single instruction. Leave it alone.
- **When pursuing predication removes side effects that matter.** Predicated loads/stores never evaluate addresses or read operands (BP §13.2), which is correct for pure-arithmetic predication but matters when the "other branch" had a memory read whose address computation had to happen. Check SASS for the actual shape.

## Measured Characteristics

Measured on H200-SXM (sm_90a, CUDA 12.9, driver 570.124.06) using [sources/experience/hw-probes/warp-divergence-cost/](../../../sources/experience/hw-probes/warp-divergence-cost/) — per-lane predicate with probability `p` of taking path B; compute-bound workload (~1024 FMAs per lane per kernel, 1 M lanes, Compute SM throughput ~89 %). Unlocked clock logged at 1980 MHz. Full record: [sources/experience/hw-probes/warp-divergence-cost/2026-04-22-warp-divergence-cost.md](../../../sources/experience/hw-probes/warp-divergence-cost/2026-04-22-warp-divergence-cost.md).

| Kernel               | p=0.00 | p=0.25 | p=0.50 | p=0.75 | p=1.00 | Slowdown vs own p=0 |
| -------------------- | -----: | -----: | -----: | -----: | -----: | ------------------: |
| `branch_variant`     | 0.0418 | 0.0760 | 0.0760 | 0.0760 | 0.0416 |        **1.82×**    |
| `predicated_variant` | 0.0766 | 0.0819 | 0.0820 | 0.0819 | 0.0766 |              1.07×  |
| `warp_uniform`       | 0.0409 | 0.0415 | 0.0415 | 0.0415 | 0.0408 |              1.01×  |

Units: median ms across 20 iterations (5 warmup).

Key measured findings:

- **Divergence cost kicks in at ANY p ∈ (0, 1), not just p=0.5.** With 32-lane warps, the probability any single warp avoids divergence at p=0.25 is (0.75)³² ≈ 0.01 %; essentially every warp diverges and pays the full **1.82× slowdown**. The "cost curve" folklore (smooth ramp from 1× at p=0 to 2× at p=0.5) is a per-warp-averaged measure, not what wall-clock shows.
- **Predication is flat across p** (1.00×–1.07×) but always pays the "both bodies execute" cost (absolute 0.0766 ms vs the branch's 0.0418 ms at p=0). Trade: predication eliminates p-dependence at a fixed ~1.83× compute cost. Use when p is unpredictable per-lane.
- **`warp_uniform` (S3) at 1.01× is the upper bound** on this technique's benefit — achievable only when the "pessimistic body" cost equals the optimistic body cost. In production kernels where path B is materially heavier than path A, `warp_uniform` pays the full path-B cost on every warp that has any lane needing it.
- **NCU `smsp__thread_inst_executed_per_inst_executed.ratio` stays at 32 across all p** for the branching kernel — this is a new pitfall (P10): the SIMT-efficiency metric that pedagogy recommends for divergence hunting can look perfect at 32/32 even when the warp is actually serializing. The authoritative signal is wall-clock at constant Compute(SM) Throughput (~89 %) — doubled wall-clock at unchanged throughput means doubled total instructions issued.
- **Divergence cost is invisible in memory-bound kernels.** A parallel run of the same probe at N=16 M and only 64 FMAs per path showed zero slowdown across p on all three variants (memory latency hiding it). This is the empirical basis for skill §"When NOT to use" item 1.

See probe record for NCU breakdown, the memory-bound vs compute-bound comparison, and the full interpretation.

## Principles

1. **Divergence cost is proportional to the LONGER path × the number of distinct paths taken.** A 2-way 50/50 branch doubles the cost of one side; a 4-way branch can quadruple it.
2. **Predication is the compiler's escape hatch; stay short and side-effect-free inside branches.** Function calls, volatile writes, and printf block predication entirely.
3. **Sub-warp branching is always worse than no branching; warp- aligned branching is usually better than sub-warp branching but not always.** Check the memory coalescing side before committing to S1 (pitfall P6).
4. **ITS (Volta+) requires explicit `__syncwarp()` after any divergent region followed by warp-synchronous code.** This is a correctness rule, not a performance one, but the performance cost is negligible — always include the barrier.
5. **Profile first.** If SIMT efficiency > 75 % and the kernel is memory-bound, divergence is not the bottleneck; rewriting it produces churn without wall-clock win.

## Open questions

- Q1. What is the exact predication threshold for `nvcc 12.9` on sm_90a? Legacy guidance says "approximately 7 instructions"; the probe opens with a compile-time sweep to land the modern number.
- Q2. How does divergence cost interact with Hopper's thread-block- cluster scheduling? Currently no data; follow-up probe blocked on `wiki/nvidia/hardware/thread-block-cluster/` bootstrap (bucket F).
- Q3. For the "early-exit via warp vote" idiom (S3), what fraction of warps-with-zero-work is needed for the vote-early-exit to pay back its own `__any_sync` cost? Open follow-up: `sources/experience/hw-probes/warp-vote-early-exit-amortization/`.

## Legacy references

- `legacy_sandbox_path`: `KernelPilot/optimization/latency/warp-divergence/skill.md`. Legacy kept five sub-skills (S1-S5); this port collapses S1+S4 of the legacy version into the measured quadrant (S1 warp-align, S4 data-compaction), keeps S2 predication / S3 warp-vote, and promotes S5 `__syncwarp()` from "skill" to a pitfall (P1 in the sibling pitfalls file) because it is a **correctness** fix, not a throughput optimization.
- Legacy L3 sandbox findings P6-P9 are retained in `pitfalls.md` pending H200 re-measurement.
- **Related but distinct**: `wiki/nvidia/foundations/compute/branch-elimination/` (pending migration) covers the catalog of branch-rewriting patterns; this skill covers the divergence cost model and measurement. Cross-ref is one-line in each skill's body.
