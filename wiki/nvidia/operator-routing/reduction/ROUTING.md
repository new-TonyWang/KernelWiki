---
title: Reduction Pattern -- Skill Routing
pattern_class: cuda-core
op: reduction
status: draft
id: routing-reduction-ROUTING
type: operator-routing
vendor: nvidia
operator: reduction
---
# Reduction Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a custom reduction kernel, in recommended application order. Only skills that currently exist under `wiki/nvidia/foundations/` (with a completed `skill.md`) are listed.

Skills that are conceptually relevant but not yet built (e.g., vectorized-access, ilp, register-pressure) are NOT included. Do not reference them until their `skill.md` files exist.

## Applicable skills

### 1. Global Memory Coalescing

- **Skill path**: `wiki/nvidia/foundations/memory/coalescing/`
- **Why it matters for reduction**: the input-load phase of any reduction kernel reads the full input array from global memory. Coalesced (stride-1) loads are essential for achieving peak memory bandwidth. On H200, non-coalesced (stride-32) access is 2.85x slower than coalesced access (measured).
- **When to apply**: always. Every reduction kernel's load phase should be verified for coalescing. The standard pattern `tid = blockIdx.x * blockDim.x + threadIdx.x; val = input[tid]` is coalesced. Watch for multi-dimensional reductions where the reduction axis may not be the contiguous dimension.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Reads/writes memory with stride > 1 per thread."

### 2. Warp Primitives (Shuffle Reduction)

- **Skill path**: `wiki/nvidia/foundations/compute/warp-primitives/`
- **Why it matters for reduction**: warp-level butterfly reduction via `__shfl_xor_sync` (or `__shfl_down_sync`) replaces shared-memory tree reduction for the final 32-element stage. On H200, a full 5-step butterfly reduction takes ~145 cycles (measured), eliminating shared-memory round-trips entirely within a warp.
- **When to apply**: always for the intra-warp reduction stage. The canonical block-level reduction pattern (from cuda-samples reduce4+) uses shared memory for inter-warp communication but warp shuffles for the intra-warp stage.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Thread 0 (or lane 0) does a serial reduction."

### 3. Shared Memory Cache (prerequisite for the block-level tree)

- **Skill path**: `wiki/nvidia/foundations/memory/shared-memory-cache/`
- **Why it matters for reduction**: the block-level reduction phase (§1 Q4 "single block" branch in `INDEX.md`) uses shared memory as the inter-warp staging area — warp leaders write their partial sums to `smem[warpId]`, then a final warp reduces those 32 partials via shuffles. For grid-level reductions, each block also optionally uses a shared-memory tile to cache strides of the input before reducing. The sibling skill's sub-skill S1 (tile reuse) + S2 (coalesce-then-reorder) both apply.
- **When to apply**: always, for reductions over > 32 elements (i.e., anything that leaves a single warp). For pure warp-level reductions (`<= 32` elements), smem is not needed — the final shuffle suffices.
- **Specific guidance for reduction**:
  - The classic `__shared__ float smem[32];` for warp-leader partials is bank-conflict-free on stride-1 lane access; no padding required.
  - When caching a multi-element-per-thread tile before reducing (Brent's theorem pattern), declare `__shared__ float tile[TILE][TILE+1]` if any column access is planned — measured on H200: unpadded `[32][32]` → 16.3 M bank conflicts, padded → 33 K (488× drop; see `sources/experience/hw-probes/smem-tile-reuse/`).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Block-level reduction without smem staging / with redundant global loads."

### 4. Barrier Optimization (scope narrowing + async-barrier overlap)

- **Skill path**: `wiki/nvidia/foundations/sync/barrier-optimization/`
- **Why it matters for reduction**: the block-level reduction tree uses `__syncthreads()` between warp-leader writes to smem and the final warp's read of those partials. On H200 sm_90a, each `__syncthreads()` costs ~31–62 ns depending on block size (measured: `sources/experience/hw-probes/barrier-cost/`). For reductions that loop over many tiles with a `__syncthreads()` per tile, this cost compounds.
- **When to apply**:
  - **S1 narrow scope**: if the reduction is warp-local (N ≤ 32 per reduction instance), drop `__syncthreads()` entirely — the `_sync` shuffle intrinsics embed their own warp sync. Saves ~2× the per-barrier cost per iteration.
  - **S1 narrow scope**: if the reduction is block-local but only a subset of warps need to agree (e.g. after warp 0 has written the final smem slot, other warps don't need to wait), use a named `bar.sync` with subset count or `cuda::barrier` with `expected_count < blockDim.x`.
  - **S2 arrive/wait split (async barrier)**: only beneficial if the post-barrier consumer warp has independent work to do BEFORE needing the barrier result. For a standard block-level reduction the consumer depends immediately on the barrier, so S2 does NOT help here — pitfall P6 shows bare mbarrier cost is 1.6–2.2× a `__syncthreads`.
- **When NOT to apply**: grid-stride reduction kernels with just one `__syncthreads()` per block are already barrier-light — the next optimization target is elsewhere (NCU `stall_barrier` < 5 %).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Reduction kernel with `stall_barrier` above ~15 %."

### 5. Bank-Conflict Avoidance

- **Skill path**: `wiki/nvidia/foundations/memory/bank-conflict/`
- **Why it matters for reduction**: block-level reduction kernels that use shared memory (skill 3 above) for the tree-reduction phase are susceptible to bank conflicts. Interleaved addressing (cuda-samples reduce1) creates conflicts; sequential addressing (reduce2) avoids them. When warp leaders write partial sums to `smem[warpId]`, 32-way conflicts can occur if all warps' lane-0 threads hit bank 0.
- **When to apply**: whenever the kernel uses `__shared__` memory for inter-warp communication or the reduction tree. Check the access pattern: if consecutive active threads access addresses with a stride that is a multiple of 32, apply the +1 padding trick or redesign the index mapping.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "Narrow reduction into smem with heavy bank contention."

### 6. Vectorized Access

- **Skill path**: `wiki/nvidia/foundations/memory/vectorized-access/`
- **Why**: loading float4 in the input phase increases bytes-in-flight per thread. On H200, float4 vs scalar shows 2.59× bandwidth gain.
- **When**: input array is 16-byte aligned and N is divisible by 4.

### 7. Warp Divergence (reduction-tail `if (tid < offset)` shapes)

- **Skill path**: `wiki/nvidia/foundations/compute/warp-divergence/`
- **Why it matters for reduction**: the classical tree-reduction tail has a stride halving loop with `if (tid < offset)` guards. Unguarded, the active lane count halves each iteration — and the inactive lanes still occupy the warp. Fortunately nvcc typically predicates the body (short: one load + one add + one store to smem), so the divergence is a non-issue for stride-halving tails. It becomes an issue only when the per-lane body grows (e.g. reduction with fused fmad + reciprocal, or reduction over struct fields) and falls outside nvcc's predication threshold.
- **When to apply**:
  - Keep the reduction-tail body short and arithmetic-only so nvcc predicates it; let the last few iterations converge within a single warp using shuffles instead of smem (skill #2 warp-primitives already replaces the tail with branch-free shuffles).
  - For reductions with a fused compute (e.g. softmax's exp + sum-reduction + divide-by-sum), the reduction kernel is typically compute-bound at the compute step and memory-bound at the tail — measured divergence cost on the tail is usually invisible. Verify before rewriting.
- **When NOT to apply**: pure sum-reduction kernels using shuffle- based warp reduction (the canonical H200 pattern from `atomic-reduction` skill). These have no divergent tail — the warp-reduction folds 32 elements into lane 0 with five shuffles and zero divergence.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — "Reduction-tail body too large for predication."

### 8. Instruction-Level Parallelism (ILP)

- **Skill path**: `wiki/nvidia/foundations/compute/ilp/`
- **Why**: multiple independent accumulator chains hide FMA pipeline latency (4 cycles on H200). 4-acc achieves 3.97× throughput vs 1-acc.
- **When**: the reduction accumulation loop processes one element at a time; unrolling with multiple accumulators helps.

### 9. Register Pressure Management

- **Skill path**: `wiki/nvidia/foundations/memory/register-pressure/`
- **Why**: vectorized access + ILP increase register usage, potentially reducing occupancy. On H200, forcing 32 regs caused 4.84× slowdown from spilling.
- **When**: after applying skills 4-5, check regs/thread via `-Xptxas=-v`.

### 10. Compiler Hints

- **Skill path**: `wiki/nvidia/foundations/compute/compiler-hints/`
- **Why**: `__launch_bounds__` controls register allocation ceiling; `#pragma unroll` exposes ILP.
- **When**: tuning the final kernel after functional correctness is established.

### 11. Async Copy (LDGSTS Prefetching)

- **Skill path**: `wiki/nvidia/foundations/memory/async-copy/`
- **Why**: for iterative / compute-heavy reduction kernels, LDGSTS async copies to shared memory increase bytes-in-flight without consuming registers. Little's Law: H200 needs ~64 KiB/SM for >90% BW.
- **When**: reduction kernel is iterative (processes multiple tiles per block) AND compute per element is non-trivial (e.g., involves sqrt/exp). Simple sum/max over a flat array does NOT benefit (GTC25-S72683 showed 10% regression for trivial `a*b`).
- **Caveat**: evaluate carefully — async copies add code complexity and only help when "Stall Long Scoreboard" is the dominant bottleneck in ncu.

### 12. Memory Ordering (correctness)

- **Skill path**: `wiki/nvidia/foundations/sync/memory-ordering/`
- **Why**: grid-level reductions with `atomicAdd` or `__threadfence` patterns require correct memory ordering to avoid data races.
- **When**: multi-block reduction that communicates via global memory atomics or flags.

### 13. Atomic Reduction Contention Control (critical at grid level)

- **Skill path**: `wiki/nvidia/foundations/sync/atomic-reduction/`
- **Why it matters for reduction**: once the block-level reduction produces a partial sum, the grid-level combining step uses global atomics. A naive kernel that has **every thread** atomicAdd into a single global scalar is both contention-pathological and numerically wrong (on H200, 33M-element FP32 reduction measured 248× slower AND `rel_err = 5.03e-02` vs double reference -- the running sum saturates the FP32 mantissa, see skill's pitfall P7). The S1 hierarchical pattern commits **one atomic per block** after in-warp shuffle + shared-memory fan-in; S1 + grid-stride loop (fewer, fatter blocks) lifts this to 1498× and 75% DRAM SoL on H200.
- **When to apply**: every multi-block (`>1024` elements per instance) reduction that terminates with a global atomic. Required for the `>1024` branch of Q4 in `INDEX.md`. Pair with skill #9 (memory-ordering) for scope / ordering correctness, and with skill #2 (warp-primitives) since S1's inner loop uses `__shfl_down_sync`.
- **Measured impact**: 247.8× speedup of S1 over naive; 1498× for S1+grid-stride. See `sources/experience/hw-probes/atomic-reduction-contention/2026-04-20-atomic-reduction.md`.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` -- "many threads contending on the same atomic address" / high `stall_long_scoreboard` on `RED`/`ATOM`.

### 14. L2 Access Policy (hitRatio-tuned persisting window)

- **Skill path**: `wiki/nvidia/foundations/memory/l2-access-policy/`
- **Why it matters for reduction**: multi-stage reduction pipelines that re-read a hot buffer across kernels (e.g. grid-level combine that touches the per-block partials buffer twice, or a softmax sum-then-divide shape that re-reads the input between the two passes) can pin that buffer into H200's 37.5 MiB L2 set-aside via `cudaStreamAttributeAccessPolicyWindow`. Measured on H200 at WS = 80 MiB / set_aside = 37.5 MiB: `hitRatio = set_aside / WS` recovers **+17.7%** effective BW; `hitRatio = 1.0` silently degenerates to no-op.
- **When to apply**:
  - The hot buffer is repeatedly read **across more than one kernel launch** on the same stream. A single kernel's grid-stride pass does not trigger eviction pressure worth pinning against (measured: WS = 40 MiB single-kernel soft-null).
  - WS approaches or exceeds the H200 set-aside cap of 37.5 MiB. Below that, normal LRU keeps the buffer hot for free.
  - Set the `hitRatio` formula: `hitRatio = min(1.0f, (float)set_aside / (float)ws_bytes)`. Do **not** use `hitRatio = 1.0` when WS > set_aside — measured as a silent null on H200.
- **When NOT to apply**:
  - Single-kernel MVPs with no competing workload on the L2 (skill's P8 directly applies to reduction microbenchmarks).
  - MIG partitions (set-aside reservation is a silent no-op; skill P3).
  - Buffers already below set-aside with no concurrent pressure — policy is overhead without benefit (measured neutral at WS = 4 / 40 MiB).
- **Measured impact**: +17.7% effective BW at WS = 80 MiB on H200 when `hitRatio` is tuned. See `sources/experience/hw-probes/l2-residency/2026-04-23-l2-residency.md`.
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — multi-pass reductions where the second kernel's DRAM throughput stays at 100% Roofline even though the same buffer was just fully read by the first kernel (no L2 reuse).

### 15. Branch Elimination (max/min reductions; tail already predicated)

- **Skill path**: `wiki/nvidia/foundations/compute/branch-elimination/`
- **Why it matters for reduction**:
  - **Max-reduction / min-reduction / argmax / argmin** kernels have a comparator at every reduction step. Using `fmaxf(a, b)` / `fminf(a, b)` in place of `(a > b) ? a : b` saves **one SASS op per comparison** on H200 (FMNMX vs FSETP+FSEL). Over a log2(N)-depth tree this compounds.
  - **Tree-reduction tail** (`for (offset = blockDim/2; offset > 0; offset >>= 1) if (tid < offset) smem[tid] += smem[tid + offset];`) contains an `if (tid < offset)` guard on each step. nvcc 12.9 already predicates this simple-body guard to `FSEL`; `cuobjdump --dump-sass` confirms no inner-loop `BRA`. No rewrite needed.
  - **Clamped-sum reductions** (e.g. `sum(max(x - threshold, 0))` as in quantile estimators or hinge loss): use `fmaxf` for the clamp — 1 SASS op vs 2 for the equivalent `if/else`.
- **When to apply**:
  - Max/min-reduction: prefer `fmaxf`/`fminf` over explicit ternary. Measured saving 1.56× per comparison on H200.
  - Argmax: use `fmaxf` for the value comparison but still need an `if`/`selp` to carry the index — compiler predicates the index select automatically. Do NOT rewrite as arithmetic.
  - Sum-of-clamped: `sum += fmaxf(x - t, 0.0f)` — one FMNMX per element vs two-op guarded add.
- **When NOT to apply**:
  - Do NOT rewrite `if (tid < offset) smem[tid] += smem[tid + offset]` as arithmetic — it is already predicated by nvcc, and the arithmetic form would add FADD/FMUL ops (measured 1.27× slower in the branchless-patterns probe cond-arith variant).
  - Do NOT hand-code bit tricks for argmax sign handling — `copysignf` or `fabsf` + a predicated index carry is cleaner.
- **Measured on H200**: 1.56× per comparison from switching `ternary` to `fmaxf` in a max-reduction tree (branchless-patterns probe Group A).
- **Relevance to bottleneck triage**: Q4 in `reasoning/bottleneck-triage.md` — max/min reduction kernel where hot path shows `FSETP + FSEL` pairs instead of `FMNMX`. Correct response: replace ternary with `fmaxf`/`fminf`.
