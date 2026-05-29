---
title: Indexing Pattern -- Skill Routing
pattern_class: cuda-core
op: indexing
status: draft
id: routing-indexing-ROUTING
type: operator-routing
vendor: nvidia
operator: indexing
---
# Indexing Pattern -- Skill Whitelist

This file lists the optimization skills applicable to a custom indexing kernel, in recommended application order. Only skills that currently exist under `knowledge/30-skill/` (with a completed `skill.md`) are listed.

---

## Applicable skills

### 1. Global Memory Coalescing (critical)

- **Skill path**: `30-skill/memory/coalescing/`
- **Why it matters for indexing**: indexing kernels are dominated by indirect memory access. In a gather kernel, the output write is coalesced but the source read is non-coalesced (random). In a scatter kernel, the input read is coalesced but the destination write is non-coalesced. Understanding the coalescing cost is essential for predicting and improving performance. On H200, non-coalesced (stride-32) access is 2.85x slower than coalesced access (measured).
- **When to apply**: always. Every indexing kernel should be inspected to ensure that the coalesced side (output for gather, input for scatter) maintains stride-1 access. For index_select along dim=0, both sides can be coalesced per-slice.
- **Specific guidance for indexing**:
  - Sort the index array before the gather/scatter kernel to improve spatial locality on the non-coalesced side. Sorted indices increase L2 cache hit rate because adjacent threads access nearby source addresses.
  - For 2-D gather along the non-contiguous dimension, consider transposing the source tensor so that the gather dimension becomes contiguous.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Reads/writes memory with stride > 1 per thread."

### 2. Layout Transform (pre-gather fix when source is stride-unfriendly)

- **Skill path**: `30-skill/memory/layout-transform/`
- **Why it matters for indexing**: the `src[idx[i]]` random-access pattern of gather is already non-coalesced; that's the pattern's defining cost. But when the source buffer is *also* AoS-shaped and the gather touches one field per indexed element, the waste multiplies — each random load fetches 32 B of struct bytes but uses 4 B. Sub-skill S1 (AoS → SoA) collapses the struct tax into a stride-1 problem before the gather even runs. For scatter-add the same logic applies symmetrically on the destination side.
- **When to apply**:
  - **AoS source**: apply S1 if >= ~3 gather/scatter kernels will consume the SoA form (break-even is measured 3.65 on a 24-B struct). If only one kernel consumes the split, skip — the conversion kernel is itself more expensive than a single non-coalesced gather.
  - **Non-128-B-aligned 2-D source**: apply S2 (`cudaMallocPitch`) before the gather kernel — otherwise the first-column coalescing penalty stacks on top of the random-index penalty.
- **Specific guidance for indexing**:
  - Gather with sorted indices (Q4a YES branch in `INDEX.md`) benefits most from SoA because the L2 locality increase on the stride-1 SoA layout compounds with the sorted-index L2 hit rate gain.
  - For `torch.embedding` / embedding-lookup style kernels where the inner dim is already stride-1, layout transform is not needed — the gather already reaches HBM peak on coalesced load-slice.
- **Measured on H200**: AoS-one-field gather is ~2× slower than SoA even with stride-1 access patterns; on random-index gather the penalty compounds. See `80-experience/hw-probes/aos-vs-soa/`.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — "Struct-shaped source amplifies the random-access penalty."

### 3. Vectorized Access (conditional)

- **Skill path**: `30-skill/memory/vectorized-access/`
- **Why it matters for indexing**: when indices select contiguous ranges (index_select on dim=0, embedding lookup), the copy of each selected row is a contiguous memcpy that benefits from float4 vectorized loads/stores. On H200, float4 achieves 2.59x bandwidth over scalar loads.
- **When to apply**: only when the operation moves contiguous chunks of data (not random-element gather). Specifically:
  - index_select along dim=0 where each selected slice is >= 16 bytes.
  - Embedding lookup where the embedding dimension is a multiple of 4 (for float32) or 8 (for float16/bfloat16).
  - Scatter of full rows.
- **When NOT to apply**: random-element gather/scatter where each thread accesses a single element at an unpredictable address. Vectorized loads at random addresses waste 3/4 of the fetched data.

### 4. Warp Primitives

- **Skill path**: `30-skill/compute/warp-primitives/`
- **Why it matters for indexing**: useful for two sub-patterns:
  1. **Warp-level topk**: a small topk (k <= 32) can be implemented as a warp-level tournament using `__shfl_xor_sync` / `__shfl_down_sync` to find the k largest values without shared memory.
  2. **Warp-level coordination for scatter conflicts**: when multiple lanes in a warp scatter to the same destination, `__ballot_sync` + `__popc` can detect duplicates and serialize or aggregate the conflicting writes within the warp before issuing a single atomic.
- **When to apply**: topk with k <= 32, or scatter-add/scatter-max when intra-warp index collisions are expected.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Thread 0 (or lane 0) does a serial reduction."

### 5. Warp Divergence (masked gather / conditional scatter)

- **Skill path**: `30-skill/compute/warp-divergence/`
- **Why it matters for indexing**: gather with a validity mask (`if (mask[i]) dst[i] = src[idx[i]]`) and scatter-with-predicate are natural divergence sources — adjacent lanes have independent mask values. If the gather/scatter body is longer than nvcc's predication threshold (function call, nested indexing expression), the branch becomes a real serialization. Measured on H200 (compute-bound proxy): divergent-branch kernel is **1.82× slower** than the uniform-warp baseline at any p ∈ (0, 1).
- **When to apply**:
  - Collapse masked reads into `dst[i] = mask[i] ? src[idx[i]] : fallback` (single selp, no branch). For writes, prefer `dst[i] = mask[i] ? new : dst[i]` over a guarded `if (mask[i]) dst[i] = new;`.
  - For very skewed mask distributions (most lanes inactive), use `__any_sync(0xFFFFFFFFu, mask[i])` to early-exit entire inactive warps (S3 in the skill).
  - For mask-with-reorder (compact active indices into a dense prefix then gather), fall through to the skill's S4 data-compaction guidance — note pitfall P8: compaction overhead can exceed the divergence it was meant to fix.
- **When NOT to apply**: gathers with random-access patterns already dominated by L2 miss latency are typically memory-bound — divergence cost is invisible, and code churn to predicate doesn't move wall-clock.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — "Masked gather/scatter with non-predicable body on compute-bound kernel."

### 6. Memory Ordering (correctness for scatter with conflicts)

- **Skill path**: `30-skill/sync/memory-ordering/`
- **Why it matters for indexing**: scatter operations where multiple source elements map to the same destination index require atomic operations (atomicAdd, atomicMax, etc.) or careful memory ordering to avoid data races. Without atomics, concurrent writes to the same address produce undefined results.
- **When to apply**: any scatter kernel where index collisions are possible (scatter_add, scatter_max, histogram-style scatter).
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` -- "Data race or inconsistent cross-thread reads."

### 7. Bank-Conflict Avoidance (conditional)

- **Skill path**: `30-skill/memory/bank-conflict/`
- **Why it matters for indexing**: when a topk or index_select kernel stages data through shared memory (e.g., block-level radix select uses shared-memory histograms, or index_select tiles a source matrix through shared memory to coalesce strided access), bank conflicts can degrade shared memory throughput.
- **When to apply**: only when the kernel uses `__shared__` memory. Simple gather/scatter kernels that operate entirely in registers and global memory do not need this skill.

### 8. Instruction-Level Parallelism (ILP)

- **Skill path**: `30-skill/compute/ilp/`
- **Why it matters for indexing**: for gather kernels with a grid-stride loop, processing multiple elements per thread (with independent load-store chains) can hide memory latency. On H200, 4-accumulator chains achieve 3.97x throughput vs 1-accumulator.
- **When to apply**: when the gather kernel processes many elements per thread and the loads are independent. Less effective when loads are already latency-bound by random access (cache misses dominate over pipeline latency).

### 9. Compiler Hints

- **Skill path**: `30-skill/compute/compiler-hints/`
- **Why it matters for indexing**: `__restrict__` on pointer arguments enables the compiler to assume no aliasing between src, idx, and dst, which may allow better scheduling. `__launch_bounds__` controls register allocation.
- **When to apply**: tuning phase after functional correctness is established.

### 10. Register Pressure Management

- **Skill path**: `30-skill/memory/register-pressure/`
- **Why it matters for indexing**: gather/scatter kernels are generally register-light (few live variables), but ILP unrolling or vectorized access can increase register usage. Monitor with `-Xptxas=-v`.
- **When to apply**: after applying skills 2 and 6, if occupancy drops below 50%.

### 11. Atomic Reduction Contention Control (scatter-add / histogram)

- **Skill path**: `30-skill/sync/atomic-reduction/`
- **Why it matters for indexing**: scatter-add (`dst[idx[i]] += val[i]`) and histogram (`hist[data[i]]++`) are conflict-resolving scatters. When the index distribution is skewed -- a handful of destinations receiving most of the atomics -- contention on those "hot" addresses dominates latency. Two mitigations from this skill apply directly: **S1 hierarchical fan-in** (every block accumulates a private copy, then commits one atomic per block per bucket) and **S4 shared-memory atomics** (block-local histogram in `__shared__` that is only flushed to global memory at the end of the kernel). For a 256-bucket histogram, S4 issues 256 global atomics per block regardless of input size, versus one per sample in the naive version.
- **When to apply**:
  - **Small destination set** (histogram, bincount, top-k accumulation) where `sizeof(dst_bucket) * num_buckets` fits in shared memory -- use S4 shared-memory atomics.
  - **Large destination set with skewed index distribution** (scatter-add with a few hot indices) -- use S1 hierarchical: sort or group the indices first so each block's writes concentrate on few addresses, then fan in with one global atomic per (block, key).
  - **Uniform / random index distribution** -- contention is already low; the naive `atomicAdd(&dst[idx[i]], val[i])` is fine, and this skill adds overhead without gain.
- **Measured impact (reduction-to-scalar variant)**: on H200, S1 gave 247.8x speedup over naive per-thread atomics; grid-stride + S1 gave 1498x. Scatter-add wins will be smaller (hot-spot skew determines contention), but the pattern is identical.
- **Pair with**: skill 5 (memory-ordering, for thread-scope / memory- order correctness around the atomic) and, for FP16/BF16 gradient scatter-add, the `atom.add.noftz.f16` caveat in atomic-reduction pitfall P5.

### 12. L2 Access Policy (hot-index residency)

- **Skill path**: `30-skill/memory/l2-access-policy/`
- **Why it matters for indexing**: the canonical indexing workload that benefits most is **lookup-table reads** where the table is hot but the index stream is not: `out[i] = lut[idx[i]]`. If the `lut` buffer is in the 37.5 MiB–128 MiB range **and** the kernel is invoked repeatedly on a stream that also runs other kernels (distinct weights, activations, etc.), tagging `lut` with a persisting `accessPolicyWindow` keeps it resident in L2 across those interleaved launches. Measured on H200 at WS = 80 MiB with tuned `hitRatio = set_aside / WS`: **+17.7%** effective BW over no policy.
- **When to apply**:
  - Indexing kernel reads a **shared** lookup/embedding/weight table that is reused by multiple subsequent kernels on the same stream, and that table exceeds the 37.5 MiB H200 set-aside.
  - Always set `hitRatio = min(1.0, (float)set_aside / (float)table_bytes)` — the naive `hitRatio = 1.0` is a measured silent null on H200 at WS > set-aside.
  - Reserve set-aside with `cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, set_aside)` before the first kernel launch. Skipping this is the most common silent failure mode (skill pitfall P11).
- **When NOT to apply**:
  - Table fits easily in L2 naturally (< 40 MiB on H200) — policy adds overhead without benefit (measured neutral at WS = 4 / 40 MiB when no competing pressure exists).
  - One-shot indexing kernels with no follow-up — the pinning benefit only manifests across more than one launch.
  - MIG partitions (skill P3: `cudaDeviceSetLimit` silent no-op).
- **Specific guidance for indexing**:
  - For embedding lookup (`embed[tok_id]`) in LLM inference: pin the embedding table (often 200 MiB–1 GiB). Use `accessPolicyMaxWindowSize` cap on H200 = 128 MiB; window covers only a prefix, so `num_bytes = min(embed_bytes, 128 MiB)` and accept that only the first 128 MiB gets the policy tag.
  - For one-hot / gather-one-field: the input **indices** are streamed (missProp = Streaming), the **source** table is pinned (hitProp = Persisting); this asymmetric tag is the canonical shape for indexing.
- **Measured impact**: +17.7% effective BW at WS = 80 MiB on H200 when `hitRatio` is tuned. See `80-experience/hw-probes/l2-residency/2026-04-23-l2-residency.md`.
- **Relevance to bottleneck triage**: Q2/Q4 in `70-reasoning/bottleneck-triage.md` — indexing kernels with high `stall_long_scoreboard` or persistent L2 miss rate on the table buffer across launches.

### 13. Cache Load Hints (source-table reads; one measurably-bad trap)

- **Skill path**: `30-skill/memory/cache-load-hints/`
- **Why it matters for indexing**: gather / scatter kernels touch two buffers with very different access patterns — the **index stream** `idx[]` (sequential, streamed) and the **source table** `lut[]` (random, hot if reused). The legacy KB's advice to tag the source with `__ldg` is redundant on H200 (measured 0.1-0.4% vs default with `const __restrict__`). What **is** measurably harmful: reaching for `__ldcg` on the source table under the "bypass L1 for read-only data" legacy framing. When the table is L2-resident and re-accessed by the same SM (the common indexing case), bypassing L1 costs **2.26×** (skill's L2 regime, measured).
- **When to apply**:
  - Declare indexing-kernel pointers with `const __restrict__` on the source / index / output that are truly read-only or non-aliased. The compiler already emits the read-only cache path; no explicit `__ldg` needed.
  - For scatter kernels that read both `idx[]` and `src[]` before writing `dst[]`, keep all reads on default cache operators; the mixed access pattern is what L1 is for.
- **When NOT to apply**:
  - Never use `__ldcg` on the source table of a gather. The "L1 only holds a few KB per SM, bypass to save L1 for others" reasoning is legacy Kepler-era; on H200 the L1/TEX unit is unified and staging every read through it is the fast path.
  - `__ldcs` on the index stream looks superficially right ("streaming, one-shot"), but measured on H200 as a null vs default in the un-contended case. Skip it.
- **Measured on H200**: `__ldg ≡ default` (within 0.5%); `__ldcg` at L2-resident reuse is 2.26× slower. See `80-experience/hw-probes/cache-hint/2026-04-23-cache-hint.md`.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — an indexing kernel where agent annotated `__ldcg(&src[idx])` and reports "slower than default". Correct response: revert to default load. Pair with skill 12 (l2-access-policy) if the table genuinely needs cross-launch pinning.

### 14. Branch Elimination (masked gather / conditional scatter)

- **Skill path**: `30-skill/compute/branch-elimination/`
- **Why it matters for indexing**: indexing kernels frequently have predicates — validity masks (`if (mask[i]) dst[i] = src[idx[i]]`), scatter-with-guard (`if (keep[i]) dst[idx[i]] = val[i]`), conditional bounds (`idx = max(0, min(idx, N-1))`). On H200 sm_9.0a, the simple-body forms of these predicates compile to `FSEL` (branchless SASS) automatically; the measurable wins come from using single-op intrinsics (`fmaxf` / `fabsf` / `fminf`) where they apply.
- **When to apply**:
  - **Bounds clamping** on indices: `idx = min(max(idx, 0), N-1)` compiles to two `IMNMX` (integer min/max) — single-op per clamp. Prefer this over branchful bounds checks.
  - **Masked load-with-fallback**: `dst[i] = mask[i] ? src[idx[i]] : fallback;` emits one FSEL; **do not** rewrite as `dst[i] = src[idx[i]] * mask[i] + fallback * (1 - mask[i])` which emits 4 FP ops and is 1.27× slower (measured branchless-patterns probe's cond-arith variant).
  - **Conditional scatter-add**: use `if (mask[i]) atomicAdd(&dst[idx[i]], v)` directly — the compiler preserves the predicate on the atomic. Rewriting the atomic value as `mask[i] * v` to "always atomic-add" is incorrect on its face (adds zero noisily) and adds ops.
- **When NOT to apply**:
  - Do not rewrite gather-with-mask as arithmetic; the measured 1.27× slowdown from cond-arith pattern applies here too.
  - Do not hand-code bit tricks for abs of indices — `abs(i - pivot)` using `fabsf` after casting is simpler and faster. For integer `abs`, the compiler does the right thing with the standard `abs(int)`.
  - For data-dependent branches with heavy bodies (calling a helper, doing a store), the compiler emits a real `BRA` and warp-divergence cost applies — see the `warp-divergence` skill for measurement and mitigation (warp-vote, data compaction).
- **Measured on H200**: `fmaxf` / `fabsf` 1.56× / 2.36× faster than the `if/else` form; arithmetic simulation of select is 1.27× *slower* than `if/else`. See `80-experience/hw-probes/branchless-patterns/2026-04-23-branchless-patterns.md`.
- **Relevance to bottleneck triage**: Q4 in `70-reasoning/bottleneck-triage.md` — indexing kernel with explicit branchless bit-twiddling that does not improve wall-clock. Correct response: swap to the intrinsic form; stop counting branches; count SASS ops.
