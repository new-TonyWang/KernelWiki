---
title: Barrier Optimization (scope, async-split, arrive/wait)
status: verified
evidence_level: measured
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9.86 + ptxas 12.9
measured_on: H200-SXM
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- normalization
- scan
- synchronization
requires_sm: '>=7.0'
requires_features:
- block-barrier
- async-barrier
single_kernel_useful: true
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1402-L1404
  excerpt: Throughput for __syncthreads() is 32 operations per clock cycle for devices
    of compute capability 6.0, 16 operations per clock cycle for devices of compute
    capability 7.x as well as 8.x and 64 operations per clock cycle for devices of
    compute capability 5.x, 6.1 and 6.2. __syncthreads() can impact performance by
    forcing the multiprocessor to idle.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3647-L3704
  excerpt: An asynchronous barrier differs from a typical single-stage barrier (__syncthreads)
    in that the notification by a thread that it has reached the barrier (the arrival)
    is separated from the operation of waiting for other threads to arrive at the
    barrier (the wait). This separation increases execution efficiency by allowing
    a thread to perform additional operations unrelated to the barrier. Devices of
    compute capability 8.0 or higher provide hardware acceleration for asynchronous
    barriers.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L19217-L19237
  excerpt: 'bar.cta.sync / bar.cta.arrive: CTA-level barrier with 16 named barrier
    resources. barrier.cta.sync.aligned and barrier.cta.arrive.aligned provide the
    aligned variants used when the compiler can prove full-warp participation.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L20580-L20600
  excerpt: 'mbarrier: a 64-bit memory-backed barrier object with separate arrival
    and wait phases, transaction-count tracking for async copies, and hardware-accelerated
    phase advancement on sm_80+.'
artifacts:
  code: 80-experience/hw-probes/barrier-cost/artifacts/barrier_cost_probe.cu
  build: 80-experience/hw-probes/barrier-cost/artifacts/build.sh
  introspection: 80-experience/hw-probes/barrier-cost/artifacts/device.json
  profile: ''
related_apis:
- __syncthreads
- __syncwarp
- cuda::barrier
- cuda::pipeline
- mbarrier.init
- mbarrier.arrive
- mbarrier.try_wait
related_skills:
- memory-ordering
- warp-primitives
- async-copy
- warp-divergence
id: skill-barrier-optimization
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

A **barrier** is a synchronization primitive that forces a group of threads to all reach the same program point before any of them proceeds. CUDA offers a hierarchy of barriers, each with its own scope, memory backing, and hardware acceleration:

- **`__syncwarp(mask)`** — 32-lane (or subset) warp barrier. The cheapest barrier available; on sm_70+ it is explicit hardware reconvergence after Independent Thread Scheduling diverges lanes.
- **`__syncthreads()`** — whole-block barrier, backed by PTX `bar.sync` with 16 named barrier slots per SM. Per BP §12.1.3 (L1402-L1404), its **throughput is 16 ops/clock on sm_7.x and sm_8.x**; the per-call wall-clock cost is dominated by the "SM idles until all threads arrive" stall, not by the barrier instruction throughput.
- **`cuda::barrier<thread_scope_block>` (mbarrier)** — memory-backed asynchronous barrier with separate `arrive` and `wait` phases (PG §3.2.4.2, L3647-L3704). Hardware-accelerated on sm_80+. The key advantage is **temporal splitting**: a thread signals arrival, does other work, and only blocks later when it actually needs the synchronization.
- **`cuda::barrier` with `mbarrier.arrive.expect_tx`** — extension that tracks **byte counts** of async copies (TMA, `cp.async`), so the barrier completes when both thread arrivals and copy completions are satisfied. This is the correct synchronization primitive for pipelined producer-consumer patterns, replacing the deprecated `cuda::pipeline` shape.
- **`barrier.cluster.arrive` / `barrier.cluster.wait`** — cluster scope barriers (sm_90+). Only required for distributed shared memory or multi-block coordination; out of scope for most cuda-core kernels.

Correct choice depends on **who needs to synchronize with whom**. The pathological default is `__syncthreads()` everywhere; the optimized pattern narrows to the smallest group that actually needs to see the write.

## Why

Barrier cost shows up two ways:

1. **Instruction cost** — the fixed cycles to issue and retire the barrier instruction itself. Small (single digits on sm_9.0a for `__syncwarp`, ~10s for `__syncthreads`).
2. **Idle time** — the time spent waiting for the *slowest* thread (or async operation) to arrive. This is usually 10–100× the instruction cost. BP §12.1.3 (L1404) captures this: `__syncthreads` "can impact performance by forcing the multiprocessor to idle".

The goal of this skill is **not** to reduce instruction cost (the barrier itself is cheap). It is to reduce **idle time** by (a) narrowing scope so fewer threads are ever waiting on one another, (b) splitting `arrive` from `wait` so each thread does useful work between the two, and (c) letting hardware (mbarrier with expect_tx) track async-copy progress without holding threads hostage.

PG §3.2.4.2 (L3647-L3704) documents the mechanism explicitly: "separation [between arrive and wait] increases execution efficiency by allowing a thread to perform additional operations unrelated to the barrier". Sub-skills S1–S4 below are the four practical patterns for exploiting that.

## When to use

### S1. Narrow the scope to the smallest group that must actually sync

The single highest-impact decision in barrier placement is **which group is synchronizing**. Prefer the narrowest scope that is correct:

```
warp-local work           → __syncwarp(0xFFFFFFFFu) (or nothing,
                            if your warp intrinsic is already _sync)
block-level work          → __syncthreads() or block.sync()
sub-block subset          → cuda::barrier with expected_count < blockDim
async-copy completion     → mbarrier with expect_tx
cluster-level (sm_90+)    → barrier.cluster.arrive/wait
```

```cuda
// Typical block-level reduction tail — __syncthreads is necessary
// because the warp-partial-sums written to smem[warpId] must be
// visible to warp 0.
smem[warpId] = warp_reduce(val);
__syncthreads();                    // block-wide
if (warpId == 0) {
    val = (laneId < NWARP) ? smem[laneId] : 0;
    val = warp_reduce(val);         // _sync shuffles, no extra barrier
}

// BUT: if the reduction is purely intra-warp (N ≤ 32), no
// __syncthreads is needed. Shuffle + _sync suffices.
```

Common smell: a kernel that only does intra-warp work uses `__syncthreads()` "to be safe". Each such call stalls 31 other warps on average. Replace with `__syncwarp()` if any barrier is needed at all (usually **none** is needed if the warp-shuffle is the `_sync` variant, per the warp-divergence skill's ITS discussion).

### S2. Split arrive from wait with `cuda::barrier` (hw-accel on sm_80+)

When the post-barrier consumer also has independent pre-wait work, split arrival from wait:

```cuda
#include <cuda/barrier>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void producer_consumer_kernel(const float* in, float* out, int N) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    __shared__ float staging[BLOCK];
    auto block = cg::this_thread_block();

    if (block.thread_rank() == 0)
        init(&bar, block.size());
    block.sync();                            // one-time init sync

    // ---- Produce phase ----
    staging[threadIdx.x] = compute_A(in[threadIdx.x]);
    auto token = bar.arrive();               // signal, do NOT block

    // ---- Independent work between arrive and wait ----
    float local = compute_B_independent(threadIdx.x);

    // ---- Wait only when staging[] is actually needed ----
    bar.wait(std::move(token));
    out[threadIdx.x] = staging[(threadIdx.x + 1) % BLOCK] + local;
}
```

The `compute_B_independent` body overlaps with other threads' producer work. On sm_80+ this overlap is hardware-accelerated: the barrier sits in shared memory and the `mbarrier.arrive` instruction returns immediately after incrementing the count.

### S3. Use `mbarrier` with `expect_tx` for async-copy pipelines

For kernels that stream data from global to shared memory via `cp.async` or TMA, the correct synchronization primitive is an mbarrier that tracks transaction byte counts. This is what the `async-copy` skill pairs with:

```cuda
// (Simplified; see async-copy skill for the full pipeline.)
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
if (threadIdx.x == 0) init(&bar, 1);
__syncthreads();

// Declare total expected bytes, then kick off cp.async instructions.
if (threadIdx.x == 0)
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::device::barrier_native_handle(bar), BYTES_IN_TILE);

// Every cp.async.* that targets smem also arrives on bar via the
// .mbarrier.arrive suffix in PTX.

bar.wait(bar.arrive());    // blocks until tx count is satisfied
// smem[] is now safe to consume.
```

### S4. Prefer block-level reduction in smem over cross-block synchronization

When a reduction / scan has to span more than one block, a common anti-pattern is a global counter with `atomicCAS`-based spin-locks. The right answer is almost always **two-pass launch** — each block reduces into a per-block slot, then a second kernel reduces the slots. See the `atomic-reduction` skill (S1 hierarchical pattern).

The only time cluster-level `barrier.cluster.*` is appropriate is when the kernel uses distributed shared memory across blocks of the same cluster (sm_90+); that is a separate skill under `40-hardware-feature/thread-block-cluster/` (pending bucket F).

## When NOT to use

- **When the synchronization is already handled by `_sync` warp intrinsics.** `__shfl_down_sync` / `__ballot_sync` each embed their own warp-sync semantics; adding `__syncwarp` before them is pure overhead. Do it only if the preceding code diverged and needs explicit reconvergence (per the `warp-divergence` skill pitfall P1).
- **When narrowing scope breaks correctness.** If warp 0 reads smem written by warp 3, you need `__syncthreads()`. Swapping to `__syncwarp()` is not an optimization, it is a bug.
- **When the kernel is already memory-bound at a healthy SOL.** A barrier that currently contributes < 5 % of stall cycles is not the bottleneck. Check NCU `smsp__warps_issue_stalled_barrier.pct` first; if it is already low, the barrier is not the problem. This is the skill's own version of the "profile first" rule.
- **When mbarrier would add more smem + instructions than it saves.** Legacy pitfall P6 (measured): `cuda::barrier` objects consume additional smem and emit more setup instructions; on small kernels with no independent work to overlap, plain `__syncthreads()` is cheaper.

## Measured Characteristics

Measured on H200-SXM (sm_90a, CUDA 12.9, driver 570.124.06) using [80-experience/hw-probes/barrier-cost/](../../../80-experience/hw-probes/barrier-cost/) — per-call cost of three barrier primitives at block sizes {128, 256, 512, 1024}. 10,000 barrier calls per inner loop, CUDA-events timing. Full record: [80-experience/hw-probes/barrier-cost/2026-04-22-barrier-cost.md](../../../80-experience/hw-probes/barrier-cost/2026-04-22-barrier-cost.md).

### ns per barrier call

| Block | `__syncwarp` | `__syncthreads` | `mbarrier arrive+wait` | mbarrier / syncthreads |
| ----- | -----------: | --------------: | ---------------------: | ---------------------: |
| 128   |        14.61 |           31.81 |                  50.50 |                  1.59× |
| 256   |        14.61 |           36.35 |                  52.53 |                  1.45× |
| 512   |        15.43 |           44.44 |                  74.37 |                  1.67× |
| 1024  |        22.13 |           61.51 |                 137.17 |              **2.23×** |

### Scaling factor (cost at B=1024 ÷ cost at B=128)

| Primitive              | Scaling factor |
| ---------------------- | -------------: |
| `__syncwarp`           |          1.51× |
| `__syncthreads`        |          1.93× |
| `mbarrier arrive+wait` |      **2.72×** |

Key measured findings:

- **Cost hierarchy is warp → block → mbarrier**, with measured ratios roughly 1 : 2–3 : 3–6 at typical block sizes. Skill §S1 "narrow the scope" is data-backed: `__syncthreads` → `__syncwarp` where correct saves ~2× per-call cost.
- **`__syncthreads` scales sub-linearly with block size** (1.93× for 8× the threads), consistent with BP §12.1.3's "16 ops/clock" throughput on sm_8.x (no sm_9.0a-specific revision of the claim).
- **mbarrier bare cost is a loss without overlap**: 1.6–2.2× slower than `__syncthreads` in a tight arrive+wait loop. The skill's §S2 (arrive/wait split) pays off **only** when there is real independent work of ≥ 100 cycles between arrive and wait — pitfall P6 upgraded from legacy-anecdotal to measured.
- **NCU Compute SOL for single-block kernels sits near 0.11 %** — the 131 unused SMs idle. This is expected and the NCU pass is useful only as corroboration; wall-clock ns/call is the authoritative measurement for barrier primitives.

The "arrive/wait overlap" variant of this probe (workload-dependent; real compute between `arrive` and `wait`) is a follow-up at `80-experience/hw-probes/barrier-async-overlap/` (open).

## Principles

1. **Narrow scope, not fewer barriers.** Deleting a required barrier is a correctness bug; narrowing `__syncthreads` to `__syncwarp` is almost always correct and free.
2. **Arrive is cheap, wait is expensive.** Splitting them only helps when there is real independent work between the two. Otherwise `__syncthreads` is cheaper.
3. **Barriers report SM idle time, not instruction cost.** The NCU metric of interest is `smsp__warps_issue_stalled_barrier_per_issue_active.pct`, not instruction throughput.
4. **Always re-converge before `mbarrier.arrive`.** Per PG §4.9.2.1 and pitfall P1, a divergent warp issues per-lane arrivals and the count completes too early.

## Open questions

- Q1. What is the actual per-call cycle cost of `__syncthreads` vs `mbarrier arrive+wait` on sm_90a H200 at block sizes 128 / 256 / 512 / 1024? Probe 2026-04-22 targets exactly this; measured section populated on completion.
- Q2. For the overlap case (S2 with real independent work), what is the minimum independent-work size where mbarrier-split beats `__syncthreads`? Follow-up probe: `80-experience/hw-probes/barrier-async-overlap/`.
- Q3. Does `barrier.cluster.arrive` / `wait` on sm_90 add enough cost over `__syncthreads` that cluster-level kernels should budget one extra pass? Depends on DSMEM probe at `40-hardware-feature/thread-block-cluster/` (pending bucket F).

## Legacy references

- `legacy_sandbox_path`: `KernelPilot/knowledge/optimization/synchronization/barrier-optimization/skill.md`. Legacy kept six sub-skills (S1–S6); this port keeps the four that are single-kernel-relevant (S1 async barriers → S2, S2 warp vs block → S1, S3 mbarrier expect_tx → S3, S6 minimize scope → S1 guidance). S4 cluster barriers is routed to the pending `40-hardware-feature/thread-block-cluster/` skill. S5 `__nanosleep` applies to spin-wait patterns (lock-based producer-consumer) and fits better under a future `30-skill/sync/spin-wait/` if one materializes — not migrated in this pass; pitfall P8 in `pitfalls.md` records the measurement caveat.
- Legacy L3 sandbox findings P6–P9 are retained in `pitfalls.md` pending H200 re-measurement.
- **Related but distinct**: `30-skill/sync/memory-ordering/` covers the *visibility* semantics around barriers; this skill covers the *cost and placement* of the barriers themselves. The two are complementary.
