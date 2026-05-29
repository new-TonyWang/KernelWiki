---
title: Warp-specialized GEMM mainloop (producer/consumer warpgroups)
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
- tensor-core/gemm
applies_to_ops:
- gemm
requires_sm: '>=9.0a'
requires_features:
- tma
- wgmma
- mbarrier
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11680-L11686
  excerpt: 4.11.1.3. Producer-Consumer Pattern Through Warp Specialization — implement
    a producer-consumer pattern where a single warp is specialized as the producer
    performing asynchronous data copies from global to shared memory, while the remaining
    warps consume the data from shared memory and perform computations. To enable
    concurrency between the producer and the consumer threads, we use double-buffering
    in shared memory.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L10720
  excerpt: A thread block can be spatially partitioned to allow different threads
    to perform independent operations. This is most commonly done by assigning threads
    from different warps within the thread block to specific tasks. This technique
    is referred to as warp specialization.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L16316
  excerpt: The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk
    variant uses the mbarrier complete-tx byte tracking — the synchronization primitive
    that connects the producer's TMA load to the consumer's wgmma issue.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L20810-L20825
  excerpt: 9.7.13.15.9. mbarrier.init initializes the mbarrier object at the location
    specified by the address operand addr with the unsigned 32-bit integer count —
    the bar_full / bar_empty pair in the WS skeleton.
- path: blogs/colfax/developing-cuda-kernels-for-gemm-on-nvidia-hopper-architecture-using-cutlass
  anchor: warp-specialized GEMM mainloop walkthrough — producer/consumer warpgroups
    + mbarrier-pipelined TMA→wgmma
- path: blogs/colfax/cutlass-tutorial-efficient-gemm-kernel-designs-with-pipelining
  anchor: pipelining strategy for warp-specialized GEMM
- path: 40-hardware-feature/tma-ptx/skill.md
  anchor: cutlass-free TMA primitive (the producer issues these)
- path: 40-hardware-feature/wgmma-ptx/skill.md
  anchor: cutlass-free wgmma primitive (the consumer issues these)
artifacts:
  code: 80-experience/api-probes/gemm/artifacts/gemm_compare_ws.cu
  build: 80-experience/api-probes/gemm/artifacts/build_warpspec.sh
  introspection: 80-experience/api-probes/gemm/artifacts/device.json
  profile: 80-experience/api-probes/gemm/artifacts/profiles/2026-04-28-warp-specialization-ablation.csv
  ablation: 80-experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md
upstream_repo: cutlass@f74fea9c (one possible implementation; see "References")
related_apis: []
related_skills:
- tma-ptx
- wgmma-ptx
- gemm-aligned
- persistent-kernel
id: algo-warp-specialization
type: algorithm
vendor: nvidia
tags:
- cuda-cpp
---
# Warp-specialized GEMM mainloop (algorithm skeleton)

## What it is

An *algorithmic pattern* for hiding TMA latency behind wgmma compute on Hopper. The threadblock is split into:

- **Producer warps** — issue TMA loads of A and B tiles into a multi-stage smem ring buffer; signal completion via a per-stage mbarrier.
- **Consumer warpgroup(s)** — wait on the mbarrier (TMA done), issue `wgmma.mma_async` against the loaded tile, signal back that the smem stage is consumed and may be refilled.

The pattern is independent of cutlass; cutlass is one realization (`MainloopSm90TmaGmmaWarpSpecialized`). The cutlass-free realization composes the primitives from [`40-hardware-feature/tma-ptx`](../../40-hardware-feature/tma-ptx/skill.md) (producer side) and [`40-hardware-feature/wgmma-ptx`](../../40-hardware-feature/wgmma-ptx/skill.md) (consumer side).

## Why the pattern works

A naive Hopper GEMM mainloop interleaves TMA-load and wgmma-issue serially in one warpgroup. The wgmma engine then stalls every iteration waiting on the next TMA tile. By pinning the TMA issuer to a *separate* warp(s) and letting the consumer keep computing on the previous tile, the TMA latency is hidden behind compute.

Three variants of the pattern, each adding an axis of parallelism:

| Variant | Producer warps | Consumer warpgroups | Per-CTA tile cadence |
|---|---|---|---|
| **Plain WS** | 1 warp (or 1 warpgroup) | 1 warpgroup | one tile at a time |
| **Pingpong** | 1 | 2 (alternating) | while consumer-A computes tile T, consumer-B prepares tile T+1 |
| **Cooperative** | 1 | 2 (cooperating on same tile) | both consumers split the wgmma rows of one tile; usually paired with cluster-multicast TMA so the producer's tile is broadcast to ≥2 CTAs (see [`80-experience/hw-probes/tma-ptx/2026-04-30-tma-multicast.md`](../../80-experience/hw-probes/tma-ptx/2026-04-30-tma-multicast.md) — measured 1.26× / 1.79× effective-bandwidth amplification at C=2 / C=4) |

(Cutlass exposes these as `KernelTmaWarpSpecialized*` dispatch policies; the names are implementation labels, not algorithm names.)

## When to use it

- Hopper sm_90a GEMM-class kernels where TMA latency is non-trivial relative to wgmma compute (essentially all production GEMMs).
- Any cutlass-free Hopper kernel that loads with TMA and computes with wgmma — the pattern is the canonical way to overlap them.

## When NOT to use it

- Single-tile-grid problems (problem ≤ one CtaTile in M and N). The producer/consumer split adds coordination overhead with no latency to hide.
- Kernels that don't use TMA (Ampere `cp.async`, manual `cp.async.bulk` without tensor maps). Without TMA's high-latency-but-async issue, there is nothing to hide.

## Algorithm skeleton (cutlass-free)

The pattern below uses `S` smem stages (typical 2–4) and the TMA / wgmma primitives from the sibling skills. The producer stage `s` and consumer stage `s` use two mbarriers per stage: `bar_full[s]` (producer → consumer) signals the load is complete; `bar_empty[s]` (consumer → producer) signals the smem slot may be refilled.

```
__shared__ {A_tile[S], B_tile[S], bar_full[S], bar_empty[S]};

if (warp_id == PRODUCER_WARP) {
    // Initial fill of S stages.
    for (s = 0; s < S; ++s) {
        wait(bar_empty[s], phase=0);                  // smem slot s is free
        arrive_expect_tx(bar_full[s], tile_bytes);
        cp.async.bulk.tensor.2d(A_tile[s], tensor_map_A, coord(0, s));
        cp.async.bulk.tensor.2d(B_tile[s], tensor_map_B, coord(0, s));
    }
    // Steady state.
    for (k = S; k < N_TILES_K; ++k) {
        s = k % S;
        wait(bar_empty[s], phase=(k/S - 1) & 1);
        arrive_expect_tx(bar_full[s], tile_bytes);
        cp.async.bulk.tensor.2d(A_tile[s], tensor_map_A, coord(0, k));
        cp.async.bulk.tensor.2d(B_tile[s], tensor_map_B, coord(0, k));
    }
} else {  // CONSUMER_WARPGROUP — 128 threads of 4 warps
    accum = {0};
    for (k = 0; k < N_TILES_K; ++k) {
        s = k % S;
        wait(bar_full[s], phase=(k/S) & 1);
        wgmma.fence;
        wgmma.mma_async.m64nNk16.f32.bf16.bf16 accum,
             desc(A_tile[s]), desc(B_tile[s]), 1, 1, 1, 0, 0;
        wgmma.commit_group;
        wgmma.wait_group 0;
        arrive(bar_empty[s]);                         // signal slot s is free
    }
    epilogue.store(accum, D);
}
```

Pingpong variant: keep two consumer warpgroups; route even-k tiles to consumer-A and odd-k to consumer-B; each consumer touches ⌈N_TILES_K/2⌉ tiles. Both consumers share one producer.

Cooperative variant: keep two consumer warpgroups, both wait on the same `bar_full[s]`; consumer-A handles wgmma rows [0..31] of the m64 atom, consumer-B rows [32..63]; both arrive on `bar_empty[s]` (= 2 expected arrivals per slot). Pair with cluster-multicast TMA (`cp.async.bulk.tensor.shared::cluster.global.tile.bulk_group.cta_group::2`) so the producer's load broadcasts to both CTAs in a 2-CTA cluster — single load feeds two consumers' wgmma.

`tile_bytes` per stage = `(BOX_M*BOX_K + BOX_K*BOX_N) * sizeof(elem)`. mbarrier expected-tx counts both A and B in one stage; the producer issues two TMA loads but the barrier expects the sum.

## Implementations on disk

- **Cutlass realization** — `cutlass::gemm::kernel::sm90_gemm_tma_warpspecialized.hpp` (`KernelTmaWarpSpecialized`), `_pingpong.hpp` (Pingpong), `_cooperative.hpp` (Cooperative). Cluster-shape constraints, EpilogueSchedule co-constraints, and stage-count autocarvers are cutlass-implementation details; see `pitfalls.md`.
- **Cutlass-free building blocks** — `40-hardware-feature/tma-ptx/skill.md` (producer's `cp.async.bulk.tensor.*` + mbarrier protocol), `40-hardware-feature/wgmma-ptx/skill.md` (consumer's `wgmma.mma_async.*` family). The cutlass-free GEMM at `30-skill/compute/gemm-ptx/` composes them but is not yet warp-specialized; adding the producer/consumer split following the skeleton above is the natural extension.

## Measured Characteristics

H200-SXM, sm_90a, CUDA 12.9.86. The algorithm has been validated in the cutlass realization at problem 2048 × 2048 × 2048 in TF32 (the cutlass-free realization is queued as a follow-up). Numbers below characterize the *algorithm pattern* on H200, observed via the cutlass implementation:

| Variant | Latency (μs) | TFLOPS | cuBLAS diff | Note |
|---|---|---|---|---|
| Serial baseline (no warp-spec) | 96.02 | 178.9 | bit-identical | One warpgroup interleaves TMA + wgmma |
| Plain WS | 87.94 | 195.4 | bit-identical | Producer/consumer split, no persistence |
| Pingpong | 88.64 | 193.8 | bit-identical | 2 consumer warpgroups alternating tiles, persistent |
| Cooperative | 91.92 | 186.9 | bit-identical | 2 consumers cooperating on same tile + cluster-multicast |

**On/off A/B at 2048³**: enabling warp-specialization (serial → plain WS) gives **+9.2% throughput** (178.9 → 195.4 TFLOPS) at bit-identical correctness. The improvement comes entirely from hiding TMA latency behind wgmma compute via the producer/consumer pipeline.

At 2048³ all three WS variants are within 5% of each other; plain WS leads narrowly. At 8192³ (cooperative + cluster-multicast amortizes its setup) the cooperative variant climbs to 292.7 TFLOPS — see `30-skill/compute/gemm/aligned/skill.md` for the larger-shape data.

## Cross-references

- TMA-PTX primitive (cutlass-free producer side): `40-hardware-feature/tma-ptx/skill.md` — measured throughput sweep at `80-experience/hw-probes/tma-ptx/2026-04-29-tma-throughput.md`.
- wgmma-PTX primitive (cutlass-free consumer side): `40-hardware-feature/wgmma-ptx/skill.md` — atom zoo at `80-experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md`.
- Aligned GEMM consumer: `30-skill/compute/gemm/aligned/skill.md`
- Persistent-kernel sibling: `50-classical-algo/persistent-kernel/skill.md`
- Cutlass example 48 source-reading notes: `60-code/cutlass-cute/example48-hopper-warp-specialized-gemm/{README.md, mainloop_skeleton.md}`
- Probe + ablation: `80-experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md`
- Failure modes: `50-classical-algo/warp-specialization/pitfalls.md`
