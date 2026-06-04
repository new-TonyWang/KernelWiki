---
title: Persistent kernel (grid = SM count, internal tile loop)
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
- tensor-core/gemm
- cuda-core
applies_to_ops:
- gemm
- reduction
- attention
requires_sm: '>=7.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: wiki/nvidia/hardware/tma/skill-tma-ptx.md
  anchor: cutlass-free TMA primitive (composes per-tile in the persistent loop)
- path: wiki/nvidia/hardware/wgmma/skill-wgmma-ptx.md
  anchor: cutlass-free wgmma primitive
artifacts:
  code: artifacts/experience/api-probes/gemm/gemm_compare_pingpong.cu
  build: artifacts/experience/api-probes/gemm/build_persistent.sh
  introspection: artifacts/experience/api-probes/gemm/device.json
  profile: artifacts/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.csv
  ablation: artifacts/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.csv
upstream_repo: cutlass@f74fea9c (one possible implementation; see "References")
related_apis: []
related_skills:
- warp-specialization
- gemm-aligned
- tma-ptx
- wgmma-ptx
id: algo-persistent-kernel
type: algorithm
vendor: nvidia
tags:
- cuda-cpp
- wgmma
- tma
- mbarrier
- cluster
- warp-specialization
- persistent-kernel
- pipeline-stages
- data-reuse
- shared-memory-optimization
- swizzling
- tile-scheduling
- tma-multicast
- gemm
- fused-kernel
- attention
- quantization
- ptx
- cute-dsl
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L28042
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11680-L11686
- source_id: blogs/colfax
  path: cutlass-tutorial-persistent-kernels-and-stream-k
  anchor: persistent-kernel + stream-K walkthrough — tile-scheduler abstraction
- source_id: blogs/colfax
  path: developing-cuda-kernels-for-gemm-on-nvidia-hopper-architecture-using-cutlass
  anchor: pingpong vs cooperative scheduling section
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
- mbarrier
- cluster
techniques:
- warp-specialization
- persistent-kernel
- pipeline-stages
- data-reuse
- shared-memory-optimization
- swizzling
- tile-scheduling
- tma-multicast
kernel_types:
- gemm
- fused-kernel
- attention
- quantization
confidence: experimental
artifact_dir: artifacts/experience/api-probes/gemm
---
# Persistent kernel (algorithm skeleton)

## What it is

An *algorithmic pattern* where the launch grid size equals the device's SM count (or a small multiple) and the kernel loops internally over the work units, instead of one CTA per work unit. The tile-to-CTA assignment is computed inside the kernel by a **tile scheduler** rather than by `dim3 grid`.

The pattern is independent of cutlass. cutlass packages it as `PersistentScheduler` / `StreamKScheduler` driving its `KernelTmaWarpSpecialized{Pingpong,Cooperative}` mainloops; the same idea works in any kernel that processes a uniform set of work tiles.

## Why the pattern works

Two structural wins over a non-persistent launch:

1. **Eliminates per-tile launch overhead.** A non-persistent kernel with N tiles incurs one launch + grid setup; a persistent kernel pays once and amortizes across all tiles via the inner loop. For small per-tile work, the overhead can be a noticeable fraction.
2. **Smarter rasterization without re-launching.** The scheduler can swizzle, linearize, or stream-K the tile order to improve L2 reuse, balance work across SMs, or split a single tile across multiple CTAs — all decided per work-tile inside the kernel.

The pattern composes naturally with warp-specialization: the producer warps stream TMA loads continuously while the consumer warpgroups process tiles handed to them by the scheduler. The two patterns are orthogonal: warp-specialization is about *intra-CTA* parallelism (producer vs consumer); persistent is about *inter-tile* sharing of the same CTA.

## When to use it

- Large work-tile counts (≥ a few hundred tiles) where the launch + scheduler-startup cost can amortize.
- Stream-K decompositions where work is split across SMs at the K dimension; the persistent grid is the substrate that lets stream-K rebalance load.
- LLM serving / repeated launches with similar shapes — persistent reduces launch jitter.

## When NOT to use it

- Tiny problems (≤ a few CtaTiles in M and N): the inner-loop setup overhead dominates a non-persistent launch's tiny gain.
- Kernels that need fine-grained per-tile compile-time specialization (different epilogues per tile region): persistent kernels share the body across tiles.

## Variants

| Variant | Consumer warpgroups per CTA | Per-CTA tile cadence in the persistent loop |
|---|---|---|
| **Plain persistent** | 1 | one tile, then advance to next |
| **Pingpong persistent** | 2 (alternating) | consumer-A computes tile T, consumer-B prepares T+1; on tile T+2 they swap |
| **Cooperative persistent** | 2 (cooperating on same tile) | both consumers split the wgmma rows of one tile; advance jointly to next; pair with cluster-multicast TMA (cutlass-free PTX measured at `sources/experience/hw-probes/tma-ptx.md`, 1.26× / 1.79× effective-bandwidth at C=2 / C=4) |

Cutlass spells these as `KernelTmaWarpSpecializedPingpong` and `KernelTmaWarpSpecializedCooperative`; those names label *the warp-spec variant + persistent loop combination*. The persistent loop itself is independent of warp-spec — a non-warp-specialized persistent kernel also exists in principle, just not landed in cutlass for sm_90.

## Algorithm skeleton (cutlass-free)

```
__global__ void persistent_kernel(/* tensor maps, problem dims */) {
    // 1. Compute the work-tile partition for this CTA.
    int n_tiles_total  = N_TILES_M * N_TILES_N;
    int n_tiles_per_cta = (n_tiles_total + gridDim.x - 1) / gridDim.x;
    int my_first_tile  = blockIdx.x * n_tiles_per_cta;
    int my_last_tile   = min(my_first_tile + n_tiles_per_cta, n_tiles_total);

    // 2. Persistent loop — process all tiles assigned to this CTA in order.
    for (int t = my_first_tile; t < my_last_tile; ++t) {
        int tile_m = scheduler_row(t);   // e.g. linear, swizzled, or stream-K
        int tile_n = scheduler_col(t);

        // 3. Per-tile work — composes the warp-specialized mainloop from
        //    wiki/nvidia/techniques/warp-specialization.md (producer/consumer
        //    + mbarrier-pipelined TMA→wgmma), accumulating into per-thread
        //    fragment registers across all K iterations of this tile.
        accum = {0};
        for (int k = 0; k < N_TILES_K; ++k) {
            // ... pipelined TMA load A[tile_m, k] + B[k, tile_n] ...
            // ... wgmma against the loaded smem tiles ...
            // ... accumulate into accum ...
        }

        // 4. Per-tile epilogue — quantize / activate / store accum to D[tile_m, tile_n].
        epilogue.store(accum, D, tile_m, tile_n);
    }
}

// Launch with grid = SM count (× cluster size for cooperative).
persistent_kernel<<<N_SMS, /*threads*/, smem>>>(...);
```

Variants:

- **Pingpong** — split `for t` into two interleaved streams (even-t vs odd-t) and route them to consumer-A vs consumer-B. The producer warp keeps streaming for both; the two consumers each touch ⌈n_tiles_per_cta / 2⌉ tiles.
- **Cooperative** — keep `for t` as-is but split the per-tile wgmma work across two consumer warpgroups (rows [0..31] and [32..63] of the m64 atom). Pair with cluster-multicast TMA so the producer's load broadcasts to two CTAs in a 2-CTA cluster.
- **Stream-K** — replace `scheduler_row/col(t)` with a stream-K decomposition that lets one tile's K-loop be split across multiple CTAs, with the partial reductions joined via a final atomic or split-K commit. The persistent grid is the substrate; stream-K is a different scheduler choice on top.

The scheduler functions `scheduler_row/col` are pure host-precomputable functions of `t, blockIdx, problem_dims`. Common choices:

| Scheduler | When |
|---|---|
| Linear (`t / N_TILES_N, t % N_TILES_N`) | Simplest; near-optimal when N_TILES_M × N_TILES_N is small |
| Swizzled (Hilbert-like) | Improves L2 reuse when adjacent tiles share input rows or cols |
| Stream-K | Splits one tile's K across SMs; helps load balance when tile count is not a multiple of SM count |

## Implementations on disk

- **Cutlass realization** — `cutlass::gemm::kernel::tile_scheduler.hpp` (`PersistentScheduler` / `StreamKScheduler`), composed into `sm90_gemm_tma_warpspecialized_pingpong.hpp` and `_cooperative.hpp` mainloops. Cluster-shape constraints, multicast pairings, and per-stage smem allocators are cutlass-implementation details; see `pitfalls.md`.
- **Cutlass-free building blocks** — TMA-PTX (`wiki/nvidia/hardware/tma/skill-tma-ptx.md`) and wgmma-PTX (`wiki/nvidia/hardware/wgmma/skill-wgmma-ptx.md`) provide the per-tile inner kernel; the persistent loop above is plain CUDA C++. No cutlass dependency at the algorithm level.

## Measured Characteristics

H200-SXM, sm_90a. Algorithm validated in the cutlass realization at problem 2048 × 2048 × 2048, TF32 inputs / F32 accumulator (the cutlass-free realization is queued):

| Variant | Persistent? | Cluster | Latency (μs) | TFLOPS | cuBLAS diff |
|---|---|---|---|---|---|
| Plain WS (non-persistent) | OFF | `<1, 1, 1>` | 87.94 | 195.4 | bit-identical |
| Pingpong persistent | ON | `<2, 1, 1>` | 88.64 | 193.8 | bit-identical |
| Cooperative persistent | ON | `<4, 2, 1>` | 91.92 | 186.9 | bit-identical |

At 2048³ on H200 the persistent advantage is small (within 1% of non-persistent plain WS for pingpong; cooperative pays 4% for its setup cost). The persistent advantage compounds at LARGER problems: the aligned-GEMM 8192³ measurement records 292.7 TFLOPS for the cooperative persistent schedule — substantially above the 2048³ ceiling because the persistent grid keeps all 132 H200 SMs busy across many tiles without launch-overhead bubbles, and cluster-multicast TMA amortizes once the per-CTA tile count is large.

## Cross-references

- Warp-specialization sibling (intra-CTA pattern): `wiki/nvidia/techniques/warp-specialization.md`
- TMA-PTX (cutlass-free producer side): `wiki/nvidia/hardware/tma/skill-tma-ptx.md`
- wgmma-PTX (cutlass-free consumer side): `wiki/nvidia/hardware/wgmma/skill-wgmma-ptx.md`
- Aligned GEMM consumer: `wiki/nvidia/foundations/compute/gemm.md`
- Cutlass library-usage notes + tuning log: `wiki/nvidia/code-walkthroughs/cutlass-cute/persistent-kernel/{README.md, tuning.md}`
- Probe + ablation: `sources/experience/api-probes/gemm.md`
- Underlying TMA hw-probe (composed per tile): `sources/experience/hw-probes/tma-ptx.md`
- Underlying wgmma hw-probe (composed per tile): `sources/experience/hw-probes/wgmma-ptx.md`
- Failure modes: `wiki/nvidia/techniques/persistent-kernel/pitfalls.md`
