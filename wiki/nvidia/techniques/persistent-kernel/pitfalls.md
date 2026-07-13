---
id: algo-persistent-kernel-pitfalls
type: algorithm
vendor: nvidia
title: Pitfalls
tags:
- cuda-cpp
- tma
- cluster
- warp-specialization
- persistent-kernel
- pipeline-stages
- shared-memory-optimization
- swizzling
- tile-scheduling
- tma-multicast
- gemm
evidence_level: spec
source:
- path: spec
  anchor: Algorithm reference
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
hardware_features:
- tma
- cluster
techniques:
- warp-specialization
- persistent-kernel
- pipeline-stages
- shared-memory-optimization
- swizzling
- tile-scheduling
- tma-multicast
kernel_types:
- gemm
confidence: source-reported
---
# Persistent kernel — pitfalls

## 1. Grid size must equal device SM count, not problem tile count

The whole point of the persistent pattern is to launch one CTA per SM (× cluster size) and have each CTA consume many tiles via the inner loop. Launching with `gridDim = N_TILES_M × N_TILES_N` defeats the pattern and pays both the persistent overhead AND the per-tile launch overhead. Use `cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, ...)` (or hard-code 132 for H200) and pass that as the launch grid.

## 2. Pingpong needs an even tile count per CTA, OR the tail runs one consumer

The pingpong variant alternates work between two consumer warpgroups within each CTA. If `n_tiles_per_cta` is odd, one consumer takes the tail tile alone. This is fine for correctness but the per-tile latency at the tail differs from steady-state — don't infer steady-state per-tile timing from a small problem where the tail dominates.

## 3. Cooperative is meaningful only with cluster-multicast TMA

The cooperative variant pairs two consumer warpgroups on the same tile and pairs that with cluster-multicast TMA so the load broadcasts to ≥ 2 CTAs in a cluster. Setting cluster `<1, 1, 1>` defeats the multicast and you pay coordination overhead for nothing. Always set cluster ≥ `<2, 1, 1>` (typically `<4, 2, 1>`) and use the cluster-multicast TMA instruction variant when running cooperative.

## 4. Stream-K is a scheduler choice, not a separate variant

Stream-K decomposes one tile's K-loop across multiple CTAs and joins via partial-sum reductions. It is a *different scheduler* on top of the same persistent grid + warp-specialized mainloop — replace `scheduler_row/col(t)` with the stream-K decomposition; the rest of the kernel doesn't change. Stream-K and the cooperative variant are independent axes.

## 5. The persistent grid does not oversubscribe

A persistent kernel always launches at SM-count × cluster-size occupancy. You cannot scale by launching more CTAs than the grid — the scheduler indexes assume one persistent CTA per SM (or per cluster slot). For concurrency, use multiple streams; each stream gets its own persistent kernel and the GPU schedules them across SMs.

## 6. Per-tile inner work must be uniform for the scheduler to balance

Linear / swizzled schedulers assume every tile takes the same inner-loop time. If some tiles are masked / shorter / cheaper, the scheduler does not adapt and the cheap-tile CTAs sit idle. Stream-K can rebalance, but only along the K dimension; M/N skew requires a different (work-stealing) scheduler.

## 7. The persistent advantage is shape-dependent

At 2048³ on H200, persistent pingpong (193.8 TFLOPS) is within 1 % of non-persistent plain WS (195.4 TFLOPS); cooperative is 4 % slower. Persistent wins at larger problems — at 8192³ the cooperative variant reaches 292.7 TFLOPS (cluster-multicast finally amortizes; the persistent grid keeps all 132 SMs busy without launch-overhead bubbles between tiles). Don't pick persistent + cooperative for small problems; the coordination cost dominates.

## 8. The scheduler functions are host-precomputable — don't recompute per iteration

`scheduler_row(t)` and `scheduler_col(t)` are pure functions of `t, blockIdx, problem_dims`. For complex schedulers (Hilbert / stream-K) precompute the lookup table on host and pass via `__grid_constant__`; recomputing inside the persistent loop wastes cycles in the hot path.

## 9. Cutlass implementation specifics (PersistentScheduler / StreamKScheduler)

These are pitfalls of cutlass's *implementation* of the persistent pattern, not of the pattern itself. Listed for reference:

- `PersistentScheduler` launches `(SMs × CTAs-per-cluster, 1, 1)` — for cluster `<4, 2, 1>` on H200 (132 SMs), that is approximately ⌈132 / 8⌉ × 8 ≈ 136 CTAs total. The grid shape changes if you switch GPUs but the problem doesn't.
- `TileSchedulerType` is a template parameter independent of the mainloop schedule — set it to `cutlass::gemm::StreamKScheduler` to get stream-K on top of `KernelTmaWarpSpecializedCooperative`.
- The mainloop's `Stages` parameter is unrelated to persistence. Stages = smem ring depth in the warp-specialized mainloop; persistence = inner tile loop. They compose orthogonally.

A cutlass-free implementation skips all of these by making the scheduler a plain device function and the launch grid an explicit `N_SMS` constant.
