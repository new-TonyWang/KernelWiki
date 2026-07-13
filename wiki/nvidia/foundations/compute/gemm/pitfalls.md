---
id: pitfall-gemm
type: pitfall
vendor: nvidia
title: Pitfalls
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
hardware_features:
- wgmma
- tma
- cluster
techniques:
- warp-specialization
- tma-multicast
kernel_types:
- gemm
confidence: inferred
tags:
- wgmma
- tma
- cluster
- warp-specialization
- tma-multicast
- gemm
- cuda-cpp
---
# Non-aligned GEMM tail handling — pitfalls

## 1. Sub-CtaTile shapes are REJECTED, not silently padded

The cooperative kernel's `gemm.can_implement(arguments)` rejects `(M, N) < CtaTile` (here, smaller than 128 × 128). At 30³ the kernel returns `cutlass::Status::kErrorInvalidProblem` and example 48 prints `Got cutlass error: Invalid status at: 415`. This is the **explicit** rejection path; the kernel never silently runs at a sub-tile shape. The "silent padding without recording the assumption" anti-pattern is therefore not violated by this kernel — but if you build your own kernel you must replicate this rejection or you risk silent miscompute. Always check `can_implement` before launching.

## 2. The cooperative tail path is not a runtime toggle

`MainloopSm90TmaGmmaWarpSpecialized` always runs the predicate-based tail path at boundary tiles when the problem is not a CtaTile multiple. There is no flag to disable it; the only way to "remove" the tail strategy is to pick a different mainloop template (`MainloopSm90TmaGmma` — the non-warpspecialized path) or to manually pad the input tensors before launching. Manual padding is the silent-padding anti-pattern unless documented.

## 3. The "tail cost" at moderate sizes is real but not catastrophic

At 1440³ the cooperative kernel runs at 129.4 TFLOPS vs. 188.7 TFLOPS at 2048³ (aligned). That is a ~32% throughput cost for a shape that is just one CtaTile away from being fully aligned. The cost is **not** purely the predicate-tail logic; it is the combined effect of (a) the boundary CTA tile having reduced effective work, (b) the cluster-multicast TMA losing some efficiency at the boundary, and (c) the wgmma atom not fully utilized at the tail. Don't try to attribute the full 32% to a single cause.

## 4. cuBLAS auto-selects different kernels for different non-aligned shapes

cuBLAS at 200³ achieves 2,621 GFLOPS; at 1440³ it achieves 153,708 GFLOPS. The ratio of cutlass to cuBLAS varies by shape (0.71× at 200³ vs 0.84× at 1440³) because cuBLAS's selector picks per-shape kernels. cutlass example 48 uses the same cooperative template across all shapes; the gap is expected and is not a regression. If your operator must always beat cuBLAS, you either match cuBLAS's per-shape kernel choice or accept the gap.

## 5. ClusterShape constrains which non-aligned shapes work

The cooperative kernel uses ClusterShape `<4, 2, 1>`. The cluster covers 4 CTAs in M and 2 in N, so the kernel's effective minimum problem size in (M-CTAs, N-CTAs) is at least one full cluster. At CtaTile 128 in M and N, that is 4 × 128 = 512 in M and 2 × 128 = 256 in N. If the problem is smaller than this in either dimension (e.g. 200×200×200), the cluster-multicast still fires but launches fewer-than-one-cluster's worth of CTAs in the boundary direction; this is fine as long as `can_implement` accepts the shape, but it reduces multicast efficiency. At 30×30×30 the shape is below the cluster minimum and `can_implement` rejects.

