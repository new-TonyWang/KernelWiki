---
title: Non-aligned GEMM tail handling on Hopper via cutlass cooperative kernel
status: partial
evidence_level: measured
applies_to_pattern_class:
- tensor-core
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
- path: spec
  anchor: Reference
artifacts:
  code: sources/experience/api-probes/gemm/artifacts/gemm_tail.cu
  build: sources/experience/api-probes/gemm/artifacts/build_tail.sh
  introspection: sources/experience/api-probes/gemm/artifacts/device.json
  profile: sources/experience/api-probes/gemm/artifacts/profiles/2026-04-28-gemm-tail.csv
upstream_repo: cutlass@f74fea9c
related_apis: []
related_skills:
- tma
- wgmma
- gemm-aligned
id: skill-gemm
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
  anchor: MainloopSm90TmaGmmaWarpSpecialized + can_implement
- source_id: source-code/cutlass
  path: examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu
  anchor: Lall (gemm.can_implement(arguments) on line 415)
- source_id: blogs/colfax
  path: developing-cuda-kernels-for-gemm-on-nvidia-hopper-architecture-using-cutlass
  anchor: Hopper warp-specialized cooperative GEMM
---
# Non-aligned GEMM tail handling on Hopper

## What it is

The Hopper warp-specialized cooperative GEMM kernel from cutlass example 48 has a **predicate-based tail-handling code path** built into its `MainloopSm90TmaGmmaWarpSpecialized` mainloop: when the problem size is not an exact multiple of the CTA tile (`Shape<_128, _128, _32>` here), the boundary CTA tiles issue predicated loads that mask out elements past the problem boundary, so the kernel produces correct output without the user having to pad the tensors.

What the kernel does NOT do is silently extend itself to handle every conceivable shape. Specifically, when the problem is smaller than the cluster's effective M-N footprint, `gemm.can_implement(arguments)` returns `kErrorInvalidProblem` and the kernel refuses to launch. The tail strategy is therefore **"predicate at the boundary tile + reject sub-tile problems"**, not "extend to arbitrary shape".

## When to use it

- GEMMs where the problem dimensions are reasonably close to the CTA tile size (≥ tile-M × tile-N elements per dim) but not necessarily exact multiples. The cooperative kernel's cost-of-tail is small as long as the boundary CTAs are a small fraction of the total grid.
- LLM attention-projection layers where MNK are between several hundred and several thousand elements with non-power-of-two dimensions; this is the empirical sweet spot.

## When NOT to use it

- Problem sizes smaller than ~128×128 (sub-CtaTile in M and N) — the kernel will reject. Switch to a smaller CTA tile (e.g. `<64, 64, 32>` + the non-cooperative scheduler) or grouped GEMM.
- Problem sizes where most of the grid is boundary tiles (e.g. shapes just above 128×128 — the entire grid is the tail path; throughput is launch-bound rather than compute-bound).

## Measured Characteristics

The cooperative kernel's behavior at three non-aligned categories, measured on H200-SXM at TF32 inputs / F32 accumulator, vs. the aligned baseline at similar problem sizes:

The skill's primary hardware-probe evidence comes from the wgmma + TMA primitives the cooperative kernel composes. See:

- TMA primitive (cutlass-API path): sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md. Cutlass-free isolated TMA bandwidth: sources/experience/hw-probes/tma-ptx/2026-04-29-tma-throughput.md.
- wgmma atom (cutlass-API path): sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md. Cutlass-free isolated wgmma family (N-shape × dtype × layout × A-source): sources/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md.
- wgmma atom-shape sweep: sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md.

The non-aligned-shape-specific measurements live in the api-probe at sources/experience/api-probes/gemm/2026-04-28-gemm-tail.md:

| Shape (MNK) | Category | Kernel template | cutlass result | cutlass μs | cutlass GFLOPS | cutlass-vs-cuBLAS direct diff |
|---|---|---|---|---|---|---|
| 80 × 80 × 80 | far smaller than tile (genuinely non-aligned in all 3 dims on small-tile kernel: 80 = 1.25 × CtaTile.M=64, 80 = 2.5 × CtaTile.K=32) | TileShape `<64, 64, 32>` + Cluster `<1, 1, 1>` (smaller-tile non-cooperative) | Passed | 5.52 | 185 | **max_abs=0, max_rel=0, 0 mismatches** (bit-identical) |
| 200 × 200 × 200 | tile-misaligned (between 1 and 2 CtaTiles of 128) | TileShape `<128, 128, 32>` + Cluster `<4, 2, 1>` (cooperative) | Passed | 8.8 | 1,820 | **max_abs=0, max_rel=0, 0 mismatches** (bit-identical) |
| 1440 × 1440 × 1440 | far larger with non-aligned tail | same cooperative template | Passed | 45.8 | 130,260 | **max_abs=0, max_rel=0, 0 mismatches** (bit-identical) |

A/B comparison: the predicate-tail strategy is compiled unconditionally into `MainloopSm90TmaGmmaWarpSpecialized`; there is no runtime flag. A literal **same-shape on/off** measurement is **structurally infeasible** here: a kernel template that does NOT compile in tail handling cannot correctly run a non-aligned shape (it either rejects via `can_implement` or computes wrong output past the boundary). The closest practical measurement is **the same kernel binary at two adjacent shapes** — one where the predicates fire (boundary CTA tile present) and one where the predicates are compiled-in but never mask anything out (every CTA tile is full). This is recorded below as a proxy, not a closure-grade isolation of the boundary mechanism — see the report's confounders section.

| Configuration | Shape | Predicate-tail active? | TFLOPS | Latency | Boundary-strategy cost |
|---|---|---|---|---|---|
| With boundary strategy active | 1440 × 1440 × 1440 | Yes (32-elt tail per dim, cluster non-aligned) | 130.2 | 45.86 μs | — |
| Without boundary strategy active | 1536 × 1536 × 1536 | No (every CTA tile full, cluster aligned) | 149.4 | 48.51 μs | **12.85% throughput drop** when the predicate-tail path activates |

Both runs use the same `gemm_compare` binary (same kernel template, same compile-time predicate logic). Both are bit-identical to cuBLAS (`max_abs=0, max_rel=0`). 1536³ is the smallest fully-tile-and-cluster-aligned shape ≥ 1440³ on this kernel; **the throughput delta is therefore an adjacent-shape proxy with documented confounders (cluster-multicast alignment delta, total-work delta, wave-quantization), not a clean isolation of the boundary mechanism**. The skill's overall `status: partial` reflects this — the three shape categories are correctness-verified, but the strategy A/B is a documented proxy.

Other useful (but less tightly controlled) comparisons:

| Shape | Aligned reference (different scale) | Aligned TFLOPS | Non-aligned TFLOPS | Note |
|---|---|---|---|---|
| 80³ (small-tile fallback) | n/a — different kernel template | n/a | 0.185 | Launch-bound (4-CTA grid). The predicate-tail path is active in M, N, and K; cost cannot be cleanly separated from per-launch overhead at this scale. |
| 200³ | 512³ (cooperative aligned anchor) | 21.7 | 1.85 | Mostly absolute-size effect (launch-bound across scales), not isolated tail cost. |
| 1440³ | 2048³ aligned | 188.7 | 130.2 | 31% gap — confounds scale + tail. The 1440-vs-1536 A/B above is the cleaner isolation. |
| 30³ | smaller than supported domain | n/a | n/a | "Boundary strategy" at this scale is `kErrorInvalidProblem`; both the cooperative and small-tile templates refuse M < 64. |

## Cross-references

- Aligned baseline: `wiki/nvidia/foundations/compute/gemm/aligned/skill.md`
- Library-usage notes + tuning log: `wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-tail/{README.md, tuning.md}`. Canonical buildable artifact at `sources/experience/api-probes/gemm/artifacts/{gemm_tail.cu, gemm_tail_small.cu, helper.h, build_tail.sh, run_tail.sh}`.
- Probe record: `sources/experience/api-probes/gemm/2026-04-28-gemm-tail.md`
- Failure modes: `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/pitfalls.md`
- Tuning parameter space: tuning.md.
