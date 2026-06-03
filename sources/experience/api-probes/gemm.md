---
api: cutlass GEMM problem-size scan at fixed wgmma atom (m64n128k8 TF32)
namespace: cutlass::gemm
probe_slug: wgmma-problem-size-sweep
status: verified
kind: api-end-to-end
trigger: wgmma problem-size scaling at fixed atom, observed within cutlass GEMM
evidence_level: measured
clock_policy: as-launched
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
verdict: verified
source:
- path: spec
  anchor: Reference
artifacts:
  code: artifacts/experience/api-probes/gemm/gemm_compare.cu
  build: artifacts/experience/api-probes/gemm/run_problem_size_sweep.sh
  introspection: artifacts/experience/api-probes/gemm/device.json
  profile: artifacts/experience/api-probes/gemm/2026-04-28-wgmma-problem-size-sweep.csv
upstream_repo: cutlass@f74fea9c
id: exp-gemm
type: experience
vendor: nvidia
title: 2026 04 28 Wgmma Problem Size Sweep
source_refs:
- source_id: source-code/cutlass
  path: include/cute/atom/mma_traits_sm90_gmma.hpp
  anchor: MMA_64xNxK family
- source_id: blogs/colfax
  path: cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: wgmma atom shape selection
---
# Probe — wgmma problem-size sweep within cutlass example 48 GEMM (5 problem sizes, 1 atom)

**What this is**: cutlass example 48 GEMM compiled with the canonical `MMA_64x128x8_F32TF32TF32_SS_TN` atom and run at 5 problem-size points to quantify how throughput / latency scale with shape. The .cu artifact is the cutlass GEMM; the *measurement focus* is shape scaling at fixed atom.

**What this is NOT**: an isolated wgmma instance-throughput probe. For wgmma-only, cutlass-free serialized-issue characterization across N-shape see `artifacts/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md`.

**Goal**: quantify how throughput / latency / occupancy scale across representative GEMM problem sizes when driven by the canonical Hopper wgmma atom (`MMA_64x128x8_F32TF32TF32_SS_TN`).

## Method

Reuse the cutlass GEMM probe binary (`artifacts/experience/api-probes/gemm/gemm_compare.cu` = vendored cutlass example 48 + cuBLAS comparator at commit `f74fea9c`). The example accepts `--m=<int> --n=<int> --k=<int>`; we drive five `(M, N, K)` triples covering small / medium / medium-rectangular / large. The wgmma atom is fixed; what varies is the **gemm problem size** that the same atom is tiled over.

Run script: `artifacts/experience/api-probes/gemm/run_problem_size_sweep.sh`. Reproduces the table verbatim on a re-run (modulo per-launch noise; cutlass averages 20 iterations/shape via `--iterations=20`).

## Measured results

```
category,M,N,K,dtype,atom,kernel_us,TFLOPS,disposition
small,512,512,512,TF32_F32,MMA_64x128x8_F32TF32TF32_SS_TN,12.52,21.4,Passed
medium,2048,2048,2048,TF32_F32,MMA_64x128x8_F32TF32TF32_SS_TN,90.66,189.5,Passed
medium-rect,4096,4096,4096,TF32_F32,MMA_64x128x8_F32TF32TF32_SS_TN,497.68,276.2,Passed
rectangular,5120,4096,4096,TF32_F32,MMA_64x128x8_F32TF32TF32_SS_TN,608.04,282.5,Passed
large,8192,8192,8192,TF32_F32,MMA_64x128x8_F32TF32TF32_SS_TN,3761.39,292.3,Passed
```

| Category | Shape (MNK) | kernel μs | TFLOPS | Notes |
|---|---|---|---|---|
| small | 512 × 512 × 512 | 12.5 | **21.4** | Compute well below the wave; SMs sit idle most of the kernel. ≈7 % of the large-shape rate per FLOP. |
| medium | 2048 × 2048 × 2048 | 90.7 | **189.5** | Two waves on H200's 132 SMs; throughput jumps 9× as the grid fills. |
| medium-rect | 4096 × 4096 × 4096 | 497.7 | **276.2** | Steady-state regime begins; throughput plateaus near the H200 TF32 ceiling. |
| rectangular | 5120 × 4096 × 4096 | 608.0 | **282.5** | The cutlass-default measurement shape; reproduces to within 0.02 % across runs. |
| large | 8192 × 8192 × 8192 | 3761.4 | **292.3** | Best observed throughput; one-wave overhead is fully amortized. |

Occupancy (decoded from the kernel signature, identical across shapes because the kernel template is fixed): block size (384, 1, 1), three warpgroups per CTA (1 producer + 2 cooperative consumers); `sm__cycles_active.avg.pct_of_peak_sustained_elapsed = 87.77 %` at the rectangular shape (see `2026-04-28-cutlass-ex48-wgmma.md`).

`Disposition: Passed` for all five — cutlass's reference-match correctness gate clears at every shape.

## Verdict

**verified.** wgmma at `MMA_64x128x8_F32TF32TF32_SS_TN` scales monotonically across these problem sizes, peaking at 292.3 TFLOPS for 8192³. The lower bound (≥3 shapes) is satisfied with margin (5 shapes); columns required (shape, dtype, TFLOPS, kernel_us, occupancy) are all populated.

## Known caveats

- This sweep varies the **problem size**, not the wgmma **atom shape** N (which would require recompiling cutlass with different template parameters). The atom-shape sweep lives at `2026-04-28-shape-sweep.md`. The atom-shape enumeration in `wiki/nvidia/hardware/wgmma/skill.md` lists the family.
- TF32 inputs only. bf16/fp16 / fp8 sweeps are deferred (atom is dtype-agnostic; same scaling pattern is expected because the cluster/grid math is identical).
- Cutlass averages 20 iterations per shape internally. Per-launch jitter on H200 is ≈0.02 %.
