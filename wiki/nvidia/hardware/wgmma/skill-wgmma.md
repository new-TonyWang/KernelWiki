---
title: 'wgmma: Warpgroup MMA on Hopper via cutlass-cute'
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
applies_to_ops:
- gemm
- attention
requires_sm: '>=9.0a'
requires_features:
- wgmma
- tma
- mbarrier
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: blogs/colfax/cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: wgmma atom + warpgroup synchronization
  excerpt: Warpgroup MMA = 4 warps × M64 rows; SS-form consumes A and B from smem;
    F32 accumulator stays in registers.
- path: '{{CUTLASS_REPO_REF}}/include/cute/atom/mma_traits_sm90_gmma.hpp'
  anchor: MMA_64x{N}x{K}_F32{TF32|BF16|FP16|FP8}_SS_TN
- path: '{{CUTLASS_REPO_REF}}/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu'
  anchor: Lall
artifacts:
  code: sources/experience/api-probes/gemm/artifacts/gemm_compare.cu
  build: sources/experience/api-probes/gemm/artifacts/build.sh
  introspection: sources/experience/api-probes/gemm/artifacts/device.json
  profile: sources/experience/api-probes/gemm/artifacts/profiles/2026-04-28-wgmma-counters.csv
upstream_repo: cutlass@f74fea9c
related_apis: []
related_skills:
- tma
- warp-specialization
id: skill-wgmma
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
# wgmma — Warpgroup MMA on Hopper

## What it is

`wgmma` is the Hopper instruction family that lets one **warpgroup** (4 warps = 128 threads) issue a single matrix-multiply-accumulate covering an `M64 × N × K` tile in a single asynchronous PTX call. The atom shape is `M=64`, `N ∈ {8, 16, 32, 64, 96, 128, 192, 256}`, `K = 16` for bf16/fp16 / `K = 8` for tf32 / `K = 32` for fp8. A `wgmma.fence` + `wgmma.commit_group` + `wgmma.wait_group` triple separates issue from completion, and the F32 accumulator lives in the consumer warpgroup's registers throughout — no smem round-trip.

In cutlass-cute the entire pattern lives in `MMA_Atom<MMA_64xNxK_F32{tf32|bf16|fp16|fp8}_{SS|RS}_TN<..>>` plus a `TiledMMA<...>` wrapper.

## Why it helps

- **One instruction per warpgroup tile** vs. dozens of `mma.sync` / `wmma` issues on Ampere. Issue bandwidth on the warp scheduler stops being the bottleneck for compute-bound GEMM.
- **Asynchronous**. `wgmma` issues, then `wgmma.commit_group` releases the warpgroup; the warpgroup can do other work (or just spin on `wgmma.wait_group`) until the result lands in registers.
- **Pairs naturally with TMA**. Together the producer/consumer pattern (TMA producer → mbarrier → wgmma consumer) keeps both pipes saturated.

## When to use it

- Any single-kernel gemm/attention compute path on Hopper. wgmma is the only path to ≥80 % of H200 peak TFLOPS at gemm.
- Mixed-dtype mainloops (cutlass `MainloopSm90TmaGmmaWarpSpecialized*MixedInput…`) where the input dtype differs from the accumulator dtype.

## When **not** to use it

- Targets older than sm_90a (Ampere/Turing — use `mma.sync` or `wmma`).
- Operators dominated by memory traffic, where the wgmma issue rate is irrelevant.
- Kernels that cannot afford a 128-thread minimum block — wgmma fundamentally consumes one warpgroup.

## Atom shapes supported on H200 (from `mma_traits_sm90_gmma.hpp`)

| Family | M | N range | K |
|---|---|---|---|
| F32 ← BF16 × BF16 | 64 | 8, 16, 32, 64, 96, 128, 192, 256 | 16 |
| F32 ← FP16 × FP16 | 64 | 8, 16, 32, 64, 96, 128, 192, 256 | 16 |
| F32 ← TF32 × TF32 | 64 | 8, 16, 32, 64, 96, 128, 192, 256 | 8 |
| F32 ← FP8(E4M3/E5M2) × FP8(E4M3/E5M2) | 64 | 8, 16, 32, 64, 96, 128, 192, 256 | 32 |

A shape-selection guide (which N to pick for which gemm size) appears below; this skill currently pins the F32-TF32 64×128×8 shape that was measured at the canonical anchor.

## Pre-conditions to issue wgmma (measured)

1. `nvcc -arch=sm_90a` (NOT plain `sm_90`).
2. Block size that includes ≥1 warpgroup of 128 threads (the example uses 384 threads = 3 warpgroups: 1 producer + 2 consumers in the cooperative pattern).
3. SmemLayoutAtom that is wgmma-compatible — typically the `Swizzle<3,4,3>` × 32-bit layout cute provides via `cute::GMMA_64x128x8_Major<cute::GMMA::Major::K>` etc.
4. F32 accumulator in registers; no smem-resident accumulator path is taken on Hopper.

## Measured Characteristics

End-to-end wgmma-driven GEMM via `examples/48_hopper_warp_specialized_gemm` at the cutlass commit pin (see `artifacts.code`):

- 5120 × 4096 × 4096 GEMM (TF32 inputs, F32 acc) on the canonical `MMA_64x128x8_F32TF32TF32_SS_TN` atom: **0.608 ms / launch ≈ 282.6 TFLOPS** on H200-SXM, `Disposition: Passed` against the cutlass reference.
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` = **87.77 %** (decoded from the profiled kernel signature).
- `sm__warps_active.avg.pct_of_peak_sustained_active` = 14.07 % — low warp-occupancy is the **expected** signature of warp-specialized wgmma: most warps spin on `wgmma.wait_group`, not on the warp scheduler.
- 384 threads/CTA × 4×30 grid = 120 CTAs = one wave on H200's 132 SMs.

Full probe record: [sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md](../../sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md).

For the wgmma instruction in isolation (cutlass-free 11-config zoo across N-shape × dtype × layout × A-source on H200), see [sources/experience/hw-probes/wgmma-ptx/2026-04-28-wgmma-ptx-hello.md](../../sources/experience/hw-probes/wgmma-ptx/2026-04-28-wgmma-ptx-hello.md) and [sources/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md](../../sources/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md).

## Minimum repro

`sources/experience/api-probes/gemm/artifacts/gemm_compare.cu` (vendored from cutlass example 48 at commit `f74fea9c`), plus `build.sh` (nvcc invocation), `run.sh` (build → run → ncu → compute-sanitizer → CSV), `run-shape-sweep.sh` (atom-shape sweep), and `device.json` (H200 nvidia-smi snapshot) in the same `artifacts/` directory.

## Shape-selection guide

Measured on H200-SXM at problem size 4096 × 4096 × 4096, TF32 input / F32 accumulator, varying only the wgmma atom selected by cute's `CollectiveBuilder` from `TileShape.N`. Three atoms from the `MMA_64xNx8_F32TF32TF32_SS_TN` family. Cluster shape adjusted for the largest atom so the cluster-N × tile-N product still tiles the 4096-wide grid.

| Atom | TileShape (M×N×K) | Cluster | Kernel μs | TFLOPS | SM cycles active | Warps active | Use when |
|---|---|---|---|---|---|---|---|
| `MMA_64x64x8_F32TF32TF32_SS_TN`  (small)  | 128×64×32  | (4,2,1) | 765.3 | **179.6** | 85.45 % | 14.07 % | Smaller per-CTA register footprint; useful when the kernel must coexist with a large epilogue, with persistent-kernel state, or when the operator's N dimension is itself small. About 58 % of the medium-atom rate at this problem size. |
| `MMA_64x128x8_F32TF32TF32_SS_TN` (medium) | 128×128×32 | (4,2,1) | 497.8 | **276.1** | 85.81 % | 14.07 % | The cutlass-example-48 default. Best general-purpose choice for square TF32 GEMM in the 4K–8K range. Well within the H200 cooperative kernel's compute-bound sweet spot. |
| `MMA_64x256x8_F32TF32TF32_SS_TN` (large)  | 128×256×32 | (2,2,1) | 444.8 | **309.0** | 77.23 % | 14.07 % | Largest atom in the family. Best peak at large square shapes, but the smaller cluster (`<2,2,1>` instead of `<4,2,1>`) cuts multicast efficiency, so the gain over medium is shape-dependent and shrinks for smaller problems. The grid drops to (2, 60, 1) = 120 CTAs ≈ one wave, hence the lower SM-cycles-active. |

**Heuristic for picking N:**

- Default is **medium (N=128)** unless you have reason otherwise. It pairs cleanly with the standard `<4,2,1>` cluster and gives ~85 % SM time at typical LLM GEMM shapes.
- Pick **large (N=256)** only when (a) the problem is square at ≥4K and (b) you can afford the smaller cluster (or you can rebalance ClusterShape to keep multicast). At 4K² the gain is ~12 % over medium; at smaller shapes the gain disappears or reverses because the grid drops below one wave.
- Pick **small (N=64)** only when register pressure or epilogue-co-resident state forces it. Throughput drops sharply (~35 % below medium) because the kernel issues 2× as many wgmma atoms per CTA tile and the wgmma issue pipeline becomes the bottleneck.
- The `sm__warps_active` 14.07 % across all three atoms is the warp-specialized-wgmma signature, not a regression. Don't try to "improve" it; warps in the consumer wg are *meant* to spend most of their time on `wgmma.wait_group`.

For complementary guidance on **problem-size scaling at a fixed atom** (small/medium/large GEMM under the medium atom), see the supplemental record [sources/experience/api-probes/gemm/2026-04-28-wgmma-problem-size-sweep.md](../../sources/experience/api-probes/gemm/2026-04-28-wgmma-problem-size-sweep.md). Headline: at the medium atom the kernel goes 21.4 → 189.5 → 276.2 → 282.5 → 292.3 TFLOPS as MNK grows from 512³ through 8192³; below 2K total problem size the kernel-launch/wave-amortization overheads dominate, so adjust CTA tile or grouped-GEMM rather than the wgmma atom.

**Authoritative evidence:**
- Atom-shape sweep (this section): [sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md](../../sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md). Reproducible from `sources/experience/api-probes/gemm/artifacts/run_atom_sweep.sh`.
- Problem-size sweep at the medium atom: [sources/experience/api-probes/gemm/2026-04-28-wgmma-problem-size-sweep.md](../../sources/experience/api-probes/gemm/2026-04-28-wgmma-problem-size-sweep.md). Reproducible from `sources/experience/api-probes/gemm/artifacts/run_problem_size_sweep.sh`.

## Atomic Usage (cute)

Each row maps a cute `MMA_Atom` specialization (from `mma_traits_sm90_gmma.hpp`) to the PTX probe config that exercises the same instruction. Every atom listed has a 1:1 correspondence with a measured config in the wgmma-ptx probe zoo at `sources/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md`.

| cute MMA_Atom struct | PTX mnemonic shape | Probe config | Probe record |
|---|---|---|---|
| `SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n8k16.f32.bf16.bf16` | `ss_tn_bf16_n8` | `sources/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md` |
| `SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n16k16.f32.bf16.bf16` | `ss_tn_bf16_n16` | same |
| `SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n32k16.f32.bf16.bf16` | `ss_tn_bf16_n32` | same |
| `SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n64k16.f32.bf16.bf16` | `ss_tn_bf16_n64` | same |
| `SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n128k16.f32.bf16.bf16` | `ss_tn_bf16_n128` | same |
| `SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n256k16.f32.bf16.bf16` | `ss_tn_bf16_n256` | same |
| `SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>` | `m64n64k16.f32.f16.f16` | `ss_tn_fp16_n64` | same |
| `SM90_64x64x8_F32TF32TF32_SS_TN<>` | `m64n64k8.f32.tf32.tf32` | `ss_tn_tf32_n64` | same |
| `SM90_64x64x32_S32S8S8S32_SS_TN<>` | `m64n64k32.s32.s8.s8` | `ss_tn_s8_n64` | same |
| `SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>` | `m64n64k16.f32.bf16.bf16` (NT) | `ss_nt_bf16_n64` | same |
| `SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>` | `m64n64k16.f32.bf16.bf16` (RS) | `rs_tn_bf16_n64` | same |

Minimal cute-API snippet (the pattern used by cutlass example 48 and the `gemm_compare.cu` harness):

```cpp
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

// Choose an atom (this is the medium N=128 TF32 default):
using MMA_Op = SM90_64x128x8_F32TF32TF32_SS_TN<>;
using MMA = MMA_Atom<MMA_Traits<MMA_Op>>;
// Tile it across the CTA with TiledMMA:
using TiledMma = TiledMMA<MMA, Layout<Shape<_2,_1,_1>>>;  // 2 warpgroups consumer
```

Families NOT exercised in the probe zoo but present in `mma_traits_sm90_gmma.hpp`:
- fp8 atoms (`SM90_64xNx32_F32E4M3E4M3_SS_TN`, `SM90_64xNx32_F32E5M2E4M3_SS_TN`, etc.) — K=32, same issue port as s8.
- Sparse atoms (`SM90_64xNx64_F32E4M3E4M3_SS_TN<SparseConfig::StructuredN>`) — requires structured sparsity metadata.
- Mixed-input atoms (e.g. `SM90_64xNx32_F32E4M3E5M2_RS_TN`) — A and B may differ in dtype.

## How it connects to the rest of the KB

- Pairs with `wiki/nvidia/hardware/tma/skill.md` — the cooperative kernel exercises both. wgmma without TMA falls back to `cp.async`, which collapses the issue-rate advantage.
- Pre-condition for `wiki/nvidia/techniques/warp-specialization/` and `wiki/nvidia/techniques/persistent-kernel/`.
- Shape-selection guide is the dedicated section above, backed by the atom-shape sweep at [sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md](../../sources/experience/api-probes/gemm/2026-04-28-wgmma-atom-shape-sweep.md).
