---
id: code-cutlass-cute-example48-hopper-warp-specialized-gemm-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# 60-code/cutlass-cute/example48-hopper-warp-specialized-gemm

This directory is a CUTLASS reading guide for:

```text
examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu
```

It explains the example as one complete Hopper GEMM, not as separate TMA and warp-specialization topics. The runnable artifact for this path lives at:

- `80-experience/api-probes/gemm/artifacts/gemm_aligned.cu`

No buildable code lives here.

## What This Example Demonstrates

Example 48 is the smallest useful CUTLASS example where these Hopper pieces appear together:

| Layer | CUTLASS / CuTe object | What it means |
|---|---|---|
| CTA tile | `TileShape = Shape<_128,_128,_32>` | one CTA computes a 128 x 128 output tile over K chunks of 32 |
| CTA cluster | `ClusterShape = Shape<_4,_2,_1>` | 8 CTAs cooperate as a 4-by-2 cluster |
| Producer side | `cute::copy(tma_atom, gmem, smem)` | TMA moves A/B tiles from global memory to shared memory |
| Consumer side | `cute::gemm(tiled_mma, sA, sB, accum)` | WGMMA consumes shared-memory operands |
| Synchronization | `PipelineTmaAsync` / mbarriers | producer commits smem stages; consumers wait and release |
| Schedule | `KernelTmaWarpSpecializedCooperative` via `KernelScheduleAuto` | producer / consumer warpgroup split with cooperative cluster behavior |

Read this directory when you want to understand the source file's structure: how the template parameters, TMA copy path, WGMMA issue path, cluster shape, and warp-specialized schedule fit together.

## What Belongs Elsewhere

- WGMMA atom name decoding: `../wgmma-atom-decoding/`
- TMA hardware concept and pitfalls: `40-hardware-feature/tma/`
- Warp-specialization as a general algorithm: `50-classical-algo/warp-specialization/`
- Performance probes and measured evidence: `80-experience/api-probes/gemm/`

## File Map

- `mainloop_skeleton.md` — the compact tour through example 48's template setup and producer / consumer mainloop.

## Key Reading Order

1. Start with `mainloop_skeleton.md`.
2. If `MMA_64x128x8_F32TF32TF32_SS_TN` is unclear, branch to `../wgmma-atom-decoding/wgmma_skeleton.md`.
3. If `ClusterShape` or schedule selection is unclear, stay in this directory: example 48's TMA multicast and cooperative schedule only make sense together.

## Cross-references

- Aligned GEMM probe: `80-experience/api-probes/gemm/2026-04-28-gemm-aligned.md`
- TMA counters: `80-experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md`
- WGMMA counters: `80-experience/api-probes/gemm/2026-04-28-wgmma-counters.md`
- Warp-specialization ablation: `80-experience/api-probes/gemm/2026-04-28-warp-specialization-ablation.md`
- Upstream (pinned): `cutlass@f74fea9c` (`{{CUTLASS_REPO_REF}}`)
