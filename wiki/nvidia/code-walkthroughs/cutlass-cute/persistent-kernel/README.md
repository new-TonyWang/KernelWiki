---
id: code-cutlass-cute-persistent-kernel-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# wiki/nvidia/code-walkthroughs/cutlass-cute/persistent-kernel — persistent-schedule library usage

Library-usage knowledge for the persistent flavour of cutlass's `KernelTmaWarpSpecialized*` schedules. The canonical reproducible artifacts:

- `sources/experience/api-probes/gemm/artifacts/gemm_compare_pingpong.cu` (pingpong, Cluster `<2,1,1>`)
- `sources/experience/api-probes/gemm/artifacts/gemm_aligned.cu` (cooperative, Cluster `<4,2,1>` — auto-selected at large aligned shapes)

This directory is the **distilled-knowledge view**. No buildable code lives here.

## When to use

A persistent CTA processes many output tiles serially, amortizing kernel-launch cost and stabilizing wgmma issue-port utilization across tile boundaries. Pair with pingpong / cooperative warp-spec; plain WS does not persist.

## Persistent schedule landscape

| Schedule | Cluster | Persistent | Tile scheduler |
|---|---|---|---|
| `KernelTmaWarpSpecialized` | `<1,1,1>` | no | `PersistentScheduler` not engaged |
| `KernelTmaWarpSpecializedPingpong` | `<2,1,1>` | yes | `PersistentScheduler` round-robin across tiles, alternating consumer-A / consumer-B |
| `KernelTmaWarpSpecializedCooperative` | `<4,2,1>` | yes | `PersistentScheduler` + cluster-multicast TMA |
| StreamK schedule | varies | yes | `StreamKScheduler` — split-K tiles distributed across CTAs |

## Measured A/B at 2048³

| Schedule | Persistent | TFLOPS |
|---|---|---|
| `KernelTmaWarpSpecialized` | OFF | 195.4 |
| `KernelTmaWarpSpecializedPingpong` | ON (pingpong) | 193.8 |
| `KernelTmaWarpSpecializedCooperative` | ON (cooperative) | 186.9 |

Persistent on/off at 2048³ is a wash; the persistent variants pay back only at 8192³+ where cluster-multicast amortizes its setup.

## Key cutlass entry points

- `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` — pingpong kernel body.
- `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` — cooperative kernel body.
- `include/cutlass/gemm/kernel/tile_scheduler.hpp` — `PersistentScheduler` and `StreamKScheduler`.
- `examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu` — multi-schedule sweep.

## Cross-references

- Skill: `wiki/nvidia/techniques/persistent-kernel/skill.md`
- Pitfalls: `wiki/nvidia/techniques/persistent-kernel/pitfalls.md`
- Example 48 cooperative mainloop notes: `wiki/nvidia/code-walkthroughs/cutlass-cute/example48-hopper-warp-specialized-gemm/`
- Ablation data: `sources/experience/api-probes/gemm/2026-04-28-persistent-kernel-ablation.md`
