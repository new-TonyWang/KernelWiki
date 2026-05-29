---
id: code-cutlass-cute-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
---
# FMHA Collective Tuning Parameters

## Builder-level knobs (resolved before collective instantiation)

These knobs are passed as template options to `FmhaKernelBuilder` and determine which collective template is instantiated.

| Knob | Default | Range | Effect on collective |
|---|---|---|---|
| `kNumMmaWarpGroups` | 2 | 1, 2 | Determines BLOCK_M (64 per warpgroup). Default 2 gives BLOCK_M=128. 1 warpgroup gives BLOCK_M=64 with less register/smem pressure. |
| `kStagesKV` | auto | 2, 3, 4 | Number of smem pipeline stages for K/V tiles. Each stage holds one (BLOCK_N × HEAD_DIM) K tile + one (BLOCK_N × HEAD_DIM) V tile. More stages hide TMA load latency. |
| `kStagesQ` | auto | 1, 2 | Q pipeline stages. Usually 1 (Q is loaded once and reused across all KV tiles). 2 enables prefetch overlap for the next Q tile in persistent mode. |
| `kClusterM` | 1 | 1, 2, 4 | CTA cluster along the Q-tile dimension. Enables TMA multicast — one TMA load distributes K/V tiles to multiple clustered CTAs. Reduces HBM bandwidth per CTA. |
| `kIsPersistent` | false | true, false | Persistent kernel mode. CTAs loop over work items (batch, head, Q-tile) instead of one-shot. Improves SM utilization for small problem sizes. |

## Kernel-level knobs (inside the collective)

These are derived or fixed inside the collective implementation.

| Knob | Source | Typical value | Effect |
|---|---|---|---|
| BLOCK_M | `64 × kNumMmaWarpGroups` | 64, 128 | Q-tile row count. Must be a multiple of wgmma M=64. |
| BLOCK_N | derived from HEAD_DIM | 64, 128 | KV-tile sequence chunk. Larger BLOCK_N means fewer mainloop iterations but more smem. |
| HEAD_DIM | template param | 64, 128, 256 | Inner dimension of Q@K^T. Fixed per model. |
| Softmax precision | `kAccQK` option | f32 | Accumulator type for Q@K^T scores. f16 reduces register pressure but loses precision for long sequences. |
| Producer warps | 1 warpgroup | 128 threads | Fixed at 1 warpgroup for TMA load; not tunable. |
| Pipeline barrier type | mbarrier (default) or lock | mbarrier | `kIsMainloopLocked` switches to lock-based sync. |

## Recommended configurations

| Scenario | kNumMmaWarpGroups | kStagesKV | kClusterM | Notes |
|---|---|---|---|---|
| Large batch, D=128 | 2 | 2 | 1 | High compute; 2 consumer warpgroups fully utilize SM. |
| Small batch, D=128 | 1 | 3 | 2 | Multicast saves bandwidth; persistent mode recommended. |
| D=64 | 1 | 4 | 1 | Smaller tiles → more iterations → more stages to hide latency. |
| D=256 | 2 | 2 | 1 | Large D already fills smem; minimize stages. |

## Interaction with the rest of the tuning space

The FMHA collective's tile shapes interact with the GEMM-level wgmma atom selection (the collective uses `SM90_64xNxK_*_SS_TN` atoms internally). The atom N-size is determined by BLOCK_N and HEAD_DIM. See `40-hardware-feature/wgmma/skill.md` for atom shape selection guidance.

## Source reference

Builder-level knobs: `kernel/fmha_options.hpp` (`Tag` enum) + `kernel/fmha_kernel_builder.hpp` (option resolution).
Collective implementation: `collective/fmha_collective_tma_warpspecialized.hpp`.
