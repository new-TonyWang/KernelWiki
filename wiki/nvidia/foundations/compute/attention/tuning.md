---
id: skill-attention-tuning
type: skill
vendor: nvidia
title: Tuning
tags:
- cuda-cpp
applies_to:
- general
source:
- path: spec
  anchor: CUDA Programming Guide
evidence_level: spec
---
# MVP Minimal Flash-Attention Tuning Parameters

## Primary kernel: fixed-config TMA+wgmma

The primary artifact (`flash_attn_tma_wgmma.cu`) uses fixed constants — not template parameters:

| Constant | Value | Code location | Notes |
|---|---|---|---|
| BM | 64 | `static constexpr int BM = 64` | Matches wgmma M=64. Not tunable without recompilation. |
| BN | 64 | `static constexpr int BN = 64` | KV-tile width. Fixed for this MVP. |
| HD | 64 | `static constexpr int HD = 64` | Head dimension. Fixed per model. |
| NTH | 128 | `static constexpr int NTH = 128` | 1 warpgroup (4 warps × 32 threads). |

Future tuning axes (not yet explored):
- Pipeline depth (single-buffer → double-buffer TMA)
- Smem swizzle mode (SWIZZLE_128B for bank-conflict-free access)
- Larger N per wgmma (m64n32k16 or m64n64k16 for fewer wgmma calls)

## Runtime parameters

| Parameter | Effect |
|---|---|
| B (batch) | Number of independent sequences. Grid Y dimension. |
| H (heads) | Number of attention heads. Grid Y dimension (B×H). |
| S (seq_len) | Sequence length. Must be a multiple of 64 (BM=BN=64). Total work is O(S² × D) per batch-head. |
| D (head_dim) | Must equal 64 (HD constant). |
| scale | Softmax temperature, typically 1/sqrt(D). |

## Smem budget (primary kernel)

Total dynamic smem = ~40.8 KB (fits in default 48 KB):
- Q tile: 64×64 fp16 = 8 KB
- KV tile: 64×64 fp16 = 8 KB (shared for K then V)
- S matrix: 64×64 f32 = 16 KB (wgmma output / softmax workspace)
- P matrix: 64×64 fp16 = 8 KB (softmax output for second wgmma)
- Per-row state: 3 × 64 × f32 = 768 B (max, sum, rescale)
- mbarrier: 8 B

## Secondary reference: thread-level tuning sweep

The secondary reference kernel (`flash_attn_minimal.cu`) is templatized on BLOCK_M × BLOCK_N and has a measured tile-shape sweep on H200. This data characterizes thread-level scalar performance, not the primary TMA+wgmma kernel.

H200-SXM, sm_90a, cuda 12.9.86. Fixed workload: B=1, H=2, S=256, D=64. CSV at `80-experience/api-probes/attention/artifacts/profiles/attention-sweep.csv`. Full analysis at `80-experience/hw-probes/attention-tuning/2026-05-08-attention-tile-sweep.md`.

| BLOCK_M | BLOCK_N | kernel_ms | throughput_gflops | Disposition |
|---|---|---|---|---|
| 32 | 64 | **0.151** | **221.77** | Passed |
| 32 | 32 | 0.166 | 201.77 | Passed |
| 64 | 64 | 0.298 | 112.67 | Passed |
| 64 | 128 | 0.310 | 108.38 | Passed |
| 64 | 32 | 0.341 | 98.49 | Passed |
| 128 | 64 | 0.502 | 66.89 | Passed |

BLOCK_M is the dominant knob (2x cost per 2x BLOCK_M). These results characterize thread-level scalar performance, not tensor-core capability.
