---
id: code-flash-attention-v3-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/flash-attention-v3
---
# FlashAttention v3 Tuning Parameters

## Compile-time parameters (from `hopper/block.h`)

| Knob | Typical values | Effect |
|---|---|---|
| kBlockM | 64, 128, 192 | Q-tile row count. Must be a multiple of wgmma M=64. Larger kBlockM increases compute per KV iteration. |
| kBlockN | 64, 128, 176, 256 | KV-tile sequence chunk. Larger kBlockN reduces number of KV iterations but increases smem. |
| kHeadDim | 64, 128, 192, 256 | Head dimension. Fixed per model. Affects the K-dimension of both Q@K^T and P@V tiles. |
| kNWarps | 8, 12, 16 | Warps per CTA. More warps enable more warpgroups (wgmma consumers). Typical: 8 = 2 warpgroups, 12 = 3 warpgroups. |
| kStages | 1, 2 | smem pipeline stages for KV tiles. |
| Is_causal | true, false | Enables triangular causal mask. |
| Is_local | true, false | Enables sliding window / local attention. |

## Runtime parameters (from Python/C++ API)

| Parameter | Type | Effect |
|---|---|---|
| batch_size | int | Number of sequences. |
| seqlen_q, seqlen_k | int | Query and Key/Value sequence lengths. |
| num_heads, num_heads_k | int | Number of Q heads and KV heads (for GQA). |
| head_dim | int | Must match kHeadDim. |
| softmax_scale | float | Typically 1/sqrt(head_dim). |
| window_size_left, window_size_right | int | For local attention. |
| num_splits | int | Split-KV parallelism along sequence dimension. |

## Key differences from cutlass FMHA tuning

- FlashAttention v3 uses a fixed set of pre-tuned tile configurations selected via head_dim and dtype, rather than cutlass's fully template-parameterized builder.
- The `num_splits` parameter enables split-KV parallelism (analogous to split-K in GEMM), which cutlass FMHA does not expose.
- kNWarps directly maps to warpgroup count, whereas cutlass FMHA uses `kNumMmaWarpGroups` as a separate option.

## Source reference

Tile configurations: `hopper/block.h`
Python API: `flash_attn/flash_attn_interface.py`
C++ dispatch: `hopper/flash_api.cpp`
