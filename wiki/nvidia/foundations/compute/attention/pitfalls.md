---
id: pitfall-attention
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
techniques:
- pipeline-stages
- shared-memory-optimization
- software-exp
- conditional-rescaling
kernel_types:
- attention
- flash-attention
confidence: inferred
tags:
- wgmma
- tma
- pipeline-stages
- shared-memory-optimization
- software-exp
- conditional-rescaling
- attention
- flash-attention
- cuda-cpp
---
# MVP Minimal Flash-Attention Pitfalls

## 1. Online softmax numerical stability

**Problem**: The online softmax algorithm requires tracking a running maximum `m_i` per row. If `m_i` is not updated before computing `exp(S - m_new)`, the exponential overflows for large logits, producing `inf` values that corrupt the output.

**Trigger**: Processing long sequences (S > 2048) with large attention scores, especially with fp16 inputs where the dynamic range is limited.

**Mitigation**: Always update `m_new = max(m_i, rowmax(S))` before computing `exp(S - m_new)`. The rescaling factor `exp(m_i - m_new)` is always <= 1.0, so it never overflows. Use f32 accumulators for the softmax computation even when inputs are fp16.

## 2. QK^T vs Q@K layout confusion

**Problem**: The attention score `S = Q @ K^T` requires K to be transposed. If K is loaded in the same layout as Q (row-major with head_dim as the fast dimension), the wgmma instruction computes Q @ K instead of Q @ K^T, producing a (BLOCK_M x BLOCK_N) result from (BLOCK_M x D) @ (D x BLOCK_N) only if K is transposed.

**Trigger**: Loading K tile with the same TMA descriptor as Q without accounting for the transpose.

**Mitigation**: Either transpose K explicitly in smem before the wgmma, or use a different wgmma layout variant (e.g., `.TN` with K loaded in N-major layout). The cutlass FMHA example uses the latter approach.

## 3. Accumulator rescaling precision

**Problem**: Each KV-tile iteration rescales the running output accumulator by `exp(m_old - m_new)`. Over many iterations, this multiplicative rescaling can accumulate rounding error, especially in fp16.

**Trigger**: Very long sequences (S > 8192) with many KV-tile iterations.

**Mitigation**: Keep the output accumulator in f32 throughout the loop. Only convert to fp16 at the final writeback after the `O /= l_i` normalization.

## 4. Shared memory capacity for Q + K + V tiles

**Problem**: The kernel needs smem for Q (BLOCK_M x D), K (BLOCK_N x D), and V (BLOCK_N x D) tiles simultaneously. For BLOCK_M=128, BLOCK_N=128, D=128 in fp16, this is 128x128x2 + 128x128x2 + 128x128x2 = 96 KB — close to the H200 per-SM smem limit of 228 KB but feasible. With multi-stage pipelining, the requirement multiplies.

**Trigger**: Large tile sizes with multiple pipeline stages exceeding available smem.

**Mitigation**: Size tiles to fit within the target SM's smem budget. H200 supports up to 228 KB dynamic smem per CTA (with `cudaFuncSetAttribute`). Use 2-stage pipelining for large tiles or reduce BLOCK_N.

## 5. wgmma atom shape mismatch with attention dimensions

**Problem**: wgmma atoms have M=64 fixed. If BLOCK_M < 64 or BLOCK_M is not a multiple of 64, the kernel cannot directly use wgmma for the Q @ K^T tile.

**Trigger**: Choosing BLOCK_M=32 for small sequence lengths.

**Mitigation**: Always set BLOCK_M to a multiple of 64. For small sequences, use BLOCK_M=64 as the minimum.
