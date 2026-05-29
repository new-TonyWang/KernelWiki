---
title: MVP Minimal Flash-Attention Kernel (TMA + wgmma, cutlass-template-free)
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
applies_to_ops:
- attention
requires_sm: '>=9.0a'
requires_features:
- wgmma
- tma
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: sources/experience/hw-probes/wgmma-ptx/artifacts/wgmma_hello.cu
  anchor: wgmma PTX inline-asm pattern reused for attention Q@K^T and P@V tiles
- path: sources/experience/hw-probes/tma-ptx/artifacts/tma_hello.cu
  anchor: TMA load PTX pattern reused for Q/K/V tile loads
- path: sources/experience/api-probes/attention/artifacts/flash_attn_tma_wgmma.cu
  anchor: flash_attn_tma_wgmma_kernel — TMA+wgmma online-softmax attention
artifacts:
  code: sources/experience/api-probes/attention/artifacts/flash_attn_tma_wgmma.cu
  build: sources/experience/api-probes/attention/artifacts/build.sh
  introspection: sources/experience/api-probes/attention/artifacts/device.json
  profile: sources/experience/api-probes/attention/artifacts/profiles/tma-wgmma-ncu.csv
related_apis: []
related_skills:
- compute/attention/cutlass-fmha
upstream_repo: none (hand-built from TMA + wgmma primitives)
id: skill-attention
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: examples/88_hopper_fmha/88_hopper_fmha.cu
  anchor: Hopper FMHA example — used as algorithmic reference only, not as code dependency
---
# MVP Minimal Flash-Attention Kernel

## What it is

A hand-built flash-attention kernel that computes scaled dot-product attention `softmax(QK^T / sqrt(d)) * V` using TMA for tile loads and wgmma for both matmuls, without depending on cutlass FMHA templates. This mirrors the existing "aligned GEMM minimal kernel" pattern but for the attention operator.

**Status**: `verified` / `measured` — the TMA+wgmma kernel (`flash_attn_tma_wgmma.cu`) passes correctness on H200 with `max_abs_err = 0.000061` (well within 1e-2 tolerance), 0 mismatches out of 16384 elements, and `compute-sanitizer memcheck: 0 errors`. Kernel time: 0.024 ms at B=1 H=2 S=128 D=64. Uses canonical CUTLASS smem layout (Layout_K_SW128_Atom with Swizzle<3,4,3> + CU_TENSOR_MAP_SWIZZLE_128B). A thread-level reference kernel (`flash_attn_minimal.cu`) is also provided as a secondary correctness baseline.

The kernel implements the FlashAttention-2 online softmax algorithm:
1. Load Q tile from HBM via TMA (`cp.async.bulk.tensor.2d`) into smem
2. For each K/V tile along the sequence dimension:
   a. Load K tile via TMA into smem
   b. Compute `S = Q @ K^T` using wgmma m64n16k16 (f16 inputs, f32 accumulator), iterating over K and N groups
   c. Scatter S from wgmma output registers to smem
   d. Apply online softmax in smem: track running max and running sum, rescale previous O accumulator
   e. Load V tile via TMA into smem
   f. Compute `O += softmax(S) @ V` using wgmma m64n16k16
3. Final normalization and writeback O to global memory

## Algorithmic skeleton

```
HOST: create CUtensorMap for Q, K, V (2D: fastest=HD, slow=BH*S)

DEVICE (1 warpgroup = 128 threads):
for each batch b, head h:
    for each Q-tile (rows [qStart..qStart+BLOCK_M)):
        TMA-load Q[b,h,qStart:qStart+BM, :] → Q_smem
        init: O_acc[4][8] = 0, rmax = -inf, rsum = 0

        for each KV-tile (cols [kvStart..kvStart+BLOCK_N)):
            TMA-load K[b,h,kvStart:kvStart+BN, :] → KV_smem

            // wgmma: S = Q_smem @ K_smem^T
            wgmma.fence
            for ng in 0..3:  // 4 N-groups of 16
                for k in 0..3:   // 4 K-tiles of 16
                    descA = Q_smem[:, k*16], descB = K_smem[ng*16:, k*16:]
                    wgmma.mma_async m64n16k16 S_acc[ng] += A × B  (transA=0, transB=0)
            wgmma.commit; wgmma.wait

            // scatter S from registers to S_smem (f32)
            // thread-level softmax in S_smem → P_smem (f16)
            // rescale O_acc per owned rows

            TMA-load V[b,h,kvStart:kvStart+BN, :] → KV_smem

            // wgmma: O_acc += P_smem @ V_smem
            wgmma.fence
            for ng in 0..3:
                for k in 0..3:
                    descA = P_smem[:, k*16], descB = V_smem[k*16:, ng*16:]
                    wgmma.mma_async m64n16k16 O_acc[ng] += A × B  (transA=0, transB=1)
            wgmma.commit; wgmma.wait

        O_acc /= rsum; global-store O
```

## Tuning parameters

See `tuning.md` for the full parameter space. Fixed config for MVP:
- `BLOCK_M` = 64 (matches wgmma M=64)
- `BLOCK_N` = 64
- `HEAD_DIM` = 64
- `NTHREADS` = 128 (1 warpgroup)

## Measured Characteristics

H200-SXM, sm_90a, cuda 12.9.86, driver 570.124.06. fp16 inputs, f32 accumulator.

**Correctness gate**: `max_abs_err = 0.000061` at B=1, H=2, S=128, D=64, 0/16384 mismatches (>1e-2). `compute-sanitizer memcheck: 0 errors`. Disposition: **Passed**.

**Timing**: kernel_ms = 0.024 at B=1, H=2, S=128, D=64 (12.4x faster than thread-level reference at same config: 0.298 ms).

**Throughput**: 0.35 TFLOPS (total_flops = 4×1×2×128²×64 = 8.39M FLOPs). Far below H200 peak (~989 TFLOPS) due to tiny problem size (2 CTAs on 132 SMs).

**ncu profiler metrics** (representative run at B=1 H=2 S=128 D=64):

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 0.35 TFLOPS | 8.39M FLOPs / 0.024 ms |
| HBM bandwidth | 5.15 GB/s (0.11% of 4800 GB/s peak) | 123,648 B read / 0.024 ms |
| SM utilization | 2.75% peak active cycles | Extreme under-utilization at this size |
| SM throughput | 0.20% peak | |
| Active warps | 6.25% peak sustained | Single warpgroup per CTA |
| Tensor ops (hmma) | 1,024 instructions | 32 wgmma calls |

Evidence:
- Primary probe: 2026-05-08-mvp-attention-tma-wgmma.md
- ncu CSV: `sources/experience/api-probes/attention/artifacts/profiles/tma-wgmma-ncu.csv`
- Correctness log: `sources/experience/api-probes/attention/artifacts/profiles/tma-wgmma-correctness.log`
- Sanitizer log: `sources/experience/api-probes/attention/artifacts/profiles/tma-wgmma-sanitizer.log`

Secondary thread-level reference:
- Probe record: sources/experience/api-probes/attention/2026-05-08-mvp-attention.md
- Tuning sweep: sources/experience/hw-probes/attention-tuning/2026-05-08-attention-tile-sweep.md

## When to use it

- Learning how flash-attention works at the TMA + wgmma primitive level.
- As a starting point for custom attention patterns without cutlass templates.
- Correctness reference for cutlass FMHA or FlashAttention v3.

## When NOT to use it

- Production attention: use cutlass FMHA or FlashAttention v3 for optimized performance.
- Non-Hopper targets: this kernel requires sm_90a (TMA + wgmma).

## How it connects to the rest of the KB

- Builds on `wiki/nvidia/hardware/wgmma-ptx/` (wgmma PTX patterns) and `wiki/nvidia/hardware/tma-ptx/` (TMA load patterns).
- Complemented by `wiki/nvidia/foundations/compute/attention/cutlass-fmha/` (cutlass's production FMHA).
- Compared against FlashAttention v3 at `wiki/nvidia/code-walkthroughs/flash-attention-v3/`.
- Tuning parameter space: tuning.md.
