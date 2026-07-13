---
title: Normalization Pattern -- Decision Tree
pattern_class: cuda-core
op: normalization
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: spec
  anchor: Reference
id: routing-normalization-INDEX
type: operator-routing
vendor: nvidia
operator: normalization
source_refs:
- source_id: source-code/cuda-samples
  path: Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
  anchor: L75-L81
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23945-L23973
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cub/cub.md
  anchor: L5931-L5997
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- vectorized-loads
- kernel-fusion
- shared-memory-optimization
kernel_types:
- fused-kernel
confidence: inferred
tags:
- vectorized-loads
- kernel-fusion
- shared-memory-optimization
- fused-kernel
- cuda-cpp
---
# Normalization Pattern -- Decision Tree

This document guides the kernel-writing agent through a normalization task (LayerNorm, RMSNorm, BatchNorm, GroupNorm) from a custom-kernel requirement to a working, optimized kernel.

## Background -- what normalization operators share

All four normalization variants follow a two-phase structure:

1. **Reduction phase** -- compute statistics (mean, variance, or root-mean-square) over a normalization axis. This is a per-token / per-channel / per-group reduction identical in structure to the reduction pattern (`wiki/nvidia/operator-routing/cuda-core/reduction/`).
2. **Elementwise scaling phase** -- subtract the mean (if applicable), divide by `sqrt(variance + eps)`, and optionally apply learned affine parameters (gamma, beta). This is a pointwise map over every element.

The reduction axis distinguishes the variants:

| Variant    | Typical shape          | Reduction axis        | Statistics per |
|------------|------------------------|-----------------------|----------------|
| LayerNorm  | (B, S, H)             | H (hidden dim)        | token          |
| RMSNorm    | (B, S, H)             | H (hidden dim)        | token          |
| BatchNorm  | (B, C, H, W)          | B, H, W               | channel        |
| GroupNorm  | (B, C, H, W), G groups| H, W within each group| group          |

Because the kernel fuses both phases (reduction + scaling) into a single launch, custom normalization kernels avoid the extra global-memory round-trip that separate reduce-then-scale would incur. This fusion is the primary reason custom kernels outperform naive compositions.

## Scope

This decision tree covers custom-kernel implementation choices only. It starts after the task has been classified as requiring a dedicated kernel implementation.

## Step 1 -- Choose the custom kernel strategy

All normalization variants share the same kernel skeleton. The choice depends on the reduction width (number of elements reduced per output normalization instance).

```
Q4. How many elements per normalization instance (reduction width)?
    <=32 (single warp)
        --> Single-warp kernel: one warp handles one normalization
            instance. Each lane holds one or more elements. Reduce via
            __shfl_down_sync. No shared memory needed for the reduction.
            Ideal for small hidden dims (H <= 32).

    33..1024 (single block, one instance per block)
        --> Block-level kernel: one thread block handles one instance.
            Phase 1: each thread loads and accumulates elements via
            a grid-stride loop within the instance.
            Phase 2: block-level reduction via warp shuffle + shared
            memory (same as reduction pattern Step 1, 33..1024 case).
            Phase 3: broadcast mean/variance to all threads, apply
            elementwise scaling.
            This is the most common case for transformer LayerNorm /
            RMSNorm where H is typically 768, 1024, 2048, 4096, etc.

    >1024 (multiple blocks per instance)
        --> Multi-block kernel: multiple blocks cooperate on one
            normalization instance. Requires a two-pass approach:
            Pass 1: each block reduces its tile, writes partial sums
            to global memory.
            Pass 2: a single block reduces the partials and broadcasts
            the statistics.
            Pass 3 (or fused into Pass 2): apply elementwise scaling.
            Rare in practice -- most norm hidden dims fit in a single
            block. Consider this only for very large hidden dims.
```

## Step 1b -- Choose dtypes (precision/range decision; correctness-critical)

Normalization has a **reduction phase** (mean / variance / RMS over the axis) and an **elementwise scaling phase** (`(x - mean) * rstd * gamma + beta`). These two phases have different numerical requirements and are usually assigned different dtypes. This is a **correctness decision**, not a perf tuning step. Reference: `wiki/nvidia/foundations/compute/half-precision-math/` §Precision.

```
Q3a. Storage dtype (input activation, gamma, beta, output) — fp16 / bf16 / fp32?
     - fp16: 2x memory BW vs fp32, ULP ~4.88e-4 (~3 decimal digits).
       Safe if |activation| in [6.1e-5, 6.55e4]. Typical for inference.
     - bf16: 2x memory BW, fp32-level dynamic range, ULP ~7.81e-3 (~2 decimal digits).
       Preferred for training (no 65k ceiling); coarser than fp16.
     - fp32: 4x memory BW cost but no precision concerns. Use when any of:
       (a) activation magnitudes cross fp16's 6.55e4 ceiling AND bf16's coarser
           mantissa is unacceptable for downstream (e.g. per-channel scale near 0);
       (b) user provides fp32 tensors already.
     Default for transformer LayerNorm / RMSNorm: bf16 (training) or fp16 (inference);
     BatchNorm weights typically fp32 (small tensor, negligible memory cost).

Q3b. Accumulator dtype (mean / variance / RMS partial sums) — ALWAYS fp32.
     DO NOT accumulate in fp16 or bf16 regardless of storage dtype. Reasons:
     - fp16 overflow: sum of H terms of magnitude ~1 hits the 6.55e4 ceiling at
       H > 65504 (catastrophic) and accumulates 1 ULP of noise at H ≈ 2000
       (silent correctness drift).
     - bf16 mantissa: 7-bit mantissa → ~0.8% relative error per add; after H=4096
       adds the mean drifts by a full decimal digit. Not overflow, but wrong.
     - Canonical pattern: `float acc = 0.0f; acc += __half2float(row[i]);` inside
       the reduction loop; produce `mean, rstd` in fp32; cast back at store.
     See half-precision-math §S3 for the full mixed-precision shape.

Q3c. Epsilon — verify against the storage dtype's min-normal.
     - fp16 min-normal ≈ 6.1e-5. Any eps < 1e-4 risks denormal FTZ (flush-to-zero)
       making `sqrt(var + eps)` produce 0 when var ≈ 0, leading to inf rstd.
     - bf16 min-normal is fp32-level (1.18e-38), so denormal FTZ is never an
       issue. But the mantissa is coarse: if `eps << var`, bf16 may round
       `var + eps` down to just `var`, losing the epsilon's stabilising effect.
     - Compute (var + eps) and the rsqrt in fp32 regardless of storage dtype.

Q3d. Output cast — round-to-nearest-even.
     After computing `(x - mean) * rstd * gamma + beta` in fp32, cast to the
     storage dtype with `__float2half_rn` / `__float2bfloat16_rn`. Do NOT cast
     intermediates inside the expression (each cast is a rounding step and
     compounds error).

Q3e. Atomics for multi-block reduction (Q4 >1024 case) — fp32 destination.
     Even if final output is fp16/bf16, the grid-level combine atomic must be
     fp32. Native fp16 atomicAdd has worse contention throughput AND worse
     numerics (legacy half-precision-math P12). Pattern: one fp32 atomic per
     block into a fp32 partial-sum buffer, cast to fp16/bf16 in a final
     kernel's elementwise store.
```

## Step 2 -- Kernel skeleton (LayerNorm / RMSNorm)

The canonical single-block normalization kernel follows this structure:

```cuda
// One block per normalization instance (one row of shape [B*S, H])
// blockDim.x = min(H, 1024), one thread per element (or grid-stride if H > blockDim.x)

__global__ void layernorm_kernel(
    const float* __restrict__ input,    // [N, H]
    const float* __restrict__ gamma,    // [H]
    const float* __restrict__ beta,     // [H]
    float* __restrict__ output,         // [N, H]
    int H, float eps)
{
    int row = blockIdx.x;  // one block per row
    const float* row_in = input + row * H;
    float* row_out = output + row * H;

    // Phase 1: compute mean (reduction over H)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        local_sum += row_in[i];
    // block-level reduce (warp shuffle + smem)
    float mean = blockReduceSum(local_sum) / H;

    // Phase 2: compute variance (reduction over H)
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    float variance = blockReduceSum(local_var) / H;
    float inv_std = rsqrtf(variance + eps);

    // Phase 3: elementwise scaling (no reduction, pure map)
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
}
```

For **RMSNorm**, skip the mean computation; compute only the root-mean-square: `rms = sqrt(sum(x_i^2) / H + eps)`, then scale by `x_i / rms * gamma[i]`. This saves one reduction pass and one global read pass, making RMSNorm approximately 1.5x faster than LayerNorm for the same shape.

For **BatchNorm**, the reduction axis spans (B, H, W) for each channel, so the kernel layout changes: one block per channel, threads iterate over spatial locations and batch elements. Running mean/variance must be maintained for inference mode.

For **GroupNorm**, the reduction is over (H/G, W) spatial dims within each group, with G groups per channel dimension. Kernel layout: one block per (batch, group) pair.

## Step 3 -- Optimization via ROUTING.md skills

After the basic custom kernel is working and correct, apply optimization skills from ROUTING.md in priority order:

1. **Warp primitives** (`wiki/nvidia/foundations/compute/warp-primitives/`) -- the reduction phase must use `__shfl_down_sync` butterfly reduction within each warp, avoiding shared-memory round-trips for intra-warp communication.

2. **Coalescing** (`wiki/nvidia/foundations/memory/coalescing/`) -- the elementwise scaling phase reads input[row, i] and writes output[row, i] with stride-1 across threads. Verify that the load/store phase is coalesced. For BatchNorm, the NCHW layout may cause non-coalesced access along the spatial axes; consider NHWC layout.

3. **Vectorized access** (`wiki/nvidia/foundations/memory/vectorized-access/`) -- load/store `float4` (128-bit) in the elementwise phase to increase bytes-in-flight. Requires H to be divisible by 4 and 16-byte alignment.

4. **Bank-conflict avoidance** (`wiki/nvidia/foundations/memory/bank-conflict/`) -- if the block-level reduction uses shared memory for inter-warp communication (warp leaders writing partials to smem[warpId]), ensure no bank conflicts.

After each skill application, re-benchmark against the task-provided baseline and follow the bottleneck-triage procedure in `reasoning/bottleneck-triage.md`.

## Step 4 -- Advanced techniques

- **Two-pass to one-pass variance**: Welford's online algorithm computes mean and variance in a single pass, avoiding a second read of the input. Trades one extra global-memory read for slightly more per-element arithmetic. Beneficial when H is large and the kernel is memory-bandwidth-bound.

- **Fused residual + normalization**: for transformer architectures, the pattern `x = layernorm(x + residual)` can be fused into a single kernel that reads residual and x, adds them, computes statistics, and writes the normalized output. This halves the global-memory traffic compared to separate add + norm kernels.

## Cross-references

- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Reduction pattern (shared primitive)**: `wiki/nvidia/operator-routing/cuda-core/reduction/INDEX.md`
- **Bottleneck triage after benchmarking**: `reasoning/bottleneck-triage.md`
