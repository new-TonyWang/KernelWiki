---
title: Normalization Pattern -- Decision Tree
pattern_class: cuda-core
op: normalization
status: draft
hardware:
  device: H200
  sm: 9.0a
source:
- path: '{{CUDA_SAMPLES_REPO_REF}}/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu'
  anchor: L75-L81
  excerpt: warpReduceSum using __shfl_down_sync with offset halving loop -- same primitive
    used for the reduction phase of normalization
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23945-L23973
  excerpt: 'Warp Shuffle Functions: __shfl_down_sync copies from a lane with a higher
    ID'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cub/cub.md
  anchor: L5931-L5997
  excerpt: cub::BlockReduce provides collective methods for computing a parallel reduction
    across a CUDA thread block
id: routing-normalization-INDEX
type: operator-routing
vendor: nvidia
operator: normalization
---
# Normalization Pattern -- Decision Tree

This document guides the kernel-writing agent through a normalization task (LayerNorm, RMSNorm, BatchNorm, GroupNorm) from initial problem statement to a working, optimized kernel. The decision tree enforces a **library-first** policy: only proceed to a custom kernel when the library path has been proven insufficient.

## Background -- what normalization operators share

All four normalization variants follow a two-phase structure:

1. **Reduction phase** -- compute statistics (mean, variance, or root-mean-square) over a normalization axis. This is a per-token / per-channel / per-group reduction identical in structure to the reduction pattern (`20-pattern/cuda-core/reduction/`).
2. **Elementwise scaling phase** -- subtract the mean (if applicable), divide by `sqrt(variance + eps)`, and optionally apply learned affine parameters (gamma, beta). This is a pointwise map over every element.

The reduction axis distinguishes the variants:

| Variant    | Typical shape          | Reduction axis        | Statistics per |
|------------|------------------------|-----------------------|----------------|
| LayerNorm  | (B, S, H)             | H (hidden dim)        | token          |
| RMSNorm    | (B, S, H)             | H (hidden dim)        | token          |
| BatchNorm  | (B, C, H, W)          | B, H, W               | channel        |
| GroupNorm  | (B, C, H, W), G groups| H, W within each group| group          |

Because the kernel fuses both phases (reduction + scaling) into a single launch, custom normalization kernels avoid the extra global-memory round-trip that separate reduce-then-scale would incur. This fusion is the primary reason custom kernels outperform naive compositions.

## Step 0 -- Try the library first

Before writing any custom CUDA code, check whether a production-quality library already handles the normalization.

```
Q0. Is the caller's environment PyTorch-based?
    YES --> Can torch.nn.functional.layer_norm / rms_norm /
            batch_norm / group_norm handle the shape + dtype?
            YES --> Use the PyTorch op. DONE.
            NO  --> Continue to Q1.
    NO  --> Continue to Q1.

Q1. Is the normalization a standard BatchNorm and is cuDNN available?
    YES --> Use cudnnBatchNormalizationForwardTraining (train) or
            cudnnBatchNormalizationForwardInference (eval).
            See library-fallback.md for API details. DONE.
    NO  --> Continue to Q2.

Q2. Is an optimized fused kernel available via a third-party library
    (e.g., Apex fused LayerNorm / RMSNorm)?
    YES --> Benchmark the fused kernel against the PyTorch op.
            If faster, use it. DONE.
    NO  --> Continue to Q3.

Q3. Does the library path fail to meet performance requirements after
    benchmarking (e.g., fused normalization + residual add, custom
    norm variant, or measured >10% overhead vs. theoretical peak)?
    YES --> Proceed to Step 1 (custom kernel).
    NO  --> Re-examine the library path. Most normalization workloads
            in transformer models are well-served by PyTorch or Apex.
            Only proceed to custom if benchmark evidence shows the
            library is insufficient.
```

**When to skip the library**: the library path is insufficient when:
- The normalization must be fused with adjacent operations (e.g., residual add + layernorm, or layernorm + dropout) to avoid an extra global-memory round-trip.
- A non-standard normalization variant is needed (e.g., RMSNorm before PyTorch native support, or a custom normalization axis).
- The measured library latency exceeds the theoretical bandwidth-bound limit by more than 10% for the given shape.
- The model uses a custom epsilon, clamping, or other non-standard behavior not supported by the library.

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

Normalization has a **reduction phase** (mean / variance / RMS over the axis) and an **elementwise scaling phase** (`(x - mean) * rstd * gamma + beta`). These two phases have different numerical requirements and are usually assigned different dtypes. This is a **correctness decision**, not a perf tuning step. Reference: [`30-skill/compute/half-precision-math/`](../../../30-skill/compute/half-precision-math/) §Precision.

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

1. **Warp primitives** (`30-skill/compute/warp-primitives/`) -- the reduction phase must use `__shfl_down_sync` butterfly reduction within each warp, avoiding shared-memory round-trips for intra-warp communication.

2. **Coalescing** (`30-skill/memory/coalescing/`) -- the elementwise scaling phase reads input[row, i] and writes output[row, i] with stride-1 across threads. Verify that the load/store phase is coalesced. For BatchNorm, the NCHW layout may cause non-coalesced access along the spatial axes; consider NHWC layout.

3. **Vectorized access** (`30-skill/memory/vectorized-access/`) -- load/store `float4` (128-bit) in the elementwise phase to increase bytes-in-flight. Requires H to be divisible by 4 and 16-byte alignment.

4. **Bank-conflict avoidance** (`30-skill/memory/bank-conflict/`) -- if the block-level reduction uses shared memory for inter-warp communication (warp leaders writing partials to smem[warpId]), ensure no bank conflicts.

After each skill application, re-benchmark against the baseline (torch.nn.functional op or Apex fused kernel) and follow the bottleneck-triage procedure in `70-reasoning/bottleneck-triage.md`.

## Step 4 -- Advanced techniques

- **Two-pass to one-pass variance**: Welford's online algorithm computes mean and variance in a single pass, avoiding a second read of the input. Trades one extra global-memory read for slightly more per-element arithmetic. Beneficial when H is large and the kernel is memory-bandwidth-bound.

- **Fused residual + normalization**: for transformer architectures, the pattern `x = layernorm(x + residual)` can be fused into a single kernel that reads residual and x, adds them, computes statistics, and writes the normalized output. This halves the global-memory traffic compared to separate add + norm kernels.

## Cross-references

- **Library fallback details**: `library-fallback.md`
- **Skill whitelist for this pattern**: `ROUTING.md`
- **Task packet template**: `TASK-PACKET.md`
- **Reduction pattern (shared primitive)**: `20-pattern/cuda-core/reduction/INDEX.md`
- **Bottleneck triage after benchmarking**: `70-reasoning/bottleneck-triage.md`
