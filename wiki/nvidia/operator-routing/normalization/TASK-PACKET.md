---
title: Normalization Pattern -- Task Packet Template
pattern_class: cuda-core
op: normalization
status: draft
source:
- path: reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This
    file is the task packet.'
id: routing-normalization-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: normalization
---
# Normalization -- Task Packet Template

This document defines the operator-specific task packet fields for a normalization kernel-writing task. It refines the generic task packet contract in `reasoning/task-packet.md` with normalization-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from reasoning/task-packet.md) ---
task_id: "2026-04-XX-normalization-<variant>"   # date-prefixed, kebab-case
task_type: write-kernel                          # or benchmark-kernel
target_path: kernels/normalization/<variant>/    # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- Normalization-specific fields ---
op: normalization

norm_type: layernorm
  # Which normalization variant. One of:
  #   layernorm   -- Layer Normalization (mean + variance over hidden dim)
  #   rmsnorm     -- Root Mean Square Normalization (RMS over hidden dim, no mean)
  #   batchnorm   -- Batch Normalization (mean + variance over batch + spatial, per channel)
  #   groupnorm   -- Group Normalization (mean + variance over spatial within each group)

shape:
  # Specify the input tensor shape. Format depends on norm_type:
  # - layernorm / rmsnorm: { B: 32, S: 512, H: 1024 }
  #     B = batch size, S = sequence length, H = hidden dimension
  # - batchnorm:           { B: 64, C: 256, H: 14, W: 14 }
  #     B = batch, C = channels, H = height, W = width
  # - groupnorm:           { B: 32, C: 256, H: 14, W: 14, G: 32 }
  #     G = number of groups
  B: 32
  S: 512
  H: 1024

dtype: float32
  # Supported: float16, bfloat16, float32, float64

eps: 1.0e-5
  # Epsilon for numerical stability in the denominator.
  # LayerNorm/BatchNorm typical: 1e-5
  # RMSNorm typical: 1e-6

affine: true
  # Whether learned affine parameters (gamma, beta) are applied.
  # true  --> output = (x - mean) * inv_std * gamma + beta
  # false --> output = (x - mean) * inv_std
  # RMSNorm: beta is typically omitted (only gamma).

baseline: "torch.nn.functional.layer_norm"
  # The library baseline to benchmark against. One of:
  #   torch.nn.functional.layer_norm
  #   torch.nn.functional.rms_norm
  #   torch.nn.functional.batch_norm
  #   torch.nn.functional.group_norm
  #   apex.normalization.FusedLayerNorm
  #   apex.normalization.FusedRMSNorm
  # The custom kernel's latency must approach or beat this baseline.
```

## Optional fields

```yaml
fuse_residual: false
  # If true, the kernel fuses residual addition with normalization:
  #   output = norm(x + residual)
  # The kernel reads both x and residual, adds them, then normalizes.
  # This avoids an extra global-memory round-trip.

fuse_dropout: false
  # If true, the kernel fuses dropout with normalization:
  #   output = norm(dropout(x, p))
  # Requires a PRNG state per thread (e.g., Philox).

training: true
  # For batchnorm: whether to compute batch statistics (training=true)
  # or use running statistics (training=false / inference mode).

momentum: 0.1
  # For batchnorm training: exponential moving average momentum for
  # running mean/variance updates.

num_groups: 32
  # For groupnorm: number of groups G. C must be divisible by G.

welford: false
  # If true, use Welford's online algorithm to compute mean and
  # variance in a single pass (instead of two separate reductions).
  # Saves one global-memory read at the cost of slightly more ALU.

elements_per_thread: 4
  # How many input elements each thread processes in the grid-stride
  # loop. Higher values improve ILP and vectorization opportunities.
  # Default: let the implementation choose.

success_criteria:
  - correctness: "max_abs_error < 1e-3 vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
  # The kernel must be within 10% of the library baseline.
  # Note: normalization correctness threshold is looser than reduction
  # (1e-3 vs 1e-5) because the division by sqrt(var + eps) amplifies
  # floating-point rounding differences.

notes: |
  Free-form guidance for the kernel-writing agent.
  Example: "This layernorm is part of a transformer block; fuse with
  the preceding residual add to avoid a separate kernel launch."
```

## Example: LayerNorm fp32, H=1024

```yaml
task_id: 2026-04-20-normalization-layernorm-fp32-h1024
task_type: write-kernel
target_path: kernels/normalization/layernorm-fp32-h1024/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: normalization
norm_type: layernorm
shape:
  B: 32
  S: 512
  H: 1024
dtype: float32
eps: 1.0e-5
affine: true
baseline: "torch.nn.functional.layer_norm"
success_criteria:
  - correctness: "max_abs_error < 1e-3"
  - performance: "median_latency <= 1.1 * baseline_latency"
references:
  - wiki/nvidia/operator-routing/cuda-core/normalization/INDEX.md
  - wiki/nvidia/operator-routing/cuda-core/normalization/ROUTING.md
  - reasoning/bottleneck-triage.md
```

## Example: RMSNorm bf16, H=4096 with fused residual

```yaml
task_id: 2026-04-21-normalization-rmsnorm-bf16-h4096-fused
task_type: write-kernel
target_path: kernels/normalization/rmsnorm-bf16-h4096-fused/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: normalization
norm_type: rmsnorm
shape:
  B: 8
  S: 2048
  H: 4096
dtype: bfloat16
eps: 1.0e-6
affine: true
fuse_residual: true
baseline: "apex.normalization.FusedRMSNorm"
success_criteria:
  - correctness: "max_abs_error < 5e-3"
  - performance: "median_latency <= 1.05 * baseline_latency"
notes: |
  Fused residual + RMSNorm for LLaMA-style transformer block.
  The kernel reads both x and residual from global memory, adds them,
  computes RMS, and writes the normalized output. This saves one
  global-memory round-trip vs separate add + rmsnorm.
  Use vectorized float4 loads (H=4096 is divisible by 4).
  Consider Welford's algorithm since we only need sum-of-squares.
references:
  - wiki/nvidia/operator-routing/cuda-core/normalization/INDEX.md
  - wiki/nvidia/operator-routing/cuda-core/normalization/ROUTING.md
  - reasoning/bottleneck-triage.md
```

## Example: BatchNorm fp32, NCHW

```yaml
task_id: 2026-04-22-normalization-batchnorm-fp32-nchw
task_type: write-kernel
target_path: kernels/normalization/batchnorm-fp32-nchw/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: normalization
norm_type: batchnorm
shape:
  B: 64
  C: 256
  H: 14
  W: 14
dtype: float32
eps: 1.0e-5
affine: true
training: true
momentum: 0.1
baseline: "torch.nn.functional.batch_norm"
success_criteria:
  - correctness: "max_abs_error < 1e-4"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Standard BatchNorm for a ResNet-style conv layer.
  The reduction spans (B, H, W) for each of C=256 channels.
  Each block handles one channel.
  Consider NHWC layout for better coalescing on Hopper.
references:
  - wiki/nvidia/operator-routing/cuda-core/normalization/INDEX.md
  - wiki/nvidia/operator-routing/cuda-core/normalization/ROUTING.md
  - reasoning/bottleneck-triage.md
```

## Agent workflow when receiving a normalization task packet

1. Read `wiki/nvidia/operator-routing/cuda-core/normalization/INDEX.md` -- follow the decision tree starting at Step 0.
2. If the library path suffices, report the recommended library call and stop (no custom kernel needed).
3. If a custom kernel is needed, read `ROUTING.md` for the skill whitelist, then implement the kernel following Step 1/Step 2 of INDEX.md.
4. Benchmark against `baseline` and apply bottleneck triage (`reasoning/bottleneck-triage.md`) if the success criteria are not met.
5. After at most 3 optimization iterations, finalize or report `status: stuck`.
