---
title: Pooling Pattern -- Task Packet Template
pattern_class: cuda-core
op: pooling
status: draft
source:
- path: knowledge/70-reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This
    file is the task packet.'
id: routing-pooling-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: pooling
---
# Pooling -- Task Packet Template

This document defines the operator-specific task packet fields for a pooling kernel-writing task. It refines the generic task packet contract in `70-reasoning/task-packet.md` with pooling-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from 70-reasoning/task-packet.md) ---
task_id: "2026-04-XX-pooling-<variant>"        # date-prefixed, kebab-case
task_type: write-kernel                         # or benchmark-kernel
target_path: kernels/pooling/<variant>/         # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- Pooling-specific fields ---
op: pooling

pool_type: max
  # The pooling variant. One of:
  #   max       -- maximum over the window
  #   avg       -- average over the window (sum / window_area)
  #   adaptive  -- output size is fixed; window size is computed per position

shape:
  # Input tensor shape in NCHW format.
  B: 32            # batch size
  C: 64            # channels
  H: 224           # input height
  W: 224           # input width

dtype: float32
  # Supported: float16, bfloat16, float32, float64

kernel_size: [3, 3]
  # Pooling window size [kH, kW] (or scalar for square windows).
  # For adaptive pooling, this is computed from input/output sizes and
  # should be omitted; specify output_size instead.

stride: [2, 2]
  # Stride [sH, sW] (or scalar for equal strides).
  # Default: same as kernel_size (non-overlapping windows).

padding: [1, 1]
  # Zero-padding [pH, pW] applied to the input before pooling.
  # Default: [0, 0] (no padding).

baseline: "torch.nn.functional.max_pool2d"
  # The library baseline to benchmark against. One of:
  #   torch.nn.functional.max_pool2d
  #   torch.nn.functional.avg_pool2d
  #   torch.nn.functional.adaptive_avg_pool2d
  #   torch.nn.functional.adaptive_max_pool2d
  #   cudnnPoolingForward
  # The custom kernel's latency must approach or beat this baseline
  # to justify its existence (see INDEX.md Step 0).
```

## Optional fields

```yaml
output_size: [7, 7]
  # Required for pool_type: adaptive. Specifies the target output
  # spatial dimensions (oH, oW). The effective kernel_size and stride
  # are computed per output position.

count_include_padding: true
  # For avg pooling only. Whether zero-padded positions are included
  # in the divisor. Default: true.

dilation: [1, 1]
  # Dilation factor for the pooling window. Default: [1, 1] (no dilation).
  # Dilated pooling (atrous pooling) expands the effective window size.

ceil_mode: false
  # If true, use ceil instead of floor for output size computation.
  # Default: false.

return_indices: false
  # For max-pool only. If true, the kernel also outputs the indices
  # of the maximum elements (needed for max_unpool). Default: false.

layout: NCHW
  # Memory layout of the input/output tensors.
  # NCHW: channels before spatial dims (PyTorch default).
  # NHWC: channels last (sometimes faster on GPU for certain ops).
  # Default: NCHW.

use_shared_memory: auto
  # Whether to stage input tiles in shared memory.
  # auto  -- use shared memory when stride < kernel_size (overlapping windows).
  # always -- force shared memory tiling.
  # never -- direct global reads only.
  # Default: auto.

success_criteria:
  - correctness: "max_abs_error < 1e-5 vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
  # The kernel must be within 10% of the library baseline.
  # If it cannot meet this bar, the library fallback should be used.

notes: |
  Free-form guidance for the kernel-writing agent.
  Example: "This pooling kernel will be fused with a subsequent ReLU
  activation to avoid an extra global memory round-trip."
```

## Example: 2D max-pool (ImageNet-style)

```yaml
task_id: 2026-04-20-pooling-maxpool2d-imagenet
task_type: write-kernel
target_path: kernels/pooling/maxpool2d-imagenet/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: pooling
pool_type: max
shape:
  B: 32
  C: 64
  H: 224
  W: 224
dtype: float32
kernel_size: [3, 3]
stride: [2, 2]
padding: [1, 1]
baseline: "torch.nn.functional.max_pool2d"
success_criteria:
  - correctness: "max_abs_error < 1e-5"
  - performance: "median_latency <= 1.1 * baseline_latency"
references:
  - 20-pattern/cuda-core/pooling/INDEX.md
  - 20-pattern/cuda-core/pooling/ROUTING.md
  - 70-reasoning/bottleneck-triage.md
```

## Example: global average pooling (adaptive)

```yaml
task_id: 2026-04-21-pooling-global-avgpool
task_type: write-kernel
target_path: kernels/pooling/global-avgpool/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: pooling
pool_type: adaptive
shape:
  B: 32
  C: 2048
  H: 7
  W: 7
dtype: float32
output_size: [1, 1]
baseline: "torch.nn.functional.adaptive_avg_pool2d"
success_criteria:
  - correctness: "max_abs_error < 1e-5"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Global average pooling: reduce each (7, 7) spatial map to a single
  scalar per channel. This is a per-channel reduction of 49 elements.
  Since 49 > 32, this needs a block-level reduction if one block handles
  one (n, c) pair, or a single-thread serial reduction since 49 is small.
  Consider assigning one warp per (n, c) pair: 32 threads reduce 49
  elements (first 32 loaded, then the remaining 17 by a subset of threads),
  followed by a warp-level butterfly reduction via __shfl_xor_sync.
```

## Example: 2D avg-pool with padding

```yaml
task_id: 2026-04-22-pooling-avgpool2d-padded
task_type: write-kernel
target_path: kernels/pooling/avgpool2d-padded/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: pooling
pool_type: avg
shape:
  B: 16
  C: 128
  H: 56
  W: 56
dtype: float32
kernel_size: [3, 3]
stride: [1, 1]
padding: [1, 1]
count_include_padding: false
baseline: "torch.nn.functional.avg_pool2d"
success_criteria:
  - correctness: "max_abs_error < 1e-6"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Stride-1 pooling with 3x3 window: maximum overlap. Each input element
  is read by up to 9 output elements. Shared memory tiling is strongly
  recommended to amortize redundant global reads. count_include_padding=false
  means the divisor varies at the borders.
```

## Agent workflow when receiving a pooling task packet

1. Read `20-pattern/cuda-core/pooling/INDEX.md` -- follow the decision tree starting at Step 0.
2. If the library path suffices, report the recommended library call and stop (no custom kernel needed).
3. If a custom kernel is needed, read `ROUTING.md` for the skill whitelist, then implement the kernel following Step 1/Step 2 of INDEX.md.
4. Benchmark against `baseline` and apply bottleneck triage (`70-reasoning/bottleneck-triage.md`) if the success criteria are not met.
5. After at most 3 optimization iterations, finalize or report `status: stuck`.
