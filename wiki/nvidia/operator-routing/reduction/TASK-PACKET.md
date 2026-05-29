---
title: Reduction Pattern -- Task Packet Template
pattern_class: cuda-core
op: reduction
status: draft
source:
- path: reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This
    file is the task packet.'
id: routing-reduction-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: reduction
---
# Reduction -- Task Packet Template

This document defines the operator-specific task packet fields for a reduction kernel-writing task. It refines the generic task packet contract in `reasoning/task-packet.md` with reduction-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from reasoning/task-packet.md) ---
task_id: "2026-04-XX-reduction-<variant>"      # date-prefixed, kebab-case
task_type: write-kernel                         # or benchmark-kernel
target_path: kernels/reduction/<variant>/       # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- Reduction-specific fields ---
op: reduction

shape:
  # Specify the input tensor shape. Use one of:
  # - Flat (1-D): { N: 1048576 }
  # - 2-D:        { M: 128, N: 4096 }
  # - General:    { dims: [B, M, N], sizes: [32, 128, 4096] }
  N: 1048576

dtype: float32
  # Supported: float16, bfloat16, float32, float64, int32, int64

reduction_axis: -1
  # Which axis to reduce over.
  # -1 means "all axes" (full/flat reduction to a scalar).
  # 0, 1, 2, ... for axis-specific reduction (e.g., per-row reduction
  # of a 2-D tensor uses reduction_axis: 1).

reduction_op: sum
  # The reduction operator. One of:
  #   sum     -- additive reduction (most common)
  #   max     -- maximum reduction
  #   min     -- minimum reduction
  #   mean    -- average (sum / count)
  #   argmax  -- index of maximum element
  #   argmin  -- index of minimum element
  #   prod    -- product reduction (rare, overflow-prone)
  #   custom  -- user-defined binary op (specify in notes)

baseline: "torch.sum"
  # The library baseline to benchmark against. One of:
  #   torch.sum, torch.mean, torch.max, torch.min,
  #   torch.argmax, torch.argmin,
  #   cub::DeviceReduce::Sum, cub::DeviceReduce::Min, etc.
  # The custom kernel's latency must approach or beat this baseline
  # to justify its existence (see INDEX.md Step 0).
```

## Optional fields

```yaml
keepdim: false
  # If true, the output tensor retains the reduced dimension with size 1.
  # Relevant for axis-specific reductions.

contiguous_axis: true
  # Whether the reduction axis is the innermost (contiguous) dimension
  # of the input tensor in memory. If false, the kernel must handle
  # strided access or transpose the data before reducing.

multi_block_strategy: "two-pass"
  # For grid-level reductions (shape.N > 1024):
  #   two-pass       -- launch block-reduce kernel, then reduce partials
  #   atomic          -- each block does atomicAdd to a single output
  #   thread-fence    -- single-kernel with __threadfence() last-block pattern
  # Default: two-pass (safest, deterministic for floats)

elements_per_thread: 4
  # Brent's theorem tuning: how many input elements each thread
  # accumulates serially before the parallel reduction tree.
  # Higher values improve arithmetic intensity for large N.
  # Default: let the implementation choose.

success_criteria:
  - correctness: "max_abs_error < 1e-5 vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
  # The kernel must be within 10% of the library baseline.
  # If it cannot meet this bar, the library fallback should be used.

notes: |
  Free-form guidance for the kernel-writing agent.
  Example: "This reduction is part of a layernorm kernel; the
  per-row sum must be fused with the subsequent normalization pass
  to avoid writing and re-reading the intermediate sum."
```

## Example: flat fp32 sum reduction

```yaml
task_id: 2026-04-20-reduction-flat-fp32-sum
task_type: write-kernel
target_path: kernels/reduction/flat-fp32-sum/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: reduction
shape:
  N: 1048576
dtype: float32
reduction_axis: -1
reduction_op: sum
baseline: "torch.sum"
success_criteria:
  - correctness: "max_abs_error < 1e-5"
  - performance: "median_latency <= 1.1 * baseline_latency"
references:
  - wiki/nvidia/operator-routing/cuda-core/reduction/INDEX.md
  - wiki/nvidia/operator-routing/cuda-core/reduction/ROUTING.md
  - reasoning/bottleneck-triage.md
```

## Example: per-row fp32 max reduction

```yaml
task_id: 2026-04-21-reduction-row-fp32-max
task_type: write-kernel
target_path: kernels/reduction/row-fp32-max/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: reduction
shape:
  M: 256
  N: 4096
dtype: float32
reduction_axis: 1
reduction_op: max
baseline: "torch.max"
keepdim: false
contiguous_axis: true
success_criteria:
  - correctness: "exact match vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Per-row max of a (256, 4096) tensor.  Each row has 4096 elements,
  requiring a block-level reduction.  Consider assigning one block
  per row (blockDim.x = 128, each thread processes 32 elements).
```

## Agent workflow when receiving a reduction task packet

1. Read `wiki/nvidia/operator-routing/cuda-core/reduction/INDEX.md` -- follow the decision tree starting at Step 0.
2. If the library path suffices, report the recommended library call and stop (no custom kernel needed).
3. If a custom kernel is needed, read `ROUTING.md` for the skill whitelist, then implement the kernel following Step 1/Step 2 of INDEX.md.
4. Benchmark against `baseline` and apply bottleneck triage (`reasoning/bottleneck-triage.md`) if the success criteria are not met.
5. After at most 3 optimization iterations, finalize or report `status: stuck`.
