---
title: Scan (Cumulative) Pattern -- Task Packet Template
pattern_class: cuda-core
op: scan-cumulative
status: draft
source:
- path: reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This file is the task packet.'
id: routing-scan-cumulative-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: scan-cumulative
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- pipeline-stages
- kernel-fusion
kernel_types:
- fused-kernel
confidence: inferred
tags:
- pipeline-stages
- kernel-fusion
- fused-kernel
- cuda-cpp
---
# Scan (Cumulative) -- Task Packet Template

This document defines the operator-specific task packet fields for a scan (prefix sum / cumulative) kernel-writing task. It refines the generic task packet contract in `reasoning/task-packet.md` with scan-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from reasoning/task-packet.md) ---
task_id: "2026-04-XX-scan-<variant>"           # date-prefixed, kebab-case
task_type: write-kernel                         # or benchmark-kernel
target_path: kernels/scan/<variant>/            # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- Scan-specific fields ---
op: scan

scan_type: inclusive
  # One of:
  #   inclusive  -- Output[i] = Op(Input[0], Input[1], ..., Input[i])
  #   exclusive  -- Output[i] = Op(Input[0], Input[1], ..., Input[i-1])
  #                 Output[0] = identity element (e.g., 0 for sum, INT_MAX for min)
  # Default: inclusive

scan_op: sum
  # The binary scan operator. One of:
  #   sum     -- additive prefix sum (most common)
  #   max     -- prefix maximum
  #   min     -- prefix minimum
  #   prod    -- prefix product (rare, overflow-prone)
  #   custom  -- user-defined binary op (specify functor in notes)
  # The operator must be associative. Commutativity is not required
  # (CUB supports non-commutative operators since CCCL 2.2.0).

shape:
  # Specify the input tensor shape. Use one of:
  # - Flat (1-D): { N: 1048576 }
  # - 2-D:        { M: 128, N: 4096 }
  # - General:    { dims: [B, M, N], sizes: [32, 128, 4096] }
  N: 1048576

dtype: float32
  # Supported: float16, bfloat16, float32, float64, int32, int64

scan_axis: -1
  # Which axis to scan along.
  # -1 means "all axes" (flat 1-D scan over the entire buffer).
  # 0, 1, 2, ... for axis-specific scan (e.g., per-row scan of a
  # 2-D tensor uses scan_axis: 1).

baseline: "torch.cumsum"
  # The library baseline to benchmark against. One of:
  #   torch.cumsum, torch.cumprod, torch.cummax, torch.cummin,
  #   cub::DeviceScan::InclusiveSum, cub::DeviceScan::ExclusiveSum,
  #   cub::DeviceScan::InclusiveScan, cub::DeviceScan::ExclusiveScan
  # The custom kernel's latency must approach or beat this baseline
  # to justify its existence (see INDEX.md Step 0).
```

## Optional fields

```yaml
init_value: 0
  # For exclusive scans, the identity/initial value.
  # Default: 0 for sum, INT_MAX/FLT_MAX for min, INT_MIN/-FLT_MAX for max.

segmented: false
  # If true, this is a segmented scan. Segments are defined by one of:
  #   segment_offsets: a device array of segment start indices
  #   segment_flags:   a device array of boolean flags (1 = new segment)
  # Segmented scans reset the running accumulator at each segment boundary.

contiguous_axis: true
  # Whether the scan axis is the innermost (contiguous) dimension
  # of the input tensor in memory. If false, the kernel must handle
  # strided access or transpose the data before scanning.

multi_block_strategy: "three-pass"
  # For grid-level scans (shape.N > 1024):
  #   three-pass           -- scan tiles, scan block totals, uniform add
  #   decoupled-lookback   -- single-pass with inter-block lookback
  # Default: three-pass (simpler, deterministic for floats)

elements_per_thread: 4
  # How many input elements each thread processes serially before
  # the parallel scan tree. Higher values reduce launch overhead
  # for large N. Default: let the implementation choose.

in_place: false
  # If true, the scan may be performed in-place (output overwrites input).
  # Both CUB and the custom kernel support this.

success_criteria:
  - correctness: "max_abs_error < 1e-5 vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
  # The kernel must be within 10% of the library baseline.
  # If it cannot meet this bar, the library fallback should be used.

notes: |
  Free-form guidance for the kernel-writing agent.
  Example: "This scan is part of a stream-compaction pipeline;
  the exclusive prefix sum produces scatter indices for a subsequent
  gather kernel. Fusion with the predicate evaluation is desired."
```

## Example: flat fp32 inclusive prefix sum

```yaml
task_id: 2026-04-20-scan-flat-fp32-inclusive-sum
task_type: write-kernel
target_path: kernels/scan/flat-fp32-inclusive-sum/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: scan
scan_type: inclusive
scan_op: sum
shape:
  N: 1048576
dtype: float32
scan_axis: -1
baseline: "torch.cumsum"
success_criteria:
  - correctness: "max_abs_error < 1e-5"
  - performance: "median_latency <= 1.1 * baseline_latency"
references:
  - wiki/nvidia/operator-routing/cuda-core/scan-cumulative/INDEX.md
  - wiki/nvidia/operator-routing/cuda-core/scan-cumulative/ROUTING.md
  - reasoning/bottleneck-triage.md
```

## Example: flat int32 exclusive prefix sum (stream compaction)

```yaml
task_id: 2026-04-21-scan-flat-int32-exclusive-sum
task_type: write-kernel
target_path: kernels/scan/flat-int32-exclusive-sum/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: scan
scan_type: exclusive
scan_op: sum
shape:
  N: 4194304
dtype: int32
scan_axis: -1
baseline: "cub::DeviceScan::ExclusiveSum"
multi_block_strategy: "three-pass"
success_criteria:
  - correctness: "exact match vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Exclusive prefix sum for stream compaction scatter indices.
  Integer dtype so exact correctness is required (no FP rounding).
```

## Example: per-row fp32 inclusive prefix sum

```yaml
task_id: 2026-04-22-scan-row-fp32-inclusive-sum
task_type: write-kernel
target_path: kernels/scan/row-fp32-inclusive-sum/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: scan
scan_type: inclusive
scan_op: sum
shape:
  M: 256
  N: 4096
dtype: float32
scan_axis: 1
baseline: "torch.cumsum"
contiguous_axis: true
success_criteria:
  - correctness: "max_abs_error < 1e-5"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Per-row inclusive prefix sum of a (256, 4096) tensor. Each row has
  4096 elements, requiring a block-level scan. Consider assigning one
  block per row (blockDim.x = 256, each thread processes 16 elements
  serially, then participates in a warp + inter-warp parallel scan).
```

## Agent workflow when receiving a scan task packet

1. Read `wiki/nvidia/operator-routing/cuda-core/scan-cumulative/INDEX.md` -- follow the decision tree starting at Step 0.
2. If the library path suffices, report the recommended library call and stop (no custom kernel needed).
3. If a custom kernel is needed, read `ROUTING.md` for the skill whitelist, then implement the kernel following Step 1/Step 2 of INDEX.md.
4. Benchmark against `baseline` and apply bottleneck triage (`reasoning/bottleneck-triage.md`) if the success criteria are not met.
5. After at most 3 optimization iterations, finalize or report `status: stuck`.
