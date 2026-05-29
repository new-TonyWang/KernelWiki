---
title: Indexing Pattern -- Task Packet Template
pattern_class: cuda-core
op: indexing
status: draft
source:
- path: reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This
    file is the task packet.'
id: routing-indexing-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: indexing
---
# Indexing -- Task Packet Template

This document defines the operator-specific task packet fields for an indexing kernel-writing task. It refines the generic task packet contract in `reasoning/task-packet.md` with indexing-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from reasoning/task-packet.md) ---
task_id: "2026-04-XX-indexing-<variant>"       # date-prefixed, kebab-case
task_type: write-kernel                        # or benchmark-kernel
target_path: kernels/indexing/<variant>/        # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- Indexing-specific fields ---
op: indexing

index_type: gather
  # The sub-operation. One of:
  #   gather          -- indirect load:  dst[i] = src[idx[i]]
  #   scatter         -- indirect store: dst[idx[i]] = val[i]
  #   scatter_add     -- scatter with additive conflict resolution
  #   index_select    -- select full slices along a dimension
  #   topk            -- find the k largest/smallest elements
  #   one_hot         -- convert class indices to one-hot vectors
  #   embedding       -- lookup rows from an embedding table

shape:
  # Specify the source and index tensor shapes.
  # For gather/scatter:
  #   src: { N: 1048576 }       -- 1-D source array
  #   idx: { N: 65536 }         -- 1-D index array (number of gathers)
  # For 2-D gather along an axis:
  #   src: { M: 1024, N: 256 }
  #   idx: { M: 1024, K: 16 }  -- K indices per row
  # For index_select:
  #   src: { M: 10000, N: 256 }
  #   idx: { K: 128 }          -- selecting 128 rows
  # For topk:
  #   src: { M: 128, N: 4096 } -- per-row topk
  src:
    N: 1048576
  idx:
    N: 65536

dtype: float32
  # Supported: float16, bfloat16, float32, float64, int32, int64

baseline: "torch.gather"
  # The library baseline to benchmark against. One of:
  #   torch.gather, torch.scatter_, torch.scatter_add_,
  #   torch.index_select, torch.topk,
  #   thrust::gather, thrust::scatter,
  #   cub::DeviceRadixSort (for topk via full sort)
  # The custom kernel's latency must approach or beat this baseline
  # to justify its existence (see INDEX.md Step 0).
```

## Optional fields

```yaml
k: 10
  # For topk only: the number of top elements to select.
  # Must satisfy 1 <= k <= shape.src.N (or the reduction dimension).

gather_dim: 1
  # For multi-dimensional gather/scatter: which dimension to index into.
  # Default: -1 (last / innermost dimension).

conflict_resolution: "add"
  # For scatter only. How to resolve index collisions:
  #   none    -- undefined behavior (last write wins); default for scatter
  #   add     -- atomicAdd (scatter_add)
  #   max     -- atomicMax (scatter_max)
  #   min     -- atomicMin (scatter_min)

index_pattern: "random"
  # Describes the expected distribution of indices, which affects
  # optimization strategy:
  #   random          -- uniformly random indices (worst case for caching)
  #   sorted          -- indices are pre-sorted (locality-friendly)
  #   clustered       -- indices have spatial locality (block-friendly)
  #   contiguous_range -- indices are a contiguous sub-range (trivial case)

num_classes: 1000
  # For one_hot only: the number of classes (output width).

embedding_dim: 256
  # For embedding only: the embedding vector dimension.

success_criteria:
  - correctness: "max_abs_error < 1e-5 vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
  # The kernel must be within 10% of the library baseline.
  # For indexing ops, this bar is harder to meet because the library
  # implementations are well-optimized for the common case.

notes: |
  Free-form guidance for the kernel-writing agent.
  Example: "This gather is part of an attention score computation;
  the gathered values will immediately be multiplied by a query
  vector, so fusing the gather + multiply is the primary goal."
```

## Example: random gather of 64K elements from a 1M-element array

```yaml
task_id: 2026-04-20-indexing-gather-random-fp32
task_type: write-kernel
target_path: kernels/indexing/gather-random-fp32/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: indexing
index_type: gather
shape:
  src:
    N: 1048576
  idx:
    N: 65536
dtype: float32
index_pattern: random
baseline: "torch.gather"
success_criteria:
  - correctness: "exact match vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
references:
  - wiki/nvidia/operator-routing/cuda-core/indexing/INDEX.md
  - wiki/nvidia/operator-routing/cuda-core/indexing/ROUTING.md
  - reasoning/bottleneck-triage.md
```

## Example: scatter-add with conflicts

```yaml
task_id: 2026-04-21-indexing-scatter-add-fp32
task_type: write-kernel
target_path: kernels/indexing/scatter-add-fp32/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: indexing
index_type: scatter_add
shape:
  src:
    N: 65536
  idx:
    N: 65536
  dst:
    N: 10000
dtype: float32
conflict_resolution: add
index_pattern: random
baseline: "torch.scatter_add_"
success_criteria:
  - correctness: "max_abs_error < 1e-5 vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Scatter-add with approximately 6.5 average writes per destination
  index. Index collisions are expected and must be handled with
  atomicAdd.
```

## Example: per-row topk

```yaml
task_id: 2026-04-22-indexing-topk-row-fp32
task_type: write-kernel
target_path: kernels/indexing/topk-row-fp32/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: indexing
index_type: topk
shape:
  src:
    M: 128
    N: 4096
k: 10
gather_dim: 1
dtype: float32
baseline: "torch.topk"
success_criteria:
  - correctness: "exact match of indices (order-independent)"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Per-row top-10 of a (128, 4096) tensor. Each row has 4096
  elements; k=10 is small relative to N, making a partial
  radix-select approach potentially faster than a full sort.
```

## Example: embedding lookup

```yaml
task_id: 2026-04-23-indexing-embedding-fp32
task_type: write-kernel
target_path: kernels/indexing/embedding-fp32/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"
op: indexing
index_type: embedding
shape:
  src:
    M: 50000
    N: 256
  idx:
    K: 512
embedding_dim: 256
dtype: float32
baseline: "torch.nn.functional.embedding"
success_criteria:
  - correctness: "exact match vs baseline"
  - performance: "median_latency <= 1.1 * baseline_latency"
notes: |
  Embedding table with 50K entries of dimension 256. Lookup 512
  indices. Each lookup selects a contiguous row of 256 floats,
  so vectorized loads (float4) should be applied per row.
```

## Agent workflow when receiving an indexing task packet

1. Read `wiki/nvidia/operator-routing/cuda-core/indexing/INDEX.md` -- follow the decision tree starting at Step 0.
2. If the library path suffices, report the recommended library call and stop (no custom kernel needed).
3. If a custom kernel is needed, read `ROUTING.md` for the skill whitelist, then implement the kernel following Step 1 of INDEX.md, selecting the strategy for the specific `index_type`.
4. Benchmark against `baseline` and apply bottleneck triage (`reasoning/bottleneck-triage.md`) if the success criteria are not met.
5. After at most 3 optimization iterations, finalize or report `status: stuck`.
