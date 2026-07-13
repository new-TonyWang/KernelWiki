---
id: skill-ascend-block-level-design
title: "Ascend Block-Level Kernel Design Methodology"
type: skill
vendor: ascend
tags:
- ai-core
- cube-unit
- vector-unit
- tilelang
evidence_level: spec
applies_to:
- ascend910b
source:
- path: local
  anchor: AscendOpGenAgent/skills
architectures:
- ascend910b
languages:
- tilelang
- ascendc
techniques:
- tile-scheduling
- pipeline-stages
- double-buffering
---
# Block Level Design

## Hardware Architecture

AI Core consists of three parts: `Cube` (matrix compute), `Vector` (elementwise, reduction, gather, scatter), and `Scalar` (instruction dispatch, loop control). Cube and Vector can run in parallel, enabling C/V pipeline designs.

Storage hierarchy (GPU analogy):
- `global memory` ↔ `global memory`
- `shared memory` ↔ Cube-side `L1 buffer` / Vector-side `Unified Buffer`
- `register memory` ↔ `L0A/L0B/L0C buffer`

Cube and Vector cores communicate through global memory. Workspace tensors also reside in global memory.

## Block Level Design Overview

Block-level design answers four questions:

1. Which output region does each block produce?
2. What data must the block iterate over?
3. How do C and V cooperate and interleave?
4. When different shapes need different block organizations, should multiple `T.prim_func` templates be designed?

## Task Partitioning

1. Prefer no write-conflict, no extra synchronization splits between blocks
2. If this loses parallelism, introduce controlled cross-block reduction
3. When layout is unfavorable, do axis merging, splitting, or reordering

### Example: Matmul

Split along M/N across blocks, K loops within block. Each block owns exclusive output region.

### Example: Flash Attention

Each block handles one `(batch, head, q_block)` output region, iterating all KV chunks within the block.

### Example: Pooling / Scan

Reorder layouts first: NCHW → NHWC for pooling, move scan axis to axis 0 for cumsum/cumprod.

## Template Design: Multiple prim_func by Shape

```python
@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def op(shape0, shape1, ..., dtype="float32"):
    @T.prim_func
    def fast_path(...):
        with T.Kernel(block_num_fast, is_npu=True) as (cid, vid):
            ...

    @T.prim_func
    def fallback_path(...):
        with T.Kernel(block_num_fallback, is_npu=True) as (cid, vid):
            ...

    if some_shape_condition:
        return fast_path
    return fallback_path
```

## Pipeline Design

Goal: arrange block-internal stages so Cube and Vector stay busy in steady-state.

### Matmul Pipeline
- C: matmul main compute
- V: epilogue (activation, cast, writeback)

### Flash Attention Pipeline (interleaved)

```text
t=0:  C: C1(0)          V: V1(0)
t=1:  C: C1(1)          V: V1(1)
t=2:  C: C1(2) + C2(0)  V: V1(2) + V2(0)
t=3:  C: C1(3) + C2(1)  V: V1(3) + V2(1)
tail: C: C2(...)         V: V2(...)
```

Preload/prelaunch pattern with ring buffer and cross flags for stage handoff.
