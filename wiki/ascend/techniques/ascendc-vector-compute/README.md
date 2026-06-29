---
id: skill-ascend-vector-compute
title: "AscendC Vector Compute Reference: Init and Process"
type: skill
vendor: ascend
tags:
- ascendc
- vector-unit
- ub
evidence_level: spec
applies_to:
- ascend910b
source:
- path: local
  anchor: AscendOpGenAgent/skills
architectures:
- ascend910b
languages:
- ascendc
hardware_features:
- vector-unit
- ub
---
## Vector Compute Reference: Init and Process

For pure Vector operators or AscendC implementations with Vector compute stages.

### vec_num and Block Composition

| DSL vec_num | KERNEL_TYPE | Block Composition |
|:---|:---|:---|
| 1 | KERNEL_TYPE_MIX_AIC_1_1 | 1 AIC + 1 AIV |
| 2 | KERNEL_TYPE_MIX_AIC_1_2 | 1 AIC + 2 AIV |

### TPosition Mapping (Vector side)

| TPosition | Physical Storage | Purpose |
|:---|:---|:---|
| VECIN / VECOUT | Unified Buffer | Vector compute input/output |
| VECCALC | Unified Buffer | Temporary variables |

### Buffer/Queue Selection Rules

- UB buffer in `T.serial` loop + GM copy → `TQue + VECIN/VECOUT`
- UB buffer for intermediate computation only → `TBuf + VECCALC`
- UB buffer loaded once, reused in multiple loops → `TBuf + PipeBarrier<PIPE_MTE2>()`

### Three-Stage Data Flow

1. **CopyIn**: `AllocTensor` → `DataCopyPad` (GM→UB) → `EnQue`
2. **Compute**: `DeQue` → compute → `EnQue` (or `FreeTensor`)
3. **CopyOut**: `DeQue` → `DataCopyPad` (UB→GM) → `FreeTensor`

### TQue depth=0 Constraint

VECIN/VECOUT queues (depth=0) must use reference form:

```cpp
queue.AllocTensor<T>(localVar);
queue.DeQue<T>(localVar);
```
