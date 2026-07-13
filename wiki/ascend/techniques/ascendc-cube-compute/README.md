---
id: skill-ascend-cube-compute
title: "AscendC Cube Compute Reference: Init and Process"
type: skill
vendor: ascend
tags:
- ascendc
- cube-unit
- l0a
- l0b
- l0c
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
- cube-unit
- l1-buffer
- l0a
- l0b
- l0c
---
## Cube Compute Reference: Init and Process

For pure Cube operators or AscendC implementations with independent Cube compute stages.

### Kernel Entry

```cpp
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    KernelClass kernel;
    kernel.Init(a, b, c, tiling, &pipe);
    kernel.Process();
}
```

### Init(): Tiling, GM Binding, Queue Setup

- Read tiling fields via `CopyTiling(&tiling_, tilingGM)`
- Bind GM tensors via `SetGlobalBuffer(...)`
- Derive runtime params: `baseM`, `baseN`, `baseK`, etc.

#### TPosition Mapping (Cube side)

| TPosition | Physical Storage | Purpose |
|:---|:---|:---|
| A1 / B1 | L1 Buffer | Large matrix tiles |
| A2 / B2 | L0A / L0B | Small MMA input tiles |
| CO1 | L0C | Matrix compute results |

### Process(): Workload Loop and Cube Pipeline

- **CopyA / CopyB**: GM → L1
- **SplitA / SplitB**: L1 → L0, extract current baseK sub-tile
- **Compute**: `Mmad` on current A2/B2
- **CopyOut**: `Fixpipe` writes CO1 to GM or workspace

Key points:
- DeQue once per K_L1 iteration, pass offset pointers to inner loop
- FreeTensor once after all inner iterations complete
- `Fixpipe` dstStride must be the full row width
