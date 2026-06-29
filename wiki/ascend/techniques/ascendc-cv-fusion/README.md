---
id: skill-ascend-cv-fusion
title: "AscendC Cube/Vector Fusion Reference"
type: skill
vendor: ascend
tags:
- ascendc
- cube-unit
- vector-unit
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
- vector-unit
- ub
- l1-buffer
techniques:
- kernel-fusion
- pipeline-stages
---
## C/V Fusion Reference

For operators with both Cube and Vector stages running cooperatively (AIC/AIV).

### Kernel Entry

```cpp
extern "C" __global__ __aicore__ void kernel_custom(GM_ADDR ...inputs..., GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    KernelClass kernel;
    kernel.Init(..., workspace, tiling, &pipe);
    kernel.Process();
}
```

| DSL vec_num | KERNEL_TYPE | Block Composition | GetSubBlockNum() |
|:---|:---|:---|:---|
| 1 | KERNEL_TYPE_MIX_AIC_1_1 | 1 AIC + 1 AIV | 2 |
| 2 | KERNEL_TYPE_MIX_AIC_1_2 | 1 AIC + 2 AIV | 3 |

### Init(): GM, Workspace, Submodule Setup

- Tiling + GM binding
- Workspace base address and per-core offset calculation
- Ring buffer / WorkspaceQueue initialization
- Cube submodule init under `ASCEND_IS_AIC` branch
- Vector submodule init under `ASCEND_IS_AIV` branch

### Process(): AIC/AIV Branch Dispatch

```cpp
__aicore__ inline void KernelClass::Process()
{
    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);
        if ASCEND_IS_AIC { /* Cube side */ }
        if ASCEND_IS_AIV { /* Vector side */ }
    }
}
```

- **AIC branch**: fetch input, acquire workspace slot, call Cube submodule, release slot
- **AIV branch**: acquire consumer slot, compute sub-block offset via `GetSubBlockIdx()`, call Vector submodule, write back to GM

### When to Split Submodules

Split when: multiple clear Scope roles, both Cube and Vector stages exist, or compute logic needs reuse. One TileLang Scope ↔ one AscendC submodule.
