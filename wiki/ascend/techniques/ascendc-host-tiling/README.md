---
id: skill-ascend-host-tiling
title: "AscendC Host-Side Tiling and Pybind11 Binding"
type: skill
vendor: ascend
tags:
- ascendc
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
---
## Host-Side Preparation

### Tiling Parameter Consistency

All kernel components must use consistent tiling parameters:

```cpp
constexpr uint32_t baseM = 64;
constexpr uint32_t baseN = 64;
constexpr uint32_t baseK = 64;
constexpr uint32_t subBlockM = baseM;  // or baseM / vec_num
```

### Tiling Struct: Pre-compute on Host

Avoid computing derived values like `nTiles = N / baseN` in the kernel. Pre-compute in host and write to tiling struct:

```cpp
struct Tiling {
    int32_t M, N, H_K;
    int32_t baseM, baseN, baseK, K_L1;
    int32_t nTiles;       // = N / baseN
    int32_t nTilesPerH;   // = H_K / baseN
};
```

### Pybind11 Binding

The binding function:
- Receives only explicit input tensors (not outputs or workspace)
- Checks input shape and dtype
- Derives runtime params from input shape
- Allocates output tensors and workspace
- Constructs tiling tensor
- Calls `extern "C"` kernel launch function
- Returns output tensors

Module naming: use `_<op_name>_ext` to avoid name collision with task directory.

### Workspace Allocation

When DSL declares workspace or `@tilelang.jit(workspace_idx=...)` is specified, pybind11 must allocate workspace. Total bytes must match DSL block organization, accumulation dtype, and parallelism.
