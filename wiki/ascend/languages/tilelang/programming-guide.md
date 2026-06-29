---
id: skill-tilelang-ascend-programming-guide
title: "TileLang Ascend Programming Guide"
type: skill
vendor: ascend
tags:
- tilelang
- ai-core
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
- tilelang
hardware_features:
- ai-core
- cube-unit
- vector-unit
- l1-buffer
- l0a
- l0b
- l0c
- ub
---
# TileLang-Ascend Programming Guide

## 1. Introduction

TileLang is a tile-level DSL for kernel programming. On Ascend, a practical and stable programming style is to organize computation around:

- L1 as Cube-side staging memory
- UB as Vector-side working memory
- L0A / L0B / L0C as matrix compute buffers

This guide focuses on that programming model and covers:

- kernel definition and launch
- L1 / UB / L0 memory allocation
- data movement with `T.copy`
- matrix compute with `T.mma`
- vector and tile compute with `T.tile.*`
- scope and synchronization control

### 1.1 Programming Guidelines

- Prefer `T.tile.*` APIs for compute whenever possible, and avoid scalar or element-by-element operations in hot paths.
- Use `T.tile.broadcast` sparingly because it can consume large UB temporary space, and prefer row-wise or column-wise tile compute patterns when UB is constrained.
- On the Vector side, in practice, you should copy inputs into UB and then cast them to `float32` at the beginning of `T.Scope("V")`, because this ensures better numerical stability and consistent behavior across the subsequent vector compute path.

## 2. Basic Structure

### 2.1 JIT Kernel Definition

Use `@tilelang.jit(...)` to define a kernel generator and `@T.prim_func` to define the device kernel body.

The basic structure is:

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[...])
def kernel(...):
    @T.prim_func
    def main(...):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            ...
    return main
```

### 2.2 Data Types

TileLang Ascend kernels commonly use dtype strings such as:

- `float16`
- `float32`
- `bfloat16`
- `int8`
- `int16`
- `int32`
- `uint8`
- `uint16`
- `uint32`

### 2.3 Kernel Launch

Use `T.Kernel(grid, is_npu=True)` to create an NPU kernel region:

```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    ...
```

- `cid`: block or tile id
- `vid`: Vector-side split id when a kernel has Cube/Vector cooperation

### 2.4 Loops and Control Flow

Use `T.serial` for explicit serial iteration:

```python
for k in T.serial(loop_k):
    if k == 0:
        ...
    else:
        ...
```

## 3. On-Chip Memory

Ascend kernels in this guide only use L1, UB, and L0 storage.

### 3.1 `T.alloc_L1`

Allocates an L1 buffer.

```python
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_K, block_N), "float16")
```

### 3.2 `T.alloc_ub`

Allocates a UB buffer.

```python
x_ub = T.alloc_ub((block_M, block_N), "float16")
tmp_ub = T.alloc_ub((tmp_size,), "uint8")
```

### 3.3 `T.alloc_L0A` / `T.alloc_L0B` / `T.alloc_L0C`

```python
A_L0 = T.alloc_L0A((block_M, block_K), "float16")
B_L0 = T.alloc_L0B((block_K, block_N), "float16")
C_L0 = T.alloc_L0C((block_M, block_N), "float32")
```

## 4. Data Movement

### 4.1 `T.copy`

`T.copy(src, dst)` is the primary movement primitive. Common patterns:

- GM -> L1, L1 -> L0A/L0B, L0C -> GM, GM -> UB, UB -> GM

### 4.2 Layout Annotation

```python
from tilelang.intrinsics import make_zn_layout
T.annotate_layout({
    A_L1: make_zn_layout(A_L1),
    B_L1: make_zn_layout(B_L1),
})
```

## 5. Matrix Compute with `T.mma`

```python
T.mma(A_L0, B_L0, C_L0, init=True)
```

- `A_L0` in L0A, `B_L0` in L0B, `C_L0` in L0C
- `init=True` initializes the accumulation tile; `init=False` accumulates

## 6. Tile Compute with `T.tile.*`

### 6.1 Math and Logical Ops

| API | Semantics |
| --- | --- |
| `T.tile.add(dst, src0, src1)` | elementwise add |
| `T.tile.sub(dst, src0, src1)` | elementwise subtract |
| `T.tile.mul(dst, src0, src1)` | elementwise multiply |
| `T.tile.div(dst, src0, src1)` | elementwise divide |
| `T.tile.max/min(dst, src0, src1)` | elementwise max/min |
| `T.tile.exp/ln/abs/sqrt/rsqrt(dst, src)` | unary ops |
| `T.tile.relu/leaky_relu(dst, src, ...)` | activation |
| `T.tile.sin/cos(dst, src, tmp)` | trig ops |
| `T.tile.bitwise_and/or/not/xor(...)` | bitwise ops |

### 6.2 Compare and Select

- `T.tile.compare(dst, src0, src1, mode)` — modes: EQ, NE, GT, GE, LT, LE
- `T.tile.select(dst, selMask, src0, src1, selMode)`
- `T.tile.gather_mask(dst, src, pattern)` — fixed patterns: P0101, P1010, etc.

### 6.3 Cast and Data Transform

- `T.tile.cast(dst, src, mode, count)` — modes: CAST_NONE, CAST_RINT, CAST_FLOOR, etc.
- `T.tile.transpose(dst, src)` — 16x16 2D matrix-tile transpose

### 6.4 Fill and Index Helpers

- `T.tile.fill(buffer, value)`
- `T.tile.createvecindex(dst, first_value)`
- `T.tile.arith_progression(buffer, first_value, diff_value, count)`

### 6.5 Sort and Gather

- `T.tile.sort(dst, src, indices, tmp_buffer, repeat_time)`
- `T.tile.merge_sort(dst, src, block_size, block_num, is_copy)`
- `T.tile.topk(dst, src, tmp_buffer, block_size)`
- `T.tile.gather(dst, src, src_offset, src_base_addr)`

### 6.6 Broadcast

- `T.tile.broadcast(dst, src, tmp)` — requires additional UB, use sparingly

### 6.7 Reduce

- `T.reduce_sum/max/min(buffer, out, tmp, dim)` — tmp must be uint8, size 2x buffer

## 7. Scope and Synchronization

### 7.1 Scope Control

- `T.Scope("C")` for Cube-side logic (GM/L1/L0 movement, `T.mma`)
- `T.Scope("V")` for Vector-side logic (UB tiles, `T.tile.*`)

### 7.2 Pipe Flags

- `T.set_flag(src, dst, id)` / `T.wait_flag(src, dst, id)`
- Common pipe names: mte1, mte2, mte3, m, v, fix

### 7.3 Cross-Scope Synchronization

- `T.set_cross_flag(pipe, id)` / `T.wait_cross_flag(id)`

### 7.4 Pipe Barrier and Auto Sync

- `T.pipe_barrier(pipe)` for full pipe barrier
- `tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True` for auto-inferred sync
