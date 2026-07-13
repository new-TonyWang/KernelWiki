---
id: skill-ascend-manual-loop-persistent-tiling
title: "Ascend Triton Manual Multi-Tile Loop and Persistent Blocks"
type: skill
vendor: ascend
tags:
- triton-ascend
- tiling-optimization
- vector-core-partition
- persistent-kernel
- loop-unrolling
- avoid-scalar-lowering
- elementwise
evidence_level: measured
applies_to:
- ascend910b
- ascend910b2c
- triton-ascend
- elementwise
source:
- path: sources/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling.md
  anchor: Ascend 910B2C Manual Multi-Tile Loop and Persistent Blocks
architectures:
- ascend910b
- ascend910b2
- ascend910b2c
languages:
- triton-ascend
- python
techniques:
- tiling-optimization
- vector-core-partition
- persistent-kernel
- loop-unrolling
- avoid-scalar-lowering
kernel_types:
- elementwise
confidence: experimental
aliases:
- manual loop
- manual-loop
- loop
- loops
- for loop
- static loop
- multi tile loop
- multi-tile
- multiple tiles
- tile loop
- tile-loop
- loop tiles
- loop factor
- LOOP_TILES
- persistent
- persistent tiling
- persistent blocks
- persistent kernel
- persistent-kernel
- persistent program
- strided loop
- grid stride
- grid-stride
- scalar overhead
- scalar time
- scalar ratio
- control overhead
- scheduler overhead
- dispatch overhead
- launch overhead
- launch blocks
- fewer blocks
- reduce blocks
- block count
- program count
- program_id
- num programs
- logical tile
- physical blocks
- NUM_BLOCKS
- MAX_ITERS
- VECTOR_CORE_COUNT
- vector cores
- 48 cores
- 48 blocks
- elementwise tiling
- streaming kernel
- high block count
- too many blocks
- overlaunch
- reduce scalar
- amortize overhead
- optimize
- optimization
- optimise
- optimisation
- performance
- speedup
- faster
- fast
- tune
- tuning
- improve performance
- optimize kernel
- optimize triton
- optimize ascend
- performance optimization
- 性能优化
- 优化
- 提速
- 加速
- 变快
- 调优
- 算子优化
- kernel优化
- triton优化
- ascend优化
- 分块
- 多块循环
- 手动循环
- 持久化
- 持久化块
- 持久化kernel
- 常驻块
- 循环展开
- 减少block
- 减少program
- 标量开销
- 控制开销
- 调度开销
- 启动开销
- 发射块数
- 逻辑tile
- 物理核数
- 向量核
- 分核
- 元素级算子
artifact_dir: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling
artifacts:
  manual: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/docs/manual_loop_persistent_tiling.md
  summary_csv: artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_manual_loop/analysis/summary.csv
related:
- skill-ascend-vector-core-partition
- routing-ascend-elementwise
- skill-triton-ascend-elementwise
---
# Ascend Triton Manual Multi-Tile Loop and Persistent Blocks

## Search keywords

optimize, optimization, performance, speedup, faster, tune, tuning, improve performance, optimize kernel, optimize triton, optimize ascend, performance optimization, manual loop, loop, for loop, static loop, multi tile loop, tile loop, multiple tiles, persistent, persistent blocks, persistent kernel, persistent tiling, strided loop, grid stride, scalar overhead, scalar time, scalar ratio, control overhead, scheduler overhead, dispatch overhead, launch overhead, launch blocks, block count, program count, fewer blocks, reduce blocks, overlaunch, high block count, too many blocks, logical tile, physical blocks, loop factor, `LOOP_TILES`, `NUM_BLOCKS`, `MAX_ITERS`, `VECTOR_CORE_COUNT`, vector cores, 48 cores, 48 blocks, elementwise tiling, streaming kernel, amortize overhead, Triton Ascend elementwise, Ascend 910B2C.

中文关键词：优化，性能优化，提速，加速，变快，调优，算子优化，kernel 优化，triton 优化，ascend 优化，手动循环，多块循环，tile 循环，多个 tile，持久化，持久化块，持久化 kernel，常驻块，strided loop，grid stride，减少 block，减少 program，block 数量，program 数量，发射块数，启动开销，调度开销，标量开销，控制开销，scalar 比例，逻辑 tile，物理核数，向量核，48 核，分核，元素级算子，高 block 数，block 太多，摊销开销。

## When to use

Use this skill for Triton-Ascend elementwise or streaming kernels when profiling shows many logical programs and a high scalar/control ratio. The goal is to keep a correctness-safe logical tile size while reducing the number of launched programs.

Typical symptoms:

- thousands of logical tiles/programs for one large tensor;
- scalar/control time tracks program count more than element count;
- increasing `BLOCK` or reducing block count helps, but a huge tile risks UB/register pressure or precision changes;
- the kernel is vector/elementwise rather than CUBE/GEMM heavy.

## Pattern A: contiguous multi-tile loop

One launched program processes `LOOP_TILES` adjacent logical tiles:

```python
@triton.jit
def kernel_loop(x_ptr, y_ptr, n_elements,
                LOOP_TILES: tl.constexpr,
                TILE_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base_tile = pid * LOOP_TILES

    for i in range(LOOP_TILES):
        tile_id = base_tile + i
        offsets = tile_id * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # compute one logical tile
        tl.store(y_ptr + offsets, x, mask=mask)
```

Host grid:

```python
num_tiles = triton.cdiv(n_elements, TILE_SIZE)
grid = (triton.cdiv(num_tiles, LOOP_TILES),)
```

This is simple and locality-friendly. Sweep `LOOP_TILES`; do not assume the smallest block count wins.

## Pattern B: persistent strided blocks

Launch a fixed number of programs, often near the vector-core count, and assign each program `pid + i * NUM_BLOCKS` logical tiles:

```python
@triton.jit
def kernel_persistent(x_ptr, y_ptr, n_elements,
                      NUM_BLOCKS: tl.constexpr,
                      MAX_ITERS: tl.constexpr,
                      TILE_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, TILE_SIZE)

    for i in range(MAX_ITERS):
        tile_id = pid + i * NUM_BLOCKS
        active = tile_id < num_tiles
        offsets = tile_id * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = (offsets < n_elements) & active
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # compute one logical tile
        tl.store(y_ptr + offsets, x, mask=mask)
```

Host grid:

```python
num_tiles = triton.cdiv(n_elements, TILE_SIZE)
num_blocks = min(num_tiles, VECTOR_CORE_COUNT)
max_iters = triton.cdiv(num_tiles, num_blocks)
grid = (num_blocks,)
```

For Ascend 910B2C vector-heavy kernels, the captured case used `VECTOR_CORE_COUNT = 48` as a first persistent-block target.

## Correctness checklist

Every logical tile must be processed exactly once:

- contiguous loop: `tile_id = pid * LOOP_TILES + i`;
- persistent loop: `tile_id = pid + i * NUM_BLOCKS`;
- guard tails with `active = tile_id < num_tiles` and combine with the element mask;
- keep dtype materialization and numerical boundaries unchanged; this optimization changes scheduling, not math semantics.

For `Inf`/`NaN`-prone elementwise kernels, use exact equality or explicit finite/non-finite masks in addition to max-absolute-difference summaries, because `Inf - Inf` can become `NaN` even when tensors are equal.

## Measurement checklist

For every candidate loop factor or persistent variant, record:

- launch block count;
- event/device time;
- profiler duration;
- scalar time and scalar ratio;
- vector and MTE ratios when available;
- exact/tolerance equality versus the original logical-tile implementation.

The optimization is successful only if total device time improves, not merely scalar time.

## Case study: large bf16 elementwise workload on Ascend 910B2C

Workload characteristics:

- dtype: bf16;
- shape: `15x255x1x1x256x8`;
- elements: `7,833,600`;
- logical `TILE_SIZE=4096`;
- logical tile count: `1913`;
- persistent target: `48` launch blocks.

Measured variants from the localized artifact:

| Variant | Launch blocks | Event us | Prof duration us | Scalar us | Scalar ratio | Exact equal |
|---|---:|---:|---:|---:|---:|---|
| `loop1` | 1913 | 195.120 | 166.186 | 59.787 | 0.364 | yes |
| `loop2` | 957 | 130.570 | 88.764 | 13.870 | 0.161 | yes |
| `loop4` | 479 | 125.490 | 83.384 | 10.622 | 0.132 | yes |
| `loop8` | 240 | 122.100 | 81.404 | 9.326 | 0.119 | yes |
| `loop16` | 120 | 133.230 | 92.863 | 10.147 | 0.110 | yes |
| `loop32` | 60 | 166.690 | 121.085 | 12.211 | 0.101 | yes |
| `loop40` | 48 | 120.120 | 77.403 | 6.710 | 0.090 | yes |
| `persistent_48` | 48 | 119.970 | 78.603 | 7.068 | 0.094 | yes |

Findings:

1. Manual looping reduced scalar time from `59.787 us` to roughly `6-9 us`.
2. Event time improved from `195.120 us` to about `120 us`.
3. `loop32` regressed despite fewer blocks, so loop factor must be selected by measurement.
4. `persistent_48` matched the 48-vector-core target and remained exact-equal to `loop1`.

## Artifacts

- Source record: `sources/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling.md`
- Manual note: `artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/docs/manual_loop_persistent_tiling.md`
- Sweep script: `artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_manual_loop/scripts/manual_loop_case024.py`
- Summary CSV: `artifacts/experience/hw-probes/ascend910b2c-manual-loop-persistent-tiling/profile/round11_manual_loop/analysis/summary.csv`

## Takeaway

Manual multi-tile loops and persistent blocks are a measured Ascend/Triton-Ascend way to amortize scalar/control overhead in high-block-count streaming kernels. Preserve the per-tile math and precision semantics, then sweep loop count or persistent block count around the hardware vector parallelism instead of blindly minimizing the number of launched programs.
