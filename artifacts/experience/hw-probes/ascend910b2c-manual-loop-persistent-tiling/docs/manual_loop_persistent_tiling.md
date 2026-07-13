# Manual Multi-Tile Loop / Persistent Blocks Optimization Pattern

## Summary

When any kernel launches far more programs/blocks than the physical core count, scalar/control overhead can dominate. A common fix is to decouple **logical tiles** from **launch blocks**:

- Keep a convenient logical `TILE_SIZE` for vector work and UB/register pressure.
- Launch fewer programs.
- Let each program process multiple logical tiles in a static loop or a persistent strided loop.

This reduces repeated per-program overhead such as `program_id`, offset generation, mask/predicate setup, masked load/store bookkeeping, and scheduler overhead while preserving the same logical work.

## When to consider it

Use this pattern for any kernel type when profiling shows:

1. **High scalar/control ratio** in a high-block-count kernel.
2. **Large logical block count**, e.g. thousands of Triton programs for one tensor.
3. **Per-tile work is not enough** to amortize per-program overhead.
4. Increasing `BLOCK` or reducing block count improves performance, but simply making `BLOCK` huge risks UB pressure, register pressure, lower occupancy, or correctness/parity issues.

Do not use it blindly for kernels where every program already has enough work, or for small tensors where launching too many persistent blocks would create idle work.

## Terminology

- **Logical tile**: The unit of data processed by one iteration, e.g. `TILE_SIZE=4096` elements.
- **Launch block/program**: One Triton program instance, indexed by `tl.program_id(0)`.
- **Loop factor**: Number of logical tiles processed by one program in a contiguous manual loop.
- **Persistent blocks**: A fixed small number of programs, usually near the hardware core/vector-core count, each processing multiple logical tiles in a strided loop.

## Pattern A: contiguous multi-tile loop

Each program handles `LOOP_TILES` adjacent logical tiles.

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
        # ... compute one logical tile ...
        tl.store(y_ptr + offsets, x, mask=mask)
```

Host-side grid:

```python
num_tiles = triton.cdiv(n_elements, TILE_SIZE)
grid = (triton.cdiv(num_tiles, LOOP_TILES),)
kernel_loop[grid](x, y, n_elements,
                  LOOP_TILES=loop_tiles,
                  TILE_SIZE=tile_size)
```

### Pros

- Simple addressing.
- Adjacent tiles may have better locality.
- Easy to tune `LOOP_TILES`.

### Cons

- Tail program may execute inactive loop iterations unless guarded by masks.
- Too large `LOOP_TILES` can reduce parallelism and regress performance.

## Pattern B: persistent strided blocks

Launch a fixed number of programs, often `min(num_tiles, vector_core_count)`, and let each program process tiles `pid + k * NUM_BLOCKS`.

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
        # ... compute one logical tile ...
        tl.store(y_ptr + offsets, x, mask=mask)
```

Host-side grid:

```python
num_tiles = triton.cdiv(n_elements, TILE_SIZE)
num_blocks = min(num_tiles, vector_core_count)
max_iters = triton.cdiv(num_tiles, num_blocks)

grid = (num_blocks,)
kernel_persistent[grid](x, y, n_elements,
                        NUM_BLOCKS=num_blocks,
                        MAX_ITERS=max_iters,
                        TILE_SIZE=tile_size)
```

### Pros

- Caps launch block count at hardware parallelism.
- Good when scalar/scheduler overhead dominates.
- Naturally balances work when many logical tiles exist.

### Cons

- Strided tile order may reduce locality compared with contiguous loops.
- `MAX_ITERS` must be constexpr/static in Triton-style loops.
- Too few programs can underutilize hardware if each iteration has long-latency memory or special-function work and insufficient overlap.

## Choosing `NUM_BLOCKS`

For Ascend 910B2C, the project-local KernelWiki hardware page records:

- 24 AI Core
- 2 VEC per AI Core
- 48 VEC total

For kernels whose logical tiles primarily occupy vector-side work, a useful first choice is:

```python
VECTOR_CORE_COUNT = 48
num_blocks = min(num_tiles, VECTOR_CORE_COUNT)
```

For other devices, use the actual vector-core / AI-core count for that chip. If unknown, measure a small sweep around likely values.

## Choosing `TILE_SIZE` and loop factor

`TILE_SIZE` and `LOOP_TILES` are different knobs:

- `TILE_SIZE` controls vector width per iteration, UB/register pressure, and per-iteration memory behavior.
- `LOOP_TILES` controls how many iterations one program performs, reducing launch block count.

Guidelines:

1. Pick a `TILE_SIZE` that is already correctness-safe and does not exceed UB/register constraints.
2. Sweep `LOOP_TILES` such that launch blocks approach, but do not necessarily go below, hardware vector parallelism.
3. Test `LOOP_TILES` around:
   - 1, 2, 4, 8, 16
   - `ceil(num_tiles / vector_core_count)`
4. Do not assume fewer blocks is always faster. Excessive serial work per program may reduce parallelism or increase pressure.

## Correctness requirements

Every logical tile must be processed exactly once.

For contiguous loop:

```text
tile_id = pid * LOOP_TILES + i
```

For persistent strided loop:

```text
tile_id = pid + i * NUM_BLOCKS
```

Tail handling must guard inactive tiles:

```python
active = tile_id < num_tiles
mask = (offsets < n_elements) & active
```

For kernels that may produce `Inf` or `NaN`, exact equality checks can be stronger than max-absolute-difference summaries, because `Inf - Inf` may become `NaN` even when tensors are equal.

## Measurement checklist

For every candidate loop factor or persistent variant, collect:

- Launch block count.
- Event/device time.
- Profiler duration.
- Scalar time and scalar ratio.
- Vector ratio/time.
- MTE2/MTE3 ratios if memory movement is relevant.
- Exact or tolerance-based equality versus the original logical-tile implementation.

The important question is not only whether scalar time decreases, but whether total device time decreases.

## Case study: large bf16 high-block-count workload on Ascend 910B2C

Workload:

- dtype: bf16
- shape: `15x255x1x1x256x8`
- elements: `7,833,600`
- logical `TILE_SIZE=4096`
- logical tile count: `1913`
- vector-core limit: `48`

Measured variants from `profile/round11_manual_loop/analysis/summary.csv`:

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

1. Manual looping reduced scalar time from `59.787 us` to about `6-9 us`.
2. Event time improved from `195.120 us` to about `120 us`.
3. `loop32` regressed despite fewer blocks, proving loop factor must be measured.
4. `persistent_48` implemented the same logical work with launch blocks equal to the 48 VEC count and was exact-equal to `loop1`.

## Common pitfalls

- **Confusing bigger tile with more tiles per program.** Increasing `TILE_SIZE` changes vector/memory shape; looping multiple tiles preserves logical tile shape while reducing program count.
- **Ignoring tails.** Persistent loops need `active` masks to avoid out-of-range work.
- **Over-serializing.** Too few programs or too many loop iterations per program can reduce parallelism.
- **Using this as a correctness shortcut.** This only changes scheduling/tiling; all dtype materialization and numerical boundaries must be preserved.
- **Promoting from one-case evidence.** A production route must pass full correctness and benchmark coverage.

## Recommended implementation flow

1. Profile the original kernel and confirm scalar/control overhead with high block count.
2. Write an isolated ablation kernel using the same per-tile computation.
3. Sweep contiguous `LOOP_TILES`.
4. Test a persistent `NUM_BLOCKS = min(num_tiles, vector_core_count)` variant.
5. Verify exact/tolerance equality against the original per-tile kernel.
6. If useful, integrate as a shape/dtype-specific production route.
7. Re-run full static checks, correctness, perturbation if required, and benchmark gates.

## Takeaway

Manual multi-tile loops and persistent blocks are a general way to reduce scalar/control overhead in high-block-count kernels of any type. They are especially useful when profiling shows scalar time tracking block count more strongly than element count. The best configuration is hardware- and workload-dependent, so it must be selected by measurement.
