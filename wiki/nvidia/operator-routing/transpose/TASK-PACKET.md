---
title: Transpose Pattern -- Task Packet Template
pattern_class: cuda-core
op: transpose
status: draft
source:
- path: reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This
    file is the task packet.'
id: routing-transpose-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: transpose
---
# Transpose -- Task Packet Template

This document defines the operator-specific task packet fields for a transpose kernel-writing task. It refines the generic task packet contract in `reasoning/task-packet.md` with transpose-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from reasoning/task-packet.md) ---
task_id: "2026-04-XX-transpose-<variant>"      # date-prefixed, kebab-case
task_type: write-kernel                         # or benchmark-kernel
target_path: kernels/transpose/<variant>/       # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- Transpose-specific fields ---
op: transpose

shape:
  # Specify the input tensor shape. Use one of:
  # - 2-D matrix: { M: 4096, N: 4096 }
  # - N-D tensor: { dims: [B, H, M, N], sizes: [4, 8, 2048, 2048] }
  M: 4096
  N: 4096

dtype: float32
  # Supported: float16, bfloat16, float32, int32. fp64 valid but unusual.

transpose_kind: matrix_2d
  # One of:
  #   matrix_2d            -- 2-D dim0↔dim1 swap (the canonical case)
  #   permute_nd           -- general N-D axis permute
  #   aos_to_soa           -- struct-of-fields → array-per-field (layout)
  #   soa_to_aos           -- reverse (rare, usually only for host export)

permutation: [1, 0]
  # For matrix_2d, this is always [1, 0].
  # For permute_nd, list the target axis order, e.g. [0, 2, 1, 3].
  # For aos_to_soa / soa_to_aos, omit (implied by struct definition).

# AoS/SoA specifics (only when transpose_kind is aos_to_soa / soa_to_aos)
struct_fields:
  - { name: x,  dtype: float32 }
  - { name: y,  dtype: float32 }
  - { name: z,  dtype: float32 }
  - { name: vx, dtype: float32 }
  - { name: vy, dtype: float32 }
  - { name: vz, dtype: float32 }
# Struct size in bytes is inferred from the field list.

amortization_context:
  # Estimated number of downstream kernels that will consume the
  # transposed layout. Drives the decision between "materialize" and
  # "strided view" in INDEX Step 1 Q5d.
  downstream_kernel_count: 8
  # Rule of thumb (from aos-vs-soa probe): materialize when
  # downstream_kernel_count >= 4 for 24-B struct; matrix transpose
  # usually materializes at >= 2 downstream kernels.
```

## Optional fields

```yaml
# Skill whitelist override: normally derived from ROUTING.md; override
# only when the task explicitly excludes a skill (e.g. "no smem for
# this variant because we want to benchmark the naive baseline").
skill_whitelist:
  - shared-memory-cache
  - bank-conflict
  - layout-transform
  - vectorized-access
  - coalescing

# Tile-size override: default is [32][33] for fp32. Supply only for
# small-matrix regimes where the default collapses occupancy.
tile_size:
  rows: 16
  cols: 17

# Baseline for benchmarking. Required if task_type = benchmark-kernel.
baseline:
  # One of: torch_permute | torch_contiguous | cublaslt_transform | naive_nosmem
  name: torch_contiguous
  tolerance_pct: 10
  # Accept the library path if your kernel is within tolerance_pct of it.
```

## Seed tasks (canonical, used for pattern onboarding)

These four seed tasks cover the dtype × shape matrix from the transpose seed-task matrix and are the acceptance benchmarks for this pattern.

### T1. Large fp32 transpose

```yaml
task_id: "2026-04-22-transpose-t1-fp32-large"
task_type: benchmark-kernel
shape: { M: 4096, N: 4096 }
dtype: float32
transpose_kind: matrix_2d
permutation: [1, 0]
baseline: { name: torch_contiguous, tolerance_pct: 10 }
expected_bw_gb_s_min: 1500
# On H200 the custom padded-smem kernel delivers ~1685 GB/s (measured
# in sources/experience/hw-probes/smem-tile-reuse/). Any kernel below 1500
# GB/s is likely missing padding or smem.
```

### T2. Small fp32 transpose (occupancy regime)

```yaml
task_id: "2026-04-22-transpose-t2-fp32-small"
task_type: benchmark-kernel
shape: { M: 256, N: 256 }
dtype: float32
transpose_kind: matrix_2d
permutation: [1, 0]
tile_size: { rows: 16, cols: 17 }
baseline: { name: torch_contiguous, tolerance_pct: 15 }
# Small matrix: smem staging may not pay off (pitfall P7). Allow wider
# tolerance; the library path is competitive here.
```

### T3. Large bf16 transpose

```yaml
task_id: "2026-04-22-transpose-t3-bf16-large"
task_type: benchmark-kernel
shape: { M: 8192, N: 8192 }
dtype: bfloat16
transpose_kind: matrix_2d
permutation: [1, 0]
# bf16 is 2 bytes, so [32][TILE+k] padding needs k such that the stride
# in 32-bit bank units becomes odd. [32][34] (in bf16 units) = +2 bytes
# = +1 32-bit bank — canonical.
tile_size: { rows: 32, cols: 34 }
baseline: { name: torch_contiguous, tolerance_pct: 10 }
```

### T4. Small bf16 transpose

```yaml
task_id: "2026-04-22-transpose-t4-bf16-small"
task_type: benchmark-kernel
shape: { M: 512, N: 512 }
dtype: bfloat16
transpose_kind: matrix_2d
permutation: [1, 0]
tile_size: { rows: 16, cols: 18 }
baseline: { name: torch_contiguous, tolerance_pct: 15 }
# Small bf16: warp-level tile strategy (one warp = one row-column pair)
# starts to dominate. Measuring the occupancy-tile size tradeoff is
# part of the acceptance criteria.
```

Results matrix (dtype × shape) across all four seed tasks is the acceptance deliverable for the pattern — one table with measured median-ms and effective BW, plus a speedup column vs each task's baseline.

## Notes

- For `transpose_kind: aos_to_soa`, the task defers to `wiki/nvidia/foundations/memory/layout-transform/` sub-skill S1. The kernel is a one-pass copy, not a smem-tiled transpose.
- For `transpose_kind: permute_nd`, collapse to one or more 2-D transposes over the batched remaining dims. See INDEX step 1 Q5b.
- The task packet should reference ROUTING.md for the applicable skills and INDEX.md for the decision tree that leads to each kernel variant.
