---
title: Hopper TMA via raw PTX (cutlass-free)
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
- tensor-core/gemm
applies_to_ops:
- memory-load
- gemm
requires_sm: '>=9.0a'
requires_features:
- tma
- mbarrier
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9 + libcuda (driver API for cuTensorMapEncodeTiled)
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: sources/experience/hw-probes/tma-ptx/artifacts/tma_hello.cu
  anchor: cutlass-free hello-world (correctness)
- path: sources/experience/hw-probes/tma-ptx/artifacts/tma_throughput_probe.cu
  anchor: cutlass-free throughput sweep over swizzle × pipeline-depth × box-rows (16
    configs)
artifacts:
  code: sources/experience/hw-probes/tma-ptx/artifacts/tma_hello.cu
  build: sources/experience/hw-probes/tma-ptx/artifacts/build.sh
  throughput_code: sources/experience/hw-probes/tma-ptx/artifacts/tma_throughput_probe.cu
  throughput_build: sources/experience/hw-probes/tma-ptx/artifacts/build_throughput.sh
  throughput_run: sources/experience/hw-probes/tma-ptx/artifacts/run_throughput.sh
  multicast_code: sources/experience/hw-probes/tma-ptx/artifacts/tma_multicast_probe.cu
  multicast_build: sources/experience/hw-probes/tma-ptx/artifacts/build_multicast.sh
  multicast_run: sources/experience/hw-probes/tma-ptx/artifacts/run_multicast.sh
  multicast_v2_code: sources/experience/hw-probes/tma-ptx/artifacts/tma_multicast_v2.cu
  multicast_v2_run: sources/experience/hw-probes/tma-ptx/artifacts/run_multicast_v2.sh
  profile: sources/experience/hw-probes/tma-ptx/artifacts/profiles/2026-04-29-tma-throughput.csv
related_apis: []
related_skills:
- tma
- wgmma-ptx
- gemm-ptx
id: skill-tma-ptx
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: include/cute/arch/copy_sm90_tma.hpp
  anchor: cutlass's TMA inline-PTX wrapper (used as a reference; not included in our
    binary)
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L16866-L16880
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L16316
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L20810-L20825
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11833
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3459
- source_id: blogs/colfax
  path: cutlass-tutorial-mastering-the-nvidia-tensor-memory-accelerator-tma
  anchor: Hopper TMA walkthrough — cuTensorMapEncodeTiled + cp.async.bulk.tensor +
    mbarrier protocol
---
# Hopper TMA via raw PTX (cutlass-free)

## What it is

A minimal implementation of Hopper's Tensor Memory Accelerator (TMA) tile load in pure CUDA + inline PTX, with **no cutlass or cute headers**. The host builds a `CUtensorMap` via the CUDA Driver API (`cuTensorMapEncodeTiled`); the device kernel issues `cp.async.bulk.tensor.2d.shared::cluster.global` from inline PTX and synchronizes via `mbarrier.init` / `mbarrier.arrive.expect_tx` / `mbarrier.try_wait.parity`.

Compared to the cutlass-API track (which uses cutlass's `SM90_TMA_LOAD` cute atom), this skill:

- Constructs the `CUtensorMap` directly with `cuTensorMapEncodeTiled` from `<cuda.h>` (driver API).
- Issues the TMA load and mbarrier protocol from inline PTX in a plain CUDA kernel.
- Verifies zero `cutlass::` / `cute::` symbols in both the preprocessed source (`nvcc -E | grep`) and the linked binary (`cuobjdump --dump-elf-symbols | grep`).

## When to use it

- Building cutlass-free kernels (composes with `wgmma-ptx` for a full cutlass-free GEMM).
- Auditing the TMA protocol at the PTX level — cutlass's `SM90_TMA_LOAD` cute atom is a thin wrapper over the same instructions.
- Educational: understanding `cuTensorMapEncodeTiled` parameter conventions (the global_stride array has rank-1 entries, dimension order is fastest-moving first, etc.).

## When NOT to use it

- Production GEMM kernels where cutlass's tile / pipeline / scheduler abstractions would save thousands of lines of code.
- Use cases that need swizzled smem layouts (the hello-world uses `CU_TENSOR_MAP_SWIZZLE_NONE`); swizzle-128B / 64B / 32B require additional bit-layout work in both the encode call and the descriptor consumer (wgmma).

## PTX mnemonic inventory (this skill)

| Mnemonic | Purpose |
|---|---|
| `mbarrier.init.shared.b64` | Initialize a shared-memory mbarrier for `n_threads` arrivals. |
| `mbarrier.arrive.expect_tx.shared::cta.b64` | Arrive at the barrier and declare an expected-bytes transaction count. |
| `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` | Issue the TMA load: copies a 2D tile from global to shared, signals the mbarrier when the transaction completes. |
| `mbarrier.try_wait.parity.shared::cta.b64` | Spin-wait for the barrier's phase to flip (the load to complete). Idiomatic spin-pattern wraps this in a `bra`-loop until predicate true. |
| `cp.async.bulk.tensor.<n>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` | **Cluster-multicast** variant: one CTA issues; all CTAs whose bit is set in the trailing 16-bit `cta_mask` receive the data into their local smem and have their local mbarrier signalled. Same DRAM read serves multiple CTAs. Requires the kernel to be launched with `__cluster_dims__(C, 1, 1)` (or runtime cluster attribute). |
| `barrier.cluster.arrive.aligned` / `barrier.cluster.wait.aligned` | Cluster-wide barrier. Used between mbarrier init and the first multicast issue to ensure all CTAs in the cluster have their local mbarriers ready before the leader signals them. |

## Host API: cuTensorMapEncodeTiled

```cpp
CUtensorMap tensor_map;
cuuint64_t global_dim[2]    = {SRC_COLS, SRC_ROWS};                  // fastest-moving first
cuuint64_t global_stride[1] = {SRC_COLS * sizeof(float)};            // rank-1 entries (no stride for fastest dim)
cuuint32_t box_dim[2]       = {TILE_COLS, TILE_ROWS};                // tile shape
cuuint32_t elem_stride[2]   = {1, 1};                                // element stride within the tile

cuTensorMapEncodeTiled(
    &tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    /*tensor_rank=*/2,
    /*global_address=*/d_src,
    global_dim, global_stride, box_dim, elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

The `CUtensorMap` is opaque (128 bytes) and must live in either `__grid_constant__` device-side memory or be passed by-pointer to the kernel as cutlass example 48 does. This skill's hello-world copies it into an explicit device-side allocation for simplicity.

## Measured Characteristics

### Correctness (hello-world)

H200-SXM, sm_90a, cuda 12.9.86. The hello-world kernel loads a 16 × 32 float tile from a 64 × 32 source tensor and verifies element-by-element:

| Output | Expected | Measured |
|---|---|---|
| matched / total | 512 / 512 | 512 / 512 ✓ |
| `h_out[0]` | 0 | 0 ✓ |
| `h_out[1]` | 1 | 1 ✓ |
| `h_out[TILE_COLS]` (= start of row 1) | 100 | 100 ✓ |
| `h_out[TILE_ROWS*TILE_COLS - 1]` | 1531 | 1531 ✓ |

Cutlass-free verification:

| Check | Expected | Result |
|---|---|---|
| `nvcc -E tma_hello.cu \| grep -E 'cutlass::\|cute::'` | 0 matches | 0 matches ✓ |
| `cuobjdump --dump-elf-symbols tma_hello \| grep -E 'cutlass::\|cute::'` | 0 matches | 0 matches ✓ |

### Throughput (16-config sweep)

H200-SXM, sm_90a. 132 CTAs × 1 warp cooperatively stream a 128 MiB bf16 source (> 60 MiB L2 → DRAM-bound) through pipelined TMA tile loads. 5 warmup + 20 timed launches, median ms. Source `2026-04-29-tma-throughput.csv`.

**Peak observed: 3.72 TB/s = 77.5 % of H200 HBM3e datasheet peak (~4.8 TB/s)** at `SWIZZLE_128B + depth=4 + box=128×64 bf16` (16 KiB tile).

| swizzle | fast-axis B | depth=1 GB/s | depth=2 GB/s | depth=4 GB/s |
|---------|------------:|-------------:|-------------:|-------------:|
| NONE    | 128 |       1444.9 |       2363.3 |       3309.8 |
| 32B     |  32 |        540.6 |        769.0 |       1326.4 |
| 64B     |  64 |        764.4 |       1344.7 |       2093.0 |
| 128B    | 128 |       1442.4 |       2376.8 |   **3322.9** |

Box-row sweep at SWIZZLE_128B / depth=4 / fast-axis 64 bf16:

| box_rows | tile bytes | GB/s |
|---------:|-----------:|-----:|
|        8 |      1 KiB |  847 |
|       16 |      2 KiB | 1514 |
|       32 |      4 KiB | 2410 |
|       64 |      8 KiB | 3323 |
|      128 |     16 KiB | **3718** |

### Cluster-multicast (3-config sweep)

H200-SXM, sm_90a. 132 CTAs arranged as 132/C clusters of size C ∈ {1, 2, 4}; each cluster's leader issues 124 multicast TMA loads via `cp.async.bulk.tensor.2d…multicast::cluster` with `cta_mask = (1<<C)-1`; all C CTAs receive the same tile sequence into their own smem. Per-CTA workload held constant; DRAM bytes scale 1/C while smem-bytes-delivered stays at 128 MiB. Source `2026-04-30-tma-multicast.csv`.

| config       | C | DRAM bytes | DRAM GB/s | effective smem GB/s | amplification vs c=1 |
|--------------|--:|-----------:|----------:|--------------------:|---------------------:|
| baseline     | 1 |    128 MiB |      3299 |                3299 |                1.00× |
| multicast c2 | 2 |     64 MiB |      2068 |            **4136** |          **1.26×**   |
| multicast c4 | 4 |     32 MiB |      1480 |            **5918** |          **1.79×**   |

Cluster-multicast is the hardware mechanism that makes cooperative warp-specialised GEMM bandwidth-efficient — when both CTAs in a cluster need the same A or B tile, multicast loads it from DRAM once instead of `C` times. The amplification is sub-linear vs the C× ideal at this 132-CTA grid because shrinking the issuer count (132 → 33) drops DRAM-controller parallelism faster than per-issuer bandwidth recovers. For real GEMM grids (1000s of CTAs) the multiplier approaches C×.

## Recommended configuration (cutlass-free TMA pipelines)

| If feeding wgmma | If pure streaming |
|---|---|
| `SWIZZLE_128B` (wgmma's smem descriptor expects it) | `SWIZZLE_NONE` (bandwidth-equivalent, fewer constraints) |
| fast-axis = 64 bf16 / 32 fp32 (= 128 B) | any 16-byte-aligned fast-axis |
| `box_rows ≥ 64` | `box_rows ≥ 64` |
| pipeline depth ≥ 4 (smem ≈ 32 KiB at 8 KiB tiles) | pipeline depth ≥ 4 |

The two main throughput levers are **tile size** (linear in TFLOPS up to ~16 KiB tiles) and **pipeline depth** (≥ 4 reaches 69 % HBM peak; depth=1 caps at 30 %). Swizzle mode itself is bandwidth-neutral when fast-axis bytes are held equal — `SWIZZLE_NONE @ 128 B fast-axis` matches `SWIZZLE_128B @ 128 B` to within 0.4 %. Swizzle's role is downstream (wgmma smem-descriptor compatibility), not upstream (TMA load bandwidth).

## Cross-references

- TMA reference (cutlass-API path): `wiki/nvidia/hardware/tma/skill.md` + `sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md`.
- Throughput probe with full sweep + open questions: `sources/experience/hw-probes/tma-ptx/2026-04-29-tma-throughput.md`.
- Cluster-multicast probe (this skill's `multicast::cluster` PTX variant): `sources/experience/hw-probes/tma-ptx/2026-04-30-tma-multicast.md` (v1, 3 cluster sizes) + `sources/experience/hw-probes/tma-ptx/2026-04-30-tma-multicast-v2.md` (follow-up: cluster sizes 1–16, multi-producer-warp, L2 promotion, cross-CTA empty mbarrier protocol via `mapa` + `mbarrier.arrive.release.cluster`, numeric correctness).
- wgmma-PTX sibling (also cutlass-free): `wiki/nvidia/hardware/wgmma-ptx/skill.md`.
- Cutlass-free GEMM (composes both PTX primitives): `wiki/nvidia/foundations/compute/gemm-ptx/skill.md`.
- Failure modes: `wiki/nvidia/hardware/tma-ptx/pitfalls.md`.
