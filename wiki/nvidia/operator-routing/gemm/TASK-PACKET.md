---
title: Tensor-core GEMM Pattern -- Task Packet Template
pattern_class: tensor-core
op: gemm
status: draft
source:
- path: reasoning/task-packet.md
  anchor: L1-L109
  excerpt: 'The KB-gen agent takes exactly one input: a YAML file under tasks/. This file is the task packet.'
- path: wiki/nvidia/operator-routing/reduction/TASK-PACKET.md
  anchor: Reduction -- Task Packet Template
  excerpt: Reference shape for the operator-specific task packet refinement
id: routing-gemm-TASK-PACKET
type: operator-routing
vendor: nvidia
operator: gemm
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
- cute-dsl
hardware_features:
- wgmma
- tma
- mbarrier
- cluster
techniques:
- warp-specialization
- persistent-kernel
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- tma-multicast
- software-exp
kernel_types:
- gemm
- attention
- fused-kernel
- quantization
confidence: inferred
tags:
- wgmma
- tma
- mbarrier
- cluster
- warp-specialization
- persistent-kernel
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- tma-multicast
- software-exp
- gemm
- attention
- fused-kernel
- quantization
- ptx
- cuda-cpp
- cute-dsl
---
# Tensor-core GEMM -- Task Packet Template

This document defines the operator-specific task packet fields for a Hopper tensor-core GEMM kernel-writing task. It refines the generic task packet contract in `reasoning/task-packet.md` with GEMM-specific required and optional fields.

## Required fields (in addition to base task-packet fields)

```yaml
# --- Base fields (from reasoning/task-packet.md) ---
task_id: "2026-04-XX-gemm-<variant>"            # date-prefixed, kebab-case
task_type: write-kernel                          # or benchmark-kernel
target_path: kernels/gemm/<variant>/             # output directory

hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

# --- GEMM-specific fields ---
op: gemm

shape:
  M: 2048
  N: 2048
  K: 2048
  # Logical D[M, N] = A[M, K] * B[K, N]
  # Specify exact integers; if the kernel must serve a range, list the
  # representative shapes the benchmark must cover.

dtype:
  A: bfloat16
  B: bfloat16
  C: float32
  D: float32
  accumulator: float32
  # Supported combos on Hopper sm_90a:
  #   bf16 / bf16 / f32 / f32  (most common)
  #   fp16 / fp16 / f32 / f32
  #   fp16 / fp16 / f16 / f16
  #   tf32 / tf32 / f32 / f32  (cutlass MMA_64x128x8_F32TF32TF32_SS_TN family)
  #   fp8_e4m3 / fp8_e4m3 / f32 / bf16  (with scale; outside MVP)
  #   int8 / int8 / int32 / int32

layout:
  A: row-major     # "row-major" or "col-major"
  B: row-major
  C: row-major
  D: row-major
  # Cutlass-free PTX track: B in conceptual K-major (= col-major K x N) is
  # required for wgmma .SS_TN (see wiki/nvidia/foundations/compute/gemm-ptx/pitfalls.md #0).
  # The host driver typically transposes B accordingly. Document this in
  # `notes:` when relevant.

baseline:
  library: "cublasLtMatmul"
  # The library baseline to benchmark against. One of:
  #   torch.matmul, torch.nn.functional.linear,
  #   cublasGemmEx, cublasLtMatmul,
  #   cutlass::gemm::device::GemmUniversalAdapter.
  # The custom kernel's latency must approach or beat this baseline
  # to justify its existence (see INDEX.md Step 0).

  # OR for a cutlass-free build, an additional structural baseline:
  cutlass_free: true
  # When set, the build script MUST run BOTH:
  #   nvcc -E <src>.cu | grep -E 'cutlass::|cute::'        # expect 0
  #   cuobjdump --dump-elf-symbols <bin> | grep -E 'cutlass::|cute::'  # expect 0
  # See wiki/nvidia/foundations/compute/gemm-ptx.md "Cutlass-free verification".

track:
  - cutlass-api    # one of:  cutlass-api  |  cutlass-free
  # Picks the implementation track per INDEX.md Q4. Determines which
  # skills in ROUTING.md apply.

ws_variant: plain  # one of: plain | pingpong | cooperative
  # Picks the warp-specialization variant per INDEX.md Q5. Plain is
  # the safe default for problems below ~a few cluster-grids of CTAs.
```

## Optional fields

```yaml
fusion:
  prologue: none           # none | dequantize | scale-cast
  epilogue:                # none, OR a list:
    - bias-add
    - relu
  # Standard fused epilogues that cuBLASLt can absorb (BIAS / RELU / GELU
  # and combinations) -- listing these here is a hint that the library
  # path Q2 in INDEX.md likely covers the case. For non-standard fusion
  # (softmax, per-row reduction, custom quantize), describe in `notes:`.

cluster_shape: "1,1,1"
  # Hopper cluster geometry. Pingpong implies "2,1,1"; cooperative implies
  # "4,2,1" or larger with cluster-multicast. Plain WS uses "1,1,1".

stages: 4
  # TMA pipeline depth in smem stages. 2 is the WS minimum, 3-4 is the
  # H200 sweet spot, >4 rarely justified (WS pitfall #1).

split_k: 1
  # If > 1, the K dimension is split across split_k CTAs whose partial
  # outputs are reduced. Useful when the M*N tile grid leaves SMs idle
  # but K is large enough to absorb the split.

persistent: false
  # Whether the CTA persists across multiple output tiles. True for
  # pingpong / cooperative; false for plain WS. Pair with `ws_variant`.

success_criteria:
  - correctness: "max_rel_diff < 1e-2 vs cublasLtMatmul baseline"
  - performance: "median_latency <= 1.05 * cublasLtMatmul_latency"
  # Cutlass-free track also asserts:
  - cutlass_free_preprocessor: "nvcc -E <src> | grep -E 'cutlass::|cute::' == 0 matches"
  - cutlass_free_binary:       "cuobjdump --dump-elf-symbols <bin> | grep -E 'cutlass::|cute::' == 0 matches"

notes: |
  Free-form guidance for the kernel-writing agent.
  Example: "Cutlass-free build: B must be physically transposed to col-major K x N
  before TMA load (wgmma .SS_TN expects K-major B); use the host-side transpose
  pattern from artifacts/experience/api-probes/gemm-ptx/gemm_ptx.cu. Inherit the per-thread fragment-store
  layout from wiki/nvidia/foundations/compute/gemm-ptx.md verbatim until the upstream
  fix for pitfall #1 lands."
```

## Example A: cutlass-API plain warp-specialized GEMM at 2048^3 bf16

```yaml
task_id: 2026-04-30-gemm-aligned-bf16-2048
task_type: write-kernel
target_path: kernels/gemm/aligned-bf16-2048/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

op: gemm
shape: { M: 2048, N: 2048, K: 2048 }
dtype: { A: bfloat16, B: bfloat16, C: float32, D: float32, accumulator: float32 }
layout: { A: row-major, B: col-major, C: row-major, D: row-major }
baseline:
  library: cublasLtMatmul
track: cutlass-api
ws_variant: plain
cluster_shape: "1,1,1"
stages: 4
persistent: false

success_criteria:
  - correctness: "max_rel_diff < 1e-2 vs cublasLtMatmul baseline"
  - performance: "median_latency <= 1.05 * cublasLtMatmul_latency"

references:
  - wiki/nvidia/operator-routing/tensor-core/gemm/INDEX.md
  - wiki/nvidia/operator-routing/tensor-core/gemm/ROUTING.md
  - wiki/nvidia/foundations/compute/gemm.md
  - wiki/nvidia/techniques/warp-specialization.md
  - reasoning/bottleneck-triage.md

notes: |
  Aligned shape (2048 % 64 == 0, % 128 == 0). Use the cutlass collective
  builder's KernelTmaWarpSpecialized + NoSmemWarpSpecialized epilogue.
  Drop to KernelScheduleAuto if benchmarking shows the explicit pinning
  is sub-optimal at this shape.
```

## Example B: cutlass-free warp-specialized GEMM (single CTA, 2 K-tiles)

```yaml
task_id: 2026-04-29-gemm-ws-ptx-min
task_type: write-kernel
target_path: sources/experience/kernel-records/2026-04-29-gemm-ws-ptx/
hardware:
  device: H200
  sm: "9.0a"
  cuda_runtime: "12.9"

op: gemm
shape: { M: 64, N: 8, K: 32 }
dtype: { A: bfloat16, B: bfloat16, C: float32, D: float32, accumulator: float32 }
layout: { A: row-major, B: col-major, C: row-major, D: row-major }
baseline:
  library: cublasGemmEx
  cutlass_free: true
track: cutlass-free
ws_variant: plain
cluster_shape: "1,1,1"
stages: 2
persistent: false

success_criteria:
  - correctness: "kernel runs to completion (= producer/consumer mbarrier protocol does not deadlock)"
  - cutlass_free_preprocessor: "nvcc -E gemm_ws_ptx.cu | grep -E 'cutlass::|cute::' == 0 matches"
  - cutlass_free_binary:       "cuobjdump --dump-elf-symbols gemm_ws_ptx | grep -E 'cutlass::|cute::' == 0 matches"
  # Numeric vs cuBLAS is *not* a pass criterion at this commit because the
  # kernel inherits the per-thread fragment-store residual from gemm-ptx
  # pitfall #1. Once that fix lands upstream, add:
  #   - correctness: "max_rel_diff < 1e-2 vs cublasGemmEx"

references:
  - wiki/nvidia/operator-routing/tensor-core/gemm/INDEX.md
  - wiki/nvidia/operator-routing/tensor-core/gemm/ROUTING.md
  - wiki/nvidia/foundations/compute/gemm-ptx.md
  - wiki/nvidia/foundations/compute/gemm-ptx/pitfalls.md
  - wiki/nvidia/hardware/tma/skill-tma-ptx.md
  - wiki/nvidia/hardware/wgmma/skill-wgmma-ptx.md
  - wiki/nvidia/techniques/warp-specialization.md
  - wiki/nvidia/techniques/warp-specialization/pitfalls.md

notes: |
  Smallest shape that exercises the WS pipeline: STAGES=2, N_K_TILES=2.
  Producer warp = warp 4; consumer warpgroup = warps 0..3. mbarrier
  full / empty pair per stage, expected_tx = TILE_BYTES_A + TILE_BYTES_B.
  B is host-transposed to col-major K x N before TMA load (gemm-ptx pitfall #0).
  See sources/experience/kernel-records/2026-04-29-gemm-ws-ptx.md for
  the full design walkthrough.
```

## Agent workflow when receiving a tensor-core GEMM task packet

1. Read `INDEX.md` -- follow Step 0 (library), Step 1 (track choice), Step 2 (warp-spec variant).
2. If the library path suffices, report the recommended call shape and stop (no custom kernel needed).
3. If a custom kernel is needed, read `ROUTING.md` for the skill whitelist; instantiate the producer side (skill #6) and consumer side (skill #7) under the warp-spec mainloop (skill #1).
4. (Cutlass-free track only) Run BOTH cutlass-free gates (`nvcc -E | grep` and `cuobjdump --dump-elf-symbols | grep`) before reporting success. The structural gate is non-negotiable; the numeric gate may be deferred if a known upstream issue applies, in which case document the deferral in `notes:`.
5. Benchmark against the `baseline.library` and apply bottleneck triage (`reasoning/bottleneck-triage.md`) if the success criteria are not met.
6. After at most 3 optimization iterations, finalize or report `status: stuck`.
