---
title: 'TMA: Tensor Memory Accelerator end-to-end via cutlass-cute on Hopper'
status: verified
evidence_level: measured
applies_to_pattern_class:
- tensor-core
applies_to_ops:
- gemm
- attention
requires_sm: '>=9.0a'
requires_features:
- tma
- mbarrier
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: spec
  anchor: Reference
artifacts:
  code: sources/experience/api-probes/gemm/artifacts/gemm_compare.cu
  build: sources/experience/api-probes/gemm/artifacts/build.sh
  introspection: sources/experience/api-probes/gemm/artifacts/device.json
  profile: sources/experience/api-probes/gemm/artifacts/profiles/2026-04-28-tma-counters.csv
upstream_repo: cutlass@f74fea9c
related_apis: []
related_skills:
- wgmma
- warp-specialization
id: skill-tma
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: include/cute/atom/copy_traits_sm90_tma.hpp
  anchor: SM90_TMA_LOAD / SM90_TMA_LOAD_MULTICAST
- source_id: source-code/cutlass
  path: examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu
  anchor: Lall
- source_id: blogs/colfax
  path: cutlass-tutorial-mastering-the-nvidia-tensor-memory-accelerator-tma
  anchor: TMA descriptor + cp.async.bulk.tensor end-to-end
---
# TMA — Tensor Memory Accelerator on Hopper

## What it is

TMA is the Hopper-and-newer hardware copy engine that moves multi-dimensional tensor tiles between global and shared memory **without occupying compute warps**. The host encodes a `CUtensorMap` descriptor (shape, stride, swizzle, element type, …) once, copies it to a device-side symbol, and the device-side `cp.async.bulk.tensor.{1d|2d|3d|…}.{shared::cta|shared::cluster}.global` PTX instructions issue the actual transfer asynchronously. Completion is signaled via `mbarrier`, and only the warp that issues the bulk-load needs to be involved in scheduling the transfer; data arrives on its own and the consumer wgmma threads wait on the mbarrier.

In cutlass-cute the entire pattern is wrapped in two atoms — `SM90_TMA_LOAD` (single-CTA destination) and `SM90_TMA_LOAD_MULTICAST` (cluster-CTA destination, used by the cooperative GEMM kernel). `cute::copy` dispatches to the right atom based on the partition's layout.

## Why it helps

- **Decouples address generation from compute warps.** Pre-Hopper, every `cp.async` issued by the loader warps occupied warp scheduler slots and stole issue bandwidth from compute. TMA fires `cp.async.bulk.tensor` once and the engine handles the rest.
- **Multidimensional + swizzle-aware in a single instruction.** A 2-D tile with bank-conflict-free swizzle can be loaded with one PTX call instead of N nested loops.
- **Cluster multicast.** With the `MULTICAST` form, one global-memory read fans out to several CTAs in the same cluster, halving DRAM traffic in cooperative GEMMs.

## When to use it

- Any single-kernel gemm/attention that targets ≥80 % of H200 peak.
- Operators with **regular tile geometry** that fits one of the supported descriptor shapes (1-D, 2-D, 3-D, 4-D, 5-D); irregular/scatter loads still need `cp.async`.
- The kernel has a clear producer/consumer split (cutlass cooperative + ping-pong patterns).

## When **not** to use it

- Variable-stride or scatter loads — TMA descriptors are static.
- Targets older than sm_90a — TMA only exists on Hopper and Blackwell.
- Tiles that don't satisfy the bytes-per-element + alignment constraints (see `pitfalls.md`).

## Measured Characteristics

End-to-end TMA-driven GEMM via `examples/48_hopper_warp_specialized_gemm` at the cutlass commit pin (see `artifacts.code`):

- 5120 × 4096 × 4096 GEMM (TF32 inputs, F32 acc), **0.608 ms / launch ≈ 282.6 TFLOPS** on H200-SXM, `Disposition: Passed` against the cutlass reference.
- Sustained SM utilization: `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` = **87.77 %** (kernel name decoded; `SM90_TMA_LOAD_MULTICAST` atom + `Swizzle<3,4,3>` smem layout).
- Compute-bound at this shape: SM throughput 30.95 % of SoL, DRAM throughput 3.32 % — TMA is keeping wgmma fed without saturating DRAM.

Full probe record: sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md.

For the TMA primitive in isolation (single-tile correctness + 16-config DRAM-bound bandwidth sweep, peak 3.72 TB/s = 77.5 % HBM3e on H200), see the cutlass-free hw-probes sources/experience/hw-probes/tma-ptx/2026-04-28-tma-ptx-hello.md and sources/experience/hw-probes/tma-ptx/2026-04-29-tma-throughput.md.

## Minimum repro

`sources/experience/api-probes/gemm/artifacts/gemm_compare.cu` (vendored from cutlass example 48 at commit `f74fea9c`), plus `build.sh` (nvcc invocation), `run.sh` (build → run → ncu CSV → compute-sanitizer), and `device.json` (H200 nvidia-smi snapshot) in the same `artifacts/` directory. Build requires the cutlass include tree at the same pin and CUDA 12.9 on H200.

## How it connects to the rest of the KB

- Pairs with `wiki/nvidia/hardware/wgmma/skill.md` — the same kernel exercises both. TMA loads the operands; wgmma consumes them.
- Used by `wiki/nvidia/techniques/warp-specialization/` — warp-specialization without TMA defeats the purpose, since the producer warpgroup needs a non-blocking load primitive.
- Pre-condition for `wiki/nvidia/foundations/compute/gemm/aligned/`.
