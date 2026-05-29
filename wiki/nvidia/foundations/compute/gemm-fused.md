---
title: Hopper fused-GEMM (epilogue activation fusion via cutlass; prologue queued)
status: partial
evidence_level: measured
applies_to_pattern_class:
- tensor-core
- tensor-core/gemm
- fused
applies_to_ops:
- gemm
requires_sm: '>=9.0a'
requires_features:
- tma
- wgmma
single_kernel_useful: true
cuda_version_tested: 12.9.86
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM | sm_90a | cuda 12.9.86 | driver 570.124.06
source:
- path: blogs/colfax/epilogue-fusion-in-cutlass-with-epilogue-visitor-trees
  anchor: epilogue-visitor-tree (EVT) abstraction; LinearCombination / Activation
    / Bias / Aux fusion patterns
- path: blogs/colfax/cutlass-tutorial-fast-matrix-multiplication-with-wgmma-on-nvidia-hopper-gpus
  anchor: epilogue customization for Hopper wgmma GEMM
artifacts:
  code: sources/experience/api-probes/gemm/artifacts/gemm_compare_relu.cu
  build: sources/experience/api-probes/gemm/artifacts/build_fused.sh
  introspection: sources/experience/api-probes/gemm/artifacts/device.json
  profile: sources/experience/api-probes/gemm/artifacts/profiles/2026-04-28-gemm-fused-epilogue.csv
upstream_repo: cutlass@f74fea9c
related_skills:
- warp-specialization
- persistent-kernel
- gemm-aligned
id: skill-gemm-fused
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: source-code/cutlass
  path: examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu
  anchor: S8 GEMM + epilogue-swizzle fusion (output-tensor stride/swizzle absorbed
    into the epilogue)
- source_id: source-code/cutlass
  path: examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu
  anchor: mixed-dtype GEMM (int4 × bf16) — int4 dequantization is the prologue fusion
    absorbed into the mainloop
- source_id: source-code/cutlass
  path: examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu
  anchor: epilogue-fused topK + softmax for output projection
- source_id: source-code/cutlass
  path: include/cutlass/epilogue/thread/linear_combination_relu.h
  anchor: LinearCombinationRelu — Activation fusion as a thread-level epilogue
---
# Hopper fused-GEMM (epilogue + prologue fusion via cutlass)

> **Status: partial.** Full characterization requires both an epilogue and a prologue fusion measured against non-fused reference chains. The **epilogue path** is verified: the fused kernel (`gemm_compare_relu.cu`, using `LinCombEltAct<ReLu, ...>`) is bit-faster than the non-fused chain (`gemm_compare` + standalone `relu_kernel`) by ~13% wall-clock at 2048³ on H200, with `verify()` enforcing a real fused-vs-fused-reference correctness gate (Disposition: Passed). The **prologue path** (mixed-dtype int4 × bf16 dequant absorbed into the mainloop, cutlass example 55) is **NOT yet measured**; the skill therefore stays `partial` until the prologue A/B lands. The epilogue measurements remain valid; promotion to `verified` requires the prologue half.

## What it is

Cutlass on Hopper exposes two fusion abstractions that absorb pre-/post-GEMM elementwise work into the same kernel as the matmul, eliminating a separate kernel launch + a full read/write of the intermediate tensor:

1. **Epilogue fusion** via `cutlass::epilogue::collective::CollectiveBuilder` with a thread-level epilogue functor (`LinearCombination`, `LinearCombinationRelu`, `LinearCombinationGelu`, …) OR a **Epilogue Visitor Tree (EVT)** for arbitrary multi-input fusion (bias add, residual add, top-K, softmax, channel-wise scale).
2. **Prologue fusion** via the mainloop's input element type (e.g. `MainloopMixedDtype` accepts `int4` × `bf16` and dequantizes `int4` → `bf16` inside the mainloop's TMA-load → smem path). This is exercised by cutlass example 55 (int4 × bf16 GEMM).

## When to use it

- LLM linear layers where the operator chain is `Y = activation(W·X + b)`: fuse bias + activation into the GEMM epilogue. Saves one kernel launch and one full read/write of `(M × N)` elements.
- Quantized inference: int4 weight × bf16 activation with dequantization absorbed into the mainloop (prologue fusion). Saves a separate `int4 → bf16` upcast pass.
- Output projection in attention: top-K + softmax fused via EVT (cutlass example 61).

## When NOT to use it

- When the elementwise work is bandwidth-bound and the GEMM is compute-bound and they share no producer/consumer relationship — fusing adds smem pressure to the GEMM kernel without saving bandwidth.
- When the fused operation needs a different tile size than the GEMM (e.g. attention rows aggregated across the tile boundary) — the fusion would either reduce GEMM occupancy or require cross-tile sync that doesn't fit the WS pipeline.

## Measured Characteristics

H200-SXM, sm_90a, cuda 12.9.86, problem 2048 × 2048 × 2048 in TF32 → F32:

| Path | Variant | Latency (μs) | TFLOPS | Total operation latency |
|---|---|---|---|---|
| **Fused** | `gemm_compare_relu` — cooperative GEMM with `LinCombEltAct<ReLu>` fused into the epilogue | **86.83** | 196.3 | **86.83 μs** |
| Non-fused (GEMM portion) | `gemm_compare` — cooperative GEMM, default `LinearCombination` epilogue | 91.92 | 186.9 | — |
| Non-fused (standalone ReLU on D) | `relu_kernel` — bandwidth-bound D round-trip at 4045 GB/s | 8.29 | n/a | — |
| **Non-fused total** | sum of the two | — | — | **100.21 μs** |

**Fusion savings**: 12.68 μs / 12.65% wall-clock at 2048³. The cleanly-isolated lower-bound saving is the standalone `relu_kernel`'s 8.29 μs (8.3% of non-fused total) — the bandwidth round-trip of D that fusion eliminates. The remaining 4.4 μs gap between the two cooperative variants is partly a schedule-spec confounder (the GEMM-only baseline uses `KernelScheduleAuto`; the fused variant uses explicit `KernelTmaWarpSpecializedCooperative` because Auto's epilogue does not support fusion — see `sm90_builder.inl` L615).

Memory traffic delta: fused = 50.33 MB, non-fused = 83.88 MB → **−33.55 MB per launch (40% less DRAM traffic)**.

Other fusion patterns documented but not timed yet:

| Pattern | Status | Where |
|---|---|---|
| Epilogue activation fusion (ReLU) | **Verified — measured A/B at 2048³** | `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md` |
| Epilogue swizzle (S8 GEMM, output stride/swizzle) | Correctness-only via cutlass example 50 | upstream `examples/50_hopper_gemm_with_epilogue_swizzle/` |
| Prologue dequant (int4 × bf16 mixed-dtype) | Reference only; queued for follow-up | upstream `examples/55_hopper_mixed_dtype_gemm/` |
| Epilogue topK + softmax via EVT | Reference only; queued for follow-up | upstream `examples/61_hopper_gemm_with_topk_and_softmax/` |

## Cross-references

- Aligned GEMM consumer: `wiki/nvidia/foundations/compute/gemm/aligned/skill.md` (the unfused baseline)
- Warp-specialization (the schedule that hosts the fused mainloop): `wiki/nvidia/techniques/warp-specialization/skill.md`
- TMA primitive (the prologue dequant uses TMA): `wiki/nvidia/hardware/tma/skill.md` (see also `sources/experience/api-probes/gemm/2026-04-28-tma-bandwidth-counters.md` for cutlass-API observations and `sources/experience/hw-probes/tma-ptx/2026-04-29-tma-throughput.md` for the isolated TMA bandwidth sweep)
- wgmma atom (the compute primitive): `wiki/nvidia/hardware/wgmma/skill.md` (see also `sources/experience/api-probes/gemm/2026-04-28-wgmma-counters.md` for cutlass-API observations and `sources/experience/hw-probes/wgmma-ptx/2026-04-29-wgmma-zoo.md` for the isolated 11-config wgmma family zoo)
- Library-usage notes + tuning log: `wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-fused/{README.md, tuning.md}`. Canonical reproducible artifact: `sources/experience/api-probes/gemm/artifacts/gemm_compare_relu.cu` + upstream `examples/50_hopper_gemm_with_epilogue_swizzle/` built in place by `run_fused.sh`.
- Probe (verified epilogue, partial prologue): `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md`
- Failure modes: `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/pitfalls.md`
- Tuning parameter space: tuning.md.
