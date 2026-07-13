---
id: pitfall-gemm-fused
type: pitfall
vendor: nvidia
title: Pitfalls
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
hardware_features:
- wgmma
- tma
techniques:
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- swizzling
kernel_types:
- gemm
- fused-kernel
- quantization
confidence: inferred
tags:
- wgmma
- tma
- pipeline-stages
- kernel-fusion
- shared-memory-optimization
- swizzling
- gemm
- fused-kernel
- quantization
- cuda-cpp
---
# Hopper fused-GEMM (epilogue + prologue) — pitfalls

## 1. Epilogue smem footprint reduces mainloop occupancy

`LinearCombination` is small. `LinearCombinationRelu` is the same size (one fmaxf added). But the EVT for multi-input fusion (bias + residual + activation) consumes additional shared-memory tiles because each auxiliary tensor must be loaded by the epilogue. The `StageCountAutoCarveout` template parameter computes the maximum mainloop pipeline depth that fits given the epilogue's smem footprint; a heavier EVT shrinks `Stages`, which shrinks the wgmma pipeline depth, which can drop throughput. Always measure `Stages` after composing the EVT and re-measure throughput at the production shape.

## 2. Prologue dequant only works when the dequant function is bandwidth-light enough to fit in TMA cycles

cutlass's mixed-dtype prologue (example 55) absorbs the int4 → bf16 dequantization into the TMA-load → smem path. This works because the dequant is a single multiply-add per element and the TMA load is bandwidth-bound. If your "prologue" is a 32-tap convolution or a layer-norm, it does NOT fit in the prologue and you must run a separate kernel.

## 3. Epilogue swizzle changes the output access pattern, not the math

Cutlass example 50 demonstrates an "epilogue swizzle" that reshapes the output tensor's storage layout for downstream consumption. The math is identical to a plain `LinearCombination`; only the store stride/swizzle changes. Don't confuse "epilogue swizzle" with "fused activation"; they are independent fusion axes.

## 4. EVT compile times are long

EVT-based fusion produces a heavily templated kernel; compile times of 60-300 seconds per kernel are routine. For development iteration, prefer the thread-level epilogue functors (`LinearCombination*`) and fall back to EVT only when you need multi-input fusion (bias + residual + …).

## 5. Numerical equivalence with a non-fused chain is NOT guaranteed bit-for-bit

A fused `Y = ReLU(α·AB + β·C)` does the ReLU before the F32→F16 (or whatever the output dtype is) downcast. The non-fused chain `Y' = ReLU(downcast(α·AB + β·C))` does the downcast first. The two are numerically very close at TF32/F32 → F32 (no downcast); they can differ at TF32 → bf16 / int8 because the downcast quantizes negative-near-zero values that ReLU would have killed. Always document the numerical boundary in a `pitfalls.md` for the specific operator chain.

## 6. reference GEMM does not expose all the same fusion patterns

reference GEMM (the heuristic-driven matmul) supports linear-combination + bias + ReLU/GELU as fused epilogues (`named epilogue`, …) but does NOT expose arbitrary EVT trees. If your fusion needs a custom pattern that reference GEMM can't express, cutlass + EVT is the production path; if your fusion is one of reference GEMM's named patterns, reference GEMM may be faster because of its per-shape kernel selection.
