---
id: code-cutlass-cute-gemm-fused-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
languages:
- cuda-cpp
- cute-dsl
techniques:
- warp-specialization
- register-budgeting
- kernel-fusion
- software-exp
kernel_types:
- gemm
- attention
- fused-kernel
- quantization
confidence: inferred
tags:
- warp-specialization
- register-budgeting
- kernel-fusion
- software-exp
- gemm
- attention
- fused-kernel
- quantization
- cuda-cpp
- cute-dsl
architectures:
- sm90
- sm90a
---
# gemm-fused — tuning log (skeleton)

Fusion decision tree + EVT plumbing notes. Numbers are link-only — pull from `sources/experience/api-probes/gemm.md` (ReLU fusion A/B probe).

## When to fuse (vs run a separate post-kernel)

```
Q1. Does the post-GEMM op need to run inside this custom kernel?
    YES -> Q2.
    NO  -> keep the unfused GEMM path for this walkthrough.

Q2. Is the GEMM at a shape where the post-GEMM kernel's DRAM round-trip
    (read D, write D') dominates its own compute?
    YES -> fuse via cutlass custom epilogue (LinearCombinationXxx, EVT, or
           Sm90TmaWarpSpecializedAdapter). Round-trip cost saved.
    NO  -> separate kernel. Fusion's added register pressure / code complexity
           does not pay back when the post-kernel is compute-bound.
```

## Epilogue functor catalogue

| Functor | Operation | When |
|---|---|---|
| `LinearCombination<T, ...>` | `D = alpha * acc + beta * C` | always available |
| `LinearCombinationRelu<T, ...>` | `D = max(alpha * acc + beta * C, 0)` | ReLU after GEMM |
| `LinearCombinationGELU<T, ...>` | GELU after GEMM | activation epilogue |
| `LinearCombinationClamp<T, ...>` | bounded output (quantize) | scale + clamp |
| EVT (Epilogue Visitor Tree) | arbitrary expression tree | when no canned functor matches |

## EVT plumbing (Sm90)

- Compose tree nodes via `cutlass::epilogue::fusion::Sm90TreeVisitor`.
- Producer/consumer split inside the epilogue is handled by `sm90_callbacks_tma_warpspecialized.hpp`.
- TopK + softmax (example 61) demonstrates a non-trivial EVT.

## Open questions

- Quantitative fused-vs-non-fused timing A/B at 2048³ (ReLU + bias-add). Probe is in the repo; numbers not yet posted to skill.
- Whether `LinearCombinationGELU` and `LinearCombinationRelu` give bit-identical output to a separate kernel implementation (numerical sanity).

## References

- Skill: `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md`
- Pitfalls: `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/pitfalls.md`
- Probe: `sources/experience/api-probes/gemm.md`
- Custom-kernel decision tree: `wiki/nvidia/operator-routing/gemm/INDEX.md`
