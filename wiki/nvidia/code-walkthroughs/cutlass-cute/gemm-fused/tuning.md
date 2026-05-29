---
id: code-cutlass-cute-gemm-fused-tuning
type: code-walkthrough
vendor: nvidia
title: Tuning
upstream_repo: NVIDIA/cutlass-cute
---
# gemm-fused — tuning log (skeleton)

Fusion decision tree + EVT plumbing notes. Numbers are link-only — pull from `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md` (ReLU fusion A/B probe).

## When to fuse (vs run a separate post-kernel)

```
Q1. Is the post-GEMM op in cuBLASLt's epilogue catalogue?
    YES -> use cuBLASLt with the corresponding CUBLASLT_EPILOGUE_*. Done.
    NO  -> Q2.

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
- Probe: `sources/experience/api-probes/gemm/2026-04-28-gemm-fused.md`
- cuBLASLt fallback: `wiki/nvidia/operator-routing/tensor-core/gemm/library-fallback.md` §3
