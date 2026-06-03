---
id: code-cutlass-cute-gemm-fused-README
type: code-walkthrough
vendor: nvidia
title: Readme
upstream_repo: NVIDIA/cutlass-cute
---
# wiki/nvidia/code-walkthroughs/cutlass-cute/gemm-fused — fused-GEMM library usage

Library-usage knowledge for cutlass's fused-GEMM patterns (epilogue + prologue fusion). The canonical reproducible artifacts:

- Reference build of upstream `examples/50_hopper_gemm_with_epilogue_swizzle/` lives at `artifacts/experience/api-probes/gemm/` (built via the run scripts there); the upstream `.cu` is **not** vendored — it stays in the cutlass tree at the pinned commit.
- Custom epilogue probe: `artifacts/experience/api-probes/gemm/gemm_compare_relu.cu` + `relu_kernel.cu` (fused vs non-fused ReLU A/B).

This directory is the **distilled-knowledge view**. No buildable code lives here.

## When to use

Whenever the GEMM is followed (or preceded) by an elementwise / reduction operation that the cuBLASLt epilogue catalogue does **not** cover. Typical examples: custom quantize/dequantize scales, EVT topK+softmax, output-stride / swizzle absorption, mixed-dtype prologue dequant.

For epilogues that *are* in cuBLASLt's catalogue (`CUBLASLT_EPILOGUE_BIAS / RELU / GELU / DRELU_BGRAD / ...`) library-first is the default — see `wiki/nvidia/operator-routing/tensor-core/gemm/library-fallback.md`.

## Fusion pattern landscape

| Pattern | Upstream example | Mainloop | Epilogue | Status |
|---|---|---|---|---|
| Epilogue swizzle (output stride / swizzle absorbed) | `examples/50_hopper_gemm_with_epilogue_swizzle/` | warp-specialized cooperative | custom Sm90TmaWarpSpecializedAdapter + `LinearCombination<int32_t, ...>` | **Built + correctness-passed on H200** (`WGMMA GEMM with Epilogue Swizzle : Passed`) |
| Prologue dequant (int4 → bf16) | `examples/55_hopper_mixed_dtype_gemm/` | mixed-dtype warp-specialized | `LinearCombination` | Reference only; follow-up |
| Epilogue topK + softmax (EVT) | `examples/61_hopper_gemm_with_topk_and_softmax/` | warp-specialized cooperative | EVT (Epilogue Visitor Tree) | Reference only; follow-up |
| Activation epilogue (ReLU / GELU) | thread functor `cutlass::epilogue::thread::LinearCombinationRelu` | any sm_90 mainloop | `LinearCombinationRelu` thread functor | Probe at `artifacts/experience/api-probes/gemm/gemm_compare_relu.cu` |

## Key cutlass entry points

- `include/cutlass/epilogue/collective/collective_builder.hpp` — `CollectiveBuilder<...>` composes the epilogue from a thread functor + tile-handler.
- `include/cutlass/epilogue/thread/{linear_combination,linear_combination_relu,linear_combination_gelu,...}.h` — thread-level epilogue functors.
- `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` — EVT plumbing for Sm90.
- `include/cutlass/gemm/collective/collective_builder.hpp` (mixed-dtype path) — prologue dequant for int4 / int8 inputs.

## Cross-references

- Skill: `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/skill.md`
- Pitfalls: `wiki/nvidia/foundations/compute/gemm-fused/cutlass-epilogue-prologue/pitfalls.md`
- ReLU fusion probe: `sources/experience/api-probes/gemm.md`
- Library-first decision: `wiki/nvidia/operator-routing/tensor-core/gemm/library-fallback.md` §3 cuBLASLt
