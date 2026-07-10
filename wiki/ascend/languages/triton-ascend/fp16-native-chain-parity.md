---
id: skill-triton-ascend-fp16-native-chain-parity
title: "Triton Ascend fp16 Native Chain Parity"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: fp16 native lowprecision rounding parity precision probe
architectures:
- ascend910b
languages:
- triton-ascend
kernel_types:
- elementwise
- reduction
- attention
aliases:
- fp16 native chain
- float16 native chain
- half native math
- fp16 transcendental
- fp16 exp
- fp16 tanh
- fp16 rsqrt
- do not upcast fp16
- fp16 per op materialization
- fp16 barrier invariant
- fp16 NaN Inf parity
- torch_npu fp16 native
---
# Triton Ascend fp16 Native Chain Parity

## Search Keywords

Simple search terms: fp16 native chain, float16 native chain, half native math, fp16 transcendental, fp16 exp, fp16 tanh, fp16 rsqrt, do not upcast fp16, fp16 per-op materialization, fp16 barrier invariant, fp16 NaN Inf parity, torch_npu fp16 native.

## 1. Key rule

Do not copy the bf16 recipe of “fp32 intrinsic + bit-round” to fp16. torch_npu fp16 transcendental behavior is closer to a **native fp16 op chain**, so Triton-Ascend should keep the computation chain in fp16:

```python
xh = x.to(tl.float16)
z  = (input_scale_h * xh).to(tl.float16)
e  = tl.exp(z).to(tl.float16)
m1 = (e - 1.0).to(tl.float16)
out = (alpha_scale_h * m1).to(tl.float16)
```

For fp16, upcasting to fp32 and rounding back can diverge from torch_npu fp16 intrinsics by a few ULPs, enough to cross a `1e-2` gate.

## 2. Difference from bf16

| dtype | Empirical reference model | Triton-Ascend parity strategy |
|---|---|---|
| bf16 | fp32 intrinsic followed by bf16 rounding | fp32 op + explicit bf16 RTNE bit-round at each edge |
| fp16 | native fp16 intrinsic / op chain | keep the chain native fp16; do not upcast first |

Match the reference’s **compute precision**, not just its rounding mode.

## 3. NaN/Inf also depend on fp16 materialization

The native fp16 chain affects not only finite values but also NaN/Inf masks. For example, `x ** -3` or `1 / tiny` may overflow to `inf` in fp16 and propagate to `NaN` downstream. If the Triton kernel computes the same intermediate in fp32, it may remain finite and fail the verifier’s NaN/Inf mask.

If Triton-Ascend saturates fp16 overflow to `65504` instead of producing `inf`, see [fp16 overflow semantics](fp16-overflow-semantics.md).

## 4. When to use

Use this rule when:

- fp16 fails against torch_npu while fp32 passes;
- the expression contains `exp`, `tanh`, `rsqrt`, pow, reciprocal, or similar operations;
- the reference is a decomposed op graph with per-op materialization;
- NaN/Inf positions must match, not only finite allclose values.

## 5. Checklist

1. Do not assume fp32 intermediates are better for verification; the evaluator wants reference parity.
2. Run fp16 transcendentals on fp16 operands when the reference does so.
3. Materialize each reference fp16 boundary with `.to(tl.float16)`.
4. If NaN/Inf masks differ, combine this with [fp16 overflow semantics](fp16-overflow-semantics.md).
5. For bf16, use the generic [low-precision materialization pitfall](../../../generic/pitfalls/triton-low-precision-materialization.md) guidance instead.
