---
id: skill-triton-ascend-bf16-rounding-parity
title: "Triton Ascend bf16 Rounding Parity"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: bf16 lowprecision rounding parity precision probe
architectures:
- ascend910b
languages:
- triton-ascend
kernel_types:
- elementwise
- reduction
- attention
aliases:
- bf16 rounding parity
- bfloat16 rounding
- bf16 RTNE
- round-half-to-even
- round to nearest even
- bf16 bit round
- bf16 bitcast rounding
- round_bf16
- bf16 precision mismatch
- bf16 allclose fail
- Triton-Ascend bf16 cast
- torch_npu bf16
- per op rounding
- per-edge rounding
- materialize bf16
- fp_downcast_rounding
- TRITON_DEFAULT_FP_FUSION
---
# Triton Ascend bf16 Rounding Parity

## Search Keywords

Simple search terms: bf16 rounding parity, bfloat16 rounding, bf16 RTNE, round-half-to-even, round to nearest even, bf16 bit round, bf16 bitcast rounding, round_bf16, bf16 precision mismatch, bf16 allclose fail, per-op rounding, per-edge rounding, materialize bf16, fp_downcast_rounding, TRITON_DEFAULT_FP_FUSION.

## 1. Problem

When the reference is a torch_npu decomposed op graph and the candidate is a fused Triton-Ascend kernel, bf16 can fail because it is “more accurate” than the reference:

- torch_npu usually executes op by op, materializing each op output to bf16.
- Triton-Ascend fused kernels often keep intermediates in fp32 registers.
- Plain `.to(tl.bfloat16)` may be optimized away, or its rounding mode may differ from torch_npu.
- Adjacent `mul + add` may be fused into FMA, changing the rounding boundary.

The goal is not to improve numerical accuracy. The goal is to reproduce the reference’s **per-op bf16 round-half-to-even** boundaries.

## 2. Scope

Use this page when:

- fp32 passes but bf16 fails against torch_npu;
- the op is elementwise, normalization, attention, or reduction with decomposed `exp`, `tanh`, `rsqrt`, mul/add, or reductions;
- the mismatch is finite allclose, not a NaN/Inf mask mismatch.

Do not use this as a universal fix for closed fused vendor ops; those may require vendor-op reproduction or reverse engineering.

## 3. Core technique: explicit bit-round at every materialized edge

```python
@triton.jit
def round_bf16(v):  # f32 in -> bf16-precision value, kept as f32
    bits = v.to(tl.uint32, bitcast=True)
    is_special = (bits & 0x7F800000) == 0x7F800000
    r = bits + 0x7FFF + ((bits >> 16) & 1)  # round-half-to-even
    r = (r >> 16) << 16
    return tl.where(is_special, bits, r).to(tl.float32, bitcast=True)
```

The helper rounds a fp32 value to bf16 precision while keeping the storage type as fp32. It avoids store/reload and is robust against floating-point fusion.

Example:

```python
z  = round_bf16(input_scale * x)
e  = round_bf16(tl.exp(z))
m1 = round_bf16(e - 1.0)
out = round_bf16(alpha_scale * m1)
```

Apply it at every bf16 edge that the reference materializes, not only at the final output.

## 4. Why not a plain cast

| Method | Limitation |
|---|---|
| `.to(tl.bfloat16)` | May be eliminated in a fused kernel; even if kept, backend downcast may not match torch_npu RTNE |
| store + reload | Forces one materialization but costs a DRAM round trip and fixes only one edge |
| `TRITON_DEFAULT_FP_FUSION=0` | Disables FMA but does not guarantee bf16 cast rounding parity |
| `tl.cast(..., fp_downcast_rounding="rtne")` | Semantically relevant, but backend support must be verified; bit-round is the portable fallback |

Integer bitcast arithmetic cannot be folded into a floating-point FMA, so it is a reliable rounding barrier.

## 5. Checklist

1. fp32 passes, bf16 fails, and NaN/Inf masks match.
2. The reference is a decomposed torch_npu op graph, not an opaque fused vendor op.
3. Identify every reference edge that materializes to bf16.
4. Insert `round_bf16` at each corresponding Triton edge.
5. Pre-round scalar parameters as needed; pre-round scalar parameters explicitly when they participate in low-precision expressions.
6. Do not use this bf16 bit-round recipe for fp16; see [fp16 native chain parity](fp16-native-chain-parity.md).
