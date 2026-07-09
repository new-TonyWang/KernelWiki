---
id: skill-triton-ascend-fp16-overflow-semantics
title: "Triton Ascend fp16 Overflow Semantics"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: fp16-overflow-semantics precision probe
architectures:
- ascend910b
languages:
- triton-ascend
kernel_types:
- elementwise
- reduction
- attention
aliases:
- fp16 overflow
- half overflow
- float16 overflow
- NaN mask mismatch
- Inf mask mismatch
- NaN/Inf mask
- saturate to 65504
- clamp to 65504
- fp16 saturate
- overflow to inf
- 65504
- 65520
- std**-3
- reciprocal overflow
- pow negative overflow
- torch_npu fp16
- Triton-Ascend fp16
- materialize fp16 overflow
- Inf sign mismatch
---
# Triton Ascend fp16 Overflow Semantics

## Search Keywords

Simple search terms: fp16 overflow, half overflow, float16 overflow, NaN mask mismatch, Inf mask mismatch, NaN/Inf mask, saturate to 65504, clamp to 65504, overflow to inf, 65504, 65520, reciprocal overflow, pow negative overflow, std**-3, materialize fp16 overflow, torch_npu fp16, Triton-Ascend fp16.

## 1. Problem

torch_npu fp16 ops usually follow fp16 overflow semantics: when the true result magnitude exceeds the largest finite fp16 value (`65504`), the result becomes `+inf` or `-inf`. These infinities then propagate into downstream `NaN`s through expressions such as `inf * 0`, `inf - inf`, or `inf + (-inf)`.

A common Triton-Ascend path differs: fp16 arithmetic may be kept in fp32 registers, and the final `fp32 -> fp16` cast can **saturate** to `65504` instead of producing `inf`. Finite allclose may therefore look fine, but the verifier fails first because NaN/Inf positions or Inf signs do not match.

## 2. When to use this page

Check this rule first when:

- an fp16 kernel against a torch_npu reference reports `NaN mask mismatch` or `Inf mask mismatch`;
- the reference contains division by a tiny value or a negative power, such as `1/x`, `x**-1`, `std**-3`, or `rsqrt(tiny)`;
- fp32/bf16 paths pass, but fp16 NaN/Inf positions differ;
- the mismatch appears in normalization backward, attention scaling, or elementwise reciprocal/pow code.

If only finite values exceed the tolerance and NaN/Inf masks match, look at low-precision rounding parity instead.

## 3. Core technique: overflow-aware materialization

When a fp32 intermediate must emulate torch_npu fp16 materialization, do not rely only on `.to(tl.float16)`. Explicitly map values that would round to fp16 infinity to signed `inf`; otherwise use the normal fp16 round:

```python
@triton.jit
def materialize_fp16_overflow(v):  # v: fp32 value, returns fp16-semantics value kept as fp32
    zero = tl.full((), 0.0, tl.float32)
    pinf = 1.0 / zero
    ninf = -1.0 / zero
    BIG = 65520.0  # fp16 round-to-inf boundary; max finite fp16 is 65504
    return tl.where(v >= BIG, pinf,
           tl.where(v <= -BIG, ninf,
                    v.to(tl.float16).to(tl.float32)))
```

Example:

```python
# The reference materializes this reciprocal-cube boundary in fp16.
cube = materialize_fp16_overflow(std * std * std)
inv_cube = materialize_fp16_overflow(1.0 / cube)
```

Notes:

- `65520.0` is the fp16 round-to-nearest-even overflow boundary above max finite `65504`.
- `1.0 / 0.0` is a reliable way to construct `+inf` on Ascend; the negative case is analogous.
- Inf sign must match the true fp32 value because the verifier checks Inf signs.
- bf16 usually does not need this overflow patch because its exponent range is close to fp32; bf16 mainly needs rounding parity.

## 4. This is not a universal patch

For graphs with multiple overflow sites, inserting this helper everywhere does not guarantee a better result. The NaN/Inf mask depends on:

- dtype at every intermediate node;
- op execution order;
- reduction/accumulation order;
- where combinations such as `inf + (-inf)` or `inf * 0` occur.

If the Triton kernel structure differs substantially from the reference, for example a two-pass reduction or a long fp32 fused chain replacing a per-op fp16 graph, NaN/Inf positions may still differ. Exact matching requires mirroring the torch_npu fp16 op graph node-by-node and materializing the same boundaries.

## 5. Diagnostic flow

1. If the failure is a NaN/Inf mask mismatch, use this flow. If it is finite allclose only, use rounding parity first.
2. Locate overflow sources: `1/x`, negative `pow`, `rsqrt`, `std**-3`, softmax/exp, and similar expressions.
3. Add `materialize_fp16_overflow` at the most suspicious boundary first and check whether ref/impl NaN/Inf counts move closer.
4. If multiple boundaries are needed, extend in reference op-graph order; do not blindly patch every edge at once.
5. Apply only to fp16. bf16 and fp32 should use their own parity rules.

## 6. Common mistakes

| Mistake | Problem | Recommended fix |
|---|---|---|
| Direct `v.to(tl.float16)` | Triton-Ascend may saturate to `65504` and never produce `inf` | Use `materialize_fp16_overflow` at the required boundary |
| Compute everything in fp32 and cast once | Skips the reference fp16 overflow point | Materialize at the same fp16 op boundaries as the reference |
| Patch every possible site blindly | Multi-term NaN/Inf interactions may move to the wrong positions | Validate one site first, then extend in op-graph order |
| Apply the same logic to bf16 | bf16 normally has enough exponent range | Use bf16 round-half-to-even parity instead |

## 7. Related pages

- Finite low-precision mismatch: [low-precision rounding parity](lowprecision-rounding-parity.md).
- fp16 compute precision mismatch: [fp16 native chain parity](fp16-native-chain-parity.md).
- Math intrinsic behavior: check the libdevice usage page when the overflow source is `exp`, `tanh`, or `pow`.
