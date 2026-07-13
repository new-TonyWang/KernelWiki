---
id: skill-triton-ascend-lowprecision-rounding-parity
title: "Triton Ascend Low-Precision Rounding Parity Overview"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: lowprecision rounding parity overview
architectures:
- ascend910b
languages:
- triton-ascend
kernel_types:
- elementwise
- reduction
- attention
- quantization
aliases:
- low precision rounding parity
- torch_npu precision mismatch
- Triton-Ascend precision mismatch
- bf16 fp16 parity
- rounding boundary mismatch
- low dtype materialization
- per op materialization
- allclose fail low precision
---
# Triton Ascend Low-Precision Rounding Parity Overview

## Search Keywords

Simple search terms: low precision rounding parity, torch_npu precision mismatch, Triton-Ascend precision mismatch, bf16 fp16 parity, rounding boundary mismatch, low dtype materialization, per-op materialization, allclose fail low precision.

## 1. Triage

When the reference is torch_npu and the candidate is a fused Triton-Ascend kernel, low-precision failures often come from missing reference materialization boundaries rather than from an incorrect algorithm.

Route by dtype and symptom:

| Symptom | First page to check |
|---|---|
| bf16 finite allclose fails while fp32 passes | [low-precision materialization pitfall](../../../generic/pitfalls/triton-low-precision-materialization.md) |
| fp16 finite values or intrinsic behavior differ | [fp16 native chain parity](fp16-native-chain-parity.md) |
| fp16 NaN/Inf mask differs | [fp16 overflow semantics](fp16-overflow-semantics.md) |
| Scalar parameters participate in low-precision expressions | pre-round scalar parameters explicitly |
| int8 dynamic quantization has off-by-one values | [int8 quant rounding parity](int8-quant-rounding-parity.md) |

## 2. Core principles

- Do not blindly promote every intermediate to fp32; the evaluator wants parity with the reference, not maximum mathematical accuracy.
- bf16 and fp16 require different strategies: bf16 often needs fp32 op + explicit RTNE bit-round; fp16 often needs a native fp16 op chain.
- Scalars may have been materialized to a low dtype in the reference; pre-round them on the host when needed.
- int8 quantization `.5` boundaries are sensitive to division order, rounding mode, and scalar-vs-vector codegen.
- If the reference is an opaque fused vendor op, a per-op rounding recipe may not apply; first confirm that a decomposed op graph exists.

## 3. Related non-precision traps

- Duplicate scatter with undefined reference behavior is not a precision issue; see [scatter duplicate semantics](scatter-duplicate-semantics.md).
- Wrong N-D gather addressing is not a precision issue; see [N-D index addressing](nd-index-addressing.md).
- fp16 overflow belongs to NaN/Inf mask parity; see [fp16 overflow semantics](fp16-overflow-semantics.md).
