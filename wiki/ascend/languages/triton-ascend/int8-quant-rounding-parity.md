---
id: skill-triton-ascend-int8-quant-rounding-parity
title: "Triton Ascend int8 Quantization Rounding Parity"
type: skill
vendor: ascend
tags:
- triton-ascend
- quantization
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: int8 dynamic quant rounding parity precision probe
architectures:
- ascend910b
languages:
- triton-ascend
kernel_types:
- quantization
aliases:
- int8 quant rounding
- dynamic quant parity
- per-token quant
- per-row quant
- DequantSwigluQuant
- divide by scale
- do not multiply reciprocal
- absmax divided by 127
- round half to even int8
- rint int8
- nearbyint int8
- half to even quant
- scalar vector division
- BLOCK_SIZE 1 scalar lowering
- vector division path
- int8 off by one
- golden -1 actual -2
---
# Triton Ascend int8 Quantization Rounding Parity

## Search Keywords

Simple search terms: int8 quant rounding, dynamic quant parity, per-token quant, per-row quant, DequantSwigluQuant, divide by scale, do not multiply reciprocal, absmax divided by 127, round half to even int8, rint, nearbyint, scalar vector division, BLOCK_SIZE 1 scalar lowering, vector division path, int8 off by one, golden -1 actual -2.

## 1. Problem

Per-row / per-token dynamic int8 quantization is highly sensitive at 1-ULP boundaries. The reference is usually:

```python
absmax = s.abs().amax(-1)
scale = absmax / 127
q = (s / scale).round().clamp(-128, 127)
```

If a Triton-Ascend kernel rewrites this as:

```python
q = (s * (127 / absmax)).round()
```

the math is equivalent, but the fp32 rounding path is different. Values near a `.5` quantization boundary can flip from `-1` to `-2`, producing int8 off-by-one mismatches.

## 2. Rule 1: divide by scale; do not multiply by reciprocal

Correct order:

```python
scale = absmax / 127.0
qf = s / scale
q = rint(qf)
q = clamp(q, -128, 127)
```

Do not rewrite it as:

```python
qf = s * (127.0 / absmax)
```

`x / (absmax / 127)` and `x * (127 / absmax)` can differ by around 1 ULP in fp32, enough to flip a `.5` rounding boundary.

## 3. Rule 2: use half-to-even rounding

PyTorch `torch.round` is round-half-to-even. In Triton-Ascend, use a half-to-even `rint` / `nearbyint` path. Avoid:

```python
floor(x + 0.5)       # half-up
trunc/away-from-zero # half-away
```

After rounding, clamp:

```python
q = clamp(q, -128, 127)
```

If `absmax == 0`, force the entire row to zero to avoid division-by-zero artifacts.

## 4. Rule 3: avoid scalar division codegen

Even with the correct `s / scale` order, Triton-Ascend may generate different code for scalar and vector float division, causing 1–2 ULP differences. A common sensitive step is:

```python
scale = max_abs / 127.0
```

If this lowers to the scalar path, the later `round(x / scale)` can flip an int8 boundary.

Recommendations:

- Compute scale division on a vector path with `tl.arange(0, BLOCK_SIZE)` and `BLOCK_SIZE >= 2`.
- Avoid `BLOCK_SIZE == 1`; the compiler can re-scalarize it.
- If scale is one value per row, a separate `compute_scale_kernel` over the row-max vector is often clean.
- Vectorize the entire quantization chain, including `max/127` and `x/scale + round`.

## 5. Evaluation gate matters

Different verifiers lead to different conclusions:

- Mean-error gate: a few off-by-one values are averaged out; the scale-division fix may be sufficient.
- Strict per-element gate: any int8 off-by-one fails, so the whole quantization chain must match as closely as possible.

Check the metric before promising a complete pass.

## 6. Checklist

1. If int8 passes almost all cases but has small off-by-one failures, check division order first.
2. Write `scale = absmax / 127`, then `s / scale`.
3. Use half-to-even `rint` / `nearbyint`.
4. Avoid `floor(x + 0.5)` and half-away-from-zero.
5. Use `BLOCK_SIZE >= 2` to avoid scalar lowering.
6. Vectorize both scale division and `x / scale + round`.
