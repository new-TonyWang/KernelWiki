---
id: skill-triton-ascend-lowprecision-scalar-rounding
title: "Triton Ascend Low-Precision Scalar Rounding"
type: skill
vendor: ascend
tags:
- triton-ascend
evidence_level: measured
applies_to:
- ascend910b
source:
- path: local
  anchor: scalar lowprecision rounding parity
architectures:
- ascend910b
languages:
- triton-ascend
kernel_types:
- elementwise
- reduction
- attention
aliases:
- scalar rounding
- bf16 scalar rounding
- fp16 scalar rounding
- host scalar pre-round
- alpha scale rounding
- alpha_scale
- scalar params low dtype
- torch.as_tensor scalar dtype
- bf16 host round
- scalar precision mismatch
---
# Triton Ascend Low-Precision Scalar Rounding

## Search Keywords

Simple search terms: scalar rounding, bf16 scalar rounding, fp16 scalar rounding, host scalar pre-round, alpha scale rounding, alpha_scale, scalar params low dtype, torch.as_tensor scalar dtype, bf16 host round, scalar precision mismatch.

## 1. Problem

Low-precision parity applies to scalar parameters too, not only tensor intermediates. A torch_npu reference often casts scalar parameters to the tensor dtype before using them:

```python
alpha_t = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
scale_t = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
alpha_scale = alpha_t * scale_t
```

If the Triton-Ascend host code passes a Python float or fp32 scalar directly into the kernel, the kernel uses an unrounded fp32 scalar and its multiplications no longer match the reference bf16/fp16 scalar boundary.

## 2. Rules

- If the reference first materializes a scalar to bf16/fp16, pre-round it on the host to the same dtype.
- Scalar products must follow the reference order; `alpha * scale` is not necessarily one fp32 multiply.
- bf16 uses round-half-to-even; fp16 uses fp16 cast semantics.
- Pass pre-rounded scalar values or scalar products into the Triton kernel.

## 3. bf16 host-side RTNE example

```python
import struct

def round_bf16_host(x):
    b = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (b & 0x7F800000) == 0x7F800000:
        return float(x)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]

alpha_b = round_bf16_host(alpha)
scale_b = round_bf16_host(scale)
alpha_scale_b = round_bf16_host(alpha_b * scale_b)
```

Then pass the rounded value:

```python
kernel[grid](x, out, alpha_scale_b, ...)
```

## 4. Common mistakes

| Mistake | Consequence |
|---|---|
| Passing a Python float directly | Kernel uses fp32 scalar while the reference used low dtype |
| Rounding individual scalars but not scalar products | The `alpha*scale` boundary still differs |
| Mixing fp32 scalar with bf16 tensor inside the kernel | Multiplication input dtypes differ from the reference |
| Ignoring special values | NaN/Inf scalar bit patterns may be changed incorrectly |

## 5. Related pages

- Tensor bf16 edges: [bf16 rounding parity](bf16-rounding-parity.md).
- fp16 op chains: [fp16 native chain parity](fp16-native-chain-parity.md).
