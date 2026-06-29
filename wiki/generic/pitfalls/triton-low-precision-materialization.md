---
id: pitfall-triton-low-precision-materialization
title: "Triton Low-Precision Intermediate Materialization Pitfall"
type: pitfall
vendor: generic
tags:
- triton
- triton-ascend
- quantization
architectures:
- sm90
- sm100
- ascend910b
languages:
- triton
- triton-ascend
confidence: source-reported
related:
- skill-ascend-autotune
sources: []
---
# Triton Low-Precision Intermediate Materialization Pitfall

## Symptom

A Triton kernel computes `tl.exp(x).to(bf16) - 1.0` (or similar cast-then-arithmetic chains) in the same SSA expression. The result differs from the PyTorch baseline `torch.exp(x) - 1` by a few ULP in bf16/fp16, causing precision failures when the error is amplified by subsequent scaling.

Typical failure pattern:
- Most test cases pass, but a handful of bf16/fp16 cases fail with `rtol=1e-2, atol=1e-2`
- fp32 cases always pass
- The error is small per-element but gets amplified by multiplication (e.g. `alpha * (exp(x) - 1)`)

## Root Cause

In a single Triton kernel SSA chain:

```python
exp_val = tl.exp(prod.to(tl.float32)).to(tl.bfloat16)
result = (exp_val - 1.0).to(tl.bfloat16)  # exp_val not materialized!
```

Although `.to(tl.bfloat16)` appears in code, the compiler may keep `exp_val` at higher precision in registers. The subsequent `- 1.0` operates on the un-materialized value, producing a different rounding path than PyTorch's bf16 tensor semantics where `torch.exp(x)` returns a bf16 tensor that is physically stored before the subtraction.

This is **not** a hardware bug — it affects both NVIDIA GPU (H200/CUDA) and Huawei Ascend NPU (910B) identically. It is a fundamental property of how Triton compilers optimize SSA chains.

## Verified Evidence

### On NPU (910B)

```text
# Without materialization: 34/39 passed, 5 bf16 cases failed
# With store+reload:       39/39 passed
```

### On GPU (H200)

```text
cast-fp32 tl.exp vs torch.exp:                              neq=0
cast-fp32 tl.exp(x)-1 same kernel vs torch.exp(x)-1:       neq=3179  max_abs=0.0078125
cast-fp32 stored tl.exp then torch sub vs torch.exp(x)-1:   neq=0
```

Both platforms show the same pattern: un-materialized low-precision intermediates diverge from PyTorch.

## Fix: Store + Reload to Force Materialization

Insert a `tl.store` / `tl.load` round-trip to force the intermediate value through memory at the target precision:

```python
# WRONG: exp_val stays in registers at higher precision
exp_val = tl.exp(prod.to(tl.float32)).to(tl.bfloat16)
result = (exp_val - 1.0).to(tl.bfloat16)

# CORRECT: store+reload forces bf16 rounding boundary
exp_val = tl.exp(prod.to(tl.float32)).to(tl.bfloat16)
tl.store(out_ptr + offs, exp_val, mask=mask)
exp_val = tl.load(out_ptr + offs, mask=mask)
result = (exp_val - 1.0).to(tl.bfloat16)
```

The store+reload creates a physical bf16 rounding boundary that matches PyTorch's tensor semantics.

## Additional Measure: Disable FP Fusion

Set `enable_fp_fusion=False` in the kernel launch to prevent the compiler from fusing floating-point operations across the materialization boundary:

```python
kernel[grid](x, out, ..., enable_fp_fusion=False)
```

This is a necessary but not sufficient fix — `enable_fp_fusion=False` alone does not eliminate the rounding divergence. Both measures together provide robust precision alignment.

## Platform-Specific Notes

### NVIDIA GPU (CUDA Triton)

`tl.exp` only accepts fp32/fp64 inputs. Direct `tl.exp(bf16_tensor)` causes a compile error:

```text
Expected dtype ['fp32', 'fp64'] but got bf16
```

Must explicitly upcast: `tl.exp(x.to(tl.float32))`.

### Ascend NPU (Triton Ascend)

`tl.exp` accepts bf16/fp16 inputs directly. However the materialization issue still applies — the store+reload fix is equally necessary.

## General Rule

**Whenever a Triton kernel performs arithmetic on a low-precision (bf16/fp16) intermediate that was cast from a higher-precision computation, and the result must match a PyTorch baseline, force materialization via store+reload before the next arithmetic operation.**

This applies to any pattern of the form:

```python
intermediate = high_prec_op(x).to(low_prec_dtype)
result = f(intermediate)  # intermediate not yet in memory
```

Common instances beyond `exp - 1`:
- `softmax(x).to(fp16)` followed by matmul
- `gelu(x).to(bf16)` followed by scaling
- Any `cast → subtract/add/multiply` chain where the cast result feeds directly into the next op
