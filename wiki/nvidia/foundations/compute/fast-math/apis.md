---
func_name: __expf
namespace: math
header: math_functions.h
signature: float __expf(float x)
status: documented
has_end_to_end_example: true
source:
- path: spec
  anchor: Reference
id: api-fast-math-ref
type: api-definition
vendor: nvidia
title: Apis
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L26993-L26995
languages:
- ptx
- cuda-cpp
techniques:
- kernel-fusion
- software-exp
kernel_types:
- fused-kernel
confidence: inferred
tags:
- kernel-fusion
- software-exp
- fused-kernel
- ptx
- cuda-cpp
---
# Fast Math Intrinsics API Reference

This file lists the APIs touched by the fast-math skill. Each entry records the function name, namespace, signature, semantics, ULP error bound, and upstream source reference.

## Approximate Transcendental Intrinsics (Single-Precision Only)

### `__expf`

- **Namespace**: math (CUDA intrinsic)
- **Header**: `math_functions.h` (included via `cuda_runtime.h`)
- **Signature**: `float __expf(float x)`
- **PTX**: compiles to `ex2.approx.f32` with range-reduction scaling
- **Semantics**: Computes `e^x` using the SFU approximate exponential.
- **Max ULP error**: `2 + floor(abs(1.173 * x))`
- **Source**: Programming guide Table 58, L26993-L26995
- **Probed**: expf-vs-fast-expf

### `__exp10f`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __exp10f(float x)`
- **Semantics**: Computes `10^x`.
- **Max ULP error**: `2 + floor(abs(2.97 * x))`
- **Source**: Programming guide Table 58, L26997-L26999

### `__logf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __logf(float x)`
- **PTX**: compiles to `lg2.approx.f32` with base-conversion scaling
- **Semantics**: Computes `ln(x)`.
- **Max ULP error**: `2^-21.41` absolute error for `x in [0.5, 2]`; 3 ULP otherwise
- **Source**: Programming guide Table 58, L27005-L27007

### `__log2f`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __log2f(float x)`
- **PTX**: `lg2.approx.f32`
- **Semantics**: Computes `log2(x)`.
- **Max ULP error**: `2^-22` absolute error for `x in [0.5, 2]`; 2 ULP otherwise
- **Source**: Programming guide Table 58, L27009-L27011

### `__log10f`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __log10f(float x)`
- **Semantics**: Computes `log10(x)`.
- **Max ULP error**: `2^-24` absolute error for `x in [0.5, 2]`; 3 ULP otherwise
- **Source**: Programming guide Table 58, L27013-L27015

### `__sinf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __sinf(float x)`
- **PTX**: `sin.approx.f32`
- **Semantics**: Computes `sin(x)`.
- **Max ULP error**: `2^-21.41` absolute error for `x in [-pi, pi]`; larger otherwise
- **Source**: Programming guide Table 58, L27017-L27019

### `__cosf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __cosf(float x)`
- **PTX**: `cos.approx.f32`
- **Semantics**: Computes `cos(x)`.
- **Max ULP error**: `2^-21.41` absolute error for `x in [-pi, pi]`; larger otherwise
- **Source**: Programming guide Table 58, L27021-L27023

### `__sincosf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `void __sincosf(float x, float* sptr, float* cptr)`
- **Semantics**: Computes `sin(x)` and `cos(x)` simultaneously.
- **Max ULP error**: Component-wise, same as `__sinf` and `__cosf`.
- **Source**: Programming guide Table 58, L27025-L27027

### `__tanf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __tanf(float x)`
- **Semantics**: Computes `tan(x)`.
- **Max ULP error**: Derived from `__sinf(x) * (1 / __cosf(x))`
- **Source**: Programming guide Table 58, L27029-L27031

### `__tanhf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __tanhf(float x)`
- **Semantics**: Computes `tanh(x)`.
- **Max ULP error**: Max relative error `2^-11`. Subnormal results are not flushed even under `-ftz=true`.
- **Source**: Programming guide Table 58, L27033-L27035

### `__powf`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __powf(float x, float y)`
- **Semantics**: Computes `x^y`.
- **Max ULP error**: Derived from `exp2f(y * __log2f(x))` -- compounds errors.
- **Source**: Programming guide Table 58, L27001-L27003

### `__fdividef`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __fdividef(float x, float y)`
- **Semantics**: Computes `x / y`.
- **Max ULP error**: 2 ULP for `|y| in [2^-126, 2^126]`. Returns 0 for `|y| > 2^126`.
- **Source**: Programming guide Table 58, L26985-L26988

## Rounded Arithmetic Intrinsics (IEEE-754 Exact, 0 ULP)

### `__fadd_rn`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __fadd_rn(float x, float y)`
- **Semantics**: `x + y` with round-to-nearest-even. Prevents FFMA contraction.
- **Max ULP error**: 0 (IEEE-754 compliant)
- **Source**: Programming guide Table 57, L26937-L26938

### `__fmul_rn`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __fmul_rn(float x, float y)`
- **Semantics**: `x * y` with round-to-nearest-even. Prevents FFMA contraction.
- **Max ULP error**: 0
- **Source**: Programming guide Table 57, L26945-L26947

### `__fmaf_rn`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __fmaf_rn(float x, float y, float z)`
- **Semantics**: `x * y + z` as a single fused multiply-add. Round-to-nearest-even.
- **Max ULP error**: 0
- **Source**: Programming guide Table 57, L26949-L26951

### `__frcp_rn`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __frcp_rn(float x)`
- **Semantics**: `1 / x` with round-to-nearest-even.
- **Max ULP error**: 0
- **Source**: Programming guide Table 57, L26957-L26959

### `__fsqrt_rn`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __fsqrt_rn(float x)`
- **Semantics**: `sqrt(x)` with round-to-nearest-even.
- **Max ULP error**: 0
- **Source**: Programming guide Table 57, L26961-L26963

### `__frsqrt_rn`

- **Namespace**: math
- **Header**: `math_functions.h`
- **Signature**: `float __frsqrt_rn(float x)`
- **Semantics**: `1 / sqrt(x)` with round-to-nearest-even.
- **Max ULP error**: 0
- **Source**: Programming guide Table 58, L26989-L26991

### `rsqrtf`

- **Namespace**: math (CUDA Math API, not prefixed with `__`)
- **Header**: `math_functions.h`
- **Signature**: `float rsqrtf(float x)`
- **Semantics**: Fast reciprocal square root. Maps to SFU `rsqrt.approx.f32`.
- **Max ULP error**: 2 ULP
- **Source**: Programming guide standard functions table, L26781; Best-practices guide L1372
- **Note**: Recommended to invoke directly rather than writing `1.0f / sqrtf(x)`, which the compiler only optimizes to rsqrt under specific flags.
