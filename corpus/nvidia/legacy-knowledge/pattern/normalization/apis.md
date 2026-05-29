# Normalization -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__expf(x)` | Math Intrinsics | Fast variant of expf(x) |
| `__frsqrt_rn(x)` | Math Intrinsics | Fast variant of rsqrtf(x) |
| `__fsqrt_rn(x)` | Math Intrinsics | Fast variant of sqrtf(x) |
| `__hfma(a,b,c)` | Math Intrinsics | Fused multiply-add a*b+c |
| `__hfma2(a,b,c)` | Math Intrinsics | Packed fused multiply-add |
| `__logf(x)` | Math Intrinsics | Fast variant of logf(x) |
| `erff(x)` | Math Intrinsics | Error function erf(x) |
| `expf(x)` | Math Intrinsics | Base-e exponential e^x |
| `fmaf(x,y,z)` | Math Intrinsics | Fused multiply-add x*y+z (standard) |
| `h2log(__half2)` | Math Intrinsics | Fast variant of hlog(__half) |
| `h2log(__nv_bfloat162)` | Math Intrinsics | Fast variant of hlog(__nv_bfloat16) |
| `h2rsqrt(__half2)` | Math Intrinsics | Fast variant of hrsqrt(__half) |
| `h2rsqrt(__nv_bfloat162)` | Math Intrinsics | Fast variant of hrsqrt(__nv_bfloat16) |
| `h2sqrt(__half2)` | Math Intrinsics | Fast variant of hsqrt(__half) |
| `h2sqrt(__nv_bfloat162)` | Math Intrinsics | Fast variant of hsqrt(__nv_bfloat16) |
| `hlog(__half)` | Math Intrinsics | Natural logarithm |
| `hlog(__nv_bfloat16)` | Math Intrinsics | Natural logarithm |
| `hrsqrt(__half)` | Math Intrinsics | Reciprocal square root |
| `hrsqrt(__nv_bfloat16)` | Math Intrinsics | Reciprocal square root |
| `hsqrt(__half)` | Math Intrinsics | Square root |
| `hsqrt(__nv_bfloat16)` | Math Intrinsics | Square root |
| `logf(x)` | Math Intrinsics | Natural logarithm ln(x) |
| `normcdff(x)` | Math Intrinsics | Normal CDF (used in GELU) |
| `rsqrtf(x)` | Math Intrinsics | Reciprocal square root 1/sqrt(x) |
| `sqrtf(x)` | Math Intrinsics | Square root |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__dsqrt_rn(x)` | Math Intrinsics | Fast variant of sqrt(x) |
| `exp(x)` | Math Intrinsics | Base-e exponential |
| `log(x)` | Math Intrinsics | Natural logarithm |
| `rsqrt(x)` | Math Intrinsics | Reciprocal square root |
| `sqrt(x)` | Math Intrinsics | Square root |
