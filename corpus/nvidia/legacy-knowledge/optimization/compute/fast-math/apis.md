# Fast Math -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__cosf(x)` | Math Intrinsics | Fast variant of cosf(x) |
| `__expf(x)` | Math Intrinsics | Fast variant of expf(x) |
| `__fdividef(x,y)` | Math Intrinsics | Fast variant of fdividef(x,y) |
| `__frsqrt_rn(x)` | Math Intrinsics | Fast variant of rsqrtf(x) |
| `__log2f(x)` | Math Intrinsics | Fast variant of log2f(x) |
| `__logf(x)` | Math Intrinsics | Fast variant of logf(x) |
| `__powf(x,y)` | Math Intrinsics | Fast variant of powf(x,y) |
| `__sincosf(x,s,c)` | Math Intrinsics | Fast variant of sincosf(x,s,c) |
| `__sinf(x)` | Math Intrinsics | Fast variant of sinf(x) |
| `__tanhf(x)` | Math Intrinsics | Fast variant of tanhf(x) |
| `cosf(x)` | Math Intrinsics | Cosine |
| `expf(x)` | Math Intrinsics | Base-e exponential e^x |
| `fdividef(x,y)` | Math Intrinsics | Fast approximate divide |
| `log2f(x)` | Math Intrinsics | Base-2 logarithm |
| `logf(x)` | Math Intrinsics | Natural logarithm ln(x) |
| `powf(x,y)` | Math Intrinsics | x raised to power y |
| `rsqrtf(x)` | Math Intrinsics | Reciprocal square root 1/sqrt(x) |
| `sincosf(x,s,c)` | Math Intrinsics | Sine and cosine together |
| `sinf(x)` | Math Intrinsics | Sine |
| `tanhf(x)` | Math Intrinsics | Hyperbolic tangent |
| `cos.approx[.ftz].f32` | PTX ISA | Fast approximate cosine |
| `cvt[.rnd][.sat].dtype.stype` | PTX ISA | Type conversion (int/float, narrowing/widening, rounding modes) |
| `div.approx[.ftz].f32` | PTX ISA | Fast approximate FP32 divide |
| `ex2.approx[.ftz].f32` | PTX ISA | Fast approximate 2^x |
| `fma.rnd.f64` | PTX ISA | FP64 fused multiply-add |
| `fma.rnd[.ftz][.sat].f32` | PTX ISA | FP32 fused multiply-add; .f32x2 variant (sm_100+, **Blackwell**) |
| `lg2.approx[.ftz].f32` | PTX ISA | Fast approximate log2 |
| `rcp.approx[.ftz].f32` | PTX ISA | Fast reciprocal (1/x), max 1 ulp error |
| `rsqrt.approx[.ftz].f32` | PTX ISA | Fast reciprocal square root (1/sqrt(x)) |
| `sin.approx[.ftz].f32` | PTX ISA | Fast approximate sine |
| `sqrt.approx[.ftz].f32` | PTX ISA | Fast approximate square root |
| `tanh.approx.f32` | PTX ISA | Fast approximate tanh (activation function) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__exp10f(x)` | Math Intrinsics | Fast variant of exp10f(x) |
| `__log10f(x)` | Math Intrinsics | Fast variant of log10f(x) |
| `__saturatef(x)` | Math Intrinsics | Clamp to [0.0, 1.0] |
| `__tanf(x)` | Math Intrinsics | Fast variant of tanf(x) |
| `exp10f(x)` | Math Intrinsics | Base-10 exponential 10^x |
| `exp2f(x)` | Math Intrinsics | Base-2 exponential 2^x |
| `log10f(x)` | Math Intrinsics | Base-10 logarithm |
| `tanf(x)` | Math Intrinsics | Tangent |
| `add.rnd.f64` | PTX ISA | FP64 add |
| `add.rnd[.ftz][.sat].f32` | PTX ISA | FP32 add; .f32x2 SIMD variant (sm_100+, **Blackwell**) |
| `div.full[.ftz].f32` | PTX ISA | Full-range approximate FP32 divide |
| `div.rnd.f64` | PTX ISA | IEEE-754 compliant FP64 divide |
| `div.rnd[.ftz].f32` | PTX ISA | IEEE-754 compliant FP32 divide |
| `div[.type]` | PTX ISA | Integer divide (slow) |
| `max[.ftz][.NaN].f32` | PTX ISA | FP32 max; .xorsign.abs variants (sm_86+); 3-input (sm_100+, **Blackwell**) |
| `min[.ftz][.NaN].f32` | PTX ISA | FP32 min; .xorsign.abs variants (sm_86+); 3-input (sm_100+, **Blackwell**) |
| `mul.rnd.f64` | PTX ISA | FP64 multiply |
| `mul.rnd[.ftz][.sat].f32` | PTX ISA | FP32 multiply; .f32x2 SIMD variant (sm_100+, **Blackwell**) |
| `rcp.approx.ftz.f64` | PTX ISA | Fast approximate f64 reciprocal |
| `rcp.rnd[.ftz].f32` | PTX ISA | IEEE-compliant reciprocal |
| `rsqrt.approx.f64` | PTX ISA | Approximate f64 reciprocal square root (emulated, slow) |
| `sqrt.rnd[.ftz].f32` | PTX ISA | IEEE-compliant square root |
| `sub.rnd.f64` | PTX ISA | FP64 subtract |
| `sub.rnd[.ftz][.sat].f32` | PTX ISA | FP32 subtract |
