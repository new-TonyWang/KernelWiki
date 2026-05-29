# Math Intrinsics Index

Source: CUDA Math API Reference Manual 13.2
Header files: `<math.h>` (single/double), `cuda_fp16.h` (half), `cuda_bf16.h` (bfloat16)

> **Reading guide**: "Fast Variant" column shows the `__`-prefixed intrinsic that trades precision
> for speed. Functions marked with `use_fast_math` are automatically replaced by the compiler
> when `-use_fast_math` is passed. The "Knowledge Node" column maps each function to the
> kernel-knowledge tree path under `optimization/` or `pattern/`.

---

## 1. Exponential and Logarithmic Functions

Critical for softmax, normalization, and activation functions in ML kernels.

### 1.1 Single Precision (float)

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `expf(x)` | `__expf(x)` | Base-e exponential e^x | single | optimization/compute/fast-math, pattern/normalization, pattern/attention | core |
| `exp2f(x)` | -- | Base-2 exponential 2^x | single | optimization/compute/fast-math | related |
| `exp10f(x)` | `__exp10f(x)` | Base-10 exponential 10^x | single | optimization/compute/fast-math | related |
| `expm1f(x)` | -- | e^x - 1 (accurate near zero) | single | pattern/elementwise | low-relevance |
| `logf(x)` | `__logf(x)` | Natural logarithm ln(x) | single | optimization/compute/fast-math, pattern/normalization | core |
| `log2f(x)` | `__log2f(x)` | Base-2 logarithm | single | optimization/compute/fast-math | core |
| `log10f(x)` | `__log10f(x)` | Base-10 logarithm | single | optimization/compute/fast-math | related |
| `log1pf(x)` | -- | ln(1+x) (accurate near zero) | single | pattern/elementwise | low-relevance |
| `powf(x,y)` | `__powf(x,y)` | x raised to power y | single | optimization/compute/fast-math, pattern/elementwise | core |

### 1.2 Double Precision (double)

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `exp(x)` | -- | Base-e exponential | double | pattern/normalization | related |
| `exp2(x)` | -- | Base-2 exponential | double | pattern/elementwise | low-relevance |
| `exp10(x)` | -- | Base-10 exponential | double | pattern/elementwise | low-relevance |
| `expm1(x)` | -- | e^x - 1 | double | pattern/elementwise | low-relevance |
| `log(x)` | -- | Natural logarithm | double | pattern/normalization | related |
| `log2(x)` | -- | Base-2 logarithm | double | pattern/elementwise | low-relevance |
| `log10(x)` | -- | Base-10 logarithm | double | pattern/elementwise | low-relevance |
| `log1p(x)` | -- | ln(1+x) | double | pattern/elementwise | low-relevance |
| `pow(x,y)` | -- | x raised to power y | double | pattern/elementwise | related |

### 1.3 Half Precision (__half / half2)

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hexp(__half)` | `h2exp(__half2)` | Base-e exponential | half | optimization/compute/half-precision-math, pattern/attention | core |
| `hexp2(__half)` | `h2exp2(__half2)` | Base-2 exponential | half | optimization/compute/half-precision-math | related |
| `hexp10(__half)` | `h2exp10(__half2)` | Base-10 exponential | half | optimization/compute/half-precision-math | related |
| `hlog(__half)` | `h2log(__half2)` | Natural logarithm | half | optimization/compute/half-precision-math, pattern/normalization | core |
| `hlog2(__half)` | `h2log2(__half2)` | Base-2 logarithm | half | optimization/compute/half-precision-math | related |
| `hlog10(__half)` | `h2log10(__half2)` | Base-10 logarithm | half | optimization/compute/half-precision-math | related |

### 1.4 BFloat16 (__nv_bfloat16 / __nv_bfloat162)

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hexp(__nv_bfloat16)` | `h2exp(__nv_bfloat162)` | Base-e exponential | bfloat16 | optimization/compute/half-precision-math, pattern/attention | core |
| `hexp2(__nv_bfloat16)` | `h2exp2(__nv_bfloat162)` | Base-2 exponential | bfloat16 | optimization/compute/half-precision-math | related |
| `hexp10(__nv_bfloat16)` | `h2exp10(__nv_bfloat162)` | Base-10 exponential | bfloat16 | optimization/compute/half-precision-math | related |
| `hlog(__nv_bfloat16)` | `h2log(__nv_bfloat162)` | Natural logarithm | bfloat16 | optimization/compute/half-precision-math, pattern/normalization | core |
| `hlog2(__nv_bfloat16)` | `h2log2(__nv_bfloat162)` | Base-2 logarithm | bfloat16 | optimization/compute/half-precision-math | related |
| `hlog10(__nv_bfloat16)` | `h2log10(__nv_bfloat162)` | Base-10 logarithm | bfloat16 | optimization/compute/half-precision-math | related |

---

## 2. Square Root and Reciprocal Functions

Critical for normalization layers (LayerNorm, RMSNorm) and distance computations.

### 2.1 Single Precision (float)

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `sqrtf(x)` | `__fsqrt_rn(x)` | Square root | single | pattern/normalization | core |
| `rsqrtf(x)` | `__frsqrt_rn(x)` | Reciprocal square root 1/sqrt(x) | single | pattern/normalization, optimization/compute/fast-math | core |
| `cbrtf(x)` | -- | Cube root x^(1/3) | single | pattern/elementwise | low-relevance |
| `rcbrtf(x)` | -- | Reciprocal cube root 1/cbrt(x) | single | pattern/elementwise | low-relevance |
| `hypotf(x,y)` | -- | sqrt(x^2 + y^2) without overflow | single | pattern/elementwise | low-relevance |
| `rhypotf(x,y)` | -- | 1/sqrt(x^2 + y^2) | single | pattern/elementwise | low-relevance |
| `norm3df(a,b,c)` | -- | sqrt(a^2 + b^2 + c^2) | single | pattern/elementwise | low-relevance |
| `rnorm3df(a,b,c)` | -- | 1/sqrt(a^2 + b^2 + c^2) | single | pattern/elementwise | low-relevance |
| `norm4df(a,b,c,d)` | -- | sqrt(a^2 + b^2 + c^2 + d^2) | single | pattern/elementwise | low-relevance |
| `rnorm4df(a,b,c,d)` | -- | 1/sqrt(a^2 + b^2 + c^2 + d^2) | single | pattern/elementwise | low-relevance |
| `normf(dim, p)` | -- | sqrt(sum of squares) for N coords | single | pattern/elementwise | low-relevance |
| `rnormf(dim, p)` | -- | 1/sqrt(sum of squares) for N coords | single | pattern/elementwise | low-relevance |

### 2.2 Single Precision Intrinsics (rounding-mode control)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__fsqrt_rd(x)` | sqrt in round-down mode | single | optimization/compute/compiler-hints | related |
| `__fsqrt_rn(x)` | sqrt in round-to-nearest-even mode | single | optimization/compute/compiler-hints | related |
| `__fsqrt_ru(x)` | sqrt in round-up mode | single | optimization/compute/compiler-hints | related |
| `__fsqrt_rz(x)` | sqrt in round-towards-zero mode | single | optimization/compute/compiler-hints | related |
| `__frsqrt_rn(x)` | 1/sqrt(x) in round-to-nearest-even mode | single | pattern/normalization, optimization/compute/compiler-hints | core |

### 2.3 Double Precision (double)

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `sqrt(x)` | `__dsqrt_rn(x)` | Square root | double | pattern/normalization | related |
| `rsqrt(x)` | -- | Reciprocal square root | double | pattern/normalization | related |
| `cbrt(x)` | -- | Cube root | double | pattern/elementwise | low-relevance |
| `rcbrt(x)` | -- | Reciprocal cube root | double | pattern/elementwise | low-relevance |

### 2.4 Double Precision Intrinsics (rounding-mode control)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__dsqrt_rd(x)` | sqrt in round-down mode | double | optimization/compute/compiler-hints | low-relevance |
| `__dsqrt_rn(x)` | sqrt in round-to-nearest-even mode | double | optimization/compute/compiler-hints | low-relevance |
| `__dsqrt_ru(x)` | sqrt in round-up mode | double | optimization/compute/compiler-hints | low-relevance |
| `__dsqrt_rz(x)` | sqrt in round-towards-zero mode | double | optimization/compute/compiler-hints | low-relevance |

### 2.5 Half Precision

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hsqrt(__half)` | `h2sqrt(__half2)` | Square root | half | optimization/compute/half-precision-math, pattern/normalization | core |
| `hrsqrt(__half)` | `h2rsqrt(__half2)` | Reciprocal square root | half | optimization/compute/half-precision-math, pattern/normalization | core |

### 2.6 BFloat16

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hsqrt(__nv_bfloat16)` | `h2sqrt(__nv_bfloat162)` | Square root | bfloat16 | optimization/compute/half-precision-math, pattern/normalization | core |
| `hrsqrt(__nv_bfloat16)` | `h2rsqrt(__nv_bfloat162)` | Reciprocal square root | bfloat16 | optimization/compute/half-precision-math, pattern/normalization | core |

---

## 3. Trigonometric Functions

Used in positional encodings (RoPE), activation functions, and signal processing kernels.

### 3.1 Single Precision (float)

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `sinf(x)` | `__sinf(x)` | Sine | single | optimization/compute/fast-math, pattern/elementwise | core |
| `cosf(x)` | `__cosf(x)` | Cosine | single | optimization/compute/fast-math, pattern/elementwise | core |
| `tanf(x)` | `__tanf(x)` | Tangent | single | optimization/compute/fast-math | related |
| `sincosf(x,s,c)` | `__sincosf(x,s,c)` | Sine and cosine together | single | optimization/compute/fast-math, pattern/elementwise | core |
| `sinpif(x)` | -- | sin(x * pi) | single | pattern/elementwise | low-relevance |
| `cospif(x)` | -- | cos(x * pi) | single | pattern/elementwise | low-relevance |
| `sincospif(x,s,c)` | -- | sin(x*pi) and cos(x*pi) together | single | pattern/elementwise | low-relevance |
| `asinf(x)` | -- | Arc sine | single | pattern/elementwise | low-relevance |
| `acosf(x)` | -- | Arc cosine | single | pattern/elementwise | low-relevance |
| `atanf(x)` | -- | Arc tangent | single | pattern/elementwise | low-relevance |
| `atan2f(y,x)` | -- | Arc tangent of y/x | single | pattern/elementwise | low-relevance |

### 3.2 Hyperbolic Functions (float)

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `tanhf(x)` | `__tanhf(x)` | Hyperbolic tangent | single | optimization/compute/fast-math, pattern/elementwise | core |
| `sinhf(x)` | -- | Hyperbolic sine | single | pattern/elementwise | low-relevance |
| `coshf(x)` | -- | Hyperbolic cosine | single | pattern/elementwise | low-relevance |
| `atanhf(x)` | -- | Inverse hyperbolic tangent | single | pattern/elementwise | low-relevance |
| `asinhf(x)` | -- | Inverse hyperbolic sine | single | pattern/elementwise | low-relevance |
| `acoshf(x)` | -- | Inverse hyperbolic cosine | single | pattern/elementwise | low-relevance |

### 3.3 Double Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `sin(x)` | Sine | double | pattern/elementwise | low-relevance |
| `cos(x)` | Cosine | double | pattern/elementwise | low-relevance |
| `tan(x)` | Tangent | double | pattern/elementwise | low-relevance |
| `sincos(x,s,c)` | Sine and cosine together | double | pattern/elementwise | low-relevance |
| `tanh(x)` | Hyperbolic tangent | double | pattern/elementwise | related |

### 3.4 Half Precision

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hsin(__half)` | `h2sin(__half2)` | Sine | half | optimization/compute/half-precision-math | related |
| `hcos(__half)` | `h2cos(__half2)` | Cosine | half | optimization/compute/half-precision-math | related |

### 3.5 BFloat16

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hsin(__nv_bfloat16)` | `h2sin(__nv_bfloat162)` | Sine | bfloat16 | optimization/compute/half-precision-math | related |
| `hcos(__nv_bfloat16)` | `h2cos(__nv_bfloat162)` | Cosine | bfloat16 | optimization/compute/half-precision-math | related |

---

## 4. Basic Arithmetic Intrinsics

Low-level intrinsics with explicit rounding-mode control. Used when precise numeric behavior
is required or when preventing FMA contraction.

### 4.1 Single Precision Arithmetic Intrinsics

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__fadd_rn(x,y)` | Add, round-to-nearest-even | single | optimization/compute/compiler-hints, pattern/elementwise | core |
| `__fadd_rd(x,y)` | Add, round-down | single | optimization/compute/compiler-hints | related |
| `__fadd_ru(x,y)` | Add, round-up | single | optimization/compute/compiler-hints | related |
| `__fadd_rz(x,y)` | Add, round-towards-zero | single | optimization/compute/compiler-hints | related |
| `__fsub_rn(x,y)` | Subtract, round-to-nearest-even | single | optimization/compute/compiler-hints, pattern/elementwise | core |
| `__fsub_rd(x,y)` | Subtract, round-down | single | optimization/compute/compiler-hints | related |
| `__fsub_ru(x,y)` | Subtract, round-up | single | optimization/compute/compiler-hints | related |
| `__fsub_rz(x,y)` | Subtract, round-towards-zero | single | optimization/compute/compiler-hints | related |
| `__fmul_rn(x,y)` | Multiply, round-to-nearest-even | single | optimization/compute/compiler-hints, pattern/elementwise | core |
| `__fmul_rd(x,y)` | Multiply, round-down | single | optimization/compute/compiler-hints | related |
| `__fmul_ru(x,y)` | Multiply, round-up | single | optimization/compute/compiler-hints | related |
| `__fmul_rz(x,y)` | Multiply, round-towards-zero | single | optimization/compute/compiler-hints | related |
| `__fdiv_rn(x,y)` | Divide, round-to-nearest-even | single | optimization/compute/compiler-hints | related |
| `__fdiv_rd(x,y)` | Divide, round-down | single | optimization/compute/compiler-hints | related |
| `__fdiv_ru(x,y)` | Divide, round-up | single | optimization/compute/compiler-hints | related |
| `__fdiv_rz(x,y)` | Divide, round-towards-zero | single | optimization/compute/compiler-hints | related |
| `fdividef(x,y)` | `__fdividef(x,y)` | Fast approximate divide | single | optimization/compute/fast-math | core |
| `__frcp_rn(x)` | Reciprocal 1/x, round-to-nearest-even | single | optimization/compute/compiler-hints | related |
| `__frcp_rd(x)` | Reciprocal, round-down | single | optimization/compute/compiler-hints | related |
| `__frcp_ru(x)` | Reciprocal, round-up | single | optimization/compute/compiler-hints | related |
| `__frcp_rz(x)` | Reciprocal, round-towards-zero | single | optimization/compute/compiler-hints | related |

### 4.2 Single Precision FMA Intrinsics

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `fmaf(x,y,z)` | Fused multiply-add x*y+z (standard) | single | pattern/elementwise, pattern/normalization | core |
| `__fmaf_rn(x,y,z)` | FMA, round-to-nearest-even | single | optimization/compute/compiler-hints | core |
| `__fmaf_rd(x,y,z)` | FMA, round-down | single | optimization/compute/compiler-hints | related |
| `__fmaf_ru(x,y,z)` | FMA, round-up | single | optimization/compute/compiler-hints | related |
| `__fmaf_rz(x,y,z)` | FMA, round-towards-zero | single | optimization/compute/compiler-hints | related |
| `__fmaf_ieee_rn(x,y,z)` | FMA, round-to-nearest, ignore -ftz | single | optimization/compute/compiler-hints | related |
| `__fmaf_ieee_rd(x,y,z)` | FMA, round-down, ignore -ftz | single | optimization/compute/compiler-hints | low-relevance |
| `__fmaf_ieee_ru(x,y,z)` | FMA, round-up, ignore -ftz | single | optimization/compute/compiler-hints | low-relevance |
| `__fmaf_ieee_rz(x,y,z)` | FMA, round-towards-zero, ignore -ftz | single | optimization/compute/compiler-hints | low-relevance |

### 4.3 Single Precision Vector Arithmetic Intrinsics (float2, CC >= 10.0)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__fadd2_rn(x,y)` | Vector add (float2), round-to-nearest | single x2 | optimization/compute/compiler-hints | related |
| `__fadd2_rd/ru/rz(x,y)` | Vector add with other rounding modes | single x2 | optimization/compute/compiler-hints | low-relevance |
| `__fmul2_rn(x,y)` | Vector multiply (float2), round-to-nearest | single x2 | optimization/compute/compiler-hints | related |
| `__fmul2_rd/ru/rz(x,y)` | Vector multiply with other rounding modes | single x2 | optimization/compute/compiler-hints | low-relevance |
| `__ffma2_rn(x,y,z)` | Vector FMA (float2), round-to-nearest | single x2 | optimization/compute/compiler-hints | related |
| `__ffma2_rd/ru/rz(x,y,z)` | Vector FMA with other rounding modes | single x2 | optimization/compute/compiler-hints | low-relevance |

### 4.4 Double Precision Arithmetic Intrinsics

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__dadd_rn(x,y)` | Add, round-to-nearest-even | double | optimization/compute/compiler-hints | related |
| `__dadd_rd/ru/rz(x,y)` | Add with other rounding modes | double | optimization/compute/compiler-hints | low-relevance |
| `__dmul_rn(x,y)` | Multiply, round-to-nearest-even | double | optimization/compute/compiler-hints | related |
| `__dmul_rd/ru/rz(x,y)` | Multiply with other rounding modes | double | optimization/compute/compiler-hints | low-relevance |
| `__ddiv_rn(x,y)` | Divide, round-to-nearest-even | double | optimization/compute/compiler-hints | related |
| `__ddiv_rd/ru/rz(x,y)` | Divide with other rounding modes | double | optimization/compute/compiler-hints | low-relevance |
| `__dsub_rn(x,y)` | Subtract, round-to-nearest-even | double | optimization/compute/compiler-hints | related |
| `__dsub_rd/ru/rz(x,y)` | Subtract with other rounding modes | double | optimization/compute/compiler-hints | low-relevance |
| `__drcp_rn(x)` | Reciprocal, round-to-nearest-even | double | optimization/compute/compiler-hints | related |
| `__drcp_rd/ru/rz(x)` | Reciprocal with other rounding modes | double | optimization/compute/compiler-hints | low-relevance |
| `fma(x,y,z)` | Fused multiply-add (standard) | double | pattern/elementwise | related |
| `__fma_rn(x,y,z)` | FMA, round-to-nearest-even | double | optimization/compute/compiler-hints | related |
| `__fma_rd/ru/rz(x,y,z)` | FMA with other rounding modes | double | optimization/compute/compiler-hints | low-relevance |

---

## 5. Half Precision Arithmetic (cuda_fp16.h)

Packed half2 operations process two FP16 values in a single instruction, doubling throughput.
This is the primary data type for inference kernels and many training kernels.

### 5.1 Half Scalar Arithmetic

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__hadd(a,b)` | Add two half values | half | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hsub(a,b)` | Subtract two half values | half | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hmul(a,b)` | Multiply two half values | half | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hdiv(a,b)` | Divide two half values | half | optimization/compute/half-precision-math | related |
| `__hfma(a,b,c)` | Fused multiply-add a*b+c | half | optimization/compute/half-precision-math, pattern/normalization | core |
| `__hfma_relu(a,b,c)` | FMA with ReLU fusion | half | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hneg(a)` | Negate | half | optimization/compute/half-precision-math | related |
| `__habs(a)` | Absolute value | half | optimization/compute/half-precision-math | related |
| `__hmax(a,b)` | Maximum | half | optimization/compute/half-precision-math, pattern/attention | core |
| `__hmin(a,b)` | Minimum | half | optimization/compute/half-precision-math | related |
| `__hfma_sat(a,b,c)` | FMA with saturation to [0,1] | half | optimization/compute/half-precision-math | related |
| `__hadd_sat(a,b)` | Add with saturation to [0,1] | half | optimization/compute/half-precision-math | low-relevance |
| `__hsub_sat(a,b)` | Subtract with saturation to [0,1] | half | optimization/compute/half-precision-math | low-relevance |
| `__hmul_sat(a,b)` | Multiply with saturation to [0,1] | half | optimization/compute/half-precision-math | low-relevance |

### 5.2 Half2 Packed Arithmetic (2x throughput)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__hadd2(a,b)` | Packed add (two half values at once) | half2 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hsub2(a,b)` | Packed subtract | half2 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hmul2(a,b)` | Packed multiply | half2 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__h2div(a,b)` | Packed divide | half2 | optimization/compute/half-precision-math | related |
| `__hfma2(a,b,c)` | Packed fused multiply-add | half2 | optimization/compute/half-precision-math, pattern/normalization | core |
| `__hfma2_relu(a,b,c)` | Packed FMA with ReLU fusion | half2 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hneg2(a)` | Packed negate | half2 | optimization/compute/half-precision-math | related |
| `__habs2(a)` | Packed absolute value | half2 | optimization/compute/half-precision-math | related |
| `__hmax2(a,b)` | Packed maximum | half2 | optimization/compute/half-precision-math, pattern/attention | core |
| `__hmin2(a,b)` | Packed minimum | half2 | optimization/compute/half-precision-math | related |
| `__hfma2_sat(a,b,c)` | Packed FMA with saturation | half2 | optimization/compute/half-precision-math | related |
| `__hadd2_sat(a,b)` | Packed add with saturation | half2 | optimization/compute/half-precision-math | low-relevance |
| `__hsub2_sat(a,b)` | Packed subtract with saturation | half2 | optimization/compute/half-precision-math | low-relevance |
| `__hmul2_sat(a,b)` | Packed multiply with saturation | half2 | optimization/compute/half-precision-math | low-relevance |

### 5.3 Half Math Functions

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hrcp(__half)` | `h2rcp(__half2)` | Reciprocal 1/x | half | optimization/compute/half-precision-math | related |
| `hceil(__half)` | `h2ceil(__half2)` | Ceiling | half | optimization/compute/half-precision-math | low-relevance |
| `hfloor(__half)` | `h2floor(__half2)` | Floor | half | optimization/compute/half-precision-math | low-relevance |
| `hrint(__half)` | `h2rint(__half2)` | Round to nearest integer | half | optimization/compute/half-precision-math | low-relevance |
| `htrunc(__half)` | `h2trunc(__half2)` | Truncate to integer | half | optimization/compute/half-precision-math | low-relevance |

### 5.4 Half Comparison Functions

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `__heq(a,b)` | `__heq2(a,b)` | Equal comparison | half | optimization/compute/half-precision-math | related |
| `__hne(a,b)` | `__hne2(a,b)` | Not-equal comparison | half | optimization/compute/half-precision-math | related |
| `__hgt(a,b)` | `__hgt2(a,b)` | Greater-than comparison | half | optimization/compute/half-precision-math | related |
| `__hge(a,b)` | `__hge2(a,b)` | Greater-or-equal comparison | half | optimization/compute/half-precision-math | related |
| `__hlt(a,b)` | `__hlt2(a,b)` | Less-than comparison | half | optimization/compute/half-precision-math | related |
| `__hle(a,b)` | `__hle2(a,b)` | Less-or-equal comparison | half | optimization/compute/half-precision-math | related |
| `__hisnan(a)` | `__hisnan2(a)` | Check if NaN | half | optimization/compute/half-precision-math | low-relevance |
| `__hisinf(a)` | -- | Check if infinite | half | optimization/compute/half-precision-math | low-relevance |

---

## 6. BFloat16 Arithmetic (cuda_bf16.h)

Same API pattern as half-precision but for BFloat16 format (8-bit exponent, 7-bit mantissa).
Preferred for training due to wider dynamic range matching float32 exponent.

### 6.1 BFloat16 Scalar Arithmetic

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__hadd(a,b)` | Add two bfloat16 values | bfloat16 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hsub(a,b)` | Subtract two bfloat16 values | bfloat16 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hmul(a,b)` | Multiply two bfloat16 values | bfloat16 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hdiv(a,b)` | Divide two bfloat16 values | bfloat16 | optimization/compute/half-precision-math | related |
| `__hfma(a,b,c)` | Fused multiply-add | bfloat16 | optimization/compute/half-precision-math, pattern/normalization | core |
| `__hfma_relu(a,b,c)` | FMA with ReLU fusion | bfloat16 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hneg(a)` | Negate | bfloat16 | optimization/compute/half-precision-math | related |
| `__habs(a)` | Absolute value | bfloat16 | optimization/compute/half-precision-math | related |
| `__hmax(a,b)` | Maximum | bfloat16 | optimization/compute/half-precision-math, pattern/attention | core |
| `__hmin(a,b)` | Minimum | bfloat16 | optimization/compute/half-precision-math | related |

### 6.2 BFloat162 Packed Arithmetic (2x throughput)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__hadd2(a,b)` | Packed add | bfloat162 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hsub2(a,b)` | Packed subtract | bfloat162 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hmul2(a,b)` | Packed multiply | bfloat162 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__h2div(a,b)` | Packed divide | bfloat162 | optimization/compute/half-precision-math | related |
| `__hfma2(a,b,c)` | Packed FMA | bfloat162 | optimization/compute/half-precision-math, pattern/normalization | core |
| `__hfma2_relu(a,b,c)` | Packed FMA with ReLU | bfloat162 | optimization/compute/half-precision-math, pattern/elementwise | core |
| `__hneg2(a)` | Packed negate | bfloat162 | optimization/compute/half-precision-math | related |
| `__habs2(a)` | Packed absolute value | bfloat162 | optimization/compute/half-precision-math | related |
| `__hmax2(a,b)` | Packed maximum | bfloat162 | optimization/compute/half-precision-math, pattern/attention | core |
| `__hmin2(a,b)` | Packed minimum | bfloat162 | optimization/compute/half-precision-math | related |

### 6.3 BFloat16 Math Functions

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `hrcp(__nv_bfloat16)` | `h2rcp(__nv_bfloat162)` | Reciprocal | bfloat16 | optimization/compute/half-precision-math | related |
| `hceil(__nv_bfloat16)` | `h2ceil(__nv_bfloat162)` | Ceiling | bfloat16 | optimization/compute/half-precision-math | low-relevance |
| `hfloor(__nv_bfloat16)` | `h2floor(__nv_bfloat162)` | Floor | bfloat16 | optimization/compute/half-precision-math | low-relevance |
| `hrint(__nv_bfloat16)` | `h2rint(__nv_bfloat162)` | Round to nearest integer | bfloat16 | optimization/compute/half-precision-math | low-relevance |
| `htrunc(__nv_bfloat16)` | `h2trunc(__nv_bfloat162)` | Truncate to integer | bfloat16 | optimization/compute/half-precision-math | low-relevance |

### 6.4 BFloat16 Comparison Functions

| Function | Packed Variant | Description | Precision | Knowledge Node | Relevance |
|----------|---------------|-------------|-----------|----------------|-----------|
| `__heq(a,b)` | `__heq2(a,b)` | Equal | bfloat16 | optimization/compute/half-precision-math | related |
| `__hne(a,b)` | `__hne2(a,b)` | Not-equal | bfloat16 | optimization/compute/half-precision-math | related |
| `__hgt(a,b)` | `__hgt2(a,b)` | Greater-than | bfloat16 | optimization/compute/half-precision-math | related |
| `__hge(a,b)` | `__hge2(a,b)` | Greater-or-equal | bfloat16 | optimization/compute/half-precision-math | related |
| `__hlt(a,b)` | `__hlt2(a,b)` | Less-than | bfloat16 | optimization/compute/half-precision-math | related |
| `__hle(a,b)` | `__hle2(a,b)` | Less-or-equal | bfloat16 | optimization/compute/half-precision-math | related |

---

## 7. Error and Special Functions

Used in GELU activation, probability distributions, and specialized ML layers.

### 7.1 Single Precision

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `erff(x)` | -- | Error function erf(x) | single | pattern/elementwise, pattern/normalization | core |
| `erfcf(x)` | -- | Complementary error function 1-erf(x) | single | pattern/elementwise | related |
| `erfinvf(x)` | -- | Inverse error function | single | pattern/elementwise | low-relevance |
| `erfcinvf(x)` | -- | Inverse complementary error function | single | pattern/elementwise | low-relevance |
| `erfcxf(x)` | -- | Scaled complementary error function | single | pattern/elementwise | low-relevance |
| `tgammaf(x)` | -- | Gamma function | single | pattern/elementwise | low-relevance |
| `lgammaf(x)` | -- | Log-gamma function | single | pattern/elementwise | low-relevance |
| `normcdff(x)` | -- | Normal CDF (used in GELU) | single | pattern/elementwise, pattern/normalization | core |
| `normcdfinvf(x)` | -- | Inverse normal CDF | single | pattern/elementwise | low-relevance |

### 7.2 Double Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `erf(x)` | Error function | double | pattern/elementwise | related |
| `erfc(x)` | Complementary error function | double | pattern/elementwise | low-relevance |
| `erfinv(x)` | Inverse error function | double | pattern/elementwise | low-relevance |
| `erfcinv(x)` | Inverse complementary error function | double | pattern/elementwise | low-relevance |
| `erfcx(x)` | Scaled complementary error function | double | pattern/elementwise | low-relevance |
| `tgamma(x)` | Gamma function | double | pattern/elementwise | low-relevance |
| `lgamma(x)` | Log-gamma function | double | pattern/elementwise | low-relevance |
| `normcdf(x)` | Normal CDF | double | pattern/elementwise | related |
| `normcdfinv(x)` | Inverse normal CDF | double | pattern/elementwise | low-relevance |

---

## 8. Min / Max / Abs / Clamp

Critical for attention score clamping, activation functions, and numerical stability.

### 8.1 Single Precision

| Function | Fast Variant | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-------------|-----------|----------------|-----------|
| `fmaxf(x,y)` | -- | Maximum (NaN-safe) | single | pattern/attention, pattern/elementwise | core |
| `fminf(x,y)` | -- | Minimum (NaN-safe) | single | pattern/attention, pattern/elementwise | core |
| `fabsf(x)` | -- | Absolute value | single | pattern/elementwise | core |
| `__saturatef(x)` | -- | Clamp to [0.0, 1.0] | single | optimization/compute/fast-math, pattern/elementwise | related |
| `copysignf(x,y)` | -- | Copy sign of y to magnitude of x | single | pattern/elementwise | low-relevance |
| `fdimf(x,y)` | -- | Positive difference max(x-y, 0) | single | pattern/elementwise | low-relevance |

### 8.2 Double Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `fmax(x,y)` | Maximum (NaN-safe) | double | pattern/attention | related |
| `fmin(x,y)` | Minimum (NaN-safe) | double | pattern/attention | related |
| `fabs(x)` | Absolute value | double | pattern/elementwise | related |
| `copysign(x,y)` | Copy sign | double | pattern/elementwise | low-relevance |

---

## 9. Rounding and Remainder Functions

### 9.1 Single Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `floorf(x)` | Floor (largest integer <= x) | single | pattern/elementwise | related |
| `ceilf(x)` | Ceiling (smallest integer >= x) | single | pattern/elementwise | related |
| `truncf(x)` | Truncate to integer part | single | pattern/elementwise | related |
| `rintf(x)` | Round to nearest integer (in float) | single | pattern/elementwise | related |
| `roundf(x)` | Round to nearest integer, halfway away from zero | single | pattern/elementwise | related |
| `nearbyintf(x)` | Round to nearest integer | single | pattern/elementwise | low-relevance |
| `fmodf(x,y)` | Floating-point remainder | single | pattern/elementwise | low-relevance |
| `remainderf(x,y)` | IEEE remainder | single | pattern/elementwise | low-relevance |
| `modff(x, iptr)` | Decompose into integer + fractional | single | pattern/elementwise | low-relevance |
| `frexpf(x, nptr)` | Extract mantissa and exponent | single | pattern/elementwise | low-relevance |
| `ldexpf(x, exp)` | Compute x * 2^exp | single | pattern/elementwise | low-relevance |
| `scalbnf(x, n)` | Scale by power of 2 | single | pattern/elementwise | low-relevance |
| `logbf(x)` | Extract exponent | single | pattern/elementwise | low-relevance |
| `ilogbf(x)` | Extract unbiased integer exponent | single | pattern/elementwise | low-relevance |

### 9.2 Double Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `floor(x)` | Floor | double | pattern/elementwise | low-relevance |
| `ceil(x)` | Ceiling | double | pattern/elementwise | low-relevance |
| `trunc(x)` | Truncate | double | pattern/elementwise | low-relevance |
| `rint(x)` | Round to nearest | double | pattern/elementwise | low-relevance |
| `round(x)` | Round, halfway away from zero | double | pattern/elementwise | low-relevance |
| `fmod(x,y)` | Floating-point remainder | double | pattern/elementwise | low-relevance |
| `remainder(x,y)` | IEEE remainder | double | pattern/elementwise | low-relevance |

---

## 10. Type Conversion Intrinsics

### 10.1 Half <-> Float Conversions (cuda_fp16.h)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__float2half(x)` | float -> half (round-to-nearest) | float->half | optimization/compute/half-precision-math | core |
| `__float2half_rn(x)` | float -> half, round-to-nearest-even | float->half | optimization/compute/half-precision-math | core |
| `__float2half_rd(x)` | float -> half, round-down | float->half | optimization/compute/compiler-hints | related |
| `__float2half_ru(x)` | float -> half, round-up | float->half | optimization/compute/compiler-hints | related |
| `__float2half_rz(x)` | float -> half, round-towards-zero | float->half | optimization/compute/compiler-hints | related |
| `__half2float(x)` | half -> float | half->float | optimization/compute/half-precision-math | core |
| `__float2half2_rn(x)` | float -> half2 (both components = x) | float->half2 | optimization/compute/half-precision-math | core |
| `__floats2half2_rn(x,y)` | Two floats -> half2 | float->half2 | optimization/compute/half-precision-math | core |
| `__half22float2(x)` | half2 -> float2 | half2->float2 | optimization/compute/half-precision-math | core |
| `__halves2half2(a,b)` | Two halves -> half2 | half->half2 | optimization/compute/half-precision-math | core |
| `__low2half(x)` | Extract low half from half2 | half2->half | optimization/compute/half-precision-math | related |
| `__high2half(x)` | Extract high half from half2 | half2->half | optimization/compute/half-precision-math | related |
| `__low2half2(x)` | Replicate low half to half2 | half2->half2 | optimization/compute/half-precision-math | related |
| `__high2half2(x)` | Replicate high half to half2 | half2->half2 | optimization/compute/half-precision-math | related |
| `__lowhigh2highlow(x)` | Swap low and high in half2 | half2->half2 | optimization/compute/half-precision-math | related |
| `__half2short_rn(x)` | half -> short | half->int | optimization/compute/half-precision-math | low-relevance |
| `__half2int_rn(x)` | half -> int | half->int | optimization/compute/half-precision-math | low-relevance |
| `__short2half_rn(x)` | short -> half | int->half | optimization/compute/half-precision-math | low-relevance |
| `__int2half_rn(x)` | int -> half | int->half | optimization/compute/half-precision-math | low-relevance |

### 10.2 BFloat16 <-> Float Conversions (cuda_bf16.h)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__float2bfloat16(x)` | float -> bfloat16 (round-to-nearest) | float->bf16 | optimization/compute/half-precision-math | core |
| `__float2bfloat16_rn(x)` | float -> bfloat16, round-to-nearest | float->bf16 | optimization/compute/half-precision-math | core |
| `__float2bfloat16_rd(x)` | float -> bfloat16, round-down | float->bf16 | optimization/compute/compiler-hints | related |
| `__float2bfloat16_ru(x)` | float -> bfloat16, round-up | float->bf16 | optimization/compute/compiler-hints | related |
| `__float2bfloat16_rz(x)` | float -> bfloat16, round-towards-zero | float->bf16 | optimization/compute/compiler-hints | related |
| `__bfloat162float(x)` | bfloat16 -> float | bf16->float | optimization/compute/half-precision-math | core |
| `__float2bfloat162_rn(x)` | float -> bfloat162 (both = x) | float->bf162 | optimization/compute/half-precision-math | core |
| `__floats2bfloat162_rn(x,y)` | Two floats -> bfloat162 | float->bf162 | optimization/compute/half-precision-math | core |
| `__bfloat1622float2(x)` | bfloat162 -> float2 | bf162->float2 | optimization/compute/half-precision-math | core |
| `__halves2bfloat162(a,b)` | Two bfloat16 -> bfloat162 | bf16->bf162 | optimization/compute/half-precision-math | core |
| `__low2bfloat16(x)` | Extract low from bfloat162 | bf162->bf16 | optimization/compute/half-precision-math | related |
| `__high2bfloat16(x)` | Extract high from bfloat162 | bf162->bf16 | optimization/compute/half-precision-math | related |
| `__low2bfloat162(x)` | Replicate low to bfloat162 | bf162->bf162 | optimization/compute/half-precision-math | related |
| `__high2bfloat162(x)` | Replicate high to bfloat162 | bf162->bf162 | optimization/compute/half-precision-math | related |

### 10.3 Half <-> BFloat16 Conversions

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__half2bfloat16(x)` | half -> bfloat16 | half->bf16 | optimization/compute/half-precision-math | related |
| `__bfloat162half(x)` | bfloat16 -> half | bf16->half | optimization/compute/half-precision-math | related |

### 10.4 Float <-> Double Casting Intrinsics

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__double2float_rn(x)` | double -> float, round-to-nearest | double->float | optimization/compute/compiler-hints | related |
| `__double2float_rd(x)` | double -> float, round-down | double->float | optimization/compute/compiler-hints | low-relevance |
| `__double2float_ru(x)` | double -> float, round-up | double->float | optimization/compute/compiler-hints | low-relevance |
| `__double2float_rz(x)` | double -> float, round-towards-zero | double->float | optimization/compute/compiler-hints | low-relevance |

### 10.5 Float/Double <-> Integer Casting Intrinsics

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__float2int_rn(x)` | float -> int, round-to-nearest | float->int | optimization/compute/compiler-hints | related |
| `__float2int_rd/ru/rz(x)` | float -> int, other rounding modes | float->int | optimization/compute/compiler-hints | low-relevance |
| `__float2uint_rn(x)` | float -> unsigned int, round-to-nearest | float->uint | optimization/compute/compiler-hints | related |
| `__float2uint_rd/ru/rz(x)` | float -> unsigned int, other modes | float->uint | optimization/compute/compiler-hints | low-relevance |
| `__float2ll_rn(x)` | float -> long long, round-to-nearest | float->int64 | optimization/compute/compiler-hints | low-relevance |
| `__float2ull_rn(x)` | float -> unsigned long long | float->uint64 | optimization/compute/compiler-hints | low-relevance |
| `__int2float_rn(x)` | int -> float, round-to-nearest | int->float | optimization/compute/compiler-hints | related |
| `__uint2float_rn(x)` | unsigned int -> float | uint->float | optimization/compute/compiler-hints | related |
| `__double2int_rn(x)` | double -> int | double->int | optimization/compute/compiler-hints | low-relevance |
| `__double2ll_rn(x)` | double -> long long | double->int64 | optimization/compute/compiler-hints | low-relevance |
| `__ll2float_rn(x)` | long long -> float | int64->float | optimization/compute/compiler-hints | low-relevance |
| `__ll2double_rn(x)` | long long -> double | int64->double | optimization/compute/compiler-hints | low-relevance |

### 10.6 Bit Reinterpretation Intrinsics

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__float_as_int(x)` | Reinterpret float bits as int | float<->int | optimization/compute/compiler-hints | core |
| `__float_as_uint(x)` | Reinterpret float bits as unsigned int | float<->uint | optimization/compute/compiler-hints | core |
| `__int_as_float(x)` | Reinterpret int bits as float | int<->float | optimization/compute/compiler-hints | core |
| `__uint_as_float(x)` | Reinterpret unsigned int bits as float | uint<->float | optimization/compute/compiler-hints | core |
| `__double_as_longlong(x)` | Reinterpret double bits as long long | double<->int64 | optimization/compute/compiler-hints | related |
| `__longlong_as_double(x)` | Reinterpret long long bits as double | int64<->double | optimization/compute/compiler-hints | related |
| `__double2hiint(x)` | High 32 bits of double as int | double->int | optimization/compute/compiler-hints | low-relevance |
| `__double2loint(x)` | Low 32 bits of double as int | double->int | optimization/compute/compiler-hints | low-relevance |
| `__hiloint2double(hi, lo)` | Two ints -> double | int->double | optimization/compute/compiler-hints | low-relevance |

---

## 11. Integer Intrinsics (Kernel-Relevant Subset)

### 11.1 Bit Manipulation

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__clz(x)` | Count leading zeros (32-bit) | int | optimization/compute/compiler-hints | related |
| `__clzll(x)` | Count leading zeros (64-bit) | int64 | optimization/compute/compiler-hints | related |
| `__ffs(x)` | Find first set bit (32-bit) | int | optimization/compute/compiler-hints | related |
| `__ffsll(x)` | Find first set bit (64-bit) | int64 | optimization/compute/compiler-hints | related |
| `__popc(x)` | Population count / bit count (32-bit) | int | optimization/compute/compiler-hints | related |
| `__popcll(x)` | Population count (64-bit) | int64 | optimization/compute/compiler-hints | related |
| `__brev(x)` | Bit reverse (32-bit) | int | optimization/compute/compiler-hints | low-relevance |
| `__brevll(x)` | Bit reverse (64-bit) | int64 | optimization/compute/compiler-hints | low-relevance |
| `__byte_perm(x,y,s)` | Byte permutation from two 32-bit values | int | optimization/compute/compiler-hints | related |
| `__funnelshift_l(lo,hi,shift)` | Funnel shift left | int | optimization/compute/compiler-hints | related |
| `__funnelshift_r(lo,hi,shift)` | Funnel shift right | int | optimization/compute/compiler-hints | related |
| `__funnelshift_lc(lo,hi,shift)` | Funnel shift left clamped | int | optimization/compute/compiler-hints | low-relevance |
| `__funnelshift_rc(lo,hi,shift)` | Funnel shift right clamped | int | optimization/compute/compiler-hints | low-relevance |

### 11.2 Integer Dot Product (Quantization Kernels)

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `__dp4a(a,b,c)` | 4-way int8 dot product with int32 accumulate | int8->int32 | optimization/compute/compiler-hints, pattern/elementwise | core |
| `__dp2a_lo(a,b,c)` | 2-way int16*int8 dot product (low half) | int16->int32 | optimization/compute/compiler-hints | related |
| `__dp2a_hi(a,b,c)` | 2-way int16*int8 dot product (high half) | int16->int32 | optimization/compute/compiler-hints | related |

---

## 12. Bessel and Other Specialized Functions

Low relevance for typical ML/HPC kernel optimization but included for completeness.

### 12.1 Single Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `j0f(x)` | Bessel function of first kind, order 0 | single | pattern/elementwise | low-relevance |
| `j1f(x)` | Bessel function of first kind, order 1 | single | pattern/elementwise | low-relevance |
| `jnf(n,x)` | Bessel function of first kind, order n | single | pattern/elementwise | low-relevance |
| `y0f(x)` | Bessel function of second kind, order 0 | single | pattern/elementwise | low-relevance |
| `y1f(x)` | Bessel function of second kind, order 1 | single | pattern/elementwise | low-relevance |
| `ynf(n,x)` | Bessel function of second kind, order n | single | pattern/elementwise | low-relevance |
| `cyl_bessel_i0f(x)` | Modified cylindrical Bessel, order 0 | single | pattern/elementwise | low-relevance |
| `cyl_bessel_i1f(x)` | Modified cylindrical Bessel, order 1 | single | pattern/elementwise | low-relevance |

### 12.2 Double Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `j0(x)` | Bessel function of first kind, order 0 | double | pattern/elementwise | low-relevance |
| `j1(x)` | Bessel function of first kind, order 1 | double | pattern/elementwise | low-relevance |
| `jn(n,x)` | Bessel function of first kind, order n | double | pattern/elementwise | low-relevance |
| `y0(x)` | Bessel function of second kind, order 0 | double | pattern/elementwise | low-relevance |
| `y1(x)` | Bessel function of second kind, order 1 | double | pattern/elementwise | low-relevance |
| `yn(n,x)` | Bessel function of second kind, order n | double | pattern/elementwise | low-relevance |
| `cyl_bessel_i0(x)` | Modified cylindrical Bessel, order 0 | double | pattern/elementwise | low-relevance |
| `cyl_bessel_i1(x)` | Modified cylindrical Bessel, order 1 | double | pattern/elementwise | low-relevance |

---

## 13. Classification and Utility Functions

### 13.1 Single Precision

| Function | Description | Precision | Knowledge Node | Relevance |
|----------|-------------|-----------|----------------|-----------|
| `isfinite(x)` | Check if finite | single | pattern/elementwise | related |
| `isinf(x)` | Check if infinite | single | pattern/elementwise | related |
| `isnan(x)` | Check if NaN | single | pattern/elementwise | related |
| `signbit(x)` | Get sign bit | single | pattern/elementwise | low-relevance |
| `nanf(tagp)` | Generate NaN | single | pattern/elementwise | low-relevance |
| `nextafterf(x,y)` | Next representable float after x toward y | single | pattern/elementwise | low-relevance |

---

## 14. Fast-Math Compiler Flag Summary

When `-use_fast_math` is passed to nvcc, the following standard functions are automatically
replaced by their fast intrinsic counterparts:

| Standard Function | Replaced By | Precision Trade-off |
|-------------------|-------------|---------------------|
| `sinf(x)` | `__sinf(x)` | Max 2 ULP error for [-pi, pi]; larger outside |
| `cosf(x)` | `__cosf(x)` | Max 2 ULP error for [-pi, pi]; larger outside |
| `tanf(x)` | `__tanf(x)` | Derived from __sinf/__cosf |
| `sincosf(x,s,c)` | `__sincosf(x,s,c)` | Same as __sinf/__cosf |
| `logf(x)` | `__logf(x)` | Max 1 ULP error; not guaranteed monotonic |
| `log2f(x)` | `__log2f(x)` | Max 1 ULP error |
| `log10f(x)` | `__log10f(x)` | Max 1 ULP error |
| `expf(x)` | `__expf(x)` | Max 2 ULP error |
| `exp10f(x)` | `__exp10f(x)` | Max 2 ULP error |
| `powf(x,y)` | `__powf(x,y)` | Max 8 ULP error |
| `fdividef(x,y)` | `__fdividef(x,y)` | Returns 0 for 2^126 < |y| < 2^128 |

Additional `-use_fast_math` effects:
- Enables `--ftz=true` (flush denormals to zero)
- Enables `--prec-div=false` (fast division)
- Enables `--prec-sqrt=false` (fast square root)
- Enables `--fmad=true` (fused multiply-add contraction)

---

## Summary

### Counts by Category

| Category | Total Functions | Core | Related | Low-Relevance |
|----------|----------------|------|---------|---------------|
| 1. Exponential / Logarithmic | 42 | 12 | 16 | 14 |
| 2. Square Root / Reciprocal | 24 | 6 | 10 | 8 |
| 3. Trigonometric / Hyperbolic | 27 | 6 | 8 | 13 |
| 4. Basic Arithmetic Intrinsics | 53 | 8 | 27 | 18 |
| 5. Half Precision Arithmetic | 38 | 14 | 14 | 10 |
| 6. BFloat16 Arithmetic | 32 | 10 | 14 | 8 |
| 7. Error / Special Functions | 18 | 2 | 3 | 13 |
| 8. Min / Max / Abs / Clamp | 10 | 4 | 3 | 3 |
| 9. Rounding / Remainder | 21 | 0 | 5 | 16 |
| 10. Type Conversions | 48 | 16 | 18 | 14 |
| 11. Integer Intrinsics | 16 | 1 | 8 | 7 |
| 12. Bessel / Specialized | 16 | 0 | 0 | 16 |
| 13. Classification / Utility | 6 | 0 | 3 | 3 |
| **Total** | **351** | **79** | **129** | **143** |

### Counts by Knowledge Node

| Knowledge Node | Core | Related | Low-Relevance |
|----------------|------|---------|---------------|
| optimization/compute/fast-math | 16 | 7 | 0 |
| optimization/compute/half-precision-math | 38 | 52 | 20 |
| optimization/compute/compiler-hints | 8 | 48 | 36 |
| pattern/normalization | 14 | 6 | 0 |
| pattern/attention | 6 | 2 | 0 |
| pattern/elementwise | 12 | 18 | 82 |

### Key Takeaways for Kernel Optimization

1. **Fast-math intrinsics** (`__expf`, `__logf`, `__sinf`, `__cosf`, `__rsqrtf`, `__fdividef`, `__powf`, `__tanhf`) provide 2-8x speedup over standard library functions with 1-8 ULP error -- sufficient for ML inference.

2. **Half2 packed operations** (`__hadd2`, `__hmul2`, `__hfma2`) double arithmetic throughput by processing two FP16 values per instruction. Always prefer packed variants in performance-critical loops.

3. **BFloat16** mirrors the half API but offers wider dynamic range (same exponent as float32). Use for training workloads; use half for inference.

4. **`rsqrtf` / `__frsqrt_rn`** is the single most important function for normalization kernels (LayerNorm, RMSNorm, BatchNorm). Maps to a single SFU instruction.

5. **`erff` / `normcdff`** are critical for GELU activation. Consider polynomial approximations in custom kernels for higher throughput.

6. **`__dp4a`** enables int8 dot products with int32 accumulation -- essential for quantized inference kernels.

7. **Type conversion intrinsics** (`__float2half_rn`, `__half2float`, `__float2bfloat16_rn`) are on the critical path for mixed-precision kernels. The rn (round-to-nearest) variants are the default choice.

8. **Bit reinterpretation** (`__float_as_int`, `__int_as_float`) is zero-cost and used for branchless comparisons, NaN handling, and fast absolute value via bit masking.
