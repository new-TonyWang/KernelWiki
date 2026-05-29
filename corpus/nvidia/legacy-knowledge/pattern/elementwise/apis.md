# Elementwise -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__cosf(x)` | Math Intrinsics | Fast variant of cosf(x) |
| `__dp4a(a,b,c)` | Math Intrinsics | 4-way int8 dot product with int32 accumulate |
| `__fadd_rn(x,y)` | Math Intrinsics | Add, round-to-nearest-even |
| `__fmul_rn(x,y)` | Math Intrinsics | Multiply, round-to-nearest-even |
| `__fsub_rn(x,y)` | Math Intrinsics | Subtract, round-to-nearest-even |
| `__hadd(a,b)` | Math Intrinsics | Add two half values |
| `__hadd2(a,b)` | Math Intrinsics | Packed add (two half values at once) |
| `__hfma2_relu(a,b,c)` | Math Intrinsics | Packed FMA with ReLU fusion |
| `__hfma_relu(a,b,c)` | Math Intrinsics | FMA with ReLU fusion |
| `__hmul(a,b)` | Math Intrinsics | Multiply two half values |
| `__hmul2(a,b)` | Math Intrinsics | Packed multiply |
| `__hsub(a,b)` | Math Intrinsics | Subtract two half values |
| `__hsub2(a,b)` | Math Intrinsics | Packed subtract |
| `__powf(x,y)` | Math Intrinsics | Fast variant of powf(x,y) |
| `__sincosf(x,s,c)` | Math Intrinsics | Fast variant of sincosf(x,s,c) |
| `__sinf(x)` | Math Intrinsics | Fast variant of sinf(x) |
| `__tanhf(x)` | Math Intrinsics | Fast variant of tanhf(x) |
| `cosf(x)` | Math Intrinsics | Cosine |
| `erff(x)` | Math Intrinsics | Error function erf(x) |
| `fabsf(x)` | Math Intrinsics | Absolute value |
| `fmaf(x,y,z)` | Math Intrinsics | Fused multiply-add x*y+z (standard) |
| `fmaxf(x,y)` | Math Intrinsics | Maximum (NaN-safe) |
| `fminf(x,y)` | Math Intrinsics | Minimum (NaN-safe) |
| `normcdff(x)` | Math Intrinsics | Normal CDF (used in GELU) |
| `powf(x,y)` | Math Intrinsics | x raised to power y |
| `sincosf(x,s,c)` | Math Intrinsics | Sine and cosine together |
| `sinf(x)` | Math Intrinsics | Sine |
| `tanhf(x)` | Math Intrinsics | Hyperbolic tangent |
| `CUBLASLT_EPILOGUE_BIAS` | cuBLAS | Add broadcast bias vector |
| `CUBLASLT_EPILOGUE_GELU` | cuBLAS | Apply GELU activation |
| `CUBLASLT_EPILOGUE_GELU_BIAS` | cuBLAS | Bias + GELU |
| `CUBLASLT_EPILOGUE_RELU` | cuBLAS | Apply ReLU: x = max(x, 0) |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | cuBLAS | Bias + ReLU |
| `cublas<t>axpy()` | cuBLAS | y = alpha*x + y |
| `cublas<t>scal()` | cuBLAS | x = alpha*x (vector scale) |
| `cublasAxpyEx()` | cuBLAS | y = alpha*x + y with mixed precision |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__saturatef(x)` | Math Intrinsics | Clamp to [0.0, 1.0] |
| `ceilf(x)` | Math Intrinsics | Ceiling (smallest integer >= x) |
| `erf(x)` | Math Intrinsics | Error function |
| `erfcf(x)` | Math Intrinsics | Complementary error function 1-erf(x) |
| `fabs(x)` | Math Intrinsics | Absolute value |
| `floorf(x)` | Math Intrinsics | Floor (largest integer <= x) |
| `fma(x,y,z)` | Math Intrinsics | Fused multiply-add (standard) |
| `isfinite(x)` | Math Intrinsics | Check if finite |
| `isinf(x)` | Math Intrinsics | Check if infinite |
| `isnan(x)` | Math Intrinsics | Check if NaN |
| `normcdf(x)` | Math Intrinsics | Normal CDF |
| `pow(x,y)` | Math Intrinsics | x raised to power y |
| `rintf(x)` | Math Intrinsics | Round to nearest integer (in float) |
| `roundf(x)` | Math Intrinsics | Round to nearest integer, halfway away from zero |
| `tanh(x)` | Math Intrinsics | Hyperbolic tangent |
| `truncf(x)` | Math Intrinsics | Truncate to integer part |
| `cublas<t>copy()` | cuBLAS | y = x (vector copy) |
| `cublas<t>dgmm()` | cuBLAS | C = A * diag(x) or diag(x) * A |
| `cublas<t>geam()` | cuBLAS | C = alpha*op(A) + beta*op(B) (matrix add/transpose) |
| `cublasScalEx()` | cuBLAS | Vector scale with mixed precision |
