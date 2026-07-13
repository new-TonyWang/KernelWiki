# Half Precision Math -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__bfloat1622float2(x)` | Math Intrinsics | bfloat162 -> float2 |
| `__bfloat162float(x)` | Math Intrinsics | bfloat16 -> float |
| `__float2bfloat16(x)` | Math Intrinsics | float -> bfloat16 (round-to-nearest) |
| `__float2bfloat162_rn(x)` | Math Intrinsics | float -> bfloat162 (both = x) |
| `__float2bfloat16_rn(x)` | Math Intrinsics | float -> bfloat16, round-to-nearest |
| `__float2half(x)` | Math Intrinsics | float -> half (round-to-nearest) |
| `__float2half2_rn(x)` | Math Intrinsics | float -> half2 (both components = x) |
| `__float2half_rn(x)` | Math Intrinsics | float -> half, round-to-nearest-even |
| `__floats2bfloat162_rn(x,y)` | Math Intrinsics | Two floats -> bfloat162 |
| `__floats2half2_rn(x,y)` | Math Intrinsics | Two floats -> half2 |
| `__hadd(a,b)` | Math Intrinsics | Add two half values |
| `__hadd2(a,b)` | Math Intrinsics | Packed add (two half values at once) |
| `__half22float2(x)` | Math Intrinsics | half2 -> float2 |
| `__half2float(x)` | Math Intrinsics | half -> float |
| `__halves2bfloat162(a,b)` | Math Intrinsics | Two bfloat16 -> bfloat162 |
| `__halves2half2(a,b)` | Math Intrinsics | Two halves -> half2 |
| `__hfma(a,b,c)` | Math Intrinsics | Fused multiply-add a*b+c |
| `__hfma2(a,b,c)` | Math Intrinsics | Packed fused multiply-add |
| `__hfma2_relu(a,b,c)` | Math Intrinsics | Packed FMA with ReLU fusion |
| `__hfma_relu(a,b,c)` | Math Intrinsics | FMA with ReLU fusion |
| `__hmax(a,b)` | Math Intrinsics | Maximum |
| `__hmax2(a,b)` | Math Intrinsics | Packed maximum |
| `__hmul(a,b)` | Math Intrinsics | Multiply two half values |
| `__hmul2(a,b)` | Math Intrinsics | Packed multiply |
| `__hsub(a,b)` | Math Intrinsics | Subtract two half values |
| `__hsub2(a,b)` | Math Intrinsics | Packed subtract |
| `h2exp(__half2)` | Math Intrinsics | Fast variant of hexp(__half) |
| `h2exp(__nv_bfloat162)` | Math Intrinsics | Fast variant of hexp(__nv_bfloat16) |
| `h2log(__half2)` | Math Intrinsics | Fast variant of hlog(__half) |
| `h2log(__nv_bfloat162)` | Math Intrinsics | Fast variant of hlog(__nv_bfloat16) |
| `h2rsqrt(__half2)` | Math Intrinsics | Fast variant of hrsqrt(__half) |
| `h2rsqrt(__nv_bfloat162)` | Math Intrinsics | Fast variant of hrsqrt(__nv_bfloat16) |
| `h2sqrt(__half2)` | Math Intrinsics | Fast variant of hsqrt(__half) |
| `h2sqrt(__nv_bfloat162)` | Math Intrinsics | Fast variant of hsqrt(__nv_bfloat16) |
| `hexp(__half)` | Math Intrinsics | Base-e exponential |
| `hexp(__nv_bfloat16)` | Math Intrinsics | Base-e exponential |
| `hlog(__half)` | Math Intrinsics | Natural logarithm |
| `hlog(__nv_bfloat16)` | Math Intrinsics | Natural logarithm |
| `hrsqrt(__half)` | Math Intrinsics | Reciprocal square root |
| `hrsqrt(__nv_bfloat16)` | Math Intrinsics | Reciprocal square root |
| `hsqrt(__half)` | Math Intrinsics | Square root |
| `hsqrt(__nv_bfloat16)` | Math Intrinsics | Square root |
| `add[.rnd].bf16 / .bf16x2` | PTX ISA | BF16 add; SIMD 2-wide |
| `add[.rnd][.ftz][.sat].f16 / .f16x2` | PTX ISA | FP16 add; SIMD 2-wide on .f16x2 |
| `atom[.sem][.scope].add.noftz.f16/bf16` | PTX ISA | Atomic FP16/BF16 add |
| `cvt[.rnd][.sat].dtype.stype` | PTX ISA | Type conversion (int/float, narrowing/widening, rounding modes) |
| `fma.rnd[.ftz][.sat][.relu].f16 / .f16x2` | PTX ISA | FP16 fused multiply-add; relu/oob variants |
| `fma.rnd[.relu].bf16 / .bf16x2` | PTX ISA | BF16 fused multiply-add; .oob variant (sm_90+) |
| `fma.rnd[.sat].f32.{f16/bf16}` | PTX ISA | Mixed-precision fma: f16/bf16 inputs -> f32 output |
| `mul[.rnd].bf16 / .bf16x2` | PTX ISA | BF16 multiply; SIMD 2-wide |
| `mul[.rnd][.ftz][.sat].f16 / .f16x2` | PTX ISA | FP16 multiply; SIMD 2-wide |
| `sub[.rnd].bf16 / .bf16x2` | PTX ISA | BF16 subtract; SIMD 2-wide |
| `sub[.rnd][.ftz][.sat].f16 / .f16x2` | PTX ISA | FP16 subtract; SIMD 2-wide |
| `tanh.approx .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision approximate tanh |
| `cublas<t>gemmEx()` | cuBLAS | GEMM with lower-precision inputs, higher-precision compute |
| `cublasAxpyEx()` | cuBLAS | y = alpha*x + y with mixed precision |
| `cublasDotEx()` | cuBLAS | Dot product with mixed precision |
| `cublasGemmEx()` | cuBLAS | Fully flexible GEMM: individual data types + algorithm selection |
| `cublasHgemm()` | cuBLAS | Half-precision GEMM (included in above) |
| `cublasLtMatmul()` | cuBLAS | D = alpha*op(A)*op(B) + beta*C with epilogue fusion |
| `cublasNrm2Ex()` | cuBLAS | Euclidean norm with mixed precision |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__bfloat162half(x)` | Math Intrinsics | bfloat16 -> half |
| `__h2div(a,b)` | Math Intrinsics | Packed divide |
| `__habs(a)` | Math Intrinsics | Absolute value |
| `__habs2(a)` | Math Intrinsics | Packed absolute value |
| `__half2bfloat16(x)` | Math Intrinsics | half -> bfloat16 |
| `__hdiv(a,b)` | Math Intrinsics | Divide two half values |
| `__heq(a,b)` | Math Intrinsics | Equal comparison |
| `__heq2(a,b)` | Math Intrinsics | Fast variant of __heq(a,b) |
| `__hfma2_sat(a,b,c)` | Math Intrinsics | Packed FMA with saturation |
| `__hfma_sat(a,b,c)` | Math Intrinsics | FMA with saturation to [0,1] |
| `__hge(a,b)` | Math Intrinsics | Greater-or-equal comparison |
| `__hge2(a,b)` | Math Intrinsics | Fast variant of __hge(a,b) |
| `__hgt(a,b)` | Math Intrinsics | Greater-than comparison |
| `__hgt2(a,b)` | Math Intrinsics | Fast variant of __hgt(a,b) |
| `__high2bfloat16(x)` | Math Intrinsics | Extract high from bfloat162 |
| `__high2bfloat162(x)` | Math Intrinsics | Replicate high to bfloat162 |
| `__high2half(x)` | Math Intrinsics | Extract high half from half2 |
| `__high2half2(x)` | Math Intrinsics | Replicate high half to half2 |
| `__hle(a,b)` | Math Intrinsics | Less-or-equal comparison |
| `__hle2(a,b)` | Math Intrinsics | Fast variant of __hle(a,b) |
| `__hlt(a,b)` | Math Intrinsics | Less-than comparison |
| `__hlt2(a,b)` | Math Intrinsics | Fast variant of __hlt(a,b) |
| `__hmin(a,b)` | Math Intrinsics | Minimum |
| `__hmin2(a,b)` | Math Intrinsics | Packed minimum |
| `__hne(a,b)` | Math Intrinsics | Not-equal comparison |
| `__hne2(a,b)` | Math Intrinsics | Fast variant of __hne(a,b) |
| `__hneg(a)` | Math Intrinsics | Negate |
| `__hneg2(a)` | Math Intrinsics | Packed negate |
| `__low2bfloat16(x)` | Math Intrinsics | Extract low from bfloat162 |
| `__low2bfloat162(x)` | Math Intrinsics | Replicate low to bfloat162 |
| `__low2half(x)` | Math Intrinsics | Extract low half from half2 |
| `__low2half2(x)` | Math Intrinsics | Replicate low half to half2 |
| `__lowhigh2highlow(x)` | Math Intrinsics | Swap low and high in half2 |
| `h2cos(__half2)` | Math Intrinsics | Fast variant of hcos(__half) |
| `h2cos(__nv_bfloat162)` | Math Intrinsics | Fast variant of hcos(__nv_bfloat16) |
| `h2exp10(__half2)` | Math Intrinsics | Fast variant of hexp10(__half) |
| `h2exp10(__nv_bfloat162)` | Math Intrinsics | Fast variant of hexp10(__nv_bfloat16) |
| `h2exp2(__half2)` | Math Intrinsics | Fast variant of hexp2(__half) |
| `h2exp2(__nv_bfloat162)` | Math Intrinsics | Fast variant of hexp2(__nv_bfloat16) |
| `h2log10(__half2)` | Math Intrinsics | Fast variant of hlog10(__half) |
| `h2log10(__nv_bfloat162)` | Math Intrinsics | Fast variant of hlog10(__nv_bfloat16) |
| `h2log2(__half2)` | Math Intrinsics | Fast variant of hlog2(__half) |
| `h2log2(__nv_bfloat162)` | Math Intrinsics | Fast variant of hlog2(__nv_bfloat16) |
| `h2rcp(__half2)` | Math Intrinsics | Fast variant of hrcp(__half) |
| `h2rcp(__nv_bfloat162)` | Math Intrinsics | Fast variant of hrcp(__nv_bfloat16) |
| `h2sin(__half2)` | Math Intrinsics | Fast variant of hsin(__half) |
| `h2sin(__nv_bfloat162)` | Math Intrinsics | Fast variant of hsin(__nv_bfloat16) |
| `hcos(__half)` | Math Intrinsics | Cosine |
| `hcos(__nv_bfloat16)` | Math Intrinsics | Cosine |
| `hexp10(__half)` | Math Intrinsics | Base-10 exponential |
| `hexp10(__nv_bfloat16)` | Math Intrinsics | Base-10 exponential |
| `hexp2(__half)` | Math Intrinsics | Base-2 exponential |
| `hexp2(__nv_bfloat16)` | Math Intrinsics | Base-2 exponential |
| `hlog10(__half)` | Math Intrinsics | Base-10 logarithm |
| `hlog10(__nv_bfloat16)` | Math Intrinsics | Base-10 logarithm |
| `hlog2(__half)` | Math Intrinsics | Base-2 logarithm |
| `hlog2(__nv_bfloat16)` | Math Intrinsics | Base-2 logarithm |
| `hrcp(__half)` | Math Intrinsics | Reciprocal 1/x |
| `hrcp(__nv_bfloat16)` | Math Intrinsics | Reciprocal |
| `hsin(__half)` | Math Intrinsics | Sine |
| `hsin(__nv_bfloat16)` | Math Intrinsics | Sine |
| `add.rnd[.sat].f32.{f16/bf16}` | PTX ISA | Mixed-precision add: f16/bf16 input -> f32 output |
| `cvt.pack[.type]` | PTX ISA | Pack multiple values into a register |
| `ex2.approx .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision approximate 2^x |
| `min/max [.NaN][.xorsign.abs] .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision min/max with NaN handling |
| `neg/abs .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision negate/absolute value |
| `set.CmpOp .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision typed comparison |
| `setp.CmpOp .f16/.f16x2/.bf16/.bf16x2` | PTX ISA | Half-precision predicate comparison |
| `sub.rnd[.sat].f32.{f16/bf16}` | PTX ISA | Mixed-precision sub: f16/bf16 input -> f32 output |
| `cublasScalEx()` | cuBLAS | Vector scale with mixed precision |
