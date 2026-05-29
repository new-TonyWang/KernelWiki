# Compiler Hints -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__dp4a(a,b,c)` | Math Intrinsics | 4-way int8 dot product with int32 accumulate |
| `__fadd_rn(x,y)` | Math Intrinsics | Add, round-to-nearest-even |
| `__float_as_int(x)` | Math Intrinsics | Reinterpret float bits as int |
| `__float_as_uint(x)` | Math Intrinsics | Reinterpret float bits as unsigned int |
| `__fmaf_rn(x,y,z)` | Math Intrinsics | FMA, round-to-nearest-even |
| `__fmul_rn(x,y)` | Math Intrinsics | Multiply, round-to-nearest-even |
| `__frsqrt_rn(x)` | Math Intrinsics | 1/sqrt(x) in round-to-nearest-even mode |
| `__fsub_rn(x,y)` | Math Intrinsics | Subtract, round-to-nearest-even |
| `__int_as_float(x)` | Math Intrinsics | Reinterpret int bits as float |
| `__uint_as_float(x)` | Math Intrinsics | Reinterpret unsigned int bits as float |
| `cudaFuncSetAttribute` | Runtime API | Set attributes for a function (max dynamic shared mem, cluster dims, shared mem carveout) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `__byte_perm(x,y,s)` | Math Intrinsics | Byte permutation from two 32-bit values |
| `__clz(x)` | Math Intrinsics | Count leading zeros (32-bit) |
| `__clzll(x)` | Math Intrinsics | Count leading zeros (64-bit) |
| `__dadd_rn(x,y)` | Math Intrinsics | Add, round-to-nearest-even |
| `__ddiv_rn(x,y)` | Math Intrinsics | Divide, round-to-nearest-even |
| `__dmul_rn(x,y)` | Math Intrinsics | Multiply, round-to-nearest-even |
| `__double2float_rn(x)` | Math Intrinsics | double -> float, round-to-nearest |
| `__double_as_longlong(x)` | Math Intrinsics | Reinterpret double bits as long long |
| `__dp2a_hi(a,b,c)` | Math Intrinsics | 2-way int16*int8 dot product (high half) |
| `__dp2a_lo(a,b,c)` | Math Intrinsics | 2-way int16*int8 dot product (low half) |
| `__drcp_rn(x)` | Math Intrinsics | Reciprocal, round-to-nearest-even |
| `__dsub_rn(x,y)` | Math Intrinsics | Subtract, round-to-nearest-even |
| `__fadd2_rn(x,y)` | Math Intrinsics | Vector add (float2), round-to-nearest |
| `__fadd_rd(x,y)` | Math Intrinsics | Add, round-down |
| `__fadd_ru(x,y)` | Math Intrinsics | Add, round-up |
| `__fadd_rz(x,y)` | Math Intrinsics | Add, round-towards-zero |
| `__fdiv_rd(x,y)` | Math Intrinsics | Divide, round-down |
| `__fdiv_rn(x,y)` | Math Intrinsics | Divide, round-to-nearest-even |
| `__fdiv_ru(x,y)` | Math Intrinsics | Divide, round-up |
| `__fdiv_rz(x,y)` | Math Intrinsics | Divide, round-towards-zero |
| `__ffma2_rn(x,y,z)` | Math Intrinsics | Vector FMA (float2), round-to-nearest |
| `__ffs(x)` | Math Intrinsics | Find first set bit (32-bit) |
| `__ffsll(x)` | Math Intrinsics | Find first set bit (64-bit) |
| `__float2bfloat16_rd(x)` | Math Intrinsics | float -> bfloat16, round-down |
| `__float2bfloat16_ru(x)` | Math Intrinsics | float -> bfloat16, round-up |
| `__float2bfloat16_rz(x)` | Math Intrinsics | float -> bfloat16, round-towards-zero |
| `__float2half_rd(x)` | Math Intrinsics | float -> half, round-down |
| `__float2half_ru(x)` | Math Intrinsics | float -> half, round-up |
| `__float2half_rz(x)` | Math Intrinsics | float -> half, round-towards-zero |
| `__float2int_rn(x)` | Math Intrinsics | float -> int, round-to-nearest |
| `__float2uint_rn(x)` | Math Intrinsics | float -> unsigned int, round-to-nearest |
| `__fma_rn(x,y,z)` | Math Intrinsics | FMA, round-to-nearest-even |
| `__fmaf_ieee_rn(x,y,z)` | Math Intrinsics | FMA, round-to-nearest, ignore -ftz |
| `__fmaf_rd(x,y,z)` | Math Intrinsics | FMA, round-down |
| `__fmaf_ru(x,y,z)` | Math Intrinsics | FMA, round-up |
| `__fmaf_rz(x,y,z)` | Math Intrinsics | FMA, round-towards-zero |
| `__fmul2_rn(x,y)` | Math Intrinsics | Vector multiply (float2), round-to-nearest |
| `__fmul_rd(x,y)` | Math Intrinsics | Multiply, round-down |
| `__fmul_ru(x,y)` | Math Intrinsics | Multiply, round-up |
| `__fmul_rz(x,y)` | Math Intrinsics | Multiply, round-towards-zero |
| `__frcp_rd(x)` | Math Intrinsics | Reciprocal, round-down |
| `__frcp_rn(x)` | Math Intrinsics | Reciprocal 1/x, round-to-nearest-even |
| `__frcp_ru(x)` | Math Intrinsics | Reciprocal, round-up |
| `__frcp_rz(x)` | Math Intrinsics | Reciprocal, round-towards-zero |
| `__fsqrt_rd(x)` | Math Intrinsics | sqrt in round-down mode |
| `__fsqrt_rn(x)` | Math Intrinsics | sqrt in round-to-nearest-even mode |
| `__fsqrt_ru(x)` | Math Intrinsics | sqrt in round-up mode |
| `__fsqrt_rz(x)` | Math Intrinsics | sqrt in round-towards-zero mode |
| `__fsub_rd(x,y)` | Math Intrinsics | Subtract, round-down |
| `__fsub_ru(x,y)` | Math Intrinsics | Subtract, round-up |
| `__fsub_rz(x,y)` | Math Intrinsics | Subtract, round-towards-zero |
| `__funnelshift_l(lo,hi,shift)` | Math Intrinsics | Funnel shift left |
| `__funnelshift_r(lo,hi,shift)` | Math Intrinsics | Funnel shift right |
| `__int2float_rn(x)` | Math Intrinsics | int -> float, round-to-nearest |
| `__longlong_as_double(x)` | Math Intrinsics | Reinterpret long long bits as double |
| `__popc(x)` | Math Intrinsics | Population count / bit count (32-bit) |
| `__popcll(x)` | Math Intrinsics | Population count (64-bit) |
| `__uint2float_rn(x)` | Math Intrinsics | unsigned int -> float |
