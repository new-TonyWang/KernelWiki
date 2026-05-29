---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Best Practices Guide, Section 12.1 (Arithmetic Instructions)"
  - "Best Practices Guide, Section 12.1.1 (Throughput of Native Arithmetic Instructions)"
  - "Best Practices Guide, Section 12.1.9 (Math Libraries)"
  - "Best Practices Guide, Section 12.1.10 (Precision-related Compiler Flags)"
  - "Programming Guide, Section 5.5.9 (Intrinsic Functions)"
  - "Programming Guide, Section 5.5.9.3 (--use_fast_math Effect)"
cross_ref:
  - "Best Practices Guide, Section 12.1.4 (Division Modulo Operations)"
  - "Best Practices Guide, Section 12.1.6 (Reciprocal Square Root)"
  - "Best Practices Guide, Section 12.1.8 (Exponentiation With Small Fractional Arguments)"
  - "Programming Guide, Section 5.5.1.5 (Fused Multiply-Add FMA)"
related_apis:
  - __sinf
  - __cosf
  - __expf
  - __logf
  - __log2f
  - __powf
  - __sincosf
  - tanhf  # Note: __tanhf does NOT exist; use tanhf() which maps to tanh.approx.f32 with --use_fast_math
  - __fdividef
  - __frsqrt_rn
  - rsqrtf
  - sin.approx.f32
  - cos.approx.f32
  - ex2.approx.f32
  - lg2.approx.f32
  - rcp.approx.f32
  - rsqrt.approx.f32
  - div.approx.f32
  - tanh.approx.f32
related_experience: []
unlocks:
  - "half-precision-math: reduced precision strategy combines with half-precision for maximum throughput"
  - "operator-fusion: fast-math intrinsics can be fused with FMA chains"
conflicts_with:
  - "tensor-core: TC has its own precision pipeline, fast-math intrinsics do not apply"
---

# Fast Math

## Skill 1: Selective Intrinsic Function Replacement
### When to Use
When profiling reveals that standard math functions (`sinf`, `cosf`, `expf`, `logf`) are throughput bottlenecks, and reduced accuracy is acceptable for the application.

### How to Apply
1. Replace `sinf(x)` with `__sinf(x)`, `cosf(x)` with `__cosf(x)`, etc.
2. Intrinsic functions map directly to hardware SFU (Special Function Unit) instructions.
3. SFU throughput is 16 operations per clock per SM (1/4 of FP32 add throughput).
4. Selectively replace only the functions in hot paths where accuracy loss is tolerable.

### Code Template
```cuda
__global__ void activation_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Fast tanh: maps to tanh.approx.f32 (single instruction)
        output[idx] = tanhf(x);  // maps to tanh.approx.f32 with --use_fast_math
    }
}
```

### Source
Best Practices Guide, Section 12.1.9 (Math Libraries); Programming Guide, Section 5.5.9.2 (Single-Precision-Only Intrinsic Functions)

## Skill 2: --use_fast_math Compiler Flag
### When to Use
When the entire kernel or compilation unit can tolerate reduced single-precision accuracy and denormal flushing. This is an all-or-nothing approach.

### How to Apply
1. Add `-use_fast_math` to nvcc command line.
2. This implicitly sets `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false`.
3. All `functionName()` calls are replaced with `__functionName()` equivalents.
4. Only affects single-precision; double-precision is unchanged.

### Code Template
```bash
# Compile with fast math enabled globally
nvcc -use_fast_math -o kernel kernel.cu

# Or set individual flags for finer control
nvcc -ftz=true -prec-div=false -prec-sqrt=false -o kernel kernel.cu
```

### Source
Best Practices Guide, Section 12.1.10 (Precision-related Compiler Flags); Programming Guide, Section 5.5.9.3 (--use_fast_math Effect)

## Skill 3: Reciprocal Square Root Instead of Division by sqrt
### When to Use
When computing `1.0f / sqrtf(x)`, which appears frequently in normalization (LayerNorm, BatchNorm, vector normalization).

### How to Apply
1. Call `rsqrtf(x)` explicitly instead of `1.0f / sqrtf(x)`.
2. The compiler may not always optimize `1.0f / sqrtf(x)` to `rsqrtf()` due to IEEE-754 compliance.
3. `rsqrtf` maps to `rsqrt.approx.f32` which runs on the SFU at 16 ops/clock.

### Code Template
```cuda
__device__ float normalize_vector(float x, float y, float z) {
    float inv_len = rsqrtf(x*x + y*y + z*z);  // 1 SFU instruction
    return x * inv_len;  // scale component
}

// For LayerNorm variance normalization
__device__ float layernorm_scale(float variance, float eps) {
    return rsqrtf(variance + eps);
}
```

### Source
Best Practices Guide, Section 12.1.6 (Reciprocal Square Root)

## Skill 4: Bitwise Shift for Power-of-Two Division and Modulo
### When to Use
When dividing or taking modulo by a power-of-two constant. Integer division is very slow (20+ cycles) compared to shift/AND (1 cycle).

### How to Apply
1. Replace `i / N` with `i >> log2(N)` when N is a power of 2.
2. Replace `i % N` with `i & (N - 1)` when N is a power of 2.
3. The compiler does this automatically for literal power-of-two constants, but not always for variables or non-literal expressions.

### Code Template
```cuda
// Slow: integer division (20+ cycles)
int block = idx / 32;
int lane  = idx % 32;

// Fast: bitwise shift and AND (1 cycle each)
int block = idx >> 5;     // idx / 32
int lane  = idx & 31;     // idx % 32
```

### Source
Best Practices Guide, Section 12.1.4 (Division Modulo Operations)

## Skill 5: Exponentiation with Specialized Functions
### When to Use
When computing `pow(x, y)` where y is a small integer, a fraction, or a base-2/base-10 exponentiation. `pow()` is very heavy (high register pressure, many instructions).

### How to Apply
1. For small integer powers, use explicit multiplication: `x*x` instead of `powf(x, 2.0f)`.
2. For base-2: use `exp2f(y)` instead of `powf(2.0f, y)`.
3. For base-10: use `exp10f(y)` instead of `powf(10.0f, y)`.
4. For cube root: use `cbrtf(x)` instead of `powf(x, 1.0f/3.0f)`.
5. For common fractional exponents, use the lookup table from the Best Practices Guide.

### Code Template
```cuda
__device__ float compute_powers(float x) {
    // Bad: heavy-weight pow
    // float r = powf(x, 0.25f);

    // Good: use rsqrt chain (from BP Guide Table 6)
    float r = rsqrtf(rsqrtf(x));  // x^(1/4)

    // Bad: powf(x, 2.0f)
    // Good: explicit multiply
    float sq = x * x;

    // Bad: powf(2.0f, y)
    // Good: specialized base-2 exponential
    float e2 = exp2f(3.5f);

    return r + sq + e2;
}
```

### Source
Best Practices Guide, Section 12.1.8 (Exponentiation With Small Fractional Arguments); Section 12.1.9 (Math Libraries)

## Skill 6: Use sincosf for Simultaneous Sine and Cosine
### When to Use
When both sine and cosine of the same argument are needed (e.g., rotation matrices, Fourier transforms).

### How to Apply
1. Replace separate `sinf(x)` and `cosf(x)` calls with `sincosf(x, &s, &c)`.
2. For fast variant: `__sincosf(x, &s, &c)`.
3. This computes both in a single pass, roughly the cost of one transcendental.

### Code Template
```cuda
__device__ void rotate_2d(float angle, float in_x, float in_y,
                           float *out_x, float *out_y) {
    float s, c;
    sincosf(angle, &s, &c);  // or __sincosf for fast variant
    *out_x = in_x * c - in_y * s;
    *out_y = in_x * s + in_y * c;
}
```

### Source
Best Practices Guide, Section 12.1.9 (Math Libraries)

## Skill 7: Use Float Literal Suffixes to Avoid Double Promotion
### When to Use
Always, when writing single-precision CUDA code. Double-precision constants cause implicit promotion and use the slow FP64 pipe.

### How to Apply
1. Append `f` suffix to all floating-point constants: `3.14159f` not `3.14159`.
2. This avoids the compiler inserting `cvt.f64.f32` and `cvt.f32.f64` conversion instructions.

### Code Template
```cuda
// Bad: 1.0 is double, causes promotion
float result = x * 1.0 + 0.5;

// Good: f suffix keeps everything in FP32
float result = x * 1.0f + 0.5f;
```

### Source
Best Practices Guide, Section 12.1.7 (Other Arithmetic Instructions)

## Cascading Opportunities (unlocks)
- **half-precision-math**: fast-math philosophy of trading precision for speed extends naturally to fp16/bf16.
- **operator-fusion**: fast intrinsics reduce instruction count, making fusion more effective.

## Conflicts
- **tensor-core**: TC operations use their own precision pipeline; fast-math intrinsics do not affect TC behavior.

## Principles
- SFU (Special Function Unit) provides approximate transcendental functions at 16 ops/clock/SM -- 4x slower than FP32 add/mul.
- `--use_fast_math` is a sledgehammer: it affects ALL single-precision math in the compilation unit. Prefer selective intrinsic replacement for precision-critical code.
- FP64 throughput is typically 1/32 of FP32 on consumer GPUs. Always use single-precision with `f` suffixes unless double is truly needed.
- The FMA operation (`a*b+c`) is a single instruction with one rounding step; prefer it over separate multiply and add.

## Open Questions
- What is the ULP error budget for `tanhf` (fast path) in transformer attention softmax pipelines?
- How does `-ftz=true` affect convergence in iterative algorithms (e.g., Newton-Raphson)?
