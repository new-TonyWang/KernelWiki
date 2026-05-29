---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Best Practices Guide, Section 7.3.1 (Single vs. Double Precision)"
  - "Best Practices Guide, Section 12.1.1 (Throughput of Native Arithmetic Instructions)"
  - "Programming Guide, Section 5.4.11.2 (Alternate Floating Point)"
  - "Programming Guide, Section 5.5.2 (Floating-Point Data Types)"
  - "PTX ISA, f16/f16x2/bf16/bf16x2 arithmetic instructions"
cross_ref:
  - "Programming Guide, Section 5.4.11 (Warp Matrix Functions)"
  - "Best Practices Guide, Section 12.1 (Arithmetic Instructions)"
related_apis:
  - __hadd
  - __hadd2
  - __hmul
  - __hmul2
  - __hfma
  - __hfma2
  - __hfma2_relu
  - __float2half_rn
  - __half2float
  - __float2bfloat16_rn
  - __bfloat162float
  - __halves2half2
  - __floats2half2_rn
  - hexp
  - hlog
  - hrsqrt
  - h2exp
  - h2rsqrt
  - add.f16x2
  - fma.rnd.f16x2
  - fma.rnd.bf16x2
  - tanh.approx.f16
  - atom.add.noftz.f16
related_experience: []
unlocks:
  - "tensor-core: fp16/bf16 are the native input types for Tensor Cores"
  - "vectorized-access: half2 packing doubles effective memory bandwidth"
  - "fast-math: half-precision intrinsics provide fast approximate math at fp16 level"
conflicts_with:
  - "fast-math: __use_fast_math only affects fp32 intrinsics, not fp16/bf16"
---

# Half Precision Math

## Skill 1: Packed Half2 Arithmetic for 2x Throughput
### When to Use
When performing element-wise operations on fp16 data. Using `__half2` (packed pair) doubles throughput because a single SIMD instruction processes two fp16 values simultaneously.

### How to Apply
1. Pack two `__half` values into a `__half2` using `__halves2half2(a, b)` or `__floats2half2_rn(a, b)`.
2. Use `__hadd2`, `__hmul2`, `__hfma2` for packed arithmetic.
3. FP16 SIMD throughput: 128 ops/clock/SM on sm_80 (2x the scalar fp16 throughput).

### Code Template
```cuda
#include <cuda_fp16.h>

__global__ void half2_add(const half2* __restrict__ a,
                           const half2* __restrict__ b,
                           half2* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        c[idx] = __hadd2(a[idx], b[idx]);  // 2 fp16 adds in 1 instruction
    }
}
```

### Source
Best Practices Guide, Section 12.1.1 (Throughput of Native Arithmetic Instructions) -- "16-bit floating-point add, multiply, multiply-add (2-way SIMD): add.f16x2"

## Skill 2: BF16 for Training-Friendly Half Precision
### When to Use
When working with deep learning training where the fp32 dynamic range is needed but fp32 precision is not. BF16 has the same exponent range as fp32 (8 bits) with reduced mantissa (7 bits).

### How to Apply
1. Include `<cuda_bf16.h>` and use `__nv_bfloat16` / `__nv_bfloat162` types.
2. Convert with `__float2bfloat16_rn()` and `__bfloat162float()`.
3. Use packed `__nv_bfloat162` for 2x throughput SIMD operations.
4. Available on sm_80+.

### Code Template
```cuda
#include <cuda_bf16.h>

__global__ void bf16_scale(const __nv_bfloat162* __restrict__ input,
                            __nv_bfloat162* __restrict__ output,
                            float scale_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        __nv_bfloat162 val = input[idx];
        __nv_bfloat162 s = __float2bfloat162_rn(scale_val);
        output[idx] = __hmul2(val, s);  // Packed bf16 multiply
    }
}
```

### Source
Programming Guide, Section 5.4.11.2 (Alternate Floating Point); Section 5.5.2 (Floating-Point Data Types)

## Skill 3: Mixed-Precision Accumulation Pattern
### When to Use
When computation involves fp16/bf16 inputs but requires fp32 accumulation for numerical stability (e.g., reduction, dot product, normalization).

### How to Apply
1. Load fp16/bf16 data.
2. Convert to fp32 for accumulation using `__half2float()` or `__bfloat162float()`.
3. Accumulate in fp32.
4. Convert result back to fp16/bf16 for storage.
5. For FMA, use `fma.rnd.f32.f16` (mixed-precision FMA) to fuse the conversion.

### Code Template
```cuda
#include <cuda_fp16.h>

__global__ void mixed_precision_dot(const half* __restrict__ a,
                                     const half* __restrict__ b,
                                     float* __restrict__ result, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        sum = fmaf(fa, fb, sum);  // FP32 accumulation
    }

    // Warp reduction in fp32
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (threadIdx.x % 32 == 0)
        atomicAdd(result, sum);
}
```

### Source
Programming Guide, Section 5.5.2 (Floating-Point Data Types); Best Practices Guide, Section 7.3.1 (Single vs. Double Precision)

## Skill 4: Half-Precision Transcendental Functions
### When to Use
When computing transcendental functions (exp, log, rsqrt, tanh) on fp16/bf16 data. Half-precision intrinsics are faster and avoid fp32 conversion overhead.

### How to Apply
1. Use `hexp()`, `hlog()`, `hrsqrt()`, `hsqrt()` for scalar fp16.
2. Use `h2exp()`, `h2log()`, `h2rsqrt()`, `h2sqrt()` for packed fp16x2.
3. For tanh, PTX provides `tanh.approx.f16` / `tanh.approx.bf16` as single instructions.

### Code Template
```cuda
#include <cuda_fp16.h>

__global__ void half_softmax_exp(const half2* __restrict__ input,
                                  half2* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        half2 val = input[idx];
        output[idx] = h2exp(val);  // Packed fp16 exp, 2 results per instruction
    }
}
```

### Source
PTX ISA, tanh.approx.f16/bf16 instructions; CUDA Math API cuda_fp16.h

## Skill 5: FP16 ReLU Fusion via hfma_relu
### When to Use
When applying FMA followed by ReLU activation on fp16 data. The fused variant `__hfma_relu` / `__hfma2_relu` combines both operations into a single instruction.

### How to Apply
1. Use `__hfma_relu(a, b, c)` for scalar: `max(a*b+c, 0)`.
2. Use `__hfma2_relu(a, b, c)` for packed half2.
3. Maps to PTX `fma.rnd.relu.f16` instruction.

### Code Template
```cuda
#include <cuda_fp16.h>

__global__ void fused_linear_relu(const half2* __restrict__ input,
                                   const half2* __restrict__ weight,
                                   const half2* __restrict__ bias,
                                   half2* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 2) {
        half2 a = input[idx];
        half2 w = weight[idx];
        half2 b = bias[idx];
        // Fused: max(a*w + b, 0) in one instruction
        output[idx] = __hfma2_relu(a, w, b);
    }
}
```

### Source
PTX ISA, fma.rnd.relu.f16/f16x2 instruction

## Skill 6: Atomic FP16/BF16 Addition
### When to Use
When performing reductions to fp16/bf16 output using atomics, available on sm_70+ (fp16) and sm_80+ (bf16).

### How to Apply
1. Use `atomicAdd(half_ptr, half_val)` -- supported natively.
2. Maps to PTX `atom.add.noftz.f16`.
3. The `.noftz` qualifier means denormals are not flushed to zero during the atomic.

### Code Template
```cuda
#include <cuda_fp16.h>

__global__ void fp16_atomic_reduce(const half* __restrict__ input,
                                    half* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&output[0], input[idx]);  // Native fp16 atomic add
    }
}
```

### Source
PTX ISA, atom.add.noftz.f16/bf16 instruction

## Cascading Opportunities (unlocks)
- **tensor-core**: fp16/bf16 are the primary input types for Tensor Cores; half-precision math enables TC utilization.
- **vectorized-access**: `half2` naturally aligns to 4 bytes, enabling efficient 32-bit loads/stores per pair.
- **fast-math**: half-precision intrinsics provide approximate math at fp16 level, complementing fp32 fast-math.

## Conflicts
- **fast-math**: `--use_fast_math` only affects fp32 functions; it does not map half-precision functions to faster variants.

## Principles
- FP16 SIMD (f16x2) throughput matches FP32 throughput in ops/clock on most architectures, but processes 2 values per instruction, yielding 2x effective throughput.
- BF16 has the same dynamic range as FP32 (exponent: 8 bits) but reduced precision (mantissa: 7 bits). Choose BF16 for training and FP16 for inference.
- Always accumulate reductions in FP32 to avoid catastrophic cancellation in half-precision.
- Memory bandwidth is halved for fp16 vs fp32, making half-precision doubly beneficial for memory-bound kernels.

## Open Questions
- What is the optimal strategy for mixed fp16/fp32 in LayerNorm: accumulate in fp32 for variance only, or for both mean and variance?
- How does FP8 (E4M3/E5M2) on Hopper/Blackwell compare to fp16/bf16 for inference throughput and accuracy?
