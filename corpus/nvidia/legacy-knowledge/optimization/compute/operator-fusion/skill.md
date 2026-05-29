---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Best Practices Guide, Section 12.1 (Arithmetic Instructions)"
  - "Programming Guide, Section 5.5.1.5 (Fused Multiply-Add FMA)"
  - "PTX ISA, fma.rnd.f32 / fma.rnd.f16 / fma.rnd.bf16 / lop3.b32 instructions"
cross_ref:
  - "Best Practices Guide, Section 12.1.9 (Math Libraries)"
  - "Programming Guide, Section 5.5.9.1 (Basic Intrinsic Functions)"
related_apis:
  - fma.rnd.f32
  - fma.rnd.f64
  - fma.rnd.f16
  - fma.rnd.f16x2
  - fma.rnd.bf16
  - fma.rnd.bf16x2
  - fma.rnd.f32.f16
  - fma.rnd.f32.bf16
  - lop3.b32
  - mad
related_experience: []
unlocks:
  - "tensor-core: epilogue fusion in TC accumulator avoids extra store/load"
  - "half-precision-math: mixed-precision FMA (f16 inputs -> f32 output) fuses type conversion with arithmetic"
  - "instruction-level-parallelism: fewer total instructions from fusion means more ILP headroom"
conflicts_with:
  - "compiler-hints: __fadd_rn / __fmul_rn explicitly prevent FMA fusion"
---

# Operator Fusion

## Skill 1: Fused Multiply-Add (FMA) for Precision and Throughput
### When to Use
Always, for `a*b+c` patterns. FMA is a single instruction with one rounding step (more accurate) and the same throughput as a simple add or multiply.

### How to Apply
1. Write `a*b + c` in your code; the compiler generates FMA by default when `-fmad=true` (the default).
2. For explicit control, use `fmaf(a, b, c)` or the intrinsic `__fmaf_rn(a, b, c)`.
3. The PTX instruction `fma.rn.f32` performs the operation with round-to-nearest-even.

### Code Template
```cuda
__global__ void fma_example(const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ c,
                             float* __restrict__ d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compiler generates a single FMA instruction
        d[idx] = a[idx] * b[idx] + c[idx];

        // Or explicitly:
        // d[idx] = fmaf(a[idx], b[idx], c[idx]);
    }
}
```

### Source
Programming Guide, Section 5.5.1.5 (Fused Multiply-Add FMA)

## Skill 2: Mixed-Precision FMA for Half-to-Float Accumulation
### When to Use
When inputs are fp16 or bf16 but accumulation must be in fp32 for numerical stability. The mixed-precision FMA does the type widening and arithmetic in one instruction.

### How to Apply
1. Use PTX `fma.rnd.f32.f16` or `fma.rnd.f32.bf16` via inline assembly.
2. Inputs are fp16/bf16, output is fp32, with a single rounding step.
3. This avoids separate `cvt` + `fma.f32` instruction sequences.

### Code Template
```cuda
__device__ float mixed_fma_f16_f32(uint16_t a_f16, uint16_t b_f16, float c_f32) {
    float result;
    asm volatile(
        "fma.rn.f32.f16 %0, %1, %2, %3;\n"
        : "=f"(result)
        : "h"(a_f16), "h"(b_f16), "f"(c_f32)
    );
    return result;
}
```

### Source
PTX ISA, fma.rnd.f32.{f16/bf16} instruction

## Skill 3: Kernel Fusion to Eliminate Intermediate Global Memory Traffic
### When to Use
When two successive operations (e.g., GEMM + bias + activation) would each be separate kernels requiring a global memory round-trip for intermediate results.

### How to Apply
1. Combine the operations into a single kernel.
2. Compute the intermediate result in registers or shared memory.
3. Apply the subsequent operation before writing to global memory.
4. This eliminates one global memory write + read cycle per fused operation.

### Code Template
```cuda
// Fused: GEMM output + bias + ReLU in a single kernel
__global__ void gemm_bias_relu(const float* __restrict__ gemm_out,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = gemm_out[row * N + col];
        val += bias[col];                // bias addition
        val = fmaxf(val, 0.0f);          // ReLU activation
        output[row * N + col] = val;
    }
}
```

### Source
Best Practices Guide, Section 12.1 (Arithmetic Instructions) -- minimize memory instructions; general optimization principle

## Skill 4: lop3.b32 for Multi-Operand Logical Fusion
### When to Use
When you need to compute a 3-input logical operation (AND, OR, XOR combinations) in a single instruction. Useful in bitwise processing, hash functions, or predicate manipulation.

### How to Apply
1. `lop3.b32` takes three 32-bit inputs and a lookup table (LUT) encoding the desired logical function.
2. The LUT byte encodes all 256 possible 3-input Boolean functions.
3. Use inline PTX to issue the instruction.

### Code Template
```cuda
// lop3: result = (a & b) | (b & c) -- encoded as LUT 0xE8
__device__ uint32_t majority3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t result;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xE8;\n"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c)
    );
    return result;
}
```

### Source
PTX ISA, lop3.b32 instruction

## Skill 5: Epilogue Fusion with WMMA Fragment Access
### When to Use
When the output of a Tensor Core MMA needs an element-wise epilogue (scale, bias, activation) before writing to global memory.

### How to Apply
1. After `wmma::mma_sync`, access accumulator elements via `frag.x[i]`.
2. Apply the epilogue in a loop over `frag.num_elements`.
3. This keeps the data in registers, avoiding a shared/global memory store-load pair.

### Code Template
```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
// ... mma_sync ...

// Fused epilogue: bias + GELU approximation
float bias_val = bias[col];
for (int i = 0; i < c_frag.num_elements; i++) {
    float x = c_frag.x[i] + bias_val;
    // Fast GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    c_frag.x[i] = x * 0.5f * (1.0f + __tanhf(0.7978846f * (x + 0.044715f * x3)));
}

wmma::store_matrix_sync(out_ptr, c_frag, ldc, wmma::mem_row_major);
```

### Source
Programming Guide, Section 5.4.11.1 (Description) -- fragment element access; PTX ISA fma instructions

## Cascading Opportunities (unlocks)
- **tensor-core**: epilogue fusion avoids extra memory traffic after MMA.
- **half-precision-math**: mixed-precision FMA fuses precision conversion with arithmetic.
- **instruction-level-parallelism**: fewer instructions from fusion leaves more room for ILP.

## Conflicts
- **compiler-hints**: Using `__fadd_rn(a, b)` and `__fmul_rn(a, b)` explicitly prevents the compiler from fusing them into FMA. Use these only when you need to control rounding or prevent FMA for numerical reasons.

## Principles
- FMA is the fundamental fused instruction on NVIDIA GPUs. It is not only faster but more accurate than separate multiply+add.
- Kernel fusion is the most impactful form of operator fusion: it eliminates global memory round-trips that dominate latency in memory-bound kernels.
- The compiler fuses `a*b+c` into FMA by default (`-fmad=true`). To prevent this, use `-fmad=false` or explicit intrinsics.
- For cuBLAS, `cublasLtMatmul` supports epilogue fusion (bias, ReLU, GELU) natively via epilogue descriptors.

## Open Questions
- How to optimally fuse multi-operator chains (e.g., GEMM + LayerNorm + residual add) at the PTX level?
- What is the register pressure impact of in-register epilogues for large WMMA tile sizes?
