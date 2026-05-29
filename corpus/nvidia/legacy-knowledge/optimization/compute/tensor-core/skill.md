---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - "Programming Guide, Section 5.4.11 (Warp Matrix Functions)"
  - "Programming Guide, Section 5.4.11.2 (Alternate Floating Point)"
  - "PTX ISA, wmma / mma.sync / wgmma / tcgen05 instruction families"
cross_ref:
  - "Best Practices Guide, Section 12.1.1 (Throughput of Native Arithmetic Instructions)"
  - "Programming Guide, Section 3.2.2.3 (Asynchronous Execution Features)"
related_apis:
  - nvcuda::wmma::load_matrix_sync
  - nvcuda::wmma::mma_sync
  - nvcuda::wmma::store_matrix_sync
  - mma.sync.aligned
  - wgmma.mma_async.sync.aligned
  - tcgen05.mma
  - cublasGemmEx
  - cublasLtMatmul
related_experience: []
unlocks:
  - "half-precision-math: lower-precision inputs feed TC at higher throughput"
  - "data-prefetch: async copy to shared memory enables pipelined TC feeding"
  - "warp-specialization: producer-consumer pattern keeps TC pipeline full"
  - "shared-memory-cache: TC operands staged in shared memory"
conflicts_with:
  - "fast-math: TC has fixed precision pipeline; fast-math approximations are irrelevant for MMA ops"
  - "register-pressure: WMMA fragments consume many registers per thread"
---

# Tensor Core

## Skill 1: WMMA API for Portable Tensor Core Usage
### When to Use
When you need a simple, portable way to use Tensor Cores across sm_70+ architectures for matrix multiply-accumulate (D = A * B + C) on fp16, bf16, tf32, fp64, or integer types.

### How to Apply
1. Declare `wmma::fragment` objects for matrices A, B, and accumulator C/D.
2. Load operand tiles from global or shared memory using `load_matrix_sync`.
3. Call `mma_sync` to execute the MMA.
4. Store the result with `store_matrix_sync`.
5. All 32 threads in the warp must participate; call only within uniform control flow.

### Code Template
```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm_kernel(half *A, half *B, float *C,
                                  int M, int N, int K, int lda, int ldb, int ldc) {
    // Each warp computes a 16x16 output tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;
    int warpN = blockIdx.y * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warpM * lda + k, lda);
        wmma::load_matrix_sync(b_frag, B + k * ldb + warpN, ldb);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * ldc + warpN, c_frag, ldc, wmma::mem_row_major);
}
```

### Source
Programming Guide, Section 5.4.11.1 (Description) and Section 5.4.11.7 (Example)

## Skill 2: TF32 Tensor Core for FP32-Range Inputs
### When to Use
When input data is originally fp32, full fp32 precision is not required, but fp32 dynamic range is needed. Available on sm_80+.

### How to Apply
1. Convert fp32 inputs to tf32 precision using `__float_to_tf32()`.
2. Use `wmma::fragment` with `wmma::precision::tf32` and `float` storage type.
3. Accumulator must be `float`. Supported tile size is 16x16x8.
4. This provides >=10 bits of mantissa precision with full fp32 range.

### Code Template
```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void tf32_mma_kernel(float *A, float *B, float *C, int lda, int ldb, int ldc) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B, ldb);

    // Convert elements to tf32 precision
    for (int i = 0; i < a_frag.num_elements; i++)
        a_frag.x[i] = __float_to_tf32(a_frag.x[i]);
    for (int i = 0; i < b_frag.num_elements; i++)
        b_frag.x[i] = __float_to_tf32(b_frag.x[i]);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
}
```

### Source
Programming Guide, Section 5.4.11.2 (Alternate Floating Point)

## Skill 3: PTX mma.sync for Fine-Grained Register Control
### When to Use
When WMMA abstraction is too coarse and you need explicit control over register-to-register MMA at the warp level (sm_75+).

### How to Apply
1. Use inline PTX `asm` to issue `mma.sync.aligned` instructions.
2. Map thread registers to the specific fragment layout for your target shape.
3. This gives direct control over register allocation and scheduling.

### Code Template
```cuda
__device__ void mma_m16n8k16_f16(float *d, const uint32_t *a, const uint32_t *b, const float *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
```

### Source
PTX ISA, mma.sync.aligned instruction

## Skill 4: cuBLAS Tensor Core Enablement via Math Mode
### When to Use
When using cuBLAS for GEMM and you want to ensure Tensor Cores are used. This is the easiest path for applications that use library-level GEMM.

### How to Apply
1. Set the math mode to `CUBLAS_TF32_TENSOR_OP_MATH` (for tf32) or `CUBLAS_DEFAULT_MATH` (auto-selection).
2. Use `cublasGemmEx()` or `cublasLtMatmul()` with appropriate data types.
3. For fp16 inputs with fp32 accumulate, use `CUDA_R_16F` compute type.

### Code Template
```cuda
cublasHandle_t handle;
cublasCreate(&handle);

// Enable TF32 Tensor Cores (default on Ampere+)
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

float alpha = 1.0f, beta = 0.0f;
cublasGemmEx(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             d_A, CUDA_R_16F, M,
             d_B, CUDA_R_16F, K,
             &beta,
             d_C, CUDA_R_32F, M,
             CUBLAS_COMPUTE_32F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

cublasDestroy(handle);
```

### Source
Programming Guide, Section 5.4.11.2 (Alternate Floating Point); cuBLAS documentation

## Skill 5: Fragment Element-Wise Operations for Epilogue Fusion
### When to Use
When you need to apply an element-wise operation (e.g., scaling, bias, activation) to the accumulator before storing results, avoiding an extra global memory round-trip.

### How to Apply
1. After `mma_sync`, iterate over `c_frag.x[]` array using `c_frag.num_elements`.
2. Apply the operation element-wise across all fragment elements.
3. Call `store_matrix_sync` to write results.

### Code Template
```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// ... perform mma_sync ...

// Apply ReLU + scale epilogue in registers
float scale = 0.5f;
for (int t = 0; t < c_frag.num_elements; t++) {
    c_frag.x[t] = fmaxf(c_frag.x[t] * scale, 0.0f);  // scale + ReLU
}

wmma::store_matrix_sync(C_out + row * ldc + col, c_frag, ldc, wmma::mem_row_major);
```

### Source
Programming Guide, Section 5.4.11.1 (Description) -- fragment element access

## Skill 6: Sparse MMA with Structured 2:4 Sparsity
### When to Use
When the weight matrix has been pruned to 2:4 structured sparsity pattern, doubling effective throughput on sm_80+.

### How to Apply
1. Prune weight matrix to 2:4 format: exactly 2 non-zero values per group of 4 elements.
2. Compress the sparse matrix and generate the metadata index.
3. Use `mma.sp.sync.aligned` PTX instruction with the metadata operand.

### Code Template
```cuda
// PTX inline assembly for sparse MMA (m16n8k32, fp16)
__device__ void sparse_mma_m16n8k32_f16(
    float *d, const uint32_t *a, const uint32_t *b,
    const float *c, const uint32_t *meta) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(meta[0])
    );
}
```

### Source
PTX ISA, mma.sp.sync.aligned instruction

## Cascading Opportunities (unlocks)
- **half-precision-math**: fp16/bf16 inputs are the primary data types for TC, enabling 2x throughput vs fp32.
- **data-prefetch**: async global-to-shared copies (cp.async, TMA) keep the TC pipeline fed without stalling.
- **warp-specialization**: producer warps prefetch data while consumer warps issue MMA instructions.
- **shared-memory-cache**: operand tiles are loaded into shared memory before being consumed by TC.
- **operator-fusion**: epilogue operations (bias, activation) can be fused into the TC accumulator before store.

## Conflicts
- **register-pressure**: WMMA fragments consume many registers (8-16 per fragment). Heavy TC usage can reduce occupancy. Use `__launch_bounds__` to manage.
- **fast-math**: approximate math functions have no effect on TC pipeline; TC precision is determined by input data type.

## Principles
- All 32 threads in a warp must participate in WMMA calls; conditional execution must be uniform across the warp.
- Fragment layout is opaque and architecture-specific; never pass fragments between separately compiled translation units for different architectures.
- Pointer alignment of 256 bits (32 bytes) is required for `load_matrix_sync` and `store_matrix_sync`.
- The `ldm` stride must be a multiple of 16 bytes.

## Open Questions
- What is the optimal ratio of MMA instructions to memory instructions for saturating TC throughput on Hopper (sm_90) wgmma vs Ampere (sm_80) mma.sync?
- How to best combine tcgen05 Tensor Memory with TC5 operations on Blackwell (sm_120)?
