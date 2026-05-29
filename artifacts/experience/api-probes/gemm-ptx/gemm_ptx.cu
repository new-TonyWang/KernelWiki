// AC-11 minimal cutlass-free GEMM.
//
// Composes the AC-10 TMA-PTX + wgmma-PTX primitives into a single GEMM kernel:
//   - Host: cuTensorMapEncodeTiled to build CUtensorMap descriptors for A and B.
//   - Device: mbarrier-pipelined TMA load A + B, then wgmma.mma_async, then store D.
//
// The minimum acceptable shape is one wgmma atom: M=64, N=8, K=16 (bf16 inputs, f32 accumulator).
// The kernel performs:
//   D[64,8] = sum_{k=0..15} A[64,k] * B[k,8]
//
// Verification: compares output element-by-element against cuBLAS at the same shape.
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
//        gemm_ptx.cu -lcuda -lcublas -o gemm_ptx
//
// Cutlass-free verification:
//   nvcc -E ... gemm_ptx.cu | grep -E 'cutlass::|cute::'  -> 0 matches
//   cuobjdump --dump-elf-symbols gemm_ptx | grep -E 'cutlass::|cute::' -> 0 matches

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

#define CU_CHECK(x) do { CUresult err = (x); if (err != CUDA_SUCCESS) { \
    const char* msg = nullptr; cuGetErrorString(err, &msg); \
    fprintf(stderr, "CU error %s at %s:%d\n", msg ? msg : "(?)", __FILE__, __LINE__); \
    std::exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

constexpr int M = 64;
constexpr int N = 8;
constexpr int K = 16;

// === inline-PTX helpers (replicated from wgmma-ptx + tma-ptx hello-worlds) ===

__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr, uint32_t ld_bytes, uint32_t sd_bytes, uint32_t swizzle = 0)
{
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_int >> 4)) & 0x3FFF;
    desc |= ((uint64_t)(ld_bytes >> 4) & 0xFFFF) << 14;
    desc |= ((uint64_t)(sd_bytes >> 4) & 0xFFFF) << 30;
    desc |= ((uint64_t)(swizzle & 0x3)) << 62;
    return desc;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t n_threads) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(bar_int), "r"(n_threads));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(bar_int), "r"(bytes));
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{ .reg .pred P1;\n"
        "  LAB_W%=: \n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "  @P1 bra DONE_W%=; \n"
        "  bra LAB_W%=; \n"
        "  DONE_W%=: }\n"
        :: "r"(bar_int), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* tmap, int32_t r, int32_t c, uint64_t* mbar)
{
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t bar_int  = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t map_addr = reinterpret_cast<uint64_t>(tmap);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(smem_int), "l"(map_addr), "r"(bar_int),
           "r"(c), "r"(r));
}

// === GEMM kernel: TMA-load A + B, then wgmma m64n8k16 bf16, then store D. ===

__global__ void gemm_ptx_kernel(
    const CUtensorMap* tmap_A,   // 64 × 16 bf16
    const CUtensorMap* tmap_B,   // 16 × 8 bf16
    float* __restrict__ D)       // 64 × 8 f32
{
    extern __shared__ uint8_t smem[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem + M * K * sizeof(__nv_bfloat16));
    uint64_t* smem_bar    = reinterpret_cast<uint64_t*>(smem + (M * K + K * N) * sizeof(__nv_bfloat16));

    int tid = threadIdx.x;

    // Stage 1: TMA load A and B into smem.
    if (tid == 0) {
        mbarrier_init(smem_bar, 1);
        constexpr uint32_t bytes_a = M * K * sizeof(__nv_bfloat16);  // 64*16*2 = 2048
        constexpr uint32_t bytes_b = K * N * sizeof(__nv_bfloat16);  // 16*8*2 = 256
        mbarrier_arrive_expect_tx(smem_bar, bytes_a + bytes_b);
        tma_load_2d(smem_a, tmap_A, 0, 0, smem_bar);
        tma_load_2d(smem_b, tmap_B, 0, 0, smem_bar);
    }
    __syncthreads();
    if (tid == 0) {
        mbarrier_wait(smem_bar, 0);
    }
    __syncthreads();

    // Stage 2: wgmma m64n8k16 bf16 → f32.
    // The exact wgmma SS-variant smem-descriptor LBO/SBO encoding for non-swizzled
    // smem at non-canonical (M,K) shapes is non-trivial to derive from PTX docs
    // alone; cutlass's GMMA::DescriptorEncode helpers compute these per atom-shape
    // pair. Round-12 attempted both (256,256) and (256,16/128,16) — neither
    // produced a numeric match against cuBLAS at this M=64 N=8 K=16 shape with
    // non-trivial inputs (the all-1.0 wgmma_hello case happens to look correct
    // for ANY descriptor + ANY fragment-store permutation because all entries are
    // identical). The fix path for the next round: extract the descriptor encoder
    // from cute/atom/mma_traits_sm90_gmma.hpp and reproduce it cutlass-free in
    // helper.h.
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    // For tnspA=0 (K-major), descA's LBO = stride from one 8-row block to next = 8 * K * sizeof(bf16) = 256 bytes;
    // SBO = stride within an 8-row block to skip 8 K-elements = 8 * sizeof(bf16) = 16 bytes.
    uint64_t descA = make_smem_desc(smem_a, /*ld=*/256, /*sd=*/16, 0);
    // For tnspB=0 (K-major) with col-major K×N (= row-major N×K storage in smem):
    // each row of physical storage is K elements (32 bytes); LBO = 8 N-rows × K bytes = 8 * 32 = 256 bytes;
    // SBO = stride within an 8-N-row block to skip 8 K-cols = 8 * sizeof(bf16) = 16 bytes.
    uint64_t descB = make_smem_desc(smem_b, /*ld=*/256, /*sd=*/16, 0);

    // wgmma immediate args (PTX ISA 8.x §9.7.14.5.1): scale_d, scale_a, scale_b, transA, transB.
    // transA = GMMA::Major::K (= 0): A is K-major — matches our row-major M×K storage.
    // transB = GMMA::Major::K (= 0): B is K-major — col-major K×N storage; achieved by host-side
    //                                  pre-transpose: hB is allocated as row-major N×K, indexed as
    //                                  hB[n*K + k] = B[k,n], so the byte sequence is B[*,0], B[*,1],
    //                                  ..., B[*,N-1] = col-major K×N. See Round-14 BL-20260428-wgmma-ssTN-B-layout.
    asm volatile("wgmma.fence.sync.aligned;\n");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(descA), "l"(descB));
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");

    // Stage 3: store the 4 fragment elements per thread to D.
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row_base = warp_id * 16 + (lane_id >> 2);
    int col_base = (lane_id & 0x3) * 2;
    if (row_base < M && col_base + 1 < N) {
        D[row_base * N + col_base + 0]       = d0;
        D[row_base * N + col_base + 1]       = d1;
        D[(row_base + 8) * N + col_base + 0] = d2;
        D[(row_base + 8) * N + col_base + 1] = d3;
    }
}

int main() {
    // 1. Build A (64 × 16 bf16), B (16 × 8 bf16) on host with deterministic small values
    //    so cuBLAS comparison is meaningful (not all-1.0).
    __nv_bfloat16* hA = new __nv_bfloat16[M * K];
    __nv_bfloat16* hB_row = new __nv_bfloat16[K * N];   // row-major K×N (for cuBLAS path; keeps Round-12 cuBLAS call correct)
    __nv_bfloat16* hB_col = new __nv_bfloat16[K * N];   // col-major K×N = row-major N×K storage (for wgmma .SS_TN)
    for (int i = 0; i < M * K; ++i) hA[i] = __float2bfloat16(((i % 7) - 3) * 0.25f);
    for (int i = 0; i < K * N; ++i) hB_row[i] = __float2bfloat16(((i % 5) - 2) * 0.5f);
    // Physical transpose: hB_col[n*K + k] = hB_row[k*N + n] (= conceptual B[k,n]).
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            hB_col[n * K + k] = hB_row[k * N + n];

    __nv_bfloat16 *dA, *dB_row, *dB_col;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB_row, K * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB_col, K * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_row, hB_row, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_col, hB_col, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // 2. Build CUtensorMaps for A (64×16, row-major) and B (col-major K×N for wgmma).
    cuInit(0);
    CUtensorMap tmap_A = {};
    CUtensorMap tmap_B = {};
    {
        cuuint64_t global_dim[2]    = {K, M};
        cuuint64_t global_stride[1] = {K * sizeof(__nv_bfloat16)};
        cuuint32_t box_dim[2]       = {K, M};
        cuuint32_t elem_stride[2]   = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tmap_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA,
            global_dim, global_stride, box_dim, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        // For col-major K×N B: physical storage shape is N rows × K cols (row-major N×K).
        // CUtensorMap dim order is fastest-moving first, so {K, N} with stride {K * sizeof}.
        cuuint64_t global_dim[2]    = {K, N};
        cuuint64_t global_stride[1] = {K * sizeof(__nv_bfloat16)};
        cuuint32_t box_dim[2]       = {K, N};
        cuuint32_t elem_stride[2]   = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tmap_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB_col,
            global_dim, global_stride, box_dim, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    CUtensorMap *d_tmap_A, *d_tmap_B;
    CUDA_CHECK(cudaMalloc(&d_tmap_A, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tmap_B, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tmap_A, &tmap_A, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmap_B, &tmap_B, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // 3. Allocate output and launch.
    float* dD;
    CUDA_CHECK(cudaMalloc(&dD, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dD, 0, M * N * sizeof(float)));

    size_t smem_bytes = (M * K + K * N) * sizeof(__nv_bfloat16) + sizeof(uint64_t);
    gemm_ptx_kernel<<<1, 128, smem_bytes>>>(d_tmap_A, d_tmap_B, dD);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* hD_ptx = new float[M * N];
    CUDA_CHECK(cudaMemcpy(hD_ptx, dD, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 4. Compute cuBLAS reference. cublasGemmEx with bf16 inputs / f32 accumulator.
    //    cuBLAS uses col-major; our row-major A (M×K) and B (K×N) appear transposed
    //    to cuBLAS as A^T (K×M col-major) and B^T (N×K col-major). For C = A·B (M×N row-major),
    //    cuBLAS computes (B^T)·(A^T) col-major = (A·B)^T col-major = A·B row-major when read back.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float* dD_cublas;
    CUDA_CHECK(cudaMalloc(&dD_cublas, M * N * sizeof(float)));
    const float alpha = 1.f, beta = 0.f;
    // Compute D[M×N] = A[M×K] · B[K×N] in row-major.
    // cuBLAS call: D^T[N×M] = B^T[N×K] · A^T[K×M] (all col-major) -> equivalent to row-major D = A·B
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB_row, CUDA_R_16BF, N,    // cuBLAS path uses the original row-major K×N layout (lda for col-major B^T = N)
        dA, CUDA_R_16BF, K,    // lda for col-major A^T = K
        &beta,
        dD_cublas, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* hD_cublas = new float[M * N];
    CUDA_CHECK(cudaMemcpy(hD_cublas, dD_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Compare element-by-element. Allow a small tolerance for bf16 rounding-order
    //    differences between our wgmma path and cuBLAS's path.
    double max_abs = 0.0;
    double max_rel = 0.0;
    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        double d_ptx = hD_ptx[i];
        double d_cub = hD_cublas[i];
        double abs_diff = std::fabs(d_ptx - d_cub);
        double rel_diff = (std::fabs(d_cub) > 1e-6) ? abs_diff / std::fabs(d_cub) : abs_diff;
        if (abs_diff > max_abs) max_abs = abs_diff;
        if (rel_diff > max_rel) max_rel = rel_diff;
        if (abs_diff > 1e-2 && rel_diff > 1e-2) ++mismatches;
    }

    printf("=== gemm_ptx (cutlass-free GEMM at M=%d N=%d K=%d, bf16 inputs / f32 accum) ===\n", M, N, K);
    printf("  ptx     hD[0..3]: %.4f %.4f %.4f %.4f\n", hD_ptx[0], hD_ptx[1], hD_ptx[2], hD_ptx[3]);
    printf("  cuBLAS  hD[0..3]: %.4f %.4f %.4f %.4f\n", hD_cublas[0], hD_cublas[1], hD_cublas[2], hD_cublas[3]);
    printf("  max_abs_diff = %.6f\n", max_abs);
    printf("  max_rel_diff = %.6f\n", max_rel);
    printf("  mismatches (both abs > 1e-2 AND rel > 1e-2): %d / %d\n", mismatches, M * N);
    printf("  verdict: %s\n", mismatches == 0 ? "PASS (matches cuBLAS within tolerance)" : "FAIL");

    delete[] hA; delete[] hD_ptx; delete[] hD_cublas;
    cudaFree(dA); cudaFree(dB_row); cudaFree(dB_col); cudaFree(dD); cudaFree(dD_cublas);
    delete[] hB_row; delete[] hB_col;
    cudaFree(d_tmap_A); cudaFree(d_tmap_B);
    cublasDestroy(handle);
    return mismatches == 0 ? 0 : 1;
}
