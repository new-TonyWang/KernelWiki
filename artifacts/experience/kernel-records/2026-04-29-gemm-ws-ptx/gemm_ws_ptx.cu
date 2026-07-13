// Cutlass-free warp-specialized GEMM (Hopper, sm_90a).
//
// Composes:
//   - 40-hardware-feature/tma-ptx     -- producer side (cp.async.bulk.tensor + mbarrier)
//   - 40-hardware-feature/wgmma-ptx   -- consumer side (wgmma.mma_async.m64n8k16 bf16)
//   - 50-classical-algo/warp-specialization (algorithm skeleton: producer/consumer
//     split + smem ring buffer + per-stage full/empty mbarrier pair)
//
// CTA layout:
//   1 CTA x 5 warps = 160 threads.
//     warps 0..3 (tid 0..127)   = consumer warpgroup (wgmma + epilogue store).
//     warp  4    (tid 128..159) = producer warp      (TMA loads of A and B).
//
// Problem:
//   D[M=64, N=8] = sum_{k=0..K_TOTAL-1} A[M, K_TOTAL] * B[K_TOTAL, N]
//   K_TOTAL = N_K_TILES * K_TILE, K_TILE = 16, N_K_TILES = 2 (=> K_TOTAL = 32).
//   STAGES  = 2 (smem ring depth).
//
// Why N_K_TILES = 2:
//   The pipeline is non-trivial only when N_K_TILES >= STAGES, since the producer's
//   steady-state branch (k >= STAGES) is what waits on bar_empty before refilling a
//   stage. STAGES=2 + N_K_TILES=2 covers exactly the boundary case (one initial-fill
//   iteration per stage, no steady-state reuse) -- which is the minimum that proves
//   the producer/consumer split runs at all. Bumping N_K_TILES beyond 2 exercises
//   the stage-reuse path; the kernel is parametric so changing the constant alone
//   is enough to retest deeper K.
//
// Cutlass-free verification (the hard gate, same as 30-skill/compute/gemm-ptx):
//   nvcc -E gemm_ws_ptx.cu | grep -E 'cutlass::|cute::'   -> 0 matches
//   cuobjdump --dump-elf-symbols gemm_ws_ptx | grep cutlass::|cute::  -> 0 matches
//
// Numeric correctness vs cuBLAS at M=64 N=8 K=32 bf16:
//   Inherits the per-thread fragment-store mapping from 60-code/ptx-gemm/gemm_ptx.cu.
//   That mapping is currently flagged as suspect by 30-skill/compute/gemm-ptx/pitfalls.md
//   item #1 (residual 497/512 mismatches at the single-tile shape). The warp-spec
//   layer in this file is independent of that bug -- if upstream lands a fix, this
//   kernel inherits it without further changes.

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

// ----- problem and pipeline shape ---------------------------------------------------
constexpr int M           = 64;
constexpr int N           = 8;
constexpr int K_TILE      = 16;
constexpr int N_K_TILES   = 2;
constexpr int K_TOTAL     = N_K_TILES * K_TILE;     // 32
constexpr int STAGES      = 2;

constexpr int CONSUMER_THREADS = 128;               // 4 warps
constexpr int PRODUCER_THREADS = 32;                // 1 warp
constexpr int CTA_THREADS      = CONSUMER_THREADS + PRODUCER_THREADS;
constexpr int PRODUCER_WARP    = 4;                 // warp index of producer

constexpr uint32_t TILE_BYTES_A = M * K_TILE * sizeof(__nv_bfloat16);  // 2048
constexpr uint32_t TILE_BYTES_B = K_TILE * N * sizeof(__nv_bfloat16);  //  256
constexpr uint32_t TILE_BYTES   = TILE_BYTES_A + TILE_BYTES_B;         // 2304

// ===== inline-PTX primitives (replicated from tma-ptx + wgmma-ptx hello-worlds) =====

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

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t n_arrivals) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(bar_int), "r"(n_arrivals));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar_int));
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

// ===== kernel ========================================================================

__global__ void __launch_bounds__(CTA_THREADS, 1) gemm_ws_ptx_kernel(
    const CUtensorMap* tmap_A,   // logical M x K_TOTAL bf16 (row-major; K is fastest)
    const CUtensorMap* tmap_B,   // logical K_TOTAL x N bf16 col-major (= N x K_TOTAL row-major in storage)
    float* __restrict__ D)       // 64 x 8 f32 row-major
{
    extern __shared__ __align__(16) uint8_t smem_raw[];

    // Smem layout (16-byte aligned regions; bf16 tiles + uint64 mbarriers):
    //   [0                                ) smem_a[STAGES * M * K_TILE]
    //   [STAGES*TILE_BYTES_A              ) smem_b[STAGES * K_TILE * N]
    //   [STAGES*(TILE_BYTES_A+TILE_BYTES_B)) bar_full[STAGES]
    //   [+ STAGES*8                       ) bar_empty[STAGES]
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(
        smem_raw + STAGES * TILE_BYTES_A);
    uint64_t* bar_full  = reinterpret_cast<uint64_t*>(
        smem_raw + STAGES * (TILE_BYTES_A + TILE_BYTES_B));
    uint64_t* bar_empty = bar_full + STAGES;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    // ---- mbarrier init: one thread sets up all 2*STAGES barriers, then a CTA-wide
    //      __syncthreads before either side touches them. Each barrier expects exactly
    //      one explicit arrival; bar_full additionally tracks the TMA's expect_tx bytes.
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbarrier_init(&bar_full[s],  1);   // producer arrives (with expect_tx)
            mbarrier_init(&bar_empty[s], 1);   // consumer arrives once per stage
        }
    }
    __syncthreads();

    if (warp_id == PRODUCER_WARP) {
        // ---- Producer warp (32 lanes; only lane 0 issues the protocol).
        //
        // Phase tracking on bar_empty[s]:
        //   Initial parity = 0. Consumer's first arrive on stage s flips parity -> 1.
        //   Producer at iteration k >= STAGES waits before refilling stage s = k%STAGES,
        //   waiting for the flip caused by consumer iteration (k - STAGES). The phase
        //   to pass to mbarrier.try_wait.parity is the parity that the barrier currently
        //   holds (the wait succeeds when it differs). Across stage reuses, that parity
        //   is ((k - STAGES) / STAGES) & 1.
        if (lane == 0) {
            for (int k = 0; k < N_K_TILES; ++k) {
                int s = k % STAGES;

                if (k >= STAGES) {
                    uint32_t empty_phase = ((k - STAGES) / STAGES) & 1u;
                    mbarrier_wait(&bar_empty[s], empty_phase);
                }

                mbarrier_arrive_expect_tx(&bar_full[s], TILE_BYTES);
                tma_load_2d(&smem_a[s * (M * K_TILE)],
                            tmap_A, /*r=*/0, /*c=*/k * K_TILE, &bar_full[s]);
                tma_load_2d(&smem_b[s * (K_TILE * N)],
                            tmap_B, /*r=*/0, /*c=*/k * K_TILE, &bar_full[s]);
            }
        }
        // Other producer lanes have nothing to do.
    } else {
        // ---- Consumer warpgroup (warps 0..3, all 128 threads execute the wgmma).
        //
        // Per-thread accumulator fragment for m64n8k16 f32:
        //   64*8 = 512 outputs / 128 threads = 4 floats per thread (= 2 register pairs).
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

        // K-loop: every iteration waits for stage s's TMA to complete, issues a wgmma
        // that ACCUMULATES (scaleD=1) into {d0,d1,d2,d3}, fences, and signals bar_empty
        // so the producer may refill stage s.
        for (int k = 0; k < N_K_TILES; ++k) {
            int s = k % STAGES;
            uint32_t full_phase = (k / STAGES) & 1u;
            mbarrier_wait(&bar_full[s], full_phase);

            // wgmma SS-variant smem descriptors. Encoding follows
            // 40-hardware-feature/wgmma-ptx and 30-skill/compute/gemm-ptx/pitfalls.md #2
            // (LD = byte stride between consecutive 8-row groups along the wgmma-M axis;
            //  SD = byte stride between consecutive 8-col groups along the contracting axis).
            //   A : row-major M x K_TILE bf16 -> LD = 8*K_TILE*2 = 256, SD = 8*2 = 16.
            //   B : col-major K_TILE x N (= row-major N x K_TILE physical), wgmma-M axis is N
            //       -> LD = 8*K_TILE*2 = 256, SD = 8*2 = 16.
            uint64_t descA = make_smem_desc(&smem_a[s * (M * K_TILE)], 256, 16, 0);
            uint64_t descB = make_smem_desc(&smem_b[s * (K_TILE * N)], 256, 16, 0);

            asm volatile("wgmma.fence.sync.aligned;\n");
            // Immediates: scaleD=1 (accumulate), scaleA=1, scaleB=1, transA=0 (K-major A),
            // transB=0 (K-major B = SS_TN). See 40-hardware-feature/wgmma-ptx/pitfalls.md #8.
            asm volatile(
                "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
                "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "l"(descA), "l"(descB));
            asm volatile("wgmma.commit_group.sync.aligned;\n");
            asm volatile("wgmma.wait_group.sync.aligned 0;\n");

            // After wait_group, this stage's smem is no longer in flight.
            // Any single thread of the consumer warpgroup signals bar_empty[s].
            if (tid == 0) {
                mbarrier_arrive(&bar_empty[s]);
            }
        }

        // ---- Epilogue: store the per-thread fragment to D[M, N].
        // Hopper wgmma f32 m64nNk16 fragment, per PTX ISA "Asynchronous Warpgroup-Level
        // Matrix Operations / Matrix Fragments":
        //   Each warp w (0..3) owns rows [w*16, w*16+16) of the M=64 output tile.
        //   Within the warp's 16 rows, lane l holds two 2-element row segments:
        //     row_top = w*16 + (l/4)
        //     row_bot = w*16 + (l/4) + 8
        //     col0    = (l%4) * 2,   col1 = col0 + 1
        //     {d0, d1} -> (row_top, col0..col1)
        //     {d2, d3} -> (row_bot, col0..col1)
        //
        // CAVEAT: 30-skill/compute/gemm-ptx/pitfalls.md #1 flags this exact mapping as
        // a *suspected* layout bug at the single-tile shape (497/512 mismatches against
        // cuBLAS). It is the same mapping used by gemm_ptx.cu and is preserved here
        // verbatim so this kernel's correctness behaviour is the WS-pipeline delta on
        // top of gemm_ptx, not a different layout convention.
        int row_top = warp_id * 16 + (lane >> 2);
        int row_bot = row_top + 8;
        int col0    = (lane & 0x3) * 2;
        int col1    = col0 + 1;
        if (row_top < M && col1 < N) {
            D[row_top * N + col0] = d0;
            D[row_top * N + col1] = d1;
            D[row_bot * N + col0] = d2;
            D[row_bot * N + col1] = d3;
        }
    }
}

// ===== host driver: build CUtensorMaps, launch, compare against cuBLAS =============

int main() {
    // ---- 1. Host buffers with non-trivial values (avoids the all-1.0 trap that masks
    //         layout permutations -- see 50-classical-algo/warp-specialization/pitfalls
    //         and 30-skill/compute/gemm-ptx/pitfalls.md #6).
    __nv_bfloat16* hA      = new __nv_bfloat16[M * K_TOTAL];
    __nv_bfloat16* hB_row  = new __nv_bfloat16[K_TOTAL * N];   // row-major K x N for cuBLAS
    __nv_bfloat16* hB_col  = new __nv_bfloat16[K_TOTAL * N];   // col-major K x N (= row-major N x K) for wgmma SS_TN

    for (int i = 0; i < M * K_TOTAL; ++i)  hA[i]     = __float2bfloat16(((i % 7) - 3) * 0.25f);
    for (int i = 0; i < K_TOTAL * N; ++i)  hB_row[i] = __float2bfloat16(((i % 5) - 2) * 0.5f);
    for (int k = 0; k < K_TOTAL; ++k)
        for (int n = 0; n < N; ++n)
            hB_col[n * K_TOTAL + k] = hB_row[k * N + n];

    __nv_bfloat16 *dA, *dB_row, *dB_col;
    CUDA_CHECK(cudaMalloc(&dA,     M * K_TOTAL * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB_row, K_TOTAL * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB_col, K_TOTAL * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(dA,     hA,     M * K_TOTAL * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_row, hB_row, K_TOTAL * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_col, hB_col, K_TOTAL * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // ---- 2. CUtensorMaps. global_dim is fastest-moving first
    //         (40-hardware-feature/tma-ptx/skill.md, pitfalls.md #4).
    cuInit(0);
    CUtensorMap tmap_A = {};
    CUtensorMap tmap_B = {};
    {
        // A : row-major M x K_TOTAL ; box = M x K_TILE ; we slide along K.
        cuuint64_t global_dim[2]    = {(cuuint64_t)K_TOTAL, (cuuint64_t)M};
        cuuint64_t global_stride[1] = {(cuuint64_t)(K_TOTAL * sizeof(__nv_bfloat16))};
        cuuint32_t box_dim[2]       = {(cuuint32_t)K_TILE, (cuuint32_t)M};
        cuuint32_t elem_stride[2]   = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tmap_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA,
            global_dim, global_stride, box_dim, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    {
        // B : col-major K_TOTAL x N stored as row-major N x K_TOTAL ; box = N x K_TILE ; slide along K.
        cuuint64_t global_dim[2]    = {(cuuint64_t)K_TOTAL, (cuuint64_t)N};
        cuuint64_t global_stride[1] = {(cuuint64_t)(K_TOTAL * sizeof(__nv_bfloat16))};
        cuuint32_t box_dim[2]       = {(cuuint32_t)K_TILE, (cuuint32_t)N};
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

    // ---- 3. Launch.
    float* dD;
    CUDA_CHECK(cudaMalloc(&dD, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dD, 0, M * N * sizeof(float)));

    // smem footprint:
    //   STAGES * (TILE_BYTES_A + TILE_BYTES_B) + 2 * STAGES * sizeof(uint64_t)
    //   = 2 * 2304 + 2 * 2 * 8 = 4640 bytes
    size_t smem_bytes = STAGES * (TILE_BYTES_A + TILE_BYTES_B)
                      + 2 * STAGES * sizeof(uint64_t);

    gemm_ws_ptx_kernel<<<1, CTA_THREADS, smem_bytes>>>(d_tmap_A, d_tmap_B, dD);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* hD_ptx = new float[M * N];
    CUDA_CHECK(cudaMemcpy(hD_ptx, dD, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 4. cuBLAS reference.
    //         D[M,N] = A[M,K_TOTAL] * B[K_TOTAL,N] in row-major.
    //         cuBLAS is col-major; the canonical idiom is to call (B^T)*(A^T) col-major
    //         with sizes swapped (see 30-skill/compute/gemm-ptx/pitfalls.md #3).
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float* dD_cublas;
    CUDA_CHECK(cudaMalloc(&dD_cublas, M * N * sizeof(float)));
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K_TOTAL,
        &alpha,
        dB_row,    CUDA_R_16BF, N,
        dA,        CUDA_R_16BF, K_TOTAL,
        &beta,
        dD_cublas, CUDA_R_32F,  N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* hD_cublas = new float[M * N];
    CUDA_CHECK(cudaMemcpy(hD_cublas, dD_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 5. Compare.
    double max_abs = 0.0, max_rel = 0.0;
    int mismatches = 0;
    int nonzero_outputs = 0;
    for (int i = 0; i < M * N; ++i) {
        if (hD_ptx[i] != 0.f) ++nonzero_outputs;
        double a = hD_ptx[i], b = hD_cublas[i];
        double abs_d = std::fabs(a - b);
        double rel_d = (std::fabs(b) > 1e-6) ? abs_d / std::fabs(b) : abs_d;
        if (abs_d > max_abs) max_abs = abs_d;
        if (rel_d > max_rel) max_rel = rel_d;
        if (abs_d > 1e-2 && rel_d > 1e-2) ++mismatches;
    }

    printf("=== gemm_ws_ptx (cutlass-free WS GEMM at M=%d N=%d K=%d, "
           "STAGES=%d, %d K-tiles, bf16 in / f32 acc) ===\n",
           M, N, K_TOTAL, STAGES, N_K_TILES);
    printf("  ws_ptx hD[0..3]: %.4f %.4f %.4f %.4f\n",
           hD_ptx[0], hD_ptx[1], hD_ptx[2], hD_ptx[3]);
    printf("  cuBLAS hD[0..3]: %.4f %.4f %.4f %.4f\n",
           hD_cublas[0], hD_cublas[1], hD_cublas[2], hD_cublas[3]);
    printf("  non-zero outputs: %d / %d  (proves the wgmma fragment + epilogue store reached D)\n",
           nonzero_outputs, M * N);
    printf("  max_abs_diff = %.6f\n", max_abs);
    printf("  max_rel_diff = %.6f\n", max_rel);
    printf("  mismatches (abs>1e-2 AND rel>1e-2): %d / %d\n", mismatches, M * N);
    printf("  pipeline check: kernel ran to completion -> producer/consumer mbarrier "
           "protocol did not deadlock across %d stages and %d K-tiles.\n",
           STAGES, N_K_TILES);

    delete[] hA;
    delete[] hB_row;
    delete[] hB_col;
    delete[] hD_ptx;
    delete[] hD_cublas;
    cudaFree(dA); cudaFree(dB_row); cudaFree(dB_col);
    cudaFree(dD); cudaFree(dD_cublas);
    cudaFree(d_tmap_A); cudaFree(d_tmap_B);
    cublasDestroy(handle);
    return 0;
}
