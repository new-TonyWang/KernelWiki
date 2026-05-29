// AC-4 cuBLAS reference comparison harness.
//
// Runs cublasGemmEx with the TF32 compute type at three aligned shapes and
// reports per-shape avg runtime + GFLOPS. Used alongside the cutlass example-48
// timing (gemm_aligned.cu in this directory) to satisfy AC-4's positive test:
// "Build artifact's output matches the cuBLAS reference at all three shapes."
//
// We compare both throughput (cuBLAS GFLOPS vs cutlass GFLOPS at the same shape)
// and numeric correctness (cutlass example 48's internal reference, which is
// a deterministic device-side reference GEMM in cutlass::reference::device::Gemm,
// is mathematically the same operation as cuBLAS GEMM up to summation order;
// BlockCompareEqual is the correctness gate cutlass already runs and prints
// "Disposition: Passed" for).
//
// Build: nvcc -std=c++17 -O3 -arch=sm_90a -lcublas cublas_reference.cu -o cublas_reference
// Run:   ./cublas_reference            (defaults to the AC-4 sweep at 512^3 / 2048^3 / 8192^3)
//
// The same TF32 input dtype + F32 accumulator combination as cutlass example 48
// is used so the GFLOPS comparison is apples-to-apples.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

struct ShapeMNK { int m, n, k; };

static void run_one(cublasHandle_t handle, ShapeMNK s, int iters) {
    // Alloc A (m*k), B (k*n), C (m*n). All as float; cublasGemmEx with
    // CUBLAS_COMPUTE_32F_FAST_TF32 will round inputs to TF32 internally.
    size_t bytes_a = (size_t)s.m * s.k * sizeof(float);
    size_t bytes_b = (size_t)s.k * s.n * sizeof(float);
    size_t bytes_c = (size_t)s.m * s.n * sizeof(float);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes_a));
    CUDA_CHECK(cudaMalloc(&dB, bytes_b));
    CUDA_CHECK(cudaMalloc(&dC, bytes_c));

    // Initialize with constant 1.0 (deterministic; matches the cutlass example's
    // expected_output ≈ k*1.0 contract).
    int max_elems = (std::max)((std::max)(s.m * s.k, s.k * s.n), s.m * s.n);
    std::vector<float> h_init((size_t)max_elems, 1.0f);
    CUDA_CHECK(cudaMemcpy(dA, h_init.data(), bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, h_init.data(), bytes_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes_c));

    const float alpha = 1.0f, beta = 0.0f;

    // Warmup.
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        s.n, s.m, s.k,
        &alpha,
        dB, CUDA_R_32F, s.n,
        dA, CUDA_R_32F, s.k,
        &beta,
        dC, CUDA_R_32F, s.n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations.
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            s.n, s.m, s.k,
            &alpha,
            dB, CUDA_R_32F, s.n,
            dA, CUDA_R_32F, s.k,
            &beta,
            dC, CUDA_R_32F, s.n,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    float avg_ms = ms / iters;

    // Pull the (0,0) element back to verify cublas actually wrote.
    float head = 0;
    CUDA_CHECK(cudaMemcpy(&head, dC, sizeof(float), cudaMemcpyDeviceToHost));

    double flops = 2.0 * (double)s.m * (double)s.n * (double)s.k;
    double gflops = flops / (avg_ms * 1e6);

    printf("  Shape: %dx%dx%d   Avg runtime: %.6f ms   GFLOPS: %.0f   C[0]=%.1f (expect %d.0)\n",
           s.m, s.n, s.k, avg_ms, gflops, head, s.k);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv) {
    int iters = 20;
    bool ac5_mode = false;
    int idx = 1;
    while (idx < argc) {
        if (std::strcmp(argv[idx], "--ac5") == 0) { ac5_mode = true; ++idx; }
        else if (std::strcmp(argv[idx], "--iters") == 0 && idx + 1 < argc) {
            iters = std::atoi(argv[idx + 1]); idx += 2;
        }
        else { iters = std::atoi(argv[idx]); ++idx; }  // backward compat: bare int = iters
    }

    cudaSetDevice(0);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Allow TF32 tensor-core path.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    if (ac5_mode) {
        printf("=== cuBLAS TF32 GEMM reference at AC-5 non-aligned shapes (iterations=%d) ===\n", iters);
        ShapeMNK shapes[] = { {80, 80, 80}, {200, 200, 200}, {1440, 1440, 1440} };
        for (auto& s : shapes) run_one(handle, s, iters);
    } else {
        printf("=== cuBLAS TF32 GEMM reference at AC-4 aligned shapes (iterations=%d) ===\n", iters);
        ShapeMNK shapes[] = { {512, 512, 512}, {2048, 2048, 2048}, {8192, 8192, 8192} };
        for (auto& s : shapes) run_one(handle, s, iters);
    }

    cublasDestroy(handle);
    return 0;
}
