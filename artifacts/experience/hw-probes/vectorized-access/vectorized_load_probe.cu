// vectorized_load_probe.cu -- Microbenchmark: scalar float vs float4 vectorized load
// Target: H200 (sm_90a), CUDA 12.9
//
// Measures effective bandwidth for two access patterns:
//   (a) Scalar:     thread i reads input[i] one float at a time
//   (b) Vectorized:  thread i reads ((float4*)input)[i], processes 4 floats at once
//
// Both kernels read the same total number of floats from global memory.
// The vectorized kernel uses 1/4 the threads, each loading a float4 (16 bytes).
//
// Protocol: warmup 5, measure 20, CUDA events, median/p10/p90.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while(0)

// Kernel A: scalar load -- each thread reads one float
__global__ void scalar_load_kernel(const float* __restrict__ input,
                                   float*       __restrict__ output,
                                   int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (tid < n) {
        val = input[tid];
    }
    // Warp-reduce to prevent DCE and avoid output bottleneck
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    if (threadIdx.x % 32 == 0) {
        int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        output[warpId] = val;
    }
}

// Kernel B: vectorized float4 load -- each thread reads one float4 (4 floats)
__global__ void vectorized_load_kernel(const float4* __restrict__ input,
                                       float*        __restrict__ output,
                                       int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (tid < n4) {
        float4 v = input[tid];
        val = v.x + v.y + v.z + v.w;
    }
    // Warp-reduce to prevent DCE and avoid output bottleneck
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    if (threadIdx.x % 32 == 0) {
        int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        output[warpId] = val;
    }
}

struct BenchResult {
    float median_ms;
    float p10_ms;
    float p90_ms;
    double bw_gbps;
};

BenchResult bench_scalar(const float* d_input, float* d_output,
                         int n, int block_size, int warmup, int repeats) {
    int num_blocks = (n + block_size - 1) / block_size;
    // Warmup
    for (int i = 0; i < warmup; i++)
        scalar_load_kernel<<<num_blocks, block_size>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t e_start, e_end;
    CHECK_CUDA(cudaEventCreate(&e_start));
    CHECK_CUDA(cudaEventCreate(&e_end));

    std::vector<float> times_ms(repeats);
    for (int i = 0; i < repeats; i++) {
        CHECK_CUDA(cudaEventRecord(e_start));
        scalar_load_kernel<<<num_blocks, block_size>>>(d_input, d_output, n);
        CHECK_CUDA(cudaEventRecord(e_end));
        CHECK_CUDA(cudaEventSynchronize(e_end));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e_start, e_end));
        times_ms[i] = ms;
    }
    std::sort(times_ms.begin(), times_ms.end());

    BenchResult r;
    r.median_ms = times_ms[repeats / 2];
    r.p10_ms    = times_ms[repeats / 5];
    r.p90_ms    = times_ms[repeats * 9 / 10];
    double bytes = (double)n * sizeof(float);
    r.bw_gbps   = bytes / (r.median_ms * 1e-3) / 1e9;

    CHECK_CUDA(cudaEventDestroy(e_start));
    CHECK_CUDA(cudaEventDestroy(e_end));
    return r;
}

BenchResult bench_vectorized(const float4* d_input, float* d_output,
                             int n4, int block_size, int warmup, int repeats) {
    int num_blocks = (n4 + block_size - 1) / block_size;
    // Warmup
    for (int i = 0; i < warmup; i++)
        vectorized_load_kernel<<<num_blocks, block_size>>>(d_input, d_output, n4);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t e_start, e_end;
    CHECK_CUDA(cudaEventCreate(&e_start));
    CHECK_CUDA(cudaEventCreate(&e_end));

    std::vector<float> times_ms(repeats);
    for (int i = 0; i < repeats; i++) {
        CHECK_CUDA(cudaEventRecord(e_start));
        vectorized_load_kernel<<<num_blocks, block_size>>>(d_input, d_output, n4);
        CHECK_CUDA(cudaEventRecord(e_end));
        CHECK_CUDA(cudaEventSynchronize(e_end));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e_start, e_end));
        times_ms[i] = ms;
    }
    std::sort(times_ms.begin(), times_ms.end());

    BenchResult r;
    r.median_ms = times_ms[repeats / 2];
    r.p10_ms    = times_ms[repeats / 5];
    r.p90_ms    = times_ms[repeats * 9 / 10];
    double bytes = (double)n4 * sizeof(float4);  // same total bytes as scalar
    r.bw_gbps   = bytes / (r.median_ms * 1e-3) / 1e9;

    CHECK_CUDA(cudaEventDestroy(e_start));
    CHECK_CUDA(cudaEventDestroy(e_end));
    return r;
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("HBM bandwidth (theoretical): %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    int driverVersion, runtimeVersion;
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    printf("CUDA Runtime: %d.%d, Driver: %d.%d\n",
           runtimeVersion / 1000, (runtimeVersion % 100) / 10,
           driverVersion / 1000, (driverVersion % 100) / 10);

    // Configuration
    const int BLOCK_SIZE = 256;
    const int WARMUP = 5;
    const int REPEATS = 20;

    // Use 64M floats = 256 MB, well above L2 cache (51.2 MB on H200),
    // so we measure HBM bandwidth, not cache effects.
    const int N = 64 * 1024 * 1024;  // 64M floats
    const int N4 = N / 4;            // 16M float4 elements

    printf("\nConfiguration:\n");
    printf("  Total floats: %d (%.1f MB)\n", N, (double)N * 4.0 / (1 << 20));
    printf("  Scalar kernel: %d threads, %d blocks x %d\n",
           N, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("  Vectorized kernel: %d threads, %d blocks x %d\n",
           N4, (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("  Warmup: %d, Repeats: %d\n", WARMUP, REPEATS);

    // Allocate device memory
    float *d_input;
    float *d_output_scalar, *d_output_vec;
    CHECK_CUDA(cudaMalloc(&d_input, (size_t)N * sizeof(float)));
    int n_warps_scalar = (N + 31) / 32;
    int n_warps_vec    = (N4 + 31) / 32;
    CHECK_CUDA(cudaMalloc(&d_output_scalar, n_warps_scalar * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_vec,    n_warps_vec    * sizeof(float)));

    // Initialize input with deterministic values
    std::vector<float> h_input(N);
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), (size_t)N * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Verify correctness: both kernels should produce the same warp-reduced
    // sums. Each warp's lane-0 output is the sum of 32 input values (scalar)
    // or the sum of 32 float4-sums (vectorized, so sum of 128 values).
    // We compare the total sum across all warps.
    {
        int nb_s = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scalar_load_kernel<<<nb_s, BLOCK_SIZE>>>(d_input, d_output_scalar, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> h_out_s(n_warps_scalar);
        CHECK_CUDA(cudaMemcpy(h_out_s.data(), d_output_scalar,
                               n_warps_scalar * sizeof(float), cudaMemcpyDeviceToHost));
        double sum_s = 0.0;
        for (int i = 0; i < n_warps_scalar; i++) sum_s += h_out_s[i];

        int nb_v = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorized_load_kernel<<<nb_v, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(d_input), d_output_vec, N4);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> h_out_v(n_warps_vec);
        CHECK_CUDA(cudaMemcpy(h_out_v.data(), d_output_vec,
                               n_warps_vec * sizeof(float), cudaMemcpyDeviceToHost));
        double sum_v = 0.0;
        for (int i = 0; i < n_warps_vec; i++) sum_v += h_out_v[i];

        printf("\nCorrectness check:\n");
        printf("  Scalar total sum:     %.1f (expected %.1f)\n", sum_s, (double)N);
        printf("  Vectorized total sum: %.1f (expected %.1f)\n", sum_v, (double)N);
        double err = fabs(sum_s - sum_v);
        printf("  Absolute error:       %.6f\n", err);
        if (err > 1.0) {
            fprintf(stderr, "ERROR: correctness check failed (err=%.6f)\n", err);
            return 1;
        }
        printf("  PASS\n");
    }

    // Benchmark
    printf("\n--- Benchmarking ---\n");

    BenchResult scalar_r = bench_scalar(d_input, d_output_scalar,
                                        N, BLOCK_SIZE, WARMUP, REPEATS);
    printf("\n=== SCALAR (float) ===\n");
    printf("  Latency (ms): median=%.4f  p10=%.4f  p90=%.4f\n",
           scalar_r.median_ms, scalar_r.p10_ms, scalar_r.p90_ms);
    printf("  Effective BW: %.2f GB/s\n", scalar_r.bw_gbps);

    BenchResult vec_r = bench_vectorized(
        reinterpret_cast<const float4*>(d_input), d_output_vec,
        N4, BLOCK_SIZE, WARMUP, REPEATS);
    printf("\n=== VECTORIZED (float4) ===\n");
    printf("  Latency (ms): median=%.4f  p10=%.4f  p90=%.4f\n",
           vec_r.median_ms, vec_r.p10_ms, vec_r.p90_ms);
    printf("  Effective BW: %.2f GB/s\n", vec_r.bw_gbps);

    printf("\n=== SUMMARY ===\n");
    printf("  Scalar  median: %.4f ms, BW: %.2f GB/s\n",
           scalar_r.median_ms, scalar_r.bw_gbps);
    printf("  Float4  median: %.4f ms, BW: %.2f GB/s\n",
           vec_r.median_ms, vec_r.bw_gbps);
    printf("  Speedup (scalar_ms / vec_ms): %.2fx\n",
           scalar_r.median_ms / vec_r.median_ms);
    printf("  BW ratio (vec / scalar):      %.2fx\n",
           vec_r.bw_gbps / scalar_r.bw_gbps);

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_scalar));
    CHECK_CUDA(cudaFree(d_output_vec));

    return 0;
}
