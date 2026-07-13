// expf_probe.cu -- Compute-bound throughput comparison: expf vs __expf on H200
// Protocol: benchmark-protocol.md (warmup=5, repeat=20, CUDA events)
// Target: sm_90a (H200)
//
// Design: Each thread computes ITERS chained expf calls. To prevent overflow
// and dead-code elimination, each expf result is scaled back into a safe
// input range via a cheap fma: acc = expf(acc) * SCALE + BIAS, where
// SCALE and BIAS map [0, exp(4)] ~ [0, 54.6] back to [-4, 4].
// The fma cost is constant between variants and thus cancels in the ratio.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

constexpr int N     = 1 << 20;  // 1M threads
constexpr int BLOCK = 256;
constexpr int GRID  = (N + BLOCK - 1) / BLOCK;
constexpr int ITERS = 128;      // iterations per thread

// Rescale constants: map [0, ~54.6] -> [-4, 4]
// scale = 8.0/54.6 ~ 0.1465; bias = -4.0
constexpr float SCALE = 8.0f / 54.598f;
constexpr float BIAS  = -4.0f;

// Standard expf kernel
__global__ void kernel_expf(const float* __restrict__ in,
                            float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = in[i];
    #pragma unroll 8
    for (int iter = 0; iter < ITERS; ++iter) {
        acc = expf(acc) * SCALE + BIAS;
    }
    out[i] = acc;
}

// Fast __expf kernel
__global__ void kernel_fast_expf(const float* __restrict__ in,
                                 float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = in[i];
    #pragma unroll 8
    for (int iter = 0; iter < ITERS; ++iter) {
        acc = __expf(acc) * SCALE + BIAS;
    }
    out[i] = acc;
}

// Precision probe: single call
__global__ void kernel_expf_single(const float* __restrict__ in,
                                   float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

__global__ void kernel_fast_expf_single(const float* __restrict__ in,
                                        float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __expf(in[i]);
}

static float compute_max_ulp_error(const float* result, const float* input,
                                   int n) {
    float max_ulp = 0.0f;
    for (int i = 0; i < n; ++i) {
        double ref = exp((double)input[i]);
        float  ref_f = (float)ref;
        float  got = result[i];
        if (std::isinf(ref_f) || std::isnan(ref_f) ||
            std::isinf(got)   || std::isnan(got)) continue;
        if (ref_f == got) continue;
        int32_t ri, gi;
        memcpy(&ri, &ref_f, 4);
        memcpy(&gi, &got,   4);
        float ulp = fabsf((float)(ri - gi));
        if (ulp > max_ulp) max_ulp = ulp;
    }
    return max_ulp;
}

struct BenchResult {
    float median_ms;
    float p10_ms;
    float p90_ms;
};

static BenchResult bench(void(*launcher)(), int warmup, int repeats) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) launcher();
    CHECK(cudaDeviceSynchronize());

    std::vector<float> times(repeats);
    for (int i = 0; i < repeats; ++i) {
        CHECK(cudaEventRecord(start));
        launcher();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
    }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return {times[repeats/2], times[repeats/5], times[(repeats*9)/10]};
}

static float *d_in, *d_out;
static int g_n;

static void launch_expf()            { kernel_expf<<<GRID, BLOCK>>>(d_in, d_out, g_n); }
static void launch_fast_expf()       { kernel_fast_expf<<<GRID, BLOCK>>>(d_in, d_out, g_n); }
static void launch_expf_single()     { kernel_expf_single<<<GRID, BLOCK>>>(d_in, d_out, g_n); }
static void launch_fast_expf_single(){ kernel_fast_expf_single<<<GRID, BLOCK>>>(d_in, d_out, g_n); }

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("N = %d threads, ITERS = %d, grid = %d, block = %d\n",
           N, ITERS, GRID, BLOCK);
    long long total_ops = (long long)N * ITERS;
    printf("Total expf ops per kernel = %lld\n", total_ops);
    g_n = N;

    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Input: uniform in [-4, 4]
    srand(42);
    for (int i = 0; i < N; ++i)
        h_in[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // ======== THROUGHPUT ========
    printf("\n==== Throughput (compute-bound, %d iters/thread) ====\n", ITERS);

    auto r_std = bench(launch_expf, 5, 20);
    double gops_std  = (double)total_ops / (r_std.median_ms * 1e-3) / 1e9;
    printf("[expf]   median=%.4f ms  p10=%.4f ms  p90=%.4f ms  (%.2f Gop/s)\n",
           r_std.median_ms, r_std.p10_ms, r_std.p90_ms, gops_std);

    auto r_fast = bench(launch_fast_expf, 5, 20);
    double gops_fast = (double)total_ops / (r_fast.median_ms * 1e-3) / 1e9;
    printf("[__expf] median=%.4f ms  p10=%.4f ms  p90=%.4f ms  (%.2f Gop/s)\n",
           r_fast.median_ms, r_fast.p10_ms, r_fast.p90_ms, gops_fast);

    float speedup = r_std.median_ms / r_fast.median_ms;
    printf("Speedup (__expf / expf): %.2fx\n", speedup);

    // ======== PRECISION ========
    printf("\n==== Precision (single-call, input in [-10, 10]) ====\n");
    for (int i = 0; i < N; ++i)
        h_in[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    launch_expf_single();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    float ulp_std = compute_max_ulp_error(h_out, h_in, N);
    printf("[expf]   max ULP error vs f64 ref: %.0f\n", ulp_std);

    launch_fast_expf_single();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    float ulp_fast = compute_max_ulp_error(h_out, h_in, N);
    printf("[__expf] max ULP error vs f64 ref: %.0f\n", ulp_fast);

    printf("\n=== FINAL SUMMARY ===\n");
    printf("expf   : %.4f ms => %.2f Gop/s, max ULP = %.0f\n",
           r_std.median_ms, gops_std, ulp_std);
    printf("__expf : %.4f ms => %.2f Gop/s, max ULP = %.0f\n",
           r_fast.median_ms, gops_fast, ulp_fast);
    printf("Speedup: %.2fx\n", speedup);

    free(h_in); free(h_out);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    return 0;
}
