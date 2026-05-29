// Probe: __ldg read-only cache load (end-to-end)
// Loads N floats via __ldg, adds 1.0, stores to output.
// Verifies against CPU reference (normal load + add).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N (1 << 20)  // 1M elements
#define BLK 256
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void ldg_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldg(in + tid) + 1.0f;
}

// Baseline: plain global load (compiler may also use LDG for const __restrict__,
// but we omit __restrict__ here to force the default path).
__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    // warmup
    for (int i = 0; i < 5; i++) ldg_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms_ldg(20), ms_plain(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        ldg_add<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_ldg[i], t0, t1);
    }
    // warmup baseline
    for (int i = 0; i < 5; i++) plain_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        plain_add<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_plain[i], t0, t1);
    }
    std::sort(ms_ldg.begin(), ms_ldg.end());
    std::sort(ms_plain.begin(), ms_plain.end());

    // verify __ldg kernel
    ldg_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < N; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - (h_in[i] + 1.0f)));

    printf("PASS=%s  max_abs_err=%.6e\n", max_err <= 1e-5 ? "true" : "false", max_err);
    printf("ldg    median=%.6f p10=%.6f p90=%.6f ms\n", ms_ldg[10], ms_ldg[2], ms_ldg[18]);
    printf("plain  median=%.6f p10=%.6f p90=%.6f ms\n", ms_plain[10], ms_plain[2], ms_plain[18]);
    printf("ratio=%.4f\n", ms_plain[10] / ms_ldg[10]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err <= 1e-5) ? 0 : 1;
}
