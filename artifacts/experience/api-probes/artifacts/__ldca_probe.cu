// Probe: __ldca cache-at-all-levels load (end-to-end)
// Loads N floats via __ldca (PTX ld.ca: cache in L1 and L2, default policy),
// adds 1.0, stores to output. Verifies against CPU reference.
// Baseline: plain global load (default ld.ca path).
// Buffer size: 256 MB (> H200 L2 ~60 MB) so the load path is HBM-bound.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N (1 << 26)  // 64M elements -> 256 MB for fp32
#define BLK 256
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void ldca_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldca(in + tid) + 1.0f;
}

__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    for (int i = 0; i < 5; i++) ldca_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms_api(20), ms_plain(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        ldca_add<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_api[i], t0, t1);
    }
    for (int i = 0; i < 5; i++) plain_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        plain_add<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_plain[i], t0, t1);
    }
    std::sort(ms_api.begin(), ms_api.end());
    std::sort(ms_plain.begin(), ms_plain.end());

    ldca_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < N; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - (h_in[i] + 1.0f)));

    printf("PASS=%s  max_abs_err=%.6e\n", max_err <= 1e-5 ? "true" : "false", max_err);
    printf("ldca   median=%.6f p10=%.6f p90=%.6f ms\n", ms_api[10], ms_api[2], ms_api[18]);
    printf("plain  median=%.6f p10=%.6f p90=%.6f ms\n", ms_plain[10], ms_plain[2], ms_plain[18]);
    printf("ratio=%.4f\n", ms_plain[10] / ms_api[10]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err <= 1e-5) ? 0 : 1;
}
