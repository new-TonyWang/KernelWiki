// Probe: __shfl_xor_sync butterfly warp reduction (end-to-end)
// Demonstrates: host alloc -> H2D -> kernel -> D2H -> verify
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024  // one value per thread
#define BLK 256
#define WARP 32
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__device__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}

// Each warp reduces its 32 elements; lane 0 writes result.
__global__ void shfl_xor_reduce(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    float sum = warp_reduce_sum(val);
    if ((threadIdx.x & (WARP - 1)) == 0 && tid < n)
        out[tid / WARP] = sum;
}

int main() {
    int nwarps = N / WARP;
    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(nwarps * sizeof(float));
    float *h_ref = (float*)calloc(nwarps, sizeof(float));
    for (int i = 0; i < N; i++) { h_in[i] = 1.0f + (i % WARP) * 0.01f; }
    for (int i = 0; i < N; i++) h_ref[i / WARP] += h_in[i];

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, nwarps * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    // warmup
    for (int i = 0; i < 5; i++) shfl_xor_reduce<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    // measure
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        shfl_xor_reduce<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());

    CHECK(cudaMemcpy(h_out, d_out, nwarps * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < nwarps; i++)
        max_err = fmax(max_err, fabsf(h_out[i] - h_ref[i]));

    printf("PASS=%s  max_abs_err=%.6e\n", max_err <= 1e-5 ? "true" : "false", max_err);
    printf("median=%.6f p10=%.6f p90=%.6f ms\n", ms[10], ms[2], ms[18]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err <= 1e-5) ? 0 : 1;
}
