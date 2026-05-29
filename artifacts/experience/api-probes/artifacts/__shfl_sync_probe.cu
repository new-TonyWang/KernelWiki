// Probe: __shfl_sync direct-indexed lane copy (end-to-end)
// Demonstrates: host alloc -> H2D -> kernel -> D2H -> verify
//   Mode A: broadcast from lane 0 to all lanes.
//   Mode B: permutation -- lane t reads from lane (t + k) % 32.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024  // one value per thread
#define BLK 256
#define WARP 32
#define K 5     // permutation shift within a warp
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Each lane writes:
//   out_bcast[tid] = value held by lane 0 of the same warp
//   out_perm[tid]  = value held by lane ((lane + K) % 32) of the same warp
__global__ void shfl_sync_kernel(const float* in, float* out_bcast,
                                 float* out_perm, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    float val = (tid < n) ? in[tid] : 0.0f;

    // A: broadcast from lane 0 (srcLane argument = 0)
    float b = __shfl_sync(0xFFFFFFFF, val, 0);

    // B: permutation -- srcLane = (lane + K) % 32
    int src = (lane + K) & (WARP - 1);
    float p = __shfl_sync(0xFFFFFFFF, val, src);

    if (tid < n) {
        out_bcast[tid] = b;
        out_perm[tid]  = p;
    }
}

int main() {
    float *h_in    = (float*)malloc(N * sizeof(float));
    float *h_bcast = (float*)malloc(N * sizeof(float));
    float *h_perm  = (float*)malloc(N * sizeof(float));
    float *h_refB  = (float*)malloc(N * sizeof(float));
    float *h_refP  = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) h_in[i] = 1.0f + (i % WARP) * 0.01f;
    // CPU reference
    for (int i = 0; i < N; i++) {
        int warp_base = (i / WARP) * WARP;
        int lane = i & (WARP - 1);
        h_refB[i] = h_in[warp_base + 0];                       // broadcast from lane 0
        h_refP[i] = h_in[warp_base + ((lane + K) & (WARP-1))]; // permutation
    }

    float *d_in, *d_b, *d_p;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_b,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_p,  N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    for (int i = 0; i < 5; i++) shfl_sync_kernel<<<grid, BLK>>>(d_in, d_b, d_p, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        shfl_sync_kernel<<<grid, BLK>>>(d_in, d_b, d_p, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());

    CHECK(cudaMemcpy(h_bcast, d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_perm,  d_p, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        max_err = fmax(max_err, fabsf(h_bcast[i] - h_refB[i]));
        max_err = fmax(max_err, fabsf(h_perm[i]  - h_refP[i]));
    }

    printf("PASS=%s  max_abs_err=%.6e\n", max_err <= 1e-5 ? "true" : "false", max_err);
    printf("median=%.6f p10=%.6f p90=%.6f ms\n", ms[10], ms[2], ms[18]);

    cudaFree(d_in); cudaFree(d_b); cudaFree(d_p);
    free(h_in); free(h_bcast); free(h_perm); free(h_refB); free(h_refP);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err <= 1e-5) ? 0 : 1;
}
