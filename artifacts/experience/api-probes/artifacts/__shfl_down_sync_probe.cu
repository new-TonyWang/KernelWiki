// Probe: __shfl_down_sync (shift down by delta; lanes with lane >= 32-delta keep own)
// Demonstrates: host alloc -> H2D -> kernel -> D2H -> verify
// Three deltas are tested in one kernel: 1, 4, 16.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024
#define BLK 256
#define WARP 32
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Each lane reads from (lane + delta). Lanes with lane + delta >= 32 keep own
// value (source lane ID does not wrap; upper `delta` lanes remain unchanged).
__global__ void shfl_down_kernel(const float* in, float* o1, float* o4,
                                 float* o16, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    float v1  = __shfl_down_sync(0xFFFFFFFF, val, 1);
    float v4  = __shfl_down_sync(0xFFFFFFFF, val, 4);
    float v16 = __shfl_down_sync(0xFFFFFFFF, val, 16);
    if (tid < n) { o1[tid] = v1; o4[tid] = v4; o16[tid] = v16; }
}

static void ref_down(const float* in, float* out, int n, int delta) {
    for (int i = 0; i < n; i++) {
        int warp_base = (i / WARP) * WARP;
        int lane = i & (WARP - 1);
        int src_lane = lane + delta;
        if (src_lane >= WARP) out[i] = in[i];                    // own value
        else                  out[i] = in[warp_base + src_lane];
    }
}

int main() {
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h1   = (float*)malloc(N * sizeof(float));
    float *h4   = (float*)malloc(N * sizeof(float));
    float *h16  = (float*)malloc(N * sizeof(float));
    float *r1   = (float*)malloc(N * sizeof(float));
    float *r4   = (float*)malloc(N * sizeof(float));
    float *r16  = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) h_in[i] = 1.0f + (i % WARP) * 0.01f;
    ref_down(h_in, r1,  N, 1);
    ref_down(h_in, r4,  N, 4);
    ref_down(h_in, r16, N, 16);

    float *d_in, *d1, *d4, *d16;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d1,   N * sizeof(float)));
    CHECK(cudaMalloc(&d4,   N * sizeof(float)));
    CHECK(cudaMalloc(&d16,  N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    for (int i = 0; i < 5; i++) shfl_down_kernel<<<grid, BLK>>>(d_in, d1, d4, d16, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        shfl_down_kernel<<<grid, BLK>>>(d_in, d1, d4, d16, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());

    CHECK(cudaMemcpy(h1,  d1,  N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h4,  d4,  N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h16, d16, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0;
    for (int i = 0; i < N; i++) {
        max_err = fmax(max_err, fabsf(h1[i]  - r1[i]));
        max_err = fmax(max_err, fabsf(h4[i]  - r4[i]));
        max_err = fmax(max_err, fabsf(h16[i] - r16[i]));
    }
    printf("PASS=%s  max_abs_err=%.6e\n", max_err <= 1e-5 ? "true" : "false", max_err);
    printf("median=%.6f p10=%.6f p90=%.6f ms\n", ms[10], ms[2], ms[18]);

    cudaFree(d_in); cudaFree(d1); cudaFree(d4); cudaFree(d16);
    free(h_in); free(h1); free(h4); free(h16); free(r1); free(r4); free(r16);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err <= 1e-5) ? 0 : 1;
}
