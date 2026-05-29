// Probe: __ballot_sync warp vote ballot (end-to-end)
// Each warp counts how many of its lanes satisfy a predicate using
// __ballot_sync + __popc, then compares against a CPU reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024
#define BLK 256
#define WARP 32
#define THRESH 0.5f
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Each warp: ballot lanes where in[tid] > THRESH, lane 0 writes popcount.
__global__ void ballot_count(const float* in, int* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pred = (tid < n && in[tid] > THRESH) ? 1 : 0;
    unsigned mask = __ballot_sync(0xFFFFFFFF, pred);
    if ((threadIdx.x & (WARP - 1)) == 0)
        out[tid / WARP] = __popc(mask);
}

int main() {
    int nwarps = N / WARP;
    float *h_in = (float*)malloc(N * sizeof(float));
    int   *h_out = (int*)malloc(nwarps * sizeof(int));
    int   *h_ref = (int*)calloc(nwarps, sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N; i++)
        if (h_in[i] > THRESH) h_ref[i / WARP]++;

    float *d_in; int *d_out;
    CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, nwarps * sizeof(int)));
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    for (int i = 0; i < 5; i++) ballot_count<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        ballot_count<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());

    CHECK(cudaMemcpy(h_out, d_out, nwarps * sizeof(int), cudaMemcpyDeviceToHost));
    int max_err = 0;
    for (int i = 0; i < nwarps; i++)
        max_err = abs(h_out[i] - h_ref[i]) > max_err ?
                  abs(h_out[i] - h_ref[i]) : max_err;

    printf("PASS=%s  max_abs_err=%d\n", max_err == 0 ? "true" : "false", max_err);
    printf("median=%.6f p10=%.6f p90=%.6f ms\n", ms[10], ms[2], ms[18]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err == 0) ? 0 : 1;
}
