// Probe: __all_sync warp vote all (end-to-end)
// Each warp tests __all_sync with two scenarios:
//   1) all lanes predicate=1 → expect result=1
//   2) some lanes predicate=0 → expect result=0
// Results are verified against a CPU reference.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024
#define BLK 256
#define WARP 32
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Each warp: test __all_sync in two scenarios.
// out[warp_id * 2 + 0] = __all_sync result when all predicates are 1
// out[warp_id * 2 + 1] = __all_sync result when lane 0 predicate is 0
__global__ void all_sync_probe(int* out, int nwarps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp_id = tid / WARP;
    if (warp_id >= nwarps) return;

    // Scenario 1: all lanes predicate = 1
    int pred_all = 1;
    int r1 = __all_sync(0xFFFFFFFF, pred_all);

    // Scenario 2: lane 0 predicate = 0, others = 1
    int pred_mixed = (lane == 0) ? 0 : 1;
    int r2 = __all_sync(0xFFFFFFFF, pred_mixed);

    if (lane == 0) {
        out[warp_id * 2 + 0] = r1;
        out[warp_id * 2 + 1] = r2;
    }
}

int main() {
    int nwarps = N / WARP;
    int *h_out = (int*)malloc(nwarps * 2 * sizeof(int));
    int *h_ref = (int*)malloc(nwarps * 2 * sizeof(int));

    // CPU reference: scenario 1 → all true → 1; scenario 2 → not all true → 0
    for (int w = 0; w < nwarps; w++) {
        h_ref[w * 2 + 0] = 1;  // all predicates true
        h_ref[w * 2 + 1] = 0;  // lane 0 predicate false
    }

    int *d_out;
    CHECK(cudaMalloc(&d_out, nwarps * 2 * sizeof(int)));

    int grid = (N + BLK - 1) / BLK;

    // Warmup: 5 launches
    for (int i = 0; i < 5; i++)
        all_sync_probe<<<grid, BLK>>>(d_out, nwarps);
    CHECK(cudaDeviceSynchronize());

    // Measurement: 20 launches via CUDA events
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        all_sync_probe<<<grid, BLK>>>(d_out, nwarps);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());

    // Verify correctness
    CHECK(cudaMemcpy(h_out, d_out, nwarps * 2 * sizeof(int), cudaMemcpyDeviceToHost));
    int max_err = 0;
    for (int i = 0; i < nwarps * 2; i++)
        max_err = std::max(max_err, std::abs(h_out[i] - h_ref[i]));

    printf("PASS=%s  max_abs_err=%d\n", max_err == 0 ? "true" : "false", max_err);
    printf("median=%.6f p10=%.6f p90=%.6f ms\n", ms[10], ms[2], ms[18]);

    cudaFree(d_out);
    free(h_out); free(h_ref);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err == 0) ? 0 : 1;
}