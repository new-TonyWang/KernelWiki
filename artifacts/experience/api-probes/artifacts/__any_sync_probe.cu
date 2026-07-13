// Probe: __any_sync warp vote any (end-to-end)
// Each warp tests __any_sync with three scenarios:
//   1) all lanes predicate=0 → expect result=0
//   2) lane 0 predicate=1, others=0 → expect result!=0
//   3) all lanes predicate=1 → expect result!=0
// Results are verified against a CPU OR-reduction reference.
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

// Each warp: test __any_sync in three scenarios.
// out[warp_id * 3 + 0] = __any_sync when all predicates are 0
// out[warp_id * 3 + 1] = __any_sync when only lane 0 predicate is 1
// out[warp_id * 3 + 2] = __any_sync when all predicates are 1
__global__ void any_sync_probe(int* out, int nwarps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & (WARP - 1);
    int warp_id = tid / WARP;
    if (warp_id >= nwarps) return;

    // Scenario 1: all lanes predicate = 0
    int pred_none = 0;
    int r1 = __any_sync(0xFFFFFFFF, pred_none);

    // Scenario 2: only lane 0 predicate = 1, others = 0
    int pred_one = (lane == 0) ? 1 : 0;
    int r2 = __any_sync(0xFFFFFFFF, pred_one);

    // Scenario 3: all lanes predicate = 1
    int pred_all = 1;
    int r3 = __any_sync(0xFFFFFFFF, pred_all);

    if (lane == 0) {
        // Normalize to 0/1 for exact-match comparison with CPU OR-reduction.
        out[warp_id * 3 + 0] = (r1 != 0) ? 1 : 0;
        out[warp_id * 3 + 1] = (r2 != 0) ? 1 : 0;
        out[warp_id * 3 + 2] = (r3 != 0) ? 1 : 0;
    }
}

int main() {
    int nwarps = N / WARP;
    int *h_out = (int*)malloc(nwarps * 3 * sizeof(int));
    int *h_ref = (int*)malloc(nwarps * 3 * sizeof(int));

    // CPU reference via OR-reduction of per-lane predicates.
    for (int w = 0; w < nwarps; w++) {
        int or1 = 0, or2 = 0, or3 = 0;
        for (int lane = 0; lane < WARP; lane++) {
            or1 |= 0;
            or2 |= (lane == 0) ? 1 : 0;
            or3 |= 1;
        }
        h_ref[w * 3 + 0] = (or1 != 0) ? 1 : 0;  // 0
        h_ref[w * 3 + 1] = (or2 != 0) ? 1 : 0;  // 1
        h_ref[w * 3 + 2] = (or3 != 0) ? 1 : 0;  // 1
    }

    int *d_out;
    CHECK(cudaMalloc(&d_out, nwarps * 3 * sizeof(int)));

    int grid = (N + BLK - 1) / BLK;

    // Warmup: 5 launches
    for (int i = 0; i < 5; i++)
        any_sync_probe<<<grid, BLK>>>(d_out, nwarps);
    CHECK(cudaDeviceSynchronize());

    // Measurement: 20 launches via CUDA events
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        any_sync_probe<<<grid, BLK>>>(d_out, nwarps);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());

    // Verify correctness
    CHECK(cudaMemcpy(h_out, d_out, nwarps * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    int max_err = 0;
    for (int i = 0; i < nwarps * 3; i++)
        max_err = std::max(max_err, std::abs(h_out[i] - h_ref[i]));

    printf("PASS=%s  max_abs_err=%d\n", max_err == 0 ? "true" : "false", max_err);
    printf("median=%.6f p10=%.6f p90=%.6f ms\n", ms[10], ms[2], ms[18]);

    cudaFree(d_out);
    free(h_out); free(h_ref);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err == 0) ? 0 : 1;
}
