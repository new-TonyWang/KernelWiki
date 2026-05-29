---
api: __ballot_sync
namespace: runtime
probe_slug: runtime-ballot-sync
status: verified
kind: documented
trigger: init-sweep
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: 80-experience/api-probes/artifacts/__ballot_sync_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ballot_sync_probe __ballot_sync_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L23868-L23890
- path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cccl/cccl.md
  line_range: ''
- path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/CUDA Tools/compute-sanitizer/compute-sanitizer_index.html.md
  line_range: ''
- path: '{{CUDA_SAMPLES_REPO_REF}}/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu'
  line_range: L491-L507
- path: '{{LESSION_REPO_REF}}/04_warp_level_primitives/lesson.md'
  line_range: L195-L202
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23868-L23890
  excerpt: '__ballot_sync: returns a 32-bit mask where bit i is set if thread i''s
    predicate is non-zero'
conclusions:
  max_abs_err: 0
  latency_ms_median: 0.005216
  latency_ms_p10: 0.004992
  latency_ms_p90: 0.005536
  baseline_name: cpu-sequential-count
  baseline_ms: null
  ratio: null
back_filled_into:
- 10-api-raw/runtime/__ballot_sync.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU sequential count (correctness reference only, not a GPU timing
  baseline), so ratio is not reported.
id: exp-2026-04-16-runtime-ballot-sync
type: experience
vendor: nvidia
title: 2026 04 16 Runtime Ballot Sync
---
## Summary

End-to-end probe of `__ballot_sync` on H200 (sm_90a, CUDA 12.9). The probe allocates 1024 random floats, copies them to the device, and launches a kernel where each warp uses `__ballot_sync` to collect a bitmask of lanes where the value exceeds 0.5, then `__popc` counts the set bits. Results are copied back and verified against a CPU sequential count. Correctness: exact match (max_abs_err = 0). Kernel latency median: 0.005216 ms for 32 warps.

## Minimal Kernel

```cuda
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
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ballot_sync_probe __ballot_sync_probe.cu
```

## Measurement

Configuration: N = 1024 random floats, threshold = 0.5, 32 warps, grid = 4, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | fp32 | 0.005216 | 0.004992 | 0.005536 | cpu-sequential-count | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p knowledge/80-experience/api-probes/artifacts/__ballot_sync_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- Upstream documentation: CUDA Programming Guide section on warp vote functions (L23868-L23890) documents `__ballot_sync` as returning a 32-bit mask of per-lane predicates.
- The CUDA Samples reduction kernel (`reduction_kernel.cu` L491-507) uses `__ballot_sync` to compute a mask for the final warp reduction phase.
- The probe combines `__ballot_sync` with `__popc` (population count), which is the canonical pattern for counting how many warp lanes satisfy a condition.
- Correctness verification uses integer exact match since `__ballot_sync` returns a deterministic bitmask and `__popc` is exact.
- The `0xFFFFFFFF` mask means all 32 lanes participate; using a subset mask would zero out bits for non-participating lanes.
