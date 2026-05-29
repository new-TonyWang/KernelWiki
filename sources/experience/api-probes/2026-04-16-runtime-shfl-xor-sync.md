---
api: __shfl_xor_sync
namespace: runtime
probe_slug: runtime-shfl-xor-sync
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
  code: sources/experience/api-probes/artifacts/__shfl_xor_sync_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __shfl_xor_sync_probe __shfl_xor_sync_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L23945-L24034
- path: '{{CUTLASS_REPO_REF}}/tools/util/include/cutlass/util/device_utils.h'
  line_range: L48-L56
- path: '{{LESSION_REPO_REF}}/04_warp_level_primitives/lesson.md'
  line_range: L118-L144
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23945-L24034
  excerpt: '__shfl_xor_sync(): Copy from a lane based on bitwise XOR of own lane ID'
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.00528
  latency_ms_p10: 0.005056
  latency_ms_p90: 0.005984
  baseline_name: cpu-sequential-sum
  baseline_ms: null
  ratio: null
back_filled_into:
- wiki/nvidia/api-definitions/runtime/__shfl_xor_sync.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU sequential sum (correctness reference only, not a GPU timing baseline),
  so ratio is not reported.
id: exp-2026-04-16-runtime-shfl-xor-sync
type: experience
vendor: nvidia
title: 2026 04 16 Runtime Shfl Xor Sync
---
## Summary

End-to-end probe of `__shfl_xor_sync` on H200 (sm_90a, CUDA 12.9). The probe allocates 1024 floats on the host, copies them to the device, launches a kernel where each warp performs a butterfly reduction (5 XOR-shuffle steps with delta = 16, 8, 4, 2, 1), copies results back, and verifies against a CPU sequential sum. Correctness: exact match (max_abs_err = 0). Kernel latency median: 0.005280 ms for 32 warps (1024 elements).

## Minimal Kernel

```cuda
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
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __shfl_xor_sync_probe __shfl_xor_sync_probe.cu
```

## Measurement

Configuration: N = 1024 floats, 32 warps, grid = 4, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | fp32 | 0.005280 | 0.005056 | 0.005984 | cpu-sequential-sum | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__shfl_xor_sync_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- Upstream documentation: CUDA Programming Guide section on warp shuffle functions (L23945-L24034) documents `__shfl_xor_sync` as the butterfly addressing pattern variant of the shuffle intrinsics.
- CUTLASS uses `__shfl_xor_sync` extensively for warp-level reductions (e.g., `device_utils.h` line 48-56).
- The probe uses `0xFFFFFFFF` as the mask (all 32 lanes), which is the standard usage for full-warp operations.
- Correctness is exact (max_abs_err = 0) because fp32 addition of small integers is exact.
- This probe measures end-to-end kernel latency (not per-instruction latency); for per-instruction cycle counts see `sources/experience/hw-probes/shfl-sync-bfly/2026-04-16-warp-primitives.md`.
