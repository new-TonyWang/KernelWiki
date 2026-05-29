---
api: __all_sync
namespace: runtime
probe_slug: runtime-all-sync
status: verified
kind: documented
trigger: init-sweep
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f4646b8b9c4f5a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: artifacts/experience/api-probes/artifacts/__all_sync_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __all_sync_probe __all_sync_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0
  latency_ms_median: 0.005152
  latency_ms_p10: 0.00496
  latency_ms_p90: 0.005344
  baseline_name: cpu-sequential-check
  baseline_ms: null
  ratio: null
back_filled_into:
- wiki/nvidia/api-definitions/runtime/__all_sync.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU sequential check (correctness reference only, not a GPU timing
  baseline), so ratio is not reported.
- kp_introspect kernel-static failed (cuda-python not installed); no kernel introspection
  bundle available.
id: exp-2026-04-17-runtime-all-sync
type: experience
vendor: nvidia
title: 2026 04 17 Runtime All Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23872-L23882
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24197-L24217
- source_id: source-code/cuda-samples
  path: Samples/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh
  anchor: L55-L80
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23872-L23882
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24197-L24217
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24232-L24295
---
## Summary

End-to-end probe of `__all_sync` on H200 (sm_90a, CUDA 12.9). The probe launches 32 warps, each testing two scenarios: (1) all lanes predicate=1, expecting `__all_sync` to return non-zero; (2) lane 0 predicate=0 while others predicate=1, expecting `__all_sync` to return zero. Results are copied back and verified against a CPU reference. Correctness: exact match (max_abs_err = 0). Kernel latency median: 0.005152 ms for 32 warps.

## Minimal Kernel

```cuda
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
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __all_sync_probe __all_sync_probe.cu
```

## Measurement

Configuration: N = 1024 threads (32 warps), grid = 4, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | int32 | 0.005152 | 0.004960 | 0.005344 | cpu-sequential-check | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/__all_sync_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect kernel-static` bundle was generated (cuda-python not installed in this environment). Device-static info: NVIDIA H200, sm_9.0a, 132 SMs, CUDA 12.9, driver 570.124.06.

## Notes

- Upstream documentation: CUDA Programming Guide §5.4.6.2 (L23872-L23882) documents `__all_sync(unsigned mask, predicate)` as returning non-zero iff predicate is non-zero for all non-exited threads in mask.
- Warp __sync Intrinsic Constraints (L24197-L24217): mask must include all non-exited threads that reach the call; each calling thread must have its bit set in mask; all participating threads must use the same mask value.
- The CUDA Samples `simpleVoteIntrinsics` (simpleVote_kernel.cuh L55-L80) demonstrates `__all_sync` with `mask = 0xffffffff` for warp-wide vote.
- Correctness verification uses integer exact match (int32 tolerance = 0).
- The `0xFFFFFFFF` mask means all 32 lanes participate; this is the most efficient configuration per the programming guide.
- `__all_sync` does not provide memory ordering — use `__syncwarp()` or `__threadfence_block()` if memory visibility is needed after the vote.