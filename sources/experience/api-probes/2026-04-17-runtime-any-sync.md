---
api: __any_sync
namespace: runtime
probe_slug: runtime-any-sync
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
  code: sources/experience/api-probes/artifacts/__any_sync_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __any_sync_probe __any_sync_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L23864-L23895
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L24197-L24217
- path: corpus/nvidia/source-code/cuda-samples/Samples/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh
  line_range: L46-L80
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0
  latency_ms_median: 0.005024
  latency_ms_p10: 0.004832
  latency_ms_p90: 0.005376
  baseline_name: cpu-or-reduction-check
  baseline_ms: null
  ratio: null
back_filled_into:
- wiki/nvidia/api-definitions/runtime/__any_sync.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU OR-reduction check (correctness reference only, not a GPU timing
  baseline), so ratio is not reported.
- kp_introspect kernel-static failed (cuda-python not installed); no kernel introspection
  bundle available.
id: exp-2026-04-17-runtime-any-sync
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Any Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23864-L23895
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24197-L24217
- source_id: source-code/cuda-samples
  path: Samples/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh
  anchor: L46-L80
---
## Summary

End-to-end probe of `__any_sync` on H200 (sm_90a, CUDA 12.9). The probe launches 32 warps, each testing three scenarios: (1) all lanes predicate=0, expecting `__any_sync` to return zero; (2) only lane 0 predicate=1, others zero, expecting non-zero; (3) all lanes predicate=1, expecting non-zero. Results are copied back and verified against a CPU OR-reduction reference. Correctness: exact match after 0/1 normalization (max_abs_err = 0). Kernel latency median: 0.005024 ms for 32 warps.

## Minimal Kernel

```cuda
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

    int pred_none = 0;
    int r1 = __any_sync(0xFFFFFFFF, pred_none);

    int pred_one = (lane == 0) ? 1 : 0;
    int r2 = __any_sync(0xFFFFFFFF, pred_one);

    int pred_all = 1;
    int r3 = __any_sync(0xFFFFFFFF, pred_all);

    if (lane == 0) {
        out[warp_id * 3 + 0] = (r1 != 0) ? 1 : 0;
        out[warp_id * 3 + 1] = (r2 != 0) ? 1 : 0;
        out[warp_id * 3 + 2] = (r3 != 0) ? 1 : 0;
    }
}

int main() {
    int nwarps = N / WARP;
    int *h_out = (int*)malloc(nwarps * 3 * sizeof(int));
    int *h_ref = (int*)malloc(nwarps * 3 * sizeof(int));

    for (int w = 0; w < nwarps; w++) {
        int or1 = 0, or2 = 0, or3 = 0;
        for (int lane = 0; lane < WARP; lane++) {
            or1 |= 0;
            or2 |= (lane == 0) ? 1 : 0;
            or3 |= 1;
        }
        h_ref[w * 3 + 0] = (or1 != 0) ? 1 : 0;
        h_ref[w * 3 + 1] = (or2 != 0) ? 1 : 0;
        h_ref[w * 3 + 2] = (or3 != 0) ? 1 : 0;
    }

    int *d_out;
    CHECK(cudaMalloc(&d_out, nwarps * 3 * sizeof(int)));

    int grid = (N + BLK - 1) / BLK;

    for (int i = 0; i < 5; i++)
        any_sync_probe<<<grid, BLK>>>(d_out, nwarps);
    CHECK(cudaDeviceSynchronize());

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
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __any_sync_probe __any_sync_probe.cu
```

## Measurement

Configuration: N = 1024 threads (32 warps), grid = 4, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | int32 | 0.005024 | 0.004832 | 0.005376 | cpu-or-reduction-check | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__any_sync_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect kernel-static` bundle was generated (cuda-python not installed in this environment). Device-static info: NVIDIA H200, sm_9.0a, 132 SMs, CUDA 12.9, driver 570.124.06.

## Notes

- Upstream documentation: CUDA Programming Guide §5.4.6.2 (L23864-L23895) documents `__any_sync(unsigned mask, predicate)` as returning non-zero iff predicate is non-zero for one or more non-exited threads in mask. This is the OR-reduction sibling of `__all_sync` (AND-reduction) and `__ballot_sync` (per-lane bitmask).
- Warp __sync Intrinsic Constraints (L24197-L24217): mask must include all non-exited threads that reach the call; each calling thread must have its bit set in mask; all participating threads must use the same mask value.
- The CUDA Samples `simpleVoteIntrinsics` (simpleVote_kernel.cuh L46-L80) demonstrates warp vote functions with `mask = 0xffffffff` for warp-wide vote.
- `__any_sync` may return any non-zero integer (not necessarily `1`) when the OR is true; the probe normalizes the return to 0/1 before comparing with the CPU reference so the test is portable across future CUDA versions.
- The `0xFFFFFFFF` mask means all 32 lanes participate; this is the most efficient configuration per the programming guide.
- `__any_sync` does not provide memory ordering — use `__syncwarp()` or `__threadfence_block()` if memory visibility is needed after the vote.
