---
api: __ldcv
namespace: runtime
probe_slug: runtime-ldcv
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
  code: sources/experience/api-probes/artifacts/__ldcv_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ldcv_probe __ldcv_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L24559-L24565
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  line_range: L12971-L13050
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.20432
  latency_ms_p10: 0.204096
  latency_ms_p90: 0.205632
  baseline_name: plain-global-load
  baseline_ms: 0.204096
  ratio: 0.9989
back_filled_into: []
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Counter to the MVP task's expected finding, __ldcv was NOT observably slower than
  plain load on this 256 MB HBM-bound streaming workload. This is because the plain-load
  baseline already misses L2 (256 MB > 60 MB L2), so bypassing cache has no extra
  cost here. __ldcv's penalty would show up if the same address were re-read (plain
  path hits L2; __ldcv re-fetches from HBM).
- PTX spec describes __ldcv in the context of System Memory lines; its interaction
  with ordinary device global memory is less rigorously specified and may be implemented
  as a plain uncached load rather than as an L2-line invalidation.
id: exp-2026-04-17-runtime-ldcv
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Ldcv
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13023-L13032
---
## Summary

End-to-end probe of `__ldcv` on H200 (sm_90a, CUDA 12.9). The probe allocates 64M floats (256 MB, > H200 L2 of ~60 MB), copies to device, launches a kernel that loads each element via `__ldcv`, adds 1.0, and stores the result. A baseline kernel using a plain global load is measured for comparison. The output is copied back to host and verified against a CPU reference (max_abs_err = 0). Unexpectedly, `__ldcv` runs at the same speed as plain load (ratio 0.9989): on a strictly streaming workload that already misses L2, "bypass / refetch" adds no extra cost because the default path would have missed anyway.

## Minimal Kernel

```cuda
// Probe: __ldcv cache-volatile load (end-to-end)
// Loads N floats via __ldcv (PTX ld.cv: don't cache, fetch again),
// adds 1.0, stores to output. Verifies against CPU reference.
// Baseline: plain global load (default ld.ca path).
// Buffer size: 256 MB (> H200 L2 ~60 MB) so the load path is HBM-bound.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N (1 << 26)  // 64M elements -> 256 MB for fp32
#define BLK 256

__global__ void ldcv_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldcv(in + tid) + 1.0f;
}

__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}
// host code: 5 warmup + 20 measured CUDA-event launches per kernel;
// D2H copy and CPU-reference verification. See the full file under
// sources/experience/api-probes/artifacts/__ldcv_probe.cu.
```

Full source: `sources/experience/api-probes/artifacts/__ldcv_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ldcv_probe __ldcv_probe.cu
```

## Measurement

Configuration: N = 67108864 (64M) floats = 256 MB, grid = 262144, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 67108864 | fp32 | 0.204320 | 0.204096 | 0.205632 | plain-global-load | 0.204096 | 0.9989 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__ldcv_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- PTX reference (Cache Operators, Table 30): `.cv` = "don't cache and fetch again (consider cached system memory lines stale, fetch again)"; on System Memory addresses, `ld.cv` invalidates the matching L2 line and refetches.
- CUDA Programming Guide §5.4.8.3 lists `T __ldcv(const T* address)` as a low-level load intrinsic whose cache-operator semantics are defined by the PTX ISA.
- **Expected-vs-measured**: the MVP task packet predicted `__ldcv` would be *slower* than plain load because it bypasses L1 entirely. The measurement contradicts that: on a 256 MB streaming workload both paths are HBM-bound, so "bypass" costs nothing. The penalty is real only when the same address would otherwise hit in L2 -- e.g., polling a flag, re-reading a header, or mailbox-style inter-grid communication -- which a single-read streaming probe cannot exercise.
- Primary use case: forcing a fresh fetch of memory that peer devices or the host may have just updated (volatile-like semantics for System-Memory-backed addresses).
- Supported types: all C++ fundamentals, CUDA vector types (except x3), and extended float types (`__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`).
