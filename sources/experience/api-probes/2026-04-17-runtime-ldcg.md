---
api: __ldcg
namespace: runtime
probe_slug: runtime-ldcg
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
  code: artifacts/experience/api-probes/artifacts/__ldcg_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ldcg_probe __ldcg_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.204256
  latency_ms_p10: 0.203968
  latency_ms_p90: 0.205088
  baseline_name: plain-global-load
  baseline_ms: 0.204256
  ratio: 1.0
back_filled_into: []
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- For a 256 MB buffer that exceeds H200 L2 (~60 MB), __ldcg matches the default ld.ca
  path within noise. The expected L1-bypass benefit would only show up on pollution-sensitive
  workloads or when coexisting with cache-resident data, which this single-kernel
  probe does not exercise.
id: exp-2026-04-17-runtime-ldcg
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Ldcg
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L12990-L13000
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L12971-L13050
---
## Summary

End-to-end probe of `__ldcg` on H200 (sm_90a, CUDA 12.9). The probe allocates 64M floats (256 MB, > H200 L2 of ~60 MB), copies to device, launches a kernel that loads each element via `__ldcg`, adds 1.0, and stores the result. A baseline kernel using a plain global load is measured for comparison. The output is copied back to host and verified against a CPU reference (max_abs_err = 0). On this HBM-bound streaming workload, `__ldcg` and the default load path show indistinguishable latency (ratio 1.0000), as expected: both end up fetching from HBM and caching (or not caching) in L2, and there is no reuse to benefit from either L1 residence or L1 bypass.

## Minimal Kernel

```cuda
// Probe: __ldcg cache-at-global-level load (end-to-end)
// Loads N floats via __ldcg (PTX ld.cg: cache in L2, bypass L1),
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

__global__ void ldcg_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldcg(in + tid) + 1.0f;
}

__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}
// host code: 5 warmup + 20 measured CUDA-event launches per kernel;
// D2H copy and CPU-reference verification. See the full file under
// artifacts/experience/api-probes/artifacts/__ldcg_probe.cu.
```

Full source: `artifacts/experience/api-probes/artifacts/__ldcg_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ldcg_probe __ldcg_probe.cu
```

## Measurement

Configuration: N = 67108864 (64M) floats = 256 MB, grid = 262144, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 67108864 | fp32 | 0.204256 | 0.203968 | 0.205088 | plain-global-load | 0.204256 | 1.0000 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/__ldcg_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- PTX reference (Cache Operators, Table 30): `.cg` = cache at global level, bypassing L1 and caching only in L2.
- CUDA Programming Guide §5.4.8.3 lists `T __ldcg(const T* address)` as a low-level load intrinsic whose cache-operator semantics are defined by the PTX ISA.
- On this streaming 256 MB workload, `__ldcg` is indistinguishable from the default path. The intrinsic's intended benefit (reduced L1 pollution) is not observable in a single-kernel microbench that neither reuses data nor coexists with cache-resident working sets.
- Supported types: all C++ fundamentals, CUDA vector types (except x3), and extended float types (`__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`).
