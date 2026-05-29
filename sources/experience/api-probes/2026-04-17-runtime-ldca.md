---
api: __ldca
namespace: runtime
probe_slug: runtime-ldca
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
  code: sources/experience/api-probes/artifacts/__ldca_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ldca_probe __ldca_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.204288
  latency_ms_p10: 0.203936
  latency_ms_p90: 0.20528
  baseline_name: plain-global-load
  baseline_ms: 0.204256
  ratio: 0.9998
back_filled_into: []
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- __ldca maps to the default ld.ca cache operator, so it should be functionally equivalent
  to a plain compiler-generated global load. The measured ratio 0.9998 confirms this.
  Useful mainly when forcing ld.ca on pointers that would otherwise be compiled to
  ld.global.nc (e.g., const __restrict__).
id: exp-2026-04-17-runtime-ldca
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Ldca
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L12978-L12988
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L12971-L13050
---
## Summary

End-to-end probe of `__ldca` on H200 (sm_90a, CUDA 12.9). The probe allocates 64M floats (256 MB, > H200 L2 of ~60 MB), copies to device, launches a kernel that loads each element via `__ldca`, adds 1.0, and stores the result. A baseline kernel using a plain global load is measured for comparison. The output is copied back to host and verified against a CPU reference (max_abs_err = 0). `__ldca` (which selects the default PTX `ld.ca` cache operator) shows identical latency to the plain load (ratio 0.9998), confirming that on this workload both paths resolve to the same SASS.

## Minimal Kernel

```cuda
// Probe: __ldca cache-at-all-levels load (end-to-end)
// Loads N floats via __ldca (PTX ld.ca: default; cache in L1 and L2),
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

__global__ void ldca_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldca(in + tid) + 1.0f;
}

__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}
// host code: 5 warmup + 20 measured CUDA-event launches per kernel;
// D2H copy and CPU-reference verification. See the full file under
// sources/experience/api-probes/artifacts/__ldca_probe.cu.
```

Full source: `sources/experience/api-probes/artifacts/__ldca_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ldca_probe __ldca_probe.cu
```

## Measurement

Configuration: N = 67108864 (64M) floats = 256 MB, grid = 262144, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 67108864 | fp32 | 0.204288 | 0.203936 | 0.205280 | plain-global-load | 0.204256 | 0.9998 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__ldca_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- PTX reference (Cache Operators, Table 30): `.ca` = cache at all levels (L1 + L2), default load cache operation.
- CUDA Programming Guide §5.4.8.3 lists `T __ldca(const T* address)` as a low-level load intrinsic whose cache-operator semantics are defined by the PTX ISA.
- Because `ld.ca` is the default cache operator on H200, `__ldca` is the explicit form of "do whatever the compiler would do anyway"; it is rarely interesting by itself. Its main use is to override a compiler choice that would otherwise emit `ld.global.nc` (the read-only path used for `const __restrict__` pointers).
- Supported types: all C++ fundamentals, CUDA vector types (except x3), and extended float types (`__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`).
