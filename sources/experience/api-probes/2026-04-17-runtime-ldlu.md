---
api: __ldlu
namespace: runtime
probe_slug: runtime-ldlu
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
  code: artifacts/experience/api-probes/artifacts/__ldlu_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ldlu_probe __ldlu_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.20288
  latency_ms_p10: 0.202496
  latency_ms_p90: 0.20384
  baseline_name: plain-global-load
  baseline_ms: 0.204096
  ratio: 1.006
back_filled_into: []
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- '__ldlu measured ~0.6% faster than plain load, essentially matching __ldcs (ratio 1.0066). This is consistent with the PTX spec: on global addresses ld.lu is defined to perform ld.cs. The intrinsic is primarily intended for local memory / spill restore, where ''last use'' lets the compiler avoid a write-back; that path is not exercised by a global-memory probe.'
id: exp-2026-04-17-runtime-ldlu
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Ldlu
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13013-L13022
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L12971-L13050
architectures:
- sm90
- sm90a
languages:
- ptx
- cuda-cpp
techniques:
- cache-policy
- register-budgeting
confidence: experimental
tags:
- cache-policy
- register-budgeting
- ptx
- cuda-cpp
artifact_dir: artifacts/experience/api-probes/artifacts
---
## Summary

End-to-end probe of `__ldlu` on H200 (sm_90a, CUDA 12.9). The probe allocates 64M floats (256 MB, > H200 L2 of ~60 MB), copies to device, launches a kernel that loads each element via `__ldlu`, adds 1.0, and stores the result. A baseline kernel using a plain global load is measured for comparison. The output is copied back to host and verified against a CPU reference (max_abs_err = 0). `__ldlu` measured 0.60% faster than plain load (ratio 1.0060) -- essentially matching `__ldcs` (ratio 1.0066), consistent with the PTX spec that on a global address `ld.lu` behaves as `ld.cs`.

## Minimal Kernel

```cuda
// Probe: __ldlu last-use load (end-to-end)
// Loads N floats via __ldlu (PTX ld.lu: last use; on global, maps to ld.cs),
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

__global__ void ldlu_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldlu(in + tid) + 1.0f;
}

__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}
// host code: 5 warmup + 20 measured CUDA-event launches per kernel;
// D2H copy and CPU-reference verification. See the full file under
// artifacts/experience/api-probes/artifacts/__ldlu_probe.cu.
```

Full source: `artifacts/experience/api-probes/artifacts/__ldlu_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ldlu_probe __ldlu_probe.cu
```

## Measurement

Configuration: N = 67108864 (64M) floats = 256 MB, grid = 262144, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 67108864 | fp32 | 0.202880 | 0.202496 | 0.203840 | plain-global-load | 0.204096 | 1.0060 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/__ldlu_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- PTX reference (Cache Operators, Table 30): `.lu` = last use. *"The compiler/programmer may use `ld.lu` when restoring spilled registers and popping function stack frames to avoid needless write-backs of lines that will not be used again. The `ld.lu` instruction performs a load cached streaming operation (`ld.cs`) on global addresses."*
- CUDA Programming Guide §5.4.8.3 lists `T __ldlu(const T* address)` as a low-level load intrinsic whose cache-operator semantics are defined by the PTX ISA.
- **On global memory, `__ldlu` == `__ldcs`** per the PTX spec; the probe's measured ratio (1.0060 vs 1.0066 for `__ldcs`) matches that equivalence within noise.
- The intrinsic's unique value lives on *Local* (spill / stack) addresses: the "last use" hint lets the compiler avoid a write-back of the line. For explicit kernel-level use on global memory, prefer `__ldcs` for clarity.
- Supported types: all C++ fundamentals, CUDA vector types (except x3), and extended float types (`__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`).
