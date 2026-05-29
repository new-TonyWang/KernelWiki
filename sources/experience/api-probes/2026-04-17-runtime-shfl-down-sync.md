---
api: __shfl_down_sync
namespace: runtime
probe_slug: runtime-shfl-down-sync
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
  code: artifacts/experience/api-probes/artifacts/__shfl_down_sync_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __shfl_down_sync_probe __shfl_down_sync_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.005152
  latency_ms_p10: 0.00496
  latency_ms_p90: 0.0056
  baseline_name: cpu-reference-permutation
  baseline_ms: null
  ratio: null
back_filled_into:
- wiki/nvidia/api-definitions/runtime/__shfl_down_sync.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU reference permutation (correctness only, not a GPU timing baseline),
  so ratio is not reported.
id: exp-2026-04-17-runtime-shfl-down-sync
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Shfl Down Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23954-L23954
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23973-L23973
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23954-L23954
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23973-L23973
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24011-L24025
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24206-L24206
---
## Summary

End-to-end probe of `__shfl_down_sync` on H200 (sm_90a, CUDA 12.9). The probe allocates 1024 floats on the host, launches a kernel that performs three shifts (delta = 1, 4, 16), copies results back, and compares against a CPU reference. Correctness is exact (max_abs_err = 0). Kernel latency median: 0.005152 ms for 32 warps, three shifts per lane (1024 elements total). The probe confirms the documented boundary behavior: each lane reads from `lane + delta`; lanes where `lane + delta >= 32` keep their own value (the source lane ID does not wrap).

## Minimal Kernel

```cuda
// Probe: __shfl_down_sync (shift down by delta; lane >= 32-delta keep own)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024
#define BLK 256
#define WARP 32
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void shfl_down_kernel(const float* in, float* o1, float* o4,
                                 float* o16, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;
    float v1  = __shfl_down_sync(0xFFFFFFFF, val, 1);
    float v4  = __shfl_down_sync(0xFFFFFFFF, val, 4);
    float v16 = __shfl_down_sync(0xFFFFFFFF, val, 16);
    if (tid < n) { o1[tid] = v1; o4[tid] = v4; o16[tid] = v16; }
}
```

(See `artifacts/__shfl_down_sync_probe.cu` for the full host harness including CPU reference, warmup, CUDA-event measurement, and verification.)

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __shfl_down_sync_probe __shfl_down_sync_probe.cu
```

## Measurement

Configuration: N = 1024 floats, 32 warps, grid = 4, block = 256. Three `__shfl_down_sync` calls per thread (delta = 1, 4, 16). 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | fp32 | 0.005152 | 0.004960 | 0.005600 | cpu-reference-permutation | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/__shfl_down_sync_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- Upstream documentation: CUDA Programming Guide section 5.4.6.5 "Warp Shuffle Functions" defines the signature `T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width=warpSize)` and states: "Copy from a lane with a higher ID than the caller's. The intrinsic function calculates a source lane ID by adding `delta` to the caller's lane ID ... this has the effect of shifting `value` down the warp by `delta` lanes."
- Boundary behavior verified by this probe: lanes where `lane + delta >= 32` keep their own value — consistent with the Programming Guide's rule that the source lane ID does not wrap around the width.
- Typical usage pattern: warp-level reduction (shift-then-combine halving pattern), and reading "next" lane values for stencil-like computations.
- The mask `0xFFFFFFFF` selects all 32 lanes, which is the standard full-warp usage.
- `__shfl_down_sync` is the classic warp reduction primitive; the butterfly-reduction sibling `__shfl_xor_sync` (see `2026-04-16-runtime-shfl-xor-sync.md`) is generally preferred when all lanes need the final reduced value, while `__shfl_down_sync` is sufficient when only lane 0 needs the final result.
- Per-instruction (cycle-level) latency is out of scope here; see the warp-primitives hw-probe records under `artifacts/experience/hw-probes/` for that.
