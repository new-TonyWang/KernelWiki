---
api: __ldcs
namespace: runtime
probe_slug: runtime-ldcs
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
  code: sources/experience/api-probes/artifacts/__ldcs_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ldcs_probe __ldcs_probe.cu
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
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24559-L24565
  excerpt: T __ldcs(const T* address); performs a load using the cache operator specified
    in the PTX ISA guide
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13000-L13012
  excerpt: .cs -- Cache streaming, likely to be accessed once. The ld.cs load cached
    streaming operation allocates global lines with evict-first policy in L1 and L2
    to limit cache pollution by temporary streaming data that may be accessed once
    or twice.
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.202976
  latency_ms_p10: 0.202688
  latency_ms_p90: 0.20384
  baseline_name: plain-global-load
  baseline_ms: 0.20432
  ratio: 1.0066
back_filled_into: []
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- __ldcs measured ~0.66% faster than plain load on this 256 MB HBM-bound streaming
  workload. The effect is small but consistent across p10/p50/p90. Its larger payoff
  is expected in multi-kernel or concurrent workloads where the evict-first policy
  keeps useful data resident in L2 for other consumers.
id: exp-2026-04-17-runtime-ldcs
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Ldcs
---
## Summary

End-to-end probe of `__ldcs` on H200 (sm_90a, CUDA 12.9). The probe allocates 64M floats (256 MB, > H200 L2 of ~60 MB), copies to device, launches a kernel that loads each element via `__ldcs`, adds 1.0, and stores the result. A baseline kernel using a plain global load is measured for comparison. The output is copied back to host and verified against a CPU reference (max_abs_err = 0). `__ldcs` measured 0.66% faster than plain load (ratio 1.0066) -- a small but consistent improvement across p10/p50/p90, plausibly from the evict-first cache policy reducing replacement traffic on this streaming workload.

## Minimal Kernel

```cuda
// Probe: __ldcs cache-streaming load (end-to-end)
// Loads N floats via __ldcs (PTX ld.cs: evict-first, streaming),
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

__global__ void ldcs_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldcs(in + tid) + 1.0f;
}

__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}
// host code: 5 warmup + 20 measured CUDA-event launches per kernel;
// D2H copy and CPU-reference verification. See the full file under
// sources/experience/api-probes/artifacts/__ldcs_probe.cu.
```

Full source: `sources/experience/api-probes/artifacts/__ldcs_probe.cu`.

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ldcs_probe __ldcs_probe.cu
```

## Measurement

Configuration: N = 67108864 (64M) floats = 256 MB, grid = 262144, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 67108864 | fp32 | 0.202976 | 0.202688 | 0.203840 | plain-global-load | 0.204320 | 1.0066 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__ldcs_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- PTX reference (Cache Operators, Table 30): `.cs` = cache streaming; allocates global lines with *evict-first* policy in L1 and L2 to limit cache pollution from one-shot/streaming data.
- CUDA Programming Guide §5.4.8.3 lists `T __ldcs(const T* address)` as a low-level load intrinsic whose cache-operator semantics are defined by the PTX ISA.
- Use `__ldcs` for data known to be read once and never reused within the kernel, or to protect a cache-resident working set owned by another part of the kernel/pipeline from being evicted by the streaming stream.
- When applied to a *Local* window address, PTX says `ld.cs` is reinterpreted as `ld.lu` (same semantic as `__ldlu` on global addresses).
- Supported types: all C++ fundamentals, CUDA vector types (except x3), and extended float types (`__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`).
