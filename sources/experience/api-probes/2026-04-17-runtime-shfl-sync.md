---
api: __shfl_sync
namespace: runtime
probe_slug: runtime-shfl-sync
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
  code: sources/experience/api-probes/artifacts/__shfl_sync_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __shfl_sync_probe __shfl_sync_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L23949-L23975
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L11966-L11966
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L24206-L24206
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.005248
  latency_ms_p10: 0.005056
  latency_ms_p90: 0.0056
  baseline_name: cpu-reference-permutation
  baseline_ms: null
  ratio: null
back_filled_into:
- wiki/nvidia/api-definitions/runtime/__shfl_sync.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- Baseline is a CPU reference permutation (correctness only, not a GPU timing baseline),
  so ratio is not reported.
id: exp-2026-04-17-runtime-shfl-sync
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Shfl Sync
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L23952-L23952
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11966-L11966
---
## Summary

End-to-end probe of `__shfl_sync` on H200 (sm_90a, CUDA 12.9). The probe allocates 1024 floats on the host, copies them to the device, and launches a kernel that exercises two canonical usage modes: (A) broadcast from lane 0 (`srcLane = 0`) to every lane in the same warp, and (B) a permutation where lane t reads from lane `(t + 5) % 32`. Results are copied back and verified against a CPU reference that does the same per-warp permutation. Correctness is exact (max_abs_err = 0). Kernel latency median: 0.005248 ms for 32 warps (1024 elements). This confirms the Programming Guide behavior: `__shfl_sync` copies the value held by `srcLane` within the warp, with `width = warpSize` default.

## Minimal Kernel

```cuda
// Probe: __shfl_sync direct-indexed lane copy (end-to-end)
// Demonstrates: host alloc -> H2D -> kernel -> D2H -> verify
//   Mode A: broadcast from lane 0 to all lanes.
//   Mode B: permutation -- lane t reads from lane (t + k) % 32.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N 1024  // one value per thread
#define BLK 256
#define WARP 32
#define K 5     // permutation shift within a warp
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void shfl_sync_kernel(const float* in, float* out_bcast,
                                 float* out_perm, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    float val = (tid < n) ? in[tid] : 0.0f;

    // A: broadcast from lane 0 (srcLane argument = 0)
    float b = __shfl_sync(0xFFFFFFFF, val, 0);

    // B: permutation -- srcLane = (lane + K) % 32
    int src = (lane + K) & (WARP - 1);
    float p = __shfl_sync(0xFFFFFFFF, val, src);

    if (tid < n) {
        out_bcast[tid] = b;
        out_perm[tid]  = p;
    }
}
```

(See `artifacts/__shfl_sync_probe.cu` for the full host harness including CPU reference, warmup, CUDA-event measurement, and verification.)

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __shfl_sync_probe __shfl_sync_probe.cu
```

## Measurement

Configuration: N = 1024 floats, 32 warps, grid = 4, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | fp32 | 0.005248 | 0.005056 | 0.005600 | cpu-reference-permutation | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__shfl_sync_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- Upstream documentation: CUDA Programming Guide section 5.4.6.5 "Warp Shuffle Functions" defines the signature `T __shfl_sync(unsigned mask, T value, int srcLane, int width=warpSize)` and describes it as copying the value held by `srcLane` within the warp.
- The Programming Guide further recommends `cuda::device::warp_shuffle()` (libcu++) as a safer, generalized alternative (L23949), but the builtin is still universally used and is what downstream skills wire up.
- Two modes were exercised:
  - Broadcast from lane 0 — the idiomatic pattern to propagate a per-warp scalar (e.g., `uniform_warp_id`, see Programming Guide L11966).
  - Permutation with `srcLane = (lane + 5) % 32` — a rotation within the warp, confirming that any in-warp lane index is a legal source.
- The mask `0xFFFFFFFF` selects all 32 lanes, which is the standard full-warp usage.
- Per-instruction (cycle-level) latency is out of scope here; see the warp-primitives hw-probe records under `sources/experience/hw-probes/` for that.
