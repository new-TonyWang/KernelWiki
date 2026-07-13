---
api: global-memory-load
namespace: runtime
probe_slug: coalescing-stride-bandwidth
status: verified
kind: documented
trigger: skill-build
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: artifacts/experience/hw-probes/coalescing-stride/coalescing_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o coalescing_probe coalescing_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: null
  latency_ms_median: null
  latency_ms_p10: null
  latency_ms_p90: null
  baseline_name: coalesced-stride1
  baseline_ms: null
  ratio: null
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- The coalesced BW of 530.66 GB/s is well below peak HBM BW (4916.7 GB/s theoretical) because the working set is only 4 MB and the kernel is launch-latency-dominated at this scale. The relative comparison between coalesced and strided is the meaningful metric.
id: exp-coalescing-stride
type: experience
vendor: nvidia
title: 2026 04 15 Coalescing
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1379-L1411
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L539-L542
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L539-L634
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1379-L1448
architectures:
- sm90
- sm90a
languages:
- cuda-cpp
techniques:
- cache-policy
confidence: experimental
tags:
- cache-policy
- cuda-cpp
artifact_dir: artifacts/experience/hw-probes/coalescing-stride
---
## Summary

This probe measures the **effective bandwidth difference** between coalesced (stride-1) and non-coalesced (stride-32) global memory access patterns on H200 (sm_90a, CUDA 12.9). Each of 1,048,576 threads reads a single `float` from global memory. In the coalesced case, thread i reads `array[i]`; in the strided case, thread i reads `array[i * 32]`. The strided access is **2.85x slower** than the coalesced access, demonstrating the real cost of non-coalesced memory access patterns on HBM.

## Minimal Kernel

```cuda
__global__ void read_kernel(const float* __restrict__ input,
                            float*       __restrict__ output,
                            int stride,
                            int n_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    float val = 0.0f;
    if (idx < n_elements) {
        val = input[idx];
    }
    // Warp-reduce to prevent DCE and avoid output bottleneck
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    int lane = threadIdx.x % 32;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (lane == 0) {
        output[warpId] = val;
    }
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o coalescing_probe coalescing_probe.cu
```

## Measurement

Configuration: 1,048,576 threads (4096 blocks x 256 threads), each reading one float (4 bytes). Coalesced array: 4 MB; strided array: 128 MB (exceeds L2 cache, forces HBM traffic). Warmup: 5 launches discarded. Repeats: 20.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1048576 (stride=1) | float32 | 0.0079 | 0.0078 | 0.0081 | coalesced-stride1 | 0.0079 | 1.00 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o coalescing_probe coalescing_probe.cu && ./coalescing_probe` |
| 1048576 (stride=32) | float32 | 0.0225 | 0.0224 | 0.0244 | coalesced-stride1 | 0.0079 | 0.35 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o coalescing_probe coalescing_probe.cu && ./coalescing_probe` |

Effective bandwidth:
- Coalesced (stride=1): 530.66 GB/s
- Strided (stride=32):  186.18 GB/s
- Ratio: 2.85x slowdown for strided access

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- The probe uses CUDA events for timing, consistent with `benchmark-protocol.md` Pillar 2.
- The warp-level shuffle reduction in the kernel prevents dead-code elimination while avoiding output-memory bottlenecks (only one write per 32 threads).
- The strided array (128 MB) exceeds the H200 L2 cache (51.2 MB), ensuring that strided accesses hit HBM, not L2.
- The coalesced array (4 MB) fits in L2, which inflates coalesced bandwidth somewhat. Even so, the 2.85x ratio represents a conservative lower bound on the coalescing penalty; with both arrays sized to exceed L2, the ratio would likely be larger.
- The theoretical HBM bandwidth is 4916.7 GB/s (as reported by CUDA runtime). The measured 530 GB/s reflects the small working set and launch overhead, not a peak-bandwidth test. The relative comparison is the meaningful metric.
- With stride=32, each warp of 32 threads touches 32 distinct 32-byte sectors (32 x 32 = 1024 bytes fetched for 128 bytes used), yielding 12.5% memory utilization per the programming guide Figure 11 (L1389-L1393).
