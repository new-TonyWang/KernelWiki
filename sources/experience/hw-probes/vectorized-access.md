---
api: float4-vectorized-load
namespace: runtime
probe_slug: vectorized-access-bandwidth
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
  code: 80-experience/hw-probes/vectorized-access/artifacts/vectorized_load_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o vectorized_load_probe vectorized_load_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L22574-L22753
- path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L991-L1001
- path: 05-source-corpus/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1062
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22574-L22702
  excerpt: 'The following table details the byte size and alignment requirements of
    the vector types. float4: size 16, alignment 16.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L991-L1001
  excerpt: We evaluate the performance of both kernels using elements of size 4B,
    8B and 16B per thread i.e., using int, int2 and int4 for the template parameter.
    Overall, best performance is achieved when using asynchronous copies with an element
    of size 8 or 16 bytes.
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.0709
  latency_ms_p10: 0.0699
  latency_ms_p90: 0.0723
  baseline_name: scalar-float-load
  baseline_ms: 0.1835
  ratio: 2.59
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- The vectorized BW of 3783.77 GB/s is below peak HBM BW (4916.7 GB/s theoretical)
  due to non-saturated memory controller at 256 MB working set. The relative comparison
  between scalar and vectorized is the meaningful metric.
id: exp-vectorized-access
type: experience
vendor: nvidia
title: 2026 04 15 Vectorized Access
---
## Summary

This probe measures the **effective bandwidth difference** between scalar float loads and float4 vectorized loads on H200 (sm_90a, CUDA 12.9). A total of 67,108,864 floats (256 MB) are read from global memory, well above the 51.2 MB L2 cache to ensure HBM-bound traffic. In the scalar kernel, each of 67M threads loads one float (4 bytes). In the vectorized kernel, each of 16.8M threads loads one float4 (16 bytes, equivalent to 4 floats). The vectorized kernel achieves **2.59x higher effective bandwidth** than the scalar kernel.

## Minimal Kernel

```cuda
// Kernel A: scalar load -- each thread reads one float
__global__ void scalar_load_kernel(const float* __restrict__ input,
                                   float*       __restrict__ output,
                                   int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (tid < n) {
        val = input[tid];
    }
    // Warp-reduce to prevent DCE and avoid output bottleneck
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    if (threadIdx.x % 32 == 0) {
        int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        output[warpId] = val;
    }
}

// Kernel B: vectorized float4 load -- each thread reads one float4 (4 floats)
__global__ void vectorized_load_kernel(const float4* __restrict__ input,
                                       float*        __restrict__ output,
                                       int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (tid < n4) {
        float4 v = input[tid];
        val = v.x + v.y + v.z + v.w;
    }
    // Warp-reduce to prevent DCE and avoid output bottleneck
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    if (threadIdx.x % 32 == 0) {
        int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        output[warpId] = val;
    }
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o vectorized_load_probe vectorized_load_probe.cu
```

## Measurement

Configuration: 67,108,864 floats (256 MB), well above H200 L2 cache (51.2 MB). Scalar kernel: 262,144 blocks x 256 threads. Vectorized kernel: 65,536 blocks x 256 threads. Warmup: 5 launches discarded. Repeats: 20.

Correctness: both kernels produce the same total sum (67,108,864.0) with absolute error 0.0. PASS.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 67108864 (scalar) | float32 | 0.1835 | 0.1833 | 0.1841 | scalar-float-load | 0.1835 | 1.00 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o vectorized_load_probe vectorized_load_probe.cu && ./vectorized_load_probe` |
| 67108864 (float4) | float32 | 0.0709 | 0.0699 | 0.0723 | scalar-float-load | 0.1835 | 2.59 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o vectorized_load_probe vectorized_load_probe.cu && ./vectorized_load_probe` |

Effective bandwidth:
- Scalar (float):      1462.96 GB/s
- Vectorized (float4): 3783.77 GB/s
- Ratio: **2.59x** speedup with vectorized access

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- The probe uses CUDA events for timing, consistent with `benchmark-protocol.md` Pillar 2.
- The warp-level shuffle reduction in both kernels prevents dead-code elimination while avoiding output-memory bottleneck (one write per 32 threads).
- The working set (256 MB) exceeds the H200 L2 cache (51.2 MB), ensuring that all accesses hit HBM.
- The scalar kernel launches 262,144 blocks (67M threads), while the vectorized kernel launches 65,536 blocks (16.8M threads). The vectorized kernel needs 4x fewer threads because each thread processes 4 elements. This reduction in thread/block count also reduces kernel launch overhead and scheduler pressure.
- The 2.59x speedup is consistent with the expected benefit: a float4 load issues a single 128-bit (LDG.128) instruction per thread, whereas the scalar kernel issues a 32-bit (LDG.32) instruction per thread. The vectorized version uses 4x fewer load instructions to move the same data.
- The vectorized bandwidth (3783.77 GB/s) reaches 77% of the theoretical HBM peak (4916.7 GB/s). The scalar bandwidth (1462.96 GB/s) reaches only 30% of peak, limited by instruction issue rate for the large number of individual load instructions.
- Timing variance is very low (p10/p90 within 3% of median for both kernels), indicating stable measurement despite `clock_policy: unknown`.
