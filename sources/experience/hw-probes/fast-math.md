---
api: __expf
namespace: math
probe_slug: expf-vs-fast-expf
status: verified
kind: documented
trigger: skill-build
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: ''
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: sources/experience/hw-probes/fast-math/artifacts/expf_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o expf_probe expf_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L26993-L26995
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L27041-L27094
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1243-L1244
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1556-L1581
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: null
  latency_ms_median: 0.0404
  latency_ms_p10: 0.0404
  latency_ms_p90: 0.0406
  baseline_name: expf (standard math)
  baseline_ms: 0.0516
  ratio: 1.27
back_filled_into: []
open_questions:
- clock_policy is unknown; clocks were not locked during measurement.
- The 1.27x speedup reflects a kernel where each thread calls expf 128 times with
  an fma rescale between iterations. In memory-bound kernels the speedup will be near
  zero.
id: exp-fast-math
type: experience
vendor: nvidia
title: 2026 04 16 Fast Math
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L26993-L26995
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1243-L1244
---
## Summary

On NVIDIA H200 (sm_90a, CUDA 12.9, driver 570.124.06), `__expf` achieves 1.27x higher throughput than `expf` (3318 Gop/s vs 2604 Gop/s) when the kernel is compute-bound (128 chained calls per thread, 1M threads). The precision cost is modest: `__expf` has a max observed ULP error of 8 vs double-precision reference (over 1M random inputs in [-10, 10]), compared to 2 ULP for standard `expf`.

## Minimal Kernel

```cuda
constexpr int ITERS = 128;
constexpr float SCALE = 8.0f / 54.598f;
constexpr float BIAS  = -4.0f;

// Standard expf kernel
__global__ void kernel_expf(const float* __restrict__ in,
                            float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = in[i];
    #pragma unroll 8
    for (int iter = 0; iter < ITERS; ++iter) {
        acc = expf(acc) * SCALE + BIAS;
    }
    out[i] = acc;
}

// Fast __expf kernel
__global__ void kernel_fast_expf(const float* __restrict__ in,
                                 float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = in[i];
    #pragma unroll 8
    for (int iter = 0; iter < ITERS; ++iter) {
        acc = __expf(acc) * SCALE + BIAS;
    }
    out[i] = acc;
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o expf_probe expf_probe.cu
```

## Measurement

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1M threads x 128 iters | fp32 | 0.0404 | 0.0404 | 0.0406 | expf (standard math) | 0.0516 | 1.27 | unknown | `ssh h200_ncu "cd /inspire/hdd/project/qianghuaxuexi/public/wty/ai4ai/ai-infra/kernel-kb-mvp && /usr/local/cuda-12.9/bin/nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o /tmp/expf_probe sources/experience/hw-probes/fast-math/artifacts/expf_probe.cu && /tmp/expf_probe"` |

Precision (single-call, 1M inputs in [-10, 10]):
- `expf`:   max ULP error vs f64 reference = 2
- `__expf`: max ULP error vs f64 reference = 8

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not yet wired for math intrinsics). The `evidence_level` is `measured` based on the CUDA event timing and host-side ULP verification.

## Notes

1. The SFU (Special Function Unit) processes approximate transcendental functions at 16 ops/clock/SM (best-practices guide Table 5, L1243-L1244). Standard `expf` is implemented as a multi-instruction software sequence calling the SFU internally but with range-reduction and polynomial correction steps that add overhead. `__expf` maps more directly to the SFU `ex2.approx.f32` instruction with minimal overhead.

2. The documented max ULP error for `__expf(x)` is `2 + floor(abs(1.173*x))` (programming guide Table 58, L26993-L26995). For inputs in [-10, 10], the theoretical worst case is `2 + floor(11.73) = 13 ULP`. Our observed maximum of 8 ULP is consistent with this bound.

3. In memory-bound kernels (e.g., single expf per element with large arrays), both `expf` and `__expf` are bottlenecked by global memory bandwidth and show no throughput difference. The speedup materializes only in compute-bound regions.

4. The probe chain uses `acc = expf(acc) * SCALE + BIAS` to keep values in [-4, 4] across iterations. The `fma` cost is identical in both variants and cancels in the ratio.
