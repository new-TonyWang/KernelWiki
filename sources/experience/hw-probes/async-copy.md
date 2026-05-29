---
api: memcpy_async
namespace: libcupp
probe_slug: async-copy-2stage-bw
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
  code: artifacts/experience/hw-probes/async-copy/artifacts/async_copy_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o async_copy_probe async_copy_probe.cu
  introspection: ''
  profile: ''
source:
- path: <path-removed>
  anchor: part-1-maximizing-memory-bandwidth
  excerpt: LDGSTS landed in Ampere; TMA in Hopper. LDGSTS for 4/8/16-byte aligned
    loads; TMA 1D for 16-byte aligned bulk copies.
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.9775
  latency_ms_p10: 0.975
  latency_ms_p90: 0.9797
  baseline_name: vanilla-elementwise
  baseline_ms: 0.8815
  ratio: 0.902
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- 'The 2-stage LDGSTS kernel is ~10% slower than vanilla for trivial a*b compute on
  H200. This aligns with GTC25-S72683 guidance: the big wins are on iterative compute-heavy
  kernels, not simple elementwise ops. The 4-byte L1 ACCESS mode adds overhead without
  L1 BYPASS benefit.'
- A future probe with heavier compute (sqrt chains) or L1 BYPASS mode (16-byte aligned
  copies) should show the expected uplift.
id: exp-async-copy
type: experience
vendor: nvidia
title: 2026 04 16 Async Copy
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11259-L11270
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L940-L942
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L3937-L4030
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L11259-L11484
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L940-L999
---
## Summary

This probe compares a **vanilla elementwise kernel** (`c[i] = a[i] * b[i]`, grid-stride loop) against a **2-stage LDGSTS prefetch version** using `__pipeline_memcpy_async` primitives on H200 (sm_90a, CUDA 12.9). The input size is N = 256M floats (1 GiB per array). Both kernels use 256 threads/block with 1056 blocks (8 blocks/SM at 100% occupancy).

The **vanilla kernel achieves 3654 GB/s** (75% of H200 theoretical peak of ~4900 GB/s). The **2-stage async kernel achieves 3295 GB/s**, approximately **10% slower** than vanilla. This result is consistent with GTC25-S72683 findings: for trivially simple compute (`a*b`), the overhead of shared memory staging (commit/wait/release instructions + SMEM traffic) exceeds the latency-hiding benefit. The GTC talk showed that async copies deliver significant uplift only when (a) the compute intensity is higher (e.g., `sqrt` chains yielded 1.3x on H100), or (b) the kernel is iterative with genuine prefetch opportunities.

## Minimal Kernel

```cuda
// Vanilla
__global__ void kernel_vanilla(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += gridDim.x * blockDim.x) {
        c[i] = a[i] * b[i];
    }
}

// 2-stage LDGSTS prefetch (primitives API)
__global__ void kernel_async_2stage(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c, int N) {
    __shared__ float a_buf[2][256];
    __shared__ float b_buf[2][256];
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int base = blockIdx.x * blockDim.x;

    // Prologue: submit 2 stages
    for (int s = 0; s < 2; ++s) {
        int idx = base + s * stride + tid;
        if (idx < N) {
            __pipeline_memcpy_async(&a_buf[s][tid], &a[idx], sizeof(float));
            __pipeline_memcpy_async(&b_buf[s][tid], &b[idx], sizeof(float));
        }
        __pipeline_commit();
    }
    // Main loop: wait, compute, prefetch next
    for (int iter = 0; ; ++iter) {
        int stage = iter % 2;
        long long cb = (long long)base + (long long)iter * stride;
        if (cb >= N) break;
        __pipeline_wait_prior(1);
        long long ci = cb + tid;
        if (ci < N) c[ci] = a_buf[stage][tid] * b_buf[stage][tid];
        long long pb = (long long)base + (long long)(iter + 2) * stride;
        if (pb < N) {
            long long pi = pb + tid;
            if (pi < N) {
                __pipeline_memcpy_async(&a_buf[stage][tid], &a[pi], sizeof(float));
                __pipeline_memcpy_async(&b_buf[stage][tid], &b[pi], sizeof(float));
            }
        }
        __pipeline_commit();
    }
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o async_copy_probe async_copy_probe.cu
```

## Measurement

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N=256M | float32 | 0.8815 | 0.8794 | 0.8846 | vanilla-elementwise | 0.8815 | 1.000 | unknown | `./async_copy_probe` (vanilla) |
| N=256M | float32 | 0.9775 | 0.9750 | 0.9797 | vanilla-elementwise | 0.8815 | 0.902 | unknown | `./async_copy_probe` (async-2stage) |

## Introspection

Not captured in this probe (kp_introspect not available). The SM count (132) was read via `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`.

## Notes

1. **Why async is slower here**: the vanilla kernel for `c[i] = a[i] * b[i]` has only 2 loads per thread (8 bytes in-flight). At 100% occupancy with 256 threads/block and 8 blocks/SM, total bytes-in-flight is 16 KiB/SM. While this is below the ~64 KiB needed to saturate H200 fully, the kernel still achieves 75% BW utilization. The 2-stage LDGSTS version doubles bytes-in-flight to 32 KiB/SM but adds per-iteration overhead: `__pipeline_commit` + `__pipeline_wait_prior` + shared-memory reads replace direct register reads. For trivial `a*b` compute, this overhead exceeds the latency-hiding gain.

2. **GTC25-S72683 guidance**: the talk explicitly states that async copies show "minor improvement just by switching to async copies -- unless they can batch loads. The big wins are on iterative kernels that can prefetch data for future iterations, especially low-occupancy compute-heavy kernels." Our result confirms this on H200.

3. **L1 ACCESS vs L1 BYPASS**: this probe uses 4-byte copies (L1 ACCESS mode). The L1 BYPASS mode requires 16-byte aligned transfers (`aligned_size_t<16>`), which would reduce L1 cache pollution and MIO pressure. A future probe with 16-byte copies should be more favorable.

4. **Expected uplift scenarios**: the GTC talk showed 1.305x speedup on H100 when `compute(a,b) = sqrt(sqrt(a)/sqrt(b))` (heavier compute intensity). A similar probe with heavier compute on H200 should show measurable uplift.
