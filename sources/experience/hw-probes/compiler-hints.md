---
api: __launch_bounds__
namespace: runtime
probe_slug: launch-bounds-register-effect
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
  code: sources/experience/hw-probes/compiler-hints/artifacts/launch_bounds_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o launch_bounds_probe
    launch_bounds_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 1.49e-07
  latency_ms_median: 0.1203
  latency_ms_p10: 0.12
  latency_ms_p90: 0.1205
  baseline_name: heavy_no_lb
  baseline_ms: 0.1201
  ratio: 0.9989
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- The kernel is memory-bound at N=4M, which masks register-allocation differences.
  A compute-bound workload would show larger latency effects.
- The compiler allocated MORE registers with __launch_bounds__(256,4) (56 vs 48),
  consistent with the documented behavior of using more registers when an occupancy
  floor is specified.
id: exp-compiler-hints
type: experience
vendor: nvidia
title: 2026 04 15 Compiler Hints
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22819
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22837
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22816-L22905
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1099
---
## Summary

This probe measures the effect of `__launch_bounds__(256, 4)` on register allocation and latency for a register-hungry kernel on H200 (sm_90a, CUDA 12.9). The kernel uses 32 independent float accumulators with 3 rounds of cross-dependent FMA operations, designed to create high register pressure.

Key finding: `__launch_bounds__(256, 4)` caused the compiler to allocate **more** registers (56 vs 48 without launch bounds), with no spilling in either case. This confirms the documented behavior: when both `maxThreadsPerBlock` and `minBlocksPerMultiprocessor` are specified, the compiler may increase register usage up to the occupancy-derived limit L (= 65536 / (256 * 4) = 64 regs/thread) to reduce instruction count. The latency difference was negligible (~0.1% on this memory-bound workload).

## Minimal Kernel

```cuda
// Register-hungry computation: 32 accumulators, 3 rounds of cross-mixing.
__device__ __host__ __forceinline__
float heavy_compute(const float* __restrict__ in, int idx, int n) {
    float a00 = in[(idx +  0) % n];
    float a01 = in[(idx +  1) % n];
    // ... (32 loads total)
    float a31 = in[(idx + 31) % n];

    // Round 1: ring FMA
    a00 = a00 * a01 + a31;  a01 = a01 * a02 + a00;
    // ... (32 FMAs)

    // Round 2: butterfly FMA (stride 16)
    a00 = a00 * a16 + a08;  a01 = a01 * a17 + a09;
    // ... (32 FMAs)

    // Round 3: reverse ring
    a00 = a00 * a31 + a15;  a01 = a01 * a00 + a16;
    // ... (32 FMAs)

    return a00 + a01 + ... + a31;
}

// WITHOUT launch_bounds (compiler chose 48 regs, maxTPB=1024)
__global__ void heavy_no_lb(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute(in, idx, n);
}

// WITH launch_bounds (compiler chose 56 regs, maxTPB=256)
__global__ void __launch_bounds__(256, 4)
heavy_lb_256_4(const float* __restrict__ in,
               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute(in, idx, n);
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o launch_bounds_probe launch_bounds_probe.cu
```

ptxas output:
```
_Z14heavy_lb_256_4PKfPfi: Used 56 registers, 0 bytes spill stores, 0 bytes spill loads
_Z14heavy_lb_256_2PKfPfi: Used 56 registers, 0 bytes spill stores, 0 bytes spill loads
_Z11heavy_no_lbPKfPfi:    Used 48 registers, 0 bytes spill stores, 0 bytes spill loads
```

## Measurement

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| N=4194304 | fp32 | 0.1203 | 0.1200 | 0.1205 | heavy_no_lb | 0.1201 | 0.9989 | unknown | `./launch_bounds_probe` |

Additional kernel variants measured:

| kernel | regs/thread | maxTPB | smem_static | median_ms | p10_ms | p90_ms |
|---|---|---|---|---|---|---|
| heavy_no_lb | 48 | 1024 | 0 | 0.1201 | 0.1199 | 0.1220 |
| heavy_lb_256_4 | 56 | 256 | 0 | 0.1203 | 0.1200 | 0.1205 |
| heavy_lb_256_2 | 56 | 256 | 0 | 0.1202 | 0.1201 | 0.1207 |

## Introspection

Register allocation queried via `cudaFuncGetAttributes` at runtime. No `kp_introspect` bundle was produced (tool not yet wired for this probe type).

## Notes

1. **Compiler used more registers with launch_bounds, not fewer.** This is the documented behavior when `minBlocksPerMultiprocessor` is specified: the compiler knows the occupancy floor and uses more registers (up to the limit L) to reduce instruction count and better hide latency. Without launch_bounds, the compiler's heuristic conservatively used fewer registers (48) to leave room for more concurrent blocks.

2. **No spilling in any variant.** The occupancy-derived register cap (64 = 65536 / (256 * 4)) was not exceeded, so no register spilling occurred. This is the ideal outcome of a well-tuned `__launch_bounds__`.

3. **Latency difference is negligible** because this kernel is memory-bound at N=4M (dominated by 32 global loads per thread). A compute-bound kernel would show larger effects from the instruction count reduction enabled by the extra 8 registers.

4. **Occupancy analysis on H200 (sm_90a):**
   - Max warps/SM = 64, max threads/SM = 2048
   - heavy_no_lb at 48 regs, block=256: 65536 / (48 * 8) = 170 warps possible by regs alone (capped at 64), so register-unlimited. Occupancy depends only on block count: floor(2048 / 256) = 8 blocks = 64 warps = 100%.
   - heavy_lb_256_4 at 56 regs, block=256: 65536 / (56 * 8) = 146 warps (capped at 64). Still 100% occupancy with 8 blocks.
   - Both variants achieve 100% theoretical occupancy at block=256, explaining the identical latency.

5. **maxTPB was capped from 1024 to 256** by launch_bounds. Launching with more than 256 threads per block would fail at runtime for the lb variants.
