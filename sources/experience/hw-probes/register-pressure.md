---
api: --maxrregcount
namespace: ptx
probe_slug: register-pressure-spill-tradeoff
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
  code: sources/experience/hw-probes/register-pressure/artifacts/reg_pressure_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v -o probe_default reg_pressure_probe.cu
    && nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v --maxrregcount=32 -o probe_maxreg32
    reg_pressure_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 1.192093e-07
  latency_ms_median: 0.1738
  latency_ms_p10: 0.1736
  latency_ms_p90: 0.1747
  baseline_name: probe_maxreg32
  baseline_ms: 0.8414
  ratio: 4.84
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- 'ptxas emitted a warning for maxrregcount=32: ''Too big maxrregcount value specified
  32, will be ignored'' followed by ''Overriding maximum register limit 256 ... with
  32 of maxrregcount option''. The warning text is misleading but the cap was applied
  correctly.'
id: exp-register-pressure
type: experience
vendor: nvidia
title: 2026 04 15 Register Pressure
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1065
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1215
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1219-L1222
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1064-L1099
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1215-L1223
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22890-L22905
---
## Summary

This probe compares default register allocation versus forced register capping with `--maxrregcount=32` on a register-heavy kernel (48 live float accumulators, 3 rounds of cross-dependent FMA) on H200 (sm_90a, CUDA 12.9).

Key finding: the default build used **64 registers/thread** with **zero spills** and achieved **0.1738 ms** median latency. The `--maxrregcount=32` build forced registers down to **32/thread** but caused **660 bytes of spill stores** and **784 bytes of spill loads** per thread, resulting in **0.8414 ms** median latency -- a **4.84x slowdown** despite achieving **2x the occupancy** (8 vs 4 blocks/SM). This directly confirms the programming guide warning that reducing registers "may result in more register spilling" and that higher occupancy does not guarantee better performance.

## Minimal Kernel

See full source: `sources/experience/hw-probes/register-pressure/artifacts/reg_pressure_probe.cu`

The kernel uses 48 independent float accumulators loaded from global memory, then performs 3 rounds of cross-dependent FMA (ring, butterfly stride-24, reverse ring). All 48 values must be live simultaneously, creating high register pressure. The same `.cu` file is compiled twice with different flags to produce the two binaries.

```cuda
__device__ __forceinline__
float heavy_compute_48(const float* __restrict__ in, int idx, int n) {
    float a00 = in[(idx +  0) % n];
    // ... 48 loads total ...
    float a47 = in[(idx + 47) % n];

    // Round 1: ring FMA (48 FMAs, each depends on the previous)
    a00 = a00 * a01 + a47;  a01 = a01 * a02 + a00;
    // ...

    // Round 2: butterfly FMA (stride 24)
    a00 = a00 * a24 + a12;  a01 = a01 * a25 + a13;
    // ...

    // Round 3: reverse ring
    a00 = a00 * a47 + a23;  a01 = a01 * a00 + a24;
    // ...

    return a00 + a01 + ... + a47;
}

__global__ void reg_pressure_kernel(const float* __restrict__ in,
                                    float*       __restrict__ out,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute_48(in, idx, n);
}
```

## Build

Two compilations from the same source:

```bash
# (a) Default register allocation
nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v -o probe_default reg_pressure_probe.cu

# (b) Forced register cap at 32
nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v --maxrregcount=32 -o probe_maxreg32 reg_pressure_probe.cu
```

### ptxas -v output (default)

```
ptxas info    : Compiling entry function '_Z19reg_pressure_kernelPKfPfi' for 'sm_90a'
ptxas info    : Function properties for _Z19reg_pressure_kernelPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, used 0 barriers
```

### ptxas -v output (--maxrregcount=32)

```
ptxas warning : Too big maxrregcount value specified 32, will be ignored
ptxas info    : Overriding maximum register limit 256 for '_Z19reg_pressure_kernelPKfPfi' with  32 of maxrregcount option
ptxas info    : Compiling entry function '_Z19reg_pressure_kernelPKfPfi' for 'sm_90a'
ptxas info    : Function properties for _Z19reg_pressure_kernelPKfPfi
    424 bytes stack frame, 660 bytes spill stores, 784 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers, 424 bytes cumulative stack size
```

## Measurement

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| N=4194304 | fp32 | 0.1738 | 0.1736 | 0.1747 | probe_maxreg32 | 0.8414 | 4.84 | unknown | `./probe_default` |

Detailed comparison:

| variant | regs/thread | spill_stores (bytes) | spill_loads (bytes) | localSize (bytes) | blocks/SM (occupancy) | median_ms | p10_ms | p90_ms |
|---|---|---|---|---|---|---|---|---|
| probe_default | 64 | 0 | 0 | 0 | 4 (50%) | 0.1738 | 0.1736 | 0.1747 |
| probe_maxreg32 | 32 | 660 | 784 | 424 | 8 (100%) | 0.8414 | 0.8398 | 0.8430 |

## Introspection

Runtime attributes queried via `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:

- **Default**: numRegs=64, localSizeBytes=0, maxTPB=1024, blocks/SM=4
- **maxreg32**: numRegs=32, localSizeBytes=424, maxTPB=1024, blocks/SM=8

No `kp_introspect` bundle was produced (tool not yet wired for this probe type).

## Notes

1. **4.84x slowdown from spilling despite 2x occupancy.** The maxreg32 variant achieves 100% theoretical occupancy (8 blocks of 256 threads = 2048 threads = 64 warps) vs 50% for the default (4 blocks = 1024 threads = 32 warps). Yet the spill traffic to local memory (which resides in the global memory space, accessing L1/L2/HBM) completely overwhelms the occupancy benefit. Each thread generates 660 bytes of spill stores + 784 bytes of spill loads = 1444 bytes of spill traffic per thread.

2. **Default compiler chose 64 registers -- the maximum for 4 blocks/SM.** With 65536 regs/SM, block size 256: 65536 / (256 * 4) = 64 regs/thread. The compiler used exactly this budget, indicating the kernel genuinely needs all 64 registers to hold the 48 accumulators plus temporaries without any spilling.

3. **Occupancy analysis:**
   - Default (64 regs): 65536 / 64 = 1024 threads = 32 warps = 4 blocks of 256. Occupancy = 32/64 = 50%.
   - maxreg32 (32 regs): 65536 / 32 = 2048 threads = 64 warps = 8 blocks of 256. Occupancy = 64/64 = 100%.
   - Despite doubling occupancy, the local-memory spill latency (global memory speed) makes every FMA's operand reload orders of magnitude slower than register access.

4. **ptxas warning about maxrregcount is misleading.** The warning "Too big maxrregcount value specified 32, will be ignored" is followed by "Overriding maximum register limit 256 ... with 32". The cap was applied correctly despite the confusing warning. This is a known quirk of the ptxas warning system.

5. **Correctness verified.** Both variants produce identical results (max_abs_err = 1.192e-07 vs CPU reference), confirming that register spilling does not affect numerical correctness -- only performance.

6. **This probe complements the compiler-hints probe.** The compiler-hints probe showed that `__launch_bounds__` can cause the compiler to use *more* registers (up to the occupancy-derived ceiling). This probe shows the opposite extreme: forcibly reducing registers below what the kernel needs causes catastrophic spill overhead.
