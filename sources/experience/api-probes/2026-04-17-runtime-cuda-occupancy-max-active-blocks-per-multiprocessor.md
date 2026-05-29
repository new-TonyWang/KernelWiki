---
api: cudaOccupancyMaxActiveBlocksPerMultiprocessor
namespace: runtime
probe_slug: runtime-cuda-occupancy-max-active-blocks-per-multiprocessor
status: verified
kind: documented
trigger: init-sweep
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f4646b8b9c4f5a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe
    cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu
  introspection: ''
  profile: ''
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0
  latency_ms_median: 7.6e-05
  latency_ms_p10: 6.43e-05
  latency_ms_p90: 7.72e-05
  baseline_name: host-chrono bs=1024 (fewest internal branches)
  baseline_ms: 6.85e-05
  ratio: 0.9
back_filled_into:
- wiki/nvidia/api-definitions/runtime/cudaOccupancyMaxActiveBlocksPerMultiprocessor.md
open_questions:
- clock_policy is unknown — GPU clocks were not locked during measurement; host-side
  API latency is CPU-bound so clock policy has limited effect, but the kernel launch/memcpy
  in the correctness step is not clock-locked either.
- The API-predicted numBlocks is cross-checked against the static hardware model,
  but the 'actual' blocks resident per SM (e.g. via ncu ActiveCTAs.avg.peak) is not
  measured in this probe — it would require profiling the kernel at steady state.
- Only one kernel (56 regs/thread) was probed; shmem-bound or max-blocks-bound cases
  (e.g. small kernels hitting the 32 blocks/SM limit) are not exercised here.
id: exp-2026-04-17-runtime-cuda-occupancy-max-active-blocks-per-multiprocessor
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Cuda Occupancy Max Active Blocks Per Multiprocessor
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3865-L3865
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3881-L3883
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Architecture Guides/hopper-tuning-guide/cuda_hopper-tuning-guide_index.html.md
  anchor: L35-L42
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3862-L3920
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3824-L3858
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Architecture Guides/hopper-tuning-guide/cuda_hopper-tuning-guide_index.html.md
  anchor: L30-L42
---
## Summary

End-to-end probe of `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on H200 (sm_90a, CUDA 12.9, driver 570.124.06). The probe defines a 16-accumulator FMA-chain kernel that compiles to **56 regs/thread** (confirmed by `-Xptxas=-v`) and queries the API for block sizes 64/128/256/512/1024 with `dynamicSMemSize=0`. All five calls return `cudaSuccess`. The returned numBlocks/SM (18, 9, 4, 2, 1) match the hardware register model — `numBlocks * blockSize * regs/thread <= regsPerSM = 65536` holds in every case. Host-side call latency (median) is **~0.076 µs per call**, measured over 20 batches of 1000 calls. A follow-up launch at bs=256 with grid = numBlocks × 132 SMs = 528 runs to completion and produces correct output.

## Minimal Kernel

```cuda
// Probe: cudaOccupancyMaxActiveBlocksPerMultiprocessor (end-to-end)
//
// Signature:
//   __host__ __device__ cudaError_t
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//       int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
//
// The kernel is a 16-accumulator FMA chain that ptxas compiles to 56
// regs/thread. This makes the occupancy register-limited on H200
// (65536 regs/SM / 56 regs/thread = 1170 threads/SM, so numBlocks/SM
// rounds down with block-size-aware warp allocation).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void __launch_bounds__(1024, 1)
fma_chain(const float* __restrict__ A, const float* __restrict__ B,
          float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    float a8=0,a9=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    for (int i = idx; i + 15 < n; i += stride * 16) {
        a0  += A[i]    * B[i];      a1  += A[i+1]  * B[i+1];
        a2  += A[i+2]  * B[i+2];    a3  += A[i+3]  * B[i+3];
        a4  += A[i+4]  * B[i+4];    a5  += A[i+5]  * B[i+5];
        a6  += A[i+6]  * B[i+6];    a7  += A[i+7]  * B[i+7];
        a8  += A[i+8]  * B[i+8];    a9  += A[i+9]  * B[i+9];
        a10 += A[i+10] * B[i+10];   a11 += A[i+11] * B[i+11];
        a12 += A[i+12] * B[i+12];   a13 += A[i+13] * B[i+13];
        a14 += A[i+14] * B[i+14];   a15 += A[i+15] * B[i+15];
    }
    float sum = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
    if (idx < n) C[idx] = sum;
}

// Host core: for each block size, warm 5 calls, then measure 20 batches of
// 1000 calls each via std::chrono. Report median/p10/p90 in microseconds.
// Also cross-checks cudaOccupancyAvailableDynamicSMemPerBlock at bs=256.
// Finally launches grid = numBlocks × 132 SMs and verifies output.
// (Full source under artifacts/; this snippet preserves the core probe.)
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v \
     -o cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe \
     artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu
```

ptxas output: `fma_chain`: 56 regs, 0 bytes spill stores, 0 bytes spill loads.

## Measurement

Host-side API call latency. Each sample = 1000 back-to-back API calls; 20 samples collected with `std::chrono::high_resolution_clock`; per-call timing reported as `batch_us / 1000`. Kernel-launch/correctness step uses the API-returned numBlocks at bs=256 (4 blocks/SM) to pick grid = 4 × 132 = 528, then verifies the kernel runs to completion and produces positive output for the active threads.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| bs=64, dynSmem=0 | api-call | 0.0000684 | 0.0000642 | 0.0000884 | host-chrono bs=1024 | 0.0000685 | 1.00 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu && /tmp/p` |
| bs=128, dynSmem=0 | api-call | 0.0000758 | 0.0000750 | 0.0000758 | host-chrono bs=1024 | 0.0000685 | 0.90 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu && /tmp/p` |
| bs=256, dynSmem=0 | api-call | 0.0000760 | 0.0000643 | 0.0000772 | host-chrono bs=1024 | 0.0000685 | 0.90 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu && /tmp/p` |
| bs=512, dynSmem=0 | api-call | 0.0000684 | 0.0000684 | 0.0000687 | host-chrono bs=1024 | 0.0000685 | 1.00 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu && /tmp/p` |
| bs=1024, dynSmem=0 | api-call | 0.0000685 | 0.0000685 | 0.0000688 | host-chrono bs=1024 | 0.0000685 | 1.00 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p artifacts/experience/api-probes/artifacts/cudaOccupancyMaxActiveBlocksPerMultiprocessor_probe.cu && /tmp/p` |

Units: `latency_ms_*` is per-call host latency in **milliseconds** (0.0000760 ms = 0.076 µs). Single-call resolution is below std::chrono's useful precision; the table reports the batch-normalized mean per sample.

### API result table

| blockSize | dynSmem | numBlocks/SM | activeWarps | occupancy | limitingFactor |
|---|---|---|---|---|---|
| 64   | 0 | 18 | 36 | 56.2% | registers |
| 128  | 0 |  9 | 36 | 56.2% | registers |
| 256  | 0 |  4 | 32 | 50.0% | registers |
| 512  | 0 |  2 | 32 | 50.0% | registers |
| 1024 | 0 |  1 | 32 | 50.0% | registers |

### Hardware-limit cross-check (H200: 65536 regs/SM, 56 regs/thread)

| blockSize | numBlocks | used_regs (numBlocks×bs×56) | cap=65536 |
|---|---|---|---|
| 64   | 18 | 64 512 | OK |
| 128  |  9 | 64 512 | OK |
| 256  |  4 | 57 344 | OK |
| 512  |  2 | 57 344 | OK |
| 1024 |  1 | 57 344 | OK |

### Companion-API cross-check

`cudaOccupancyAvailableDynamicSMemPerBlock(fma_chain, target, bs=256)`:

| target blocks/SM | returned dynSmem cap | cudaError |
|---|---|---|
| 1 | 49 152 bytes | cudaSuccess |
| 2 | 49 152 bytes | cudaSuccess |
| 3 | 49 152 bytes | cudaSuccess |
| 4 | 49 152 bytes | cudaSuccess |
| 5..8 | — | cudaErrorInvalidValue (target exceeds max feasible) |

The cap of 49 152 bytes equals `cudaDevAttrMaxSharedMemoryPerBlock` (the default 48 KB); opt-in via `cudaFuncSetAttribute` would be required to go beyond this — see the sister probe for `cudaFuncSetAttribute`.

### Launch correctness

With bs=256, numBlocks=4, grid=528 → 135 168 threads. The kernel launch returns `cudaSuccess`, the post-launch synchronize returns `cudaSuccess`, and 135 168 output slots contain positive sums (`1.0 * 2.0 * loop_count` summed 16 ways, always > 0). The API-predicted occupancy configuration is directly launchable with no further tuning.

## Introspection

No `kp_introspect kernel-static` bundle was generated in this probe (cuda-python not available in the H200 shell environment). Device static facts used by this probe are sourced from `cudaGetDeviceProperties` at runtime: 132 SMs, 65536 regs/SM, 2048 max threads/SM, 64 max warps/SM, 32 max blocks/SM. These match the occupancy-sweep skill record (`artifacts/experience/hw-probes/occupancy-sweep/h200_device_static.json`).

## Notes

- **Signature and semantics**. Per the CUDA Runtime API reference (L3865): `__host__ __device__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize)`. L3881: "Returns in `*numBlocks` the maximum number of active blocks per streaming multiprocessor for the device function." The API does NOT launch the kernel — it only queries the occupancy model.

- **Parameter rules**. `func` is a device-kernel pointer (passed as `(const void*)kernel_name`). `blockSize` is the intended `blockDim.x*blockDim.y*blockDim.z`. `dynamicSMemSize` is the dynamic shared-memory footprint the caller plans to pass at `<<<grid, block, dynSmem>>>`. The API accounts for static shmem declared inside the kernel automatically via `cudaFuncAttributes`.

- **Return values observed**. `cudaSuccess` for all five block sizes. `cudaOccupancyAvailableDynamicSMemPerBlock` with an infeasible target returns `cudaErrorInvalidValue` (which sets a non-sticky error the caller must clear with `cudaGetLastError` before the next launch; the probe does this explicitly).

- **Cross-check with hardware model**. With 56 regs/thread, the per-SM register ceiling is `65536 / 56 = 1170` threads. The register-allocation granularity (rounded to 256 per warp) makes actual allocation 1792 regs/warp = 56 × 32 rounded to multiple of 256 = 1792. So effective threads/SM ≤ 32 warps/SM × 32 lanes = 1024 — matching the API's prediction for bs=1024 (1 block) and bs=512 (2 blocks). For bs=64 the API returns 18 blocks × 64 = 1152 threads (36 warps), consistent with the register-limited formula `floor(65536 / (56*32)) warps = 36` rounded to block-size-aware count.

- **Host call latency**. Measured 0.06–0.09 µs per call on H200 + CUDA 12.9. This is well below a typical kernel launch (~5–10 µs) and small enough to call once per operator at init time. Do NOT call it inside a hot path — the skill `wiki/nvidia/foundations/compute/occupancy-tuning/skill.md` already warns against this.

- **Reference probe exists**. The earlier skill-build probe at `artifacts/experience/hw-probes/occupancy-sweep/2026-04-16-occupancy-tuning.md` uses this exact API across block sizes on a kernel with identical register pressure and reports the same numBlocks table — this probe therefore also serves as a regression check of that prior data.

- **CUDA Samples reference**. `simpleOccupancy.cu` (L77-L122) demonstrates idiomatic use: query the API, derive active warps, and print theoretical occupancy alongside the achieved measured occupancy via `cudaEventElapsedTime` and flops/time math. That sample is the direct template for the `Launch + correctness` block of this probe.

- **Not measured / out of scope**:
  - Actual resident CTAs at steady state (would need `ncu --metrics sm__ctas_active_peak`). MVP does not ship ncu profiling yet.
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` (the `DisableCachingOverride` flag variant) — this variant is not in the F10 API batch.
  - Cluster occupancy (Hopper CGA) — out of MVP scope.
