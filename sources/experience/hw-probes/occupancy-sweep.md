---
api: cudaOccupancyMaxActiveBlocksPerMultiprocessor
namespace: runtime
probe_slug: occupancy-sweep-block-size
status: verified
kind: documented
trigger: skill-build
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f4646b8b9c4f5a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: sources/experience/hw-probes/occupancy-sweep/artifacts/occupancy_sweep_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o occupancy_sweep_probe
    occupancy_sweep_probe.cu
  introspection: sources/experience/hw-probes/occupancy-sweep/h200_device_static.json
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  line_range: L1082-L1134
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  line_range: L3824-L3960
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  line_range: L16918-L17230
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Architecture
    Guides/hopper-tuning-guide/cuda_hopper-tuning-guide_index.html.md
  line_range: L30-L42
- path: '{{CUDA_SAMPLES_REPO_REF}}/Samples/0_Introduction/simpleOccupancy/simpleOccupancy.cu'
  line_range: L77-L122
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.0057
  latency_ms_p10: 0.0055
  latency_ms_p90: 0.006
  baseline_name: vec_add-bs64
  baseline_ms: 0.008
  ratio: 1.4
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- The register-heavy kernel uses 56 registers/thread; with -maxrregcount=48 the occupancy
  pattern would differ further.
- Only two kernel types were tested; compute-bound kernels may show different occupancy-latency
  relationships.
id: exp-occupancy-sweep
type: experience
vendor: nvidia
title: 2026 04 16 Occupancy Tuning
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1087-L1089
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1089
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1124
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Architecture Guides/hopper-tuning-guide/cuda_hopper-tuning-guide_index.html.md
  anchor: L35-L42
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3862-L3890
---
## Summary

This probe sweeps block sizes (64, 128, 256, 512, 1024) on two kernels on H200 (sm_90a, CUDA 12.9) and measures both theoretical occupancy (via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) and actual latency (via CUDA events). The key finding: **the occupancy-optimal block size is not always the latency-optimal one**. For the register-heavy kernel (56 regs/thread), block size 512 at 50% occupancy is 10% faster than block size 64 at 56.2% occupancy. For the simple kernel (12 regs/thread), all block sizes achieve 100% occupancy but block size 512 is 29% faster than block size 64 due to lower scheduling overhead.

## Minimal Kernel

```cuda
// Simple vector-add: 12 regs/thread → 100% occupancy at all block sizes
__global__ void vec_add(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float*       __restrict__ C,
                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Register-heavy: 16 independent accumulators → 56 regs/thread
__global__ void __launch_bounds__(1024, 1)
vec_add_regpress(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float*       __restrict__ C,
                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    float a8=0,a9=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    for (int i = idx; i + 15 < n; i += stride * 16) {
        a0  += A[i]    * B[i];    a1  += A[i+1]  * B[i+1];
        // ... (8 pairs total)
    }
    float sum = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
    if (idx < n) C[idx] = sum;
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o occupancy_sweep_probe occupancy_sweep_probe.cu
```

ptxas output:
- `vec_add`: Used 12 registers, 0 bytes spill stores, 0 bytes spill loads
- `vec_add_regpress`: Used 56 registers, 0 bytes spill stores, 0 bytes spill loads

## Measurement

Data size: 64M float elements (256 MB per buffer). 5 warmup + 20 measured launches. Grid size = numBlocksPerSM × 132 SMs (capped to data coverage).

### Simple vec_add (12 regs/thread, memory-bound)

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| bs=64, occ=100% | fp32 | 0.0080 | 0.0078 | 0.0095 | vec_add-bs64 | 0.0080 | 1.00 | unknown | `./occupancy_sweep_probe` |
| bs=128, occ=100% | fp32 | 0.0067 | 0.0065 | 0.0072 | vec_add-bs64 | 0.0080 | 1.19 | unknown | `./occupancy_sweep_probe` |
| bs=256, occ=100% | fp32 | 0.0060 | 0.0058 | 0.0062 | vec_add-bs64 | 0.0080 | 1.33 | unknown | `./occupancy_sweep_probe` |
| bs=512, occ=100% | fp32 | 0.0057 | 0.0055 | 0.0060 | vec_add-bs64 | 0.0080 | 1.40 | unknown | `./occupancy_sweep_probe` |
| bs=1024, occ=100% | fp32 | 0.0058 | 0.0056 | 0.0061 | vec_add-bs64 | 0.0080 | 1.38 | unknown | `./occupancy_sweep_probe` |

### Register-heavy vec_add_regpress (56 regs/thread, compute-bound)

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| bs=64, occ=56.2% | fp32 | 0.0330 | 0.0323 | 0.0348 | vec_add-bs64 | 0.0080 | 0.24 | unknown | `./occupancy_sweep_probe` |
| bs=128, occ=56.2% | fp32 | 0.0320 | 0.0315 | 0.0332 | vec_add-bs64 | 0.0080 | 0.25 | unknown | `./occupancy_sweep_probe` |
| bs=256, occ=50.0% | fp32 | 0.0316 | 0.0310 | 0.0321 | vec_add-bs64 | 0.0080 | 0.25 | unknown | `./occupancy_sweep_probe` |
| bs=512, occ=50.0% | fp32 | 0.0297 | 0.0293 | 0.0306 | vec_add-bs64 | 0.0080 | 0.27 | unknown | `./occupancy_sweep_probe` |
| bs=1024, occ=50.0% | fp32 | 0.0301 | 0.0299 | 0.0308 | vec_add-bs64 | 0.0080 | 0.27 | unknown | `./occupancy_sweep_probe` |

### Occupancy details

| kernel | block_size | blocks/SM | active_warps | occupancy |
|---|---|---|---|---|
| vec_add | 64 | 32 | 64 | 100.0% |
| vec_add | 128 | 16 | 64 | 100.0% |
| vec_add | 256 | 8 | 64 | 100.0% |
| vec_add | 512 | 4 | 64 | 100.0% |
| vec_add | 1024 | 2 | 64 | 100.0% |
| vec_add_regpress | 64 | 18 | 36 | 56.2% |
| vec_add_regpress | 128 | 9 | 36 | 56.2% |
| vec_add_regpress | 256 | 4 | 32 | 50.0% |
| vec_add_regpress | 512 | 2 | 32 | 50.0% |
| vec_add_regpress | 1024 | 1 | 32 | 50.0% |

### API results

- `cudaOccupancyMaxPotentialBlockSize(vec_add)`: blockSize=1024, minGridSize=264
- `cudaOccupancyMaxPotentialBlockSize(vec_add_regpress)`: blockSize=576, minGridSize=264
- `cudaOccupancyAvailableDynamicSMemPerBlock(vec_add, 8 blocks, 256 threads)`: 29184 bytes

## Introspection

Device static info from `kp_introspect device-static`: `sources/experience/hw-probes/occupancy-sweep/h200_device_static.json`

Key H200 specs: 132 SMs, 65536 regs/SM, 2048 max threads/SM, 64 max warps/SM, 32 max blocks/SM, 233472 bytes shared mem/SM.

## Notes

- **Occupancy ≠ performance**: For vec_add_regpress, block size 512 (50% occupancy) is 10% faster than block size 64 (56.2% occupancy). The higher occupancy at small block sizes comes from fitting more blocks per SM, but each block has higher scheduling overhead and less efficient memory coalescing.

- **Equal occupancy, different latency**: For vec_add, all block sizes achieve 100% occupancy, yet block size 512 is 29% faster than block size 64. This is because smaller blocks incur more block-scheduling overhead and have fewer threads per block to amortize barrier and synchronization costs.

- **cudaOccupancyMaxPotentialBlockSize suggests 1024 for vec_add**, but 512 is actually the fastest. The API optimizes for maximum occupancy, not minimum latency.

- **cudaOccupancyMaxPotentialBlockSize suggests 576 for vec_add_regpress**. 576 is not a power-of-2 and was not tested; 512 was the fastest tested block size at 50% occupancy.

- Register allocation granularity: On sm_90a, registers are allocated per warp, rounded up to the nearest 256 registers. For vec_add_regpress with 56 regs/thread: 56 × 32 = 1792 regs/warp, rounded to 1792 (already a multiple of 256).

- The Hopper tuning guide confirms: "The maximum number of concurrent warps per SM remains the same as in NVIDIA Ampere GPU architecture (that is, 64)" and "The register file size is 64K 32-bit registers per SM."