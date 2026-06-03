---
title: Occupancy Tuning
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- scan
requires_sm: '>=3.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: spec
  anchor: Reference
artifacts:
  code: artifacts/experience/hw-probes/occupancy-sweep/occupancy_sweep_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v -o occupancy_sweep_probe occupancy_sweep_probe.cu
  introspection: artifacts/experience/hw-probes/occupancy-sweep/h200_device_static.json
  profile: ''
related_apis:
- cudaOccupancyMaxActiveBlocksPerMultiprocessor
- cudaOccupancyMaxPotentialBlockSize
- cudaOccupancyAvailableDynamicSMemPerBlock
related_skills:
- register-pressure
- compiler-hints
- ilp
id: skill-occupancy-tuning
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1087-L1089
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1089
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1093-L1097
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1123-L1124
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1133-L1134
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Architecture Guides/hopper-tuning-guide/cuda_hopper-tuning-guide_index.html.md
  anchor: L35-L42
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3862-L3890
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L17092-L17121
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3828-L3858
---
## What

Occupancy tuning is the process of choosing a launch configuration (block size, shared memory allocation, register cap) that maximizes the ratio of active warps to maximum warps per SM. On H200 (sm_90a), each SM can host up to 64 warps (2048 threads). Occupancy is defined as:

```
occupancy = active_warps_per_SM / max_warps_per_SM
```

where `max_warps_per_SM = 64` on H200 (same as Ampere, per the Hopper tuning guide). Active warps are determined by three limiting factors:

1. **Register file**: 65,536 32-bit registers per SM. If each thread uses R registers, the maximum number of threads is `65536 / R` (subject to allocation granularity: registers are allocated per warp, rounded up to the nearest 256 registers per warp).

2. **Shared memory**: 233,472 bytes per SM on H200. If each block uses S bytes of shared memory, the maximum number of blocks is `233472 / S` (subject to the shared memory carveout configuration).

3. **Max threads/blocks per SM**: 2048 threads and 32 blocks per SM. Even with zero register and shared memory pressure, occupancy is capped by these hardware limits.

The occupancy for a given kernel and block size is:

```
numBlocks = min(reg_limit, smem_limit, max_blocks_per_SM)
active_threads = numBlocks × blockSize
active_warps = active_threads / 32
occupancy = active_warps / 64
```

## Why

Occupancy determines how many warps the warp scheduler can choose from when one warp stalls (on memory, arithmetic dependency, or synchronization). More active warps means better latency hiding. However, the best-practices guide explicitly warns: "Higher occupancy does not always equate to higher performance — there is a point above which additional occupancy does not improve performance."

The measured data confirms this. On H200:

- For a simple vec_add (12 regs/thread), all block sizes from 64 to 1024 achieve 100% occupancy, yet block size 512 is **29% faster** than block size 64. The difference comes from scheduling overhead: more blocks means more block-setup cost and less efficient memory coalescing per block.

- For a register-heavy kernel (56 regs/thread), block size 64 achieves 56.2% occupancy while block size 512 achieves only 50.0% occupancy — yet block size 512 is **10% faster**. The lower-occupancy configuration has fewer blocks to schedule and better per-block memory throughput.

The key insight: **50% occupancy with zero spills often beats 100% occupancy with spills** (see register-pressure skill). Even without spills, the relationship between occupancy and performance is non-monotonic.

## When to use

- **Initial kernel development**: Use `cudaOccupancyMaxPotentialBlockSize` to get a reasonable starting block size. This API returns the block size that maximizes theoretical occupancy, which is a good first approximation.

- **Register-bound kernels**: When `-Xptxas=-v` shows high register usage (e.g., >32 regs/thread on H200), use the occupancy API to understand how register pressure limits active warps. Consider `__launch_bounds__` or `-maxrregcount` to trade registers for occupancy (see compiler-hints skill).

- **Shared-memory-bound kernels**: When a kernel uses significant shared memory per block, use `cudaOccupancyAvailableDynamicSMemPerBlock` to determine how much dynamic shared memory is available while maintaining a target number of blocks per SM.

- **Block size selection**: When choosing between block sizes, always verify occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and then benchmark. The occupancy-optimal block size is a starting point, not the final answer.

- **Diagnosing low occupancy**: When a kernel shows unexpectedly low performance, check occupancy first. If occupancy is below 50%, the limiting factor (registers, shared memory, or max threads) tells you which resource to optimize.

## When NOT to use

- **Do not blindly maximize occupancy**: The best-practices guide warns that "improving occupancy from 66 percent to 100 percent generally does not translate to a similar increase in performance." If the kernel is compute-bound with high ILP, lower occupancy with more registers per thread may be faster.

- **Do not sacrifice correctness for occupancy**: Forcing register counts down with `-maxrregcount` or aggressive `__launch_bounds__` can cause register spilling. Spills to local memory have global-memory latency and typically hurt performance far more than reduced occupancy.

- **Do not assume occupancy predicts latency**: The measured data shows that at equal occupancy (100%), block size 512 is 29% faster than block size 64 for a memory-bound kernel. Occupancy is a necessary condition for good performance, not a sufficient one.

- **Do not use occupancy APIs inside performance-critical paths**: The occupancy APIs are host-side queries that involve driver calls. Call them once during initialization, not inside a loop.

## H200 occupancy limits (sm_90a)

From the Hopper tuning guide and `kp_introspect device-static`:

| Resource | Limit |
|---|---|
| Max warps per SM | 64 |
| Max threads per SM | 2048 |
| Max blocks per SM | 32 |
| Register file per SM | 65,536 × 32-bit |
| Max registers per thread | 255 |
| Shared memory per SM | 233,472 bytes |
| Max shared memory per block (opt-in) | 232,448 bytes |
| Default shared memory per block | 49,152 bytes |
| Register allocation granularity | 256 registers per warp |

### Register-derived occupancy table (H200)

For a kernel using R registers per thread, the maximum occupancy is:

| R (regs/thread) | Max threads/SM | Occupancy | Limiting factor |
|---|---|---|---|
| ≤ 32 | 2048 | 100% | Max threads |
| 33–40 | 1536–1632 | 75–80% | Registers |
| 41–64 | 1024–1536 | 50–75% | Registers |
| 65–128 | 512–1008 | 25–50% | Registers |
| 129–255 | 256–504 | 12.5–25% | Registers |

Note: actual occupancy depends on register allocation granularity (rounded to 256 per warp) and the interaction with block size.

## Block size heuristics

The best-practices guide recommends: "The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing."

Common starting points and their tradeoffs:

| Block size | Typical use | Pros | Cons |
|---|---|---|---|
| 64 | Debugging, fine-grained work | Low latency per block | High scheduling overhead, may hit max-blocks limit |
| 128 | Moderate register pressure | Good balance, fits many blocks/SM | Less ILP opportunity per block |
| **256** | **General-purpose default** | **Sweet spot for most kernels** | May not maximize occupancy for high-register kernels |
| 512 | Memory-bound kernels | Low scheduling overhead, good coalescing | Fewer blocks/SM, less flexibility |
| 1024 | Maximize occupancy for low-reg kernels | Minimal block overhead | Only 2 blocks/SM, poor occupancy if reg-heavy |

**256 is often the sweet spot** because:
- It is a multiple of the warp size (32), ensuring full warps.
- It allows 4–8 blocks per SM on H200 (depending on register usage), giving the scheduler flexibility.
- It balances scheduling overhead against per-block resource consumption.
- It works well with warp-level primitives (see warp-primitives skill).

## Using the Occupancy API

### cudaOccupancyMaxActiveBlocksPerMultiprocessor

Returns the maximum number of blocks that can reside on an SM for a given kernel and block size:

```cuda
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, my_kernel, blockSize, dynamicSMemSize);

int activeWarps = numBlocks * blockSize / 32;
double occupancy = (double)activeWarps / 64.0;  // 64 max warps on H200
```

### cudaOccupancyMaxPotentialBlockSize

Returns the block size that maximizes occupancy and the minimum grid size needed to achieve that occupancy across the whole device:

```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize, my_kernel, dynamicSMemSize, blockSizeLimit);
```

**Important**: This API optimizes for occupancy, not latency. The measured data shows that the suggested block size (1024 for vec_add) is not always the fastest (512 was 1.7% faster). Treat the suggestion as a starting point.

### cudaOccupancyAvailableDynamicSMemPerBlock

Returns the maximum dynamic shared memory per block that allows a target number of blocks per SM:

```cuda
size_t dynamicSmemSize;
cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamicSmemSize, my_kernel, numBlocks, blockSize);
```

This is useful when you want to maximize shared memory usage while maintaining a minimum occupancy level.

## Classical example: occupancy-aware launch configuration

```cuda
__global__ void my_kernel(const float* __restrict__ in,
                          float*       __restrict__ out,
                          int n) {
    extern __shared__ float smem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... kernel body using smem ...
}

void launch_my_kernel(const float* in, float* out, int n) {
    int blockSize, minGridSize;

    // Step 1: Get occupancy-optimal block size
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, my_kernel, 0, 0);

    // Step 2: Check if we can add dynamic shared memory without
    //         reducing occupancy
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, my_kernel, blockSize, 0);

    size_t maxDynSmem;
    cudaOccupancyAvailableDynamicSMemPerBlock(
        &maxDynSmem, my_kernel, numBlocks, blockSize);

    // Step 3: Use the maximum available dynamic shared memory
    size_t dynSmem = min(maxDynSmem, (size_t)65536);  // cap at 64KB

    // Step 4: Compute grid size
    int gridSize = (n + blockSize - 1) / blockSize;

    // Step 5: Opt-in for large shared memory if needed
    if (dynSmem > 49152) {
        cudaFuncSetAttribute(my_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmem);
    }

    my_kernel<<<gridSize, blockSize, dynSmem>>>(in, out, n);
}
```

## Measured Characteristics

- Occupancy sweep probe: On H200 (sm_90a, CUDA 12.9), sweeping block sizes 64–1024 on two kernels:

  **Simple vec_add (12 regs/thread, memory-bound)**: All block sizes achieve 100% occupancy. Block size 512 is the fastest (0.0057 ms median), 29% faster than block size 64 (0.0080 ms). `cudaOccupancyMaxPotentialBlockSize` suggests 1024, but 512 is 1.7% faster.

  **Register-heavy vec_add_regpress (56 regs/thread)**: Block sizes 64/128 achieve 56.2% occupancy; block sizes 256/512/1024 achieve 50.0% occupancy. Despite lower occupancy, block size 512 (0.0297 ms) is 10% faster than block size 64 (0.0330 ms). This confirms that higher occupancy does not guarantee better performance.