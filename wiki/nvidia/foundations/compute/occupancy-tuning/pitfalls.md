---
title: Occupancy Tuning - Pitfalls
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
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
id: pitfall-occupancy-tuning
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1089
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1124
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1093-L1097
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1133-L1134
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1123
---
## P1: Assuming higher occupancy always means better performance

**Symptom**: Kernel performance does not improve (or gets worse) after increasing occupancy via register reduction or block size changes.

**Cause**: The best-practices guide explicitly states: "Higher occupancy does not always equate to higher performance — there is a point above which additional occupancy does not improve performance." The measured data confirms this: for a register-heavy kernel (56 regs/thread), block size 512 at 50% occupancy is 10% faster than block size 64 at 56.2% occupancy.

Occupancy helps hide latency by providing more warps for the scheduler to switch to. But once there are enough warps to cover the critical-path latency, additional warps provide diminishing returns. For compute-bound kernels with high ILP, even 25% occupancy may be sufficient.

**Detection**: Compare latency at different occupancy levels using the occupancy sweep probe pattern. If increasing occupancy does not reduce latency, you have reached the saturation point.

**Fix**: Do not chase 100% occupancy. Profile at multiple block sizes and register counts. Choose the configuration with the best latency, not the highest occupancy.

## P2: Using cudaOccupancyMaxPotentialBlockSize as the final answer

**Symptom**: The block size suggested by `cudaOccupancyMaxPotentialBlockSize` is not the fastest in practice.

**Cause**: This API returns the block size that maximizes **theoretical occupancy**, not the block size that minimizes latency. The measured data shows that for vec_add, the API suggests 1024 but 512 is 1.7% faster. For the register-heavy kernel, the API suggests 576 but 512 is fastest among the tested sizes.

The API does not account for:
- Block scheduling overhead (more blocks = more overhead)
- Memory coalescing efficiency (varies with block size)
- Cache behavior (L1/L2 hit rates depend on access patterns per block)
- Instruction-level parallelism within each thread

**Detection**: Always benchmark the suggested block size against nearby alternatives (e.g., 128, 256, 512).

**Fix**: Use `cudaOccupancyMaxPotentialBlockSize` as a starting point, then sweep block sizes around the suggestion. The occupancy-optimal block size is a good first guess, but the latency-optimal block size must be measured.

## P3: Forgetting register allocation granularity

**Symptom**: Occupancy calculated by hand does not match the occupancy API result.

**Cause**: On sm_90a, registers are allocated per warp, rounded up to the nearest 256 registers per warp. A kernel using 37 registers per thread requires 37 × 32 = 1184 registers per warp, which rounds up to 1280 (next multiple of 256). This rounding affects the total register consumption per block and thus the number of blocks that fit on an SM.

The best-practices guide gives an example: "on a device of compute capability 7.0, a kernel with 128-thread blocks using 37 registers per thread results in an occupancy of 75% with 12 active 128-thread blocks per multi-processor, whereas a kernel with 320-thread blocks using the same 37 registers per thread results in an occupancy of 63%."

**Detection**: Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` instead of hand calculation. The API accounts for all granularity and rounding effects.

**Fix**: Never hand-calculate occupancy for performance decisions. Always use the occupancy API or the Nsight Compute occupancy calculator.

## P4: Ignoring shared memory as an occupancy limiter

**Symptom**: Kernel has low occupancy despite low register usage.

**Cause**: Shared memory can limit occupancy independently of registers. On H200, each SM has 233,472 bytes of shared memory. If a kernel uses 128 KB of shared memory per block, only 1 block fits per SM (233472 / 131072 = 1.78 → 1 block), giving occupancy of blockSize / 2048 regardless of register usage.

The best-practices guide notes: "Shared memory can also act as a constraint on occupancy." This is especially common in tiled kernels (matrix multiply, convolution) where the tile size determines shared memory usage.

**Detection**: Check `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and compare against the register-derived limit. If the API returns fewer blocks than the register limit allows, shared memory is the bottleneck.

**Fix**: Reduce shared memory per block (smaller tiles, different data types), or use `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout` to increase the shared memory partition. Consider processing multiple elements per thread to amortize shared memory usage.

## P5: Block size not a multiple of warp size

**Symptom**: Kernel performance is lower than expected for the given occupancy.

**Cause**: The best-practices guide recommends: "The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing." A block size that is not a multiple of 32 wastes warp slots: the last partial warp still consumes a full warp's worth of scheduling resources but does less work.

**Detection**: Check if `blockSize % 32 != 0`. The occupancy API accounts for this correctly, but the effective throughput per warp is reduced.

**Fix**: Always use block sizes that are multiples of 32 (warp size). Common choices: 64, 128, 256, 512, 1024.

## P6: Forcing high occupancy causes register spilling

**Symptom**: Kernel becomes slower after adding `__launch_bounds__` with a high `minBlocksPerMultiprocessor` or after setting `-maxrregcount` to a low value. `-Xptxas=-v` shows non-zero spill stores/loads.

**Cause**: The best-practices guide warns: "A lower occupancy kernel will have more registers available per thread than a higher occupancy kernel, which may result in less register spilling to local memory." Forcing high occupancy by capping registers causes the compiler to spill excess values to local memory (which resides in global memory). The latency of spilled loads/stores far exceeds the benefit of having more active warps.

**Detection**: Compile with `-Xptxas=-v` and check for non-zero "spill stores" and "spill loads." Any non-zero spill count is a red flag.

**Fix**: Reduce `minBlocksPerMultiprocessor` or increase `-maxrregcount` until spill counts reach zero. A kernel at 50% occupancy with zero spills is almost always faster than the same kernel at 100% occupancy with spills. See the register-pressure skill for detailed guidance.

## P7: Assuming block size alone determines occupancy

**Symptom**: Changing block size does not change occupancy as expected.

**Cause**: The best-practices guide states: "When choosing the block size, it is important to remember that multiple concurrent blocks can reside on a multiprocessor, so occupancy is not determined by block size alone." A smaller block size allows more blocks per SM, potentially maintaining the same total thread count. For example, on H200 with a kernel using 12 registers per thread:
- Block size 256: 8 blocks × 256 = 2048 threads = 100% occupancy
- Block size 128: 16 blocks × 128 = 2048 threads = 100% occupancy
- Block size 64: 32 blocks × 64 = 2048 threads = 100% occupancy

All achieve the same occupancy despite different block sizes.

**Detection**: Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to check how many blocks fit per SM at each block size.

**Fix**: Understand that occupancy = (numBlocks × blockSize) / maxThreads. Both numBlocks and blockSize matter. The occupancy API handles this correctly.

## P8: Not opting in for large dynamic shared memory

**Symptom**: Kernel launch fails with "invalid configuration argument" when requesting more than 48 KB of dynamic shared memory.

**Cause**: By default, kernels are limited to 48 KB (49,152 bytes) of dynamic shared memory. Using more requires an explicit opt-in via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` before the launch. The occupancy API (`cudaOccupancyAvailableDynamicSMemPerBlock`) can tell you how much dynamic shared memory is available at a given occupancy target.

**Detection**: If the kernel uses `extern __shared__` and the third launch parameter exceeds 49152, the launch will fail without the opt-in call.

**Fix**: Call `cudaFuncSetAttribute` before the first launch:
```cuda
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, required_bytes);
my_kernel<<<grid, block, required_bytes>>>(...);
```

## P9: Overlooking the max-blocks-per-SM limit

**Symptom**: Occupancy is lower than expected even with very small block sizes and low register/shared memory usage.

**Cause**: H200 limits each SM to 32 concurrent blocks. With block size 32 (one warp per block), the maximum occupancy from blocks alone is 32 × 32 = 1024 threads = 50%, even though registers and shared memory would allow 2048 threads. This is the "max blocks per SM" limit.

**Detection**: If `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns 32 (or the device max) and `32 × blockSize < 2048`, the block count limit is the bottleneck.

**Fix**: Increase block size so that fewer blocks are needed to reach full occupancy. Block size 64 (2 warps) allows 32 × 64 = 2048 = 100%. Block size 32 (1 warp) is capped at 50%.

## P10: Using occupancy APIs in performance-critical paths

**Symptom**: Unexpected latency spikes in the host-side launch path.

**Cause**: The occupancy APIs (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`, etc.) involve driver calls that can take microseconds to milliseconds. Calling them before every kernel launch adds overhead to the launch path.

**Detection**: Profile the host-side code. If occupancy API calls appear in hot paths, they are adding unnecessary latency.

**Fix**: Call occupancy APIs once during initialization (or when the kernel configuration changes) and cache the results. The occupancy for a given kernel + block size + dynamic SMem is deterministic and does not change at runtime.