# Shared Memory Cache -- Pitfalls

## P1: Missing __syncthreads Before Reading Data Written by Other Warps
**Symptom**: Intermittent incorrect results; data races between warps accessing shared memory.
**Detection**: Results change between runs. Compute-sanitizer --tool racecheck detects the race.
**Fix**: Add `__syncthreads()` after the load phase and before the compute phase. Only `__syncwarp()` is needed for intra-warp sharing.
**Source**: Best Practices Guide, Section 10.2.3.2 (Shared Memory in Matrix Multiply C=AB)

## P2: Occupancy Drop from Excessive Shared Memory Usage
**Symptom**: Only 1 block per SM; kernel is latency-bound instead of compute/memory-bound.
**Detection**: Nsight Compute occupancy section shows shared memory as the limiting resource.
**Fix**: Reduce tile size, use dynamic shared memory with smaller allocation, or process multiple elements per thread with a smaller tile. Consider L1/shared carveout tuning.
**Source**: Best Practices Guide, Section 11.4 (Effects of Shared Memory)

## P3: Dynamic Shared Memory Alignment Errors
**Symptom**: Incorrect results or crashes when partitioning extern __shared__ into multiple arrays.
**Detection**: Second array pointer is not aligned to its type requirement (e.g., float* not 4-byte aligned).
**Fix**: Manually align partition offsets: `float* array1 = (float*)&short_array[ROUND_UP(short_count, 2)]`. Cast through char* to ensure byte-level alignment control.
**Source**: Programming Guide, Section 2.2.3.2.2 (Dynamic Allocation of Shared Memory)

## P4: Forgetting to Opt-in for >48KB Shared Memory
**Symptom**: Kernel launch fails with "too many resources requested" when using >48KB shared memory.
**Detection**: Launch returns cudaErrorLaunchOutOfResources.
**Fix**: Call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` before launch.
**Source**: Programming Guide, Section 2.2.3.2 (Shared Memory)

## P5: Shared Memory Used but No Data Reuse
**Symptom**: Kernel performance does not improve despite shared memory usage. Extra __syncthreads overhead adds latency.
**Detection**: Each element loaded into shared memory is read exactly once (no reuse). Profile shows no reduction in global memory transactions.
**Fix**: Remove unnecessary shared memory staging. Let hardware L1/L2 cache handle single-use data. Shared memory is only beneficial when data is reused or needs reordering.
**Source**: Best Practices Guide, Section 10.2.3 (Shared Memory)

## P6: Writing to Shared Memory from All Threads When Only Subset Needed
**Symptom**: Wasted instructions and potential bank conflicts from unnecessary writes.
**Detection**: Code writes to shared memory unconditionally but only a subset of values are used.
**Fix**: Guard shared memory writes with conditionals. Use cooperative group memcpy_async for collective loads.
**Source**: Programming Guide, Section 4.11.1.1 (Batching Loads in Conditional Code)

## P7: Shared memory usage tripled (1024→3072 bytes/block), reducing the occupancy limit from shared memory from 32 to 21 blocks; barrier stalls rose from 0% to 14% due to __syncthreads() — for kernels with few reuse opportunities, this overhead may negate the benefit (discovered in verification)

**Symptom**: Shared memory usage tripled (1024→3072 bytes/block), reducing the occupancy limit from shared memory from 32 to 21 blocks; barrier stalls rose from 0% to 14% due to __syncthreads() — for kernels with few reuse opportunities, this overhead may negate the benefit.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: Shared memory staging introduced significant bank conflicts (short_scoreboard stalls rose from 4 (discovered in verification)

**Symptom**: Shared memory staging introduced significant bank conflicts (short_scoreboard stalls rose from 4.5% to 32.0%) and reduced occupancy (shared_mem block limit dropped from 8 to 3), indicating the TILE layout likely causes bank conflicts that a padding strategy (TILE+1 columns) could mitigate.
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: Dynamic shared memory is not a free abstraction; when the tile size is effectively static, using extern __shared__ can introduce slight overhead compared to statically-sized shared memory, and the compiler may optimize static allocations more aggressively (discovered in verification)

**Symptom**: Dynamic shared memory is not a free abstraction; when the tile size is effectively static, using extern __shared__ can introduce slight overhead compared to statically-sized shared memory, and the compiler may optimize static allocations more aggressively.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: The carveout API successfully changed launch__occupancy_limit_shared_mem from 3 to 25 (more blocks allowed by shared memory), but occupancy was already limited by registers (limit=2), so the shared memory headroom increase was completely irrelevant (discovered in verification)

**Symptom**: The carveout API successfully changed launch__occupancy_limit_shared_mem from 3 to 25 (more blocks allowed by shared memory), but occupancy was already limited by registers (limit=2), so the shared memory headroom increase was completely irrelevant.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: Large dynamic shared memory competes with L1 cache on the same physical SRAM; over-allocating shared memory can zero out L1 hit rate and severely limit block-level occupancy even if the extra shared memory goes unused (discovered in verification)

**Symptom**: Large dynamic shared memory competes with L1 cache on the same physical SRAM; over-allocating shared memory can zero out L1 hit rate and severely limit block-level occupancy even if the extra shared memory goes unused.
**Source**: Level 3 sandbox verification (2026-04-05)
