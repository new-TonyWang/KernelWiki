# Occupancy Tuning -- Pitfalls

## P1: Launching with Non-Multiple-of-32 Block Size

**Symptom:** Reduced throughput; profiler shows under-populated warps wasting execution slots.

**Detection:** `blockDim.x % 32 != 0` in launch config. Nsight Compute warp occupancy metric shows partial warps.

**Fix:** Always use block sizes that are multiples of 32. Start with 128 or 256.

```cpp
// Bad
myKernel<<<grid, 100>>>(...);   // 3 warps + 4 threads = wasted slots

// Good
myKernel<<<grid, 128>>>(...);   // 4 full warps
```

> Source: BP 11.3 -- "Threads per block should be a multiple of warp size to avoid wasting computation on under-populated warps."

---

## P2: Maximizing Occupancy Blindly Without Profiling

**Symptom:** Occupancy is 100% but performance is worse than a lower-occupancy configuration due to increased register spilling.

**Detection:** Compare `-maxrregcount` variants. Check `ld.local` / `st.local` counts in PTX/SASS for spill evidence. Nsight Compute "register spill" metric.

**Fix:** Profile multiple configurations. For compute-intensive kernels with high ILP, try lower occupancy (e.g., 50%) with more registers per thread.

> Source: BP 11.3 -- "Higher occupancy does not always equate to better performance... A lower occupancy kernel will have more registers available per thread."

---

## P3: Missing __launch_bounds__ Causes Launch Failure on New Hardware

**Symptom:** Kernel works on one GPU but fails with `cudaErrorLaunchOutOfResources` on a different compute capability.

**Detection:** Launch failure at runtime when moving to new hardware. No compile-time warning.

**Fix:** Add `__launch_bounds__(maxThreadsPerBlock)` to every production kernel.

```cpp
// Without launch bounds -- may fail on future architectures
__global__ void kernel(float* data) { ... }

// With launch bounds -- guaranteed to work with up to 256 threads
__global__ void __launch_bounds__(256) kernel(float* data) { ... }
```

> Source: BP 11.1 -- "Failure to do so could lead to 'too many resources requested for launch' errors."

---

## P4: Register Allocation Granularity Causing Unexpected Occupancy Drops

**Symptom:** Small increase in register usage (e.g., 32 to 33 per thread) causes disproportionate occupancy drop.

**Detection:** `--ptxas-options=-v` shows register count near a boundary. Nsight Compute occupancy analysis shows register-limited occupancy.

**Fix:** Use `-maxrregcount` or `__launch_bounds__` to cap registers at the boundary (e.g., 32). Accept minor spilling if it improves net occupancy.

> Source: BP 11.1.1 -- "Register allocations are rounded up to the nearest 256 registers per warp."

---

## P5: Shared Memory Allocation Silently Limiting Occupancy

**Symptom:** Large dynamic or static shared memory allocations reduce the number of blocks that can be resident per SM.

**Detection:** `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns a low value. Nsight Compute shows shared memory as the limiting resource.

**Fix:**
1. Use the occupancy API with actual shared memory size to compute expected occupancy.
2. Reduce shared memory per block if possible (e.g., smaller tiles).
3. Consider using `cudaFuncSetAttribute` to opt into extended shared memory on architectures that support it.

```cpp
// Opt into larger shared memory on CC 8.0+
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
```

> Source: BP 11.4 -- "Shared memory can be helpful... However, it also can act as a constraint on occupancy."

---

## P6: Using Only One Large Block Per SM Instead of Multiple Smaller Blocks

**Symptom:** Kernel uses `__syncthreads()` heavily, causing the SM to idle while all warps in the single large block wait at the barrier.

**Detection:** Nsight Compute shows high "stall barrier" cycles with only 1 block per SM.

**Fix:** Use several smaller thread blocks so that warps from non-stalled blocks can fill the pipeline while other blocks are at barriers.

> Source: BP 11.3 -- "Use several smaller thread blocks rather than one large thread block per multiprocessor if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads()."

## P7: On simple, low-register kernels __launch_bounds__ is a no-op; it only matters when the compiler would otherwise spill or over-allocate registers, so applying it blindly adds annotation noise without benefit (discovered in verification)

**Symptom**: On simple, low-register kernels __launch_bounds__ is a no-op; it only matters when the compiler would otherwise spill or over-allocate registers, so applying it blindly adds annotation noise without benefit.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: L1 cache hit rate dropped from 21 (discovered in verification)

**Symptom**: L1 cache hit rate dropped from 21.78% to 0.0% after the block size change — the altered thread-to-data mapping can shift caching behavior unexpectedly, though here it was offset by better coalescing and higher DRAM throughput.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: Occupancy limit from registers stayed at 8 blocks in both cases, meaning the __launch_bounds__ change didn't actually shift the occupancy regime — the compiler simply used 2 more registers without any structural benefit, making the trade purely negative on a memory-bound kernel (discovered in verification)

**Symptom**: Occupancy limit from registers stayed at 8 blocks in both cases, meaning the __launch_bounds__ change didn't actually shift the occupancy regime — the compiler simply used 2 more registers without any structural benefit, making the trade purely negative on a memory-bound kernel.
**Source**: Level 3 sandbox verification (2026-04-04)

## P10: This skill is a diagnostic/profiling technique (sweep shared memory to find sensitivity), not a direct optimization — verifying it requires showing a performance-vs-occupancy curve, not a single before/after comparison; a kernel already at low occupancy with no sensitivity will show no effect (discovered in verification)

**Symptom**: This skill is a diagnostic/profiling technique (sweep shared memory to find sensitivity), not a direct optimization — verifying it requires showing a performance-vs-occupancy curve, not a single before/after comparison; a kernel already at low occupancy with no sensitivity will show no effect.
**Source**: Level 3 sandbox verification (2026-04-04)

## P11: Occupancy-limiting resource shifted from registers (128 blocks) to registers (4 blocks) and shared memory (8 blocks) after tuning — the optimized kernel is now tightly constrained and any increase in per-thread registers or shared memory would immediately drop occupancy (discovered in verification)

**Symptom**: Occupancy-limiting resource shifted from registers (128 blocks) to registers (4 blocks) and shared memory (8 blocks) after tuning — the optimized kernel is now tightly constrained and any increase in per-thread registers or shared memory would immediately drop occupancy.
**Source**: Level 3 sandbox verification (2026-04-04)

## P12: After optimization, occupancy is now limited by registers (limit=4 blocks) and shared memory (limit=8 blocks) rather than the block count cap (32), meaning further gains require reducing per-thread resource usage (discovered in verification)

**Symptom**: After optimization, occupancy is now limited by registers (limit=4 blocks) and shared memory (limit=8 blocks) rather than the block count cap (32), meaning further gains require reducing per-thread resource usage.
**Source**: Level 3 sandbox verification (2026-04-06)
