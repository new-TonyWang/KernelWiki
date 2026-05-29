# Bank Conflict Avoidance -- Pitfalls

## P1: Stride-32 Access Pattern Causing 32-Way Conflicts
**Symptom**: Shared memory bandwidth drops to 1/32 of peak. Kernel is much slower than expected for compute-light code.
**Detection**: Access pattern `smem[threadIdx.x * 32]` or writing columns in a 32-wide 2D array. Nsight shows high bank conflict count.
**Fix**: Pad the array inner dimension by +1: `__shared__ float tile[32][33]`.
**Source**: Best Practices Guide, Section 10.2.3.3 (Shared Memory in Matrix Multiply C=AAT)

## P2: Padding Wastes Shared Memory Capacity
**Symptom**: After padding by +1, the effective shared memory usage increases by ~3% per dimension, potentially reducing occupancy.
**Detection**: Compute total shared memory with padding: TILE * (TILE+1) * sizeof(float). Compare against per-SM limit.
**Fix**: If padding causes occupancy to drop, consider alternative conflict-avoidance strategies: swizzle-based addressing, or reducing the tile size slightly.
**Source**: Best Practices Guide, Section 10.2.3.1 (Shared Memory and Memory Banks)

## P3: Forgetting to Apply Swizzle on Both Read and Write
**Symptom**: Incorrect results when using XOR-based swizzle for writes but reading with original indices.
**Detection**: Functional test fails. Data appears scrambled.
**Fix**: The same swizzle function must be applied consistently for both store and load operations.
**Source**: Programming Guide, Section 4.11.2.2.5 (Shared-Memory Bank Swizzling)

## P4: Broadcast Assumed but Not Actually Occurring
**Symptom**: Expected broadcast (all threads reading same address) does not occur because addresses differ by a small amount.
**Detection**: Check that all 32 threads in the warp truly access the identical word, not just the same bank.
**Fix**: Verify with printf or assertion that all threads produce the same shared memory index. Broadcast only occurs when addresses match exactly.
**Source**: Programming Guide, Section 2.2.4.2 (Shared Memory Access Patterns)

## P5: Bank Size Mismatch for 64-bit Types
**Symptom**: Double-precision shared memory access shows 2x bank conflicts despite unit-stride pattern.
**Detection**: Default bank mode is 4-byte. Each double spans 2 banks, so consecutive doubles hit banks (0,1), (2,3), etc.
**Fix**: Set 8-byte bank mode: `cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)`.
**Source**: Programming Guide, Section 2.2.4.2.2 (Shared Memory Bank Conflicts)

## P6: gpu__compute_memory_throughput drops from 89% to 65 (discovered in verification)

**Symptom**: gpu__compute_memory_throughput drops from 89% to 65.4% after optimization because bank-conflict replays inflated the baseline metric — do not interpret lower compute_memory_throughput as a regression when bank conflicts are removed.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: After resolving bank conflicts, long_scoreboard stalls surged from 14 (discovered in verification)

**Symptom**: After resolving bank conflicts, long_scoreboard stalls surged from 14.6% to 57.5%, indicating that once shared memory is no longer the bottleneck, DRAM latency becomes the dominant stall — further optimization requires prefetching or increased occupancy to hide global memory latency.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: cudaDeviceSetSharedMemConfig is effectively a no-op on modern architectures; relying on it as an optimization is misleading and adds unnecessary API calls (discovered in verification)

**Symptom**: cudaDeviceSetSharedMemConfig is effectively a no-op on modern architectures; relying on it as an optimization is misleading and adds unnecessary API calls.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: Long-scoreboard stalls increased from 16 (discovered in verification)

**Symptom**: Long-scoreboard stalls increased from 16.2% to 36.3% and barrier stalls from 11.4% to 26.5%, suggesting the faster shared-memory access now exposes global memory latency and synchronization overhead as the new bottlenecks.
**Source**: Level 3 sandbox verification (2026-04-04)

## P10: Roofline efficiency_pct dropped from 89% to 65 (discovered in verification)

**Symptom**: Roofline efficiency_pct dropped from 89% to 65.4% despite a 2.2x speedup — resolving bank conflicts shifted the bottleneck from shared memory to DRAM latency, which can make the kernel appear less efficient on the roofline even though it is substantially faster.
**Source**: Level 3 sandbox verification (2026-04-04)
