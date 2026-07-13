# Layout Transform -- Pitfalls

## P1: Bank Conflicts in Transpose Tile
**Symptom**: Transpose kernel achieves only ~70% of expected bandwidth. Shared memory is the bottleneck.
**Detection**: Nsight Compute shows bank conflicts on shared memory stores (column write pattern).
**Fix**: Pad the shared memory array: `__shared__ float tile[TILE][TILE+1]` to break the stride-32 bank pattern.
**Source**: Best Practices Guide, Section 10.2.3.3 (Shared Memory in Matrix Multiply C=AAT)

## P2: Edge Tiles Not Handled for Non-Multiple Dimensions
**Symptom**: Out-of-bounds reads/writes for matrices where dimensions are not multiples of TILE.
**Detection**: Functional test with non-square, non-tile-multiple matrix sizes.
**Fix**: Add bounds checking for both read and write phases. Threads that would access out-of-bounds memory should skip both the load and store.
**Source**: Programming Guide, Section 2.2.4.2.1 (Matrix Transpose Using Shared Memory)

## P3: Transform Overhead Exceeds Benefit
**Symptom**: Total runtime increases after adding layout transform because the transform kernel is expensive relative to the savings.
**Detection**: Profile the transform kernel separately. Compare its cost to the savings in subsequent kernels.
**Fix**: Only transform if the data will be used by multiple subsequent kernels. Consider in-place transforms to avoid extra memory allocation. Batch transforms when possible.
**Source**: Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## P4: Pitch Mismatch Between Host and Device
**Symptom**: Incorrect data after cudaMemcpy2D because host pitch differs from device pitch from cudaMallocPitch.
**Detection**: Compare `cols * sizeof(float)` (host pitch) vs returned device pitch.
**Fix**: Use `cudaMemcpy2D` with explicit source and destination pitch values. Do not use `cudaMemcpy` for pitched allocations.
**Source**: Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)

## P5: Barrier stalls jumped from 0% to 20 (discovered in verification)

**Symptom**: Barrier stalls jumped from 0% to 20.27% due to __syncthreads, and roofline efficiency actually dropped (82.6%→65.0%) despite the 85% speedup — the baseline's high roofline % was misleading because it counted inefficient memory traffic as 'utilized bandwidth'.
**Source**: Level 3 sandbox verification (2026-04-04)

## P6: Using prmt (discovered in verification)

**Symptom**: Using prmt.b32 in a memory-bound kernel can shift the bottleneck profile (increasing long-scoreboard stall percentage) without improving latency, making profiling misleading if you only look at instruction counts
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: Shared memory occupancy dropped from 8 blocks/SM to 3 blocks/SM due to the 5248-byte shared memory allocation per block; on smaller matrices this reduced occupancy could hurt latency-bound workloads (discovered in verification)

**Symptom**: Shared memory occupancy dropped from 8 blocks/SM to 3 blocks/SM due to the 5248-byte shared memory allocation per block; on smaller matrices this reduced occupancy could hurt latency-bound workloads.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: Long scoreboard stalls increased from 45% to 80% after optimization — the kernel is now memory-latency bound rather than bandwidth-wasting, so further gains require increasing occupancy or hiding latency (note registers per thread rose from 16 to 18, reducing occupancy limit from 16 to 10 warps) (discovered in verification)

**Symptom**: Long scoreboard stalls increased from 45% to 80% after optimization — the kernel is now memory-latency bound rather than bandwidth-wasting, so further gains require increasing occupancy or hiding latency (note registers per thread rose from 16 to 18, reducing occupancy limit from 16 to 10 warps).
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: cudaMallocPitch padding increases total memory footprint, which can degrade L2 cache hit rate (here from 62 (discovered in verification)

**Symptom**: cudaMallocPitch padding increases total memory footprint, which can degrade L2 cache hit rate (here from 62.7% to 50.1%) and increase instruction count for pitch-based addressing — these hidden costs can neutralize the coalescing improvement
**Source**: Level 3 sandbox verification (2026-04-07)
