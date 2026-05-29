# Vectorized Access -- Pitfalls

## P1: Misaligned Vector Load Causing Multiple Transactions
**Symptom**: Bandwidth drops significantly when using float4 loads; Nsight Compute shows higher-than-expected transaction count.
**Detection**: Check `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` in Nsight Compute. If it is ~2x the expected value, alignment is wrong.
**Fix**: Ensure base pointer is aligned to vector width (16 bytes for float4). Use `cudaMalloc` (guarantees 256-byte alignment) or `__align__(16)` on host-side allocations. For sub-array offsets, ensure offset is a multiple of 4 elements (16 bytes).
**Source**: Best Practices Guide, Section 10.2.1.2 (Sequential but Misaligned Access)

## P2: Remainder Elements Not Handled
**Symptom**: Incorrect results for arrays whose length is not divisible by the vector width (2 or 4).
**Detection**: Functional test with array sizes that are not multiples of the vector width.
**Fix**: Add a scalar tail loop after the vectorized main loop to handle remaining elements.
**Source**: Best Practices Guide, Section 10.2.1 (Coalesced Access)

## P3: Register Pressure Explosion from Excessive Vectorization
**Symptom**: Occupancy drops; kernel becomes slower despite fewer instructions. `--ptxas-options=-v` reports high register count.
**Detection**: Compare register usage with and without vectorization using `--ptxas-options=-v`. Check occupancy with Nsight Compute.
**Fix**: Reduce vector width (use float2 instead of float4), or apply `__launch_bounds__` to cap register usage. Trade vectorization for occupancy.
**Source**: Best Practices Guide, Section 10.2.7.1 (Register Pressure)

## P4: Vectorized Store to Non-Contiguous Addresses
**Symptom**: Attempting to write a float4 to scattered (non-contiguous) addresses causes correctness errors or uncoalesced writes.
**Detection**: Review store pattern; each thread's float4 store must target contiguous 16-byte aligned addresses.
**Fix**: Only use vectorized stores when output addresses are contiguous and aligned. For scatter patterns, fall back to scalar stores.
**Source**: Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)

## P5: Async Copy Size Mismatch Causes Undefined Shared Memory Content
**Symptom**: Shared memory contains garbage data after `__pipeline_wait_prior(0)`.
**Detection**: Validate that the byte count in `__pipeline_memcpy_async` matches actual data size.
**Fix**: The size parameter must be exactly 4, 8, or 16 bytes. Partial copies are not supported. Pad data to match these sizes.
**Source**: Best Practices Guide, Section 10.2.3.4 (Asynchronous Copy)

## P6: Long scoreboard stalls increase from 75% to 85% of warp active cycles — vectorized loads have higher per-instruction latency, so if occupancy is already low, the kernel may become more sensitive to latency hiding (discovered in verification)

**Symptom**: Long scoreboard stalls increase from 75% to 85% of warp active cycles — vectorized loads have higher per-instruction latency, so if occupancy is already low, the kernel may become more sensitive to latency hiding.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: Despite fewer instructions and cycles, DRAM throughput only improved marginally (27 (discovered in verification)

**Symptom**: Despite fewer instructions and cycles, DRAM throughput only improved marginally (27.1%→29.7%) and long scoreboard stalls increased (58.3%→66.2%), suggesting the kernel becomes more memory-latency-bound after removing compute overhead — the optimization exposes memory latency as the new bottleneck.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: L1 caching can mask poor global load efficiency — a kernel with 25% sector utilization may still appear fast because the L1 absorbs repeated accesses to the same cache lines, making vectorization look less impactful on wall-clock time than it truly is on hardware resource waste (discovered in verification)

**Symptom**: L1 caching can mask poor global load efficiency — a kernel with 25% sector utilization may still appear fast because the L1 absorbs repeated accesses to the same cache lines, making vectorization look less impactful on wall-clock time than it truly is on hardware resource waste.
**Source**: Level 3 sandbox verification (2026-04-04)
