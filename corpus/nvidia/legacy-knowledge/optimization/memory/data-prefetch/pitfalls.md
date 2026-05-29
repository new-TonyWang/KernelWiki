# Data Prefetch -- Pitfalls

## P1: Missing __syncthreads After Pipeline Wait
**Symptom**: Data written by one warp's async copy is not visible to other warps. Intermittent incorrect results.
**Detection**: Compute-sanitizer --tool racecheck. Results vary between runs.
**Fix**: After `__pipeline_wait_prior(N)`, add `__syncthreads()` before reading data written by other threads. The pipeline wait only guarantees the issuing thread's copies are complete.
**Source**: Programming Guide, Section 4.11.1 (Using LDGSTS)

## P2: Misaligned Source or Destination Pointers for Async Copy
**Symptom**: `__pipeline_memcpy_async` produces incorrect data or falls back to slower synchronous path.
**Detection**: Check that both global and shared memory pointers are aligned to the copy size (4, 8, or 16 bytes). Best performance requires 128-byte alignment.
**Fix**: Ensure alignment with `cuda::aligned_size_t<N>()` wrapper or manual alignment. Use cudaMalloc for global (256-byte aligned) and pad shared memory declarations.
**Source**: Programming Guide, Section 4.11.1 (Using LDGSTS)

## P3: Too Many Pipeline Stages Exhausting Shared Memory
**Symptom**: Multi-buffer scheme uses 3x or 4x shared memory, causing occupancy to drop to 1 block/SM.
**Detection**: Total shared memory = stages * tile_size. Compare with per-SM shared memory limit.
**Fix**: Use the minimum number of stages that hides memory latency (usually 2-3). Reduce tile size if needed. Measure with different stage counts.
**Source**: Programming Guide, Section 3.2.4.3 (Pipelines)

## P4: Prefetching Too Far Ahead Pollutes Cache
**Symptom**: Prefetched data evicts useful data from L2 cache before it is needed. Net performance decreases.
**Detection**: L2 hit rate drops when prefetch distance is increased. Nsight Compute shows increased L2 misses.
**Fix**: Tune prefetch distance: prefetch only 1-4 iterations ahead. Use L2 cache control hints to mark prefetched data as streaming.
**Source**: Programming Guide, Section 2.4.2.4 (Memory Advise and Prefetch)

## P5: cudaMemPrefetchAsync on Non-Managed Memory
**Symptom**: `cudaMemPrefetchAsync` returns error or has no effect because memory was allocated with cudaMalloc instead of cudaMallocManaged.
**Detection**: Check allocation API. cudaMemPrefetchAsync only works with managed memory.
**Fix**: Use `cudaMallocManaged()` or switch to explicit `cudaMemcpyAsync()` for device memory.
**Source**: Programming Guide, Section 2.4.2.4 (Memory Advise and Prefetch)

## P6: Pipeline Not Committed Before Wait
**Symptom**: `__pipeline_wait_prior(0)` returns immediately but data is not ready. Shared memory contains stale values.
**Detection**: Missing `__pipeline_commit()` between memcpy_async calls and wait.
**Fix**: Always call `__pipeline_commit()` after issuing async copies and before `__pipeline_wait_prior()`.
**Source**: Best Practices Guide, Section 10.2.3.4 (Asynchronous Copy)

## P7: Async copy (LDGSTS) can regress performance on small/simple kernels where the pipeline setup cost (commit/wait intrinsics, extra registers) dominates; the global load efficiency metric drops to 0% which can be misleading since loads are reclassified as async copies rather than eliminated (discovered in verification)

**Symptom**: Async copy (LDGSTS) can regress performance on small/simple kernels where the pipeline setup cost (commit/wait intrinsics, extra registers) dominates; the global load efficiency metric drops to 0% which can be misleading since loads are reclassified as async copies rather than eliminated.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: cuda::memcpy_async with barriers introduces substantial instruction count overhead (~3 (discovered in verification)

**Symptom**: cuda::memcpy_async with barriers introduces substantial instruction count overhead (~3.2x) and extra shared memory usage; the sector-level load efficiency dropped to 0% (smsp__sass_average_data_bytes_per_sector_mem_global_op_ld), suggesting the async copy path may use a different, less efficient memory access pattern than direct loads.
**Source**: Level 3 sandbox verification (2026-04-05)

## P9: NCU kernel profiling metrics (DRAM throughput, SM throughput) do not capture page-fault overhead, so evaluating prefetch effectiveness requires wall-clock timing, not just hardware counters (discovered in verification)

**Symptom**: NCU kernel profiling metrics (DRAM throughput, SM throughput) do not capture page-fault overhead, so evaluating prefetch effectiveness requires wall-clock timing, not just hardware counters.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: Software prefetch on simple sequential-access kernels adds instruction overhead that outweighs any cache warming benefit, since the HW prefetcher already handles linear patterns efficiently (discovered in verification)

**Symptom**: Software prefetch on simple sequential-access kernels adds instruction overhead that outweighs any cache warming benefit, since the HW prefetcher already handles linear patterns efficiently.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: Long scoreboard stalls remain dominant (38%) and barrier stalls increased (6 (discovered in verification)

**Symptom**: Long scoreboard stalls remain dominant (38%) and barrier stalls increased (6.4%→10.2%), suggesting the pipeline_wait synchronization itself becomes a new bottleneck as L1 latency hiding is removed.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: Global load efficiency (smsp__sass_average_data_bytes_per_sector_mem_global_op_ld) dropped from 100% to 0% in the optimized version, suggesting async copy operations bypass the metric or introduce uncoalesced access patterns that NCU reports differently — this metric cannot be trusted when using __pipeline_memcpy_async (discovered in verification)

**Symptom**: Global load efficiency (smsp__sass_average_data_bytes_per_sector_mem_global_op_ld) dropped from 100% to 0% in the optimized version, suggesting async copy operations bypass the metric or introduce uncoalesced access patterns that NCU reports differently — this metric cannot be trusted when using __pipeline_memcpy_async.
**Source**: Level 3 sandbox verification (2026-04-07)
