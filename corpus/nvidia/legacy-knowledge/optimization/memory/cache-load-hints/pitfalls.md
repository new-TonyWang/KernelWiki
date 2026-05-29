# Cache Load Hints -- Pitfalls

## P1: Overusing __ldg() on Data That Is Modified
**Symptom**: Stale data read because __ldg() uses the non-coherent read-only cache path.
**Detection**: Incorrect results when kernel reads data that was recently written by the same or another kernel. Texture cache is not invalidated by global stores.
**Fix**: Only use __ldg() for data that is truly read-only for the entire kernel execution. For data modified by the same kernel, use default loads.
**Source**: Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## P2: Cache Hints Ignored by Compiler or Hardware
**Symptom**: No performance change despite using cache hint intrinsics.
**Detection**: Profile shows identical cache behavior with and without hints. Compiler may merge or reorder operations.
**Fix**: Cache hints are advisory. Verify with PTX output (-ptx) that the expected cache operator is present. On some architectures, certain hints have no effect.
**Source**: Programming Guide, Section 2.2.3.6 (Caches)

## P3: L1 Bypass Causing Increased L2 Pressure
**Symptom**: Using __ldcg() everywhere causes L2 miss rate to increase because more traffic goes through L2.
**Detection**: L2 utilization and miss rate increase in Nsight Compute.
**Fix**: Only bypass L1 for data that is not reused within the SM. Keep frequently reused data in L1 with default .ca loads.
**Source**: Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## P4: Streaming Hint on Reused Data Causing Cache Misses
**Symptom**: Using __ldcs() on data that is actually accessed multiple times causes repeated cache misses.
**Detection**: Data is loaded with .cs hint but profiler shows the same addresses being fetched repeatedly from memory.
**Fix**: Reserve __ldcs() for genuinely streaming (single-use) data. If data is reused, use default .ca load.
**Source**: Programming Guide, Section 5.4.8.3 (Low-Level Load and Store Functions)

## P5: cudaFuncSetCacheConfig Overridden by Runtime
**Symptom**: Requesting cudaFuncCachePreferL1 but kernel still gets default L1/shared split.
**Detection**: Profile shows shared memory carveout unchanged.
**Fix**: The cache config is a preference, not a guarantee. Use `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout` for more direct control on newer architectures.
**Source**: Programming Guide, Section 3.2.6 (Configuring L1/Shared Memory Balance)

## P6: The skill description itself acknowledges this: 'The compiler may already generate ld (discovered in verification)

**Symptom**: The skill description itself acknowledges this: 'The compiler may already generate ld.global.nc for const __restrict__ pointers' — in practice this means __ldg() is a no-op on any reasonably modern toolchain with proper pointer qualifiers.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: On simple streaming kernels the default load path already behaves like __ldcg(); the hint only matters when there is measurable L1 contention from competing data that would benefit from L1 residency (discovered in verification)

**Symptom**: On simple streaming kernels the default load path already behaves like __ldcg(); the hint only matters when there is measurable L1 contention from competing data that would benefit from L1 residency.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: DRAM bytes written dropped ~14 (discovered in verification)

**Symptom**: DRAM bytes written dropped ~14.8% (48.3M → 41.2M) with __ldcs, suggesting the cache-streaming hint altered write-back behavior, but this did not translate to any execution time improvement — metric shifts from cache hints can be misleading without end-to-end speedup.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: Store cache hints only matter when stores are the bottleneck; in load-latency-bound kernels they are irrelevant and may even slightly degrade performance by bypassing L2 write-back coalescing (discovered in verification)

**Symptom**: Store cache hints only matter when stores are the bottleneck; in load-latency-bound kernels they are irrelevant and may even slightly degrade performance by bypassing L2 write-back coalescing.
**Source**: Level 3 sandbox verification (2026-04-04)

## P10: On Ampere and later architectures, the L1/shared memory split is managed automatically by hardware; cudaFuncSetCacheConfig has no observable effect and may give a false sense of tuning (discovered in verification)

**Symptom**: On Ampere and later architectures, the L1/shared memory split is managed automatically by hardware; cudaFuncSetCacheConfig has no observable effect and may give a false sense of tuning.
**Source**: Level 3 sandbox verification (2026-04-04)

## P11: The __ldlu() hint did reduce DRAM write volume (likely fewer dirty L2 evictions), but this benefit is invisible in runtime for memory-bandwidth-bound kernels that are already underutilized — the cache pressure relief only matters when other concurrent kernels or data streams are competing for cache capacity (discovered in verification)

**Symptom**: The __ldlu() hint did reduce DRAM write volume (likely fewer dirty L2 evictions), but this benefit is invisible in runtime for memory-bandwidth-bound kernels that are already underutilized — the cache pressure relief only matters when other concurrent kernels or data streams are competing for cache capacity.
**Source**: Level 3 sandbox verification (2026-04-04)
