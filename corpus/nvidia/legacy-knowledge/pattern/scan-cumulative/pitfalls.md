# Scan (Cumulative) Pattern -- Pitfalls

## P1: Confusing Inclusive and Exclusive Scan
- **Symptom**: Output is off-by-one relative to expected results
- **Root cause**: Using inclusive scan implementation when exclusive is needed, or vice versa
- **Inclusive**: output[i] includes input[i] in the result
- **Exclusive**: output[i] does NOT include input[i] (output[0] = identity element)
- **Fix**: Clearly specify which variant is needed; convert between them via shift + identity prepend

## P2: Non-Associative Operator in Parallel Scan
- **Symptom**: Parallel scan produces different results from sequential scan
- **Root cause**: Scan algorithm requires an associative operator; floating-point addition is only approximately associative
- **Impact**: Results may differ slightly depending on reduction tree structure
- **Fix**: For exact reproducibility, use integer arithmetic or fixed-point representation
- **Mitigation**: For ML workloads, small floating-point differences are usually acceptable

## P3: Missing __threadfence in Decoupled Lookback
- **Symptom**: Blocks read stale aggregate values from other blocks, producing incorrect prefix sums
- **Root cause**: Writing aggregate to global memory and then setting the flag must be ordered; without `__threadfence()`, the flag may become visible before the aggregate
- **Fix**: Insert `__threadfence()` between writing the aggregate and setting the AGGREGATE_READY flag
- **Also**: The reading block should use appropriate memory fence when reading

## P4: Block Order Not Guaranteed Without Persistent Kernel
- **Symptom**: Decoupled lookback hangs because a block waits for a "previous" block that hasn't launched yet
- **Root cause**: CUDA does not guarantee block execution order; block N+1 may execute before block N
- **Fix**: Use a global counter (atomicAdd) for block ordering instead of blockIdx
- **Alternative**: Use persistent kernel where block assignment is explicitly controlled
- **Also**: Ensure enough blocks can be resident simultaneously for lookback chain to make progress

## P5: Spin-Wait Livelock in Lookback
- **Symptom**: Kernel hangs with some blocks spinning indefinitely
- **Root cause**: If block B waits for block A's flag, but block A can't launch because all SMs are occupied by blocks that are spin-waiting
- **Fix**: Never launch more blocks than can be simultaneously resident
- **Fix**: Use `__nanosleep()` in spin loop to reduce contention
- **Constraint**: Grid size should be <= num_SMs * max_blocks_per_SM

## P6: Warp-Level Scan Not Handling Partial Warps
- **Symptom**: Incorrect results when blockDim.x is not a multiple of 32
- **Root cause**: Last warp is partially active; `__shfl_up_sync` with incorrect active mask
- **Fix**: Always use full warps (blockDim.x should be multiple of 32)
- **Fix**: Or use proper active mask for partial warps: `__activemask()` or computed mask

## P7: Shared Memory Race in Block-Level Scan
- **Symptom**: Non-deterministic incorrect results in block-level scan
- **Root cause**: Missing `__syncthreads()` between phases (up-sweep and down-sweep, or warp-totals write and read)
- **Fix**: Insert `__syncthreads()` at every phase boundary where SMEM data dependencies exist

## P8: Large Working Set for Decoupled Lookback
- **Symptom**: Out-of-memory or excessive memory usage for very large scans
- **Root cause**: Global status array needs 3 values per block (status flag, aggregate, inclusive prefix)
- **For 1 billion elements with 256 elements/block**: 4M blocks * 12 bytes = 48MB
- **Fix**: Usually acceptable; but if memory is tight, use larger blocks to reduce block count

## P9: Segmented Scan Boundary Propagation Error
- **Symptom**: Segments bleed into each other; values from previous segment affect current segment
- **Root cause**: Segment flag not properly propagated through the warp-level or block-level scan
- **Fix**: In warp scan: `val = (flag_prefix > 0) ? thread_data : thread_data + neighbor`
- **Fix**: In block scan: warp-total for a warp containing a segment start should be the partial sum from the last segment start

## P10: Using Scan When Reduction Would Suffice
- **Symptom**: Computing prefix sums but only using the last element
- **Root cause**: If only the total (final element) is needed, scan does unnecessary work -- all intermediate prefix sums are wasted
- **Fix**: Use a simple reduction instead of scan; reduction is simpler and faster
- **When scan IS needed**: when all intermediate prefix sums are used (e.g., scatter indices)

## P11: The optimized kernel is heavily underutilized (20 (discovered in verification)

**Symptom**: The optimized kernel is heavily underutilized (20.6% compute SOL, 15.9% memory SOL) despite being fast in wall time; for this small problem size, launch overhead and insufficient parallelism dominate, so raw SOL% is misleading — wall time and instruction count are the true metrics.
**Source**: Level 3 sandbox verification (2026-04-06)

## P12: Long scoreboard stalls nearly doubled (24% → 42%) because the kernel is now so lightweight that memory latency dominates; the bottleneck shifted from compute to underutilized, meaning further gains require latency hiding (e (discovered in verification)

**Symptom**: Long scoreboard stalls nearly doubled (24% → 42%) because the kernel is now so lightweight that memory latency dominates; the bottleneck shifted from compute to underutilized, meaning further gains require latency hiding (e.g., more ILP or persistent-thread approaches).
**Source**: Level 3 sandbox verification (2026-04-06)

## P13: Roofline efficiency dropped from 55 (discovered in verification)

**Symptom**: Roofline efficiency dropped from 55.5% to 32.2% despite 35% faster wall time — the warp-scan approach finishes so quickly that hardware utilization percentages decrease; do not use roofline efficiency as a proxy for absolute performance on latency-sensitive scan kernels.
**Source**: Level 3 sandbox verification (2026-04-06)

## P14: Decoupled lookback on small arrays is counterproductive — the inter-block spin-wait and uncoalesced status-flag reads introduce overhead that dwarfs any launch-savings from single-pass execution; occupancy also dropped (register limit 16→10 blocks) due to extra state tracking (discovered in verification)

**Symptom**: Decoupled lookback on small arrays is counterproductive — the inter-block spin-wait and uncoalesced status-flag reads introduce overhead that dwarfs any launch-savings from single-pass execution; occupancy also dropped (register limit 16→10 blocks) due to extra state tracking.
**Source**: Level 3 sandbox verification (2026-04-06)

## P15: Optimized kernel dropped occupancy_limit_registers from 4 to 2 (registers per thread increased 16→17) and shifted the bottleneck profile: barrier stalls fell (40%→23%) but long_scoreboard stalls tripled (6%→21%), indicating the conversion introduced more memory-latency-dependent instruction chains that could hurt at larger problem sizes (discovered in verification)

**Symptom**: Optimized kernel dropped occupancy_limit_registers from 4 to 2 (registers per thread increased 16→17) and shifted the bottleneck profile: barrier stalls fell (40%→23%) but long_scoreboard stalls tripled (6%→21%), indicating the conversion introduced more memory-latency-dependent instruction chains that could hurt at larger problem sizes.
**Source**: Level 3 sandbox verification (2026-04-06)

## P16: The optimized kernel's L1 hit rate dropped to 0% and L2 hit rate halved (93% → 54%), yet performance improved dramatically — coalesced access patterns matter far more than cache hit rates for scan workloads, so optimizing for cache hits in isolation can be misleading (discovered in verification)

**Symptom**: The optimized kernel's L1 hit rate dropped to 0% and L2 hit rate halved (93% → 54%), yet performance improved dramatically — coalesced access patterns matter far more than cache hit rates for scan workloads, so optimizing for cache hits in isolation can be misleading.
**Source**: Level 3 sandbox verification (2026-04-06)

## P17: Occupancy is now register-limited (10 warps vs 32 block-limit) and barrier stalls appeared at 14 (discovered in verification)

**Symptom**: Occupancy is now register-limited (10 warps vs 32 block-limit) and barrier stalls appeared at 14.6%, suggesting the work-efficient scan's up/down-sweep synchronization phases are becoming visible bottlenecks at this occupancy level.
**Source**: Level 3 sandbox verification (2026-04-06)
