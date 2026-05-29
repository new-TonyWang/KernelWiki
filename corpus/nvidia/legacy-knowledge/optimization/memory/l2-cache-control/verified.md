
## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 660.09us
**Optimized**: 1133.453us
**Improvement**: -71.7%
**Key metric**: dram__bytes_write.sum (3072 bytes → 11035648 bytes (3592x increase))
**Insight**: The L2 persisting access window evicted useful cached data, causing massive DRAM write spills (3KB to 11MB) and a 72% slowdown, because the working set was already fitting well in L2 without intervention (baseline L2 hit rate ~105%).

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 87.366us
**Optimized**: 90.489us
**Improvement**: -3.6%
**Key metric**: lts__t_sector_hit_rate.pct (55.92% → 65.52%)
**Insight**: hitRatio tuning successfully improved L2 hit rate by ~10pp and cut DRAM reads by 38%, but the kernel was already at 91% memory throughput so the reduced DRAM traffic was absorbed by the memory subsystem without lowering latency — the cache policy overhead actually added ~3us.
**Conditions**: only if the kernel is not already saturating memory bandwidth (~90%+ compute_memory_throughput); when already near peak, reduced DRAM traffic from better L2 hits does not translate to wall-clock speedup

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 313.739us
**Optimized**: 313.54us
**Improvement**: 0.06%
**Key metric**: gpu__compute_memory_throughput (97.1% → 97.04%)
**Insight**: The baseline kernel is already at 97% memory roofline with 61.5% L2 hit rate — resetting L2 persisting lines has no measurable effect because the benchmark doesn't first establish persisting L2 lines that would interfere with subsequent kernels.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 862.999us
**Optimized**: 863.215us
**Improvement**: -0.025%
**Key metric**: lts__t_sector_hit_rate.pct (65.81% → 65.75%)
**Insight**: L2 persistence policy on graph kernel nodes had zero effect — L2 hit rate unchanged (65.81→65.75%) and execution time identical, likely because the working set either exceeds the persistable L2 fraction or the access pattern already achieves its natural cache residency without hints.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 111.25us
**Optimized**: 111.087us
**Improvement**: 0.15%
**Key metric**: lts__t_sector_hit_rate.pct (85.12% → 85.16%)
**Insight**: Querying L2 cache properties is a correctness/portability skill, not a performance optimization — it produces no kernel-level change, so baseline and optimized runs are effectively identical.
