
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 6.234us
**Optimized**: 4.502us
**Improvement**: 27.8%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (13.47% → 5.39%)
**Insight**: Tiled partitions replaced block-wide barriers with warp-level shuffles, cutting instructions by 59% (3.4M→1.4M) and barrier stalls from 13.5%→5.4%, yielding a 27.8% wall-clock speedup.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 205.244us
**Optimized**: 179.474us
**Improvement**: 12.6%
**Key metric**: lts__t_sector_hit_rate.pct (56.66% → 99.0%)
**Insight**: Grid-wide sync via cooperative groups keeps data resident in L2 cache across iterations (56.7% → 99% L2 hit rate), eliminating repeated kernel launch and host-device synchronization overhead, which more than compensates for the reduced occupancy and barrier stalls.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 40.755us
**Optimized**: 16.88us
**Improvement**: 58.6%
**Key metric**: cuda_extension_us (40.755us → 16.880us)
**Insight**: Using coalesced_threads() to form dynamic active-thread groups eliminated redundant warp-level work in divergent branches, cutting kernel time by 59% while slightly improving warp occupancy (6.28% → 6.94%) and compute-memory throughput (0.77% → 0.96%).

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 142.014us
**Optimized**: 142.054us
**Improvement**: -0.03%
**Key metric**: sm__throughput (84.94% → 85.09%)
**Insight**: The compiler already optimizes cooperative group handle creation and passing conventions, so manually hoisting the handle and passing by reference produces identical machine code (same register count, same instruction count, same cycle count) — this is a code-style best practice, not a performance optimization.
