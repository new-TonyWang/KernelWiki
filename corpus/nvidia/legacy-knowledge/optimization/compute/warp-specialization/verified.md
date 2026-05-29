
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 5.085us
**Optimized**: 9.45us
**Improvement**: -85.8%
**Key metric**: sm__inst_executed.sum (1138688 instructions, 100% global ld efficiency → 3518464 instructions (3.1x more), 0% global ld efficiency)
**Insight**: The producer-consumer warp partitioning adds 3x instruction overhead from pipeline bookkeeping and completely destroys global memory coalescing (100% → 0% load efficiency), making it far slower on this small (~4MB) problem where the torch baseline already finishes in 2.7µs.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.91us
**Optimized**: 11.731us
**Improvement**: -200.0%
**Key metric**: sm__cycles_active.avg (5698 → 31790)
**Insight**: The mbarrier producer-consumer pattern introduced massive synchronization overhead (barrier stalls appeared, long scoreboard stalls jumped from 54% to 77%, active cycles 5.6x higher) that far outweighs any overlap benefit on this small workload, resulting in a 3x slowdown.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 71.625us
**Optimized**: 71.609us
**Improvement**: 0.02%
**Key metric**: sm__throughput (46.17% → 45.53%)
**Insight**: Warp election via elect.sync provides no measurable benefit here because the baseline kernel has zero barrier stalls (0%) and the workload is memory-latency bound (67% long scoreboard stalls), so optimizing leader selection is irrelevant to the actual bottleneck.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 120.016us
**Optimized**: 82.432us
**Improvement**: 31.3%
**Key metric**: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct (44.34% → 15.09%)
**Insight**: Multi-stage buffering dramatically reduced long scoreboard stalls (44.3% → 15.1%) by prefetching enough tiles to hide global memory latency, cutting active cycles from 92K to 72K and boosting memory throughput from 34.3% to 43.6% SOL.
