
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 4.506us
**Optimized**: 4.544us
**Improvement**: -0.84%
**Key metric**: sm__inst_executed.sum (1933264 → 1474560 (-23.7%))
**Insight**: Warp-boundary branching cut instructions by 24% and cycles by 7%, but restructuring which threads take which path halved global load coalescing (100% -> 50%), increasing memory stalls and fully negating the divergence benefit at the wall-clock level.
**Conditions**: only if warp-boundary branch restructuring preserves memory coalescing; if threads accessing contiguous data are split across warps, global load efficiency degrades and negates the divergence reduction

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 276.534us
**Optimized**: 4.979us
**Improvement**: 98.2%
**Key metric**: sm__inst_executed.sum (101416960 → 2557450)
**Insight**: Warp vote functions (__any_sync/__all_sync) enabled entire warps to early-exit collectively, reducing executed instructions by 97.5% (101M→2.6M) and cutting kernel time from 276.5us to 5.0us by skipping all unnecessary work without divergence.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 34.304us
**Optimized**: 47.491us
**Improvement**: -38.4%
**Key metric**: cuda_extension_us (37.33% warp active, 6.65% short scoreboard stall → 34.49% warp active, 9.98% short scoreboard stall)
**Insight**: The data restructuring added overhead (sorting/compaction) without payoff — instruction count is identical (94208), warp occupancy dropped, and short scoreboard stalls increased from 6.65% to 9.98%, indicating the baseline data layout did not actually cause meaningful divergence at this problem size.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.128us
**Optimized**: 4.138us
**Improvement**: -0.24%
**Key metric**: sm__inst_executed.sum (1212416 → 1245184 (+2.7% more instructions))
**Insight**: __syncwarp() is a correctness primitive for Volta+ independent thread scheduling, not a performance optimization — it adds ~2.7% more instructions with no latency benefit, and the compiler/hardware already handled reconvergence correctly in this test case.
