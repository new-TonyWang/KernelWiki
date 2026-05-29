
## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2.576us
**Optimized**: 2.829us
**Improvement**: -9.8%
**Key metric**: sm__cycles_active.avg (3426.59 cycles → 3842.12 cycles (+12.1%))
**Insight**: The async copy pipeline adds instruction overhead (+21% instructions, +2 registers/thread) that outweighs the reduced long-scoreboard stalls (45.7%→35.4%) on this small kernel, resulting in a net 9.8% slowdown.

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2.675us
**Optimized**: 3.159us
**Improvement**: -18.1%
**Key metric**: dram_throughput (19.94% → 16.0%)
**Insight**: The barrier-based memcpy_async added significant instruction overhead (855k vs 270k instructions) and coordination cost that outweighed any prefetch benefit on this small problem size, resulting in 18% slower execution and lower DRAM throughput.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 18806.07us
**Optimized**: 122.566us
**Improvement**: 99.35%
**Key metric**: cuda_extension_us (18806.07us (page-fault-dominated) → 122.57us (prefetched))
**Insight**: Prefetch eliminates on-demand page migration during kernel execution — the baseline's 18.8ms was dominated by unified-memory page faults, not compute; NCU kernel-level metrics are identical because the kernel code is unchanged, only when data arrives differs.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 80.56us
**Optimized**: 118.783us
**Improvement**: -47.4%
**Key metric**: dram_throughput (28.96% → 17.4%)
**Insight**: Software prefetch added ~2x instruction overhead (3.3M→6.8M) and increased active cycles by 70%, hurting throughput on a simple memory-bound kernel where the hardware prefetcher already handles the sequential access pattern.

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 3.738us
**Optimized**: 2.841us
**Improvement**: 24.0%
**Key metric**: l1tex__t_sector_hit_rate.pct (73.81% → 59.01%)
**Insight**: 16-byte async copy reduces L1 hit rate from 73.8% to 59.0% confirming L1 bypass, cuts active cycles by 16% (4898→4101), and improves warp utilization from 74% to 85.5%, yielding a 24% wall-clock speedup.

## Experiment: Skill 2 verification (2026-04-07)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 8.627us
**Optimized**: 8.864us
**Improvement**: -2.7%
**Key metric**: long_scoreboard_stalls (64.59% → 41.87%)
**Insight**: Double buffering successfully hides memory latency (long scoreboard stalls dropped from 65% to 42%, active cycles reduced 22%) but the 43% instruction overhead and increased barrier synchronization (11% to 17%) negate the latency-hiding gains for this small kernel, resulting in no net wall-clock improvement.
**Conditions**: Improves hardware utilization and hides memory latency stalls, but does not translate to wall-clock speedup at this problem size due to instruction overhead and increased barrier synchronization
