
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 276.127us
**Optimized**: 281.698us
**Improvement**: -2.0%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (13.7% → 25.69%)
**Insight**: Warp specialization added producer/consumer synchronization overhead (barrier stalls nearly doubled from 13.7% to 25.7%) and reduced active warps from 90.8% to 70.4%, without enabling meaningful compute-memory overlap — tensor core utilization stayed near zero (~0.03%), so the producer warpgroup idles without benefit.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 3284.868us
**Optimized**: 2727.097us
**Improvement**: 17.0%
**Key metric**: gpu__compute_memory_throughput (85.68% → 95.02%)
**Insight**: Warp-specialization guidance led to a 17% speedup primarily by slashing barrier stalls from 56.8% to 13.7% and reducing instruction count by 27%, pushing the kernel to the roofline (95% memory SOL), consistent with the skill's claim that WS reduces synchronization overhead and register pressure.
