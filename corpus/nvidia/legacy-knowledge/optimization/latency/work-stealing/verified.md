
## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2108.209us
**Optimized**: 3587.41us
**Improvement**: -70.2%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (93.62% → 63.34%)
**Insight**: The work-stealing optimization destroyed warp occupancy (97.6% → 25.0%) and increased long scoreboard stalls (15.8% → 34.6%), likely because the atomic counter serializes tile acquisition across persistent blocks, causing most warps to idle waiting for the atomic and for irregular memory access patterns that thrash the memory subsystem.
