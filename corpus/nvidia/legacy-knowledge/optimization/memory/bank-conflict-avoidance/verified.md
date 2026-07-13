
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 84.8us
**Optimized**: 37.968us
**Improvement**: 55.2%
**Key metric**: smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct (18.76% → 5.92%)
**Insight**: Padding eliminates shared memory bank conflicts, evidenced by short scoreboard stalls dropping from 18.76% to 5.92%; the 3.17x reduction in active cycles (154878→48795) shows warps no longer serialize on conflicting bank accesses.

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 85.484us
**Optimized**: 37.44us
**Improvement**: 56.2%
**Key metric**: sm__cycles_active.avg (158258 cycles (shared mem throughput 87.96%, DRAM 21.51%) → 48386 cycles (shared mem throughput 65.71%, DRAM 65.71%))
**Insight**: Eliminating bank conflicts removed the shared-memory bottleneck (short_scoreboard stalls dropped from 10.4% to 4.5%), shifting the bottleneck to DRAM and cutting active cycles by 3.3x — the baseline's 88% gpu_compute_memory_throughput was inflated by bank-conflict serialization, not useful work.

## Experiment: Skill 3 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 26.489us
**Optimized**: 26.345us
**Improvement**: 0.54%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (83.9% → 84.36%)
**Insight**: On modern GPUs (Ampere+), the 8-byte shared memory bank mode setting has no measurable effect — the hardware already handles double-precision accesses efficiently, and the 0.5% difference is within run-to-run noise.

## Experiment: Skill 4 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 132.537us
**Optimized**: 81.484us
**Improvement**: 38.5%
**Key metric**: smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct (32.05% → 12.65%)
**Insight**: XOR-swizzling shared memory indices eliminated bank conflicts, dropping short-scoreboard stalls from 32% to 12.6% and nearly halving active cycles (222k→116k), which shifted the bottleneck from memory to compute-underutilized.

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 84.697us
**Optimized**: 37.833us
**Improvement**: 55.3%
**Key metric**: smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct (18.86% → 5.9%)
**Insight**: Padding shared memory (5120→5248 bytes/block) eliminated bank conflicts, dropping short-scoreboard stalls from 18.86% to 5.9% and cutting execution time by 55%, though this exposed global memory latency (long-scoreboard stalls rose from 20.45% to 55.87%) as the next bottleneck.
