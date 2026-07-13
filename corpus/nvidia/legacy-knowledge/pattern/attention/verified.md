# Attention — Verification Data

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2181.814us
**Optimized**: 2206.886us
**Improvement**: -1.1%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (97.83% → 80.82%)
**Insight**: The optimized kernel is 1.1% slower and drops roofline efficiency from 97.8% to 80.8%; while it drastically reduces DRAM reads (4.3MB→8KB) by hitting L2 cache (100% hit rate), this doesn't translate to wall-clock improvement because both versions remain ~15x slower than the PyTorch baseline, indicating the persistent-kernel tile-scheduling overhead dominates any cache-locality gains.

## Experiment: Skill 7 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 620.281us
**Optimized**: 611.013us
**Improvement**: 1.49%
**Key metric**: gpu__compute_memory_throughput (94.31% → 94.76%)
**Insight**: Storing intermediates in shared memory successfully halved DRAM writes (5.5MB→2.5MB) and cut barrier stalls by 47%, but DRAM reads (~33.6MB) dominate and are unchanged, so the memory-bound kernel sees negligible wall-clock gain; neither version uses tensor cores, leaving the actual backward-pass GEMM optimizations from the skill untested.
**Conditions**: only 1.5% wall-clock improvement despite halving DRAM writes; kernel remains memory-read bottlenecked at ~94% memory SOL
