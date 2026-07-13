
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 67.673us
**Optimized**: 12.528us
**Improvement**: 81.5%
**Key metric**: sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed (0.03% → 2.72%)
**Insight**: WMMA replaced scalar FMA with hardware MMA ops, cutting instruction count 32x (32.8M → 1.0M) and achieving 5.4x speedup over the baseline kernel and 1.3x over torch.compile, despite lower occupancy (78.6% → 12.1%) and L1 hit rate (92% → 55%).
