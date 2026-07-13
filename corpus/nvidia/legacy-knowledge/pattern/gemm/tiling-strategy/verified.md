
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 275.335us
**Optimized**: 85.706us
**Improvement**: 68.9%
**Key metric**: sm__inst_executed.sum (125M instructions, 32 regs/thread, 3KB smem, 90.7% warps active → 48M instructions, 72 regs/thread, 9KB smem, 24.3% warps active)
**Insight**: Larger tiles with higher arithmetic intensity reduced total instructions by 2.6x and active cycles by 3.2x; the data reuse benefit far outweighs the occupancy drop from 90% to 24% active warps.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 1858.279us
**Optimized**: 1888.778us
**Improvement**: -1.6%
**Key metric**: gpu__compute_memory_throughput (91.36% → 90.15%)
**Insight**: Increasing shared memory from 9KB to 17KB per block (more pipeline stages) added 14% more instructions without improving throughput because the kernel is already memory-bound at the L2 cache level (96% hit rate) with negligible tensor core usage, so extra staging just adds overhead.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 2078.124us
**Optimized**: 596.547us
**Improvement**: 71.3%
**Key metric**: sm__cycles_active.avg (4091721 cycles, 32 regs/thread, 3KB smem, 94.8% roofline efficiency → 1169145 cycles, 72 regs/thread, 9KB smem, 80.3% roofline efficiency)
**Insight**: Larger tiles (72 regs, 9KB smem vs 32 regs, 3KB smem) cut total instructions by 2.2x and cycles by 3.5x despite crashing occupancy from ~8 to ~3 register-limited warps, because the arithmetic intensity gain far outweighs the lost latency hiding—roofline efficiency only dropped from 94.8% to 80.3%.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 938.692us
**Optimized**: 1066.384us
**Improvement**: -13.6%
**Key metric**: short_scoreboard_stalls (38.38% → 55.73%)
**Insight**: Swizzled SMEM layout was applied without actually using WGMMA/tensor cores (both at 0.01%), so the added layout complexity increased register pressure (72→96), halved occupancy limits, and dramatically worsened short-scoreboard stalls (38%→56%) — swizzling only helps when paired with the tensor-core instructions it was designed for.

## Experiment: Skill 6 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 340.28us
**Optimized**: 339.914us
**Improvement**: 0.11%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (86.36% → 86.14%)
**Insight**: The heuristic API and exhaustive tile search produced an effectively identical configuration to the default — both runs use 254 registers/thread, 68608 bytes shared memory, identical instruction counts, and no tensor cores, indicating cuBLAS's default heuristic already selected the optimal tile for this problem size.

## Experiment: Skill 7 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 2673.869us
**Optimized**: 68.991us
**Improvement**: 97.4%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (1.93% → 75.3%)
**Insight**: Split-K tiling transformed a completely underutilized kernel (1.9% efficiency, 50% warp activity) into a memory-bound kernel at 75.3% efficiency and 81.4% warp activity by distributing the large K dimension across CTAs, achieving a 38.8x speedup.
