
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 1842.48us
**Optimized**: 10.557us
**Improvement**: 99.43%
**Key metric**: cuda_extension_us (1842.48us → 10.557us)
**Insight**: The baseline was catastrophically slow (174x slower than even the torch baseline), likely using a naive single-thread or fully-serialized atomic reduction; the optimized hierarchical warp->block->grid reduction eliminates contention and brings latency to within ~1.7x of the torch baseline, which is reasonable for a small reduction problem.

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 53.405us
**Optimized**: 53.245us
**Improvement**: 0.3%
**Key metric**: dram_throughput (24.37% → 24.35%)
**Insight**: The kernel is dominated by global memory latency (81% long scoreboard stalls), not atomic contention, so narrowing atomic scope has negligible effect when atomics are not the bottleneck.

## Experiment: Skill 3 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4320.653us
**Optimized**: 4320.936us
**Improvement**: -0.007%
**Key metric**: sm__inst_executed.sum (12058624 → 8912896 (26% fewer instructions, but zero wall-clock gain))
**Insight**: Relaxed memory ordering reduced instruction count by 26% and long scoreboard stalls from 58% to 26%, confirming the mechanism works at the instruction level, but the kernel is so severely underutilized (2.5% efficiency, 17x slower than PyTorch) that atomic ordering is not the bottleneck and the optimization has no wall-clock effect.

## Experiment: Skill 4 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 4320.897us
**Optimized**: 730.947us
**Improvement**: 83.1%
**Key metric**: cuda_extension_us (4320.897us (sm_cycles_active=9.65) → 730.947us (sm_cycles_active=8.41))
**Insight**: Shared memory atomics eliminated global atomic contention, yielding a 5.9x wall-clock speedup, though NCU shows only ~13% cycle reduction — the large wall-clock gap suggests the baseline suffered severe global atomic serialization that inflated end-to-end time far beyond raw kernel cycles.
