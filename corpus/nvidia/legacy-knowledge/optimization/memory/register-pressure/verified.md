
## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.983us
**Optimized**: 5.005us
**Improvement**: -0.44%
**Key metric**: launch__registers_per_thread (19 → 19)
**Insight**: The kernel already uses only 19 registers per thread (well below any pressure threshold), so __launch_bounds__ had no register usage to reduce and the compiler generated identical code.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 5.917us
**Optimized**: 6.019us
**Improvement**: -1.7%
**Key metric**: launch__registers_per_thread (25 regs, occupancy_limit_registers=8 → 23 regs, occupancy_limit_registers=10)
**Insight**: __maxnreg__ successfully reduced registers (25→23) and improved register-limited occupancy (8→10 blocks), but the kernel was already underutilized and not register-pressure-bound, so higher occupancy yielded no speedup — execution time was unchanged or marginally worse.

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 77.718us
**Optimized**: 91.59us
**Improvement**: -17.8%
**Key metric**: launch__registers_per_thread (40 regs, 1.15M insts, 15.5% long_scoreboard_stall → 32 regs, 1.53M insts, 55.5% long_scoreboard_stall)
**Insight**: Reducing registers from 40 to 32 via -maxrregcount forced register spilling to local memory, causing 33% more instructions, 2.67x more DRAM writes, L1 hit rate drop from 91% to 76%, and long scoreboard stalls jumping from 15% to 55%, far outweighing the marginal occupancy gain from 6 to 8 register-limited blocks.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 65.782us
**Optimized**: 44.64us
**Improvement**: 32.1%
**Key metric**: long_scoreboard_stalls (66.45% → 7.43%)
**Insight**: Diagnosing register spilling guided the optimization to move spilled data from local memory into shared memory (1KB→17.9KB), eliminating long-scoreboard stalls (66%→7%) caused by high-latency local memory accesses and yielding a 32% speedup despite unchanged register count (32/thread).

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 3.603us
**Optimized**: 3.062us
**Improvement**: 15.0%
**Key metric**: launch__registers_per_thread (56 regs (occupancy_limit_registers=4) → 32 regs (occupancy_limit_registers=8))
**Insight**: Cutting registers from 56 to 32 doubled the register-limited occupancy (4→8 blocks), letting the SM hide memory latency with more concurrent warps and reducing active cycles by 9%.
