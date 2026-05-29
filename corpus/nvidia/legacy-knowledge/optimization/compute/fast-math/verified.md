
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 11.661us
**Optimized**: 11.738us
**Improvement**: -0.66%
**Key metric**: sm__throughput (44.45% → 44.65%)
**Insight**: The kernel is memory-bound (59-60% stalled on long scoreboard / memory latency), so replacing standard math with intrinsics saves no time — the SFU was never the bottleneck; the GPU is waiting on DRAM, not compute.

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 11.705us
**Optimized**: 11.686us
**Improvement**: 0.16%
**Key metric**: sm__inst_executed.sum (3407872 → 3407872)
**Insight**: The kernel executes zero math functions affected by --use_fast_math (no sin/cos/exp/sqrt/div intrinsics), so the flag has no effect — instruction count is identical and runtime difference is within noise.

## Experiment: Skill 3 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 11.59us
**Optimized**: 11.632us
**Improvement**: -0.36%
**Key metric**: sm__inst_executed.sum (2228224 → 2228224)
**Insight**: The compiler (nvcc) already optimized 1.0f/sqrtf(x) to rsqrtf at default optimization levels, producing identical instruction counts and identical performance — the manual substitution had no effect.

## Experiment: Skill 4 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: Noneus
**Optimized**: Noneus
**Improvement**: None%
**Key metric**: profiling_timeout (timeout_300s → timeout_300s)
**Insight**: Both runs timed out during profiling (300s limit), so no performance comparison is possible; the skill cannot be verified without metrics, and the profiling timeout itself suggests the test kernel may be too large or the ncu configuration needs adjustment.
**Conditions**: both baseline and optimized profiling timed out; no measurable data

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.958us
**Optimized**: 4.038us
**Improvement**: -2.0%
**Key metric**: sm__inst_executed.sum (917504 → 1146880 (+25% more instructions))
**Insight**: The 'optimized' version emits 25% more instructions (1.15M vs 0.92M) and runs 2% slower, likely because nvcc already optimizes common powf patterns at compile time, making manual replacements with rsqrtf chains and exp2f counterproductive on this workload.

## Experiment: Skill 6 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.93us
**Optimized**: 3.93us
**Improvement**: 0.0%
**Key metric**: sm__inst_executed.sum (655360 → 655360)
**Insight**: The CUDA compiler (nvcc/ptxas) already fuses separate sinf/cosf calls on the same argument into a single sincosf internally, so explicitly writing sincosf produces identical instruction counts and runtime.

## Experiment: Skill 7 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 12.838us
**Optimized**: 11.574us
**Improvement**: 9.85%
**Key metric**: sm__inst_executed.sum (6,946,816 instructions → 2,883,584 instructions (58.5% reduction))
**Insight**: Eliminating double-promotion removed cvt.f64/f32 conversion pairs, cutting total instructions by 58.5% and active cycles by 20.6%; wall-clock improvement is a more modest 9.85% because the kernel shifts from compute-bound (60.3% SOL) to memory-latency-limited (long scoreboard stalls rise from 47.7% to 63.6%).
