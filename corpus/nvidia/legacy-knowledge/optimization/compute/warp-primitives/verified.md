
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 10.973us
**Optimized**: 10.595us
**Improvement**: 3.4%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (0.61% → 0.65%)
**Insight**: The problem size is too small (4KB read, single warp) to meaningfully demonstrate warp-level shuffle benefits — kernel launch overhead dominates, both versions are massively underutilized (<1% SOL), and the 3.4% difference is within noise.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 4.954us
**Optimized**: 4.09us
**Improvement**: 17.4%
**Key metric**: sm__inst_executed.sum (2260992 → 1015808 (55% fewer instructions))
**Insight**: Warp-level shuffle scan eliminated 55% of instructions by replacing shared-memory-based reduction with register-level __shfl_up_sync, cutting active cycles by 23% and wall-clock time by 17.4%.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 5.011us
**Optimized**: 3.994us
**Improvement**: 20.3%
**Key metric**: sm__cycles_active.avg (8466.55 → 6050.77)
**Insight**: Warp vote intrinsics enabled early exit and reduced total instructions by 53% (1.80M→0.85M), cutting active cycles by 28.5% and wall-clock time by 20%, with halved register usage (32→16) doubling occupancy potential.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 30.054us
**Optimized**: 29.916us
**Improvement**: 0.46%
**Key metric**: sm__inst_executed.sum (7602176 → 6815744 (-10.3% instructions), but sm__throughput dropped from 46.19% to 38.15%)
**Insight**: Warp broadcast reduced instruction count by 10% and shared memory usage (1152→1024 bytes), but the kernel is memory-latency-bound (long scoreboard stalls rose from 63% to 68%), so eliminating redundant compute via __shfl_sync yields no meaningful wall-clock improvement.

## Experiment: Skill 6 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 195.842us
**Optimized**: 49.433us
**Improvement**: 74.8%
**Key metric**: cuda_extension_us (195.842us → 49.433us)
**Insight**: The warp match optimization (__match_any_sync + leader election) reduced redundant atomics dramatically, bringing the CUDA extension from ~4x slower than torch baseline to roughly on par with it (~49us vs ~48us).

## Experiment: Skill 4 verification (2026-04-07)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 4.074us
**Optimized**: 3.981us
**Improvement**: 2.28%
**Key metric**: sm__inst_executed.sum (950272 → 688128 (27.6% fewer instructions))
**Insight**: The hardware redux.sync instruction eliminates the shuffle tree as expected (27.6% instruction reduction), but wall-clock improvement is only 2.3% because execution is dominated by memory-latency stalls (long_scoreboard ~47%), not reduction arithmetic.
**Conditions**: only if the kernel is compute-bound or the warp reduction constitutes a significant fraction of total execution time; in memory-latency-bound kernels the instruction savings do not translate to wall-clock gains
