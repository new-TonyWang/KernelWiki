
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 130.624us
**Optimized**: 62.736us
**Improvement**: 52.0%
**Key metric**: dram_throughput (29.04% → 61.45%)
**Insight**: With identical instruction counts (4,282,368), __restrict__ enabled the compiler to emit ld.global.nc (read-only cache path) loads and reorder memory operations, cutting active SM cycles by 55% and shifting the bottleneck from 'underutilized' to properly 'memory'-bound at 61.5% of roofline.

## Experiment: Skill 8 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 12.95us
**Optimized**: 12.96us
**Improvement**: -0.08%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (56.89% → 55.76%)
**Insight**: The kernel already uses only 16 registers per thread and is simple enough that nvcc flags like -maxrregcount, -extra-device-vectorization, and -dlto have nothing meaningful to optimize — the compiler already produces near-optimal code for this workload.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 18.864us
**Optimized**: 18.851us
**Improvement**: 0.07%
**Key metric**: launch__registers_per_thread (22 → 22)
**Insight**: The kernel already uses only 22 registers per thread, so __launch_bounds__ cannot reduce register pressure further — the compiler was already at or below the target, making the hint a no-op.

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 38.432us
**Optimized**: 35.324us
**Improvement**: 8.1%
**Key metric**: sm__inst_executed.sum (2252800 → 655360 (71% fewer instructions))
**Insight**: Full unrolling reduced instruction count by 71% and enabled enough ILP to cut long-scoreboard stalls from 95% to 37%, improving L1 hit rate from 50% to 86% and pushing memory throughput from 83% to 90% SOL — despite halving register-limited occupancy.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 5.853us
**Optimized**: 5.827us
**Improvement**: 0.4%
**Key metric**: dram_throughput (35.74% → 35.41%)
**Insight**: __builtin_assume_aligned has no effect because nvcc already generates optimal vectorized loads (float4) when __restrict__ is present and access patterns are trivially aligned — the baseline already shows 100% sector utilization (smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct=100%) and identical instruction counts.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 45.785us
**Optimized**: 16.285us
**Improvement**: 64.4%
**Key metric**: sm__inst_executed.sum (659712 → 431104)
**Insight**: Signed loop counters enabled strength reduction and more aggressive compiler optimization, reducing total executed instructions by 34.6% and active cycles by 66%, though at the cost of higher register pressure (23→32 registers per thread).

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 119.577us
**Optimized**: 119.766us
**Improvement**: -0.16%
**Key metric**: dram_throughput (18.58% → 18.73%)
**Insight**: __builtin_assume and __builtin_expect produce no measurable improvement because nvcc already optimizes simple branch patterns effectively, and this kernel is memory-bound (85%+ stalled on long scoreboard) so instruction-level hints have negligible impact.

## Experiment: Skill 7 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.55us
**Optimized**: 4.563us
**Improvement**: -0.3%
**Key metric**: sm__inst_executed.sum (2064384 → 2064384)
**Insight**: For a small 12-byte parameter struct (int + 2 floats), the compiler already handles parameter passing efficiently—identical instruction count, identical register usage (16), and no measurable latency change, so __grid_constant__ provides no benefit.
