# Pooling — Verification Data

## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 13.481us
**Optimized**: 14.95us
**Improvement**: -10.9%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (61.95% → 53.53%)
**Insight**: The optimized kernel is 10.9% slower with degraded SM throughput (62% → 53.5%), significantly worse long-scoreboard stalls (42% → 57%), and 2.3x more DRAM writes, indicating the optimization introduced memory-latency bottlenecks that outweigh the modest instruction count reduction.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 112.786us
**Optimized**: 102.454us
**Improvement**: 9.16%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (83.44% → 88.24%)
**Insight**: SMEM tiling traded L1 cache hits (87%→6%) for explicit shared memory reuse, cutting global memory stalls (long scoreboard 20%→12%) and reducing total instructions by 6%, netting a 9% speedup despite introducing 14% barrier synchronization overhead from __syncthreads().

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 15.616us
**Optimized**: 11.37us
**Improvement**: 27.2%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (25.0% → 100.0%)
**Insight**: float4 vectorized loads achieve perfect 100% sector utilization (up from 25% scalar), cutting instruction count by 35% and reducing active cycles by 29%, which translates to a 27% wall-clock speedup.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 269.929us
**Optimized**: 62.061us
**Improvement**: 77.0%
**Key metric**: dram_throughput (15.34% → 87.21%)
**Insight**: Warp-level cooperative reduction with __shfl_down_sync eliminated the massive L1 cache thrashing (87.5% → 0.6% hit rate) by having each thread load contiguous elements directly from DRAM, achieving 100% memory coalescing (12.5% → 100% bytes/sector) and 5.7x higher active cycles utilization (24% → 87% warp occupancy), turning a latency-bound scattered-access pattern into a throughput-saturated streaming reduction.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 645.918us
**Optimized**: 298.351us
**Improvement**: 53.8%
**Key metric**: sm__inst_executed.sum (593,136,192 → 119,939,072 (4.9x fewer))
**Insight**: Decomposing 2D pooling into two 1D passes reduced total instructions by ~5x (from 593M to 120M), directly matching the theoretical O(h*w) → O(h+w) complexity reduction and cutting active SM cycles by 4.7x.

## Experiment: Skill 7 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 202.825us
**Optimized**: 164.598us
**Improvement**: 18.8%
**Key metric**: sm__inst_executed.sum (175747072 → 139706368 (20.5% fewer instructions))
**Insight**: Proper padding handling with -FLT_MAX init and fmaxf eliminates conditional branches at boundaries, reducing total executed instructions by 20.5% and active cycles by 17.4%, yielding an 18.8% wall-clock speedup despite using 4 more registers per thread.

## Experiment: Skill 8 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 410.029us
**Optimized**: 7.638us
**Improvement**: 98.1%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (12.5% → 100.0%)
**Insight**: The 54x speedup came from fixing catastrophic uncoalesced global loads (12.5% → 100% sector utilization), halving register pressure (31 → 16) to double occupancy (12% → 90% active warps), and proper hierarchical reduction that cut DRAM writes by 27x (564KB → 21KB), eliminating excessive intermediate/atomic traffic.
