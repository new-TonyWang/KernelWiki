
## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 16.294us
**Optimized**: 16.368us
**Improvement**: -0.45%
**Key metric**: sm__cycles_active.avg (6615.89 → 6555.36)
**Insight**: The workload is too small and compute-underutilized (24.6% SM throughput, 0.16% DRAM throughput) for cross-stream event dependencies to produce any measurable latency benefit — kernel launch overhead dominates over any overlap gained.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 62.592us
**Optimized**: 67.673us
**Improvement**: -8.1%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (82.36% → 82.3%)
**Insight**: The kernel is already compute-bound at 82% SM throughput with identical instruction counts and cycle counts, meaning there is no default-stream serialization to eliminate — the workload is a single kernel fully saturating the GPU, so non-blocking streams provide no concurrency benefit.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 411.571us
**Optimized**: 398.4us
**Improvement**: 3.2%
**Key metric**: sm__cycles_active.avg (6114.56 → 6304.78)
**Insight**: Stream-ordered allocation reduces host-side cudaMalloc/cudaFree synchronization overhead, not kernel execution time — the kernel profile is identical (same instructions, occupancy, throughput) and the 3.2% wall-time delta is within measurement noise while sm_cycles actually increased.
**Conditions**: only if allocation overhead is a significant fraction of total execution time (not the case here with ~400us kernels)
