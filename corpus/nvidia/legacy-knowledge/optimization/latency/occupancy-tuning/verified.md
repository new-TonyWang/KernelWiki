
## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.939us
**Optimized**: 3.949us
**Improvement**: -0.25%
**Key metric**: launch__registers_per_thread (18 → 18)
**Insight**: The kernel already uses only 18 registers per thread, so __launch_bounds__ cannot reduce register pressure further — occupancy was already limited by block count (32), not registers (limit 10 blocks from registers vs 32 from blocks/shmem), meaning the compiler had already chosen near-optimal register allocation without the hint.

## Experiment: Skill 3 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 7.779us
**Optimized**: 6.387us
**Improvement**: 17.9%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (86.21% → 100.0%)
**Insight**: Warp-aligned block size eliminated under-populated warps, achieving perfect memory coalescing (86%→100%) and reducing total instructions by 22% (713K→557K), which drove an 18% latency reduction.

## Experiment: Skill 4 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 102.082us
**Optimized**: 104.585us
**Improvement**: -2.45%
**Key metric**: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct (14.29% → 18.86%)
**Insight**: The kernel is memory-bound (80.6% memory SOL, 10.6% compute SOL), so using 2 extra registers per thread (26→28) did not reduce spills (instruction count unchanged at 8,847,360) but worsened memory latency hiding, increasing long scoreboard stalls from 14.3% to 18.9% and adding ~3.8% more active cycles.
**Conditions**: kernel is memory-bound (80%+ memory SOL); skill requires compute-bound kernel with high ILP

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 5.674us
**Optimized**: 5.658us
**Improvement**: 0.28%
**Key metric**: launch__shared_mem_per_block_allocated (5120 bytes → 5120 bytes)
**Insight**: The kernel is not occupancy-sensitive — shared memory allocation, occupancy limits, and all performance metrics are identical between rounds, with the 0.28% latency difference being pure measurement noise.

## Experiment: Skill 6 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 317.249us
**Optimized**: 69.059us
**Improvement**: 78.2%
**Key metric**: sm__warps_active.avg.pct_of_peak_sustained_active (9.56% → 69.39%)
**Insight**: Tuning block size to 1024 and achieving 100% occupancy reduced SM cycle count by 5x (479k→96k), converting a severely underutilized kernel into one that saturates warp schedulers and improves DRAM throughput from 7.3% to 34.9% of peak.

## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 317.227us
**Optimized**: 69.122us
**Improvement**: 78.2%
**Key metric**: sm__warps_active.avg.pct_of_peak_sustained_active (10.22% → 69.38%)
**Insight**: The baseline used a poorly sized block configuration (only 10% warp occupancy), and cudaOccupancyMaxPotentialBlockSize selected a block size that raised active warps from 10% to 69%, cutting SM cycles by ~5x while executing the same instruction count.
