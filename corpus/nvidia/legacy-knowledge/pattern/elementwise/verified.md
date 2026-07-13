# Elementwise — Verification Data

## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 56.874us
**Optimized**: 33.574us
**Improvement**: 41.0%
**Key metric**: dram_throughput (43.04% → 71.25%)
**Insight**: float4 vectorized loads reduced total instructions by 70% (8.9M→2.6M) and shifted the kernel from underutilized to properly memory-bound, matching torch baseline performance.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 15.74us
**Optimized**: 3.923us
**Improvement**: 75.1%
**Key metric**: wall_clock_time (15.74us (4 separate kernels implied) → 3.923us (fused kernel))
**Insight**: Fusing elementwise ops into a single kernel eliminated kernel launch overhead and inter-kernel synchronization, yielding a 4x speedup despite nearly identical per-kernel hardware utilization (DRAM throughput ~14.5%, SM throughput ~29.5%).

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 139.308us
**Optimized**: 136.32us
**Improvement**: 2.1%
**Key metric**: gpu__compute_memory_throughput (88.13% → 84.81%)
**Insight**: The EVT-based fusion kernel is ~3x slower than PyTorch baseline (136us vs 48us), and the optimization attempt actually degraded throughput (88.1% → 84.8%) while increasing instruction count by 22%, indicating the CUTLASS EVT overhead dominates for this problem size.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 56.739us
**Optimized**: 113.289us
**Improvement**: -99.6%
**Key metric**: dram_throughput (42.68% → 20.25%)
**Insight**: The fixed grid size was too small, dropping warp occupancy from 71% to 24% and doubling cycle count — too few threads to hide memory latency, as shown by long scoreboard stalls rising from 72% to 93%.

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 61.167us
**Optimized**: 11.606us
**Improvement**: 81.0%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (12.5% → 100.0%)
**Insight**: Coalescing global memory loads (12.5% → 100% sector utilization) eliminated ~8x redundant memory transactions, cutting active SM cycles from 93,587 to 20,331 despite identical instruction counts.

## Experiment: Skill 7 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 64.691us
**Optimized**: 769.113us
**Improvement**: -1089.0%
**Key metric**: cuda_extension_us (64.691us (memory-bound 61.1%) → 769.113us (compute-bound 85.4%))
**Insight**: cuBLAS library overhead (dispatch, workspace setup, internal synchronization) dominates for modestly-sized elementwise ops — 14x more instructions executed and 4x more SM cycles, turning a memory-bound kernel into a compute-bound one that is 11.9x slower.
