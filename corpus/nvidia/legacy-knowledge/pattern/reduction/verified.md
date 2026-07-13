# Reduction — Verification Data

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 6.973us
**Optimized**: 6.15us
**Improvement**: 11.8%
**Key metric**: sm__warps_active.avg.pct_of_peak_sustained_active (6.58% → 4.98%)
**Insight**: Sequential addressing packs active threads contiguously within warps, reducing warp-level divergence — confirmed by warps_active dropping from 6.58% to 4.98% while instruction count stays identical at 124, meaning the same work completes with fewer warps needing to be scheduled.

## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 3.322us
**Optimized**: 2.819us
**Improvement**: 15.1%
**Key metric**: sm__inst_executed.sum (1138688 instructions, 8.17% barrier stalls, 2048B shared mem → 590848 instructions (48% fewer), 3.69% barrier stalls, 1152B shared mem)
**Insight**: Warp shuffle replaces shared-memory reduction, cutting instructions nearly in half and halving barrier stalls by eliminating __syncthreads calls, yielding a 15% wall-clock speedup.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 10.215us
**Optimized**: 9.687us
**Improvement**: 5.17%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (0.62% → 0.65%)
**Insight**: The reduction problem is too small (4KB read, single block) to meaningfully exercise the shared-memory reduction pattern — both versions are massively underutilized (<1% SOL) and dominated by kernel launch overhead, making any structural optimization invisible.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 19.795us
**Optimized**: 19.008us
**Improvement**: 3.98%
**Key metric**: dram_throughput (0.04% → 0.04%)
**Insight**: The input is too small (3840-4096 bytes DRAM read) to be bandwidth-bound at all — both kernels are massively underutilized (0.6% memory SOL, 99.4% headroom), so vectorized loads provide no measurable benefit since launch overhead dominates execution time.

## Experiment: Skill 6 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 533.038us
**Optimized**: 19.072us
**Improvement**: 96.4%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (12.5% → 100.0%)
**Insight**: The baseline performed column reduction with strided (uncoalesced) global loads at 12.5% sector utilization, relying on L1 cache (87.5% hit rate) to partially compensate; the optimized version achieves fully coalesced access (100% sector utilization) likely via transpose-then-reduce or reorganized indexing, yielding 28x speedup and 67.9% DRAM throughput.
