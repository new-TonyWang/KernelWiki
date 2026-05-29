
## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 7.949us
**Optimized**: 6.55us
**Improvement**: 17.6%
**Key metric**: sm__cycles_active.avg (14264 cycles → 11973 cycles)
**Insight**: Loop unrolling reduced total executed instructions by 17% (2.31M→1.91M) and active cycles by 16%, directly translating to wall-clock speedup, though register usage doubled (16→32) which halved the occupancy limit from registers (16→8 warps) without fully closing the memory-latency gap.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 57.286us
**Optimized**: 105.317us
**Improvement**: -83.8%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (100.0% → 12.5%)
**Insight**: Assigning ITEMS_PER_THREAD contiguous elements per thread (base = tid * N) causes adjacent threads to access addresses N elements apart, destroying warp-level coalescing (sector utilization crashed from 100% to 12.5%), which overwhelms the ILP gains from reduced long-scoreboard stalls (73.65% → 18.53%) and halved instruction count.
**Conditions**: The multi-element-per-thread pattern destroyed memory coalescing, making the kernel ~84% slower despite successfully improving ILP

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 51.392us
**Optimized**: 50.829us
**Improvement**: 1.1%
**Key metric**: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct (89.05% → 85.49%)
**Insight**: The kernel is memory-bandwidth-bound (46% DRAM throughput), not latency-bound, so software pipelining's latency hiding adds instruction overhead (+12.9% instructions) and register pressure (26→30 regs) that offset the modest reduction in long scoreboard stalls (89→85%).
