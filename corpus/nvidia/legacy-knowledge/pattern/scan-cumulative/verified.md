# Scan Cumulative — Verification Data

## Experiment: Skill 7 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 17.35us
**Optimized**: 2.835us
**Improvement**: 83.7%
**Key metric**: execution_time (17.350us → 2.835us)
**Insight**: Choosing the right scan approach by problem size (warp-scan for small arrays instead of a heavy multi-block scheme) eliminated barrier stalls entirely (40% → 0%) and reduced instructions 18× (12.5M → 688K), cutting wall time from 17.35us to 2.835us — now faster than torch baseline.

## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 5.683us
**Optimized**: 4.086us
**Improvement**: 28.1%
**Key metric**: sm__inst_executed.sum (2129920 instructions, 20% barrier stalls, 2048B shmem/block → 1015808 instructions, 0% barrier stalls, 1024B shmem/block)
**Insight**: Warp shuffle scan eliminates shared memory synchronization entirely (barrier stalls 20% → 0%) and halves instruction count, yielding 28% wall-clock speedup despite the kernel becoming memory-latency-bound.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 7.613us
**Optimized**: 4.918us
**Improvement**: 35.4%
**Key metric**: sm__cycles_active.avg (13274.87 cycles, 9216B smem, 2.72M insns, 24.68% barrier stalls → 7625.75 cycles, 1152B smem, 1.59M insns, 19.07% barrier stalls)
**Insight**: Warp-scan + SMEM combine replaces Blelloch tree scan, cutting shared memory 8x (9216→1152B), instructions 42%, and barrier stalls from 24.7%→19.1%, because shuffle-based warp scans need no __syncthreads within a warp and the meta-scan of 32 warp totals is trivially cheap.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 80.556us
**Optimized**: 92.435us
**Improvement**: -14.7%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (26.4% → 84.25%)
**Insight**: Decoupled lookback's spin-wait coordination dominates on this small input (~4MB): barrier stalls jumped from 26% to 84%, active cycles exploded 13.5x (12K→162K), and uncoalesced status-array reads dropped global load efficiency from 100% to 40%, making single-pass coordination far more expensive than the two-pass baseline it replaces.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 17.824us
**Optimized**: 5.498us
**Improvement**: 69.1%
**Key metric**: sm__cycles_active.avg (32688 → 8944)
**Insight**: Converting between inclusive and exclusive scan via shift-and-combine eliminated redundant sweep passes, reducing total instructions by 5x (10.2M→1.9M) and active cycles by 3.7x, bringing the kernel to parity with torch.compile.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 13.721us
**Optimized**: 4.611us
**Improvement**: 66.4%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (12.5% → 100.0%)
**Insight**: The 3x speedup comes almost entirely from fixing memory coalescing (12.5% → 100% sector utilization), which eliminated long-scoreboard stalls (88.75% → 13.01%) and tripled warp occupancy (22.8% → 73.8%), despite executing 12x more instructions for the segmented scan logic.

## Experiment: Skill 6 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 59301.705us
**Optimized**: 76.995us
**Improvement**: 99.87%
**Key metric**: sm__warps_active.avg.pct_of_peak_sustained_active (1.56% → 78.95%)
**Insight**: The optimized scan kernel uses proper parallel prefix-sum with shared memory and coalesced global loads (100% sector utilization vs 12.5%), reducing active cycles from ~1M to ~7.6K and achieving 770x speedup by actually utilizing GPU parallelism instead of serial execution.
