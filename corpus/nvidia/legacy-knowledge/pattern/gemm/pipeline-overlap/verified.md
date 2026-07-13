
## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2106.944us
**Optimized**: 2413.857us
**Improvement**: -14.6%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (93.69% → 87.06%)
**Insight**: The baseline was already at 93.7% memory SOL with minimal headroom; cp.async eliminated long-scoreboard stalls (15.6%→0.9%) but added 25% more instructions and doubled barrier stalls (14.2%→22.1%), netting a 14.6% regression because synchronization overhead exceeded the latency-hiding benefit.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 690.785us
**Optimized**: 2581.497us
**Improvement**: -273.6%
**Key metric**: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct (2.23% → 36.78%)
**Insight**: Double-buffering SMEM-to-RMEM fragments increased register usage from 128 to 139 per thread, dropping register-limited occupancy from 8 to 6 warps, causing massive register spills (DRAM writes 2.97MB→15.77MB) and 36.78% long-scoreboard stalls that completely negate any latency-hiding benefit.

## Experiment: Skill 6 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 1020.672us
**Optimized**: 1000.329us
**Improvement**: 2.0%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (14.53% → 9.7%)
**Insight**: Pipeline state management successfully reduced barrier stalls (14.5%→9.7%) and long scoreboard stalls (12.9%→8.5%), proving better compute/memory overlap, but the gains were offset by a sharp rise in short scoreboard stalls (21.2%→38.6%) from increased shared memory traffic (5KB→9KB per block), and the kernel was already at 93% memory SOL leaving little room for wall-clock improvement.
**Conditions**: only if the kernel is not already near memory roofline; here at 93% memory SOL, pipeline overlap yields marginal wall-clock gains despite clear stall reduction

## Experiment: Skill 7 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2392.957us
**Optimized**: 3220.292us
**Improvement**: -34.6%
**Key metric**: cuda_extension_us (2392.957us / 87.9% efficiency → 3220.292us / 86.2% efficiency)
**Insight**: Neither baseline nor optimized kernel uses tensor cores (0.01% tensor pipe activity), so the TMA-WGMMA proxy fence is pure overhead here — adding ~10% more instructions and 34% more SM cycles with no correctness benefit, while increasing long-scoreboard stalls from 2.4% to 5.4%.
