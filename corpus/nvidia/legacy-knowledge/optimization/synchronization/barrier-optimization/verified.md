
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.288us
**Optimized**: 5.078us
**Improvement**: -18.4%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (4.07% → 7.1%)
**Insight**: The async barrier added 30% more instructions (1.47M→1.92M) and nearly doubled barrier stalls (4%→7%), because this kernel is too small/simple to have meaningful independent work between arrive() and wait(), making the cuda::barrier overhead pure cost over __syncthreads().

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 5.238us
**Optimized**: 4.915us
**Improvement**: 6.2%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (11.25% → 0.0%)
**Insight**: Replacing __syncthreads with warp-level sync completely eliminated barrier stalls (11.25% → 0.0%) and reduced active SM cycles by 12.5%, translating to a 6.2% wall-clock speedup despite memory-latency stalls becoming proportionally more visible.

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 12.672us
**Optimized**: 12.643us
**Improvement**: 0.23%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (0.0% → 0.0%)
**Insight**: The test kernel has zero barrier stall pressure (0% in both runs) and only 124 instructions with trivial data (4KB), so there is no spin-wait contention for __nanosleep to alleviate — the 0.23% time difference is measurement noise.

## Experiment: Skill 6 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 6.458us
**Optimized**: 5.302us
**Improvement**: 17.9%
**Key metric**: smsp__warp_issue_stalled_barrier_per_warp_active.pct (33.33% → 0.0%)
**Insight**: Replacing __syncthreads with warp-level sync eliminated all barrier stalls (33.33% → 0.0%), reducing active SM cycles by 18% and freeing warps to issue memory/compute instructions instead of waiting at block-wide barriers.
