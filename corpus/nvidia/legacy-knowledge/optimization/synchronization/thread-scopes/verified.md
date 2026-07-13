
## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 73.696us
**Optimized**: 34.313us
**Improvement**: 53.4%
**Key metric**: sm__cycles_active.avg (139018 → 64515)
**Insight**: Replacing __threadfence with __threadfence_block eliminated expensive global memory ordering, cutting instructions by 20% and active cycles by 54%, since intra-block communication only needs block-scoped visibility guarantees.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 11.059us
**Optimized**: 14.089us
**Improvement**: -27.4%
**Key metric**: sm__inst_executed.sum (3440640 → 124)
**Insight**: The optimized kernel is effectively a no-op (124 instructions vs 3.4M, 4KB read vs 4MB) yet takes longer wall-clock time (14us vs 11us), indicating the device-scope fence restructuring eliminated useful work from the profiled kernel rather than improving inter-block communication efficiency.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 10.499us
**Optimized**: 6.214us
**Improvement**: 40.8%
**Key metric**: sm__cycles_active.avg (16865.66 → 9775.67)
**Insight**: Narrowing thread scope from system to device/block where cross-device visibility isn't needed eliminates expensive system-level fence overhead, cutting active SM cycles by 42% despite executing the same instruction count.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 15.488us
**Optimized**: 15.93us
**Improvement**: -2.85%
**Key metric**: sm__cycles_active.avg (10.64 → 8.49)
**Insight**: The kernel is too small and underutilized (0.6% SOL) for scoped atomics to produce a measurable wall-clock improvement; the slight cycle reduction from relaxed ordering is swamped by launch overhead and measurement noise.
