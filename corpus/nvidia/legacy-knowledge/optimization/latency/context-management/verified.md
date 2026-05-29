
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 9.078us
**Optimized**: 3.907us
**Improvement**: 57.0%
**Key metric**: cuda_extension_us (9.078us → 3.907us)
**Insight**: The 57% latency reduction comes entirely from eliminating host-side context creation/switching overhead — NCU kernel metrics (cycles, instructions, throughput) are identical between runs, confirming the gain is in launch-path efficiency, not kernel execution.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.006us
**Optimized**: 4.0us
**Improvement**: 0.15%
**Key metric**: cuda_extension_us (4.006us → 4.000us)
**Insight**: cudaInitDevice controls *when* initialization overhead occurs but does not reduce steady-state kernel execution time; the profiling harness already warm-starts the context before timing, so the lazy-init penalty is never captured in either run.
