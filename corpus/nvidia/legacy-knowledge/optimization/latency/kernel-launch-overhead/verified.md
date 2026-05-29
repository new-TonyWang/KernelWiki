
## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.894us
**Optimized**: 3.901us
**Improvement**: -0.18%
**Key metric**: cuda_extension_us (3.894us → 3.901us)
**Insight**: cudaEventDisableTiming saves only nanoseconds of host-side overhead per event operation, which is unmeasurable when the kernel itself runs at microsecond scale and profiling measures kernel execution time rather than host launch latency.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.894us
**Optimized**: 3.904us
**Improvement**: -0.26%
**Key metric**: sm__cycles_active.avg (5912.69 cycles → 5619.05 cycles)
**Insight**: cudaLaunchKernelExC is an API consolidation mechanism for specifying extended launch attributes (cluster dims, L2 policy, sync domains) in a single call — it provides no performance benefit for simple kernels that don't use those attributes, as the kernel execution itself is identical.
