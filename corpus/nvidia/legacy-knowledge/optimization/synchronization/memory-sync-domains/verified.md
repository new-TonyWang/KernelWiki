
## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 573.516us
**Optimized**: 571.991us
**Improvement**: 0.27%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (45.01% → 44.84%)
**Insight**: Memory sync domain mapping is a fence-scoping mechanism that only yields measurable benefit when multiple independent streams contend on the same memory fence hardware — a single-kernel benchmark with no cross-stream fence contention sees no improvement.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 59.718us
**Optimized**: 66.214us
**Improvement**: -10.9%
**Key metric**: cuda_extension_us (59.718us, 15.7M instructions, 100% global ld efficiency → 66.214us, 24.6M instructions, 0% global ld efficiency)
**Insight**: fence.proxy.async is a correctness primitive for async/generic proxy ordering, not a performance optimization; the restructured kernel using TMA+shared memory+fence executed 57% more instructions and ran 10.9% slower, likely because the problem size doesn't benefit from the async copy pattern overhead.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 154.666us
**Optimized**: 155.062us
**Improvement**: -0.26%
**Key metric**: sm__cycles_active.avg (44284 → 43908)
**Insight**: This skill describes automatic NCCL behavior (remote domain tagging) that reduces fence interference on concurrent compute kernels — it cannot be verified with an isolated single-kernel benchmark that has no concurrent NCCL traffic to interfere with.
**Conditions**: only if concurrent NCCL communication is present alongside compute kernels on Hopper GPUs
