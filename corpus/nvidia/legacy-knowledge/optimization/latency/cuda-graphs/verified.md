
## Experiment: Skill 4 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 125.356us
**Optimized**: 10.589us
**Improvement**: 91.6%
**Key metric**: cuda_extension_us (125.356us → 10.589us)
**Insight**: Device graph launch eliminates CPU-GPU round-trip latency for chained kernel dispatches, collapsing what was ~125us of launch overhead into a single ~10.6us graph execution — approaching torch.compile's 2.9us.

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 645.4us
**Optimized**: 497.452us
**Improvement**: 22.9%
**Key metric**: sm__cycles_active.avg (1939.24 → 1939.78)
**Insight**: GPU kernel execution is identical (same cycles, same instructions, same throughput) — the 148us saving is purely host-side launch overhead amortized by the CUDA Graph, confirming the technique only helps latency-bound, small-kernel workloads.
**Conditions**: only if the workload is launch-latency bound; the 23% wall-clock improvement comes entirely from reduced host-side overhead, not GPU kernel speedup

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 360.851us
**Optimized**: 394.506us
**Improvement**: -9.3%
**Key metric**: cuda_extension_us (360.851us → 394.506us)
**Insight**: Updating graph parameters without re-instantiation showed no benefit here because the kernel itself is unchanged (identical instruction counts, register usage, and memory traffic), and the overhead of parameter update APIs likely exceeded any saved re-instantiation cost for this trivially small workload.
