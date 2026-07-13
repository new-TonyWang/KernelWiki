# Normalization — Verification Data

## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 6.362us
**Optimized**: 5.542us
**Improvement**: 12.9%
**Key metric**: sm__inst_executed.sum (1568768 → 1082624 (31% fewer instructions))
**Insight**: One-pass approach eliminates the second read of input data, reducing total instructions by 31% and active cycles by 12%, directly translating to a 12.9% wall-clock speedup on this memory-latency-bound kernel.

## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 3.29us
**Optimized**: 3.002us
**Improvement**: 8.75%
**Key metric**: sm__inst_executed.sum (917504 insts, 3072B shmem/block → 753664 insts (-17.9%), 1024B shmem/block (-67%))
**Insight**: Warp shuffles replace shared-memory reductions, cutting instruction count by 18% and shared memory by 67%, yielding ~9% latency improvement through fewer total instructions and cycles despite the kernel remaining underutilized overall.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 12.147us
**Optimized**: 10.355us
**Improvement**: 14.8%
**Key metric**: sm__warps_active.avg.pct_of_peak_sustained_active (11.7% → 75.1%)
**Insight**: Using multiple warps per row massively improved warp occupancy (11.7% → 75.1%) and doubled SM throughput (15.8% → 33.2%) by distributing the feature-dimension reduction across warps, reducing long-scoreboard stalls from 66.6% to 43.8% at the cost of 10.8% barrier stalls from the required __syncthreads().

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 72.384us
**Optimized**: 45.763us
**Improvement**: 36.8%
**Key metric**: barrier_stalls_eliminated (compute-bound 73.3% SOL, 11.82% barrier stalls, 47M instructions → memory-bound 66.7% SOL, 0% barrier stalls, 7.8M instructions)
**Insight**: One-warp-per-row eliminates inter-warp synchronization (barrier stalls 11.8%→0%) and reduces instruction count by 6x, shifting the bottleneck from compute to memory bandwidth—the kernel becomes lean enough that DRAM throughput is now the limiting factor.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 24.704us
**Optimized**: 16.413us
**Improvement**: 33.6%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (51.64% → 65.1%)
**Insight**: Fusing normalization and activation into a single kernel reduced wall-clock time by 33.6% and shifted the bottleneck from underutilized to compute-bound by eliminating an extra pass over the output matrix, cutting DRAM writes by 12.5% and reducing long-scoreboard stalls from 45% to 32%.

## Experiment: Skill 6 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.73us
**Optimized**: 4.745us
**Improvement**: -0.3%
**Key metric**: sm__throughput (39.71% → 38.96%)
**Insight**: The NVCC compiler already lowers 1.0f/sqrtf(x) to a hardware rsqrt instruction at default optimization levels, so manually writing rsqrtf produces identical SASS — confirmed by identical instruction counts (2,323,456), identical register usage (16), and identical shared memory allocation.
