
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 11.337us
**Optimized**: 6.448us
**Improvement**: 43.1%
**Key metric**: sm__inst_executed.sum (2490368 → 1245184 (exactly halved))
**Insight**: Half2 packing exactly halved the instruction count (2,490,368 → 1,245,184) and active SM cycles (18,132 → 9,814), confirming that each __hadd2 processes two fp16 values in a single SIMD instruction, translating to a 43% wall-clock speedup.

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 11.373us
**Optimized**: 6.432us
**Improvement**: 43.4%
**Key metric**: sm__inst_executed.sum (2621440 → 1245184 (52.5% fewer instructions))
**Insight**: Packed __nv_bfloat162 operations halve instruction count by processing two bf16 elements per SIMD instruction, cutting active SM cycles from ~18k to ~10k and yielding a 43% wall-clock speedup.

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 16.669us
**Optimized**: 9.718us
**Improvement**: 41.7%
**Key metric**: sm__inst_executed.sum (6782976 insts / 8.4MB read → 124 insts / 4KB read)
**Insight**: The wall-clock speedup is an artifact — NCU metrics show the optimized kernel executes ~55000x fewer instructions and reads ~2000x less data, indicating the compiler optimized away the actual work or the kernel structure changed so profiling captures only a trivial stub, not a genuine mixed-precision accumulation improvement.
**Conditions**: optimized kernel appears to have eliminated actual computation — NCU shows 124 instructions vs 6.8M baseline, 4KB DRAM read vs 8.4MB baseline

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2.637us
**Optimized**: 2.691us
**Improvement**: -2.0%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (22.05% → 20.37%)
**Insight**: At this small problem size (~2MB), the kernel is launch-latency dominated and underutilized (both variants ~20% SM throughput), so swapping fp32 transcendentals for half-precision intrinsics cannot overcome the fixed overhead — the optimized version actually executed 35% more instructions (507904 vs 376832) likely due to h2exp emulation or extra packing logic, negating any ALU savings.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 2.835us
**Optimized**: 2.803us
**Improvement**: 1.1%
**Key metric**: sm__inst_executed.sum (393216 → 376832 (4.2% fewer instructions))
**Insight**: The kernel is memory-bound (both variants bottleneck on DRAM at ~24.5% throughput with 75% headroom), so fusing FMA+ReLU into one instruction saves a few ALU ops but cannot meaningfully reduce wall-clock time since execution is dominated by memory latency stalls (53% long scoreboard).

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 6.585us
**Optimized**: 155.199us
**Improvement**: -2256.4%
**Key metric**: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct (34.3% → 88.41%)
**Insight**: Naive fp16 atomicAdd to a single output address serializes all threads, causing 88% long-scoreboard stalls and a 23.6x slowdown — the instruction exists but a proper parallel reduction (shared-mem tree + single final atomic) is required to actually benefit from it.
