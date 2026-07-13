
## Experiment: Skill 1 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 73.203us
**Optimized**: 73.379us
**Improvement**: -0.24%
**Key metric**: sm__inst_executed.sum (12582912 → 12058624 (4.2% fewer instructions))
**Insight**: FMA successfully reduced instruction count by ~4.2% (fusing mul+add into one op), but the kernel is memory-bound (72.7% DRAM SOL vs 51.8% compute SOL), so the compute savings are completely hidden by memory latency.
**Conditions**: only if the kernel is compute-bound; for memory-bound kernels the reduced instruction count does not translate to wall-clock improvement

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 84.125us
**Optimized**: 80.665us
**Improvement**: 4.1%
**Key metric**: sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed (4.09% → 4.06%)
**Insight**: The kernel is memory-bound (93% memory SOL) with negligible tensor core utilization (~4%), so fusing an epilogue into WMMA fragments saves no meaningful round-trip — the bottleneck is memory latency (93% long scoreboard stalls), not register-to-shared-memory traffic, and both versions remain 20-25% slower than PyTorch.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 117.804us
**Optimized**: 62.383us
**Improvement**: 47.0%
**Key metric**: cuda_extension_us (117.804us → 62.383us)
**Insight**: The 47% wall-clock speedup comes primarily from eliminating kernel launch overhead and inter-kernel synchronization rather than reducing DRAM traffic (bytes read/written are nearly identical), indicating fusion's main benefit at this problem size is launch-overhead amortization.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 18.035us
**Optimized**: 9.709us
**Improvement**: 46.2%
**Key metric**: sm__inst_executed.sum (8323072 → 3997696 (52% reduction))
**Insight**: lop3.b32 fuses multiple Boolean operations into a single instruction, cutting total executed instructions by 52% and nearly halving kernel runtime, confirming that multi-operand logical fusion is highly effective for compound bitwise operations.
