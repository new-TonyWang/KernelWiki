
## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 418.402us
**Optimized**: 56.083us
**Improvement**: 86.6%
**Key metric**: cuda_extension_us (418.402us (naive kernel, memory-bound 93.4% SOL) → 56.083us (cuBLAS, compute-bound 69.9% SOL))
**Insight**: Switching from a naive custom GEMM kernel to cuBLAS achieved 7.5x speedup by leveraging highly optimized tiling (244 regs/thread, 19KB shared mem), reducing total instructions from 155M to 41.5M and matching PyTorch's own cuBLAS-backed performance (56.0us).

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 352.735us
**Optimized**: 341.881us
**Improvement**: 3.1%
**Key metric**: dram__bytes_write.sum (3144448 bytes → 2622208 bytes (−16.6%))
**Insight**: Epilogue fusion successfully reduces DRAM write traffic by ~16.6% (bias add folded into matmul output pass), but the kernel is compute-bound at 84.9% SM throughput so the memory savings do not translate into meaningful latency reduction (only ~3%, within noise).
**Conditions**: only if the GEMM is memory-bound or the epilogue operation (bias add) contributes meaningfully to total runtime; on compute-bound GEMMs the fused epilogue saves writes but has negligible end-to-end impact

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 100.863us
**Optimized**: 98.122us
**Improvement**: 2.72%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (85.5% → 85.31%)
**Insight**: The 2.7% speedup is within measurement noise — kernel characteristics are identical (same 244 registers, same 83.8M instructions, same occupancy, 0% tensor core usage in both), indicating the epilogue fusion skill was not actually applied or had no measurable effect on this workload.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 340.61us
**Optimized**: 340.354us
**Improvement**: 0.08%
**Key metric**: sm__throughput (84.46% → 84.1%)
**Insight**: The algorithm selection/tuning skill produced no measurable change — both rounds selected the identical cuBLAS algorithm (same register count, shared memory, tile config, instruction count), indicating the heuristic already returned the optimal algorithm for this problem size.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 798.492us
**Optimized**: 830.279us
**Improvement**: -3.98%
**Key metric**: determinism_achieved_with_acceptable_overhead (tensor_cores_0.13%_active, long_scoreboard_stall_29.4% → tensor_cores_0.01%_active, long_scoreboard_stall_52.4%)
**Insight**: Disabling atomics via cublasSetAtomicsMode forces cuBLAS into non-atomic reduction paths, which eliminates tensor core usage (0.13%→0.01%) and nearly doubles memory dependency stalls (29%→52%), confirming the documented trade-off that some faster algorithms become unavailable.
**Conditions**: only if the goal is bit-exact reproducibility rather than performance; the skill intentionally trades speed for determinism, so the ~4% slowdown is expected behavior, not a defect

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 8.554us
**Optimized**: 7.357us
**Improvement**: 14.0%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (0.04% → 43.5%)
**Insight**: Applying cuBLAS performance tips (column-major layout, aligned leading dimensions) eliminated internal transpose overhead and brought the extension to parity with torch baseline (~7.4us), while dramatically improving GPU utilization from near-zero to 43.5% SM throughput.
