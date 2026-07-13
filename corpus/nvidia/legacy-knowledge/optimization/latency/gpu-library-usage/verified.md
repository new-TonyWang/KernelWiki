
## Experiment: Skill 2 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 16.016us
**Optimized**: 6.339us
**Improvement**: 60.4%
**Key metric**: cuda_extension_us (16.016us → 6.339us)
**Insight**: Switching to batched GEMM (likely cublasGemmStridedBatched) reduced execution time by 60% and brought the CUDA extension on par with torch baseline (6.32us), by replacing a naive per-matrix loop with a single batched library call that amortizes launch overhead and enables internal parallelism across small matrices.

## Experiment: Skill 3 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 120.348us
**Optimized**: 122.492us
**Improvement**: -1.8%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (18.34% → 8.47%)
**Insight**: For small GEMM workloads (~17us torch baseline), multi-stream cuBLAS setup adds synchronization overhead (barrier stalls 2.37%→4.29%, long scoreboard stalls 16.1%→29.53%) that dominates any concurrency gains, halving SM throughput and warp activity.

## Experiment: Skill 4 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 263.605us
**Optimized**: 48.966us
**Improvement**: 81.4%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (7.51% → 41.32%)
**Insight**: Thrust's optimized reduce/sort implementations eliminated the naive baseline's memory-latency bottleneck (long scoreboard stalls dropped from 28% to 0.5%) and achieved full global load efficiency (50%→100%), beating even PyTorch's 57.5μs with 48.9μs.

## Experiment: Skill 5 verification (2026-04-06)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 3543.927us
**Optimized**: 4088.588us
**Improvement**: -15.4%
**Key metric**: cuda_extension_us (254 regs/thread, 68KB smem, 342MB DRAM read → 121 regs/thread, 13KB smem, 1983MB DRAM read)
**Insight**: Limiting SM count forced cuBLAS to select a completely different kernel with smaller tiles (121 vs 254 regs, 13KB vs 68KB smem), destroying L2 reuse (5.8x more DRAM reads) and increasing barrier stalls (3.87%→9.62%), making the single call 15% slower — the benefit only materializes when freed SMs run concurrent work.
**Conditions**: only if there is concurrent work (custom kernels in separate streams) to fill the freed SMs; in isolation, limiting SM count regresses the cuBLAS call itself
