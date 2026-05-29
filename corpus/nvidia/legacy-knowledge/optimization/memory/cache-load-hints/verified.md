
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 56.81us
**Optimized**: 56.892us
**Improvement**: -0.14%
**Key metric**: dram_throughput (43.13% → 43.25%)
**Insight**: On modern GPUs, the compiler already emits ld.global.nc (non-coherent/texture cache path) loads for const __restrict__ pointers, making explicit __ldg() redundant with no measurable benefit.

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 56.873us
**Optimized**: 56.742us
**Improvement**: 0.23%
**Key metric**: l1tex__t_sector_hit_rate.pct (0.0% → 0.0%)
**Insight**: L1 hit rate was already 0% in the baseline (streaming access pattern with no reuse), so __ldcg() had nothing to bypass — the compiler/hardware was already effectively not caching in L1.

## Experiment: Skill 3 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 56.829us
**Optimized**: 56.746us
**Improvement**: 0.15%
**Key metric**: dram_throughput (42.8% → 41.53%)
**Insight**: __ldcs() has no measurable impact on this simple elementwise kernel because the access pattern is already naturally streaming — hardware prefetching handles it, and there is no competing reuse data in cache to benefit from the evict-first hint.

## Experiment: Skill 4 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 56.697us
**Optimized**: 56.812us
**Improvement**: -0.2%
**Key metric**: dram__throughput.avg.pct_of_peak_sustained_elapsed (43.59% → 42.67%)
**Insight**: __stwt()/__stwb() store hints have no measurable effect on this workload because the kernel is not store-bottlenecked — it is dominated by long-scoreboard (load latency) stalls at ~73%, meaning store policy changes cannot improve the critical path.

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.778us
**Optimized**: 4.784us
**Improvement**: -0.13%
**Key metric**: sm__throughput.avg.pct_of_peak_sustained_elapsed (38.25% → 36.55%)
**Insight**: cudaFuncSetCacheConfig is a hint that modern GPUs (Ampere+) largely ignore since they use a unified L1/shared memory architecture with adaptive partitioning, making manual configuration ineffective on this hardware.

## Experiment: Skill 6 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 56.877us
**Optimized**: 56.666us
**Improvement**: 0.37%
**Key metric**: dram__throughput.avg.pct_of_peak_sustained_elapsed (42.88% → 41.45%)
**Insight**: On a simple streaming element-wise kernel, the hardware cache replacement policy already handles sequential one-touch data efficiently, so the __ldlu() eviction hint provides no measurable speedup despite reducing DRAM write-back traffic by 15%.
