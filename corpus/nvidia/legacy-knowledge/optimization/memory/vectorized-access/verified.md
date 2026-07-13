
## Experiment: Skill 1 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 56.767us
**Optimized**: 33.133us
**Improvement**: 41.6%
**Key metric**: dram_throughput (43.28% → 69.4%)
**Insight**: float4 loads reduce total instructions by 66% (8.9M→3.0M) and shift the kernel from underutilized to properly memory-bound, achieving near-parity with torch baseline (33.5µs vs 33.1µs).

## Experiment: Skill 2 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 4.003us
**Optimized**: 2.909us
**Improvement**: 27.3%
**Key metric**: sm__cycles_active.avg (6542.95 cycles → 5484.8 cycles)
**Insight**: float2 vectorized loads halve the number of executed instructions (655360→376832) by processing two elements per thread, reducing instruction overhead and active cycles by ~16%, which translates to a 27% wall-clock speedup.

## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 8.63us
**Optimized**: 8.329us
**Improvement**: 3.49%
**Key metric**: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (25.0% → 100.0%)
**Insight**: Vectorized 128-bit loads achieve 100% sector utilization (up from 25%) and 20.8% fewer instructions, but wall-clock improvement is only 3.5% because the baseline's inefficient scalar loads were partially masked by a 70.6% L1 cache hit rate compensating for redundant cache-line fetches.
