
## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 421.252us
**Optimized**: 276.344us
**Improvement**: 34.4%
**Key metric**: global_load_efficiency (56.25% → 100.0%)
**Insight**: Shared memory caching achieved perfect global load coalescing (56→100%) and cut long-scoreboard stalls from 44% to 18% by replacing repeated uncoalesced global reads with coalesced tile loads followed by low-latency shared memory accesses.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 252.792us
**Optimized**: 132.576us
**Improvement**: 47.6%
**Key metric**: sm__cycles_active.avg (380451 cycles, 58.1% warp active, 25.7% long_scoreboard stalls → 222708 cycles, 91.2% warp active, 16.1% long_scoreboard stalls)
**Insight**: Staging global reads through shared memory eliminated non-coalesced access patterns, cutting active cycles by 41% and raising warp utilization from 58% to 91%, despite executing 65% more instructions due to shared memory management overhead.

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 4.435us
**Optimized**: 4.691us
**Improvement**: -5.8%
**Key metric**: cuda_extension_us (4.435us → 4.691us)
**Insight**: Dynamic shared memory adds no benefit here because both versions allocate the same 2176 bytes of shared memory with identical access patterns — the flexibility of runtime-sized tiles provides no advantage when the tile size is fixed, and the minor overhead results in a ~5.8% regression.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 1857.908us
**Optimized**: 1858.38us
**Improvement**: -0.03%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (91.34% → 91.44%)
**Insight**: The L1/shared memory carveout had no measurable effect because the kernel already uses only 9KB shared memory per block, well within default limits, and the bottleneck is memory throughput at 91%+ SOL — the carveout simply doesn't change the resource balance that matters.

## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 10.521us
**Optimized**: 10.899us
**Improvement**: -3.6%
**Key metric**: launch__shared_mem_per_block_allocated (2048 bytes, occupancy_limit_shared_mem=16 → 67584 bytes, occupancy_limit_shared_mem=3)
**Insight**: Requesting 66KB shared memory when the workload only needed 2KB destroyed occupancy (16→3 blocks/SM) and eliminated L1 cache hits (50%→0%), causing a 3.6% regression despite marginally reducing long-scoreboard stalls.
