
## Experiment: Skill 5 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: conditional
**Baseline**: 123.567us
**Optimized**: 120.524us
**Improvement**: 2.46%
**Key metric**: dram__bytes_read.sum (67118848 bytes (42.85% DRAM throughput) → 16780032 bytes (24.89% DRAM throughput))
**Insight**: Per-kernel DRAM traffic dropped 75% and active cycles dropped 76% due to chunked prefetching, but wall-clock time only improved 2.5% because the test data fits in GPU memory — the chunking adds kernel launch overhead and synchronization that offsets the per-chunk efficiency gains when oversubscription pressure is absent.
**Conditions**: only if dataset actually exceeds GPU physical memory; on in-memory workloads the chunked prefetch adds launch overhead that negates per-kernel gains

## Experiment: Skill 6 verification (2026-04-04)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 1779.164us
**Optimized**: 28.073us
**Improvement**: 98.42%
**Key metric**: cuda_extension_us (1779.164us (massive page-fault overhead) → 28.073us (proper UM management))
**Insight**: The 63x wall-clock speedup with nearly identical NCU kernel metrics (same throughput, same instructions, same occupancy) indicates the baseline suffered from unified memory page-fault/migration overhead on the host side, and querying device attributes allowed selecting proper prefetching or allocation strategy that eliminated that overhead.

## Experiment: Skill 1 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 3.971us
**Optimized**: 3.987us
**Improvement**: -0.4%
**Key metric**: dram_throughput (14.61% → 14.96%)
**Insight**: cudaMallocManaged is a programming convenience, not a performance optimization — when data is already GPU-resident (as in a PyTorch extension receiving device tensors), unified memory adds page-fault migration overhead rather than removing explicit copy overhead, yielding no measurable speedup.

## Experiment: Skill 2 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 40702.976us
**Optimized**: 29190.583us
**Improvement**: 28.3%
**Key metric**: cuda_extension_us (end-to-end) (40703us → 29191us)
**Insight**: Prefetching reduced end-to-end time by 28% by migrating pages before kernel launch, but NCU kernel metrics (DRAM throughput 43.4%, SM throughput 49.3%) are identical because the improvement is in eliminating page fault stalls that occur outside the kernel's profiled cycles.

## Experiment: Skill 3 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 751.802us
**Optimized**: 753.694us
**Improvement**: -0.25%
**Key metric**: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (84.08% → 84.37%)
**Insight**: cudaMemAdviseSetReadMostly is a multi-GPU optimization that creates read-only replicas across devices; on a single GPU the data is already local so the hint is a no-op and produces no measurable benefit.

## Experiment: Skill 4 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 10245.815us
**Optimized**: 124.255us
**Improvement**: 98.79%
**Key metric**: wall_clock_time (10245.815us (page-fault dominated) → 124.255us (prefetched, no faults))
**Insight**: cudaMemAdvise + cudaMemPrefetchAsync eliminates on-demand page migration faults, collapsing wall-clock time by ~82x while the kernel's compute/memory throughput profile stays identical (42.8% DRAM, 48.5-48.8% SM), proving the gain is entirely from avoiding page-fault overhead rather than kernel-level optimization.
**Conditions**: only when baseline uses naive cudaMallocManaged without hints, causing on-demand page faults; the kernel itself is unchanged
