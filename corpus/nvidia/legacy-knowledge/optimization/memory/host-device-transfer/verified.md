
## Experiment: Skill 5 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: refuted
**Baseline**: 7.939us
**Optimized**: 7.965us
**Improvement**: -0.3%
**Key metric**: cuda_extension_us (7.939us → 7.965us)
**Insight**: Stream-ordered allocation (cudaMallocAsync/cudaFreeAsync) reduces host-side synchronization overhead from malloc/free calls, but this benchmark measures kernel execution time only — the allocation overhead is not captured in the profiled region, so the kernel itself runs identically.

## Experiment: Skill 6 verification (2026-04-05)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 6321.323us
**Optimized**: 39.885us
**Improvement**: 99.37%
**Key metric**: cuda_extension_us (end-to-end) (6321.323us (redundant H↔D transfers between kernels) → 39.885us (data stays on device))
**Insight**: Kernel-level metrics (DRAM throughput, compute throughput, cycles) are virtually identical between baseline and optimized, confirming the 158x speedup comes entirely from eliminating redundant host-device memory copies between chained kernel invocations.

## Experiment: Skill 1 verification (2026-04-07)

**Hardware**: H200 SXM, CUDA 13.2
**Verdict**: verified
**Baseline**: 9763.081us
**Optimized**: 2422.877us
**Improvement**: 75.2%
**Key metric**: host_device_transfer_time (9763.081us (pageable memory) → 2422.877us (pinned memory))
**Insight**: Pinned (page-locked) memory avoids an extra copy through a staging buffer in the DMA path, enabling the GPU to transfer directly from/to the host address at full PCIe bandwidth, yielding a ~4x speedup.
