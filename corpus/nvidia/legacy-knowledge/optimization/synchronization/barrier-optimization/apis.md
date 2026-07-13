# Barrier Optimization -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `bar.warp.sync membermask` | PTX ISA | Warp-level barrier sync (subset of threads via bitmask) |
| `barrier.cluster.arrive[.sem][.aligned]` | PTX ISA | Cluster-level barrier arrive (.release/.relaxed) |
| `barrier.cluster.wait[.acquire][.aligned]` | PTX ISA | Cluster-level barrier wait |
| `barrier{.cta}.sync[.aligned]` | PTX ISA | CTA barrier with alignment control (non-aligned supported sm_70+) |
| `bar{.cta}.arrive a, b` | PTX ISA | CTA-level barrier arrive (no wait) |
| `bar{.cta}.red.{popc/and/or}` | PTX ISA | CTA-level barrier with reduction |
| `bar{.cta}.sync a[, b]` | PTX ISA | CTA-level barrier sync (16 named barriers) |
| `cp.async.mbarrier.arrive` | PTX ISA | Signal mbarrier from async copy completion |
| `mbarrier.arrive.expect_tx` | PTX ISA | Arrive with expected async transaction byte count |
| `mbarrier.arrive[.shared{::cta/::cluster}].b64` | PTX ISA | Signal arrival at mbarrier; returns arrival token |
| `mbarrier.init[.shared::cta].b64` | PTX ISA | Initialize mbarrier object with expected arrival count |
| `mbarrier.test_wait[.acquire][.shared::cta].b64` | PTX ISA | Non-blocking test if mbarrier phase is complete |
| `mbarrier.try_wait[.acquire][.shared{::cta/::cluster}].b64` | PTX ISA | Blocking try-wait with timeout on mbarrier phase |
| `tcgen05.fence` | PTX ISA | TC5 fence (before/after operations) |
| `wgmma.fence.sync.aligned` | PTX ISA | Fence before wgmma (declare register/smem readiness) |
| `__syncthreads()` | Runtime API | Block-level barrier synchronization |
| `__syncwarp(mask)` | Runtime API | Warp-level synchronization |
| `cuda::barrier` | Runtime API | C++ barrier with arrive/wait pattern |
| `cudaGridDependencySynchronize` | Runtime API | (__device__) Programmatic grid dependency synchronization; blocks until direct grid dependencies complete |
| `cudaStreamWaitEvent` | Runtime API | Make a compute stream wait on an event |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `fence.mbarrier_init.release.cluster` | PTX ISA | Fence for mbarrier initialization |
| `mbarrier.arrive_drop[.shared::cta].b64` | PTX ISA | Arrive and drop participation from future phases |
| `mbarrier.inval[.shared::cta].b64` | PTX ISA | Invalidate mbarrier object |
| `mbarrier.pending_count.b64` | PTX ISA | Get pending arrival count from mbarrier state |
| `cudaDeviceSynchronize` | Runtime API | Wait for compute device to finish |
| `cudaEventSynchronize` | Runtime API | Waits for an event to complete |
| `cudaStreamSynchronize` | Runtime API | Waits for stream tasks to complete |
