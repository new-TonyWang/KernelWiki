# Data Prefetch -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cp.async.bulk.commit_group` | PTX ISA | Commit bulk async copies into group |
| `cp.async.bulk.prefetch` | PTX ISA | Bulk async prefetch to L2 cache |
| `cp.async.bulk.prefetch.tensor` | PTX ISA | TMA tensor prefetch to L2 |
| `cp.async.bulk.tensor[.dim]` | PTX ISA | TMA tensor async copy (1d-5d addressing) |
| `cp.async.bulk.wait_group N` | PTX ISA | Wait for bulk async group N |
| `cp.async.bulk[.dst][.src]` | PTX ISA | Bulk async copy (TMA-style, large transfers) |
| `cp.async.ca.shared.global[.vec]` | PTX ISA | Async copy global->shared (bypasses registers) |
| `cp.async.cg.shared.global[.vec]` | PTX ISA | Async copy global->shared, L2 cache only |
| `cp.async.commit_group` | PTX ISA | Commit outstanding async copies into a group |
| `cp.async.mbarrier.arrive` | PTX ISA | Signal mbarrier from async copy completion |
| `cp.async.wait_all` | PTX ISA | Wait for all async copy groups |
| `cp.async.wait_group N` | PTX ISA | Wait for async copy group N to complete |
| `cp.reduce.async.bulk.tensor[.op]` | PTX ISA | TMA tensor async copy with reduction |
| `cp.reduce.async.bulk[.op]` | PTX ISA | Async bulk copy with reduction |
| `mbarrier.arrive.expect_tx` | PTX ISA | Arrive with expected async transaction byte count |
| `prefetch{.space}` | PTX ISA | Prefetch data into cache (.L1/.L2) |
| `st.async[.type]` | PTX ISA | Asynchronous store to shared memory; .b128 variant in PTX 9.2 |
| `tcgen05.cp[.shape]` | PTX ISA | Copy data between shared memory and Tensor Memory |
| `__pipeline_commit()` | Runtime API | Commit outstanding async copies |
| `__pipeline_memcpy_async()` | Runtime API | Async copy from global to shared memory (hardware-accelerated) |
| `__pipeline_wait_prior()` | Runtime API | Wait for prior pipeline stages to complete |
| `cuda::memcpy_async()` | Runtime API | C++ async memory copy (global to shared) |
| `cudaMemAdvise` | Runtime API | Advise about usage of a memory range (read-mostly, preferred location, accessed-by) |
| `cudaMemDiscardAndPrefetchBatchAsync` | Runtime API | Batch memory discards followed by prefetches |
| `cudaMemPrefetchAsync` | Runtime API | Prefetches memory to a specified destination location |
| `cudaMemPrefetchBatchAsync` | Runtime API | Batch prefetch of multiple memory ranges |
| `cudaMemoryAdvise` | Runtime API | Memory advice (SetReadMostly, SetPreferredLocation, SetAccessedBy and Unset variants) |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `multimem.cp.async.bulk` | PTX ISA | Multi-memory bulk async copy |
| `prefetchu.L1` | PTX ISA | Prefetch uniform address into L1 |
| `cudaMemDiscardBatchAsync` | Runtime API | Batch memory discards (informs driver contents are no longer useful) |
