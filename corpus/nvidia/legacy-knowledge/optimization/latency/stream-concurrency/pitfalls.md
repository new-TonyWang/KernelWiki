# Stream Concurrency -- Pitfalls

## P1: Legacy Default Stream Serializing All Work

**Symptom:** Kernels in different streams do not overlap. Nsight Systems timeline shows sequential execution.

**Detection:** Any call to the default stream (stream 0) or `cudaMemcpy` (blocking) between stream operations.

**Fix:** Use non-blocking streams (`cudaStreamNonBlocking`), or compile with `--default-stream per-thread`.

> Source: PG 2.3.6.1 -- "When an operation is launched into this default stream, it will synchronize with all other blocking streams."

---

## P2: Implicit Synchronization from Blocking API Calls

**Symptom:** Concurrency is destroyed by `cudaMemcpy`, `cudaDeviceSynchronize`, or legacy stream operations between independent work items.

**Detection:** `cudaDeviceSynchronize` or synchronous `cudaMemcpy` appearing between kernel launches in the profiler.

**Fix:** Use `cudaMemcpyAsync` with pinned memory and explicit streams. Replace `cudaDeviceSynchronize` with stream-level or event-level synchronization.

> Source: PG 2.3.8 -- "Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in-between them."

---

## P3: Forgetting to Pin Host Memory for Async Transfers

**Symptom:** `cudaMemcpyAsync` silently falls back to synchronous behavior because host memory is pageable.

**Detection:** Nsight Systems shows `cudaMemcpyAsync` taking the same time as `cudaMemcpy` with no overlap.

**Fix:** Use `cudaHostAlloc` or `cudaHostRegister` to pin host memory before async operations.

```cpp
// Bad: pageable memory, async falls back to sync
float* h_data = (float*)malloc(size);
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

// Good: pinned memory enables true async
float* h_data;
cudaHostAlloc(&h_data, size, cudaHostAllocDefault);
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
```

> Source: BP 10.1.2 -- "the asynchronous transfer version requires pinned host memory."

---

## P4: Too Many Streams Causing Resource Contention

**Symptom:** Performance degrades when using dozens of streams due to memory pool fragmentation or workqueue contention.

**Detection:** Performance decreases with stream count beyond a threshold. Memory usage increases unexpectedly.

**Fix:** Limit stream count to match the hardware concurrency (typically 8-32 streams). Reuse streams across iterations.

> Source: PG 2.3.2 -- streams share underlying hardware work queues; excessive streams can cause false dependencies.

---

## P5: Stream Priority Not Preempting Running Work

**Symptom:** High-priority kernel still waits for low-priority kernel to complete.

**Detection:** Nsight Systems shows high-priority kernel starting only after low-priority kernel finishes.

**Fix:** Stream priorities only affect scheduling of new work, not preemption of running kernels. For true preemption, consider `green-contexts` or breaking low-priority work into smaller chunks.

> Source: PG 2.3.9.1 -- "Stream priorities will not preempt already executing work, or guarantee any specific execution order."

---

## P6: Not Issuing Independent Work Before Dependent Work

**Symptom:** Potential concurrency is lost because dependent operations are issued before independent ones.

**Detection:** Nsight Systems shows available but unused concurrency windows.

**Fix:** Issue all independent operations first, then issue dependent ones. Delay synchronization as long as possible.

> Source: PG 2.3.8 -- "All independent operations should be issued before dependent operations, and synchronization should be delayed as long as possible."

## P7: At ~16µs total execution, kernel launch overhead and measurement noise (~0 (discovered in verification)

**Symptom**: At ~16µs total execution, kernel launch overhead and measurement noise (~0.5%) dwarf any concurrency gains from stream events; this skill requires a workload with multiple substantial kernels to demonstrate overlap benefits.
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Non-blocking stream creation adds slight overhead (stream management, flag checks) that can regress latency (~8%) when the workload is already a single compute-bound kernel with no inter-stream serialization to begin with (discovered in verification)

**Symptom**: Non-blocking stream creation adds slight overhead (stream management, flag checks) that can regress latency (~8%) when the workload is already a single compute-bound kernel with no inter-stream serialization to begin with.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: cudaMallocAsync benefits are invisible to NCU kernel profiling since the savings occur in the CUDA runtime/driver path between kernel launches, not within kernels themselves; measuring this skill requires end-to-end host timing of an allocate-launch-free loop, not single-kernel profiling (discovered in verification)

**Symptom**: cudaMallocAsync benefits are invisible to NCU kernel profiling since the savings occur in the CUDA runtime/driver path between kernel launches, not within kernels themselves; measuring this skill requires end-to-end host timing of an allocate-launch-free loop, not single-kernel profiling.
**Source**: Level 3 sandbox verification (2026-04-06)
