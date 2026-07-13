# Stream Concurrency -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 10.1.2 (Asynchronous and Overlapping Transfers)"
  - "Best Practices Guide 11.5 (Concurrent Kernel Execution)"
  - "Programming Guide 2.3.1 (Asynchronous Concurrent Execution)"
  - "Programming Guide 2.3.2 (CUDA Streams)"
  - "Programming Guide 2.3.3 (CUDA Events)"
  - "Programming Guide 2.3.6 (Blocking and Non-Blocking Streams)"
  - "Programming Guide 2.3.8 (Implicit Synchronization)"
  - "Programming Guide 2.3.9.1 (Stream Prioritization)"
  - "Programming Guide 4.3 (Stream-Ordered Memory Allocator)"
cross_ref:
  - optimization/latency/cuda-graphs
  - optimization/latency/kernel-launch-overhead
  - optimization/memory/host-device-transfer
related_apis:
  - cudaStreamCreate
  - cudaStreamCreateWithFlags
  - cudaStreamCreateWithPriority
  - cudaStreamWaitEvent
  - cudaMemcpyAsync
  - cudaMallocAsync
  - cudaFreeAsync
unlocks:
  - Overlap of compute with data transfer
  - Concurrent kernel execution
  - Reduced end-to-end latency
conflicts_with:
  - optimization/latency/context-management  # Multiple contexts can serialize streams
```

---

## S1: Overlap Host-to-Device Transfer with Kernel Execution

**When to Use:** When data can be chunked so that the kernel processes one chunk while the next is being transferred.

**How to Apply:**
1. Allocate pinned host memory with `cudaHostAlloc`.
2. Create separate streams for transfer and compute.
3. Issue `cudaMemcpyAsync` in one stream and kernel launch in another.

**Code Template:**
```cpp
int nStreams = 4;
size_t chunkSize = N / nStreams;
cudaStream_t streams[4];
for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int i = 0; i < nStreams; i++) {
    size_t offset = i * chunkSize;
    cudaMemcpyAsync(d_data + offset, h_data + offset,
                    chunkSize * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    kernel<<<chunkSize/256, 256, 0, streams[i]>>>(d_data + offset);
}
```

> Source: BP 10.1.2 -- "Staged concurrent copy and execute... This approach permits some overlapping of the data transfer and execution."

---

## S2: Use Non-Blocking Streams to Avoid Default Stream Serialization

**When to Use:** Always when using multiple streams, unless you specifically need legacy default stream synchronization semantics.

**How to Apply:**
1. Create streams with `cudaStreamNonBlocking` flag.
2. Alternatively, compile with `--default-stream per-thread` for per-thread default streams.

**Code Template:**
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

// These can now run concurrently, even if something uses the default stream
kernelA<<<grid, block, 0, stream1>>>(...);
kernelB<<<grid, block, 0, stream2>>>(...);
```

> Source: PG 2.3.6 -- "In order to create a non-blocking stream, the cudaStreamCreateWithFlags function must be used with the cudaStreamNonBlocking flag."

---

## S3: Use Stream Priorities for Latency-Sensitive Work

**When to Use:** When some kernels are latency-critical (e.g., interactive rendering, real-time inference) and should be scheduled ahead of background work.

**How to Apply:**
1. Query priority range with `cudaDeviceGetStreamPriorityRange`.
2. Assign highest priority (lowest number) to latency-sensitive streams.
3. Note: priorities are hints, not guarantees; they do not preempt running work.

**Code Template:**
```cpp
int minPriority, maxPriority;
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

cudaStream_t criticalStream, backgroundStream;
cudaStreamCreateWithPriority(&criticalStream, cudaStreamDefault, maxPriority);
cudaStreamCreateWithPriority(&backgroundStream, cudaStreamDefault, minPriority);
```

> Source: PG 2.3.9.1 -- "lower numbers correspond to higher priorities."

---

## S4: Use Events for Fine-Grained Cross-Stream Dependencies

**When to Use:** When a kernel in one stream depends on output from a kernel in another stream, but you do not want to synchronize the entire device.

**How to Apply:**
1. Record an event after the producer kernel.
2. Make the consumer stream wait on that event.
3. Use `cudaEventDisableTiming` for events that do not need timestamps.

**Code Template:**
```cpp
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

kernelA<<<grid, block, 0, streamA>>>(...);
cudaEventRecord(event, streamA);

cudaStreamWaitEvent(streamB, event);
kernelB<<<grid, block, 0, streamB>>>(...);
```

> Source: PG 2.3.3 -- "CUDA Events... can be inserted into a CUDA stream to create dependencies between streams."

---

## S5: Use Stream-Ordered Memory Allocation

**When to Use:** When temporary buffers are allocated and freed per kernel launch, and you want to avoid the overhead of synchronous `cudaMalloc`/`cudaFree`.

**How to Apply:**
1. Use `cudaMallocAsync` / `cudaFreeAsync` tied to a stream.
2. The allocator pools memory and recycles it within the stream.
3. No explicit synchronization is needed for allocation/deallocation ordering.

**Code Template:**
```cpp
float* temp;
cudaMallocAsync(&temp, size, stream);
kernel<<<grid, block, 0, stream>>>(temp, ...);
cudaFreeAsync(temp, stream);
// temp is safely freed after kernel completes
```

> Source: PG 4.3 -- "Stream-Ordered Memory Allocator" enables async allocation tied to stream ordering.

---

## S6: Use Batched Memory Transfers

**When to Use:** When many small transfers need to be issued, and individual `cudaMemcpyAsync` calls create CPU overhead.

**How to Apply:**
1. Collect all source pointers, destination pointers, and sizes into arrays.
2. Issue a single `cudaMemcpyBatchAsync` call.

**Code Template:**
```cpp
std::vector<void*> srcs(batchSize), dsts(batchSize);
std::vector<size_t> sizes(batchSize);
// ... fill arrays ...

cudaMemcpyAttributes attrs = {};
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
size_t attrIdx = 0;

cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                     batchSize, &attrs, &attrIdx, 1, nullptr, stream);
```

> Source: PG 3.1.5 -- "cudaMemcpyBatchAsync function... allows batched memory transfers to be optimized."

---

## Cascading Opportunities

- Stream concurrency naturally feeds into `cuda-graphs` for repeated patterns.
- Overlapping transfers with compute requires `host-device-transfer` pinned memory.
- Stream priorities interact with `green-contexts` for resource isolation.

## Conflicts

- **context-management:** Multiple CUDA contexts on the same GPU time-slice, which can prevent streams in different contexts from running concurrently.

## Principles

1. **Streams are work queues:** Operations within a stream execute in order; across streams, they can run concurrently (PG 2.3.2).
2. **Default stream serializes:** The legacy default stream blocks all other blocking streams (PG 2.3.6).
3. **Synchronization should be lazy:** "All independent operations should be issued before dependent operations, and synchronization should be delayed as long as possible" (PG 2.3.8).

## Open Questions

- Q1: How does `CUDA_DEVICE_MAX_CONNECTIONS` affect stream concurrency on Hopper and Blackwell?
- Q2: What is the overhead of `cudaMallocAsync` vs. pre-allocated pool on repeated small allocations?
