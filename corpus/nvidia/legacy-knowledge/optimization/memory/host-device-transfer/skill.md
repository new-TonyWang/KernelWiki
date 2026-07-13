---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.1 (Data Transfer Between Host and Device)
  - Best Practices Guide, Section 10.1.1 (Pinned Memory)
  - Best Practices Guide, Section 10.1.3 (Zero Copy)
  - Programming Guide, Section 2.4.3 (Page-Locked Host Memory)
  - Programming Guide, Section 2.4.3.1 (Mapped Memory)
  - Programming Guide, Section 3.1.5 (Batched Memory Transfers)
cross_ref:
  - Best Practices Guide, Section 10.1.2 (Asynchronous and Overlapping Transfers)
  - Programming Guide, Section 3.4.2 (Multi-Device Peer-to-Peer Transfers)
related_apis:
  - cudaMemcpy, cudaMemcpyAsync, cudaMallocHost, cudaHostAlloc, cudaHostRegister, cudaFreeHost, cudaMallocAsync, cudaFreeAsync, cudaMemcpyBatchAsync, cudaDeviceEnablePeerAccess
related_experience: []
unlocks:
  - "latency/stream-concurrency: async transfers enable overlap with kernel execution"
  - "memory/unified-memory: understanding transfer cost helps decide when unified memory is preferable"
conflicts_with: []
---

# Host-Device Transfer

## Skill 1: Use Pinned (Page-Locked) Memory for Higher Bandwidth

### When to Use
- Transferring data between host and device frequently
- Need maximum PCIe/NVLink bandwidth

### How to Apply
1. Allocate host memory with `cudaMallocHost()` or `cudaHostAlloc()` instead of `malloc()`
2. Use `cudaFreeHost()` to free
3. For existing allocations, use `cudaHostRegister()` to pin on-the-fly

### Code Template
```cuda
float* h_data;
cudaMallocHost(&h_data, N * sizeof(float));

// Initialize
for (int i = 0; i < N; i++) h_data[i] = (float)i;

// Transfer at maximum bandwidth
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

// Cleanup
cudaFreeHost(h_data);
```

### Source
Best Practices Guide, Section 10.1.1 (Pinned Memory)

## Skill 2: Overlap Transfers with Computation via Async Copies

### When to Use
- Data can be divided into chunks processed independently
- GPU has concurrent copy engines (asyncEngineCount > 0)

### How to Apply
1. Create multiple streams
2. Use pinned host memory (required for async transfers)
3. In each stream: async copy chunk -> launch kernel on chunk
4. The copy in stream N+1 overlaps with kernel in stream N

### Code Template
```cuda
const int nStreams = 4;
cudaStream_t streams[nStreams];
for (int i = 0; i < nStreams; i++) cudaStreamCreate(&streams[i]);

float* h_data;
cudaMallocHost(&h_data, N * sizeof(float));

size_t chunk = N / nStreams;
for (int i = 0; i < nStreams; i++) {
    size_t offset = i * chunk;
    cudaMemcpyAsync(d_data + offset, h_data + offset,
                    chunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    kernel<<<chunk / 256, 256, 0, streams[i]>>>(d_data + offset);
}
```

### Source
Best Practices Guide, Section 10.1.2 (Asynchronous and Overlapping Transfers)

## Skill 3: Use Zero Copy (Mapped Pinned Memory) for Integrated GPUs

### When to Use
- Integrated GPU where host and device share physical memory
- Data is read/written only once by the kernel
- Want to avoid explicit copy overhead

### How to Apply
1. Check `prop.canMapHostMemory` and enable with `cudaSetDeviceFlags(cudaDeviceMapHost)`
2. Allocate with `cudaHostAlloc(&ptr, size, cudaHostAllocMapped)`
3. Get device pointer with `cudaHostGetDevicePointer()`
4. Pass device pointer to kernel

### Code Template
```cuda
cudaSetDeviceFlags(cudaDeviceMapHost);

float* h_data;
float* d_data;
cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_data, h_data, 0);

// Initialize on host
for (int i = 0; i < N; i++) h_data[i] = (float)i;

// Kernel uses d_data directly (data pulled over PCIe/NVLink on demand)
kernel<<<grid, block>>>(d_data);
```

### Source
Best Practices Guide, Section 10.1.3 (Zero Copy)

## Skill 4: Batch Small Transfers into One Large Transfer

### When to Use
- Application needs to transfer many small buffers between host and device
- Per-transfer overhead dominates for small sizes

### How to Apply
1. Pack non-contiguous host data into a contiguous staging buffer
2. Transfer the staging buffer in one large cudaMemcpy
3. Unpack on the device side, or use cudaMemcpyBatchAsync for multiple ranges

### Code Template
```cuda
// Instead of many small transfers:
// for each small buffer: cudaMemcpy(..., smallSize, ...)

// Batch into one:
float* staging;
cudaMallocHost(&staging, totalSize);
size_t offset = 0;
for (int i = 0; i < numBuffers; i++) {
    memcpy(staging + offset, hostBuffers[i], bufferSizes[i]);
    offset += bufferSizes[i] / sizeof(float);
}
cudaMemcpy(d_data, staging, totalSize, cudaMemcpyHostToDevice);
```

### Source
Best Practices Guide, Section 10.1 (Data Transfer Between Host and Device)

## Skill 5: Use Stream-Ordered Memory Allocation

### When to Use
- Kernel output size is not known until the kernel runs
- Want to avoid expensive cudaMalloc/cudaFree synchronization

### How to Apply
1. Use `cudaMallocAsync()` and `cudaFreeAsync()` with a stream
2. Allocations are pooled and reused automatically
3. No implicit synchronization unlike cudaMalloc/cudaFree

### Code Template
```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

float* d_temp;
cudaMallocAsync(&d_temp, N * sizeof(float), stream);
kernel<<<grid, block, 0, stream>>>(d_temp, N);
cudaFreeAsync(d_temp, stream);

cudaStreamSynchronize(stream);
```

### Source
Best Practices Guide, Section 10.3 (Allocation)

## Skill 6: Minimize Host-Device Transfers by Keeping Data on Device

### When to Use
- Multiple kernels process the same data sequentially
- Intermediate results are consumed only by GPU kernels

### How to Apply
1. Transfer data to device once
2. Chain kernel calls on device memory without copying back
3. Only transfer final results back to host

### Code Template
```cuda
cudaMemcpy(d_data, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

// All intermediate processing on device
kernel1<<<grid, block>>>(d_data);
kernel2<<<grid, block>>>(d_data);
kernel3<<<grid, block>>>(d_data);

// Only final result goes back
cudaMemcpy(h_output, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
```

### Source
Best Practices Guide, Section 10.1 (Data Transfer Between Host and Device)

## Cascading Opportunities (unlocks)
After optimizing host-device transfers:
1. Check latency/stream-concurrency -- overlap remaining transfers with computation
2. Check memory/unified-memory -- consider if managed memory simplifies the data flow

## Conflicts
- None significant

## Principles
- **P1**: Minimize data transfer between host and device. Even if a kernel is faster on CPU, the transfer cost may not justify moving data back.
- **P2**: Pinned memory achieves ~12 GB/s on PCIe Gen3 x16 vs ~6 GB/s for pageable memory.
- **P3**: Batch many small transfers into fewer large transfers. Per-transfer overhead is significant for small sizes.
- **P4**: Intermediate data should stay on device between kernel calls.
- **P5**: cudaMallocAsync/cudaFreeAsync avoid the implicit synchronization of cudaMalloc/cudaFree.

## Open Questions (for Level 3 verification)
- Q1: What is the crossover point where explicit transfer + kernel is faster than zero-copy for discrete GPUs?
- Q2: How does cudaMemcpyBatchAsync performance compare to packing into a contiguous buffer?
- Q3: On NVLink systems, does pinned memory still provide significant benefit over pageable memory?
