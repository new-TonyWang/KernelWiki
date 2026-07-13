---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Programming Guide, Section 2.4.2 (Unified Memory)
  - Programming Guide, Section 4.1 (Unified Memory)
  - Programming Guide, Section 4.1.1.2 (Performance Tuning)
  - Programming Guide, Section 4.1.4.2 (Data Usage Hints)
  - Programming Guide, Section 4.1.4.5 (GPU Memory Oversubscription)
  - Best Practices Guide, Section 10.1.4 (Unified Virtual Addressing)
cross_ref:
  - Programming Guide, Section 2.4.2.4 (Memory Advise and Prefetch)
related_apis:
  - cudaMallocManaged, cudaMemAdvise, cudaMemPrefetchAsync, cudaMemPrefetchBatchAsync, cudaStreamAttachMemAsync, cudaMemoryAdvise
related_experience: []
unlocks:
  - "memory/host-device-transfer: unified memory can eliminate explicit transfer management"
  - "memory/numa-binding: NUMA-aware placement hints optimize unified memory on multi-GPU systems"
conflicts_with:
  - "memory/l2-cache-control: page migration may interfere with L2 persistence windows"
---

# Unified Memory

## Skill 1: Allocate Managed Memory with cudaMallocManaged

### When to Use
- Want simplified memory management without explicit host/device transfers
- Prototyping or when programmer productivity is prioritized
- Data accessed by both CPU and GPU at different times

### How to Apply
1. Replace `cudaMalloc` + `cudaMemcpy` with `cudaMallocManaged()`
2. Access the pointer from both host and device code
3. The runtime automatically migrates data on demand

### Code Template
```cuda
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Initialize on host (data initially resides on CPU)
for (int i = 0; i < N; i++) data[i] = (float)i;

// GPU kernel (data migrates to GPU on first access)
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Read on host (data migrates back)
printf("Result: %f\n", data[0]);

cudaFree(data);
```

### Source
Programming Guide, Section 2.4.2 (Unified Memory)

## Skill 2: Use cudaMemPrefetchAsync to Avoid Page Fault Latency

### When to Use
- Performance-critical code using unified memory
- Know in advance which device will access the data next

### How to Apply
1. Call `cudaMemPrefetchAsync(ptr, size, deviceId, stream)` before the kernel
2. Data migrates in background while other work executes
3. Avoids page fault overhead during kernel execution

### Code Template
```cuda
float* data;
cudaMallocManaged(&data, N * sizeof(float));
// Initialize on host...

// Prefetch to GPU 0 before kernel launch
cudaMemPrefetchAsync(data, N * sizeof(float), 0, stream);
kernel<<<grid, block, 0, stream>>>(data, N);
cudaStreamSynchronize(stream);

// Prefetch back to CPU before host access
cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
printf("Result: %f\n", data[0]);
```

### Source
Programming Guide, Section 2.4.2.4 (Memory Advise and Prefetch)

## Skill 3: Use cudaMemAdvise for Read-Mostly Data

### When to Use
- Data is written once and read many times from GPU (weights, lookup tables)
- Want to create read-only replicas on multiple GPUs without explicit copies

### How to Apply
1. Call `cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device)` after initialization
2. The driver creates read-only copies on accessing devices
3. Writing invalidates copies (only use for truly read-mostly data)

### Code Template
```cuda
float* weights;
cudaMallocManaged(&weights, N * sizeof(float));
// Initialize weights on host...

// Mark as read-mostly: GPU gets a local copy
cudaMemAdvise(weights, N * sizeof(float), cudaMemAdviseSetReadMostly, 0);

kernel<<<grid, block>>>(weights, N);  // Fast: local read-only copy
```

### Source
Programming Guide, Section 4.1.4.2 (Data Usage Hints)

## Skill 4: Set Preferred Location to Control Initial Placement

### When to Use
- Want to control where managed memory physically resides
- Data is predominantly used by a specific device

### How to Apply
1. Call `cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device)`
2. Data will be placed on the specified device and not migrate away unless necessary
3. Other devices access it via direct access (if peer access enabled) or migration

### Code Template
```cuda
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Place data on GPU 0
cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetPreferredLocation, 0);
// Tell GPU 1 it will also access this data
cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetAccessedBy, 1);

cudaMemPrefetchAsync(data, N * sizeof(float), 0, stream);
kernel<<<grid, block, 0, stream>>>(data, N);
```

### Source
Programming Guide, Section 4.1.4.2 (Data Usage Hints)

## Skill 5: Handle GPU Memory Oversubscription

### When to Use
- Dataset exceeds GPU memory capacity
- Full unified memory support is available (Linux, CC 6.0+)

### How to Apply
1. Allocate more managed memory than GPU physical memory
2. Pages are migrated on demand between CPU and GPU
3. Use cudaMemAdvise and cudaMemPrefetchAsync to minimize thrashing
4. Partition data so the working set fits in GPU memory

### Code Template
```cuda
// Allocate more than GPU memory (oversubscription)
size_t gpu_mem;
cudaMemGetInfo(nullptr, &gpu_mem);
size_t alloc_size = gpu_mem * 2;  // 2x GPU memory

float* data;
cudaMallocManaged(&data, alloc_size);

// Process in chunks that fit in GPU memory
size_t chunk = gpu_mem / 2;
for (size_t offset = 0; offset < alloc_size / sizeof(float); offset += chunk / sizeof(float)) {
    cudaMemPrefetchAsync(data + offset, chunk, 0, stream);
    kernel<<<grid, block, 0, stream>>>(data + offset, chunk / sizeof(float));
}
```

### Source
Programming Guide, Section 4.1.4.5 (GPU Memory Oversubscription)

## Skill 6: Query Unified Memory Paradigm for Portable Code

### When to Use
- Writing code that must work across different systems (Windows, Linux, Tegra, Grace Hopper)
- Need to know if full or limited unified memory is available

### How to Apply
1. Query `cudaDevAttrConcurrentManagedAccess` to check full vs limited support
2. Query `cudaDevAttrPageableMemoryAccess` to check if all memory is unified
3. Query `cudaDevAttrPageableMemoryAccessUsesHostPageTables` for hardware vs software coherence

### Code Template
```cuda
int concurrent = 0, pageable = 0, hostPageTables = 0;
cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0);
cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0);
cudaDeviceGetAttribute(&hostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);

if (concurrent) {
    if (pageable) {
        if (hostPageTables) printf("Full UM with hardware coherence (ATS)\n");
        else printf("Full UM with software coherence (HMM)\n");
    } else {
        printf("Full UM for CUDA managed allocations only\n");
    }
} else {
    printf("Limited UM (Windows/WSL/Tegra)\n");
}
```

### Source
Programming Guide, Section 2.4.2.1 (Unified Memory Paradigms)

## Cascading Opportunities (unlocks)
After configuring unified memory:
1. Check memory/host-device-transfer -- unified memory may eliminate explicit transfers
2. Check memory/numa-binding -- NUMA-aware hints optimize placement on multi-socket systems

## Conflicts
- memory/l2-cache-control: page migration can interfere with L2 persistence windows

## Principles
- **P1**: Managed memory is first allocated where it is first touched. First-touch on CPU places it in CPU memory; first-touch on GPU places it in GPU memory.
- **P2**: Page migration granularity differs: software coherence migrates entire pages; hardware coherence (ATS) operates at cache-line granularity.
- **P3**: cudaMemPrefetchAsync is critical for performance. Without it, page faults during kernel execution cause severe stalls.
- **P4**: On limited UM systems (Windows), CPU must not access managed memory while GPU is active.
- **P5**: Oversubscription is supported only with full UM; on limited systems, total managed allocation must fit in GPU memory.

## Open Questions (for Level 3 verification)
- Q1: What is the page fault overhead on A100 for a GEMM kernel with unprefetched managed memory?
- Q2: On Grace Hopper (ATS), is there still a performance gap between managed memory and cudaMalloc?
- Q3: How does cudaMemAdviseSetReadMostly interact with atomic operations on the data?
