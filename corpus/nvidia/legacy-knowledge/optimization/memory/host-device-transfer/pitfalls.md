# Host-Device Transfer -- Pitfalls

## P1: Using Pageable Memory for Async Transfers
**Symptom**: `cudaMemcpyAsync()` behaves like a blocking call. No overlap with computation.
**Detection**: Timeline profiler shows transfer and kernel execution are sequential, not overlapped.
**Fix**: Async transfers require pinned (page-locked) host memory. Use `cudaMallocHost()` or `cudaHostAlloc()`.
**Source**: Best Practices Guide, Section 10.1.1 (Pinned Memory)

## P2: Over-Pinning Host Memory Degrades System Performance
**Symptom**: System becomes sluggish; other applications slow down. Page faults increase.
**Detection**: Pinned memory allocation consumes a large fraction of system memory. OS has less pageable memory for other processes.
**Fix**: Pin only the memory that needs async transfer. Release pinned memory promptly. There is no hard limit, but be conservative.
**Source**: Best Practices Guide, Section 10.1.1 (Pinned Memory)

## P3: Default Stream Serializing Async Operations
**Symptom**: Async copy and kernel in the default stream do not overlap.
**Detection**: Timeline shows sequential execution despite using cudaMemcpyAsync.
**Fix**: Use non-default streams (stream ID != 0) for concurrent copy and compute. The default stream has implicit synchronization.
**Source**: Best Practices Guide, Section 10.1.2 (Asynchronous and Overlapping Transfers)

## P4: Blocking Transfer Mixed with Async Transfer
**Symptom**: A blocking `cudaMemcpy()` prevents overlap of subsequent async operations.
**Detection**: Timeline shows all operations serialize around the blocking transfer.
**Fix**: Replace blocking transfers with `cudaMemcpyAsync()` on non-default streams. A blocking transfer in the default stream blocks all other streams.
**Source**: Best Practices Guide, Section 10.1.2 (Asynchronous and Overlapping Transfers)

## P5: Zero Copy on Discrete GPU with Repeated Access
**Symptom**: Kernel using mapped host memory runs extremely slowly on discrete GPU because every access traverses PCIe.
**Detection**: Profile shows massive PCIe traffic. Bandwidth is limited to PCIe speed (~12-16 GB/s) instead of device memory bandwidth.
**Fix**: Zero copy is only beneficial on integrated GPUs or for one-time access patterns. For repeated access on discrete GPUs, copy data to device memory first.
**Source**: Best Practices Guide, Section 10.1.3 (Zero Copy)

## P6: cudaMalloc/cudaFree Implicit Synchronization
**Symptom**: Kernel execution appears to stall around memory allocation calls.
**Detection**: Timeline shows synchronization points at cudaMalloc/cudaFree.
**Fix**: Use `cudaMallocAsync()` and `cudaFreeAsync()` for stream-ordered allocation without implicit synchronization.
**Source**: Best Practices Guide, Section 10.3 (Allocation)

## P7: cudaMallocAsync benefits are invisible in kernel-only profiling; the win is in reduced host-side stalls between kernel launches, which requires an end-to-end multi-kernel pipeline benchmark to measure (discovered in verification)

**Symptom**: cudaMallocAsync benefits are invisible in kernel-only profiling; the win is in reduced host-side stalls between kernel launches, which requires an end-to-end multi-kernel pipeline benchmark to measure.
**Source**: Level 3 sandbox verification (2026-04-05)

## P8: Both implementations are still ~70x slower than the torch baseline (~33us), because torch keeps tensors on-device and avoids the host-device transfer entirely — pinned memory helps when transfers are unavoidable, but eliminating transfers altogether is far superior (discovered in verification)

**Symptom**: Both implementations are still ~70x slower than the torch baseline (~33us), because torch keeps tensors on-device and avoids the host-device transfer entirely — pinned memory helps when transfers are unavoidable, but eliminating transfers altogether is far superior.
**Source**: Level 3 sandbox verification (2026-04-07)
