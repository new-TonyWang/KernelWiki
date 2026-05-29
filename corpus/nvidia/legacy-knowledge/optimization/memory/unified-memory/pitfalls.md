# Unified Memory -- Pitfalls

## P1: Page Fault Stalls During Kernel Execution
**Symptom**: Kernel runs much slower with managed memory than with explicit device memory. Profiler shows high page fault count.
**Detection**: Nsight Compute shows page fault events. Kernel time is 10-100x slower than expected.
**Fix**: Use `cudaMemPrefetchAsync()` to migrate data before kernel launch. Design access patterns to minimize cross-device page faults.
**Source**: Programming Guide, Section 4.1.4.2 (Data Usage Hints)

## P2: CPU Access During GPU Execution on Limited UM Systems
**Symptom**: Crash or undefined behavior on Windows/WSL/Tegra when CPU reads managed memory while GPU kernel is running.
**Detection**: Check if system has limited UM support (cudaDevAttrConcurrentManagedAccess == 0). CPU access during active kernel is illegal.
**Fix**: Synchronize GPU with `cudaDeviceSynchronize()` or stream sync before accessing managed memory on CPU.
**Source**: Programming Guide, Section 2.4.2.3 (Limited Unified Memory Support)

## P3: Oversubscription Thrashing
**Symptom**: Performance degrades severely when total managed allocation far exceeds GPU memory. GPU constantly migrates pages.
**Detection**: Profiler shows high page migration counts. Kernel execution time increases linearly with data size beyond GPU memory.
**Fix**: Partition workload so the active working set fits in GPU memory. Use cudaMemPrefetchAsync to stage data. Use explicit memory management for performance-critical paths.
**Source**: Programming Guide, Section 4.1.4.5 (GPU Memory Oversubscription)

## P4: ReadMostly Hint on Frequently Written Data
**Symptom**: Performance degrades because writes invalidate read-only copies across devices, causing excessive coherence traffic.
**Detection**: Profile shows high page migration after writes to read-mostly data.
**Fix**: Only use `cudaMemAdviseSetReadMostly` for data that is written once (initialization) and read many times. For frequently written data, use preferred location instead.
**Source**: Programming Guide, Section 4.1.4.2 (Data Usage Hints)

## P5: Assuming All Systems Have Full Unified Memory
**Symptom**: Code works on Linux but fails or performs poorly on Windows/Tegra due to limited UM support.
**Detection**: Check `cudaDevAttrConcurrentManagedAccess`. Value 0 means limited support.
**Fix**: Query UM paradigm at runtime and handle limited support: avoid concurrent CPU+GPU access, do not oversubscribe, synchronize before CPU reads.
**Source**: Programming Guide, Section 2.4.2.1 (Unified Memory Paradigms)

## P6: First-Touch Placement on Wrong Device
**Symptom**: Data is initialized on CPU, causing it to reside in CPU memory. GPU access triggers migration for every page.
**Detection**: Profile shows bulk page migration at start of kernel.
**Fix**: Use `cudaMemAdviseSetPreferredLocation` to place data on the target GPU, then `cudaMemPrefetchAsync` to migrate eagerly. Or initialize data on the GPU with a simple init kernel.
**Source**: Programming Guide, Section 2.4.2.2 (Full Unified Memory Feature Support)

## P7: NCU profiles individual kernel launches, so chunked approaches show misleadingly good per-kernel metrics while total end-to-end time barely changes due to accumulated launch overhead and stream synchronization costs (discovered in verification)

**Symptom**: NCU profiles individual kernel launches, so chunked approaches show misleadingly good per-kernel metrics while total end-to-end time barely changes due to accumulated launch overhead and stream synchronization costs.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: The optimized CUDA extension (28us) is still 1 (discovered in verification)

**Symptom**: The optimized CUDA extension (28us) is still 1.5x slower than the torch baseline (18us) and 5x slower than torch.compile (5.7us), suggesting unified memory still carries residual overhead compared to explicit device allocations even with correct attribute-guided management.
**Source**: Level 3 sandbox verification (2026-04-04)

## P9: In a PyTorch extension context, input tensors are already allocated via cudaMalloc by the framework, so cudaMallocManaged cannot replace the existing allocation path and any attempt to use it introduces an extra copy or migration step rather than eliminating one (discovered in verification)

**Symptom**: In a PyTorch extension context, input tensors are already allocated via cudaMalloc by the framework, so cudaMallocManaged cannot replace the existing allocation path and any attempt to use it introduces an extra copy or migration step rather than eliminating one.
**Source**: Level 3 sandbox verification (2026-04-05)

## P10: NCU metrics will not reflect the benefit of cudaMemPrefetchAsync since page fault overhead occurs in the driver/runtime path, not inside the kernel — only wall-clock timing captures the improvement, making it easy to mistakenly conclude the optimization had no effect if you only look at profiler metrics (discovered in verification)

**Symptom**: NCU metrics will not reflect the benefit of cudaMemPrefetchAsync since page fault overhead occurs in the driver/runtime path, not inside the kernel — only wall-clock timing captures the improvement, making it easy to mistakenly conclude the optimization had no effect if you only look at profiler metrics.
**Source**: Level 3 sandbox verification (2026-04-05)

## P11: cudaMemAdvise hints are silently ignored when they don't apply (single-GPU, data already resident), giving the false impression the code is 'optimized' when nothing changed (discovered in verification)

**Symptom**: cudaMemAdvise hints are silently ignored when they don't apply (single-GPU, data already resident), giving the false impression the code is 'optimized' when nothing changed.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: NCU profiles the kernel in isolation (post-migration), so DRAM throughput and SM throughput appear unchanged between baseline and optimized — the massive wall-clock improvement is invisible to roofline analysis because it comes from page-fault latency outside the kernel body (discovered in verification)

**Symptom**: NCU profiles the kernel in isolation (post-migration), so DRAM throughput and SM throughput appear unchanged between baseline and optimized — the massive wall-clock improvement is invisible to roofline analysis because it comes from page-fault latency outside the kernel body
**Source**: Level 3 sandbox verification (2026-04-05)
