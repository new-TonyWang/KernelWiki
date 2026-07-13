# GPU Library Usage -- Pitfalls

## P1: Not Setting the cuBLAS Stream (Implicit Default Stream)

**Symptom:** cuBLAS calls serialize with all other GPU work because they use the legacy default stream.

**Detection:** Nsight Systems shows cuBLAS kernels blocking other streams.

**Fix:** Always call `cublasSetStream(handle, stream)` before cuBLAS operations.

> Source: cuBLAS API -- handle defaults to legacy stream if not set.

---

## P2: Missing Workspace Allocation Causing Internal Synchronization

**Symptom:** cuBLAS internally allocates workspace on each call, causing implicit synchronization.

**Detection:** Unexpected `cudaMalloc`/`cudaFree` calls inside cuBLAS operations visible in profiler.

**Fix:** Pre-allocate workspace with `cublasSetWorkspace` to avoid internal allocation.

```cpp
void* workspace;
size_t workspaceSize = 1 << 22;  // 4 MB
cudaMalloc(&workspace, workspaceSize);
cublasSetWorkspace(handle, workspace, workspaceSize);
```

> Source: cuBLAS API -- cublasSetWorkspace.

---

## P3: Using Wrong Pointer Mode for Scalars

**Symptom:** Incorrect results or crash because alpha/beta scalars are on host but cuBLAS expects device pointers (or vice versa).

**Detection:** Incorrect GEMM results. `cudaErrorIllegalAddress` when scalars are on the wrong side.

**Fix:** Match pointer mode with where scalars reside. Default is host pointers.

```cpp
// Default: scalars on host
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, ..., &alpha, ..., &beta, ...);

// If scalars on device:
cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
cublasSgemm(handle, ..., d_alpha, ..., d_beta, ...);
```

> Source: cuBLAS API -- cublasSetPointerMode.

---

## P4: Batched GEMM with Too-Small Batch Elements

**Symptom:** Batched GEMM has poor utilization because individual matrix multiplications are too small to fill the GPU.

**Detection:** Low GPU occupancy during batched GEMM. Nsight Compute shows SM utilization below 50%.

**Fix:** Ensure batch count is large enough. Consider using `gemmStridedBatched` instead of `gemmBatched` for better memory locality. For very small matrices, consider fused custom kernels.

> Source: cuBLAS API -- batched operations have per-element overhead that needs amortization.

---

## P5: Not Destroying Library Handles Causing Resource Leaks

**Symptom:** GPU memory usage grows over time; eventual allocation failures.

**Detection:** `nvidia-smi` shows increasing memory usage. `cudaMalloc` failures after extended operation.

**Fix:** Destroy all library handles (`cublasDestroy`, `cublasLtDestroy`) during cleanup.

```cpp
cublasDestroy(handle);
cublasLtDestroy(ltHandle);
```

> Source: cuBLAS API -- resource management best practice.

## P6: Optimized kernel shows 0% L1 hit rate and only 23% warp occupancy (down from 91%), with register usage jumping from 32 to 80 per thread — the cuBLAS batched GEMM kernel trades occupancy for register-heavy computation, which works here but could become a bottleneck at larger batch counts or on register-constrained architectures (discovered in verification)

**Symptom**: Optimized kernel shows 0% L1 hit rate and only 23% warp occupancy (down from 91%), with register usage jumping from 32 to 80 per thread — the cuBLAS batched GEMM kernel trades occupancy for register-heavy computation, which works here but could become a bottleneck at larger batch counts or on register-constrained architectures.
**Source**: Level 3 sandbox verification (2026-04-06)

## P7: cublasSetStream/cublasSetWorkspace concurrent execution benefits require sufficiently large and independent workloads; on small matrices the stream management overhead causes net regression across all utilization metrics (discovered in verification)

**Symptom**: cublasSetStream/cublasSetWorkspace concurrent execution benefits require sufficiently large and independent workloads; on small matrices the stream management overhead causes net regression across all utilization metrics
**Source**: Level 3 sandbox verification (2026-04-06)

## P8: Thrust's optimized kernel uses 2x registers (32 vs 16) and 5x shared memory per block, which reduced occupancy limit from 32 to 26 blocks by shared memory — could become a problem on smaller GPUs or with larger data sizes (discovered in verification)

**Symptom**: Thrust's optimized kernel uses 2x registers (32 vs 16) and 5x shared memory per block, which reduced occupancy limit from 32 to 26 blocks by shared memory — could become a problem on smaller GPUs or with larger data sizes.
**Source**: Level 3 sandbox verification (2026-04-06)

## P9: cublasSetSmCountTarget can cause cuBLAS to pick a fundamentally different (and less efficient) kernel variant, not just run the same kernel on fewer SMs — the performance hit may be larger than a simple SM-proportional slowdown (discovered in verification)

**Symptom**: cublasSetSmCountTarget can cause cuBLAS to pick a fundamentally different (and less efficient) kernel variant, not just run the same kernel on fewer SMs — the performance hit may be larger than a simple SM-proportional slowdown
**Source**: Level 3 sandbox verification (2026-04-06)
