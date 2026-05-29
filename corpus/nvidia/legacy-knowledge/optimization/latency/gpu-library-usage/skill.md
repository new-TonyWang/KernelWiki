# GPU Library Usage -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 6.1 (Parallel Libraries)"
  - "Colfax Blog: Tutorial -- Python bindings for CUDA libraries in PyTorch"
cross_ref:
  - optimization/latency/stream-concurrency
  - optimization/compute/tensor-core
  - optimization/compute/operator-fusion
related_apis:
  - cublasSetStream
  - cublasSetWorkspace
  - "cublas<t>gemmBatched"
  - "cublas<t>gemmStridedBatched"
  - cublasGemmBatchedEx
  - cublasGemmGroupedBatchedEx
unlocks:
  - Optimized implementations without hand-tuned kernels
  - Automatic hardware-specific tuning (tensor cores, etc.)
conflicts_with:
  - None
```

---

## S1: Use cuBLAS for GEMM Instead of Hand-Written Kernels

**When to Use:** For any matrix multiplication workload. cuBLAS is highly optimized for each GPU architecture and automatically uses tensor cores when available.

**How to Apply:**
1. Create a cuBLAS handle.
2. Set the stream and workspace.
3. Call the appropriate GEMM variant (single, batched, strided batched, or grouped).

**Code Template:**
```cpp
cublasHandle_t handle;
cublasCreate(&handle);
cublasSetStream(handle, stream);

// Allocate workspace for best performance
void* workspace;
cudaMalloc(&workspace, workspaceSize);
cublasSetWorkspace(handle, workspace, workspaceSize);

// Standard GEMM: C = alpha * A * B + beta * C
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, A, M, B, K, &beta, C, M);
```

> Source: BP 6.1 -- "Applications already using other BLAS libraries can often quite easily switch to cuBLAS."

---

## S2: Use Batched GEMM for Many Small Matrices

**When to Use:** When processing many independent small matrix multiplications (e.g., batched attention, per-sample transforms).

**How to Apply:**
1. Choose between pointer-based (`gemmBatched`) and strided (`gemmStridedBatched`) variants.
2. Use `GemmBatchedEx` for mixed precision with algorithm selection.
3. For heterogeneous sizes, use `gemmGroupedBatched` (or `GemmGroupedBatchedEx`).

**Code Template:**
```cpp
// Strided batched GEMM (uniform sizes, contiguous memory)
cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K, &alpha,
    A, M, strideA,
    B, K, strideB, &beta,
    C, M, strideC,
    batchCount);

// Grouped batched GEMM (heterogeneous sizes)
cublasGemmGroupedBatchedEx(handle,
    transA_array, transB_array,
    M_array, N_array, K_array,
    alpha_array,
    A_array, Atype, lda_array,
    B_array, Btype, ldb_array,
    beta_array,
    C_array, Ctype, ldc_array,
    groupCount, groupSize_array,
    computeType, algo);
```

> Source: cuBLAS API -- batched and grouped GEMM variants.

---

## S3: Set cuBLAS Stream and Workspace for Concurrent Execution

**When to Use:** When cuBLAS calls need to run concurrently with other work on the GPU.

**How to Apply:**
1. Associate the cuBLAS handle with a specific CUDA stream using `cublasSetStream`.
2. Provide a pre-allocated workspace with `cublasSetWorkspace` to avoid internal synchronization.
3. Use separate handles or streams for independent cuBLAS calls.

**Code Template:**
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cublasHandle_t handle;
cublasCreate(&handle);

// Launch GEMM in stream1
cublasSetStream(handle, stream1);
cublasSgemm(handle, ...);

// Launch another GEMM in stream2
cublasSetStream(handle, stream2);
cublasSgemm(handle, ...);
```

> Source: cuBLAS API -- cublasSetStream and cublasSetWorkspace.

---

## S4: Use Thrust for Common Parallel Primitives

**When to Use:** For sort, reduce, scan, and other data-parallel operations where a hand-written kernel would replicate well-known algorithms.

**How to Apply:**
1. Include Thrust headers.
2. Use `thrust::device_vector` or raw pointers with execution policies.
3. Thrust automatically selects efficient implementations.

**Code Template:**
```cpp
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

thrust::device_vector<float> d_vec(h_vec.begin(), h_vec.end());
thrust::sort(d_vec.begin(), d_vec.end());
float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
```

> Source: BP 6.1 -- "Thrust provides a rich collection of data parallel primitives such as scan, sort, and reduce."

---

## S5: Limit SM Count for cuBLAS to Leave Room for Other Work

**When to Use:** When cuBLAS operations should not monopolize all SMs, leaving room for concurrent kernels.

**How to Apply:**
1. Use `cublasSetSmCountTarget` to limit the number of SMs used by cuBLAS.
2. This is particularly useful when combining cuBLAS with custom kernels in separate streams.

**Code Template:**
```cpp
// Limit cuBLAS to use at most 50% of SMs
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
cublasSetSmCountTarget(handle, numSMs / 2);
```

> Source: cuBLAS API -- cublasSetSmCountTarget.

---

## Cascading Opportunities

- cuBLAS GEMM automatically uses `tensor-core` when precision settings allow.
- Library calls benefit from `stream-concurrency` for overlapping with other work.
- cuBLASLt provides `operator-fusion` capabilities via epilogue functions.

## Conflicts

- None. Library usage is generally the safest starting point.

## Principles

1. **Libraries first, custom kernels second:** Only write custom kernels when libraries do not meet specific needs (BP 6.1).
2. **Interface familiarity:** cuBLAS mirrors BLAS, cuFFT mirrors FFTW -- migration is straightforward (BP 6.1).
3. **Architecture-specific tuning:** Libraries are tuned per GPU generation, so they automatically benefit from new hardware features.

## Open Questions

- Q1: When does a custom CUTLASS GEMM outperform cuBLAS, and by how much?
- Q2: What is the overhead of cuBLAS handle creation and workspace allocation?
