# Gpu Library Usage -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cublas<t>gemmBatched()` | cuBLAS | Batch GEMM via array of pointers |
| `cublas<t>gemmGroupedBatched()` | cuBLAS | Grouped batch GEMM (heterogeneous sizes per group) |
| `cublas<t>gemmStridedBatched()` | cuBLAS | Batch GEMM via strided memory layout |
| `cublas<t>gemvBatched()` | cuBLAS | Batched GEMV (array of pointers) |
| `cublas<t>gemvStridedBatched()` | cuBLAS | Strided batched GEMV |
| `cublasGemmBatchedEx()` | cuBLAS | Batched GemmEx with mixed types + algo selection |
| `cublasGemmGroupedBatchedEx()` | cuBLAS | Grouped batched GemmEx (heterogeneous sizes) with mixed types |
| `cublasGemmStridedBatchedEx()` | cuBLAS | Strided batched GemmEx with mixed types + algo selection |
| `cublasSetWorkspace()` | cuBLAS | Set user-managed workspace |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cublas<t>trsmBatched()` | cuBLAS | Batched triangular solve |
| `cublasCreate()` | cuBLAS | Create cuBLAS handle |
| `cublasDestroy()` | cuBLAS | Destroy cuBLAS handle |
| `cublasLtCreate()` | cuBLAS | Create cuBLASLt handle |
| `cublasLtDestroy()` | cuBLAS | Destroy cuBLASLt handle |
| `cublasLtGroupedMatrixLayoutCreate()` | cuBLAS | Create grouped matrix layout for grouped GEMM |
| `cublasLtHeuristicsCacheSetCapacity()` | cuBLAS | Set heuristics cache capacity |
| `cublasSetPointerMode()` | cuBLAS | Set host/device pointer mode for scalars |
