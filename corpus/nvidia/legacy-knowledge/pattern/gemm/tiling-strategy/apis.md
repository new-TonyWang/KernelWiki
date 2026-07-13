# Tiling Strategy -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cublas<t>gemm()` | cuBLAS | C = alpha*op(A)*op(B) + beta*C |
| `cublas<t>gemmStridedBatched()` | cuBLAS | Batch GEMM via strided memory layout |
| `cublasGemmEx()` | cuBLAS | Fully flexible GEMM: individual data types + algorithm selection |
| `cublasGemmStridedBatchedEx()` | cuBLAS | Strided batched GemmEx with mixed types + algo selection |
| `cublasLtMatmul()` | cuBLAS | D = alpha*op(A)*op(B) + beta*C with epilogue fusion |
| `cublasLtMatmulAlgoCapGetAttribute()` | cuBLAS | Query algorithm capabilities (tile size, stages, math mode) |
| `cublasLtMatmulAlgoConfigSetAttribute()` | cuBLAS | Set algorithm configuration (tile, stages, splitK, etc.) |
| `cublasLtMatmulAlgoGetHeuristic()` | cuBLAS | Get top algorithm recommendations for a GEMM problem |
| `cublasLtMatmulAlgoGetIds()` | cuBLAS | Get all supported algorithm IDs for a problem |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cublasLtMatmulAlgoConfigGetAttribute()` | cuBLAS | Query algorithm configuration |
| `cublasXt<t>gemm()` | cuBLAS | Multi-GPU GEMM |
| `cublasXtSetBlockDim()` | cuBLAS | Set tiling block dimension |
