# Cublas Integration -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `CUBLASLT_EPILOGUE_BIAS` | cuBLAS | Add broadcast bias vector |
| `CUBLASLT_EPILOGUE_DEFAULT` | cuBLAS | No epilogue; scale/quantize only |
| `CUBLASLT_EPILOGUE_GELU` | cuBLAS | Apply GELU activation |
| `CUBLASLT_EPILOGUE_GELU_AUX` | cuBLAS | GELU + save pre-activation (for backward) |
| `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` | cuBLAS | Bias + GELU + save pre-activation |
| `CUBLASLT_EPILOGUE_GELU_BIAS` | cuBLAS | Bias + GELU |
| `CUBLASLT_EPILOGUE_RELU` | cuBLAS | Apply ReLU: x = max(x, 0) |
| `CUBLASLT_EPILOGUE_RELU_AUX` | cuBLAS | ReLU + save bitmask (for backward pass) |
| `CUBLASLT_EPILOGUE_RELU_AUX_BIAS` | cuBLAS | Bias + ReLU + save bitmask |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | cuBLAS | Bias + ReLU |
| `cublas<t>gemm()` | cuBLAS | C = alpha*op(A)*op(B) + beta*C |
| `cublas<t>gemmBatched()` | cuBLAS | Batch GEMM via array of pointers |
| `cublas<t>gemmEx()` | cuBLAS | GEMM with lower-precision inputs, higher-precision compute |
| `cublas<t>gemmGroupedBatched()` | cuBLAS | Grouped batch GEMM (heterogeneous sizes per group) |
| `cublas<t>gemmStridedBatched()` | cuBLAS | Batch GEMM via strided memory layout |
| `cublas<t>gemv()` | cuBLAS | y = alpha*op(A)*x + beta*y |
| `cublas<t>gemvBatched()` | cuBLAS | Batched GEMV (array of pointers) |
| `cublas<t>gemvStridedBatched()` | cuBLAS | Strided batched GEMV |
| `cublasGemmBatchedEx()` | cuBLAS | Batched GemmEx with mixed types + algo selection |
| `cublasGemmEx()` | cuBLAS | Fully flexible GEMM: individual data types + algorithm selection |
| `cublasGemmGroupedBatchedEx()` | cuBLAS | Grouped batched GemmEx (heterogeneous sizes) with mixed types |
| `cublasGemmStridedBatchedEx()` | cuBLAS | Strided batched GemmEx with mixed types + algo selection |
| `cublasHgemm()` | cuBLAS | Half-precision GEMM (included in above) |
| `cublasLtMatmul()` | cuBLAS | D = alpha*op(A)*op(B) + beta*C with epilogue fusion |
| `cublasLtMatmulAlgoCapGetAttribute()` | cuBLAS | Query algorithm capabilities (tile size, stages, math mode) |
| `cublasLtMatmulAlgoConfigSetAttribute()` | cuBLAS | Set algorithm configuration (tile, stages, splitK, etc.) |
| `cublasLtMatmulAlgoGetHeuristic()` | cuBLAS | Get top algorithm recommendations for a GEMM problem |
| `cublasLtMatmulAlgoGetIds()` | cuBLAS | Get all supported algorithm IDs for a problem |
| `cublasLtMatmulDescCreate()` | cuBLAS | Create matmul operation descriptor |
| `cublasLtMatmulDescSetAttribute()` | cuBLAS | Set matmul descriptor attributes (epilogue, bias, pointer mode) |
| `cublasLtMatmulPreferenceCreate()` | cuBLAS | Create preference descriptor for heuristic queries |
| `cublasLtMatmulPreferenceSetAttribute()` | cuBLAS | Set workspace size, reduction scheme, numerical impl flags |
| `cublasLtMatrixLayoutCreate()` | cuBLAS | Create matrix layout descriptor |
| `cublasLtMatrixLayoutSetAttribute()` | cuBLAS | Set matrix layout attributes (batch count, stride, order) |
| `cublasSetWorkspace()` | cuBLAS | Set user-managed workspace |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `CUBLASLT_EPILOGUE_BGRADA` | cuBLAS | Bias gradient over A rows |
| `CUBLASLT_EPILOGUE_BGRADB` | cuBLAS | Bias gradient over B columns |
| `CUBLASLT_EPILOGUE_DGELU` | cuBLAS | GELU gradient (backward) |
| `CUBLASLT_EPILOGUE_DGELU_BGRAD` | cuBLAS | GELU gradient + bias gradient |
| `CUBLASLT_EPILOGUE_DRELU` | cuBLAS | ReLU gradient (backward) |
| `CUBLASLT_EPILOGUE_DRELU_BGRAD` | cuBLAS | ReLU gradient + bias gradient |
| `cublas<t>gemm3m()` | cuBLAS | Complex GEMM with Gauss reduction (3 real muls instead of 4) |
| `cublas<t>symm()` | cuBLAS | C = alpha*A*B + beta*C (A symmetric) |
| `cublas<t>syrk()` | cuBLAS | C = alpha*A*A^T + beta*C (symmetric rank-k) |
| `cublas<t>trmm()` | cuBLAS | B = alpha*op(A)*B (A triangular) |
| `cublas<t>trsm()` | cuBLAS | Triangular solve (multiple RHS) |
| `cublas<t>trsmBatched()` | cuBLAS | Batched triangular solve |
| `cublas<t>trsv()` | cuBLAS | Triangular solve (single RHS) |
| `cublasLtGroupedMatrixLayoutCreate()` | cuBLAS | Create grouped matrix layout for grouped GEMM |
| `cublasLtGroupedMatrixLayoutInit()` | cuBLAS | Initialize pre-allocated grouped matrix layout |
| `cublasLtMatmulAlgoCheck()` | cuBLAS | Validate algorithm for a specific problem |
| `cublasLtMatmulAlgoConfigGetAttribute()` | cuBLAS | Query algorithm configuration |
| `cublasLtMatmulAlgoInit()` | cuBLAS | Initialize algorithm from ID |
| `cublasLtMatmulDescDestroy()` | cuBLAS | Destroy matmul descriptor |
| `cublasLtMatmulDescGetAttribute()` | cuBLAS | Query matmul descriptor attributes (epilogue, pointer mode, etc.) |
| `cublasLtMatmulDescInit()` | cuBLAS | Initialize pre-allocated matmul descriptor |
| `cublasLtMatmulPreferenceDestroy()` | cuBLAS | Destroy preference descriptor |
| `cublasLtMatmulPreferenceGetAttribute()` | cuBLAS | Query preference attributes |
| `cublasLtMatmulPreferenceInit()` | cuBLAS | Initialize pre-allocated preference |
| `cublasLtMatrixLayoutDestroy()` | cuBLAS | Destroy matrix layout |
| `cublasLtMatrixLayoutGetAttribute()` | cuBLAS | Query matrix layout attributes |
| `cublasLtMatrixLayoutInit()` | cuBLAS | Initialize pre-allocated matrix layout |
| `cublasLtMatrixTransform()` | cuBLAS | Matrix layout transformation (transpose, type conversion) |
| `cublasLtMatrixTransformDescCreate()` | cuBLAS | Create transform descriptor |
| `cublasLtMatrixTransformDescSetAttribute()` | cuBLAS | Set transform attributes (transpose type) |
| `cublasSetAtomicsMode()` | cuBLAS | Allow/disallow atomics for reproducibility |
| `cublasXt<t>gemm()` | cuBLAS | Multi-GPU GEMM |
