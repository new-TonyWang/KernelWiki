# cuBLAS API Index

Source: cuBLAS 13.2 documentation (CUDA Toolkit)

This index covers the cuBLAS, cuBLAS Extensions, and cuBLASLt APIs with mappings to the
kernel optimization knowledge tree. Each API is classified as:
- **core**: Frequently used in kernel optimization decisions; directly relevant to performance-critical paths
- **related**: Useful context for optimization; may appear in real workloads
- **low-relevance**: Rarely relevant to kernel optimization (specialized linear algebra routines)

Data type abbreviations: S=float, D=double, C=cuComplex, Z=cuDoubleComplex, H=half(__half), BF=bfloat16

---

## BLAS Level 1 -- Vector Operations

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasI<t>amax()` | Index of max absolute value element | S, D, C, Z | pattern/reduction | related |
| `cublasI<t>amin()` | Index of min absolute value element | S, D, C, Z | pattern/reduction | related |
| `cublas<t>asum()` | Sum of absolute values | S, D, Sc, Dz | pattern/reduction | related |
| `cublas<t>axpy()` | y = alpha*x + y | S, D, C, Z | pattern/elementwise | core |
| `cublas<t>copy()` | y = x (vector copy) | S, D, C, Z | pattern/elementwise | related |
| `cublas<t>dot()` | Dot product of two vectors | S, D, C(u/c), Z(u/c) | pattern/reduction | core |
| `cublas<t>nrm2()` | Euclidean norm of vector | S, D, Sc, Dz | pattern/reduction | core |
| `cublas<t>rot()` | Apply Givens rotation | S, D, C, Cs, Z, Zd | pattern/elementwise | low-relevance |
| `cublas<t>rotg()` | Construct Givens rotation | S, D, C, Z | pattern/elementwise | low-relevance |
| `cublas<t>rotm()` | Apply modified Givens rotation | S, D | pattern/elementwise | low-relevance |
| `cublas<t>rotmg()` | Construct modified Givens rotation | S, D | pattern/elementwise | low-relevance |
| `cublas<t>scal()` | x = alpha*x (vector scale) | S, D, C, Cs, Z, Zd | pattern/elementwise | core |
| `cublas<t>swap()` | Swap vectors x and y | S, D, C, Z | pattern/elementwise | low-relevance |

### BLAS Level 1 -- Extended Precision (BLAS-like Extension)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasNrm2Ex()` | Euclidean norm with mixed precision | FP16, BF16, FP32, FP64 (via cudaDataType) | pattern/reduction, optimization/compute/half-precision-math | core |
| `cublasAxpyEx()` | y = alpha*x + y with mixed precision | FP16, BF16, FP32, FP64 (via cudaDataType) | pattern/elementwise, optimization/compute/half-precision-math | core |
| `cublasDotEx()` | Dot product with mixed precision | FP16, BF16, FP32, FP64 (via cudaDataType) | pattern/reduction, optimization/compute/half-precision-math | core |
| `cublasRotEx()` | Apply Givens rotation with mixed precision | FP16, BF16, FP32, FP64 (via cudaDataType) | pattern/elementwise | low-relevance |
| `cublasScalEx()` | Vector scale with mixed precision | FP16, BF16, FP32, FP64 (via cudaDataType) | pattern/elementwise, optimization/compute/half-precision-math | related |

---

## BLAS Level 2 -- Matrix-Vector Operations

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublas<t>gemv()` | y = alpha*op(A)*x + beta*y | S, D, C, Z | pattern/gemm/cublas-integration | core |
| `cublas<t>gemvBatched()` | Batched GEMV (array of pointers) | S, D, C, Z (also H, BF16 via HsS, BfSs) | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage | core |
| `cublas<t>gemvStridedBatched()` | Strided batched GEMV | S, D, C, Z (also H, BF16 via HsS, BfSs) | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage | core |
| `cublas<t>gbmv()` | Banded matrix-vector multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>symv()` | Symmetric matrix-vector multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>hemv()` | Hermitian matrix-vector multiply | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>sbmv()` | Symmetric banded matrix-vector multiply | S, D | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>hbmv()` | Hermitian banded matrix-vector multiply | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>spmv()` | Symmetric packed matrix-vector multiply | S, D | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>hpmv()` | Hermitian packed matrix-vector multiply | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>trmv()` | Triangular matrix-vector multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>tbmv()` | Triangular banded matrix-vector multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>tpmv()` | Triangular packed matrix-vector multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>trsv()` | Triangular solve (single RHS) | S, D, C, Z | pattern/gemm/cublas-integration | related |
| `cublas<t>tbsv()` | Triangular banded solve | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>tpsv()` | Triangular packed solve | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>ger()` | Rank-1 update: A = alpha*x*y^T + A | S, D, C(u/c), Z(u/c) | pattern/elementwise | low-relevance |
| `cublas<t>syr()` | Symmetric rank-1 update | S, D, C, Z | pattern/elementwise | low-relevance |
| `cublas<t>her()` | Hermitian rank-1 update | C, Z | pattern/elementwise | low-relevance |
| `cublas<t>spr()` | Symmetric packed rank-1 update | S, D | pattern/elementwise | low-relevance |
| `cublas<t>hpr()` | Hermitian packed rank-1 update | C, Z | pattern/elementwise | low-relevance |
| `cublas<t>syr2()` | Symmetric rank-2 update | S, D, C, Z | pattern/elementwise | low-relevance |
| `cublas<t>her2()` | Hermitian rank-2 update | C, Z | pattern/elementwise | low-relevance |
| `cublas<t>spr2()` | Symmetric packed rank-2 update | S, D | pattern/elementwise | low-relevance |
| `cublas<t>hpr2()` | Hermitian packed rank-2 update | C, Z | pattern/elementwise | low-relevance |

---

## BLAS Level 3 -- Matrix-Matrix Operations

### Standard GEMM (Most Important for Kernel Optimization)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublas<t>gemm()` | C = alpha*op(A)*op(B) + beta*C | S, D, C, Z, **H** | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, optimization/compute/tensor-core | **core** |
| `cublasHgemm()` | Half-precision GEMM (included in above) | H (__half) | pattern/gemm/cublas-integration, optimization/compute/tensor-core, optimization/compute/half-precision-math | **core** |
| `cublas<t>gemm3m()` | Complex GEMM with Gauss reduction (3 real muls instead of 4) | C, Z | pattern/gemm/cublas-integration | related |

### Batched GEMM (Critical for Operator Fusion and Throughput)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublas<t>gemmBatched()` | Batch GEMM via array of pointers | S, D, C, Z, H | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage, optimization/latency/stream-concurrency | **core** |
| `cublas<t>gemmStridedBatched()` | Batch GEMM via strided memory layout | S, D, C, Z, H (also Cgemm3mStridedBatched) | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, optimization/latency/gpu-library-usage | **core** |
| `cublas<t>gemmGroupedBatched()` | Grouped batch GEMM (heterogeneous sizes per group) | S, D, C, Z | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage | **core** |

### Other Level-3 Operations

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublas<t>symm()` | C = alpha*A*B + beta*C (A symmetric) | S, D, C, Z | pattern/gemm/cublas-integration | related |
| `cublas<t>hemm()` | C = alpha*A*B + beta*C (A Hermitian) | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>syrk()` | C = alpha*A*A^T + beta*C (symmetric rank-k) | S, D, C, Z | pattern/gemm/cublas-integration | related |
| `cublas<t>syr2k()` | C = alpha*(A*B^T + B*A^T) + beta*C | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>syrkx()` | Generalized symmetric rank-k update | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>herk()` | Hermitian rank-k update | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>her2k()` | Hermitian rank-2k update | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>herkx()` | Generalized Hermitian rank-k update | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>trmm()` | B = alpha*op(A)*B (A triangular) | S, D, C, Z | pattern/gemm/cublas-integration | related |
| `cublas<t>trsm()` | Triangular solve (multiple RHS) | S, D, C, Z | pattern/gemm/cublas-integration | related |
| `cublas<t>trsmBatched()` | Batched triangular solve | S, D, C, Z | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage | related |

---

## BLAS-like Extensions -- Mixed-Precision and Extended GEMM

### GemmEx Family (Key for Mixed Precision and Tensor Core Usage)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublas<t>gemmEx()` | GEMM with lower-precision inputs, higher-precision compute | A/B: FP16, BF16, INT8; C: FP16, BF16, FP32; compute: FP32, cuComplex | pattern/gemm/cublas-integration, optimization/compute/tensor-core, optimization/compute/half-precision-math | **core** |
| `cublasGemmEx()` | Fully flexible GEMM: individual data types + algorithm selection | FP16, BF16, FP32, FP64, INT8, FP8 (E4M3/E5M2); compute: FP16, FP32, FP64, INT32, TF32 | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, optimization/compute/tensor-core, optimization/compute/half-precision-math | **core** |
| `cublasGemmBatchedEx()` | Batched GemmEx with mixed types + algo selection | Same as cublasGemmEx | pattern/gemm/cublas-integration, optimization/compute/tensor-core, optimization/latency/gpu-library-usage | **core** |
| `cublasGemmStridedBatchedEx()` | Strided batched GemmEx with mixed types + algo selection | Same as cublasGemmEx | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, optimization/compute/tensor-core, optimization/latency/gpu-library-usage | **core** |
| `cublasGemmGroupedBatchedEx()` | Grouped batched GemmEx (heterogeneous sizes) with mixed types | Same as cublasGemmEx (no algo param) | pattern/gemm/cublas-integration, optimization/compute/tensor-core, optimization/latency/gpu-library-usage | **core** |

### Other BLAS-like Extensions

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublas<t>geam()` | C = alpha*op(A) + beta*op(B) (matrix add/transpose) | S, D, C, Z | pattern/elementwise | related |
| `cublas<t>dgmm()` | C = A * diag(x) or diag(x) * A | S, D, C, Z | pattern/elementwise | related |
| `cublasCsyrkEx()` | Symmetric rank-k update with mixed precision | A: FP16/BF16, C: FP32/cuComplex; compute: FP32 | pattern/gemm/cublas-integration, optimization/compute/half-precision-math | low-relevance |
| `cublasCsyrk3mEx()` | Symmetric rank-k with Gauss reduction + mixed precision | A: FP16/BF16, C: cuComplex; compute: cuComplex | pattern/gemm/cublas-integration | low-relevance |
| `cublasCherkEx()` | Hermitian rank-k update with mixed precision | A: FP16/BF16, C: cuComplex; compute: FP32 | pattern/gemm/cublas-integration, optimization/compute/half-precision-math | low-relevance |
| `cublasCherk3mEx()` | Hermitian rank-k with Gauss reduction + mixed precision | A: FP16/BF16, C: cuComplex; compute: cuComplex | pattern/gemm/cublas-integration | low-relevance |
| `cublas<t>getrfBatched()` | Batched LU factorization | S, D, C, Z | optimization/latency/gpu-library-usage | low-relevance |
| `cublas<t>getrsBatched()` | Batched LU solve | S, D, C, Z | optimization/latency/gpu-library-usage | low-relevance |
| `cublas<t>getriBatched()` | Batched matrix inverse via LU | S, D, C, Z | optimization/latency/gpu-library-usage | low-relevance |
| `cublas<t>matinvBatched()` | Batched matrix inverse (small matrices) | S, D, C, Z | optimization/latency/gpu-library-usage | low-relevance |
| `cublas<t>geqrfBatched()` | Batched QR factorization | S, D, C, Z | optimization/latency/gpu-library-usage | low-relevance |
| `cublas<t>gelsBatched()` | Batched least squares via QR | S, D, C, Z | optimization/latency/gpu-library-usage | low-relevance |
| `cublas<t>tpttr()` | Triangular packed to full format | S, D, C, Z | (utility) | low-relevance |
| `cublas<t>trttp()` | Triangular full to packed format | S, D, C, Z | (utility) | low-relevance |

---

## cuBLASLt API -- Lightweight GEMM with Advanced Configuration

cuBLASLt is a lightweight library dedicated to GEMM with a flexible API providing control over
data layouts, compute types, algorithm selection, workspace management, and epilogue fusion.
This is the **primary API for maximum GEMM performance tuning**.

### Core Compute Functions

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtMatmul()` | D = alpha*op(A)*op(B) + beta*C with epilogue fusion | FP16, BF16, FP32, FP64, INT8, FP8(E4M3/E5M2), TF32; compute: FP16, FP32, FP64, INT32 | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, pattern/gemm/warp-specialization, pattern/gemm/pipeline-overlap, optimization/compute/tensor-core, optimization/compute/half-precision-math | **core** |
| `cublasLtMatrixTransform()` | Matrix layout transformation (transpose, type conversion) | All supported types | pattern/gemm/cublas-integration | related |

### Matmul Descriptor Management

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtMatmulDescCreate()` | Create matmul operation descriptor | N/A | pattern/gemm/cublas-integration | **core** |
| `cublasLtMatmulDescInit()` | Initialize pre-allocated matmul descriptor | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulDescDestroy()` | Destroy matmul descriptor | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulDescGetAttribute()` | Query matmul descriptor attributes (epilogue, pointer mode, etc.) | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulDescSetAttribute()` | Set matmul descriptor attributes (epilogue, bias, pointer mode) | N/A | pattern/gemm/cublas-integration | **core** |

### Matrix Layout Management

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtMatrixLayoutCreate()` | Create matrix layout descriptor | N/A | pattern/gemm/cublas-integration | **core** |
| `cublasLtMatrixLayoutInit()` | Initialize pre-allocated matrix layout | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtGroupedMatrixLayoutCreate()` | Create grouped matrix layout for grouped GEMM | N/A | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage | related |
| `cublasLtGroupedMatrixLayoutInit()` | Initialize pre-allocated grouped matrix layout | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatrixLayoutDestroy()` | Destroy matrix layout | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatrixLayoutGetAttribute()` | Query matrix layout attributes | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatrixLayoutSetAttribute()` | Set matrix layout attributes (batch count, stride, order) | N/A | pattern/gemm/cublas-integration | **core** |

### Algorithm Selection and Heuristics (Critical for Performance Tuning)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtMatmulAlgoGetHeuristic()` | Get top algorithm recommendations for a GEMM problem | N/A | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy | **core** |
| `cublasLtMatmulAlgoGetIds()` | Get all supported algorithm IDs for a problem | N/A | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy | **core** |
| `cublasLtMatmulAlgoInit()` | Initialize algorithm from ID | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulAlgoCheck()` | Validate algorithm for a specific problem | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulAlgoCapGetAttribute()` | Query algorithm capabilities (tile size, stages, math mode) | N/A | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, pattern/gemm/pipeline-overlap | **core** |
| `cublasLtMatmulAlgoConfigGetAttribute()` | Query algorithm configuration | N/A | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy | related |
| `cublasLtMatmulAlgoConfigSetAttribute()` | Set algorithm configuration (tile, stages, splitK, etc.) | N/A | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy, pattern/gemm/pipeline-overlap | **core** |

### Matmul Preference (Workspace and Constraints)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtMatmulPreferenceCreate()` | Create preference descriptor for heuristic queries | N/A | pattern/gemm/cublas-integration | **core** |
| `cublasLtMatmulPreferenceInit()` | Initialize pre-allocated preference | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulPreferenceDestroy()` | Destroy preference descriptor | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulPreferenceGetAttribute()` | Query preference attributes | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatmulPreferenceSetAttribute()` | Set workspace size, reduction scheme, numerical impl flags | N/A | pattern/gemm/cublas-integration | **core** |

### Matrix Transform Descriptors

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtMatrixTransformDescCreate()` | Create transform descriptor | N/A | pattern/gemm/cublas-integration | related |
| `cublasLtMatrixTransformDescInit()` | Initialize pre-allocated transform descriptor | N/A | pattern/gemm/cublas-integration | low-relevance |
| `cublasLtMatrixTransformDescDestroy()` | Destroy transform descriptor | N/A | pattern/gemm/cublas-integration | low-relevance |
| `cublasLtMatrixTransformDescGetAttribute()` | Query transform attributes | N/A | pattern/gemm/cublas-integration | low-relevance |
| `cublasLtMatrixTransformDescSetAttribute()` | Set transform attributes (transpose type) | N/A | pattern/gemm/cublas-integration | related |

### Handle and Utility Functions

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtCreate()` | Create cuBLASLt handle | N/A | optimization/latency/gpu-library-usage | related |
| `cublasLtDestroy()` | Destroy cuBLASLt handle | N/A | optimization/latency/gpu-library-usage | related |
| `cublasLtGetVersion()` | Get library version | N/A | (utility) | low-relevance |
| `cublasLtGetProperty()` | Get library property (major/minor/patch) | N/A | (utility) | low-relevance |
| `cublasLtGetStatusName()` | Get error status name string | N/A | (utility) | low-relevance |
| `cublasLtGetStatusString()` | Get error status description | N/A | (utility) | low-relevance |
| `cublasLtGetCudartVersion()` | Get CUDART version used at build time | N/A | (utility) | low-relevance |
| `cublasLtDisableCpuInstructionsSetMask()` | Disable specific CPU instruction sets | N/A | (utility) | low-relevance |
| `cublasLtHeuristicsCacheGetCapacity()` | Query heuristics cache capacity | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasLtHeuristicsCacheSetCapacity()` | Set heuristics cache capacity | N/A | optimization/latency/gpu-library-usage | related |

### Emulation Descriptors (CUDA 13.2+)

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtEmulationDescCreate()` | Create floating-point emulation descriptor | N/A | optimization/compute/half-precision-math | low-relevance |
| `cublasLtEmulationDescInit()` | Initialize emulation descriptor | N/A | optimization/compute/half-precision-math | low-relevance |
| `cublasLtEmulationDescDestroy()` | Destroy emulation descriptor | N/A | optimization/compute/half-precision-math | low-relevance |
| `cublasLtEmulationDescSetAttribute()` | Set emulation attributes | N/A | optimization/compute/half-precision-math | low-relevance |
| `cublasLtEmulationDescGetAttribute()` | Get emulation attributes | N/A | optimization/compute/half-precision-math | low-relevance |

### Logging Functions

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasLtLoggerSetCallback()` | Set logging callback | N/A | (utility) | low-relevance |
| `cublasLtLoggerSetFile()` | Set log output file | N/A | (utility) | low-relevance |
| `cublasLtLoggerOpenFile()` | Open log file | N/A | (utility) | low-relevance |
| `cublasLtLoggerSetLevel()` | Set log level | N/A | (utility) | low-relevance |
| `cublasLtLoggerSetMask()` | Set log mask | N/A | (utility) | low-relevance |
| `cublasLtLoggerForceDisable()` | Force disable logging | N/A | (utility) | low-relevance |

---

## cuBLASLt Key Enumerations (Performance-Relevant)

### cublasLtEpilogue_t -- Epilogue Fusion Options

Epilogue fusion allows fusing post-GEMM operations directly into the GEMM kernel, avoiding
separate kernel launches and extra memory traffic. This is a key optimization for neural network
inference and training.

| Epilogue | Description | Knowledge Node | Relevance |
|----------|-------------|----------------|-----------|
| `CUBLASLT_EPILOGUE_DEFAULT` | No epilogue; scale/quantize only | pattern/gemm/cublas-integration | **core** |
| `CUBLASLT_EPILOGUE_RELU` | Apply ReLU: x = max(x, 0) | pattern/gemm/cublas-integration, pattern/elementwise | **core** |
| `CUBLASLT_EPILOGUE_RELU_AUX` | ReLU + save bitmask (for backward pass) | pattern/gemm/cublas-integration | **core** |
| `CUBLASLT_EPILOGUE_BIAS` | Add broadcast bias vector | pattern/gemm/cublas-integration, pattern/elementwise | **core** |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | Bias + ReLU | pattern/gemm/cublas-integration, pattern/elementwise | **core** |
| `CUBLASLT_EPILOGUE_RELU_AUX_BIAS` | Bias + ReLU + save bitmask | pattern/gemm/cublas-integration | **core** |
| `CUBLASLT_EPILOGUE_GELU` | Apply GELU activation | pattern/gemm/cublas-integration, pattern/elementwise | **core** |
| `CUBLASLT_EPILOGUE_GELU_AUX` | GELU + save pre-activation (for backward) | pattern/gemm/cublas-integration | **core** |
| `CUBLASLT_EPILOGUE_GELU_BIAS` | Bias + GELU | pattern/gemm/cublas-integration, pattern/elementwise | **core** |
| `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` | Bias + GELU + save pre-activation | pattern/gemm/cublas-integration | **core** |
| `CUBLASLT_EPILOGUE_DRELU` | ReLU gradient (backward) | pattern/gemm/cublas-integration | related |
| `CUBLASLT_EPILOGUE_DRELU_BGRAD` | ReLU gradient + bias gradient | pattern/gemm/cublas-integration | related |
| `CUBLASLT_EPILOGUE_DGELU` | GELU gradient (backward) | pattern/gemm/cublas-integration | related |
| `CUBLASLT_EPILOGUE_DGELU_BGRAD` | GELU gradient + bias gradient | pattern/gemm/cublas-integration | related |
| `CUBLASLT_EPILOGUE_BGRADA` | Bias gradient over A rows | pattern/gemm/cublas-integration, pattern/reduction | related |
| `CUBLASLT_EPILOGUE_BGRADB` | Bias gradient over B columns | pattern/gemm/cublas-integration, pattern/reduction | related |

### Key Compute Types for Tensor Core Usage

| Compute Type | Description | Tensor Core | Knowledge Node |
|-------------|-------------|-------------|----------------|
| `CUBLAS_COMPUTE_16F` | FP16 compute (uses Tensor Cores on Volta+) | Yes | optimization/compute/tensor-core, optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_32F` | FP32 compute (may use TF32 Tensor Cores on Ampere+) | Yes (TF32) | optimization/compute/tensor-core |
| `CUBLAS_COMPUTE_32F_FAST_16F` | FP32 with FP16 Tensor Core acceleration | Yes | optimization/compute/tensor-core, optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_32F_FAST_16BF` | FP32 with BF16 Tensor Core acceleration | Yes | optimization/compute/tensor-core, optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_32F_FAST_TF32` | FP32 with TF32 Tensor Core acceleration | Yes | optimization/compute/tensor-core |
| `CUBLAS_COMPUTE_64F` | FP64 compute (uses FP64 Tensor Cores on Ampere+) | Yes (A100+) | optimization/compute/tensor-core |
| `CUBLAS_COMPUTE_32I` | INT32 compute for INT8 inputs | Yes | optimization/compute/tensor-core |

---

## cuBLAS Helper Functions

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasCreate()` | Create cuBLAS handle | N/A | optimization/latency/gpu-library-usage | related |
| `cublasDestroy()` | Destroy cuBLAS handle | N/A | optimization/latency/gpu-library-usage | related |
| `cublasSetStream()` | Associate CUDA stream with handle | N/A | optimization/latency/stream-concurrency | **core** |
| `cublasGetStream()` | Get associated CUDA stream | N/A | optimization/latency/stream-concurrency | related |
| `cublasSetWorkspace()` | Set user-managed workspace | N/A | pattern/gemm/cublas-integration, optimization/latency/gpu-library-usage | **core** |
| `cublasSetMathMode()` | Enable/disable Tensor Core usage | N/A | optimization/compute/tensor-core | **core** |
| `cublasGetMathMode()` | Query current math mode | N/A | optimization/compute/tensor-core | related |
| `cublasSetSmCountTarget()` | Limit SM count for cuBLAS operations | N/A | optimization/latency/stream-concurrency | related |
| `cublasGetSmCountTarget()` | Query SM count target | N/A | optimization/latency/stream-concurrency | related |
| `cublasSetPointerMode()` | Set host/device pointer mode for scalars | N/A | optimization/latency/gpu-library-usage | related |
| `cublasGetPointerMode()` | Get current pointer mode | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasSetAtomicsMode()` | Allow/disallow atomics for reproducibility | N/A | pattern/gemm/cublas-integration | related |
| `cublasGetAtomicsMode()` | Query atomics mode | N/A | pattern/gemm/cublas-integration | low-relevance |
| `cublasSetVector()` | Copy vector from host to device | N/A | (utility) | low-relevance |
| `cublasGetVector()` | Copy vector from device to host | N/A | (utility) | low-relevance |
| `cublasSetMatrix()` | Copy matrix from host to device | N/A | (utility) | low-relevance |
| `cublasGetMatrix()` | Copy matrix from device to host | N/A | (utility) | low-relevance |
| `cublasSetVectorAsync()` | Async vector copy host to device | N/A | optimization/latency/stream-concurrency | low-relevance |
| `cublasGetVectorAsync()` | Async vector copy device to host | N/A | optimization/latency/stream-concurrency | low-relevance |
| `cublasSetMatrixAsync()` | Async matrix copy host to device | N/A | optimization/latency/stream-concurrency | low-relevance |
| `cublasGetMatrixAsync()` | Async matrix copy device to host | N/A | optimization/latency/stream-concurrency | low-relevance |
| `cublasGetVersion()` | Get library version | N/A | (utility) | low-relevance |
| `cublasGetProperty()` | Get library property | N/A | (utility) | low-relevance |
| `cublasGetStatusName()` | Get error name string | N/A | (utility) | low-relevance |
| `cublasGetStatusString()` | Get error description | N/A | (utility) | low-relevance |
| `cublasLoggerConfigure()` | Configure logging | N/A | (utility) | low-relevance |
| `cublasGetLoggerCallback()` | Get log callback | N/A | (utility) | low-relevance |
| `cublasSetLoggerCallback()` | Set log callback | N/A | (utility) | low-relevance |
| `cublasSetEmulationStrategy()` | Set FP emulation strategy | N/A | optimization/compute/half-precision-math | low-relevance |
| `cublasGetEmulationStrategy()` | Get FP emulation strategy | N/A | optimization/compute/half-precision-math | low-relevance |

---

## cuBLASXt API -- Multi-GPU GEMM

cuBLASXt dispatches operations across multiple GPUs with automatic data distribution.
Less relevant for single-GPU kernel optimization but included for completeness.

| API | Operation | Data Types | Knowledge Node | Relevance |
|-----|-----------|-----------|----------------|-----------|
| `cublasXtCreate()` | Create multi-GPU handle | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasXtDestroy()` | Destroy multi-GPU handle | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasXtDeviceSelect()` | Select GPUs for computation | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasXtSetBlockDim()` | Set tiling block dimension | N/A | pattern/gemm/tiling-strategy | related |
| `cublasXtGetBlockDim()` | Get tiling block dimension | N/A | pattern/gemm/tiling-strategy | low-relevance |
| `cublasXt<t>gemm()` | Multi-GPU GEMM | S, D, C, Z | pattern/gemm/cublas-integration, pattern/gemm/tiling-strategy | related |
| `cublasXt<t>hemm()` | Multi-GPU Hermitian matrix multiply | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>symm()` | Multi-GPU symmetric matrix multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>syrk()` | Multi-GPU symmetric rank-k | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>syr2k()` | Multi-GPU symmetric rank-2k | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>syrkx()` | Multi-GPU generalized symmetric rank-k | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>herk()` | Multi-GPU Hermitian rank-k | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>her2k()` | Multi-GPU Hermitian rank-2k | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>herkx()` | Multi-GPU generalized Hermitian rank-k | C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>trsm()` | Multi-GPU triangular solve | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>trmm()` | Multi-GPU triangular matrix multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXt<t>spmm()` | Multi-GPU symmetric packed matrix multiply | S, D, C, Z | pattern/gemm/cublas-integration | low-relevance |
| `cublasXtSetCpuRoutine()` | Set CPU fallback routine | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasXtSetCpuRatio()` | Set CPU/GPU work ratio | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasXtSetPinningMemMode()` | Set pinned memory mode | N/A | optimization/latency/gpu-library-usage | low-relevance |
| `cublasXtGetPinningMemMode()` | Get pinned memory mode | N/A | optimization/latency/gpu-library-usage | low-relevance |

---

## Summary

### API Counts by Category

| Category | Total APIs | Core | Related | Low-Relevance |
|----------|-----------|------|---------|---------------|
| BLAS Level 1 | 13 | 4 | 3 | 6 |
| BLAS Level 1 Extensions | 5 | 3 | 1 | 1 |
| BLAS Level 2 | 25 | 3 | 1 | 21 |
| BLAS Level 3 (GEMM) | 6 | 5 | 1 | 0 |
| BLAS Level 3 (Other) | 11 | 0 | 4 | 7 |
| BLAS-like Extensions (GemmEx) | 5 | 5 | 0 | 0 |
| BLAS-like Extensions (Other) | 12 | 0 | 2 | 10 |
| cuBLASLt Core Compute | 2 | 1 | 1 | 0 |
| cuBLASLt Descriptors | 12 | 4 | 6 | 2 |
| cuBLASLt Algorithm Selection | 7 | 4 | 3 | 0 |
| cuBLASLt Preferences | 5 | 2 | 2 | 1 |
| cuBLASLt Matrix Transform | 5 | 0 | 2 | 3 |
| cuBLASLt Handle/Utility | 10 | 0 | 2 | 8 |
| cuBLASLt Emulation | 5 | 0 | 0 | 5 |
| cuBLASLt Logging | 6 | 0 | 0 | 6 |
| cuBLAS Helpers | 27 | 3 | 7 | 17 |
| cuBLASXt | 20 | 0 | 2 | 18 |
| Epilogue Options | 16 | 10 | 6 | 0 |
| **Total** | **192** | **44** | **43** | **105** |

### Key Observations for Kernel Optimization

1. **GEMM is the dominant workload**: The GEMM family (`cublas<t>gemm`, `cublasGemmEx`,
   `cublasLtMatmul`) represents the most performance-critical APIs. `cublasLtMatmul` provides
   the finest-grained control over algorithm selection, tiling, and epilogue fusion.

2. **Tensor Core acceleration**: Available through `cublasGemmEx` (via algorithm selection),
   `cublasLtMatmul`, and `cublasSetMathMode()`. Key precision modes include FP16, BF16, TF32,
   FP8 (E4M3/E5M2), and INT8.

3. **Epilogue fusion in cuBLASLt**: `cublasLtMatmul` can fuse bias addition, ReLU, GELU, and
   their backward-pass gradients directly into the GEMM kernel, eliminating separate kernel
   launches and memory round-trips. This is critical for transformer and MLP layer optimization.

4. **Batched operations for throughput**: `gemmBatched`, `gemmStridedBatched`, and
   `gemmGroupedBatched` (plus their Ex variants) are essential for batched inference,
   multi-head attention, and operator fusion scenarios.

5. **Algorithm selection**: `cublasLtMatmulAlgoGetHeuristic()` and
   `cublasLtMatmulAlgoConfigSetAttribute()` allow fine-tuning tile sizes, pipeline stages,
   split-K, and reduction schemes -- directly relevant to `pattern/gemm/tiling-strategy` and
   `pattern/gemm/pipeline-overlap`.

6. **Stream and workspace management**: `cublasSetStream()` and `cublasSetWorkspace()` (or
   the cuBLASLt workspace parameter) are essential for achieving overlap with other operations
   and avoiding internal memory allocations during execution.

---

## Enum Constants Reference

Commonly used enum values in cuBLAS/cuBLASLt API calls. These are not functions but control the behavior of GEMM and other operations.

### Operation Types

| Enum | Value | Description | Usage Context |
|------|-------|-------------|---------------|
| `CUBLAS_OP_N` | 0 | No transpose | Default for row-major A or column-major B |
| `CUBLAS_OP_T` | 1 | Transpose | Transpose the matrix before operation |
| `CUBLAS_OP_C` | 2 | Conjugate transpose | Complex matrices only |

### Algorithm Selection

| Enum | Description | Knowledge Node |
|------|-------------|----------------|
| `CUBLAS_GEMM_DEFAULT` | Let cuBLAS choose the best algorithm | pattern/gemm/cublas-integration |
| `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | Prefer Tensor Core algorithms | optimization/compute/tensor-core |
| `CUBLAS_GEMM_ALGO0` .. `CUBLAS_GEMM_ALGO23` | Specific algorithm index (legacy) | pattern/gemm/cublas-integration |
| `CUBLAS_GEMM_ALGO0_TENSOR_OP` .. `CUBLAS_GEMM_ALGO15_TENSOR_OP` | Specific Tensor Core algorithm index (legacy) | optimization/compute/tensor-core |

### Math Mode / Compute Type

| Enum | Description | Knowledge Node |
|------|-------------|----------------|
| `CUBLAS_DEFAULT_MATH` | Default math operations | - |
| `CUBLAS_TF32_TENSOR_OP_MATH` | Allow TF32 Tensor Core acceleration for FP32 inputs | optimization/compute/tensor-core |
| `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` | Disable reduced-precision accumulation | optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_16F` | FP16 compute | optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_32F` | FP32 compute (strict) | - |
| `CUBLAS_COMPUTE_32F_FAST_16F` | FP32 compute with FP16 acceleration | optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_32F_FAST_16BF` | FP32 compute with BF16 acceleration | optimization/compute/half-precision-math |
| `CUBLAS_COMPUTE_32F_FAST_TF32` | FP32 compute with TF32 Tensor Core | optimization/compute/tensor-core |
| `CUBLAS_COMPUTE_64F` | FP64 compute (strict) | - |
| `CUBLAS_COMPUTE_32I` | INT32 compute | - |

### Data Type (used in GemmEx / cuBLASLt)

| Enum | Description |
|------|-------------|
| `CUDA_R_16F` | FP16 (half) |
| `CUDA_R_16BF` | BF16 (bfloat16) |
| `CUDA_R_32F` | FP32 (float) |
| `CUDA_R_64F` | FP64 (double) |
| `CUDA_R_8I` | INT8 |
| `CUDA_R_8F_E4M3` | FP8 E4M3 |
| `CUDA_R_8F_E5M2` | FP8 E5M2 |

### cuBLASLt Descriptor Attributes

| Enum | Description | Knowledge Node |
|------|-------------|----------------|
| `CUBLASLT_MATMUL_DESC_TRANSA` | Transpose mode for matrix A | pattern/gemm/cublas-integration |
| `CUBLASLT_MATMUL_DESC_TRANSB` | Transpose mode for matrix B | pattern/gemm/cublas-integration |
| `CUBLASLT_MATMUL_DESC_EPILOGUE` | Epilogue fusion type | pattern/gemm/cublas-integration |
| `CUBLASLT_MATMUL_DESC_BIAS_POINTER` | Pointer for fused bias addition | pattern/gemm/cublas-integration |
| `CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` | Data type for bias | pattern/gemm/cublas-integration |
| `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` | Target SM count for this operation | optimization/latency/green-contexts |
| `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` | Max workspace memory for algorithm | pattern/gemm/cublas-integration |
| `CUBLASLT_MATMUL_PREF_SEARCH_MODE` | Algorithm search mode (0=default, 1=best) | pattern/gemm/cublas-integration |

### Pointer Mode

| Enum | Description |
|------|-------------|
| `CUBLAS_POINTER_MODE_HOST` | Scalar parameters (alpha, beta) are on host memory |
| `CUBLAS_POINTER_MODE_DEVICE` | Scalar parameters are on device memory |

### Fill Mode / Side / Diagonal

| Enum | Description | Used By |
|------|-------------|---------|
| `CUBLAS_FILL_MODE_LOWER` | Lower triangular | trmm, trsm, symm, syrk |
| `CUBLAS_FILL_MODE_UPPER` | Upper triangular | trmm, trsm, symm, syrk |
| `CUBLAS_SIDE_LEFT` | Op applied from left: C = alpha*op(A)*B | trmm, trsm |
| `CUBLAS_SIDE_RIGHT` | Op applied from right: C = alpha*B*op(A) | trmm, trsm |
| `CUBLAS_DIAG_NON_UNIT` | Diagonal elements are not assumed to be 1 | trmm, trsm |
| `CUBLAS_DIAG_UNIT` | Diagonal elements are assumed to be 1 | trmm, trsm |
