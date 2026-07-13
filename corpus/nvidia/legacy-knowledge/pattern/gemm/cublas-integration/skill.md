# cuBLAS Integration -- Skills

## When to Apply
- Standard GEMM where writing a custom kernel is not justified
- Need epilogue fusion with common operations (bias, ReLU, GELU)
- Mixed-precision GEMM with various input/output/compute type combinations
- Batched or grouped GEMM workloads

## Skill 1: Choosing the Right cuBLAS API

### Quick selection guide:
| Scenario | API |
|----------|-----|
| Simple FP32/FP64 GEMM | `cublas<t>gemm()` |
| Mixed-precision GEMM | `cublasGemmEx()` |
| Need epilogue fusion | `cublasLtMatmul()` |
| Batched GEMM (same sizes) | `cublas<t>gemmStridedBatched()` |
| Batched GEMM (array of pointers) | `cublas<t>gemmBatched()` |
| Grouped GEMM (different sizes) | `cublas<t>gemmGroupedBatched()` |
| Full control over algorithm | `cublasLtMatmul()` with algo selection |
| Multi-GPU | `cublasXt<t>gemm()` |

## Skill 2: cublasLtMatmul Setup for Epilogue Fusion

### Step-by-step:
1. **Create matrix layouts**: `cublasLtMatrixLayoutCreate` for A, B, C, D
2. **Create matmul descriptor**: `cublasLtMatmulDescCreate(computeType, scaleType)`
3. **Set epilogue**: `cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue)`
4. **Set bias pointer** (if using bias): `cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr)`
5. **Create preference**: `cublasLtMatmulPreferenceCreate()`
6. **Set workspace size**: `cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize)`
7. **Get heuristic**: `cublasLtMatmulAlgoGetHeuristic(handle, desc, A_layout, B_layout, C_layout, D_layout, pref, maxResults, &results, &numResults)`
8. **Execute**: `cublasLtMatmul(handle, desc, alpha, A, A_layout, B, B_layout, beta, C, C_layout, D, D_layout, &results[0].algo, workspace, workspaceSize, stream)`

## Skill 3: Available Epilogue Fusions

### Forward pass epilogues:
| Epilogue | Operation |
|----------|-----------|
| `DEFAULT` | D = alpha*A*B + beta*C |
| `BIAS` | D = alpha*A*B + beta*C + bias |
| `RELU` | D = ReLU(alpha*A*B + beta*C) |
| `RELU_BIAS` | D = ReLU(alpha*A*B + beta*C + bias) |
| `GELU` | D = GELU(alpha*A*B + beta*C) |
| `GELU_BIAS` | D = GELU(alpha*A*B + beta*C + bias) |
| `RELU_AUX` | D = ReLU(...), save bitmask for backward |
| `GELU_AUX` | D = GELU(...), save pre-activation for backward |
| `RELU_AUX_BIAS` | D = ReLU(... + bias), save bitmask |
| `GELU_AUX_BIAS` | D = GELU(... + bias), save pre-activation |

### Backward pass epilogues:
| Epilogue | Operation |
|----------|-----------|
| `DRELU` | dX = dY * ReLU_mask |
| `DGELU` | dX = dY * GELU'(pre_activation) |
| `DRELU_BGRAD` | dX = dY * ReLU_mask; dbias = sum(dY) |
| `DGELU_BGRAD` | dX = dY * GELU'(...); dbias = sum(dY) |
| `BGRADA` | dbias = sum over A rows |
| `BGRADB` | dbias = sum over B columns |

## Skill 4: Algorithm Selection and Tuning
- **Heuristic-based**: `cublasLtMatmulAlgoGetHeuristic` returns ranked list (fastest method)
- **Exhaustive search**: Iterate `cublasLtMatmulAlgoGetIds` -> `cublasLtMatmulAlgoCheck` -> benchmark each
- **Query capabilities**: `cublasLtMatmulAlgoCapGetAttribute` to check tile sizes, stages, math mode
- **Force specific config**: `cublasLtMatmulAlgoConfigSetAttribute` with TILE_ID, STAGES_ID, SPLITK_NUM, etc.
- Larger workspace enables more algorithm choices; allocate at least 32MB

## Skill 5: Determinism Control
- `cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED)`: disable atomics for bit-exact reproducibility
- Trade: some algorithms become unavailable, potentially slower
- For `cublasLtMatmul`: set reduction scheme to non-atomic via preference attributes
- Critical for training reproducibility and debugging

## Skill 6: Performance Tips
- Column-major layout (Fortran order) avoids internal transpose in many cases
- Align matrix leading dimensions to 128 bytes for best memory throughput
- For batched GEMM: strided batched is faster than pointer-array batched (better memory access pattern)
- Pre-allocate workspace once and reuse across calls
- Use `cublasSetStream` to overlap cuBLAS calls on different streams

## Cross-References
- pattern/gemm/tiling-strategy -- cuBLAS tile selection via heuristic API
- optimization/compute/operator-fusion -- epilogue fusion concepts
- optimization/compute/half-precision-math -- mixed-precision GEMM types
