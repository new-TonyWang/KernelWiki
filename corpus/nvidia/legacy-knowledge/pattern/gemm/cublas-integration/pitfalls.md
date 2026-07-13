# cuBLAS Integration -- Pitfalls

## P1: Forgetting Column-Major Convention
- **Symptom**: Transposed or scrambled output matrix
- **Root cause**: cuBLAS uses column-major (Fortran) layout by default; C/C++ arrays are row-major
- **Fix**: Either transpose inputs/outputs, or set `CUBLAS_OP_T` on both matrices and swap A/B
- **Common trick**: For row-major C = A * B, call `cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)` -- swap A,B and M,N

## P2: Leading Dimension Mismatch
- **Symptom**: Garbage output or access violation
- **Root cause**: lda, ldb, ldc must reflect actual memory layout, not logical matrix dimensions
- **For column-major**: lda = number of rows of op(A) stored in memory (may differ from M if matrix is padded)
- **Fix**: For a MxK column-major matrix, lda >= M; for a KxN column-major matrix, ldb >= K

## P3: Workspace Too Small Limits Algorithm Choices
- **Symptom**: `cublasLtMatmulAlgoGetHeuristic` returns 0 results or only slow algorithms
- **Root cause**: Preference workspace size is too small; best algorithms need workspace
- **Fix**: Allocate at least 32MB workspace; pass via `cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &size)`

## P4: Epilogue Fusion Silently Falls Back to Unfused Path
- **Symptom**: Performance not improved or slightly worse after enabling epilogue
- **Root cause**: Selected algorithm does not support the requested epilogue; cuBLAS silently uses a compatible but slower algorithm
- **Detection**: Check `heuristicResult.state` -- should be `CUBLAS_STATUS_SUCCESS`
- **Fix**: Query `CUBLASLT_ALGO_CAP_EPILOGUE_MASK` to verify epilogue support for selected algorithm

## P5: Atomics Mode Causing Non-Reproducible Results
- **Symptom**: Bit-level differences across runs with same inputs
- **Root cause**: Default atomics mode allows atomic operations that introduce non-determinism
- **Fix**: `cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED)` for reproducibility
- **Trade**: Some algorithms become unavailable; may be slightly slower

## P6: Type Mismatch Between Compute and Scale Types
- **Symptom**: `CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmulDescCreate`
- **Root cause**: Incompatible combination of compute type and scale type
- **Example**: CUDA_R_16F compute with CUDA_R_32F scale -- unsupported on some architectures
- **Fix**: Check compatibility matrix in cuBLAS documentation; common safe combo: CUDA_R_32F compute + CUDA_R_32F scale

## P7: Bias Pointer Not Set for Bias Epilogues
- **Symptom**: Crash or incorrect results when using RELU_BIAS or GELU_BIAS epilogues
- **Root cause**: Epilogue references bias but `CUBLASLT_MATMUL_DESC_BIAS_POINTER` not set
- **Fix**: Must set both the epilogue type AND the bias pointer via `cublasLtMatmulDescSetAttribute`
- **Also**: Bias data type must match via `CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`

## P8: GroupedBatchedGemm with Misaligned Group Sizes
- **Symptom**: Wrong results or crash with `cublasGemmGroupedBatchedEx`
- **Root cause**: Each group must have consistent M, N, K within the group; misalignment between group layout and actual data
- **Fix**: Verify each group's dimensions are correctly specified in the grouped layout descriptor

## P9: Forgetting to Destroy Descriptors
- **Symptom**: Memory leak over many iterations (common in training loops)
- **Root cause**: `cublasLtMatmulDescCreate`, `cublasLtMatrixLayoutCreate`, `cublasLtMatmulPreferenceCreate` all allocate; must be paired with `Destroy`
- **Fix**: Use RAII wrappers or explicit destroy calls in cleanup path

## P10: AUX Epilogue Without AUX Pointer Setup
- **Symptom**: Crash when using `GELU_AUX` or `RELU_AUX` epilogues
- **Root cause**: AUX epilogues save intermediate data (pre-activation or bitmask) for backward pass; requires setting `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER` and `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`
- **Fix**: Set all required auxiliary attributes before executing matmul

## P11: cuBLAS kernel runs at only 6 (discovered in verification)

**Symptom**: cuBLAS kernel runs at only 6.3% warp occupancy (vs 92.8% baseline) yet is 7.5x faster — occupancy is a misleading optimization target for register-heavy vendor libraries; do not try to increase occupancy by reducing register usage.
**Source**: Level 3 sandbox verification (2026-04-05)

## P12: Neither baseline nor optimized version activates tensor cores (sm__pipe_tensor_cycles_active = 0%), meaning cublasLt selected a non-TC algorithm despite being an FP32 GEMM on Ampere — likely TF32 is disabled, leaving significant performance on the table regardless of epilogue fusion (discovered in verification)

**Symptom**: Neither baseline nor optimized version activates tensor cores (sm__pipe_tensor_cycles_active = 0%), meaning cublasLt selected a non-TC algorithm despite being an FP32 GEMM on Ampere — likely TF32 is disabled, leaving significant performance on the table regardless of epilogue fusion.
**Source**: Level 3 sandbox verification (2026-04-05)

## P13: Both baseline and optimized show 0% tensor core utilization despite being a GEMM workload, suggesting the cuBLAS path may be falling back to non-TC kernels; epilogue fusion benefits are likely gated on tensor core execution (discovered in verification)

**Symptom**: Both baseline and optimized show 0% tensor core utilization despite being a GEMM workload, suggesting the cuBLAS path may be falling back to non-TC kernels; epilogue fusion benefits are likely gated on tensor core execution.
**Source**: Level 3 sandbox verification (2026-04-05)

## P14: For well-matched problem sizes, cublasLt heuristic rank-1 is already optimal; exhaustive search adds startup cost with no runtime benefit (discovered in verification)

**Symptom**: For well-matched problem sizes, cublasLt heuristic rank-1 is already optimal; exhaustive search adds startup cost with no runtime benefit.
**Source**: Level 3 sandbox verification (2026-04-05)

## P15: Long scoreboard stalls nearly doubled (29 (discovered in verification)

**Symptom**: Long scoreboard stalls nearly doubled (29.4%→52.4%), indicating the deterministic code path has significantly worse memory-level parallelism; warp occupancy also dropped (98.2%→93.4%), so determinism mode can amplify latency-bound bottlenecks beyond the expected small overhead
**Source**: Level 3 sandbox verification (2026-04-05)

## P16: Baseline NCU captured a near-trivial kernel (240 instructions, 2816B DRAM read) suggesting it profiled a wrapper/setup kernel rather than the actual GEMM — cuBLAS performance tips may change which internal kernel cuBLAS selects, making before/after NCU comparisons measure fundamentally different code paths (discovered in verification)

**Symptom**: Baseline NCU captured a near-trivial kernel (240 instructions, 2816B DRAM read) suggesting it profiled a wrapper/setup kernel rather than the actual GEMM — cuBLAS performance tips may change which internal kernel cuBLAS selects, making before/after NCU comparisons measure fundamentally different code paths.
**Source**: Level 3 sandbox verification (2026-04-05)
