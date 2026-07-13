# Reduction -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `bar{.cta}.red.{popc/and/or}` | PTX ISA | CTA-level barrier with reduction |
| `red[.sem][.scope][.space].op.type` | PTX ISA | Reduction (no return value); same ops as atom |
| `redux.sync.op.type` | PTX ISA | Warp-level reduction (.add/.min/.max/.and/.or/.xor) |
| `shfl.sync[.mode].b32` | PTX ISA | Warp shuffle: exchange registers between lanes (.up/.down/.bfly/.idx) |
| `__shfl_down_sync()` | Runtime API | Warp shuffle down (used in reductions) |
| `__shfl_xor_sync()` | Runtime API | Warp shuffle XOR (butterfly reduction) |
| `cublas<t>dot()` | cuBLAS | Dot product of two vectors |
| `cublas<t>nrm2()` | cuBLAS | Euclidean norm of vector |
| `cublasDotEx()` | cuBLAS | Dot product with mixed precision |
| `cublasNrm2Ex()` | cuBLAS | Euclidean norm with mixed precision |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `CUBLASLT_EPILOGUE_BGRADA` | cuBLAS | Bias gradient over A rows |
| `CUBLASLT_EPILOGUE_BGRADB` | cuBLAS | Bias gradient over B columns |
| `cublas<t>asum()` | cuBLAS | Sum of absolute values |
| `cublasI<t>amax()` | cuBLAS | Index of max absolute value element |
| `cublasI<t>amin()` | cuBLAS | Index of min absolute value element |
