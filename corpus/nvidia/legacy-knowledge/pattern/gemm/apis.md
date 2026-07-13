# Gemm -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `dp4a.atype.btype` | PTX ISA | 4-way byte dot-product-accumulate (INT8) |
| `mma.sp.sync.aligned.shape.row.col.dtype.atype.btype.ctype` | PTX ISA | Sparse MMA (structured sparsity 2:4) |
| `mma.sync.aligned.shape.row.col.dtype.atype.btype.ctype` | PTX ISA | Warp-level MMA with explicit register mapping |
| `tcgen05.mma.sp[.kind][.shape]` | PTX ISA | 5th-gen TensorCore sparse MMA |
| `tcgen05.mma[.kind][.shape]` | PTX ISA | 5th-gen TensorCore MMA (supports M=64/128/256 via CTA groups) |
| `wgmma.mma_async.sp.sync.aligned.shape...` | PTX ISA | Sparse warpgroup MMA (structured 2:4 sparsity) |
| `wgmma.mma_async.sync.aligned.shape.dtype.atype.btype` | PTX ISA | Warpgroup-level async MMA (4 warps = 128 threads) |
| `wmma.load.a/b/c.sync[.aligned].shape.layout[.type]` | PTX ISA | Load matrix fragment from memory to registers |
| `wmma.mma.sync[.aligned].shape.dtype.atype.btype` | PTX ISA | Warp-level MxNxK matrix multiply-accumulate |
| `wmma.store.d.sync[.aligned].shape.layout[.type]` | PTX ISA | Store matrix fragment from registers to memory |
| `mma_sync()` | Runtime API | PTX-level matrix multiply-accumulate for Tensor Cores |
| `nvcuda::wmma::*` | Runtime API | Warp Matrix Multiply-Accumulate (WMMA) API for Tensor Cores |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `dp2a.mode.atype.btype` | PTX ISA | 2-way dot-product-accumulate (INT16xINT8) |
