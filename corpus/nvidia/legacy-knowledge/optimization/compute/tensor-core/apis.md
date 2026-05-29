# Tensor Core -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `dp4a.atype.btype` | PTX ISA | 4-way byte dot-product-accumulate (INT8) |
| `mma.sp.sync.aligned.shape.row.col.dtype.atype.btype.ctype` | PTX ISA | Sparse MMA (structured sparsity 2:4) |
| `mma.sync.aligned.shape.row.col.dtype.atype.btype.ctype` | PTX ISA | Warp-level MMA with explicit register mapping |
| `tcgen05.alloc` | PTX ISA | Allocate Tensor Memory (dedicated on-chip memory for TC5) |
| `tcgen05.commit` | PTX ISA | Commit outstanding TC5 async operations |
| `tcgen05.cp[.shape]` | PTX ISA | Copy data between shared memory and Tensor Memory |
| `tcgen05.dealloc` | PTX ISA | Deallocate Tensor Memory |
| `tcgen05.fence` | PTX ISA | TC5 fence (before/after operations) |
| `tcgen05.ld[.shape].tmem` | PTX ISA | Load from Tensor Memory to registers |
| `tcgen05.mma.sp[.kind][.shape]` | PTX ISA | 5th-gen TensorCore sparse MMA |
| `tcgen05.mma[.kind][.shape]` | PTX ISA | 5th-gen TensorCore MMA (supports M=64/128/256 via CTA groups) |
| `tcgen05.st[.shape].tmem` | PTX ISA | Store from registers to Tensor Memory |
| `tcgen05.wait` | PTX ISA | Wait for TC5 operations to complete |
| `wgmma.commit_group.sync.aligned` | PTX ISA | Commit outstanding wgmma.mma_async ops into a group |
| `wgmma.fence.sync.aligned` | PTX ISA | Fence before wgmma (declare register/smem readiness) |
| `wgmma.mma_async.sp.sync.aligned.shape...` | PTX ISA | Sparse warpgroup MMA (structured 2:4 sparsity) |
| `wgmma.mma_async.sync.aligned.shape.dtype.atype.btype` | PTX ISA | Warpgroup-level async MMA (4 warps = 128 threads) |
| `wgmma.wait_group.sync.aligned N` | PTX ISA | Wait for wgmma group N to complete |
| `wmma.load.a/b/c.sync[.aligned].shape.layout[.type]` | PTX ISA | Load matrix fragment from memory to registers |
| `wmma.mma.sync[.aligned].shape.dtype.atype.btype` | PTX ISA | Warp-level MxNxK matrix multiply-accumulate |
| `wmma.store.d.sync[.aligned].shape.layout[.type]` | PTX ISA | Store matrix fragment from registers to memory |
| `mma_sync()` | Runtime API | PTX-level matrix multiply-accumulate for Tensor Cores |
| `nvcuda::wmma::*` | Runtime API | Warp Matrix Multiply-Accumulate (WMMA) API for Tensor Cores |
| `cublas<t>gemm()` | cuBLAS | C = alpha*op(A)*op(B) + beta*C |
| `cublas<t>gemmEx()` | cuBLAS | GEMM with lower-precision inputs, higher-precision compute |
| `cublasGemmBatchedEx()` | cuBLAS | Batched GemmEx with mixed types + algo selection |
| `cublasGemmEx()` | cuBLAS | Fully flexible GEMM: individual data types + algorithm selection |
| `cublasGemmGroupedBatchedEx()` | cuBLAS | Grouped batched GemmEx (heterogeneous sizes) with mixed types |
| `cublasGemmStridedBatchedEx()` | cuBLAS | Strided batched GemmEx with mixed types + algo selection |
| `cublasHgemm()` | cuBLAS | Half-precision GEMM (included in above) |
| `cublasLtMatmul()` | cuBLAS | D = alpha*op(A)*op(B) + beta*C with epilogue fusion |
| `cublasSetMathMode()` | cuBLAS | Enable/disable Tensor Core usage |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `dp2a.mode.atype.btype` | PTX ISA | 2-way dot-product-accumulate (INT16xINT8) |
| `tcgen05.relinquish_alloc_permit` | PTX ISA | Relinquish allocation permit for Tensor Memory |
| `cublasGetMathMode()` | cuBLAS | Query current math mode |
