---
title: Transpose Pattern -- Library Fallback Paths
pattern_class: cuda-core
op: transpose
status: draft
source:
- path: spec
  anchor: Reference
id: routing-transpose-library-fallback
type: operator-routing
vendor: nvidia
operator: transpose
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1484-L1540
---
# Transpose -- Library Fallback Paths

Before writing a custom transpose kernel, use one of the following library paths. These are production-quality, well-tested implementations that handle edge cases (non-square matrices, high-dim permutes, batch dimensions) and integrate cleanly with the surrounding toolchain.

---

## 1. PyTorch transpose / permute

**When to use**: the caller's workflow is PyTorch-based and the transpose is a standard op on a `torch.Tensor` on CUDA.

### `tensor.transpose(dim0, dim1)`

Swaps two dimensions, producing a **view** (no data copy).

```python
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
y = x.transpose(0, 1)              # view: y.stride() == (1, 4096)
                                   # NO kernel launch at this point
```

Views are free until the downstream op requires contiguity. If the downstream op is a library call (cuBLAS matmul, cuDNN conv) that accepts a stride argument, the view suffices with zero cost.

### `tensor.contiguous()` — materialize

```python
y = x.transpose(0, 1).contiguous()  # materialization: launches a kernel
```

Use this when the downstream op requires row-major contiguous data. PyTorch's `contiguous()` for 2-D transpose is implemented as a smem-tiled kernel equivalent to the custom one in `INDEX.md` step 5a, so it sits at ~1600 GB/s on H200 (≈ the custom kernel's 1685 GB/s from the `smem-tile-reuse` probe).

### `tensor.permute(*dims)`

General N-D permutation, also produces a view.

```python
x = torch.randn(8, 16, 32, 64, device='cuda')
y = x.permute(0, 2, 1, 3)     # view; stride recomputed
y = y.contiguous()             # materialize if needed
```

For high-dim permute, `contiguous()` launches a kernel that is essentially a 2-D transpose loop over the product of unaffected batch dims. Performance is slightly below the 2-D case because of the batch-loop overhead; for very tall/narrow effective 2-D slices it can drop to ~50 % of the 2-D peak.

---

## 2. cuBLASLt matrix transform

**When to use**: the transposed matrix will immediately feed into a cuBLAS/cuBLASLt GEMM, and the surrounding type conversion (fp32↔fp16, row-major↔column-major) can be fused.

```cpp
cublasLtHandle_t handle;
cublasLtCreate(&handle);

cublasLtMatrixTransformDesc_t desc;
cublasLtMatrixTransformDescCreate(&desc, CUDA_R_32F);
cublasOperation_t op = CUBLAS_OP_T;
cublasLtMatrixTransformDescSetAttribute(desc,
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
    &op, sizeof(op));

cublasLtMatrixLayout_t layoutIn, layoutOut;
// ... configure layouts with rows, cols, lda/ldb, dtype ...

float alpha = 1.0f, beta = 0.0f;
cublasLtMatrixTransform(handle, desc,
    &alpha, d_in,  layoutIn,
    &beta,  nullptr, nullptr,
    d_out, layoutOut, stream);
```

**When this is faster than a custom kernel**: when the target downstream op is a cuBLASLt GEMM and you can wire the same `cublasLtHandle_t` through — eliminates one API boundary. Also useful when the transform includes a dtype change (fp32 → fp16), because cuBLASLt fuses the transpose and the dtype conversion into one kernel.

**Caveat**: `cublasLtMatrixTransform` has non-trivial launch overhead for small matrices (< 256² fp32) because of layout-descriptor configuration. For repeated small transposes, a custom kernel or reusing pre-configured layouts is faster.

---

## 3. Thrust `thrust::transform` (not a transpose — for completeness)

Thrust has no direct transpose op; its closest is `thrust::transform` for elementwise ops. Do not use Thrust for transpose — it will not be a smem-tiled kernel and will run at stride-N bandwidth.

---

## 4. CUTLASS transpose / epilogue

**When to use**: the transpose is part of a larger fused GEMM + epilogue pipeline and the whole operation is being authored in CUTLASS.

CUTLASS provides `cutlass::transform::threadblock::` primitives that implement the smem-tiled transpose inside a larger kernel. For a standalone transpose, CUTLASS is overkill — use PyTorch or cuBLASLt. For transpose-fused-with-GEMM / transpose-as-epilogue, see the CUTLASS docs (outside the scope of this fallback guide).

---

## Decision summary

| Caller environment | Scenario | Recommended path |
|---|---|---|
| PyTorch, view only | Downstream accepts strided view | `tensor.transpose(dim0, dim1)` (zero cost) |
| PyTorch, materialize | Downstream needs contiguous | `tensor.transpose(...).contiguous()` |
| PyTorch, high-dim | N-D permute | `tensor.permute(...).contiguous()` |
| C++/CUDA, feeds cuBLAS GEMM | Transpose + dtype + GEMM pipeline | `cublasLtMatrixTransform` |
| C++/CUDA, standalone | No GEMM downstream | Custom smem-tiled kernel (see INDEX step 1) |
| AoS ↔ SoA (not a transpose) | Struct fields split | Custom 1-pass kernel from layout-transform S1 |

**When to skip the library entirely**: the transpose must be fused with surrounding compute to avoid an extra gmem round-trip, the shape regime is too small for library launch overhead to amortize, or the dtype / layout combination is not supported (packed int8, packed fp8, custom strided views). In those cases see `INDEX.md` Step 1.
