---
title: Tensor-core GEMM Pattern -- Library Fallback Paths
pattern_class: tensor-core
op: gemm
status: draft
source:
- path: wiki/nvidia/foundations/compute/gemm.md
  anchor: Aligned GEMM on Hopper via cutlass cooperative warp-specialized kernel
  excerpt: cutlass-API track baseline at aligned shapes
- path: sources/experience/api-probes/gemm-ptx.md
  anchor: cublasGemmEx call against bf16 inputs / f32 accumulator
  excerpt: row-major-vs-col-major handling for cuBLAS reference at M=64 N=8 K=16
- path: wiki/nvidia/foundations/compute/gemm-ptx/pitfalls.md
  anchor: '#3 cuBLAS row-major-vs-col-major requires careful op_T handling'
id: routing-gemm-library-fallback
type: operator-routing
vendor: nvidia
operator: gemm
---
# Tensor-core GEMM -- Library Fallback Paths

Before writing a custom Hopper GEMM, use one of the production paths below. These are GPU-vendor maintained, tuned across CUDA toolkit releases, and cover the vast majority of GEMM shapes / dtypes / fusion patterns.

---

## 1. PyTorch matmul ops

**When to use**: the caller's workflow is PyTorch-based and the GEMM is a standard matrix multiply (optionally with bias / activation through `nn.functional.linear` or a fused module).

### torch.matmul / torch.mm / torch.bmm

```python
import torch

A = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
B = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
D = torch.matmul(A, B)               # shape: (2048, 2048), dtype bf16

# Batched matmul
A = torch.randn(8, 512, 1024, device="cuda", dtype=torch.bfloat16)
B = torch.randn(8, 1024, 512, device="cuda", dtype=torch.bfloat16)
D = torch.bmm(A, B)                  # shape: (8, 512, 512)
```

`torch.matmul` dispatches to cuBLASLt / cuBLAS / cutlass / custom kernels depending on shape, dtype, and device. For Hopper, the sm_90a path goes through cuBLASLt for most shapes and dtypes.

### torch.nn.functional.linear (with optional bias)

```python
import torch.nn.functional as F

x = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
y = F.linear(x, W, bias)             # y = x @ W^T + bias
```

This is the most common GEMM call shape in transformer code. PyTorch dispatches the bias-add via cuBLASLt's `CUBLASLT_EPILOGUE_BIAS` epilogue when the dtypes / layouts support it; otherwise a separate add kernel runs.

**Shape range**: any shape that fits in GPU memory. cuBLASLt has shape-dependent dispatch tables; tiny shapes (M*N*K < ~10^6) are launch-overhead bound and not the focus of this pattern.

**Supported dtypes**: float32, float16, bfloat16, tf32 (compute), float8\_e4m3 / e5m2 (compute), int8.

---

## 2. cuBLAS direct (cublasGemmEx)

**When to use**: writing a standalone CUDA / C++ program (not PyTorch) and the GEMM is a plain `D = alpha * A * B + beta * C` with standard transpose / dtype combos.

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// D[M, N] = alpha * A[M, K] * B[K, N] + beta * C[M, N]
// All three matrices in column-major (cuBLAS native).
const float alpha = 1.f, beta = 0.f;
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // no transpose
    M, N, K,
    &alpha,
    dA, CUDA_R_16BF, M,         // ldA = M (col-major)
    dB, CUDA_R_16BF, K,         // ldB = K
    &beta,
    dD, CUDA_R_32F,  M,         // ldD = M, dtype f32 accumulator/output
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### Row-major callers

cuBLAS is column-major. To compute `D[M, N] = A[M, K] @ B[K, N]` with all three in row-major, the canonical idiom swaps the operands and the sizes:

```cpp
// Same logical operation, all-row-major data:
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,                    // sizes swapped (compute B^T * A^T col-major)
    &alpha,
    dB, CUDA_R_16BF, N,         // B is row-major K x N -> col-major N x K, ldB = N
    dA, CUDA_R_16BF, K,         // A is row-major M x K -> col-major K x M, ldA = K
    &beta,
    dD, CUDA_R_32F,  N,         // D row-major M x N -> col-major N x M
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

This is the call shape used in the cutlass-free GEMM record's reference (`artifacts/experience/kernel-records/2026-04-29-gemm-ws-ptx/gemm_ws_ptx.cu` and `artifacts/experience/api-probes/gemm-ptx/gemm_ptx.cu`). Get `lda` / `ldb` wrong here and the diff against a custom kernel looks identical to a layout-mapped kernel bug -- which is hard to disentangle. Sanity-check with all-1.0 inputs first (output should equal K).

**Supported compute types** (CUDA 12.9):

| Compute type | Input dtypes (A, B) | Accumulator / output |
|---|---|---|
| `CUBLAS_COMPUTE_32F` | fp32, fp16, bf16 | fp32 |
| `CUBLAS_COMPUTE_32F_FAST_TF32` | fp32 (computed as tf32) | fp32 |
| `CUBLAS_COMPUTE_16F` | fp16 | fp16 |
| `CUBLAS_COMPUTE_32I` | int8 | int32 |
| `CUBLAS_COMPUTE_32F_FAST_16F` | fp32 (computed as fp16) | fp32 |
| `CUBLAS_COMPUTE_32F_FAST_16BF` | fp32 (computed as bf16) | fp32 |

---

## 3. cuBLASLt (matmul descriptor + epilogue)

**When to use**: GEMM with a fused epilogue from the cuBLASLt catalogue (bias, ReLU, GELU, dReLU + bgrad, scale, quantize). The epilogue runs in the same kernel as the GEMM body, avoiding an extra global-memory round trip.

```cpp
#include <cublasLt.h>

cublasLtHandle_t lt;
cublasLtCreate(&lt);

cublasLtMatmulDesc_t op_desc = nullptr;
cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// Fused bias + ReLU epilogue.
cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_BIAS;
cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, sizeof(dBias));

// ... allocate Layout descriptors for A, B, C, D, then call cublasLtMatmul(...).
```

**Catalogue of fused epilogues** (selected; see cuBLASLt header for the full set):

| Epilogue tag | Fused operation |
|---|---|
| `CUBLASLT_EPILOGUE_DEFAULT` | none |
| `CUBLASLT_EPILOGUE_RELU` | ReLU on D |
| `CUBLASLT_EPILOGUE_BIAS` | per-row bias add |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | bias-add then ReLU |
| `CUBLASLT_EPILOGUE_GELU` / `_GELU_BIAS` | GELU; with optional bias |
| `CUBLASLT_EPILOGUE_DRELU` / `_DRELU_BGRAD` | ReLU backward |
| `CUBLASLT_EPILOGUE_DGELU` / `_DGELU_BGRAD` | GELU backward |

**Algorithm selection**: `cublasLtMatmulAlgoGetHeuristic` queries the available algorithm IDs ranked by predicted performance for the given shape / dtype / layout. Pick the top result and pass it to `cublasLtMatmul`. For repeated calls at the same shape, cache the algo handle.

---

## 4. Cutlass via the device-API (cutlass::gemm::device::GemmUniversalAdapter)

**When to use**: cuBLASLt does not cover the dtype / layout / fusion pattern, but the project allows cutlass as a header dependency. Cutlass's collective builder selects a tile / pipeline / scheduler combination tuned for the target shape.

This is the entry point that `wiki/nvidia/foundations/compute/gemm.md` (aligned shapes) and `wiki/nvidia/foundations/compute/gemm/non-aligned-tail/skill.md` (M / N not divisible by the wgmma atom) build on. The canonical reproducible artifact lives at `artifacts/experience/api-probes/gemm/gemm_compare_ws.cu` (plain WS) / `gemm_compare_pingpong.cu` / `gemm_aligned.cu` (cooperative auto-selected). CUTLASS example 48 source-reading notes live at `wiki/nvidia/code-walkthroughs/cutlass-cute/example48-hopper-warp-specialized-gemm/`; persistent-schedule notes live at `wiki/nvidia/code-walkthroughs/cutlass-cute/persistent-kernel/`.

**Shape range**: full Hopper coverage. The cutlass `KernelTmaWarpSpecialized*` schedules fall through three variants (plain WS / pingpong / cooperative); `KernelScheduleAuto` picks one based on shape and SM target. For reproducibility, pin the schedule explicitly (WS pitfall #8 list bullet).

**When NOT to use cutlass**: if the project is required to be cutlass-free (compliance / supply-chain audit). In that case the path is the cutlass-free PTX track in `INDEX.md` Step 1 Q4 NO branch.

---

## Decision summary

| Caller environment | Standard GEMM | + standard fused epilogue | + non-standard fusion |
|---|---|---|---|
| PyTorch | `torch.matmul` / `F.linear` | `F.linear` (covers BIAS); else `torch.compile` fuses | custom kernel |
| C++ + cuBLAS allowed | `cublasGemmEx` | `cublasLtMatmul` | cutlass via skill #3/#4, OR custom |
| C++ + cutlass allowed | cutlass GemmUniversalAdapter | cutlass + custom epilogue (skill `gemm-fused/cutlass-epilogue-prologue`) | cutlass + custom epilogue |
| C++ cutlass-free | `cublasGemmEx` for the reference; **custom only** for the production kernel | custom (cutlass-free PTX track) | custom (cutlass-free PTX track) |

**Note on the cutlass-free row**: cuBLAS itself is not part of cutlass and using it as a *reference* in a cutlass-free build is fine. What the cutlass-free track forbids is shipping cutlass headers in the production kernel binary. The kernel records under `sources/experience/kernel-records/` use cuBLAS exclusively for verification.
