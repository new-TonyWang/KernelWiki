---
title: Layout Transform APIs
status: draft
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1379-L1411
  excerpt: 'Global memory coalescing: adjacent threads must land in the same 32-byte
    segment to avoid multiple transactions per warp.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
  excerpt: 'Strided accesses: stride-2 gives 50% efficiency; stride-32 gives 12.5%.
    Layout transforms convert strided patterns into unit-stride.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/parallel-thread-execution/cuda_parallel-thread-execution_index.html.md
  anchor: L13528-L13530
  excerpt: prmt.b32 byte permutation across two 32-bit registers; mode selector picks
    bytes 0..7 of (a || b).
apis:
- func_name: cudaMallocPitch
  namespace: cuda-runtime
  kind: host-alloc
  signature: cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t widthInBytes,
    size_t height);
  notes: Allocates 2D memory with each row padded to a pitch that guarantees coalescing
    on the current architecture. Returned pitch is in bytes.
- func_name: cudaMalloc3D
  namespace: cuda-runtime
  kind: host-alloc
  signature: cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);
  notes: 3D analog of cudaMallocPitch; returns pitched pointer for logical 3D layouts.
- func_name: cudaMemcpy2D
  namespace: cuda-runtime
  kind: host-copy
  signature: cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t
    spitch, size_t width, size_t height, cudaMemcpyKind kind);
  notes: Required when either side was allocated with cudaMallocPitch. Plain cudaMemcpy
    will corrupt pitched buffers.
- func_name: cudaMemcpy2DAsync
  namespace: cuda-runtime
  kind: host-copy
  notes: Async variant; takes a cudaStream_t.
- func_name: cublasLtMatrixTransform
  namespace: cublas
  kind: library
  signature: cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t lightHandle,
    cublasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A,
    cublasLtMatrixLayout_t Adesc, const void* beta, const void* B, cublasLtMatrixLayout_t
    Bdesc, void* C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream);
  notes: Performs C = alpha * opTrans(A) + beta * opTrans(B) with configurable row/column
    major, data type conversion, and stride transformation. Use for AoS ↔ packed-tile
    layouts that feed cuBLAS GEMM.
- func_name: cp.async.bulk.tensor.{dim}.{dst}.{src}.{mode}
  namespace: ptx
  kind: ptx-tma
  notes: Hopper TMA instruction; tensor descriptor can specify stride/swizzle, performing
    a layout transform during the copy. Details belong to wiki/nvidia/hardware/tma/
    (pending bucket F bootstrap).
- func_name: tensormap.replace
  namespace: ptx
  kind: ptx-tma
  notes: Dynamically modify a tensor map descriptor; lets the same kernel target different
    strides/swizzles without recompilation.
- func_name: prmt.b32
  namespace: ptx
  kind: ptx-permute
  signature: prmt.b32{.mode} d, a, b, c;
  notes: Byte permute across two 32-bit registers; selector c picks 4 bytes from the
    8-byte concatenation (a || b). Expose via inline asm or via the __byte_perm intrinsic.
- func_name: __byte_perm
  namespace: cuda-runtime
  kind: intrinsic
  signature: unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int
    s);
  notes: C++ intrinsic that lowers to prmt.b32. Use when the permutation selector
    is a compile-time constant.
- func_name: cudaFuncSetAttribute
  namespace: cuda-runtime
  kind: host-config
  notes: Use when the transform kernel itself needs carveout adjustment (shared-memory-heavy
    transpose kernel).
id: api-layout-transform-ref
type: api-definition
vendor: nvidia
func_name: Layout Transform APIs
namespace: runtime
header: cuda_runtime.h
signature: See documentation
---
## Core APIs

| API / Instruction | Layer | Purpose |
|-------------------|-------|---------|
| `cudaMallocPitch(&ptr, &pitch, widthBytes, height)` | Runtime | 2D pitched allocation (returns bytes-aligned row stride) |
| `cudaMalloc3D(&pitchedPtr, extent)` | Runtime | 3D pitched analog |
| `cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)` | Runtime | Mandatory for pitched buffers |
| `cublasLtMatrixTransform(...)` | cuBLAS | Library-level layout + dtype transform |
| `__byte_perm(x, y, selector)` | Intrinsic | Compile-time-friendly wrapper for `prmt.b32` |

## PTX-level

| Instruction | Purpose |
|-------------|---------|
| `prmt.b32{.mode} d, a, b, c` | Byte permutation across two 32-bit registers |
| `cp.async.bulk.tensor.{dim}...` | TMA bulk copy that applies layout transform via tensor descriptor (Hopper+) |
| `tensormap.replace` | Mutate a tensor descriptor's stride/swizzle at runtime |

## Cross-references

- **Coalescing**: `wiki/nvidia/foundations/memory/coalescing/` — the reason layout transform exists. Coalescing is a kernel-side fix for stride = 1; layout transform is a pre-kernel fix for when the kernel cannot be made stride-1 on its hot field.
- **Shared memory cache**: `wiki/nvidia/foundations/memory/shared-memory-cache/` — owns the transpose *mechanism* (S2 coalescing transform via smem). This skill's S3 only records the *choice* to do the transpose.
- **Vectorized access**: `wiki/nvidia/foundations/memory/vectorized-access/` — pairs naturally with S1: once SoA makes a field stride-1, `float4` / `bfloat162` loads become applicable.
- **Bank-conflict avoidance**: `wiki/nvidia/foundations/memory/bank-conflict/` — the `[TILE][TILE+1]` padding cited by S3 is this skill's rule.
- **TMA**: `wiki/nvidia/hardware/tma/` (pending bucket F) — `cp.async.bulk.tensor` is a potential single-instruction replacement for S1 + S3 on sm_90+.
