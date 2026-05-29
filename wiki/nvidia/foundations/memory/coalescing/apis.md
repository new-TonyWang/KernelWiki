---
func_name: __ldg
namespace: runtime
header: cuda_runtime.h
signature: T __ldg(const T* address)
status: documented
has_end_to_end_example: false
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24550-L24554
  excerpt: The function __ldg() performs a read-only L1/Tex cache load. It supports
    all C++ fundamental types, CUDA vector types (except x3 components), and extended
    floating-point types.
id: api-coalescing-ref
type: api-definition
vendor: nvidia
title: Apis
---
# Global Memory Coalescing API Reference

This file lists the APIs touched by the coalescing skill. Each entry records the function name, namespace, and signature as found in upstream documentation.

## Read-Only Load

### `__ldg`

- **Namespace**: runtime (CUDA C++ intrinsic)
- **Header**: `cuda_runtime.h` (via `sm_32_intrinsics.h`)
- **Signature**: `T __ldg(const T* address)`
- **PTX**: `ld.global.nc` (non-coherent load through read-only / texture cache)
- **Semantics**: Loads a value from global memory through the read-only data cache (L1/Tex cache path). The data must not be modified during the kernel's lifetime. On sm_35+, this provides a separate cache path from the normal L1, reducing pressure on the L1 and potentially improving bandwidth for read-only data. On sm_70+ (including H200), the compiler automatically generates `ld.global.nc` for `const __restrict__` pointers, making explicit `__ldg()` calls unnecessary in most cases.
- **Supported types**: all C++ fundamental types, CUDA vector types (float2, float4, int2, int4, etc., except x3 variants), `__half`, `__half2`, `__nv_bfloat16`, `__nv_bfloat162`.
- **Relation to coalescing**: `__ldg()` does not change the coalescing behavior -- the same 32-byte transaction rules apply. However, the separate cache path can improve effective bandwidth when the working set exceeds L1 capacity.
- **Source**: Programming guide L24550-L24554

### `__restrict__` pointer qualifier

- **Namespace**: runtime (C++ extension)
- **Header**: none (language keyword recognized by `nvcc`)
- **Signature**: `void kernel(const float* __restrict__ in, float* __restrict__ out)`
- **Semantics**: Promises that the pointer does not alias other pointers, enabling the compiler to: (a) cache loads in registers without re-reading, (b) reorder loads and stores, and (c) for `const __restrict__` pointers, automatically use `ld.global.nc` (equivalent to `__ldg()`).
- **Relation to coalescing**: `__restrict__` does not directly affect coalescing, but the resulting `ld.global.nc` loads use the read-only cache path, improving bandwidth utilization for read-heavy kernels.
- **Source**: Programming guide L22292-L22350

## Vector Types for Wider Loads

### `float4`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h` (included via `cuda_runtime.h`)
- **Size**: 16 bytes, alignment 16 bytes
- **Members**: `.x`, `.y`, `.z`, `.w`
- **Constructor**: `make_float4(x, y, z, w)`
- **Semantics**: A 128-bit vector type. When used for global memory access, a `float4` load compiles to a single 128-bit (`LDG.128`) instruction. A warp of 32 threads each loading one `float4` issues 32 x 16 = 512 bytes, serviced by 16 coalesced 32-byte transactions. This is perfectly coalesced and transfers 4x the data per transaction compared to scalar `float` loads.
- **Usage pattern**:
  ```cuda
  __global__ void vec_load(const float4* __restrict__ in, float* out, int n) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n / 4) {
          float4 v = in[tid];
          out[tid * 4 + 0] = v.x;
          out[tid * 4 + 1] = v.y;
          out[tid * 4 + 2] = v.z;
          out[tid * 4 + 3] = v.w;
      }
  }
  ```
- **Source**: Programming guide L22699 (float4 size=16, alignment=16)

### `float2`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 8 bytes, alignment 8 bytes
- **Members**: `.x`, `.y`
- **Constructor**: `make_float2(x, y)`
- **Semantics**: A 64-bit vector type. A `float2` load compiles to a single 64-bit (`LDG.64`) instruction. A warp of 32 threads each loading one `float2` issues 32 x 8 = 256 bytes, serviced by 8 coalesced 32-byte transactions. Useful when the data count is not a multiple of 4 but is a multiple of 2.
- **Source**: Programming guide L22691 (float2 size=8, alignment=8)

### `int4` / `uint4`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 16 bytes, alignment 16 bytes
- **Members**: `.x`, `.y`, `.z`, `.w`
- **Constructor**: `make_int4(x, y, z, w)` / `make_uint4(x, y, z, w)`
- **Semantics**: 128-bit integer vector types. Same coalescing properties as `float4`. Often used as a generic 16-byte load type via `reinterpret_cast<const int4*>(ptr)` for type-punned vectorized loads.
- **Source**: Programming guide L22635 (int4 size=16, alignment=16)

## Other Cache-Hint Load Intrinsics

### `__ldcg`, `__ldca`, `__ldcs`, `__ldlu`, `__ldcv`

- **Namespace**: runtime
- **Header**: `cuda_runtime.h`
- **Signatures**:
  - `T __ldcg(const T* address)` -- cache at global level, evict from L1
  - `T __ldca(const T* address)` -- cache at all levels
  - `T __ldcs(const T* address)` -- streaming load, evict-first
  - `T __ldlu(const T* address)` -- last-use hint
  - `T __ldcv(const T* address)` -- volatile (bypass cache)
- **Relation to coalescing**: These do not change coalescing behavior; they control cache residency policy. `__ldcs` is useful for streaming kernels (elementwise, reduction) where data is read once and should not pollute L2. `__ldcg` bypasses L1 but keeps data in L2.
- **Source**: Programming guide L24559-L24563

## Related Probes

- [`__ldg`](../../../sources/experience/api-probes/2026-04-16-runtime-ldg.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__ldcv`](../../../sources/experience/api-probes/2026-04-17-runtime-ldcv.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__ldca`](../../../sources/experience/api-probes/2026-04-17-runtime-ldca.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__ldcs`](../../../sources/experience/api-probes/2026-04-17-runtime-ldcs.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__ldcg`](../../../sources/experience/api-probes/2026-04-17-runtime-ldcg.md) — end-to-end example, build command, and H200 measurement. See probe record.
- [`__ldlu`](../../../sources/experience/api-probes/2026-04-17-runtime-ldlu.md) — end-to-end example, build command, and H200 measurement. See probe record.
