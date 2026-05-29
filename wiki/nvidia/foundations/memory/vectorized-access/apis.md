---
func_name: float4
namespace: runtime
header: vector_types.h
signature: struct float4 { float x, y, z, w; }
status: documented
has_end_to_end_example: false
source:
- path: spec
  anchor: Reference
id: api-vectorized-access-ref
type: api-definition
vendor: nvidia
title: Apis
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22574-L22702
---
# Vectorized Memory Access API Reference

This file lists the APIs touched by the vectorized-access skill. Each entry records the type name, namespace, and relevant properties as found in upstream documentation. All size and alignment values are from the programming guide Table 42 (L22578).

## 128-bit Vector Types (16 bytes)

### `float4`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h` (included via `cuda_runtime.h`)
- **Size**: 16 bytes
- **Alignment**: 16 bytes
- **Members**: `.x`, `.y`, `.z`, `.w` (each `float`)
- **Constructor**: `make_float4(float x, float y, float z, float w)`
- **PTX load instruction**: `LDG.128` (128-bit global load)
- **PTX store instruction**: `STG.128` (128-bit global store)
- **Semantics**: The primary 128-bit vector type for float data. When used for global memory access, a `float4` load compiles to a single 128-bit instruction. A warp of 32 threads each loading one `float4` requests 32 x 16 = 512 bytes, serviced by 16 coalesced 32-byte transactions.
- **Source**: Programming guide L22699 (float4 size=16, alignment=16), L22735-L22753 (vector type members and constructors)

### `int4`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 16 bytes
- **Alignment**: 16 bytes
- **Members**: `.x`, `.y`, `.z`, `.w` (each `signed int`)
- **Constructor**: `make_int4(int x, int y, int z, int w)`
- **Semantics**: 128-bit integer vector type. Often used as a generic 16-byte load type via `reinterpret_cast<const int4*>(ptr)` for type-punned vectorized loads when the actual data type differs from float.
- **Source**: Programming guide L22635 (int4 size=16, alignment=16)

### `uint4`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 16 bytes
- **Alignment**: 16 bytes
- **Members**: `.x`, `.y`, `.z`, `.w` (each `unsigned int`)
- **Constructor**: `make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)`
- **Semantics**: Unsigned 128-bit integer vector type. Same memory behavior as `int4`. Commonly used for raw byte loading of 16-byte chunks when the data needs manual unpacking (e.g., loading 8 half-precision values as one `uint4` and then reinterpreting the components).
- **Source**: Programming guide L22635 (uint4 size=16, alignment=16)

### `longlong2`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 16 bytes
- **Alignment**: 16 bytes
- **Members**: `.x`, `.y` (each `signed long long`)
- **Constructor**: `make_longlong2(long long x, long long y)`
- **Semantics**: 128-bit vector of two 64-bit integers. Generates `LDG.128`. Useful for type-punned 128-bit loads when the data elements are 8 bytes each (e.g., `double`, `int64_t`).
- **Source**: Programming guide L22667-L22670 (longlong2 size=16, alignment=16)

### `double2`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 16 bytes
- **Alignment**: 16 bytes
- **Members**: `.x`, `.y` (each `double`)
- **Constructor**: `make_double2(double x, double y)`
- **Semantics**: 128-bit vector of two double-precision floats. Generates `LDG.128`. This is the widest available vector load for double data (no `double4` with 16-byte alignment exists in the standard types).
- **Source**: Programming guide L22707-L22710 (double2 size=16, alignment=16)

## 64-bit Vector Types (8 bytes)

### `float2`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 8 bytes
- **Alignment**: 8 bytes
- **Members**: `.x`, `.y` (each `float`)
- **Constructor**: `make_float2(float x, float y)`
- **PTX load instruction**: `LDG.64` (64-bit global load)
- **Semantics**: A 64-bit vector type. A `float2` load compiles to a single 64-bit instruction. A warp of 32 threads each loading one `float2` issues 32 x 8 = 256 bytes, serviced by 8 coalesced 32-byte transactions. Useful when the data count is not a multiple of 4 but is a multiple of 2, or when 128-bit loads would cause excessive register pressure.
- **Source**: Programming guide L22691 (float2 size=8, alignment=8)

### `int2` / `uint2`

- **Namespace**: runtime (built-in type)
- **Header**: `vector_types.h`
- **Size**: 8 bytes
- **Alignment**: 8 bytes
- **Members**: `.x`, `.y` (each `int` / `unsigned int`)
- **Constructor**: `make_int2(int x, int y)` / `make_uint2(unsigned int x, unsigned int y)`
- **Semantics**: 64-bit integer vector types. Generate `LDG.64`. Often used for 8-byte type-punned loads.
- **Source**: Programming guide L22627-L22630 (int2/uint2 size=8, alignment=8)

## Constructor Functions

### `make_float4`

- **Namespace**: runtime
- **Header**: `vector_types.h`
- **Signature**: `float4 make_float4(float x, float y, float z, float w)`
- **Semantics**: Factory function that constructs a `float4` value from four float components.
- **Source**: Programming guide L22744-L22753

### `make_float2`

- **Namespace**: runtime
- **Header**: `vector_types.h`
- **Signature**: `float2 make_float2(float x, float y)`
- **Semantics**: Factory function that constructs a `float2` value from two float components.
- **Source**: Programming guide L22744-L22753

### `make_int4`

- **Namespace**: runtime
- **Header**: `vector_types.h`
- **Signature**: `int4 make_int4(int x, int y, int z, int w)`
- **Semantics**: Factory function that constructs an `int4` value.
- **Source**: Programming guide L22744-L22753

### `make_uint4`

- **Namespace**: runtime
- **Header**: `vector_types.h`
- **Signature**: `uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)`
- **Semantics**: Factory function that constructs a `uint4` value.
- **Source**: Programming guide L22744-L22753
