---
title: Vectorized Memory Access
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- scan
requires_sm: '>=6.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22574-L22702
  excerpt: 'The following table details the byte size and alignment requirements of
    the vector types. float4: size 16, alignment 16.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22735-L22753
  excerpt: Vector types are structures. Their first, second, third, and fourth components
    are accessible through the x, y, z, and w fields, respectively. They all have
    a factory function of the form make_<type_name>().
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L991-L1001
  excerpt: We evaluate the performance of both kernels using elements of size 4B,
    8B and 16B per thread i.e., using int, int2 and int4 for the template parameter.
    Overall, best performance is achieved when using asynchronous copies with an element
    of size 8 or 16 bytes.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1062
  excerpt: In particular, there is no register-related reason to pack data into vector
    data types such as float4 or int4 types.
artifacts:
  code: sources/experience/hw-probes/vectorized-access/artifacts/vectorized_load_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o vectorized_load_probe vectorized_load_probe.cu
  introspection: ''
  profile: ''
related_apis:
- float4
- float2
- int4
- uint4
- make_float4
- make_float2
- make_int4
- make_uint4
related_skills:
- coalescing
- bank-conflict
id: skill-vectorized-access
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

Vectorized memory access is a technique where each CUDA thread loads or stores multiple elements in a single memory instruction by using CUDA's built-in vector types (`float4`, `float2`, `int4`, `uint4`, `longlong2`, etc.). Instead of each thread issuing a 32-bit (4-byte) scalar load, a thread issuing a `float4` load reads 128 bits (16 bytes) in one instruction. The compiler generates a single wide memory instruction (`LDG.128` for 128-bit loads, `LDG.64` for 64-bit loads) rather than multiple narrow instructions.

The key vector types and their memory widths:

| Type | Size (bytes) | Alignment (bytes) | Load instruction |
|---|---|---|---|
| `float` (scalar) | 4 | 4 | `LDG.32` |
| `float2` | 8 | 8 | `LDG.64` |
| `float4` | 16 | 16 | `LDG.128` |
| `int4` / `uint4` | 16 | 16 | `LDG.128` |
| `longlong2` | 16 | 16 | `LDG.128` |
| `double2` | 16 | 16 | `LDG.128` |

The alignment requirements are defined in the programming guide (Table 42, L22578): `float4` requires 16-byte alignment, `float2` requires 8-byte alignment. Memory allocated by `cudaMalloc` is guaranteed to be at least 256-byte aligned, so the base pointer always satisfies these requirements.

The canonical usage pattern is to `reinterpret_cast` a `float*` pointer to a `float4*` pointer and load one `float4` per thread:

```cuda
__global__ void vec_kernel(const float* __restrict__ in,
                           float*       __restrict__ out,
                           int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n / 4) {
        float4 v = reinterpret_cast<const float4*>(in)[tid];
        // Process v.x, v.y, v.z, v.w
        float result = v.x + v.y + v.z + v.w;
        out[tid] = result;
    }
}
```

## Why

Vectorized access improves memory-bound kernel performance by reducing the number of load/store instructions the SM must issue to move the same amount of data. On H200, our microbenchmark measured a **2.59x effective bandwidth improvement** with float4 loads compared to scalar float loads (see Measured Characteristics below).

The improvement comes from two effects:

1. **Fewer instructions per byte**: a `float4` load moves 16 bytes with one instruction, vs. 4 bytes with one scalar load. This reduces pressure on the SM's instruction scheduler and load/store units. The scalar kernel must issue 4x more load instructions, which can saturate the instruction issue rate before saturating the memory bandwidth.

2. **Higher per-warp transaction efficiency**: a warp of 32 threads each loading a `float4` requests 32 x 16 = 512 bytes, serviced by 16 coalesced 32-byte transactions. While this is more transactions than the scalar case (32 x 4 = 128 bytes, 4 transactions), each transaction is fully utilized. The net effect is that the vectorized kernel saturates the memory subsystem with fewer warps and fewer instructions.

The best-practices guide (L1062) notes that there is "no register-related reason to pack data into vector data types such as `float4` or `int4` types." The benefit is purely on the memory-instruction side: fewer load instructions to move the same data.

The best-practices guide (L991-L1001) also observes that "best performance is achieved when using asynchronous copies with an element of size 8 or 16 bytes," reinforcing that 128-bit-wide memory operations are the highest-throughput path on modern NVIDIA GPUs.

## When to use

- **Elementwise kernels** (activation, scaling, bias-add, fused pointwise): these kernels are typically memory-bound and each thread processes a contiguous span of elements. Replacing scalar loads with `float4` loads is straightforward and yields immediate benefit.

- **Reduction input phase**: the initial load of reduction input from global memory benefits from vectorized access. Each thread loads 4 values at once and accumulates them locally before the warp/block reduction.

- **Normalization kernels** (layer-norm, batch-norm): the per-element read/write phases are memory-bound and benefit from vectorized access.

- **Any kernel where N is divisible by the vector width** (4 for float4, 2 for float2): the element count must divide evenly by the vector width to avoid out-of-bounds reads. If it does not divide evenly, a scalar tail loop is needed (see pitfalls.md).

- **Already-coalesced kernels**: vectorized access is a "next-level" optimization on top of coalescing. A coalesced scalar kernel (stride-1) can often be further improved by switching to vectorized loads.

## When NOT to use

- **N is not a multiple of the vector width and tail handling is complex**: if the data layout makes it difficult to handle the remainder elements cleanly, the code complexity may not justify the benefit. In practice, padding the input to a multiple of 4 is the simplest solution.

- **Gather/scatter with non-contiguous access**: vectorized loads only help when consecutive threads access consecutive memory. If the access pattern is indexed or strided, casting to `float4*` will load the wrong data. The memory must be contiguous per-thread-group.

- **Register pressure is already critical**: each `float4` occupies 4 registers (one per component). If the kernel is already register-bound and close to the occupancy limit, adding vector loads may push it over the register threshold and reduce occupancy, negating the benefit.

- **Data type is not 4-byte aligned or has odd sizes**: `float4` requires 16-byte alignment. If the underlying data type is not a multiple of 4 bytes (e.g., 3-byte packed RGB), vector loads cannot be used directly. A type-punned load via `uint4` may work but requires care.

- **The kernel is compute-bound**: if the kernel's bottleneck is arithmetic throughput rather than memory bandwidth, vectorized loads will not help.

## Classical example: scalar vs vectorized elementwise add

```cuda
// Scalar version: each thread loads and stores one float
__global__ void vecadd_scalar(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float*       __restrict__ c,
                              int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

// Vectorized version: each thread loads and stores one float4 (4 floats)
__global__ void vecadd_float4(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float*       __restrict__ c,
                              int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n / 4) {
        float4 va = reinterpret_cast<const float4*>(a)[tid];
        float4 vb = reinterpret_cast<const float4*>(b)[tid];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4*>(c)[tid] = vc;
    }
}
```

The vectorized version uses `n/4` threads (launched with `n/4` total threads) instead of `n` threads. Each thread issues one `LDG.128` per input array and one `STG.128` for the output, versus four `LDG.32` and four `STG.32` in the scalar version.

## Tail handling pattern

When `n` is not guaranteed to be a multiple of 4:

```cuda
__global__ void vecadd_with_tail(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float*       __restrict__ c,
                                 int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n / 4;
    // Vectorized body
    if (tid < n4) {
        float4 va = reinterpret_cast<const float4*>(a)[tid];
        float4 vb = reinterpret_cast<const float4*>(b)[tid];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4*>(c)[tid] = vc;
    }
    // Scalar tail for remaining elements
    int tail_start = n4 * 4;
    int tail_idx = tail_start + tid;
    if (tid < (n - tail_start) && tail_idx < n) {
        c[tail_idx] = a[tail_idx] + b[tail_idx];
    }
}
```

## Measured Characteristics

- [vectorized-access bandwidth probe](../../sources/experience/hw-probes/vectorized-access/2026-04-15-vectorized-access.md): On H200 (sm_90a, CUDA 12.9), scalar float loads achieved an effective bandwidth of 1462.96 GB/s (median latency 0.1835 ms for 256 MB), while float4 vectorized loads achieved 3783.77 GB/s (median latency 0.0709 ms for the same 256 MB). The vectorized access was **2.59x faster** than scalar access. The working set (256 MB) exceeded the L2 cache (51.2 MB), ensuring HBM-bound measurement. The vectorized kernel reached 77% of theoretical HBM peak bandwidth (4916.7 GB/s), while the scalar kernel reached only 30%.
