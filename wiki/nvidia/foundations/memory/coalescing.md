---
title: Global Memory Coalescing
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- scan
- gather
- scatter
requires_sm: '>=6.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1379-L1411
  excerpt: Global memory is accessed via 32-byte memory transactions. When a CUDA
    thread requests a word of data from global memory, the relevant warp coalesces
    the memory requests from all the threads in that warp into the number of memory
    transactions necessary to satisfy the request.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L539-L542
  excerpt: A very important performance consideration in programming for CUDA-capable
    GPU architectures is the coalescing of global memory accesses. Global memory loads
    and stores by threads of a warp are coalesced by the device into as few as possible
    transactions.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L545-L567
  excerpt: 'For devices of compute capability 6.0 or higher, the requirements can
    be summarized quite easily: the concurrent accesses of the threads of a warp will
    coalesce into a number of transactions equal to the number of 32-byte transactions
    necessary to service all of the threads of the warp.'
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
  excerpt: As illustrated in Figure 7, non-unit-stride global memory accesses should
    be avoided whenever possible.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1446-L1448
  excerpt: To alleviate these uncoalesced writes, the use of shared memory can be
    employed.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22350
  excerpt: Accesses to __global__ function const pointers marked with __restrict__
    are compiled as read-only cache loads, similar to the PTX ld.global.nc or __ldg()
    instructions.
artifacts:
  code: sources/experience/hw-probes/coalescing-stride/artifacts/coalescing_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o coalescing_probe coalescing_probe.cu
  introspection: ''
  profile: ''
related_apis:
- __ldg
- float4
- float2
- __restrict__
related_skills:
- vectorized-access
- bank-conflict
- warp-primitives
id: skill-coalescing
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
---
## What

Global memory coalescing is the hardware mechanism by which a warp's individual per-thread memory requests are combined into the minimum number of memory transactions. On compute capability 6.0 and above (including H200 / sm_90a), global memory is accessed via **32-byte transactions**. When consecutive threads in a warp access consecutive 4-byte words in a 32-byte aligned region, the warp's 32 requests (128 bytes total) are serviced by exactly **four 32-byte transactions** -- this is a perfectly coalesced access.

The coalescing unit operates at warp granularity. The hardware inspects the addresses requested by all 32 threads in the warp, computes which 32-byte segments are touched, and issues one transaction per segment. The key insight: it does not matter which thread accesses which word within a segment -- what matters is how many distinct segments the warp touches.

**Transaction math** (for 4-byte words, compute capability >= 6.0):
- Coalesced (stride-1): 32 threads x 4 bytes = 128 bytes, spanning 4 segments. 4 transactions, 100% utilization.
- Stride-2: 32 threads touch 8 segments. 8 transactions, 50% utilization.
- Stride-32: 32 threads touch 32 segments. 32 transactions, 12.5% utilization -- the pathological worst case.

On devices with L1 caching enabled (which is the default on sm_60+), the data access unit is 32 bytes regardless of whether global loads are cached in L1 or not (best-practices guide L549).

## Why

Memory coalescing is the single most impactful optimization for memory-bound CUDA-core kernels. The programming guide (L1411) states: "Ensuring proper coalescing of global memory accesses is one of the most important performance considerations for writing performant CUDA kernels."

On H200, our microbenchmark probe measured a **2.85x slowdown** for stride-32 access compared to stride-1 access (see Measured Characteristics below). This means that a kernel with non-coalesced access is leaving approximately 65% of its potential memory bandwidth on the table.

The cost of non-coalesced access arises from two effects:
1. **Wasted bandwidth**: each 32-byte transaction fetches a full sector, but only a fraction of the bytes are used.
2. **Increased transaction count**: more transactions compete for the memory subsystem's finite transaction throughput.

## When to use

Coalescing awareness should be applied to **every** kernel that accesses global memory. Specific scenarios where explicit attention is needed:

- **Elementwise kernels** (vecadd, activation, etc.): ensure thread indexing is `tid = blockIdx.x * blockDim.x + threadIdx.x` and array access is `a[tid]` (stride-1).

- **Reduction kernels**: the input-load phase should be coalesced. Even when the reduction itself uses shared memory or warp shuffles, the initial global-to-register load pattern matters.

- **Matrix operations with 2D thread blocks**: when `threadIdx.x` is the fastest-moving dimension, ensure it maps to the contiguous dimension of the matrix (the column index in row-major storage).

- **SoA vs AoS layout**: for data with multiple fields per element (e.g., RGB pixels, 3D coordinates), use Structure-of-Arrays layout so that all threads read the same field contiguously, rather than Array-of-Structures where consecutive threads stride across multiple fields.

- **Vectorized loads** (float4, float2): when each thread processes multiple elements, casting the pointer to `float4*` and loading one `float4` per thread generates a single 128-bit (16-byte) load instruction. A warp of 32 threads loading `float4` values contiguously issues 32 x 16 = 512 bytes in perfectly coalesced 128-bit transactions. This is the "next level" of coalescing -- see the vectorized-access skill.

- **Read-only data**: marking input pointers with `const ... __restrict__` or using `__ldg()` routes loads through the read-only data cache (L1/Tex cache), which can improve effective bandwidth for data read by many warps.

## When NOT to use

Coalescing is an inherent hardware behavior, not an optional optimization. However, there are situations where achieving perfect coalescing is not straightforward or not the primary concern:

- **Gather/scatter patterns**: when the access pattern is determined by an index array (indirect addressing), coalescing depends on the index distribution. If the indices are random, coalescing is poor and no simple code change fixes it -- the algorithmic design must change (e.g., sorting by key, binning).

- **Matrix transpose**: writing the transposed output inherently requires non-coalesced writes (the write index strides by the leading dimension). The standard fix is to use shared memory as a staging buffer: load a tile with coalesced reads, transpose in shared memory, then write with coalesced writes. This is documented in the programming guide at L1413-L1448.

- **Small data per thread**: if each thread accesses only 1 byte and the warp accesses 32 bytes total, one 32-byte transaction suffices regardless of layout. Coalescing only matters when the total warp footprint exceeds one segment.

## Classical example: coalesced vs non-coalesced copy

The following pair of kernels demonstrates the difference between coalesced and non-coalesced access (adapted from the best-practices guide L584-L617):

```cuda
// GOOD: coalesced -- consecutive threads read consecutive elements
__global__ void coalescedCopy(float* odata, const float* idata, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        odata[tid] = idata[tid];
}

// BAD: strided -- consecutive threads read elements stride apart
__global__ void stridedCopy(float* odata, const float* idata,
                            int stride, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    if (idx < n)
        odata[idx] = idata[idx];
}
```

With `stride = 32`, the strided kernel touches 32 distinct 32-byte segments per warp, fetching 1024 bytes but using only 128 bytes -- a 12.5% memory utilization rate.

## Shared memory staging for transpose

When an algorithm requires non-coalesced writes (such as matrix transpose), the canonical solution is to use shared memory as a coalescing buffer:

```cuda
__global__ void transpose_coalesced(float*       odata,
                                    const float* idata,
                                    int width) {
    __shared__ float tile[32][33];  // +1 padding avoids bank conflicts
    int xIdx = blockIdx.x * 32 + threadIdx.x;
    int yIdx = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from input (threadIdx.x is contiguous dimension)
    if (xIdx < width && yIdx < width)
        tile[threadIdx.y][threadIdx.x] = idata[yIdx * width + xIdx];
    __syncthreads();

    // Coalesced write to output (threadIdx.x maps to contiguous output)
    xIdx = blockIdx.y * 32 + threadIdx.x;
    yIdx = blockIdx.x * 32 + threadIdx.y;
    if (xIdx < width && yIdx < width)
        odata[yIdx * width + xIdx] = tile[threadIdx.x][threadIdx.y];
}
```

The read of `idata` is coalesced because `threadIdx.x` maps to the column index. The write of `odata` is also coalesced because, after the shared memory transpose, `threadIdx.x` again maps to the contiguous output dimension. The `[32][33]` padding avoids shared-memory bank conflicts (see the bank-conflict skill).

## Measured Characteristics

- coalescing-stride bandwidth probe: On H200 (sm_90a, CUDA 12.9), stride-1 (coalesced) access achieved an effective bandwidth of 530.66 GB/s (median latency 0.0079 ms), while stride-32 (non-coalesced) access achieved only 186.18 GB/s (median latency 0.0225 ms). The strided access was **2.85x slower** than coalesced access. With stride-32, each warp issues 32 separate 32-byte transactions for 128 bytes of useful data, yielding 12.5% memory utilization.
