---
title: Bank Conflict Avoidance
status: draft
evidence_level: measured
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- reduction
- scan
- transpose
- stencil
- normalization
- elementwise
requires_sm: '>=5.0'
requires_features: []
single_kernel_useful: true
cuda_version_tested: '12.9'
driver_version_tested: 570.124.06
toolchain: nvcc 12.9 + ptxas 12.9
measured_on: H200-SXM
source:
- path: spec
  anchor: Reference
artifacts:
  code: sources/experience/hw-probes/smem-bank-conflict/artifacts/smem_bank_conflict_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o smem_bank_conflict_probe smem_bank_conflict_probe.cu
  introspection: ''
  profile: ''
related_apis: []
related_skills:
- coalescing
- vectorized-access
- warp-primitives
id: skill-bank-conflict
type: skill
vendor: nvidia
tags:
- cuda-cpp
applies_to:
- general
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1450-L1482
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1452-L1454
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1604-L1655
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L720-L726
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L892-L900
---
## What

Bank conflict avoidance is a shared-memory layout technique that ensures threads within a warp access different memory banks when loading from or storing to `__shared__` memory.

Shared memory on all CUDA devices with compute capability >= 5.0 (including H200, sm_90a) is organized into **32 banks**, each 32 bits (4 bytes) wide. Successive 32-bit words are assigned to successive banks in round-robin fashion:

```
Address (bytes):   0    4    8   12   ...  124  128  132  ...
Bank:              0    1    2    3   ...   31    0    1   ...
```

When a warp issues a shared-memory load or store:

- **No conflict**: if all 32 threads access 32 distinct banks, the entire request is serviced in a single transaction.
- **N-way conflict**: if N threads access **different addresses** that map to the **same bank**, the hardware serializes the access into multiple rounds. This reduces effective bandwidth.
- **Broadcast**: if multiple threads read the **same address** in the same bank, the word is multicast to all requesting threads with no penalty.

The bank index for a 32-bit word at byte address `A` is:

```
bank(A) = (A / 4) % 32
```

## Why

Bank conflicts directly multiply the latency of shared-memory accesses.  On H200, a conflict-free dependent `ld.shared` takes approximately **30 cycles** (measured).  A 32-way bank conflict -- where all 32 threads hit the same bank with different addresses -- raises this to **91 cycles** (3.03x baseline), representing significant serialization overhead.

In bandwidth-sensitive kernels that rely on shared memory as a scratchpad (matrix transpose, tiled GEMM, reduction, stencil), bank conflicts can eliminate most of the bandwidth advantage of shared memory over global memory. The CUDA Best Practices Guide reports that removing 32-way bank conflicts from a matrix transpose kernel improved effective bandwidth from 140.2 GB/s to 199.4 GB/s on Tesla V100 (~42% improvement).

## When to use

Apply bank conflict avoidance whenever a kernel uses `__shared__` memory and the access pattern has a stride that is a multiple of 32 elements (128 bytes) per warp.  Common scenarios:

- **Matrix transpose**: storing to `smem[threadIdx.x][threadIdx.y]` when `threadIdx.x` is the fast-moving dimension creates stride-32 access patterns that cause 32-way bank conflicts.

- **Tiled matrix multiplication (C=AAT)**: copying a column of A into a column of shared memory while using row-major layout creates stride-32 writes.

- **Reduction to shared memory**: writing partial sums to `smem[warpId]` when 32 warps are active causes all warps' lane-0 threads to hit bank 0.

- **Column-wise access of 2D shared arrays**: any `smem[row][col]` access where consecutive threads vary `row` (with a fixed `col`) produces a stride equal to the row width, which is a 32-way conflict when the width is a multiple of 32.

## When NOT to use

- **No shared memory**: if the kernel does not use `__shared__` memory, bank conflicts do not apply.

- **Stride-1 access**: if consecutive threads read consecutive 32-bit words (`smem[threadIdx.x]` style), the access is already conflict-free.

- **Broadcast pattern**: if all threads read the same shared-memory address, the hardware broadcasts the value with no penalty.

- **Global-memory-bound kernels**: if the kernel is bottlenecked by global memory bandwidth and shared memory is used lightly, optimizing bank conflicts may not yield visible speedup.  Profile first with `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` via NCU to confirm conflicts are significant before applying padding.

## Avoidance techniques

### 1. Padding (+1 trick)

The simplest and most common fix.  Add one element of padding to the inner dimension of a 2D shared array:

```cuda
// BEFORE: 32-way bank conflict when accessing column-wise
__shared__ float tile[32][32];

// AFTER: conflict-free -- stride becomes 33, and 33 % 32 = 1
__shared__ float tile[32][33];
```

This works because padding shifts each row by one bank, so column-wise access (stride = row width) no longer maps all threads to the same bank. The extra element per row wastes 4 bytes x 32 rows = 128 bytes of shared memory, which is negligible.

### 2. Index swizzling

For more complex layouts (e.g., matrix tiles loaded via TMA or cp.async), swizzling the shared-memory index remaps addresses so that column-wise and row-wise accesses both avoid conflicts without wasting memory.  The programming guide documents `CU_TENSOR_MAP_SWIZZLE_128B` for this purpose (L12822-L12838).

### 3. Access pattern redesign

Sometimes the access pattern itself can be changed.  For example, instead of writing to `smem[threadIdx.x][threadIdx.y]` (stride-32 in the first index), transpose the indices: `smem[threadIdx.y][threadIdx.x]` (stride-1 in the second index).  This is shown in the programming guide matrix transpose example (L1624-L1639).

## Classical example: padded matrix transpose tile

```cuda
#define TILE 32

__global__ void transpose(const float* __restrict__ in,
                          float*       __restrict__ out,
                          int rows, int cols) {
    // +1 padding eliminates bank conflicts on column-wise read
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Coalesced load from global → conflict-free write to smem row
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();

    // Transposed indices for output
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Column-wise read from smem (conflict-free due to padding)
    // → coalesced write to global
    if (x < rows && y < cols)
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
```

Without the `+1` padding, the `tile[threadIdx.x][threadIdx.y]` read in the second phase would cause a 32-way bank conflict, because consecutive threads (varying `threadIdx.x`) would access addresses with stride 32.

## Measured Characteristics

- smem-bank-conflict stride sweep probe: On H200 (sm_90a, CUDA 12.9), dependent pointer-chasing in shared memory measured **30.01 cycles per load** conflict-free (stride-1), **91.00 cycles** under 32-way bank conflict (stride-32, 3.03x baseline), and **30.01 cycles** after applying the +1 padding trick (stride-33, fully restored).  Broadcast (all threads reading the same address) also measured 30.01 cycles, confirming zero-penalty hardware multicast.  2-way conflicts (stride-2) showed negligible overhead (31.01 cycles, 1.03x).
