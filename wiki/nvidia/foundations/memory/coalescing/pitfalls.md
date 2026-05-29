---
title: Global Memory Coalescing - Pitfalls
status: draft
evidence_level: spec
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
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
id: pitfall-coalescing
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L606-L634
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1389-L1393
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L1446-L1448
- source_id: cuda-official/toolkit-docs-13.2
  path: CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L569-L603
---
## P1: Array-of-Structures (AoS) layout

**Symptom**: Kernel achieves far less than expected memory bandwidth despite sequential thread indexing.

**Cause**: When data is stored as an array of structures (e.g., `struct { float x, y, z; } points[N]`), consecutive threads reading `points[tid]` each load a full struct. The hardware sees addresses spaced `sizeof(struct)` apart per thread. For a 12-byte struct, the warp touches more 32-byte segments than necessary because the x, y, and z fields of different elements are interleaved.

**Example**:
```cuda
// BAD: AoS -- threads read interleaved fields
struct Point { float x, y, z; };
__global__ void process_aos(Point* points, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = points[tid].x + points[tid].y;  // stride = 12 bytes
}

// GOOD: SoA -- each field is contiguous
__global__ void process_soa(float* x, float* y, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = x[tid] + y[tid];  // stride = 4 bytes (coalesced)
}
```

**Fix**: Convert to Structure-of-Arrays (SoA) layout: separate arrays for each field. If AoS is required by an external interface, load the data into separate shared memory arrays (SoA in shared memory) as a staging step.

**Detection**: Profile with `ncu` and check the `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` metric. If it is significantly higher than `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum * 4` (for float), the access is not coalesced.

## P2: Row-major 2D access with wrong thread dimension mapping

**Symptom**: Matrix kernels run at a fraction of expected bandwidth.

**Cause**: In a 2D thread block, `threadIdx.x` is the fastest-varying dimension. If `threadIdx.x` maps to the row index and `threadIdx.y` to the column index of a row-major matrix, consecutive threads (consecutive `threadIdx.x` values) access elements that are `width` apart in memory -- a large stride.

**Example**:
```cuda
// BAD: threadIdx.x maps to row (large stride in row-major)
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;
float val = matrix[row * width + col];

// GOOD: threadIdx.x maps to column (contiguous in row-major)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float val = matrix[row * width + col];
```

**Fix**: Ensure `threadIdx.x` maps to the contiguous storage dimension (column for row-major, row for column-major).

## P3: Misaligned base addresses

**Symptom**: 5 transactions instead of 4 for a warp reading 32 consecutive floats.

**Cause**: When the starting address of the warp's access is not aligned to a 32-byte boundary, the accessed range spans one extra 32-byte segment. For example, if the base address has a 4-byte offset, the 128 bytes accessed by the warp span 5 segments instead of 4 (best-practices guide L569-L577).

**Fix**: Use `cudaMalloc` (which returns 256-byte-aligned pointers) and ensure that per-warp offsets maintain alignment. Common violations: manually computed offsets (e.g., `ptr + 1` for a float array shifts all subsequent warps by 4 bytes), or packed struct arrays with non-power-of-two sizes.

**Mitigation**: On sm_60+ with L1 caching, the impact is modest (~10% bandwidth loss per the best-practices guide L603-604) because adjacent warps reuse the over-fetched cache lines. Still, aligning base addresses is a free optimization.

## P4: Indirect (gather) access with random indices

**Symptom**: Memory bandwidth is 10-30x below peak despite the kernel being memory-bound.

**Cause**: When threads access `data[index[tid]]` and the `index` array contains scattered values, the warp touches up to 32 distinct cache lines. No code-level change to the kernel can fix this -- the access pattern is inherently non-coalesced.

**Fix**: Algorithmic approaches:
- Sort the work items by their index values so that nearby threads access nearby memory (e.g., sort particles by cell index before a stencil kernel).
- Use binning / tiling to group elements that access the same region.
- Pre-fetch into shared memory if the same data is accessed multiple times.

**Detection**: `ncu` metric `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` significantly below 100% indicates poor coalescing.

## P5: Writing the transpose without shared memory staging

**Symptom**: Transpose kernel achieves ~12.5% of peak bandwidth.

**Cause**: A naive transpose `C[col * N + row] = A[row * N + col]` has coalesced reads (consecutive `threadIdx.x` maps to consecutive columns in row-major A) but non-coalesced writes (consecutive `threadIdx.x` maps to rows in the output, which are `N` elements apart). The write side issues up to 32 transactions per warp.

**Fix**: Load a tile into shared memory with coalesced reads, synchronize, then write from shared memory with coalesced writes (swapping row/column indices). This is the canonical shared-memory-tiled transpose (programming guide L1486-L1601).

## P6: Block size not a multiple of warp size

**Symptom**: Under-utilized warps at block boundaries, poor coalescing at the edges.

**Cause**: The best-practices guide (L1118) states: "The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing." With non-multiple block sizes, the last warp in each block is partially filled, wasting memory transaction capacity.

**Fix**: Always use block sizes that are multiples of 32 (common choices: 128, 256, 512).

## P7: Confusing L1 cache rescue with actual coalescing

**Symptom**: Slightly misaligned or stride-2 kernels perform better than expected, leading developers to ignore coalescing.

**Cause**: On sm_60+, L1 caching is enabled by default. Adjacent warps often reuse cache lines fetched by neighbors, partially masking the cost of non-coalesced access (best-practices guide L604: "the impact is not as large as we might have expected"). This can make stride-2 access appear to lose only ~10-20% bandwidth rather than the theoretical 50%.

**Risk**: Relying on L1 cache rescue is fragile. With larger strides, random access, or working sets that exceed L1 capacity, the cache cannot help and the full coalescing penalty applies.

**Fix**: Always write coalesced access patterns. Treat L1 cache rescue as a safety net, not a strategy.

## P8: Forgetting to coalesce the store side

**Symptom**: Kernel reads are coalesced but overall bandwidth is poor.

**Cause**: Developers focus on coalescing reads but forget that stores follow the same rules. Non-coalesced stores are equally costly (and on GDDR memory with ECC, even worse due to read-modify-write overhead for partial sector writes -- best-practices guide L551).

**Fix**: Apply the same stride-1 mapping to output arrays. If the algorithm inherently requires non-coalesced output (e.g., scatter), use shared memory staging or atomics as appropriate.
