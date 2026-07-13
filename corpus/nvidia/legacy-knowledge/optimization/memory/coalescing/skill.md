---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.1 (Coalesced Access to Global Memory)
  - Best Practices Guide, Section 10.2.1.1-10.2.1.4 (Access Patterns)
  - Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)
cross_ref:
  - Best Practices Guide, Section 10.2.3 (Shared Memory)
related_apis:
  - ld.global, st.global, ld.global.v2, ld.global.v4, cudaMallocPitch
related_experience: []
unlocks:
  - "memory/vectorized-access: once coalescing is achieved, widening to vector types further reduces transaction count"
  - "memory/shared-memory-cache: shared memory staging enables coalesced global access for irregular patterns"
conflicts_with:
  - "memory/layout-transform: layout changes may be needed to enable coalescing, adding pre-processing cost"
---

# Coalescing

## Skill 1: Ensure Consecutive Threads Access Consecutive Memory

### When to Use
- Any kernel reading or writing global memory arrays
- Thread index maps directly to data index

### How to Apply
1. Map thread index (threadIdx.x + blockIdx.x * blockDim.x) to array index linearly
2. Ensure the innermost loop dimension uses threadIdx.x for memory indexing
3. For 2D grids, ensure threadIdx.x maps to the fastest-varying (column) dimension

### Code Template
```cuda
__global__ void coalesced_read(const float* __restrict__ data,
                               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data[idx] * 2.0f;
    }
}
```

### Source
Best Practices Guide, Section 10.2.1.1 (Simple Access Pattern)

## Skill 2: Use cudaMallocPitch for 2D Array Coalescing

### When to Use
- Working with 2D arrays where row width is not a multiple of 32 bytes
- Row-major access by consecutive threads along columns

### How to Apply
1. Allocate with `cudaMallocPitch()` to get properly padded rows
2. Use the returned pitch (in bytes) to compute row offsets
3. Each row starts at a properly aligned address

### Code Template
```cuda
float* d_data;
size_t pitch;
cudaMallocPitch(&d_data, &pitch, width * sizeof(float), height);

__global__ void process_2d(float* data, size_t pitch, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        float* row_ptr = (float*)((char*)data + row * pitch);
        row_ptr[col] = row_ptr[col] + 1.0f;
    }
}
```

### Source
Programming Guide, Section 2.2.4.1 (Coalesced Global Memory Access)

## Skill 3: Avoid Strided Access via Shared Memory Staging

### When to Use
- Algorithm requires strided access patterns (e.g., column-major reads in a row-major array)
- Stride > 1 between consecutive threads wastes bandwidth

### How to Apply
1. Load data from global memory in a coalesced pattern into shared memory
2. Synchronize threads
3. Read from shared memory in the desired strided pattern (no penalty)

### Code Template
```cuda
#define TILE 32
__global__ void avoid_strided(const float* __restrict__ in,
                              float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE];
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    if (row < N && col < N)
        tile[threadIdx.y][threadIdx.x] = in[row * N + col];
    __syncthreads();

    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if (row < N && col < N)
        out[row * N + col] = tile[threadIdx.x][threadIdx.y];
}
```

### Source
Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## Skill 4: Choose Block Sizes as Multiples of Warp Size

### When to Use
- Configuring kernel launch parameters for any kernel with global memory access

### How to Apply
1. Set blockDim.x to a multiple of 32 (warp size)
2. Prefer block sizes of 128 or 256 threads as starting points
3. For 2D blocks, ensure blockDim.x >= 32

### Code Template
```cuda
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
my_kernel<<<grid, block>>>(data, n);
```

### Source
Best Practices Guide, Section 11.3 (Thread and Block Heuristics)

## Skill 5: Structure-of-Arrays over Array-of-Structures

### When to Use
- Data has multiple fields per element (e.g., position x, y, z)
- Threads process the same field across many elements

### How to Apply
1. Convert AoS layout to SoA layout on the host before transfer
2. Access each field array independently with coalesced patterns
3. If AoS cannot be changed, use shared memory to transpose

### Code Template
```cuda
struct SoA {
    float* x;
    float* y;
    float* z;
};

__global__ void process_soa(SoA data, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data.x[idx] + data.y[idx] + data.z[idx];
    }
}
```

### Source
Best Practices Guide, Section 10.2.1.4 (Strided Accesses)

## Skill 6: Understand the 32-byte Transaction Model

### When to Use
- Analyzing kernel memory performance in profiler output
- Diagnosing unexpected bandwidth degradation

### How to Apply
1. On CC 6.0+, each warp generates the minimum number of 32-byte transactions to service all 32 threads
2. For 4-byte (float) access: 32 threads x 4 bytes = 128 bytes = four 32-byte transactions (ideal)
3. Misaligned or strided accesses increase the transaction count per warp
4. Use Nsight Compute metric `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` to check

### Code Template
```cuda
// Analysis: offset copy kernel to measure transaction overhead
__global__ void offsetCopy(float* odata, float* idata, int offset) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    odata[xid] = idata[xid];
}
// offset=0: 4 transactions/warp (100% efficiency)
// offset=1..7: 5 transactions/warp (~80% efficiency, mitigated by L1 cache reuse)
```

### Source
Best Practices Guide, Section 10.2.1.2-10.2.1.3 (Misaligned Access)

## Cascading Opportunities (unlocks)
After achieving coalescing:
1. Check memory/vectorized-access -- widen loads to float2/float4 for fewer transactions
2. Check memory/shared-memory-cache -- if data is reused, cache it in shared memory
3. Check memory/l2-cache-control -- mark frequently accessed data as persisting

## Conflicts
- memory/layout-transform: achieving coalescing may require layout transformations that add preprocessing overhead

## Principles
- **P1**: On CC 6.0+, global memory is accessed via 32-byte transactions. Maximize the ratio of bytes used to bytes transferred.
- **P2**: A stride of S between consecutive threads wastes (S-1)/S of fetched data.
- **P3**: cudaMalloc guarantees 256-byte alignment. Block sizes that are multiples of 32 ensure aligned warp accesses.
- **P4**: On GDDR memory with ECC enabled, scattered accesses further increase overhead.

## Open Questions (for Level 3 verification)
- Q1: On V100, what is the exact bandwidth loss for stride-2 vs stride-1 in a copy kernel?
- Q2: Does L1 caching on sm_80+ hide misalignment penalties for small offsets?
- Q3: For 2D grids, does blockDim.x=32, blockDim.y=8 always outperform blockDim.x=16, blockDim.y=16?
