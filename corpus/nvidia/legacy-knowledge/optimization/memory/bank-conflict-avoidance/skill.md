---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.3.1 (Shared Memory and Memory Banks)
  - Best Practices Guide, Section 10.2.3.3 (Shared Memory in Matrix Multiply C=AAT)
  - Programming Guide, Section 2.2.4.2 (Shared Memory Access Patterns)
  - Programming Guide, Section 2.2.4.2.2 (Shared Memory Bank Conflicts)
cross_ref:
  - Programming Guide, Section 4.11.2.2.5 (Shared-Memory Bank Swizzling)
related_apis:
  - ld.shared, st.shared, cudaDeviceSetSharedMemConfig, cudaSharedMemConfig
related_experience: []
unlocks:
  - "memory/shared-memory-cache: conflict-free shared memory enables full bandwidth"
conflicts_with: []
---

# Bank Conflict Avoidance

## Skill 1: Pad Shared Memory Arrays to Eliminate Column-Access Conflicts

### When to Use
- Writing to shared memory in columns (e.g., transposed write pattern)
- Tile dimension matches bank count (32), causing all threads to hit same bank

### How to Apply
1. Add +1 padding to the inner dimension of the 2D shared memory array
2. `__shared__ float tile[TILE_DIM][TILE_DIM+1]` instead of `[TILE_DIM][TILE_DIM]`
3. The padding shifts column accesses to different banks

### Code Template
```cuda
#define TILE_DIM 32
__global__ void transpose_no_conflict(float* a, float* c, int M) {
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float transposedTile[TILE_DIM][TILE_DIM + 1];  // +1 padding

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    // Column write: stride=TILE_DIM+1=33, which maps to stride-1 in bank space
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x * blockDim.x + threadIdx.y) * TILE_DIM + threadIdx.x];
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i] * transposedTile[i][threadIdx.x];
    }
    c[row * M + col] = sum;
}
```

### Source
Best Practices Guide, Section 10.2.3.3 (Shared Memory in Matrix Multiply C=AAT)

## Skill 2: Use Stride Analysis to Detect Bank Conflicts

### When to Use
- Writing a new shared memory access pattern
- Debugging unexpectedly low shared memory bandwidth

### How to Apply
1. Shared memory has 32 banks; successive 32-bit words map to successive banks
2. Bank index = (byte_offset / 4) % 32
3. If two threads in the same warp access different words in the same bank, a conflict occurs
4. Check: stride between consecutive threads' accesses modulo 32. If 0, 32-way conflict.

### Code Template
```cuda
// Analysis example: stride-2 access
// Thread 0 -> bank 0, Thread 1 -> bank 2, Thread 2 -> bank 4, ...
// Thread 16 -> bank 0 (conflict with Thread 0)
// Result: 2-way bank conflict

__shared__ float data[1024];
// Bad: stride 2 between consecutive threads
float val = data[threadIdx.x * 2];      // 2-way conflict

// Good: stride 1 between consecutive threads
float val2 = data[threadIdx.x];          // No conflict

// Good: all threads read same address (broadcast)
float val3 = data[0];                    // No conflict (broadcast)
```

### Source
Best Practices Guide, Section 10.2.3.1 (Shared Memory and Memory Banks)

## Skill 3: Use 8-byte Bank Mode for Double-Precision Data

### When to Use
- Kernel uses double-precision (8-byte) data in shared memory
- Default 4-byte bank mode causes 2-way conflicts for double access

### How to Apply
1. Call `cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)` before kernel launch
2. Or set per-function with `cudaFuncSetSharedMemConfig(kernel, cudaSharedMemBankSizeEightByte)`
3. This doubles the bank width, aligning double accesses to avoid conflicts

### Code Template
```cuda
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

__global__ void double_kernel() {
    __shared__ double shared_data[256];
    int tid = threadIdx.x;
    // With 8-byte banks, consecutive threads access consecutive banks
    shared_data[tid] = (double)tid;
    __syncthreads();
    double val = shared_data[tid];
}
```

### Source
Programming Guide, Section 2.2.4.2 (Shared Memory Access Patterns)

## Skill 4: Swizzle Shared Memory Addresses to Avoid Conflicts

### When to Use
- Complex access patterns where simple padding is insufficient
- TMA-era code (compute capability 9.0+) with hardware swizzling support

### How to Apply
1. Compute a swizzled index by XOR-ing parts of the address
2. swizzled_col = col ^ (row % factor) for a simple swizzle
3. The XOR creates a permutation that distributes accesses across banks

### Code Template
```cuda
#define TILE 32
__global__ void swizzled_access() {
    __shared__ float tile[TILE][TILE];
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Write with swizzle
    int swizzled_col = col ^ (row & 0x3);  // XOR low 2 bits of row
    tile[row][swizzled_col] = compute_value(row, col);
    __syncthreads();

    // Read with same swizzle
    float val = tile[row][col ^ (row & 0x3)];
}
```

### Source
Programming Guide, Section 4.11.2.2.5 (Shared-Memory Bank Swizzling)

## Skill 5: Detect Bank Conflicts Using Nsight Compute

### When to Use
- Performance of shared memory operations is lower than expected
- Need to quantify bank conflict overhead

### How to Apply
1. Profile with Nsight Compute
2. Check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` for load conflicts
3. Check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` for store conflicts
4. A value > 0 indicates bank conflicts; higher values indicate worse conflicts

### Code Template
```bash
# Profile for bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./my_app
```

### Source
Best Practices Guide, Section 10.2.3.1 (Shared Memory and Memory Banks)

## Cascading Opportunities (unlocks)
After resolving bank conflicts:
1. Shared memory bandwidth becomes the full theoretical rate
2. Check memory/shared-memory-cache -- shared memory can now be used confidently as a high-bandwidth cache

## Conflicts
- None significant; bank conflict avoidance is purely beneficial

## Principles
- **P1**: Shared memory has 32 banks. Bank index = (byte_offset / 4) % 32. Access to the same bank by different threads in a warp serializes.
- **P2**: The exception is broadcast: multiple threads reading the same address incurs no conflict.
- **P3**: Padding by +1 eliminates column-access conflicts by changing the stride from 32 (same bank) to 33 (stride 1 across banks).
- **P4**: On V100, padding eliminated bank conflicts in C=AAT multiply, improving bandwidth from 140.2 to 199.4 GB/s (42% improvement).

## Open Questions (for Level 3 verification)
- Q1: On Hopper, does hardware swizzle with TMA eliminate the need for software padding?
- Q2: What is the overhead of +1 padding in terms of wasted shared memory capacity?
- Q3: For 16-byte (float4) shared memory accesses, how does the bank conflict model change?
