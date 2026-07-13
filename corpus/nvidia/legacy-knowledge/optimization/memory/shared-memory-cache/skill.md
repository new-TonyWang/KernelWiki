---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.3 (Shared Memory)
  - Best Practices Guide, Section 10.2.3.2 (Shared Memory in Matrix Multiply C=AB)
  - Programming Guide, Section 2.2.3.2 (Shared Memory)
  - Programming Guide, Section 2.2.4.2.1 (Matrix Transpose Using Shared Memory)
  - Programming Guide, Section 3.2.6 (Configuring L1/Shared Memory Balance)
cross_ref:
  - Best Practices Guide, Section 11.4 (Effects of Shared Memory)
  - Programming Guide, Section 2.2.3.8 (Distributed Shared Memory)
related_apis:
  - __syncthreads, ld.shared, st.shared, cudaFuncSetCacheConfig, cudaFuncSetAttribute, cudaFuncAttribute, cp.async.ca.shared.global
related_experience: []
unlocks:
  - "memory/bank-conflict-avoidance: once data is in shared memory, bank conflicts become the next bottleneck"
  - "memory/data-prefetch: async copy to shared memory overlaps computation with data movement"
  - "memory/coalescing: shared memory staging enables coalesced global access for irregular patterns"
conflicts_with:
  - "latency/occupancy-tuning: large shared memory per block reduces occupancy"
---

# Shared Memory Cache

## Skill 1: Use Shared Memory as User-Managed Cache for Global Data Reuse

### When to Use
- Multiple threads in a block access the same global memory data
- Data is accessed multiple times (temporal reuse)
- L1 hardware cache eviction policy does not match access pattern

### How to Apply
1. Declare `__shared__` array sized to the tile
2. Each thread loads one element from global memory into shared memory (coalesced)
3. Call `__syncthreads()` before reading shared memory
4. Perform computation using shared memory

### Code Template
```cuda
#define TILE_DIM 32
__global__ void coalescedMultiply(float* a, float* b, float* c, int N) {
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    __shared__ float bTile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    aTile[threadIdx.y][threadIdx.x] = a[row * TILE_DIM + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];
    __syncthreads();

    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
    }
    c[row * N + col] = sum;
}
```

### Source
Best Practices Guide, Section 10.2.3.2 (Shared Memory in Matrix Multiply C=AB)

## Skill 2: Stage Global Memory Reads via Shared Memory for Coalesced Access

### When to Use
- Kernel requires non-coalesced (strided or transposed) global memory access
- Shared memory can serve as an intermediate staging buffer

### How to Apply
1. Load from global memory in a coalesced pattern (row-major) into shared memory
2. Synchronize with `__syncthreads()`
3. Read from shared memory in the desired (possibly strided) pattern at no bandwidth penalty

### Code Template
```cuda
#define TILE 32
__global__ void smem_transpose(int m, float* a, float* c) {
    __shared__ float smemArray[TILE][TILE];
    const int tileCol = blockDim.x * blockIdx.x;
    const int tileRow = blockDim.y * blockIdx.y;

    // Coalesced read from global to shared
    smemArray[threadIdx.x][threadIdx.y] =
        a[(tileRow + threadIdx.y) * m + tileCol + threadIdx.x];
    __syncthreads();

    // Coalesced write from shared to global (transposed)
    c[(tileCol + threadIdx.y) * m + tileRow + threadIdx.x] =
        smemArray[threadIdx.y][threadIdx.x];
}
```

### Source
Programming Guide, Section 2.2.4.2.1 (Matrix Transpose Using Shared Memory)

## Skill 3: Use Dynamic Shared Memory for Variable-Size Tiles

### When to Use
- Tile size is determined at runtime (e.g., based on problem dimensions)
- Multiple arrays need to share the dynamic allocation

### How to Apply
1. Declare `extern __shared__` in the kernel
2. Pass the total shared memory bytes as the 3rd launch config parameter
3. Partition the buffer manually using pointer arithmetic with proper alignment

### Code Template
```cuda
__global__ void dynamic_smem_kernel(const float* in, float* out, int tile_w) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * tile_w + tid;

    if (tid < tile_w) {
        smem[tid] = in[gid];
    }
    __syncthreads();

    if (tid < tile_w) {
        out[gid] = smem[tid] * 2.0f;
    }
}

// Host launch:
int tile_w = 128;
int smem_bytes = tile_w * sizeof(float);
dynamic_smem_kernel<<<grid, block, smem_bytes>>>(in, out, tile_w);
```

### Source
Programming Guide, Section 2.2.3.2.2 (Dynamic Allocation of Shared Memory)

## Skill 4: Configure L1/Shared Memory Carveout

### When to Use
- Kernel uses a large amount of shared memory and L1 cache pressure is low
- Or kernel uses no shared memory and needs maximum L1 cache

### How to Apply
1. Use `cudaFuncSetAttribute()` with `cudaFuncAttributePreferredSharedMemoryCarveout`
2. Set to `cudaSharedmemCarveoutMaxShared` or `cudaSharedmemCarveoutMaxL1`
3. Alternatively use `cudaFuncSetCacheConfig()` for per-function preferences

### Code Template
```cuda
// Prefer maximum shared memory for this kernel
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);

// Or using the older API:
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);
```

### Source
Programming Guide, Section 3.2.6 (Configuring L1/Shared Memory Balance)

## Skill 5: Request Large Dynamic Shared Memory (>48KB)

### When to Use
- Kernel needs more than 48KB of shared memory per block
- Supported on compute capability 7.0+ (up to 96KB-228KB depending on GPU)

### How to Apply
1. Query max shared memory with `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)`
2. Set `cudaFuncAttributeMaxDynamicSharedMemorySize` for the kernel
3. Launch with the extended shared memory size

### Code Template
```cuda
int max_smem;
cudaDeviceGetAttribute(&max_smem,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    max_smem);

myKernel<<<grid, block, max_smem>>>(args...);
```

### Source
Programming Guide, Section 2.2.3.2 (Shared Memory)

## Skill 6: Use Distributed Shared Memory Across Cluster

### When to Use
- Compute capability 9.0+ (Hopper) with thread block clusters enabled
- Need more shared memory than a single block provides
- Histogram or reduction where bins exceed single-block shared memory

### How to Apply
1. Enable cluster launch with `cudaFuncSetAttribute(cudaFuncAttributeClusterDimMustBeSet, 1)`
2. Set cluster dimensions in launch configuration
3. Use `cluster.sync()` to ensure all blocks are running
4. Access remote shared memory via distributed shared memory pointers

### Code Template
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cluster_histogram(const int* data, int* bins, int n, int num_bins) {
    extern __shared__ int local_bins[];
    auto cluster = cg::this_cluster();
    auto block = cg::this_thread_block();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        local_bins[i] = 0;
    block.sync();
    cluster.sync();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx];
        int target_block = bin / (num_bins / cluster.num_blocks());
        int local_bin = bin % (num_bins / cluster.num_blocks());
        int* remote_bins = cluster.map_shared_rank(local_bins, target_block);
        atomicAdd(&remote_bins[local_bin], 1);
    }
    cluster.sync();
}
```

### Source
Programming Guide, Section 2.2.3.8 (Distributed Shared Memory)

## Cascading Opportunities (unlocks)
After placing data in shared memory:
1. Check memory/bank-conflict-avoidance -- ensure shared memory access has no bank conflicts
2. Check memory/data-prefetch -- use async copy (cp.async) to overlap transfer with computation
3. Check memory/coalescing -- verify the global-to-shared copy itself is coalesced

## Conflicts
- latency/occupancy-tuning: shared memory usage reduces the number of blocks that can co-reside on an SM, lowering occupancy

## Principles
- **P1**: Shared memory has much higher bandwidth and lower latency than global memory -- provided there are no bank conflicts.
- **P2**: Shared memory and L1 cache share the same on-chip storage. Using shared memory reduces available L1 cache.
- **P3**: Use shared memory to eliminate redundant global memory reads: on V100, caching A tile improved matrix multiply from 119.9 to 144.4 GB/s; caching both A and B tiles reached 195.5 GB/s.
- **P4**: `__syncthreads()` is required when data written by one warp is read by another warp. `__syncwarp()` suffices for intra-warp sharing.

## Open Questions (for Level 3 verification)
- Q1: What is the optimal shared memory carveout for GEMM-like kernels on A100 vs H100?
- Q2: How does distributed shared memory latency compare to direct shared memory on Hopper?
- Q3: What is the maximum practical shared memory size before occupancy drops too much on sm_90?
