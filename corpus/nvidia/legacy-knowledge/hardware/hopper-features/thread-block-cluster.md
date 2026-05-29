# Thread Block Clusters

## Concept

Thread Block Clusters are an optional hierarchy level introduced in Compute Capability 9.0 (Hopper). A cluster is a group of thread blocks that are **guaranteed to be co-scheduled on SMs within the same GPU Processing Cluster (GPC)**. This enables efficient inter-block cooperation that was previously impossible.

**Motivation:** As GPUs grew beyond 100 SMs, using a single thread block on a single SM as the only unit of locality became insufficient to maximize execution efficiency. Clusters expose control of locality at a granularity larger than one SM.

## Thread Hierarchy

```
Grid
  └── Cluster          (new in Hopper, optional)
        └── Block      (scheduled on one SM)
              └── Warp (32 threads)
                    └── Thread
```

- A grid is composed of clusters (or individual blocks if clusters are not used).
- Each cluster contains 1 to N thread blocks.
- Blocks within a cluster run concurrently on adjacent SMs in the same GPC.

## Size Limits

| Parameter | Value |
|-----------|-------|
| Maximum portable cluster size | **8 thread blocks** |
| Cluster dimensionality | 1D, 2D, or 3D (like blocks and grids) |
| Constraint on grid size | Grid dimensions must be a **multiple of cluster dimensions** |
| Query max cluster size | `cudaOccupancyMaxPotentialClusterSize()` |
| Query active clusters | `cudaOccupancyMaxActiveClusters()` |

On hardware or MIG configurations with fewer than 8 SMs, the maximum cluster size is reduced accordingly. Some architectures (Hopper H100, Blackwell B200) support clusters up to size 16 with an opt-in option.

## Intra-Cluster Synchronization

Clusters provide **hardware-accelerated barriers** via the Cooperative Groups API:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::cluster_group cluster = cg::this_cluster();

// Full cluster barrier (all blocks must reach this point)
cluster.sync();

// Split barrier pattern (arrive/wait)
auto token = cluster.barrier_arrive();
// ... do independent work ...
cluster.barrier_wait(std::move(token));
```

**Key API members of `cluster_group`:**

| Method | Description |
|--------|-------------|
| `sync()` | Barrier across all blocks in the cluster |
| `barrier_arrive()` / `barrier_wait()` | Split arrive/wait barrier |
| `num_blocks()` | Number of blocks in the cluster |
| `block_rank()` | Rank of calling block within the cluster [0, num_blocks) |
| `dim_blocks()` | Cluster dimensions in units of blocks |
| `block_index()` | 3D index of calling block within the cluster |
| `map_shared_rank(addr, rank)` | Get pointer to another block's shared memory variable |
| `query_shared_rank(addr)` | Get block rank that owns a shared memory address |

## Relationship with Distributed Shared Memory

Blocks within a cluster can **read, write, and perform atomics** on each other's shared memory. This is called Distributed Shared Memory (DSMEM).

- Total DSMEM capacity = `cluster_size * per_block_shared_memory_size`
- DSMEM segments from all blocks are mapped into each thread's generic address space
- Access remote shared memory via `cluster.map_shared_rank(local_smem_ptr, target_block_rank)`
- All blocks must be running before accessing DSMEM (enforce with `cluster.sync()`)
- All DSMEM operations must complete before any block exits

```cpp
// Access another block's shared memory
int dst_block_rank = cyclic_rank;
int *remote_smem = cluster.map_shared_rank(local_smem, dst_block_rank);
atomicAdd(remote_smem + offset, value);
```

A dedicated SM-to-SM network within the GPC provides fast, low-latency access to remote DSMEM. Compared to using global memory, DSMEM accelerates data exchange between thread blocks by approximately **7x** (per NVIDIA whitepaper).

## Launch Configuration

### Compile-time (kernel attribute)

```cpp
__global__ void __cluster_dims__(2, 1, 1) my_kernel(float *in, float *out)
{
    // cluster size fixed at compile time
}

// Grid dimension must be a multiple of cluster size
dim3 numBlocks(N / 16, N / 16);
dim3 threadsPerBlock(16, 16);
my_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
```

### Runtime (extensible launch API)

```cpp
__global__ void my_kernel(float *in, float *out) { }

cudaLaunchConfig_t config = {0};
config.gridDim = numBlocks;
config.blockDim = threadsPerBlock;

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim.x = 2;
attribute[0].val.clusterDim.y = 1;
attribute[0].val.clusterDim.z = 1;
config.attrs = attribute;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, my_kernel, input, output);
```

### Blocks-as-Clusters (alternative syntax)

```cpp
// Block size and cluster size both specified as kernel attributes
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// First arg in <<<>>> is now number of clusters, not blocks
foo<<<dim3(8, 8, 8)>>>();  // 8x8x8 clusters, each 2x2x2 blocks
```

## Performance Implications

### When to use clusters

1. **Distributed Shared Memory access** -- When multiple blocks need to share intermediate data that would otherwise go through global memory (e.g., histogram bins, halo exchanges).
2. **TMA Multicast** -- Clusters enable TMA multicast, where a single TMA load places data into the shared memory of multiple CTAs simultaneously. In GEMM, this reduces global memory traffic by a factor equal to the multicast width (e.g., 4x reduction with a 4-wide cluster row/column).
3. **Large cooperative tile sizes** -- Clusters allow cooperative execution with more threads and a larger combined shared memory pool than a single block can provide.

### Key considerations

- **Occupancy trade-off:** Larger cluster sizes reduce scheduling flexibility. All blocks in a cluster must be co-scheduled on adjacent SMs, which can reduce occupancy if the GPU does not have enough contiguous SMs available. Use `cudaOccupancyMaxActiveClusters()` to check.
- **Grid size alignment:** Grid dimensions must be exact multiples of cluster dimensions. This can require padding or careful tile-size selection.
- **Cluster size 1 is valid:** Use cluster_size=1 when DSMEM is not needed; this degenerates to standard per-block shared memory with no scheduling constraints.
- **GEMM mapping pattern:** In GEMM, the cluster shape naturally maps to the M and N dimensions of the output tile grid (K dimension is typically 1 unless using Split-K). CTAs in the same cluster row share A operand tiles; CTAs in the same column share B operand tiles.

### Performance data (from H100 whitepaper)

- DSMEM data exchange is approximately **7x faster** than going through global memory.
- H100 shared memory per SM is configurable up to **228 KB**; a cluster of 8 blocks provides up to **~1.8 MB** of combined distributed shared memory.
- Cluster-based GEMM with TMA multicast can reduce global memory bandwidth consumption proportional to the cluster dimension along the shared operand axis.
