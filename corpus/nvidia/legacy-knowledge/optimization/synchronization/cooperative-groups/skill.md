# Cooperative Groups -- Skills

```yaml
status: draft
source:
  - "Programming Guide 2.2.6 (Cooperative Groups)"
  - "Programming Guide 4.4 (Cooperative Groups)"
cross_ref:
  - optimization/synchronization/barrier-optimization
  - optimization/compute/warp-primitives
  - optimization/latency/occupancy-tuning
related_apis:
  - "cooperative_groups::this_thread_block"
  - "cooperative_groups::this_grid"
  - "cooperative_groups::this_cluster"
  - "cooperative_groups::tiled_partition"
  - "cooperative_groups::coalesced_threads"
  - cudaLaunchCooperativeKernel
unlocks:
  - Safe sub-warp and cross-block synchronization
  - Grid-wide synchronization without host round-trip
  - Portable, future-proof synchronization patterns
conflicts_with:
  - None
```

---

## S1: Use tiled_partition for Sub-Block Synchronization

**When to Use:** When you need to synchronize subsets of threads smaller than a block (e.g., warps, half-warps, quarter-warps).

**How to Apply:**
1. Get the thread block handle with `this_thread_block()`.
2. Partition into tiles of the desired size with `tiled_partition<N>`.
3. Use `tile.sync()`, `tile.thread_rank()`, and `tile.shfl()` for tile-level operations.

**Code Template:**
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void tiledReduction(float* data) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto halfWarp = cg::tiled_partition<16>(block);

    float val = data[blockIdx.x * blockDim.x + threadIdx.x];

    // Warp-level reduction using cooperative groups
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // tile.thread_rank() gives position within the tile
    if (warp.thread_rank() == 0) {
        atomicAdd(&result, val);
    }
}
```

> Source: PG 4.4.4 -- "tiled_partition divides parent group into a series of fixed-size subgroups."

---

## S2: Use Grid-Wide Synchronization for Multi-Block Algorithms

**When to Use:** When an algorithm requires all thread blocks to synchronize at a barrier (e.g., iterative solvers, multi-pass algorithms) without returning to the host.

**How to Apply:**
1. Use `cudaLaunchCooperativeKernel` instead of the normal launch syntax.
2. Inside the kernel, get the grid group with `this_grid()` and call `grid.sync()`.
3. The number of blocks is limited to what the SM can support simultaneously.

**Code Template:**
```cpp
__global__ void iterativeSolver(float* data, int iterations) {
    auto grid = cg::this_grid();

    for (int iter = 0; iter < iterations; iter++) {
        // Each block processes its portion
        computeStep(data, blockIdx.x);

        // Grid-wide synchronization
        grid.sync();  // all blocks must reach this point
    }
}

// Host launch (must use cooperative launch)
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, iterativeSolver, blockSize, 0);
void* args[] = {&d_data, &iterations};
cudaLaunchCooperativeKernel((void*)iterativeSolver,
    numBlocks * numSMs, blockSize, args);
```

> Source: PG 4.4 -- cooperative groups enable grid-wide synchronization.

---

## S3: Use coalesced_threads for Dynamic Active-Thread Groups

**When to Use:** Inside divergent code where you need to operate on whatever threads are currently active, without assuming a fixed set.

**How to Apply:**
1. Call `cg::coalesced_threads()` to get a group of currently active threads.
2. Use the group for collective operations (sync, reduce, ballot).
3. Note: the set of active threads may change at any point.

**Code Template:**
```cpp
__global__ void sparseProcess(int* flags, float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (flags[tid]) {
        // Only some threads enter this branch
        auto active = cg::coalesced_threads();

        // Collective operation on active threads only
        float val = data[tid];
        float sum = cg::reduce(active, val, cg::plus<float>());

        if (active.thread_rank() == 0) {
            // Process the sum of active threads' values
            output[blockIdx.x] = sum;
        }
    }
}
```

> Source: PG 4.4.3 -- "coalesced_threads returns the handle to a group of currently active threads in a warp."

---

## S4: Use Cluster Groups for Cross-Block Communication (CC 9.0+)

**When to Use:** When thread blocks within a cluster need to access each other's shared memory (distributed shared memory).

**How to Apply:**
1. Launch with cluster dimensions via `cudaLaunchKernelExC`.
2. Use `this_cluster()` to get the cluster group.
3. Synchronize with `cluster.sync()` before accessing other blocks' shared memory.

**Code Template:**
```cpp
__global__ void clusterKernel() {
    auto cluster = cg::this_cluster();

    __shared__ float myData[256];
    myData[threadIdx.x] = compute(threadIdx.x);

    // Sync all blocks in cluster
    cluster.sync();

    // Access another block's shared memory in the cluster
    unsigned int peerBlockRank = (cluster.block_rank() + 1) % cluster.num_blocks();
    float* peerSmem = cluster.map_shared_rank(myData, peerBlockRank);
    float peerVal = peerSmem[threadIdx.x];
}
```

> Source: PG 4.4 -- this_cluster for cluster-level cooperative groups.

---

## S5: Create Group Handles Early and Pass by Reference

**When to Use:** Always -- creating group handles is not free, and copy-constructing them is discouraged.

**How to Apply:**
1. Create group handles at the top of the kernel before any branching.
2. Pass group handles by reference to helper functions.

**Code Template:**
```cpp
__device__ void helperFunc(cg::thread_block& block, float* data) {
    // Use block handle passed by reference
    block.sync();
    // ...
}

__global__ void myKernel(float* data) {
    // Create handle early, before any branching
    auto block = cg::this_thread_block();

    helperFunc(block, data);
}
```

> Source: PG 4.4.3.1 -- "For best performance it is recommended that you create a handle for the implicit group upfront."

---

## Cascading Opportunities

- Cooperative groups' `tiled_partition` feeds into `warp-primitives` for typed shuffle operations.
- Grid sync eliminates the need for multi-kernel approaches, reducing `kernel-launch-overhead`.
- Cluster groups enable `shared-memory-cache` distributed shared memory patterns.

## Conflicts

- None. Cooperative groups provide safe, future-proof abstractions over raw synchronization.

## Principles

1. **Safe and portable:** Cooperative groups replace ad-hoc warp-synchronous code with verified abstractions (PG 4.4.1).
2. **Grid sync limits block count:** `cudaLaunchCooperativeKernel` requires all blocks to fit simultaneously on the GPU (PG 4.4).
3. **Group handle immutability:** Group handles must be initialized at declaration; no default constructor (PG 4.4.3.2).

## Open Questions

- Q1: What is the overhead of `grid.sync()` compared to a kernel boundary on Hopper?
- Q2: How do cooperative groups interact with green contexts and SM partitioning?
