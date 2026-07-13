# Work Stealing -- Skills

```yaml
status: draft
source:
  - "Programming Guide 4.12 (Work Stealing with Cluster Launch Control)"
  - "Colfax Blog: CUTLASS Tutorial -- Persistent Kernels and Stream-K"
  - "Colfax Blog: CUTLASS Tutorial -- GEMM with Thread Block Clusters on NVIDIA Blackwell GPUs"
cross_ref:
  - optimization/latency/occupancy-tuning
  - optimization/latency/kernel-launch-overhead
  - optimization/latency/dynamic-parallelism
related_apis:
  - clusterlaunchcontrol.try_cancel
  - clusterlaunchcontrol.query_cancel
  - cudaDevSmResourceSplit
unlocks:
  - Dynamic load balancing with reduced overhead
  - Preemption support for higher-priority kernels
  - Combines benefits of fixed-work and persistent kernel approaches
conflicts_with:
  - None
```

---

## S1: Implement Work Stealing with Cluster Launch Control

**When to Use:** On Blackwell (CC 10.0+) when thread block execution times vary and you want dynamic load balancing without giving up preemption.

**How to Apply:**
1. Launch the kernel with many thread blocks (fixed work per block approach).
2. In each thread block, after completing its assigned work, attempt to cancel and "steal" the work of a not-yet-launched block.
3. If cancellation succeeds, process the stolen block's work using its index.
4. If cancellation fails (no more blocks, or preemption needed), exit.

**Code Template:**
```cpp
__global__ void workStealingKernel(float* data, int totalBlocks) {
    int myBlockIdx = blockIdx.x;

    while (true) {
        // Process work for myBlockIdx
        processBlock(data, myBlockIdx);

        // Try to steal another block's work
        bool cancelled;
        int stolenIdx;
        asm volatile(
            "clusterlaunchcontrol.try_cancel %0, %1;"
            : "=r"(cancelled), "=r"(stolenIdx)
        );

        if (!cancelled) {
            // No more work to steal, or preemption requested
            break;
        }
        myBlockIdx = stolenIdx;
    }
}

// Launch with enough blocks for full workload
workStealingKernel<<<totalBlocks, blockSize>>>(data, totalBlocks);
```

> Source: PG 4.12 -- "a thread block attempts to cancel the launch of another thread block that has not started executing yet. If the cancellation request succeeds, it 'steals' the other thread block's work."

---

## S2: Stream-K Style Persistent Kernel with Work Stealing

**When to Use:** For GEMM-like workloads where tiles have varying compute costs and you want to combine persistent kernel efficiency with load balancing.

**How to Apply:**
1. Launch a fixed number of thread blocks (persistent kernel).
2. Use an atomic counter or cluster launch control to distribute work dynamically.
3. Each block processes tiles until no more are available.

**Code Template:**
```cpp
__device__ int globalWorkCounter = 0;

__global__ void streamKKernel(float* C, const float* A, const float* B,
                               int totalTiles) {
    while (true) {
        // Atomically claim the next tile
        int tileIdx = atomicAdd(&globalWorkCounter, 1);
        if (tileIdx >= totalTiles) break;

        // Compute tile
        int tileRow = tileIdx / tilesPerRow;
        int tileCol = tileIdx % tilesPerRow;
        computeTile(C, A, B, tileRow, tileCol);
    }
}

// Launch with optimal block count for the GPU
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
streamKKernel<<<numSMs * 2, 256>>>(C, A, B, totalTiles);
```

> Source: Colfax Blog -- "Persistent Kernels and Stream-K" pattern for dynamic tile distribution.

---

## S3: Combine Work Stealing with Preemption Support

**When to Use:** When a higher-priority kernel may be launched during execution, and the current kernel should yield gracefully.

**How to Apply:**
1. When `clusterlaunchcontrol.try_cancel` fails, check if it was due to preemption (higher-priority kernel scheduled).
2. Exit the kernel to allow the higher-priority work to start.
3. The GPU scheduler will resume remaining blocks of the current kernel after the higher-priority kernel completes.

**Code Template:**
```cpp
__global__ void preemptibleKernel(float* data) {
    int myIdx = blockIdx.x;

    while (true) {
        processBlock(data, myIdx);

        bool cancelled;
        int stolenIdx;
        // try_cancel returns false if preemption is needed
        asm volatile(
            "clusterlaunchcontrol.try_cancel %0, %1;"
            : "=r"(cancelled), "=r"(stolenIdx)
        );

        if (!cancelled) {
            // Exit: either no more work or preemption requested
            // Scheduler can now run higher-priority kernel
            return;
        }
        myIdx = stolenIdx;
    }
}
```

> Source: PG 4.12 -- "if a thread block exits after a cancellation failure, the scheduler can start executing the higher-priority kernel."

---

## Cascading Opportunities

- Work stealing complements `stream-concurrency` stream priorities for QoS.
- Reduces the need for `dynamic-parallelism` in many irregular workloads.
- Pairs with `occupancy-tuning` to set the right initial block count.

## Conflicts

- None inherent. Work stealing is compatible with other latency techniques.

## Principles

1. **Three approaches:** Fixed-work (good load balance, no reduced overhead), Fixed-blocks (reduced overhead, poor load balance), Cluster Launch Control (both) (PG 4.12).
2. **CC 10.0+ required:** Cluster Launch Control is a Blackwell architecture feature.
3. **Graceful degradation:** When no work is available to steal or preemption is requested, the kernel exits cleanly.

## Open Questions

- Q1: What is the overhead of `clusterlaunchcontrol.try_cancel` per invocation?
- Q2: How does work stealing interact with thread block cluster dimensions?
