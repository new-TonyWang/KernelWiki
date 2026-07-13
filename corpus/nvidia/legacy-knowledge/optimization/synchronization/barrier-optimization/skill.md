# Barrier Optimization -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 12.1.3 (Synchronization Instruction)"
  - "Programming Guide 3.2.4.2 (Asynchronous Barriers)"
  - "Programming Guide 4.9 (Asynchronous Barriers)"
  - "Programming Guide 4.9.7 (Producer-Consumer Pattern Using Barriers)"
  - "Programming Guide 5.4.4.1 (Thread Block Synchronization Functions)"
  - "Programming Guide 5.4.4.2 (Warp Synchronization Function)"
  - "Programming Guide 5.4.8.5 (__nanosleep)"
cross_ref:
  - optimization/synchronization/cooperative-groups
  - optimization/synchronization/thread-scopes
  - optimization/memory/data-prefetch
  - optimization/latency/occupancy-tuning
related_apis:
  - __syncthreads
  - __syncwarp
  - "cuda::barrier"
  - "mbarrier.init"
  - "mbarrier.arrive"
  - "mbarrier.arrive.expect_tx"
  - "mbarrier.try_wait"
  - "bar.sync"
  - "barrier.cluster.arrive"
  - "barrier.cluster.wait"
unlocks:
  - Overlap of synchronization with useful computation
  - Fine-grained subset synchronization
  - Hardware-accelerated async copy tracking
conflicts_with:
  - None
```

---

## S1: Use Asynchronous Barriers (Arrive/Wait Split)

**When to Use:** When threads have independent work to do between signaling arrival and actually needing synchronization completion.

**How to Apply:**
1. Initialize a `cuda::barrier` in shared memory.
2. Call `bar.arrive()` to signal completion (returns a token).
3. Perform independent work.
4. Call `bar.wait(std::move(token))` only when the barrier result is needed.

**Code Template:**
```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void asyncBarrierKernel(float* data) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }
    block.sync();

    // Phase 1: produce data
    data[threadIdx.x] = compute(threadIdx.x);

    // Arrive: signal that data is ready
    auto token = bar.arrive();

    // Independent work while waiting
    float localResult = independentCompute(threadIdx.x);

    // Wait: now we need other threads' data
    bar.wait(std::move(token));

    // Phase 2: consume other threads' data
    output[threadIdx.x] = data[threadIdx.x ^ 1] + localResult;
}
```

> Source: PG 3.2.4.2 -- "An asynchronous barrier differs from a typical single-stage barrier in that the notification by a thread that it has reached the barrier (the 'arrival') is separated from the operation of waiting."

---

## S2: Use __syncwarp Instead of __syncthreads for Warp-Level Sync

**When to Use:** When synchronization is only needed within a warp, not across the entire block. Much cheaper than `__syncthreads`.

**How to Apply:**
1. Replace `__syncthreads()` with `__syncwarp(mask)` when the operation is warp-local.
2. Use `0xFFFFFFFF` for full-warp sync, or a computed mask for subset sync.

**Code Template:**
```cpp
// Warp-level reduction (no __syncthreads needed)
float val = myValue;
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// val in lane 0 now holds the warp sum
// No __syncwarp needed between shuffles as _sync variants handle it
```

> Source: BP 12.1.3 -- "__syncthreads() throughput... can impact performance by forcing the multiprocessor to idle."

---

## S3: Use mbarrier with expect_tx for Async Copy Tracking

**When to Use:** When using `cp.async` or TMA to copy data from global to shared memory, and you want the barrier to track copy completion without thread-level arrival.

**How to Apply:**
1. Initialize mbarrier with `mbarrier.init`.
2. Set expected transaction count with `mbarrier.arrive.expect_tx`.
3. Issue async copies that signal the barrier upon completion.
4. Wait with `mbarrier.try_wait`.

**Code Template:**
```cpp
#include <cuda/barrier>
#include <cuda/pipeline>

__global__ void asyncCopyBarrier(const float* global_data) {
    __shared__ float smem[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, 1);  // Only 1 thread arrives; copies complete the rest
    }
    block.sync();

    // Initiate async copy with barrier tracking
    if (block.thread_rank() == 0) {
        cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::device::barrier_native_handle(bar), 256 * sizeof(float));
        // Issue TMA or cp.async that signals bar on completion
    }

    // Wait for copies to complete
    auto token = bar.arrive();
    bar.wait(std::move(token));

    // smem is now safe to read
    process(smem[threadIdx.x]);
}
```

> Source: PG 4.9 -- mbarrier with expect_tx tracks asynchronous data movement completion.

---

## S4: Use Cluster-Level Barriers for Cross-Block Synchronization

**When to Use:** On Hopper+ (CC 9.0+) when thread blocks in the same cluster need to synchronize (e.g., for distributed shared memory access).

**How to Apply:**
1. Use `barrier.cluster.arrive` to signal arrival at the cluster barrier.
2. Use `barrier.cluster.wait` to wait for all blocks in the cluster.

**Code Template:**
```cpp
__global__ void clusterSync(float* data) {
    // Access distributed shared memory from another block in cluster
    // ... setup distributed shared memory access ...

    // Cluster barrier: wait for all blocks in cluster
    asm volatile("barrier.cluster.arrive.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");

    // Safe to access other blocks' shared memory in the cluster
}
```

> Source: PTX ISA -- barrier.cluster.arrive / barrier.cluster.wait for cluster-level synchronization.

---

## S5: Use __nanosleep for Spin-Wait Back-Off

**When to Use:** When a thread must spin-wait on a condition (e.g., a lock or flag), `__nanosleep` reduces power consumption and SM resource pressure.

**How to Apply:**
1. Replace busy-wait loops with `__nanosleep` in the spin body.
2. Use exponential back-off for prolonged waits.

**Code Template:**
```cpp
// Spin-wait with nanosleep back-off
unsigned ns = 8;
while (!flag.load(cuda::memory_order_acquire)) {
    __nanosleep(ns);
    if (ns < 256) ns *= 2;  // exponential back-off
}
```

> Source: PG 5.4.8.5 -- "__nanosleep suspends a thread for approximately ns nanoseconds."

---

## S6: Minimize Barrier Scope

**When to Use:** When only a subset of threads need to synchronize. Broader barriers stall more threads unnecessarily.

**How to Apply:**
1. Use `__syncwarp` instead of `__syncthreads` when possible.
2. Use `cuda::barrier` with a smaller expected count for subset synchronization.
3. Use cooperative groups `tiled_partition` for sub-block barriers.

**Code Template:**
```cpp
auto block = cooperative_groups::this_thread_block();
auto tile = cooperative_groups::tiled_partition<32>(block);

// Only sync within the 32-thread tile (warp)
tile.sync();  // much cheaper than block.sync()
```

> Source: PG 4.9 -- "Asynchronous barriers are flexible in specifying how threads participate and which threads participate."

---

## Cascading Opportunities

- Async barriers enable `data-prefetch` multi-buffering patterns (producer-consumer with mbarrier).
- Fewer barriers improve `occupancy-tuning` by reducing warp stall time.
- Cluster barriers enable `shared-memory-cache` distributed shared memory patterns.

## Conflicts

- None inherent. Barrier optimization is always beneficial.

## Principles

1. **Split arrive from wait:** Async barriers allow useful work between arrival and wait (PG 3.2.4.2).
2. **Narrowest scope wins:** Use warp sync for warp-level, block sync for block-level, cluster sync only when needed (BP 12.1.3).
3. **Reconverge warps before arrive:** Diverged warps cause multiple barrier updates; re-converge with `__syncwarp` first (PG 4.9.2.1).

## Open Questions

- Q1: What is the latency of `mbarrier.try_wait` vs. `__syncthreads()` on Hopper for a 256-thread block?
- Q2: How does the hardware optimize barrier operations for fully-converged warps vs. diverged warps?
