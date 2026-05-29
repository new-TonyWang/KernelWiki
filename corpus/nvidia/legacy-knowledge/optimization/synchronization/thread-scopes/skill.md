# Thread Scopes -- Skills

```yaml
status: draft
source:
  - "Programming Guide 3.2.3 (Thread Scopes)"
  - "Programming Guide 3.2.4.1.2 (Performance Considerations)"
  - "Programming Guide 5.4.4.3 (Memory Fence Functions)"
  - "Programming Guide 5.7 (CUDA C++ Memory Model)"
cross_ref:
  - optimization/synchronization/atomic-reduction
  - optimization/synchronization/memory-sync-domains
  - optimization/synchronization/barrier-optimization
related_apis:
  - __threadfence
  - __threadfence_block
  - __threadfence_system
  - "cuda::atomic"
  - "fence.sem.scope (PTX)"
  - "membar.cta/.gl/.sys (PTX)"
unlocks:
  - Correct multi-level synchronization
  - Performance-optimal memory visibility
conflicts_with:
  - None
```

---

## S1: Choose the Narrowest Thread Scope for Atomics and Fences

**When to Use:** Always -- wider scopes are more expensive because they flush to a farther point in the memory hierarchy.

**How to Apply:**
1. **Block scope (`.cta`):** For communication between threads in the same block. Coherency at L1.
2. **Cluster scope (`.cluster`):** For communication between blocks in the same cluster (CC 9.0+). Coherency at L2.
3. **Device scope (`.gpu`):** For communication between any blocks on the same GPU. Coherency at L2.
4. **System scope (`.sys`):** For communication with CPU or other GPUs. Coherency at L2 + connected caches.

**Code Template:**
```cpp
#include <cuda/atomic>

// Block scope: fastest, L1 coherency
__shared__ cuda::atomic<int, cuda::thread_scope_block> blockAtomic;

// Device scope: L2 coherency
__device__ cuda::atomic<int, cuda::thread_scope_device> deviceAtomic;

// System scope: slowest, PCIe/NVLink coherency
__managed__ cuda::atomic<int, cuda::thread_scope_system> systemAtomic;
```

> Source: PG 3.2.4.1.2 -- "Use the narrowest scope possible."

---

## S2: Use __threadfence_block Instead of __threadfence for Intra-Block Communication

**When to Use:** When a thread needs to ensure its writes to shared or global memory are visible to other threads in the same block.

**How to Apply:**
1. After writing data that other threads in the block will read.
2. Before signaling (via flag or barrier) that the data is ready.
3. The receiving threads do not need a fence if they use `__syncthreads()`.

**Code Template:**
```cpp
__shared__ float data[256];
__shared__ int flag;

// Producer thread
if (threadIdx.x == 0) {
    data[0] = 42.0f;
    __threadfence_block();  // ensure data write is visible within block
    flag = 1;               // signal
}

// Consumer thread
while (flag != 1) {}       // wait for signal
// __threadfence_block not needed here if using volatile or atomic flag
float val = data[0];       // guaranteed to see 42.0f
```

> Source: PG 5.4.4.3 -- "__threadfence_block ensures writes are visible to other threads in the block."

---

## S3: Use Device Scope Fence for Inter-Block Communication

**When to Use:** When blocks need to communicate through global memory (e.g., inter-block reduction, producer-consumer across blocks).

**How to Apply:**
1. Producer block writes data, then calls `__threadfence()` (device scope).
2. Producer block sets a flag (atomic write) to signal completion.
3. Consumer block reads the flag (atomic read), then reads the data.

**Code Template:**
```cpp
__device__ int globalData[1024];
__device__ cuda::atomic<int, cuda::thread_scope_device> ready(0);

// Block A (producer)
__global__ void producer() {
    globalData[threadIdx.x] = computeResult(threadIdx.x);
    __threadfence();  // device scope: visible to all blocks
    if (threadIdx.x == 0) {
        ready.store(1, cuda::memory_order_release);
    }
}

// Block B (consumer, launched after or concurrently)
__global__ void consumer() {
    if (threadIdx.x == 0) {
        while (ready.load(cuda::memory_order_acquire) != 1) {}
    }
    __syncthreads();
    float val = globalData[threadIdx.x];  // sees producer's writes
}
```

> Source: PG 3.2.3 -- device scope ensures visibility to all threads on the GPU.

---

## S4: Use System Scope Only for GPU-CPU or Multi-GPU Communication

**When to Use:** When the GPU needs to signal the CPU (e.g., via mapped memory) or another GPU (e.g., via peer memory).

**How to Apply:**
1. Use `__threadfence_system()` or `cuda::thread_scope_system` atomics.
2. This is the most expensive scope; avoid when not needed.

**Code Template:**
```cpp
// GPU signals CPU via mapped pinned memory
__managed__ cuda::atomic<int, cuda::thread_scope_system> gpuToCpuFlag(0);
__managed__ float sharedData;

// GPU kernel
__global__ void gpuProducer() {
    sharedData = 42.0f;
    __threadfence_system();  // visible to CPU
    gpuToCpuFlag.store(1, cuda::memory_order_release);
}

// CPU code
while (gpuToCpuFlag.load(cuda::memory_order_acquire) != 1) {}
assert(sharedData == 42.0f);
```

> Source: PG 3.2.3 -- system scope makes operations visible to "other threads in the same system (CPU, other GPUs)."

---

## S5: Prefer Scoped cuda::atomic Over Legacy __threadfence

**When to Use:** In new code, always prefer `cuda::atomic` with explicit scopes over the legacy `__threadfence` / `atomicAdd` functions.

**How to Apply:**
1. Replace `atomicAdd(&var, val)` with `cuda::atomic_ref<T, Scope>(var).fetch_add(val, order)`.
2. Replace `__threadfence()` with `cuda::atomic_thread_fence(order, scope)`.
3. The explicit scope and ordering make the intent clear and enable compiler optimization.

**Code Template:**
```cpp
// Legacy (implicit device scope, implicit seq_cst-like behavior)
atomicAdd(&counter, 1);
__threadfence();

// Modern (explicit scope and ordering)
cuda::atomic_ref<int, cuda::thread_scope_device> ref(counter);
ref.fetch_add(1, cuda::memory_order_relaxed);
cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
```

> Source: PG 3.2.4.1 -- scoped atomics with C++ standard memory semantics.

---

## Cascading Opportunities

- Thread scopes directly affect `atomic-reduction` performance -- narrower scope = faster atomics.
- `memory-sync-domains` provide further isolation when device-scope fences are too broad.
- `barrier-optimization` async barriers can be scoped to block or cluster level.

## Conflicts

- None. Using the correct scope is always beneficial for both correctness and performance.

## Principles

1. **Narrowest scope = fastest:** Each wider scope adds latency as coherency extends further (PG 3.2.3, 3.2.4.1.2).
2. **Scopes map to memory hierarchy:** block->L1, cluster->L2, device->L2, system->L2+external (PG 3.2.3).
3. **Cumulativity:** Wider scopes must encompass all writes visible to narrower scopes (PG 3.2.3 + 4.14).

## Open Questions

- Q1: What is the latency difference between `.cta` and `.gpu` scoped fences on Hopper?
- Q2: How does the cluster scope (`.cluster`) perform compared to device scope (`.gpu`) for operations within a GPC?
