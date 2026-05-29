---
status: unverified
verified_date: null
verified_on: null
performance_data:
  before: null
  after: null
source:
  - Best Practices Guide, Section 10.2.3.4 (Asynchronous Copy from Global to Shared Memory)
  - Programming Guide, Section 4.11.1 (Using LDGSTS)
  - Programming Guide, Section 4.11.1.2 (Prefetching Data)
  - Programming Guide, Section 3.2.4.3 (Pipelines)
  - Programming Guide, Section 3.2.5 (Asynchronous Data Copies)
cross_ref:
  - Programming Guide, Section 4.10.7 (Producer-Consumer Pattern using Pipelines)
  - Programming Guide, Section 4.11.2 (Using TMA)
  - Programming Guide, Section 2.4.2.4 (Memory Advise and Prefetch)
related_apis:
  - cp.async.ca.shared.global, cp.async.cg.shared.global, cp.async.commit_group, cp.async.wait_group, __pipeline_memcpy_async, __pipeline_commit, __pipeline_wait_prior, cuda::memcpy_async, cudaMemPrefetchAsync, prefetch, cp.async.bulk
related_experience: []
unlocks:
  - "memory/shared-memory-cache: prefetched data lands in shared memory ready for fast access"
  - "memory/l2-cache-control: prefetch hints can warm L2 for upcoming access"
conflicts_with:
  - "memory/register-pressure: async copy bypasses registers, reducing pressure vs synchronous copy"
---

# Data Prefetch

## Skill 1: Async Copy from Global to Shared (LDGSTS) to Bypass Registers

### When to Use
- Compute capability 8.0+ (Ampere and later)
- Copying data from global to shared memory where traditional load+store wastes register bandwidth
- Want to overlap data movement with computation

### How to Apply
1. Use `__pipeline_memcpy_async()` to copy 4, 8, or 16 bytes per thread
2. Call `__pipeline_commit()` to group copies
3. Call `__pipeline_wait_prior(0)` to wait for all copies to complete
4. Call `__syncthreads()` before reading data written by other threads

### Code Template
```cuda
template <typename T>
__global__ void pipeline_kernel_async(T* global, size_t copy_count) {
    extern __shared__ char s[];
    T* shared = reinterpret_cast<T*>(s);

    for (size_t i = 0; i < copy_count; ++i) {
        __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                                &global[blockDim.x * i + threadIdx.x],
                                sizeof(T));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // Use shared[]
}
```

### Source
Best Practices Guide, Section 10.2.3.4 (Asynchronous Copy from Global to Shared Memory)

## Skill 2: Multi-Stage Pipeline for Overlapped Prefetch and Compute

### When to Use
- Kernel processes tiles of data sequentially (GEMM mainloop, stencil iteration)
- Want to overlap loading the next tile with computing on the current tile

### How to Apply
1. Allocate N shared memory buffers (double or triple buffering)
2. In the prologue, prefetch the first N-1 tiles
3. In the main loop, prefetch tile[i+N-1] while computing on tile[i]
4. Wait only on the buffer needed for current computation

### Code Template
```cuda
#define STAGES 2
__global__ void double_buffer_kernel(const float* data, float* out, int tiles) {
    __shared__ float buf[STAGES][256];
    int tid = threadIdx.x;

    // Prologue: fill first buffer
    __pipeline_memcpy_async(&buf[0][tid], &data[tid], sizeof(float));
    __pipeline_commit();

    for (int t = 0; t < tiles; t++) {
        int cur = t % STAGES;
        int nxt = (t + 1) % STAGES;

        // Prefetch next tile into next buffer
        if (t + 1 < tiles) {
            __pipeline_memcpy_async(&buf[nxt][tid],
                                    &data[(t + 1) * 256 + tid],
                                    sizeof(float));
            __pipeline_commit();
        }

        // Wait for current buffer
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        // Compute on current buffer
        out[t * 256 + tid] = buf[cur][tid] * 2.0f;
        __syncthreads();
    }
}
```

### Source
Programming Guide, Section 3.2.4.3 (Pipelines)

## Skill 3: Use cuda::memcpy_async with Barriers for Structured Prefetch

### When to Use
- Want structured async copy with barrier-based synchronization
- Using C++ cooperative groups for collective operations

### How to Apply
1. Create a `cuda::barrier<cuda::thread_scope_block>`
2. Issue `cuda::memcpy_async()` with the barrier as completion signal
3. Call `barrier.arrive_and_wait()` to wait for all copies
4. Follow with `__syncthreads()` for cross-warp visibility

### Code Template
```cuda
#include <cooperative_groups.h>
#include <cuda/barrier>

__global__ void barrier_prefetch(const float* src, float* dst) {
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier_t bar;
    __shared__ float smem[256];

    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() == 0) init(&bar, block.size());
    __syncthreads();

    cuda::memcpy_async(block, smem, src + blockIdx.x * 256,
                       cuda::aligned_size_t<4>(256 * sizeof(float)), bar);
    bar.arrive_and_wait();
    __syncthreads();

    dst[blockIdx.x * 256 + threadIdx.x] = smem[threadIdx.x] + 1.0f;
}
```

### Source
Programming Guide, Section 4.11.1.1 (Batching Loads in Conditional Code)

## Skill 4: Host-Side Prefetch with cudaMemPrefetchAsync

### When to Use
- Using unified memory (cudaMallocManaged)
- Want to migrate data to GPU before kernel launch
- Overlapping prefetch with other GPU work on a different stream

### How to Apply
1. Allocate with `cudaMallocManaged()`
2. Call `cudaMemPrefetchAsync(ptr, size, device_id, stream)` before the kernel
3. Launch kernel on the same or a dependent stream

### Code Template
```cuda
float* data;
cudaMallocManaged(&data, N * sizeof(float));
// Initialize data on host
for (int i = 0; i < N; i++) data[i] = (float)i;

// Prefetch to GPU 0
cudaMemPrefetchAsync(data, N * sizeof(float), 0, stream);

// Launch kernel -- data is already on GPU
myKernel<<<grid, block, 0, stream>>>(data, N);
```

### Source
Programming Guide, Section 2.4.2.4 (Memory Advise and Prefetch)

## Skill 5: Software Prefetch to L1/L2 Cache

### When to Use
- Want to warm cache lines before they are needed
- Data access pattern is predictable but not amenable to async copy

### How to Apply
1. Use PTX inline assembly for `prefetch.global.L1` or `prefetch.global.L2`
2. Issue prefetch several iterations ahead of actual use
3. Do not depend on prefetched data being in cache (it is a hint)

### Code Template
```cuda
__device__ void prefetch_l2(const void* addr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(addr) : "memory");
}

__global__ void prefetch_kernel(const float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Prefetch 4 iterations ahead
    if (idx + 4 * blockDim.x * gridDim.x < n)
        prefetch_l2(&data[idx + 4 * blockDim.x * gridDim.x]);
    if (idx < n)
        out[idx] = data[idx] * 2.0f;
}
```

### Source
PTX ISA, prefetch instruction

## Skill 6: 16-byte Async Copy for L1 Bypass

### When to Use
- Loading streaming data that will not be reused from L1
- Want to keep L1 cache free for other data

### How to Apply
1. Use `__pipeline_memcpy_async()` with sizeof(int4) = 16 bytes
2. 16-byte copies bypass L1, going directly global -> shared via L2
3. 4-byte and 8-byte copies still go through L1 (L1 ACCESS mode)

### Code Template
```cuda
__global__ void l1_bypass_copy(const int4* global_in) {
    extern __shared__ int4 smem[];
    int tid = threadIdx.x;

    // 16-byte async copy bypasses L1
    __pipeline_memcpy_async(&smem[tid], &global_in[blockIdx.x * blockDim.x + tid],
                            sizeof(int4));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
}
```

### Source
Programming Guide, Section 4.11.1 (Using LDGSTS)

## Cascading Opportunities (unlocks)
After applying data prefetch:
1. Check memory/shared-memory-cache -- prefetched data is already in shared memory
2. Check memory/l2-cache-control -- combine with L2 persistence for hot data

## Conflicts
- memory/register-pressure: async copy reduces register pressure compared to synchronous copy (positive interaction, not a conflict)

## Principles
- **P1**: Async copy (LDGSTS) bypasses intermediate register file, reducing register pressure and enabling better occupancy.
- **P2**: 16-byte async copies enable L1 bypass; 4-byte and 8-byte copies use L1 ACCESS mode.
- **P3**: Best async copy performance does not require copy_count to be a multiple of 4, unlike synchronous copies.
- **P4**: Multi-stage pipelines (double/triple buffering) are essential for overlapping data movement with computation in tiled kernels.

## Open Questions (for Level 3 verification)
- Q1: What is the optimal number of pipeline stages for GEMM on H100?
- Q2: How does TMA (cp.async.bulk.tensor) compare to LDGSTS for 2D tile prefetch latency?
- Q3: Does cudaMemPrefetchAsync provide measurable benefit over first-touch migration for large arrays?
