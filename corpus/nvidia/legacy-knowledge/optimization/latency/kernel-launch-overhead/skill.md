# Kernel Launch Overhead -- Skills

```yaml
status: draft
source:
  - "Best Practices Guide 11.5 (Concurrent Kernel Execution)"
  - "Best Practices Guide 11.6 (Multiple Contexts)"
  - "Programming Guide 4.7 (Lazy Loading)"
  - "Colfax Blog: CUTLASS Tutorial -- Persistent Kernels and Stream-K"
cross_ref:
  - optimization/latency/cuda-graphs
  - optimization/latency/stream-concurrency
  - optimization/latency/programmatic-dependent-launch
  - optimization/compute/operator-fusion
related_apis:
  - cudaLaunchKernel
  - cudaLaunchKernelExC
  - cudaGraphLaunch
  - cudaEventCreateWithFlags
  - cudaMemPoolCreate
unlocks:
  - Higher throughput for workloads with many small kernels
  - Better GPU utilization when kernel execution time is short
conflicts_with:
  - optimization/latency/dynamic-parallelism  # CDP has its own launch overhead
```

---

## S1: Use CUDA Graphs to Amortize Launch Overhead

**When to Use:** When the same sequence of kernels is launched repeatedly (e.g., training loop iterations, inference pipeline).

**How to Apply:**
1. Capture the kernel sequence into a graph using stream capture.
2. Instantiate the graph once.
3. Launch the executable graph in subsequent iterations.

**Code Template:**
```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;

// Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<grid, block, 0, stream>>>(...);
kernelB<<<grid, block, 0, stream>>>(...);
kernelC<<<grid, block, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);

// Instantiate once
cudaGraphInstantiate(&graphExec, graph, 0);

// Launch many times with minimal overhead
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graphExec, stream);
}
```

> Source: PG 4.2 -- "CPU launch costs are reduced compared to streams, because much of the setup is done in advance."

---

## S2: Use Persistent Kernels (Grid-Stride Loop)

**When to Use:** When launch overhead dominates execution time for many small kernel invocations. The kernel can be restructured to loop over work items.

**How to Apply:**
1. Launch a fixed number of blocks (typically `num_SMs * blocks_per_SM`).
2. Each block processes work items in a grid-stride loop.
3. Use atomics or a work queue in global memory for dynamic work distribution.

**Code Template:**
```cpp
__global__ void persistentKernel(float* data, int totalWork) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < totalWork; i += stride) {
        data[i] = process(data[i]);
    }
}

// Launch with just enough blocks to fill the GPU
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
persistentKernel<<<numSMs * 2, 256>>>(data, N);
```

> Source: Colfax Blog -- "Persistent Kernels and Stream-K" pattern for reducing launch overhead.

---

## S3: Enable Lazy Loading to Reduce Startup Latency

**When to Use:** Applications that link many CUDA libraries but only use a subset of kernels at runtime. Default since CUDA 12.3.

**How to Apply:**
1. Ensure CUDA 12.3+ is used (lazy loading is on by default).
2. For older versions, set `CUDA_MODULE_LOADING=LAZY` environment variable.
3. Verify with `CUDA_MODULE_LOADING=EAGER` to compare startup times.

**Code Template:**
```bash
# Enable lazy loading (default since CUDA 12.3)
export CUDA_MODULE_LOADING=LAZY

# Disable for debugging
export CUDA_MODULE_LOADING=EAGER
```

> Source: PG 4.7 -- "Lazy loading reduces program initialization time by waiting to load CUDA modules until they are needed."

---

## S4: Disable Timing on Events to Reduce Launch Overhead

**When to Use:** When using events only for synchronization (not timing), disabling timing eliminates per-event overhead.

**How to Apply:**
1. Create events with `cudaEventDisableTiming` flag.
2. This removes the timestamp recording overhead from each event.

**Code Template:**
```cpp
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

// Use for synchronization only
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
```

> Source: inferred from Runtime API -- cudaEventDisableTiming eliminates timestamp overhead.

---

## S5: Use cudaLaunchKernelExC for Extended Configuration

**When to Use:** When launching kernels with cluster dims, L2 access policy, memory sync domains, or programmatic events -- all specified in a single launch call rather than multiple API calls.

**How to Apply:**
1. Fill a `cudaLaunchConfig_t` structure with all attributes.
2. Launch with `cudaLaunchKernelExC` (or C++ template `cudaLaunchKernelEx`).

**Code Template:**
```cpp
cudaLaunchConfig_t config = {0};
config.gridDim = gridDim;
config.blockDim = blockDim;
config.dynamicSmemBytes = 0;
config.stream = stream;

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 1, 1};
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, myKernel, arg1, arg2);
```

> Source: PG 3.1.4 -- cudaLaunchKernelExC consolidates launch attributes.

---

## Cascading Opportunities

- After reducing launch overhead, the GPU may become memory-bound -- check `coalescing` and `data-prefetch`.
- Persistent kernels naturally combine with `stream-concurrency` for multi-stream workloads.
- CUDA Graphs pair well with `programmatic-dependent-launch` for overlapping graph nodes.

## Conflicts

- **dynamic-parallelism:** CDP incurs per-launch overhead from device code; persistent kernels often eliminate the need for CDP.

## Principles

1. **Launch overhead is fixed cost:** For short kernels (<10 us), launch overhead can dominate. Amortize it (BP 11.5, PG 4.2).
2. **Capture once, launch many:** CUDA Graphs eliminate repeated CPU-side setup (PG 4.2).
3. **Startup latency matters:** Lazy loading defers module JIT compilation until first use (PG 4.7).

## Open Questions

- Q1: What is the typical CPU-side launch overhead for `cudaLaunchKernel` vs `cudaGraphLaunch` on CUDA 13.x?
- Q2: How does `cudaInitDevice` interact with lazy loading for latency-sensitive cold-start scenarios?
