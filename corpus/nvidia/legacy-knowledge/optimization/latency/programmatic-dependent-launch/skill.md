# Programmatic Dependent Launch -- Skills

```yaml
status: draft
source:
  - "Programming Guide 3.1.4 (Programmatic Dependent Kernel Launch)"
  - "Programming Guide 4.5 (Programmatic Dependent Launch and Synchronization)"
cross_ref:
  - optimization/latency/stream-concurrency
  - optimization/latency/cuda-graphs
  - optimization/latency/kernel-launch-overhead
related_apis:
  - cudaTriggerProgrammaticLaunchCompletion
  - cudaGridDependencySynchronize
  - cudaLaunchKernelExC
  - griddepcontrol.launch_dependents
  - griddepcontrol.wait
unlocks:
  - Partial overlap of sequential kernels in the same stream
  - Reduced inter-kernel latency without separate streams
conflicts_with:
  - None
```

---

## S1: Overlap Two Sequential Kernels with PDL

**When to Use:** When a primary kernel produces results that a secondary kernel needs, but both kernels have independent work phases that can overlap.

**How to Apply:**
1. In the primary kernel, call `cudaTriggerProgrammaticLaunchCompletion()` after finishing the work the secondary kernel depends on.
2. In the secondary kernel, call `cudaGridDependencySynchronize()` after finishing independent initialization work.
3. Launch the secondary kernel with `cudaLaunchAttributeProgrammaticStreamSerialization` attribute set.

**Code Template:**
```cpp
__global__ void primary_kernel(float* shared_data, float* local_data) {
    // Phase 1: produce data the secondary kernel needs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    shared_data[tid] = compute_shared(tid);
    __syncthreads();

    // Signal: secondary kernel can start using shared_data
    cudaTriggerProgrammaticLaunchCompletion();

    // Phase 2: independent work that overlaps with secondary
    local_data[tid] = compute_local(tid);
}

__global__ void secondary_kernel(float* shared_data, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: independent initialization
    float local_val = init_local(tid);

    // Wait for primary kernel's shared_data to be ready
    cudaGridDependencySynchronize();

    // Phase 2: use shared_data from primary
    output[tid] = local_val + shared_data[tid];
}

// Host launch
cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr.val.programmaticStreamSerializationAllowed = 1;

cudaLaunchConfig_t config = {0};
config.gridDim = grid;
config.blockDim = block;
config.stream = stream;
config.attrs = &attr;
config.numAttrs = 1;

primary_kernel<<<grid, block, 0, stream>>>(shared_data, local_data);
cudaLaunchKernelEx(&config, secondary_kernel, shared_data, output);
```

> Source: PG 3.1.4 -- "CUDA provides a mechanism to allow the application developer to specify the synchronization point between the two kernels."

---

## S2: Use PDL with CUDA Graphs

**When to Use:** When building a CUDA graph with kernel nodes that can benefit from partial overlap.

**How to Apply:**
1. Set edge data type to `cudaGraphDependencyTypeProgrammatic` between kernel nodes.
2. The kernel code uses the same `cudaTriggerProgrammaticLaunchCompletion` / `cudaGridDependencySynchronize` pattern.

**Code Template:**
```cpp
cudaGraphEdgeData edgeData = {0};
edgeData.type = cudaGraphDependencyTypeProgrammatic;
edgeData.from_port = cudaGraphKernelNodePortProgrammatic;
// Add dependency with PDL edge type
cudaGraphAddDependencies(graph, &primaryNode, &secondaryNode, &edgeData, 1);
```

> Source: PG 4.5.3 -- PDL can be used with CUDA Graphs via edge data.

---

## S3: Use PTX griddepcontrol for Low-Level Control

**When to Use:** When writing inline PTX or hand-tuned kernels that need direct control over dependent launch signaling.

**How to Apply:**
1. Use `griddepcontrol.launch_dependents` to signal that dependent grids may launch.
2. Use `griddepcontrol.wait` to wait for prerequisite grids to complete.

**Code Template:**
```cpp
__global__ void primary_kernel_ptx() {
    // ... produce data ...

    // Signal dependents via inline PTX
    asm volatile("griddepcontrol.launch_dependents;");

    // ... continue independent work ...
}

__global__ void secondary_kernel_ptx() {
    // ... independent initialization ...

    // Wait for primary
    asm volatile("griddepcontrol.wait;");

    // ... use dependent data ...
}
```

> Source: PTX ISA -- griddepcontrol.launch_dependents / griddepcontrol.wait.

---

## Cascading Opportunities

- PDL reduces inter-kernel gap, complementing `cuda-graphs` for repeated workloads.
- Combine with `stream-concurrency` to overlap PDL pairs with independent streams.

## Conflicts

- None inherent. PDL is complementary to other latency techniques.

## Principles

1. **Partial overlap, not full concurrency:** PDL allows the tail of one kernel to overlap with the head of the next, controlled by the developer (PG 3.1.4).
2. **All blocks must signal:** `cudaTriggerProgrammaticLaunchCompletion` should be called by all blocks of the primary kernel, as the secondary waits for all of them.
3. **Stream ordering is preserved:** PDL operates within stream semantics; it relaxes the start condition, not the completion guarantee.

## Open Questions

- Q1: What is the practical overlap achievable with PDL when both kernels are compute-bound?
- Q2: Does PDL work across cluster boundaries on Hopper/Blackwell?
