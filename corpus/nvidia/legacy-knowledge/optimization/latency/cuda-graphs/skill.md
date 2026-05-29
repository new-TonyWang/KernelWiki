# CUDA Graphs -- Skills

```yaml
status: draft
source:
  - "Programming Guide 2.3.9.2 (Introduction to CUDA Graphs with Stream Capture)"
  - "Programming Guide 4.2 (CUDA Graphs)"
  - "Programming Guide 4.2.5.4 (Performance Considerations)"
  - "Programming Guide 4.2.6 (Device Graph Launch)"
cross_ref:
  - optimization/latency/kernel-launch-overhead
  - optimization/latency/stream-concurrency
  - optimization/latency/programmatic-dependent-launch
related_apis:
  - cudaGraphCreate
  - cudaGraphInstantiate
  - cudaGraphLaunch
  - cudaStreamBeginCapture
  - cudaStreamEndCapture
  - cudaGraphExecUpdate
  - cudaGraphUpload
unlocks:
  - Near-zero CPU launch overhead for repeated workloads
  - Whole-workflow optimization by CUDA runtime
conflicts_with:
  - optimization/latency/dynamic-parallelism  # CDP not permitted in device graphs
```

---

## S1: Stream Capture for Easy Graph Creation

**When to Use:** When an existing stream-based workflow needs to be converted to a graph with minimal code changes.

**How to Apply:**
1. Bracket existing stream operations with `cudaStreamBeginCapture` / `cudaStreamEndCapture`.
2. Instantiate and launch the captured graph.
3. Operations are not executed during capture -- they are recorded into the graph.

**Code Template:**
```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<gridA, blockA, 0, stream>>>(...);
kernelB<<<gridB, blockB, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);

cudaGraphInstantiate(&graphExec, graph, 0);

for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graphExec, stream);
}
cudaStreamSynchronize(stream);
```

> Source: PG 4.2.2.1.2 -- "Stream capture provides a mechanism to create a graph from existing stream-based APIs."

---

## S2: Cross-Stream Dependencies in Graph Capture

**When to Use:** When the graph has parallel branches (e.g., kernels that can run concurrently) expressed via multiple streams.

**How to Apply:**
1. Begin capture on the origin stream.
2. Fork into other streams using `cudaEventRecord` and `cudaStreamWaitEvent`.
3. Join all streams back to the origin stream before ending capture.

**Code Template:**
```cpp
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

kernelA<<<..., stream1>>>(...);

// Fork
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernelB<<<..., stream1>>>(...);
kernelC<<<..., stream2>>>(...);

// Join
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernelD<<<..., stream1>>>(...);

cudaStreamEndCapture(stream1, &graph);
```

> Source: PG 4.2.2.1.2.1 -- "Stream capture can handle cross-stream dependencies expressed with cudaEventRecord and cudaStreamWaitEvent."

---

## S3: Update Graph Parameters Without Re-Instantiation

**When to Use:** When only kernel parameters (e.g., pointers, grid dims) change between iterations, not the graph structure.

**How to Apply:**
1. Use `cudaGraphExecKernelNodeSetParams` to update individual node parameters.
2. Or use `cudaGraphExecUpdate` to apply changes from a modified source graph.
3. Avoid re-instantiation which is expensive.

**Code Template:**
```cpp
// Update a kernel node's parameters
cudaKernelNodeParams newParams = originalParams;
newParams.kernelParams = newArgs;
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newParams);

// Or: update grid dimensions
cudaGraphKernelNodeSetGridDim(kernelNode, newGridX, newGridY, newGridZ);
```

> Source: PG 4.2.6.1.3 -- graph structure is fixed at instantiation; parameters can be updated.

---

## S4: Device Graph Launch (Fire-and-Forget / Tail Launch)

**When to Use:** When data-dependent decisions determine which graph to run next, and you want to avoid a GPU-to-CPU-to-GPU round trip.

**How to Apply:**
1. Instantiate graphs with `cudaGraphInstantiateFlagDeviceLaunch`.
2. Upload to device via `cudaGraphUpload` or implicit host launch.
3. From device code, launch with `cudaGraphLaunch(exec, cudaStreamGraphFireAndForget)` or `cudaStreamGraphTailLaunch`.

**Code Template:**
```cpp
// Host: setup
cudaGraphInstantiate(&gExec, graph, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(gExec, stream);

// Device: launch from kernel
__global__ void controller(cudaGraphExec_t childGraph) {
    // Fire-and-forget: child runs independently
    cudaGraphLaunch(childGraph, cudaStreamGraphFireAndForget);
}

// Or tail launch: child runs after parent completes
__global__ void controller(cudaGraphExec_t nextGraph) {
    cudaGraphLaunch(nextGraph, cudaStreamGraphTailLaunch);
}
```

> Source: PG 4.2.6 -- "Device graph launch provides a convenient way to perform dynamic control flow from the device."

---

## S5: Use Explicit Graph API for Complex DAGs

**When to Use:** When the graph structure cannot be expressed naturally via stream capture (e.g., conditional nodes, complex fan-out patterns).

**How to Apply:**
1. Create the graph with `cudaGraphCreate`.
2. Add nodes manually with `cudaGraphAddKernelNode`, `cudaGraphAddMemcpyNode`, etc.
3. Specify dependencies explicitly via the dependency arrays.

**Code Template:**
```cpp
cudaGraphCreate(&graph, 0);

cudaGraphNode_t nodes[4];
cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
kParams.kernel.func = (void*)myKernel;
kParams.kernel.gridDim = {gridX, 1, 1};
kParams.kernel.blockDim = {blockX, 1, 1};

cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);
cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);
// nodes[1] and nodes[2] depend on nodes[0], can run concurrently
cudaGraphNode_t deps[] = {nodes[1], nodes[2]};
cudaGraphAddNode(&nodes[3], graph, deps, NULL, 2, &kParams);
```

> Source: PG 4.2.2.1.1 -- explicit graph API for full control over graph structure.

---

## Cascading Opportunities

- CUDA Graphs reduce launch overhead -- pair with `kernel-launch-overhead` persistent kernel patterns for further gains.
- Graph nodes can use `programmatic-dependent-launch` for PDL edge types.
- Graph memory nodes work with `stream-concurrency` stream-ordered allocator.

## Conflicts

- **dynamic-parallelism:** CDP is not permitted in device graph kernel nodes.
- Synchronous APIs (e.g., `cudaDeviceSynchronize`, `cudaMemcpy`) are invalid during stream capture.

## Principles

1. **Define once, launch many:** Graphs amortize CPU setup cost over repeated launches (PG 4.2).
2. **Structure is fixed at instantiation:** Only parameters (not topology) can be updated without re-instantiation.
3. **Device launch eliminates round-trips:** Fire-and-forget and tail launch enable GPU-side dynamic control flow (PG 4.2.6).

## Open Questions

- Q1: What is the overhead of `cudaGraphExecUpdate` compared to re-instantiation on large graphs?
- Q2: How do conditional graph nodes interact with device graph launch?
