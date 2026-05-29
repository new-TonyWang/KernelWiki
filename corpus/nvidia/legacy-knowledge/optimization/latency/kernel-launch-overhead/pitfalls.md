# Kernel Launch Overhead -- Pitfalls

## P1: Many Small Kernels Serialized by Default Stream

**Symptom:** Hundreds of short kernels launched sequentially; GPU utilization is low despite high kernel count.

**Detection:** Nsight Systems timeline shows large gaps between kernel executions. GPU utilization < 50%.

**Fix:** Use CUDA Graphs to batch the kernel sequence, or use non-default streams with concurrent execution.

> Source: BP 11.5 -- "Non-default streams are required for concurrent execution because kernel calls that use the default stream begin only after all preceding calls on the device have completed."

---

## P2: Eager Module Loading Causing Long Startup Latency

**Symptom:** Application takes seconds to start even before first kernel launch, especially when linking many CUDA libraries.

**Detection:** Profile shows time spent in `cuModuleLoad` or JIT compilation at startup.

**Fix:** Ensure `CUDA_MODULE_LOADING=LAZY` (default since CUDA 12.3). For older CUDA versions, set the environment variable explicitly.

> Source: PG 4.7 -- "Lazy loading is particularly effective for programs that only use a small number of the kernels they include."

---

## P3: Graph Instantiation Overhead Charged Per Launch

**Symptom:** Creating and instantiating a graph every iteration instead of reusing it, negating the graph's overhead reduction.

**Detection:** `cudaGraphInstantiate` appears in every iteration in the profiler timeline.

**Fix:** Instantiate once, launch many times. Use `cudaGraphExecUpdate` for parameter changes, or `cudaGraphExecKernelNodeSetParams` for node-specific updates.

```cpp
// Bad: instantiate every iteration
for (int i = 0; i < N; i++) {
    cudaGraphInstantiate(&exec, graph, 0);  // expensive!
    cudaGraphLaunch(exec, stream);
    cudaGraphExecDestroy(exec);
}

// Good: instantiate once
cudaGraphInstantiate(&exec, graph, 0);
for (int i = 0; i < N; i++) {
    cudaGraphLaunch(exec, stream);
}
```

> Source: PG 4.2 -- "presenting the whole workflow to CUDA enables optimizations... much of the setup is done in advance."

---

## P4: Timing Events Adding Unnecessary Overhead

**Symptom:** Events used only for dependency tracking still incur timestamp collection overhead.

**Detection:** Large number of `cudaEventRecord` calls visible in profiler with timing enabled.

**Fix:** Create events with `cudaEventDisableTiming` when timestamps are not needed.

> Source: Runtime API -- cudaEventDisableTiming eliminates timestamp recording overhead.

---

## P5: Using cudaDeviceSynchronize Instead of Stream/Event Sync

**Symptom:** Entire GPU pipeline drains at every synchronization point, killing concurrency.

**Detection:** `cudaDeviceSynchronize` in profiler between independent work items.

**Fix:** Use `cudaStreamSynchronize` or `cudaStreamWaitEvent` for fine-grained synchronization.

```cpp
// Bad: blocks all streams
cudaDeviceSynchronize();

// Good: synchronize only the stream you need
cudaStreamSynchronize(stream);

// Better: use events for cross-stream dependencies
cudaEventRecord(event, producerStream);
cudaStreamWaitEvent(consumerStream, event);
```

> Source: PG 2.3.7 -- "cudaDeviceSynchronize waits until all preceding commands in all streams of all host threads have completed."

## P6: This skill targets host-side launch overhead that is invisible to NCU kernel profiling; the benefit only manifests when measuring end-to-end wall-clock time across many rapid event-record calls in a tight loop, not in single-kernel latency benchmarks (discovered in verification)

**Symptom**: This skill targets host-side launch overhead that is invisible to NCU kernel profiling; the benefit only manifests when measuring end-to-end wall-clock time across many rapid event-record calls in a tight loop, not in single-kernel latency benchmarks.
**Source**: Level 3 sandbox verification (2026-04-06)

## P7: Cycle count dropped ~5% but wall-clock time was unchanged, suggesting the difference is measurement noise rather than a real improvement; warp occupancy increase (68% → 72%) is also likely noise given identical instruction counts (discovered in verification)

**Symptom**: Cycle count dropped ~5% but wall-clock time was unchanged, suggesting the difference is measurement noise rather than a real improvement; warp occupancy increase (68% → 72%) is also likely noise given identical instruction counts.
**Source**: Level 3 sandbox verification (2026-04-06)
