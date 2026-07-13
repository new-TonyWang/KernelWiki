# CUDA Graphs -- Pitfalls

## P1: Re-Instantiating Graph Every Iteration

**Symptom:** Graph launch is slower than expected because instantiation overhead is paid each iteration.

**Detection:** `cudaGraphInstantiate` appears per iteration in the profiler.

**Fix:** Instantiate once. Use `cudaGraphExecUpdate` or node-level parameter updates for changing data.

> Source: PG 4.2 -- "the resulting instance... may be launched any number of times without repeating the instantiation."

---

## P2: Synchronous Calls During Stream Capture

**Symptom:** `cudaStreamBeginCapture` / `cudaStreamEndCapture` fails or produces an invalid graph.

**Detection:** Error codes from capture APIs. `cudaMemcpy` (synchronous) used inside capture region.

**Fix:** Replace all synchronous calls with async variants. Use `cudaMemcpyAsync` instead of `cudaMemcpy`. Do not call `cudaDeviceSynchronize` during capture.

> Source: PG 4.2.2.1.2.2 -- "It is invalid to synchronize or query the execution status of a stream which is being captured."

---

## P3: Forgetting to Rejoin Forked Streams to Origin

**Symptom:** `cudaStreamEndCapture` returns an error because forked streams were not joined back.

**Detection:** `cudaStreamEndCapture` returns `cudaErrorStreamCaptureUnjoined`.

**Fix:** Every forked stream must be joined back to the origin stream via `cudaEventRecord` / `cudaStreamWaitEvent` before `cudaStreamEndCapture`.

```cpp
// Fork
cudaEventRecord(e1, originStream);
cudaStreamWaitEvent(forkStream, e1);
// ... work on forkStream ...
// Join (required!)
cudaEventRecord(e2, forkStream);
cudaStreamWaitEvent(originStream, e2);
// Now safe to end capture
cudaStreamEndCapture(originStream, &graph);
```

> Source: PG 4.2.2.1.2.1 -- "All streams being captured to the same capture graph are taken out of capture mode upon cudaStreamEndCapture. Failure to rejoin to the origin stream will result in failure."

---

## P4: Using Legacy Default Stream During Capture

**Symptom:** Capture fails when any associated stream is being captured and the legacy stream is used.

**Detection:** Error when launching to stream 0 while another stream is in capture mode.

**Fix:** Use non-blocking streams or per-thread default stream. Never use the legacy NULL stream during capture.

> Source: PG 4.2.2.1.2.2 -- "When any stream in the same context is being captured... any attempted use of the legacy stream is invalid."

---

## P5: Device Graph Launched Twice Concurrently

**Symptom:** `cudaErrorInvalidValue` when launching a device graph from the device while a previous launch is still running.

**Detection:** Error returned from device-side `cudaGraphLaunch`.

**Fix:** Ensure the previous device graph launch has completed before relaunching. Use tail launch mode for sequential chaining.

> Source: PG 4.2.6 -- "launching a device graph from the device while a previous launch of the graph is running will result in an error."

## P6: NCU metrics and roofline data are empty for the optimized run, likely because ncu cannot profile device-launched child graphs in the same way as host-launched kernels — profiling CUDA graph internals requires special ncu flags or CUPTI tracing (discovered in verification)

**Symptom**: NCU metrics and roofline data are empty for the optimized run, likely because ncu cannot profile device-launched child graphs in the same way as host-launched kernels — profiling CUDA graph internals requires special ncu flags or CUPTI tracing.
**Source**: Level 3 sandbox verification (2026-04-04)

## P7: Both baseline and optimized remain massively underutilized (~5 (discovered in verification)

**Symptom**: Both baseline and optimized remain massively underutilized (~5.7% SM throughput, ~2.7% memory throughput); applying CUDA Graphs to already compute-heavy kernels would show negligible benefit since launch overhead would be dwarfed by execution time.
**Source**: Level 3 sandbox verification (2026-04-04)

## P8: For micro-benchmarks with very short kernel runtimes, the CUDA graph parameter-update API overhead can dominate and actually regress performance compared to a simple re-instantiation or direct launch path (discovered in verification)

**Symptom**: For micro-benchmarks with very short kernel runtimes, the CUDA graph parameter-update API overhead can dominate and actually regress performance compared to a simple re-instantiation or direct launch path.
**Source**: Level 3 sandbox verification (2026-04-06)
