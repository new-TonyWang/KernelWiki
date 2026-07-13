# Green Contexts -- Pitfalls

## P1: Over-Partitioning SMs Leaving Too Few for Each Workload

**Symptom:** Each green context has so few SMs that kernels cannot achieve reasonable occupancy, resulting in lower overall throughput.

**Detection:** Total GPU throughput decreases compared to no partitioning.

**Fix:** Start with a rough 80/20 split and profile. Ensure each partition has enough SMs for the kernel's occupancy requirements.

> Source: PG 4.6.1 -- "The number of SMs each green context will have access to should be decided by the user during green context creation on a per case basis."

---

## P2: Expecting Guaranteed Concurrent Execution

**Symptom:** Independent kernels on different green contexts still do not execute concurrently.

**Detection:** Nsight Systems shows sequential execution despite separate green contexts.

**Fix:** Green contexts reduce interference but do not guarantee concurrency. Other factors (e.g., shared work queues, resource limits) may still prevent concurrent execution. Configure work queue resources as well.

> Source: PG 4.6.1 -- "even when different SM resources and work queues are provisioned per green context, concurrent execution of independent GPU work is not guaranteed."

---

## P3: Creating Green Contexts on Devices Without Support

**Symptom:** API calls fail on devices with compute capability < 9.0 (prior to Hopper) or CUDA < 13.1 for Runtime API.

**Detection:** `cudaGreenCtxCreate` returns error on unsupported hardware/software.

**Fix:** Check device compute capability. Green contexts via the Driver API require CC 9.0+. Runtime API support (`cudaGreenCtxCreate`) requires CUDA 13.1+.

> Source: PG 4.6 -- Green context support first available via Driver API; Runtime API from CUDA 13.1.

---

## P4: Forgetting to Create Streams on the Green Context

**Symptom:** Kernels launched on a regular stream bypass the green context and use all SMs.

**Detection:** Kernel uses more SMs than allocated to the green context.

**Fix:** Create streams via `cudaExecutionCtxStreamCreate` on the green context's execution context, not via `cudaStreamCreate`.

```cpp
// BAD: regular stream, ignores green context
cudaStream_t stream;
cudaStreamCreate(&stream);

// GOOD: stream bound to green context
cudaStream_t stream;
cudaExecutionCtxStreamCreate(&stream, execCtx, cudaStreamNonBlocking);
```

> Source: PG 4.6.5 -- "streams should be created on the green context's execution context."
